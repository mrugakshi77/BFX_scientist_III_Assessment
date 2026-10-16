"""End-to-end QC, grouped model evaluation, and constrained panel selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROTEIN_METADATA_COLUMNS = {"sample_id", "plate_id", "well"}


@dataclass(frozen=True)
class Inputs:
    samples: pd.DataFrame
    methylation: pd.DataFrame
    fragmentomics: pd.DataFrame
    proteins: pd.DataFrame
    annotation: pd.DataFrame


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Apply a training-fold missingness limit only to a designated feature prefix."""

    def __init__(self, max_fraction: float = 0.20, restricted_prefix: str = "CpG_"):
        self.max_fraction = max_fraction
        self.restricted_prefix = restricted_prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingnessFilter":
        frame = self._frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        applies = frame.columns.str.startswith(self.restricted_prefix)
        has_observation = frame.notna().any(axis=0).to_numpy()
        self.support_mask_ = has_observation & (~applies | (frame.isna().mean() <= self.max_fraction)).to_numpy()
        if not self.support_mask_.any():
            raise ValueError("Missingness filter removed every feature.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._frame(X)
        return frame.loc[:, self.feature_names_in_[self.support_mask_]]

    @staticmethod
    def _frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingnessFilter requires a pandas DataFrame.")
        return X


class SafeSelectKBest(SelectKBest):
    """Clip the requested univariate-feature budget to features retained in this fold."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SafeSelectKBest":
        if self.k != "all":
            self.k = min(int(self.k), X.shape[1])
        return super().fit(X, y)


def _plate_center(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    plates_train: pd.Series,
    plates_test: pd.Series,
    protein_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn per-plate protein medians on the training fold and subtract from both folds.

    Fitted on training rows only, so no outer-fold leakage. Samples whose plate was not
    seen in training (or whose plate_id is missing) fall back to the training global median.
    """
    protein_in_X = [c for c in protein_columns if c in X_train.columns]
    if not protein_in_X:
        return X_train, X_test
    X_train = X_train.copy()
    X_test = X_test.copy()
    plate_medians = (
        X_train[protein_in_X]
        .assign(_plate=plates_train.to_numpy())
        .groupby("_plate")[protein_in_X]
        .median()
    )
    global_median = X_train[protein_in_X].median()
    for frame, plates in ((X_train, plates_train), (X_test, plates_test)):
        offsets = plate_medians.reindex(plates.to_numpy()).fillna(global_median)
        frame.loc[:, protein_in_X] = frame[protein_in_X].to_numpy() - offsets.to_numpy()
    return X_train, X_test


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_inputs(data_dir: Path) -> Inputs:
    """Read and validate source tables without modifying values."""
    tables = {
        "samples": pd.read_csv(data_dir / "samples.csv"),
        "methylation": pd.read_csv(data_dir / "methylation.csv"),
        "fragmentomics": pd.read_csv(data_dir / "fragmentomics.csv"),
        "proteins": pd.read_csv(data_dir / "protein.csv"),
        "annotation": pd.read_csv(data_dir / "region_annotation.csv"),
    }
    for name, table in tables.items():
        if "sample_id" in table and table.sample_id.duplicated().any():
            raise ValueError(f"{name} contains duplicate sample_id values.")
    if not set(tables["methylation"].columns[1:]).issubset(tables["annotation"].region_id):
        raise ValueError("Methylation features do not all have region annotations.")
    return Inputs(**tables)


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def run_qc(inputs: Inputs, output_dir: Path) -> None:
    """Write transparent QC summaries; these diagnostics do not alter model inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, methylation, proteins = inputs.samples, inputs.methylation, inputs.proteins
    methyl_features = methylation.columns.drop("sample_id")
    protein_features = proteins.columns.difference(PROTEIN_METADATA_COLUMNS)

    sample_qc = samples.merge(
        methylation.assign(methylation_missing_fraction=methylation[methyl_features].isna().mean(axis=1))[
            ["sample_id", "methylation_missing_fraction"]
        ],
        on="sample_id", how="left", validate="one_to_one",
    ).merge(
        inputs.fragmentomics[["sample_id", "total_reads_millions"]], on="sample_id", how="left", validate="one_to_one"
    )
    sample_qc["protein_available"] = sample_qc.sample_id.isin(proteins.sample_id)
    _write_table(sample_qc, output_dir / "sample_level_qc.csv")

    methyl_with_status = methylation.merge(samples[["sample_id", "cancer_status"]], on="sample_id", validate="one_to_one")
    methyl_qc = pd.DataFrame({
        "region_id": methyl_features,
        "missing_fraction": methylation[methyl_features].isna().mean().to_numpy(),
        "variance": methylation[methyl_features].var().to_numpy(),
        "mean_cancer": methyl_with_status.loc[methyl_with_status.cancer_status == "cancer", methyl_features].mean().to_numpy(),
        "mean_non_cancer": methyl_with_status.loc[methyl_with_status.cancer_status == "non_cancer", methyl_features].mean().to_numpy(),
    })
    methyl_qc["mean_difference_cancer_minus_non_cancer"] = methyl_qc.mean_cancer - methyl_qc.mean_non_cancer
    _write_table(methyl_qc, output_dir / "methylation_region_qc.csv")

    fragment_features = inputs.fragmentomics.columns.drop("sample_id")
    fragment_qc = pd.DataFrame({
        "feature": fragment_features,
        "missing_fraction": inputs.fragmentomics[fragment_features].isna().mean().to_numpy(),
        "variance": inputs.fragmentomics[fragment_features].var().to_numpy(),
        "correlation_with_total_reads": [inputs.fragmentomics[column].corr(inputs.fragmentomics.total_reads_millions) for column in fragment_features],
    })
    _write_table(fragment_qc, output_dir / "fragmentomics_feature_qc.csv")

    protein_missing = pd.DataFrame({
        "protein": protein_features,
        "below_lloq_count": [proteins[column].isna().sum() for column in protein_features],
        "below_lloq_fraction": [proteins[column].isna().mean() for column in protein_features],
    }).sort_values("below_lloq_fraction", ascending=False)
    _write_table(protein_missing, output_dir / "protein_lloq_missingness.csv")

    site_summary = sample_qc.groupby(["site", "cancer_status"], as_index=False).agg(
        n_samples=("sample_id", "size"),
        median_age=("age", "median"),
        median_depth_millions=("total_reads_millions", "median"),
        median_methylation_missingness=("methylation_missing_fraction", "median"),
        protein_available_fraction=("protein_available", "mean"),
    )
    _write_table(site_summary, output_dir / "site_outcome_summary.csv")

    protein_status = proteins[["sample_id", "plate_id"]].merge(
        samples[["sample_id", "cancer_status"]], on="sample_id", how="left", validate="one_to_one"
    )
    _write_table(pd.crosstab(protein_status.plate_id, protein_status.cancer_status, margins=True).reset_index(), output_dir / "plate_by_outcome.csv")


def assemble_data(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align modalities to metadata and return all samples in sample-table order."""
    methyl = inputs.methylation.set_index("sample_id")
    frag = inputs.fragmentomics.set_index("sample_id")
    protein = inputs.proteins.set_index("sample_id")
    base = inputs.samples.set_index("sample_id")
    index = base.index
    protein_cols = [c for c in protein.columns if c not in {"plate_id", "well"}]
    features = pd.concat([methyl.reindex(index), frag.reindex(index), protein.reindex(index)[protein_cols]], axis=1)
    metadata = base.reset_index()
    metadata["plate_id"] = metadata["sample_id"].map(inputs.proteins.set_index("sample_id")["plate_id"])
    return metadata, features, protein_cols


def feature_subset(columns: list[str], prefixes: list[str], protein_columns: list[str]) -> list[str]:
    if prefixes == ["ALL"]:
        return columns
    selected: list[str] = []
    for column in columns:
        if "PROTEIN" in prefixes and column in protein_columns:
            selected.append(column)
        elif any(column.startswith(prefix) for prefix in prefixes):
            selected.append(column)
    return selected


def _pipeline(max_missingness: float, select_features: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("missingness", MissingnessFilter(max_missingness)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=False)),
        ("scale", StandardScaler()),
    ]
    if select_features:
        steps.append(("select", SafeSelectKBest(score_func=f_classif)))
    steps.append(("classifier", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000)))
    return Pipeline(steps)


def _threshold_for_specificity(y: np.ndarray, probability: np.ndarray, target: float) -> float:
    """Select a diagnostic threshold on development predictions only."""
    fpr, tpr, thresholds = roc_curve(y, probability)
    eligible = (fpr <= 1 - target) & np.isfinite(thresholds)
    if not eligible.any():
        raise ValueError("No finite development threshold achieved the requested specificity.")
    # Maximize sensitivity subject to the prespecified specificity constraint.
    best_tpr = tpr[eligible].max()
    return float(thresholds[eligible & np.isclose(tpr, best_tpr)].min())


def _safe_k_values(values: list[Any], n_features: int) -> list[Any]:
    valid = [value for value in values if value == "all" or int(value) <= n_features]
    return valid or ["all"]


def evaluate_model(
    output_name: str,
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    spec_name: str | None = None,
    protein_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested grouped CV. Each outer test fold remains untouched until scoring."""
    y = (metadata.cancer_status == "cancer").astype(int).to_numpy()
    groups = metadata.patient_id.to_numpy()
    outer = StratifiedGroupKFold(n_splits=config["outer_folds"], shuffle=True, random_state=config["random_seed"])
    predictions, fold_metrics, feature_rows = [], [], []
    spec_name = spec_name or output_name
    k_values = _safe_k_values(config["model_specs"][spec_name]["k_candidates"], X.shape[1])

    for fold, (train_ix, test_ix) in enumerate(outer.split(X, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True, random_state=config["random_seed"] + fold)
        search = GridSearchCV(
            _pipeline(config["methylation_missingness_threshold"]),
            param_grid={"select__k": k_values, "classifier__C": [0.05, 0.2, 1.0]},
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        if protein_columns:
            X_train, X_test = _plate_center(
                X_train, X_test,
                metadata.plate_id.iloc[train_ix], metadata.plate_id.iloc[test_ix],
                protein_columns,
            )
        search.fit(X_train, y[train_ix], groups=groups[train_ix])
        probability = search.predict_proba(X_test)[:, 1]
        report = _feature_report(search.best_estimator_, fold)
        feature_rows.extend(report.to_dict("records"))
        predictions.append(pd.DataFrame({"sample_id": metadata.sample_id.iloc[test_ix].to_numpy(), "fold": fold, "y_true": y[test_ix], "probability": probability}))
        fold_metrics.append({
            "model": output_name, "fold": fold, "n_test": len(test_ix),
            "roc_auc": roc_auc_score(y[test_ix], probability),
            "average_precision": average_precision_score(y[test_ix], probability),
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    _write_table(prediction_table, output_dir / f"{output_name}_oof_predictions.csv")
    _write_table(pd.DataFrame(fold_metrics), output_dir / f"{output_name}_fold_metrics.csv")
    feature_report = pd.DataFrame(feature_rows)
    _write_table(feature_report, output_dir / f"{output_name}_feature_reports.csv")
    analyte_rows = feature_report.query("feature_type == 'analyte' and passed_univariate_screen and nonzero_coefficient")
    stability = analyte_rows.groupby("feature", as_index=False).agg(
        selected_folds=("fold", "nunique"),
        mean_abs_coefficient=("coefficient", lambda values: float(np.mean(np.abs(values)))),
    )
    stability["selection_fraction"] = stability.selected_folds / config["outer_folds"]
    _write_table(stability.sort_values(["selection_fraction", "feature"], ascending=[False, True]), output_dir / f"{output_name}_selection_stability.csv")
    return stability, prediction_table


def _feature_report(pipeline: Pipeline, fold: int) -> pd.DataFrame:
    """Separate measured analytes, missingness indicators, and fitted coefficients."""
    filtered = pipeline.named_steps["missingness"].feature_names_in_[pipeline.named_steps["missingness"].support_mask_]
    imputer = pipeline.named_steps["impute"]
    indicator_features = [f"{filtered[index]}__missing" for index in imputer.indicator_.features_]
    transformed_names = list(filtered) + indicator_features
    selector = pipeline.named_steps["select"]
    support = selector.get_support()
    coefficients = pipeline.named_steps["classifier"].coef_.ravel()
    selected_names = np.asarray(transformed_names, dtype=object)[support]
    return pd.DataFrame({
        "fold": fold,
        "feature": selected_names,
        "feature_type": np.where(pd.Series(selected_names).str.endswith("__missing"), "missingness_indicator", "analyte"),
        "passed_univariate_screen": True,
        "coefficient": coefficients,
        "nonzero_coefficient": ~np.isclose(coefficients, 0.0),
    })


def choose_panel(
    methyl_stability: pd.DataFrame,
    protein_stability: pd.DataFrame,
    annotation: pd.DataFrame,
    methylation_development: pd.DataFrame,
    protein_development: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    """Use stability, feasibility, and redundancy to lock an exact 5+3 panel."""
    methyl_qc = pd.DataFrame({
        "feature": methylation_development.columns,
        "missing_fraction": methylation_development.isna().mean().to_numpy(),
        "variance": methylation_development.var().to_numpy(),
    }).merge(annotation, left_on="feature", right_on="region_id", how="left", validate="one_to_one")
    methyl_candidates = methyl_stability.merge(methyl_qc, on="feature", how="inner")
    methyl_candidates["technical_eligible"] = methyl_candidates.missing_fraction <= 0.20
    methyl_candidates["feasibility_penalty"] = (
        methyl_candidates.repeat_overlap.astype(int) + ((methyl_candidates.gc_content < 0.25) | (methyl_candidates.gc_content > 0.75)).astype(int)
    )
    methyl_candidates["panel_score"] = (
        methyl_candidates.selection_fraction * methyl_candidates.mean_abs_coefficient - 0.05 * methyl_candidates.feasibility_penalty
    )
    methyl_candidates = methyl_candidates[methyl_candidates.technical_eligible].sort_values(
        ["panel_score", "feature"], ascending=[False, True]
    )
    methyl_panel: list[str] = []
    for candidate in methyl_candidates.feature:
        correlations = [abs(methylation_development[candidate].corr(methylation_development[chosen])) for chosen in methyl_panel]
        if not correlations or max(correlations) < 0.85:
            methyl_panel.append(candidate)
        if len(methyl_panel) == 5:
            break

    protein_qc = pd.DataFrame({
        "feature": protein_development.columns,
        "below_lloq_fraction": protein_development.isna().mean().to_numpy(),
    })
    protein_candidates = protein_stability.merge(protein_qc, on="feature", how="inner")
    protein_candidates["panel_score"] = protein_candidates.selection_fraction * protein_candidates.mean_abs_coefficient - 0.05 * protein_candidates.below_lloq_fraction
    protein_candidates = protein_candidates.sort_values(["panel_score", "feature"], ascending=[False, True])
    protein_panel: list[str] = []
    for candidate in protein_candidates.feature:
        correlations = [abs(protein_development[candidate].corr(protein_development[chosen])) for chosen in protein_panel]
        if not correlations or max(correlations) < 0.85:
            protein_panel.append(candidate)
        if len(protein_panel) == 3:
            break

    if len(methyl_panel) != 5 or len(protein_panel) != 3:
        raise ValueError("Insufficient technically eligible, nonredundant candidates to lock a 5+3 panel.")
    panel = methyl_candidates[methyl_candidates.feature.isin(methyl_panel)].copy()
    panel["selection_rank"] = panel.region_id.map({feature: rank for rank, feature in enumerate(methyl_panel, 1)})
    panel["ddpcr_feasibility_flags"] = panel.apply(
        lambda row: ";".join(
            [flag for applies, flag in [
                (bool(row.repeat_overlap), "repeat_overlap"),
                (row.gc_content < 0.25 or row.gc_content > 0.75, "extreme_gc"),
            ] if applies]
        ) or "review_sequence_uniqueness_and_local_CpGs",
        axis=1,
    )
    _write_table(panel.sort_values("selection_rank"), output_dir / "methylation_panel_with_feasibility.csv")
    _write_table(protein_candidates[protein_candidates.feature.isin(protein_panel)].assign(
        selection_rank=lambda table: table.feature.map({feature: rank for rank, feature in enumerate(protein_panel, 1)})
    ).sort_values("selection_rank"), output_dir / "protein_panel.csv")
    return methyl_panel, protein_panel


def locked_panel_holdout_evaluation(
    X_development: pd.DataFrame,
    development_metadata: pd.DataFrame,
    X_holdout: pd.DataFrame,
    holdout_metadata: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    protein_columns: list[str] | None = None,
) -> None:
    """Lock panel, hyperparameters, and operating threshold before scoring holdout patients."""
    y_development = (development_metadata.cancer_status == "cancer").astype(int).to_numpy()
    y_holdout = (holdout_metadata.cancer_status == "cancer").astype(int).to_numpy()
    groups = development_metadata.patient_id.to_numpy()
    outer = StratifiedGroupKFold(n_splits=config["outer_folds"], shuffle=True, random_state=config["random_seed"])
    development_oof: list[pd.DataFrame] = []
    for fold, (train_ix, test_ix) in enumerate(outer.split(X_development, y_development, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True, random_state=config["random_seed"] + fold)
        fold_search = GridSearchCV(
            _pipeline(config["methylation_missingness_threshold"], select_features=False),
            param_grid={"classifier__C": [0.05, 0.2, 1.0]}, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True, error_score="raise",
        )
        X_tr, X_te = X_development.iloc[train_ix], X_development.iloc[test_ix]
        if protein_columns:
            X_tr, X_te = _plate_center(
                X_tr, X_te,
                development_metadata.plate_id.iloc[train_ix], development_metadata.plate_id.iloc[test_ix],
                protein_columns,
            )
        fold_search.fit(X_tr, y_development[train_ix], groups=groups[train_ix])
        development_oof.append(pd.DataFrame({"y_true": y_development[test_ix], "probability": fold_search.predict_proba(X_te)[:, 1]}))
    development_oof_table = pd.concat(development_oof, ignore_index=True)
    threshold = _threshold_for_specificity(
        development_oof_table.y_true.to_numpy(), development_oof_table.probability.to_numpy(), config["specificity_target"]
    )
    inner = StratifiedGroupKFold(
        n_splits=config["inner_folds"], shuffle=True, random_state=config["random_seed"]
    )
    search = GridSearchCV(
        _pipeline(config["methylation_missingness_threshold"], select_features=False),
        param_grid={"classifier__C": [0.05, 0.2, 1.0]},
        scoring="roc_auc", cv=inner, n_jobs=-1, refit=True, error_score="raise",
    )
    X_dev_fit, X_hold_fit = X_development, X_holdout
    if protein_columns:
        X_dev_fit, X_hold_fit = _plate_center(
            X_development, X_holdout,
            development_metadata.plate_id, holdout_metadata.plate_id,
            protein_columns,
        )
    search.fit(X_dev_fit, y_development, groups=groups)
    probability = search.predict_proba(X_hold_fit)[:, 1]
    calls = probability >= threshold
    sensitivity = float(calls[y_holdout == 1].mean()) if (y_holdout == 1).any() else np.nan
    specificity = float((~calls[y_holdout == 0]).mean()) if (y_holdout == 0).any() else np.nan
    _write_table(development_oof_table.assign(diagnostic_threshold=threshold), output_dir / "restricted_final_panel_development_oof_predictions.csv")
    _write_table(pd.DataFrame({
        "sample_id": holdout_metadata.sample_id.to_numpy(), "patient_id": holdout_metadata.patient_id.to_numpy(),
        "y_true": y_holdout, "probability": probability, "diagnostic_threshold": threshold, "positive_call": calls,
    }), output_dir / "restricted_final_panel_locked_holdout_predictions.csv")
    _write_table(pd.DataFrame([{
        "n_development": len(development_metadata), "n_holdout": len(holdout_metadata),
        "n_development_patients": development_metadata.patient_id.nunique(),
        "n_holdout_patients": holdout_metadata.patient_id.nunique(),
        "roc_auc": roc_auc_score(y_holdout, probability),
        "average_precision": average_precision_score(y_holdout, probability),
        "threshold_selected_on_development_oof": threshold,
        "holdout_sensitivity_at_locked_threshold": sensitivity,
        "holdout_specificity_at_locked_threshold": specificity,
        "best_parameters": json.dumps(search.best_params_, sort_keys=True),
    }]), output_dir / "restricted_final_panel_locked_holdout_metrics.csv")


def write_manifest(data_dir: Path, config: dict[str, Any], output_path: Path) -> None:
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(data_dir.glob("*.csv"))}
    output_path.write_text(json.dumps({"config": config, "input_sha256": hashes}, indent=2), encoding="utf-8")


def run(config_path: Path) -> None:
    config = load_config(config_path)
    root = config_path.parent.resolve()
    data_dir, results_dir = root / config["data_dir"], root / config["results_dir"]
    inputs = load_inputs(data_dir)
    run_qc(inputs, results_dir / "qc")
    metadata, features, protein_cols = assemble_data(inputs)
    specs = config["model_specs"]

    # This split occurs before any supervised model comparison or feature selection.
    splitter = StratifiedGroupKFold(
        n_splits=config["final_panel_holdout_folds"], shuffle=True, random_state=config["random_seed"]
    )
    development_ix, holdout_ix = next(splitter.split(features, metadata.cancer_status, metadata.patient_id))
    development_metadata = metadata.iloc[development_ix].reset_index(drop=True)
    development_features = features.iloc[development_ix].reset_index(drop=True)
    holdout_metadata = metadata.iloc[holdout_ix].reset_index(drop=True)
    holdout_features = features.iloc[holdout_ix].reset_index(drop=True)
    _write_table(pd.DataFrame({
        "sample_id": metadata.sample_id,
        "patient_id": metadata.patient_id,
        "partition": np.where(metadata.sample_id.isin(holdout_metadata.sample_id), "locked_holdout", "development"),
    }), results_dir / "study_partitions.csv")

    # All benchmark fitting is confined to development patients.
    for name in ("methylation", "fragmentomics", "methylation_fragmentomics"):
        cols = feature_subset(list(development_features.columns), specs[name]["feature_prefixes"], protein_cols)
        evaluate_model(name, development_features[cols], development_metadata, config, results_dir / "models")

    development_protein_available = ~development_features[protein_cols].isna().all(axis=1)
    protein_metadata = development_metadata.loc[development_protein_available].reset_index(drop=True)
    protein_features = development_features.loc[development_protein_available].reset_index(drop=True)
    for name in ("protein", "integrated", "methylation_fragmentomics"):
        model_name = f"{name}_protein_subset" if name == "methylation_fragmentomics" else name
        spec_name = name
        cols = feature_subset(list(protein_features.columns), specs[spec_name]["feature_prefixes"], protein_cols)
        evaluate_model(
            model_name, protein_features[cols], protein_metadata, config, results_dir / "models",
            spec_name=spec_name, protein_columns=protein_cols,
        )

    panel_methyl_cols = feature_subset(list(protein_features.columns), specs["methylation"]["feature_prefixes"], protein_cols)
    panel_protein_cols = feature_subset(list(protein_features.columns), specs["protein"]["feature_prefixes"], protein_cols)
    panel_methyl_stability, _ = evaluate_model(
        "panel_selection_methylation", protein_features[panel_methyl_cols], protein_metadata, config,
        results_dir / "panel", spec_name="methylation"
    )
    panel_protein_stability, _ = evaluate_model(
        "panel_selection_protein", protein_features[panel_protein_cols], protein_metadata, config,
        results_dir / "panel", spec_name="protein", protein_columns=protein_cols,
    )
    methyl_panel, protein_panel = choose_panel(
        panel_methyl_stability, panel_protein_stability, inputs.annotation,
        protein_features[panel_methyl_cols], protein_features[panel_protein_cols], results_dir / "panel"
    )
    restricted_cols = methyl_panel + protein_panel
    holdout_protein_available = ~holdout_features[protein_cols].isna().all(axis=1)
    locked_panel_holdout_evaluation(
        protein_features[restricted_cols], protein_metadata,
        holdout_features.loc[holdout_protein_available, restricted_cols].reset_index(drop=True),
        holdout_metadata.loc[holdout_protein_available].reset_index(drop=True), config, results_dir / "panel",
        protein_columns=protein_cols,
    )
    write_manifest(data_dir, config, results_dir / "run_manifest.json")

