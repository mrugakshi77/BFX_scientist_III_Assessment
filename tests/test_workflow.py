"""Regression tests for leakage and feature-selection guardrails."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from multiomic_panel.workflow import MissingnessFilter, SafeSelectKBest, _pipeline


def test_missingness_filter_drops_all_nan_and_high_missing_cpgs() -> None:
    frame = pd.DataFrame({
        "CpG_keep": [0.1, 0.2, np.nan, 0.4],
        "CpG_drop": [np.nan, np.nan, 0.3, np.nan],
        "protein_all_nan": [np.nan, np.nan, np.nan, np.nan],
        "protein_keep": [1.0, np.nan, 3.0, 4.0],
    })
    filtered = MissingnessFilter(max_fraction=0.25).fit_transform(frame)
    assert list(filtered.columns) == ["CpG_keep", "protein_keep"]


def test_safe_selector_clips_requested_budget_to_fold_feature_count() -> None:
    matrix = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]])
    selector = SafeSelectKBest(k=50).fit(matrix, np.array([0, 1, 0, 1]))
    assert selector.get_support().sum() == matrix.shape[1]


def test_grouped_split_keeps_every_patient_in_one_partition() -> None:
    groups = np.repeat(["P1", "P2", "P3", "P4", "P5", "P6"], 2)
    y = np.repeat([0, 1, 0, 1, 0, 1], 2)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=7)
    for train_index, test_index in splitter.split(np.zeros(len(y)), y, groups):
        assert set(groups[train_index]).isdisjoint(set(groups[test_index]))


def test_fixed_panel_pipeline_has_no_unrestricted_selector() -> None:
    pipeline = _pipeline(max_missingness=0.2, select_features=False)
    assert "select" not in pipeline.named_steps
