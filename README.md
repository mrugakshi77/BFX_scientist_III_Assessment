# Multiomic cancer-panel assignment

This repository implements a leakage-safe, reproducible discovery-to-panel workflow for the supplied synthetic cfDNA and protein data.

## Design choices

- All evaluation uses `StratifiedGroupKFold` grouped by `patient_id`; no patient's draws can occur in both training and test sets.
- Imputation, missingness filtering, scaling, univariate screening, and regularized model tuning occur only inside each training fold.
- Protein features are plate-median centered using per-plate medians learned on the training fold only (plates absent from training fall back to the training global median).
- Protein models use the protein-available subset. A matched methylation/fragmentomics model is also run on that same subset, making protein's incremental value interpretable.
- The final 5+3 panel is selected only in a development subset; its reported final score comes from a patient-grouped holdout untouched by panel selection.
- Stage is deliberately excluded from prediction. It is retained only for QC/exploratory summaries.
- Repeats and extreme GC are reported as ddPCR feasibility flags rather than automatic exclusions.

## Run

Pixi is the canonical environment manager for this project:

```powershell
pixi run analysis
```

This resolves the pinned Conda-forge environment and runs the workflow. `requirements.txt` and `environment.yml` are retained as portability fallbacks.

Run the guardrail tests with:

```powershell
pixi run test
```

Outputs are written to `results/`:

- `qc/`: sample-level QC tables and figures.
- `models/`: development-only out-of-fold predictions, per-fold metrics, feature-selection/indicator/coefficient reports, and selected-feature stability.
- `panel/`: the candidate 5-methylation/3-protein panel, ddPCR feasibility annotations, and its locked patient-grouped holdout evaluation.
- `run_manifest.json`: input file hashes and run configuration for reproducibility.

## Notes for interpretation

The panel is a discovery candidate, not a clinically validated signature. Before assay lock-down, selected methylation regions require sequence-level uniqueness checks, primer/probe design, local-CpG review, and wet-lab precision/LOD validation. Site and plate results should be inspected for batch effects; use the included site-stratified QC tables and consider leave-one-site-out validation if a site effect is found.

