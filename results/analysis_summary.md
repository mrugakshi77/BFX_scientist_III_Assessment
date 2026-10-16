# Multiomic panel analysis — methods and results

## Data and study design

- 350 blood draws from 300 patients (148 cancer, 202 non-cancer); 50 patients contributed two draws.
- Modalities: 2,000 methylation regions and 13 fragmentomics features (both n = 350); 40 proteins (n = 255).
- All splits are patient-grouped. Before any supervised step, a deterministic stratified patient-grouped split reserved **70 draws / 60 patients** as a locked holdout; development = **280 draws / 240 patients**.
- Final 5 + 3 panel is only evaluable on protein-available samples: **206 development** (177 patients) / **49 holdout** (43 patients).

## Pipeline (leakage-safe)

1. **QC**: sample-level, methylation-region, fragmentomics, protein LLOQ, site × outcome, plate × outcome.
2. **Nested `StratifiedGroupKFold`** (outer = 5, inner = 4) within development, grouped by `patient_id`. Imputation, missingness filtering (CpG regions > 20% missing dropped), scaling, univariate screening, and L1-logistic tuning all fit per training fold.
3. **Plate correction**: per-plate protein medians learned on the outer training fold only, subtracted from both train and test; plates unseen in training fall back to the training global median.
4. **Panel locking**: candidates ranked by `selection_fraction × mean|coef| − 0.05 × feasibility_penalty`, filtered on technical missingness, then greedy-pruned by pairwise correlation (|ρ| < 0.85), hard-stop at 5 methylation + 3 protein.
5. **Locked evaluation**: panel, hyperparameters, and a 95%-specificity threshold are all fixed on development OOF predictions, then applied once to the holdout.

## QC highlights

- Methylation coverage is strong; no region exceeds 20% missingness in the full cohort.
- Fragmentomics is complete; correlations with sequencing depth are small (|r| ≤ 0.13).
- **Structural confounds.** Site C has no protein data and is 77 non-cancer / 18 cancer. Plates are outcome-imbalanced (PLT01 62/26, PLT02 40/48, PLT03 28/51) — addressed by per-fold plate centering.
- Protein LLOQ: most proteins ≈ 4.3% below-LLOQ; IL1B, IL10, IFNG at ≈ 25%.

## Development grouped-CV comparison (5 outer folds)

| Model | Population | Mean AUC | Mean AP |
|---|---:|---:|---:|
| Fragmentomics | 280 dev | 0.62 | 0.59 |
| Methylation | 280 dev | 0.87 | 0.82 |
| Methyl + frag | 280 dev | 0.86 | 0.81 |
| Protein (plate-centered) | 206 dev | 0.79 | 0.79 |
| Methyl + frag (matched) | 206 dev | 0.82 | 0.79 |
| Integrated (plate-centered) | 206 dev | 0.83 | 0.82 |

Protein-only AUC drops 0.80 → 0.79 after plate centering, consistent with some uncorrected signal tracking plate rather than biology. Integrated gain over the matched methylation + fragmentomics baseline is small (ΔAUC ≈ 0.01).

## Locked 5 + 3 panel

**Methylation** (region · nearest gene · GC · ddPCR flag):

1. CpG_R0164 · GENE0980 · 0.867 · extreme GC
2. CpG_R1263 · GENE3373 · 0.835 · extreme GC
3. CpG_R0805 · GENE3061 · 0.854 · extreme GC
4. CpG_R1905 · GENE2844 · 0.836 · extreme GC
5. CpG_R0607 · GENE0044 · 0.589 · repeat overlap

All five were stable in ≥ 4/5 development folds. These are **discovery candidates, not ddPCR-ready markers**: four have extreme GC and one overlaps a repeat. Each requires in-silico sequence-uniqueness and local-CpG review, primer/probe design, and analytical LOD/precision testing before assay lock-down; keep a backup marker for each problematic target.

**Protein**: CYFRA21-1, GDF15, CEA — each nonzero-coefficient in all 5 plate-centered selection folds.

## Locked-holdout result (49 draws, 43 patients)

| Metric | Value |
|---|---:|
| ROC AUC | 0.958 |
| Average precision | 0.974 |
| Threshold (dev OOF, 95 % spec target) | 0.747 |
| Holdout sensitivity | 70.0 % |
| Holdout specificity | 100.0 % |

Caveat: with 43 patients, the 100 % specificity point has wide uncertainty. The 95 %-specificity target is not definitively achieved from one small holdout.

## With more time

1. Leave-one-site-out and plate-adjusted sensitivity analyses on top of the per-fold centering.
2. In-silico ddPCR design and wet-lab feasibility testing for all five methylation candidates; replace high-GC or repeat markers if assay performance is poor.
3. Patient-clustered bootstrap confidence intervals for AUC and for fixed-threshold operating points.
4. External validation in a site-balanced cohort with protein measurements at every site.
5. Try a stacked late-fusion ensemble and a nonlinear baseline (gradient boosting) to confirm the integrated-model ceiling is a data ceiling, not a modeling one.

## AI tools used

I'll be straight about this: a lot of this problem space was new to me, and I used Claude (Anthropic's Claude Code CLI) heavily — as a tutor, a sounding board, and a coding assistant. Specifically:

- **Learning the concepts.** I asked Claude to explain patient-grouped cross-validation, why leakage from preprocessing fitted outside the fold is a real problem on this kind of data, what plate/batch effects look like in protein assays, and why ddPCR cares about GC content and repeat overlap. The understanding is mine now; it wasn't at the start.
- **Weighing approaches.** For most design calls — early-fusion vs late-fusion, L1-logistic vs tree models, how strict to make the ddPCR feasibility filter, where to put the holdout boundary, whether to correct plate effects by median centering vs ComBat — I asked Claude to lay out the trade-offs and then I picked. The choices in this pipeline are the ones I'd defend; several of them I would not have known to make without that back-and-forth.
- **Code review and implementation.** Claude did a line-by-line review that flagged the uncorrected plate confound as the biggest unaddressed issue, helped me implement training-fold-safe per-plate median centering, and helped trim unused QC outputs, dead return values, and docstring typos.
- **Drafting.** This write-up was drafted with Claude and edited by me.

What I did myself: I read every change before accepting it, re-ran `pixi run test` and `pixi run analysis` after each round of edits, and sanity-checked the output numbers against the raw CSVs. If a reviewer wants to probe any specific choice in this pipeline, I can explain it on my own.
