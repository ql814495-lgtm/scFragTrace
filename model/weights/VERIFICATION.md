# Verification of the shipped model against the paper

Environment: Python 3.13, numpy 2.5.3, scikit-learn 1.9.1, torch 2.15.0 (CPU). Checked on 2026-11-22.

The bundle `cristiano_only/` was trained with

```
python model/training/train_model.py --inputs <labelled inputs of the Cristiano cohort> \
       --config model/training/configs/cristiano_only.json --out model/weights/cristiano_only --threads 4
```

on the 539 Cristiano genomes (about 55 minutes). The comparisons below used the per-sample inputs of all five cohorts
with labels (1,124 samples; controlled access, regenerated from the fragment files), and the shipped Snyder example
for the open-access part.
Paper values are the results reported in the paper.

## Tissue of origin in new cohorts (label-free model, free of trained parameters)

Balanced accuracy per cohort, patients with one of the 8 classes; mean of the four cohorts.

| | Adalsteinsson | Jiang | Snyder | Sun | Mean | Paper |
|---|---|---|---|---|---|---|
| All patients (n = 198 / 90 / 34 / 12) | 0.340 | 0.344 | 0.554 | 0.167 | **0.351** | 0.351 |
| TFx >= 5% (n = 67 / 18 / 15 / 4) | 0.591 | 0.778 | 0.806 | 0.500 | **0.669** | 0.669 |

Identical to the paper. The Snyder row is reproduced by the demo from the shipped open-access inputs.

## Cancer score in new cohorts (five components, equal weights)

AUROC, all genomes of each cohort (Jiang 90 cancer / 135 non-cancer; Snyder 48 / 10; Sun 18 / 13).

| Component | Jiang | Snyder | Sun | Mean | Paper (Jiang / Snyder / Sun / mean) |
|---|---|---|---|---|---|
| ichorCNA TFx | 0.795 | 0.715 | 0.487 | 0.665 | 0.795 / 0.715 / 0.487 / 0.665 |
| Module LR | 0.665 | 0.604 | 0.667 | 0.645 | 0.665 / 0.604 / 0.667 / 0.645 |
| Detection graph network (3 seeds) | 0.555 | 0.619 | 0.684 | 0.619 | 0.572 / 0.619 / 0.885 / 0.692 |
| Global fragmentomic LR | 0.474 | 0.785 | 0.979 | 0.746 | 0.474 / 0.785 / 0.979 / 0.746 |
| DELFI coverage LR (C = 1.0 by 5-fold CV, as in the paper) | 0.922 | 0.773 | 0.855 | 0.850 | 0.922 / 0.773 / 0.855 / 0.850 |
| **Cancer score** | **0.770** | **0.742** | **0.936** | **0.816** | 0.771 / 0.744 / 0.944 / 0.820 |

Four components are identical to the paper to three decimals. The detection graph network of the bundle is a
separate training run with explicit seeds (0, 1, 2); the cancer score agrees with the paper to within 0.008 per
cohort and 0.004 in the mean.

The graph network added to the label-free model (`mode="full"`) is the model the paper uses within the discovery
cohort; the label-free model is the default mode and the model of the new cohorts.

## Package code against the results of the paper

| Check | Result |
|---|---|
| Lineage footprint, copy-number correlation, conditioning vector, batch-referenced node features, batch-recentred global features (package output against the values behind the results reported in the paper) | identical (max abs difference 0; copy number 5e-15) |
| Graph tensors (normalised adjacency, hop distances) from `reference/module_graph.npz` | identical |
| Arm copy number from ichorCNA `.cna.seg` (`scfragtrace.features`) against the arm copy-number values used in the paper (4 samples) | identical (max abs difference 2e-16) |
| Fragment-file extractor against a direct per-site count on a synthetic fragment file | identical (`tests/test_features.py`) |

## Scope of the checks

- The shipped parameters are the full-cohort fit. The cross-validated results within the discovery cohort use a
  separate fit for each fold and are reported in the paper.
- The saved files are checked through the demo, which is deterministic (two fresh runs give identical outputs,
  `tests/test_demo.py`).
- Global-LR inputs with missing values are imputed with the training means; in the Snyder example this affects 47
  cells of the 58 x 1,005 matrix.
