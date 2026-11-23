# Trained model

| Folder | Content | Trained on |
|---|---|---|
| `reference/` | shared reference files: `module_graph.npz` (764-module single-cell co-accessibility graph, module site sets), `module_sites.npz` (40,166 site coordinates -> module), `footprint_sites.npz` (67,643 sites of the 19 footprint sets), `tcga_arm_templates.csv` (TCGA arm gain-minus-loss frequencies, 8 classes x 39 arms), `arm_bins.tsv` (5 Mb bins with arm labels), `footprint_pooled_reference.npz` (footprint regression pooled over 422 non-cancer samples of Cristiano, Jiang, Snyder and Sun; used for batches with fewer than 10 non-cancer controls) | public references. The pooled reference is an aggregate of non-cancer samples from four cohorts; it is free of trained class parameters, and among the paper's new cohorts it applies to Adalsteinsson, whose samples are all from cancer patients. |
| `cristiano_only/` | the model of the paper (3 seeds per graph network) | **Cristiano cohort (the discovery cohort).** Tissue-of-origin graph network and global classifier: the 243 Cristiano cancers with a class label (stratified 20% validation split per seed). Cancer score (module LR, detection graph network, global LR, DELFI coverage LR): the 539 Cristiano genomes (275 cancer, 264 non-cancer). Every fit and selection step used this cohort. |

The default tissue-of-origin mode of the bundle is `label_free` (47 features, free of trained parameters), which is the model
the paper uses in new cohorts; `mode="full"` adds the graph-network residual, which the paper uses within the
discovery cohort. The cancer score always uses the five components of the bundle with equal weights.

Configuration: `model/training/configs/cristiano_only.json`. Architecture and formulas:
`model/docs/ARCHITECTURE.md`. Checks against the paper: `VERIFICATION.md`.

How this bundle relates to the runs of the paper:

- The label-free tissue-of-origin model and the TFx, module-LR, global-LR and DELFI-LR components are deterministic and
  agree with the paper's values (`VERIFICATION.md`).
- The graph networks of the bundle were trained with the package code and explicit seeds (0, 1, 2); in the runs of
  the paper the network initialisation followed the random state of the process. The graph-network scores are those of a
  separate training run; `VERIFICATION.md` compares the resulting cancer score with the paper.
- The shipped parameters are the fit on the whole discovery cohort. The cross-validated results within the discovery
  cohort that the paper reports use a separate fit for each fold.
- Results with several training cohorts are reported as a reference in Supplementary Table 6 of the paper.
