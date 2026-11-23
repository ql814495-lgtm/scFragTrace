# scFragTrace

scFragTrace detects cancer and infers its tissue of origin from low-coverage plasma cell-free DNA (cfDNA) fragment
files. It reads cfDNA fragmentation at cell-type-specific open chromatin sites defined from adult single-nucleus
ATAC-seq (247,039 cells, 43 cell types) and combines four components:

1. a **single-cell lineage footprint**: the fragment-coverage dip at the open sites of each lineage, referenced to
   non-cancer samples of the same batch;
2. a **copy-number template**: the correlation of the sample's arm-level copy number (ichorCNA) with the TCGA
   gain-and-loss template of each cancer type;
3. a **lineage-constrained graph network** on 764 single-cell site modules linked by single-cell co-accessibility
   (graph convolution, graph transformer, lineage-wise pooling);
4. **global fragmentomic features** (fragment length, 5' end motifs, 5 Mb fragmentation) in a regularised logistic
   regression.

**Design.** Every trained component is fitted on one cohort, Cristiano et al. 2019
(the discovery cohort). All other cohorts are new cohorts: used for testing, each tested once.

- **Tissue of origin in new cohorts** uses the label-free model: components 1 and 2, 47 features (8 lineage
  footprints + 39 chromosome arms), free of trained parameters.
- **Within the discovery cohort**, tissue of origin adds the graph network and the global features.
- **Cancer score**: five components with equal, pre-specified weights (ichorCNA tumour fraction (TFx), module logistic
  regression, detection graph network, global fragmentomic logistic regression, DELFI coverage logistic regression),
  each standardised within the cohort under test.

Each tissue-of-origin prediction comes with its evidence decomposition: the share of the lineage footprint and of the
copy-number template (and, within the discovery cohort, of the graph network) in the score of the predicted class.

## Contents

| Folder | Content |
|---|---|
| [`model/scfragtrace/`](model/README.md) | The `scfragtrace` Python package (API and command line). |
| `model/weights/` | Trained parameters (`cristiano_only/`) and the single-cell reference site sets, co-accessibility graph and TCGA copy-number templates (`reference/`). |
| `model/docs/` | Architecture (`ARCHITECTURE.md`) and input format with the fragment-file to feature step (`INPUTS.md`). |
| `model/examples/` | A worked example on open-access plasma samples with the expected output. |
| `model/training/` | Training script and configuration. |
| `tests/` | Tests of the package, the feature extractor and the example. |

## Main results

| Task | Setting | scFragTrace |
|---|---|---|
| Tissue of origin, 8 cancer types | 334 patients of four new cohorts, TFx >= 5% (n = 104) | balanced accuracy 0.67 |
| Tissue of origin, 8 cancer types | all 334 new-cohort patients | balanced accuracy 0.35 |
| Tissue of origin, 6 classes | discovery cohort, cross-validation | balanced accuracy 0.70 |
| Cancer detection | three new cohorts | mean AUROC 0.82 |
| Cancer detection | discovery cohort, cross-validation | mean AUROC 0.98 |

## Installation

```bash
conda env create -f environment.yml          # Python 3.13, numpy, pandas, scikit-learn, PyTorch (CPU), ...
conda activate scfragtrace
pip install -e .                              # installs the package from model/ and the `scfragtrace` command
python -m pytest -q                           # package, feature extractor and demo tests (a few minutes on a laptop CPU)
```

The model runs with numpy, pandas, scipy, scikit-learn and PyTorch (CPU). Feature extraction from fragment files
additionally uses R with ichorCNA v0.3.2 (`model/docs/INPUTS.md`).

## Quick start: run the trained model on the example data

```bash
python model/examples/run_demo.py
```

This loads the trained model (`cristiano_only`), runs it on 58 open-access plasma samples of Snyder et al. 2016 (GEO
GSE71378; 48 cancers of 13 types and 10 non-cancer controls), writes
`model/examples/output/cristiano_only/predictions.csv` (cancer score, predicted tissue, class probabilities, evidence
shares), `class_scores.csv` and `evidence_terms.csv`, prints the accuracy against the known diagnoses, and checks the
result against `model/examples/expected_output/`. Snyder is a new cohort for this model.

Expected result on these 58 samples: tissue of origin (label-free model, 34 patients of the 8 classes) balanced
accuracy 0.554, and 0.806 at TFx >= 5%, as in the paper; cancer score AUROC 0.742 (0.744 in the paper, which used
a separate training run of the graph networks; `model/weights/VERIFICATION.md`).

The same from the command line or from Python:

```bash
scfragtrace predict --inputs model/examples/data/snyder_gse71378 --out predictions/
```

```python
from scfragtrace import load_model, load_inputs
model = load_model("cristiano_only")
result = model.predict(load_inputs("model/examples/data/snyder_gse71378"))
result.table[["sample_id", "detection_score", "predicted_class", "share_footprint", "share_copy_number"]]
```

For your own data: compute the inputs from fragment files and ichorCNA (`scfragtrace extract`, `model/docs/INPUTS.md`)
and include non-cancer controls processed in the same batch (about 10-20).

## Data sources

| Cohort | Source | Access |
|---|---|---|
| Cristiano et al. 2019 (discovery cohort) | FinaleDB; EGA EGAD00001005339 | controlled |
| Adalsteinsson et al. 2017 | FinaleDB; dbGaP phs001417 | controlled |
| Snyder et al. 2016 (example data of this repository) | FinaleDB; GEO GSE71378 | open |
| Jiang et al. 2015 | FinaleDB; EGA EGAS00001001024 | controlled |
| Sun et al. 2019 | FinaleDB; EGA EGAS00001003116, EGAS00001003117 | controlled |
| CRAG | Zenodo, doi:10.5281/zenodo.6914806 | open |

Reference data: adult single-nucleus ATAC-seq atlas (CATLAS, http://catlas.org/humanenhancer), sorted blood ATAC-seq
(Ulirsch et al. 2019), TCGA tumour ATAC-seq peaks (Corces et al. 2018) and TCGA arm-level copy number
(Taylor et al. 2018).

## Citation and licence

The scFragTrace authors. scFragTrace: single-cell chromatin footprints enable cancer detection and tissue-of-origin
inference from low-coverage cell-free DNA. See `CITATION.cff`. MIT licence (`LICENSE`).
