# scFragTrace model

```
model/
  scfragtrace/          Python package: load_model / predict API, command line, label-free model, graph networks,
                        training, fragment-file feature extractor
  weights/              trained model
    reference/          shared reference files: module graph, module and footprint site coordinates, TCGA arm
                        templates, DELFI arm map, pooled footprint reference
    cristiano_only/     the model of the paper: every trained component fitted on the Cristiano cohort
    README.md           what the bundle was trained on; VERIFICATION.md: checks against the paper's numbers
  docs/
    ARCHITECTURE.md     inputs, module graph, graph network, label-free model, cancer score, outputs (with diagram)
    INPUTS.md           input folder format, batch referencing, fragment file -> features (steps, ichorCNA settings)
  examples/
    run_demo.py         complete worked example (load model, predict, evaluate, compare with expected output)
    data/snyder_gse71378/   per-sample inputs of 58 open-access samples (Snyder et al. 2016, GEO GSE71378)
    data/ichorcna_seg/      three ichorCNA segment files of these samples (upstream copy-number step)
    data/synthetic/         synthetic (simulated) fragment file for the feature extractor
    expected_output/    reference predictions; output/ is written by run_demo.py
  training/
    train_model.py      trains a bundle from labelled inputs; configs/cristiano_only.json is the shipped configuration
```

## What the model does

| Output | How it is computed | Trained on |
|---|---|---|
| Tissue of origin, default (`mode="label_free"`) | label-free 47-feature model: 8 lineage footprints, each referenced to the non-cancer samples of the same batch, plus the correlation of 39 chromosome-arm copy-number values with the TCGA template of each cancer type | public references (free of trained parameters). This is the model the paper uses in new cohorts. |
| Tissue of origin, `mode="full"` | label-free score plus the residual of the lineage-constrained graph network (3 seeds) | 243 Cristiano cancers with a class label. This is the model the paper uses within the discovery cohort. |
| Cancer score | sum of five components with equal, pre-specified weights, each standardised within the batch: ichorCNA TFx, module logistic regression, detection graph network (3 seeds), global fragmentomic logistic regression, DELFI coverage logistic regression | 539 Cristiano genomes (275 cancer, 264 non-cancer) |

`run_demo.py --prior-only` and `scfragtrace.model.predict_prior_only` run the label-free model on its own, independently
of a bundle.

## Using the model

```python
from scfragtrace import load_model, load_inputs
model = load_model("cristiano_only")                 # verifies the SHA-256 checksums of all weight files
x = load_inputs("model/examples/data/snyder_gse71378")
res = model.predict(x)                               # res.table, res.scores, res.terms; mode="full" adds the graph network
res.save("predictions/")
```

```bash
scfragtrace predict --inputs <input dir> --out <output dir> [--bundle cristiano_only|<path>] [--mode label_free|full]
scfragtrace extract --manifest samples.tsv --out <input dir>      # fragment files + ichorCNA -> input folder
scfragtrace info --bundle cristiano_only                          # print the manifest
python model/examples/run_demo.py                                 # worked example
```

Without installation, put `model/` on `PYTHONPATH` (`PYTHONPATH=model python -m scfragtrace predict ...`).

### Outputs (`predictions.csv`, one row per sample)

| Column | Meaning |
|---|---|
| `detection_score` | cancer score: sum of the within-batch z-scores of the available components (`detection_components` lists them); higher = more likely cancer. It ranks the samples within a batch (a relative score). |
| `detection_module_lr`, `detection_gnn`, `detection_global_lr`, `detection_delfi_lr` | raw components of the cancer score |
| `predicted_class`, `second_class` | tissue of origin with the highest / second highest score among breast, colorectal, gastric, lung, ovary, pancreas (incl. bile duct), liver, prostate |
| `P_<class>` | softmax of the class scores (normalised relative confidence) |
| `share_footprint`, `share_copy_number`, `share_gnn`, `share_global` | evidence decomposition at the predicted class (sum to 1; the last two are 0 in label-free mode) |
| `tissue_mode` | `label_free` or `full` |

`class_scores.csv` holds the per-class label-free score, graph-network residual and final score; `evidence_terms.csv`
the per-class terms of the decomposition. The model assigns one of the 8 classes to every sample, including non-cancer
samples and cancers of other types; read the tissue call together with the cancer score and the TFx.

## Inputs

Per sample: 764 x 6 module counts, fragment counts at 19 site sets, 39 arm-level copy numbers and the TFx from
ichorCNA, and the batch it was processed in with a flag for known non-cancer controls (`docs/INPUTS.md`). The global
fragmentomic features and the DELFI coverage features are optional; the cancer score sums the components whose inputs
are present, and `detection_components` lists them. The label-free model references each footprint to non-cancer
plasma of the same batch, so a new batch needs a small set of non-cancer samples processed with the same pipeline
(about 10-20; Methods and Supplementary Table 5 of the paper).

## Retraining

`python model/training/train_model.py --inputs <labelled input dir> --config model/training/configs/cristiano_only.json --out <bundle dir>`.
The labelled training inputs (539 Cristiano genomes) derive from a controlled-access cohort (EGA EGAD00001005339);
they are rebuilt from the fragment files with `scfragtrace extract` and the steps in
`docs/INPUTS.md`.
