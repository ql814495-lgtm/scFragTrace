# Example data

## snyder_gse71378/ (58 samples, 1.9 MB)

Per-sample model inputs of the 58 plasma samples of Snyder et al., Cell 2016 (GEO GSE71378, open access; FinaleDB hg19
fragment files) used in the paper: 48 cancers (lung 9, pancreatic 8, breast 6, colorectal 3, liver 3, prostate 3,
ovarian 2, and 14 of types outside the 8 model classes: bladder, esophageal, head and neck, kidney, skin, testicular,
uterine) and 10 non-cancer controls (2 healthy, 4 inflammatory bowel disease, 4 systemic lupus erythematosus), one sample
per person. They are the inputs of the paper's models for this cohort. In the paper the Snyder cohort is a new cohort, used for testing (format: `model/docs/INPUTS.md`).

| File | Content |
|---|---|
| `samples.csv` | sample_id (FinaleDB / EE id), batch, GEO sample name, ichorCNA tumour fraction and ploidy, `reference_noncancer` (the 10 controls), diagnosis (GEO annotation), `true_class` (model class, empty for other types and controls) |
| `module_counts.npz` | 58 x 764 x 6 GC-weighted module counts + total fragments |
| `footprint_counts.npz` | centre / flank counts at the 19 footprint site sets |
| `arm_copy_number.csv` | 39 arm-level ichorCNA logR values |
| `global_features.npz` | 1,005 global fragmentomic features (global component of the cancer score) |
| `delfi_features.npz` | 1,063 DELFI features from FinaleToolkit (coverage component of the cancer score) |

## ichorcna_seg/

ichorCNA 1 Mb segment files (`<id>.cna.seg`) of three of these samples (EE86217 healthy, EE86226 ovarian, EE86238 lung):
the upstream copy-number step. `scfragtrace.features.arm_copy_number_from_seg` turns them into the rows of
`arm_copy_number.csv` (checked in `tests/test_features.py`).

## synthetic/

`synthetic_fragments.tsv.gz`: synthetic fragments around a few reference sites of chr21 and chr22 (simulated;
`model/examples/make_synthetic_fragments.py`), used to test the fragment-file feature extractor. Whole-genome fragment
files (1-10 GB each) are available from FinaleDB.
