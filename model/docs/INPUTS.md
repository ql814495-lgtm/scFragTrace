# Model inputs and the fragment-file -> feature step

## 1. Input folder (read by `scfragtrace.load_inputs` and `scfragtrace predict --inputs`)

| File | Content |
|---|---|
| `samples.csv` | one row per sample. Required columns: `sample_id`; `batch` (samples sequenced and processed together, e.g. one cohort or run); `tumour_fraction` (ichorCNA); `reference_noncancer` (True for samples known to be non-cancer controls of that batch, False otherwise). Other columns are kept. For training also `cancer` (1/0), `true_class` and `train_too` (see `scfragtrace/train.py`). |
| `module_counts.npz` | `ids` (n), `counts` (n x 764 x 6: central, flank, UL, DL, UR, DR per module), `n_frag_total` (n) |
| `footprint_counts.npz` | `ids`, `set_names` (must include the 19 sets of `scfragtrace.priors.FOOTPRINT_SETS`), `C`, `F` (n x sets) |
| `arm_copy_number.csv` | `sample_id` + 39 arm columns 1p ... 22q (order as `weights/reference/tcga_arm_templates.csv`) |
| `global_features.npz` | optional: `ids`, `X` (n x 1,005; NaN = missing bin), `feature_names`. Input of the global fragmentomics component of the cancer score, and of the global classifier in full mode. |
| `delfi_features.npz` | optional: `ids`, `X` (n x 1,063 DELFI features computed with FinaleToolkit, in the order of section 2.1 step 6). Input of the DELFI coverage component of the cancer score. |

Worked example: `model/examples/data/snyder_gse71378/` (58 open-access samples of Snyder et al. 2016).

### Batch referencing (the role of non-cancer controls)

Fragment features shift between cohorts. As in the paper, each batch is referenced to itself, using its non-cancer
samples, independently of its tissue labels:

* footprint S: regression on the control sets fitted on the batch's non-cancer samples if it has >= 10, otherwise the
  stored pooled reference of the paper's 422 non-cancer samples is used;
* tissue-of-origin node features and global features: z-scored / re-centred on the batch's non-cancer samples if it
  has >= 3, otherwise on all samples of the batch;
* cancer score: every component is standardised within the batch over all its samples (label-free); the score ranks
  samples within a batch, so a batch contains both cancer and non-cancer samples. The sum runs over the components
  whose input files are present.

A batch that consists of cancer samples gets tissue-of-origin predictions through the pooled reference. In the paper,
about 10-20 non-cancer samples of the same batch were sufficient for the label-free model, and a same-batch reference
is the recommended setting (Methods, "Reference set for deployment"; Supplementary Table 5).

## 2. From a fragment file to the inputs

### 2.1 Paper pipeline (exact features)

The shipped example inputs were produced with the following steps from FinaleDB hg19 fragment files
(`<id>.hg19.frag.tsv.bgz`):

1. **ichorCNA (tumour fraction, 1 Mb copy number).**
   Fragment midpoints (MAPQ >= 30) are counted in 1 Mb bins and written as a WIG file;
   `runIchorCNA.R` (v0.3.2) is run on it with:
   `--ploidy "c(2,3)" --normal "c(0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.995,0.999)" --maxCN 5 --gcWig gc_hg19_1000kb.wig
   --mapWig map_hg19_1000kb.wig --centromere GRCh37.p13_centromere_UCSC-gapTable.txt --normalPanel pon_half_median.rds
   --includeHOMD False --chrs "c(1:22, 'X')" --chrTrain "c(1:22)" --estimateNormal True --estimatePloidy True
   --estimateScPrevalence False --scStates "c()" --txnE 0.9999 --txnStrength 10000`.
   The panel of normals was built from 107 healthy samples of the discovery cohort (Cristiano et al.) and derives from
   controlled-access data. Build your own panel with ichorCNA's `createPanelOfNormals.R` from healthy
   samples processed like yours. Tumour fraction (TFx) = "Tumor Fraction" of `<id>.params.txt`.
2. **Arm copy number.** Column 6 (logR) of `<id>.cna.seg` (non-NA values), centred on the
   sample's median over autosomal bins, mean per arm using the DELFI 5 Mb arm map (`weights/reference/arm_bins.tsv`).
   Same arithmetic: `scfragtrace.features.arm_copy_number_from_seg` (checked exactly on `examples/data/ichorcna_seg/`).
3. **Module counts.** GC expectation per fragment length x GC bin from the hg19 2bit genome; GC-weighted site
   profiles at the module sites (fragments 60-400 bp, MAPQ >= 30, >= 90% non-N bases); sum per module with
   `weights/reference/module_sites.npz`.
4. **Footprint counts.** Fragment midpoints within 75 bp (centre) and 1,000-1,500 bp (flank) of the sites of the
   single-cell and TCGA site sets; all fragments 50-400 bp, MAPQ >= 30, unweighted. The 19 sets used are in
   `weights/reference/footprint_sites.npz`.
5. **Global features (optional).** Fragment-length histogram (121 features), end motifs (256) and 5 Mb bin ratios
   (628), 1,005 features in total.
6. **DELFI coverage features (optional).** FinaleToolkit `delfi` on the fragment file (per-sample `<id>.delfi.tsv`),
   then per 5 Mb bin `ratio_corrected`, the share of
   `num_frags_corrected` per bin and per chromosome arm, concatenated to 1,063 features.

### 2.2 Compact extractor in the package

`scfragtrace.features.extract_fragment_features(fragment_file, reference_dir)` computes the module and footprint counts
in one pass with the same windows and filters, and `scfragtrace extract --manifest samples.tsv --out <dir>` writes a
complete input folder from fragment files plus ichorCNA outputs. Manifest columns: `sample_id`, `batch`,
`reference_noncancer`, `fragments`, `seg` and `tumour_fraction` (or `params`).

Relation to the paper pipeline: the module counts of the paper are GC-weighted (per length x GC bin, from the hg19
genome) and use fragments with >= 90% non-N bases. The compact extractor counts every fragment with weight 1 by default
and applies GC weights when a `gc_weight` function is passed. The numbers of the paper were obtained with the
GC-weighted counts of section 2.1; pass `gc_weight` for the same setting. The extractor is tested on a synthetic
fragment file against a direct per-site count (`tests/test_features.py`; file `examples/data/synthetic/`).
