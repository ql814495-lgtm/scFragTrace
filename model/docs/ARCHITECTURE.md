# scFragTrace model architecture

This page describes the trained model shipped in `model/weights/`. Every formula below is implemented in
the package `model/scfragtrace/`, and its outputs were checked against the results reported in the paper
(`model/weights/VERIFICATION.md`).

Notation: sample i, class k in K = 8 tissue-of-origin classes (breast, colorectal, gastric, lung, ovary, pancreas
[incl. bile duct], liver, prostate), module m in M = 764, batch b (a cohort or sequencing run). `zs(.)` is the row-wise
z-score across the classes of one sample. TFx is the ichorCNA tumour fraction.

**Training design of the paper.** Every trained component is fitted on the Cristiano cohort (discovery cohort), which
also serves every selection step. All other cohorts are new cohorts, each tested once.

* Tissue of origin in new cohorts is the **label-free model**: 8 lineage footprints + 39 chromosome arms = 47
  features, free of trained parameters (section 3). This is the default of the shipped bundle (`mode="label_free"`).
* The graph network (section 5) and the global fragmentomic classifier (section 6) are used within the discovery
  cohort (`mode="full"`) and in the cancer score (section 7).

## 1. Overview

```
 fragment file (hg19)         ichorCNA (1 Mb bins, panel of normals)
        |                                 |
        +--> 764 module counts (x6) ------+----------------------------+
        |    -> node features x_i (764 x 3)                            |
        +--> 19 site-set counts C, F                                    |
        |    -> lineage footprint S_i (8)        arm logR a_i (39) -> CNA_i = corr(a_i, TCGA templates) (8)
        |                                                               |
        |            label-free PRIOR_i = zs(S_i) + zs(CNA_i)  <--------+   (label-free)       
        |                         |
        |      new cohorts:   score_i = PRIOR_i                                    (lambda = 0, w = 0)
        |
        |      discovery cohort (mode "full"):
        |    x_i --> [graph conv A x, A^2 x] -> [graph transformer, Graphormer bias] -> tokens h (764 x 32)
        |                    -> FiLM(TFx, CNA_i) -> lineage-masked attention readout -> residual G_i (8)
        |            final_i = PRIOR_i + mean_s lambda_s * G_i^(s)                (3 seeds)
        |            score_i = final_i + w * log p_global(i)
        |
        |            P_i(k)  = softmax_k score_i ;  evidence shares at argmax_k
        |
        +--> cancer score D_i = z_b(TFx) + z_b(module LR) + z_b(detection graph network)
                               + z_b(global fragmentomics LR) + z_b(DELFI coverage LR)      (equal, pre-specified weights)
```

## 2. Per-sample inputs (format: `model/docs/INPUTS.md`)

| Input | Shape | Definition |
|---|---|---|
| module counts | 764 x 6 + total | GC-weighted fragment counts summed over each module's sites: midpoints within 75 bp (central), midpoints 1,000-1,500 bp on both sides (flank), fragment starts / ends in [-70, -50) and [50, 70) (UL, DL, UR, DR); fragments 60-400 bp, MAPQ >= 30, autosomes |
| node features x | 764 x 3 | central/flank x 1002/151; log(1 + central per million fragments); OCF = ((UR - DR) + (DL - UL)) / (UR + DR + DL + UL) |
| footprint counts C, F | 19 sets | unweighted midpoint counts within 75 bp and 1,000-1,500 bp of the sites of 17 lineage sets and 2 control sets (closed, ubiquitously open); fragments 50-400 bp, MAPQ >= 30 |
| arm copy number a | 39 arms | ichorCNA 1 Mb logR, centred on the sample's median, mean per arm (DELFI arm map) |
| tumour fraction TFx | 1 | ichorCNA v0.3.2 (settings in INPUTS.md) |
| global features (optional) | 1,005 | fragment-length histogram (121), end motifs (256), 5 Mb bin ratios (628) |
| DELFI coverage features (optional) | 1,063 | FinaleToolkit DELFI output: 5 Mb GC-corrected short-to-long ratios, 5 Mb coverage shares, arm coverage shares |

## 3. Label-free prior (`scfragtrace.priors`)

**Lineage footprint.** For lineage k with pooled site sets P_k (single-cell set plus same-lineage TCGA tumour-gained and
tumour-lineage sets: breast = breast_luminal, tg/tl_BRCA; colorectal = crc, tg/tl_COAD; gastric = gastric_foveolar,
tg/tl_STAD; lung = lung_AT2, tg/tl_LUAD, tg/tl_LUSC; ovary = ovary; pancreas = panc_ductal; liver = liver, tg/tl_LIHC;
prostate = tg/tl_PRAD):

    R_ik = log((sum_{s in P_k} C_is + 1) / (sum_{s in P_k} F_is + 1))
    R_ik = b0_k + b1_k R_i,closed + b2_k R_i,ubiq_open + e_ik     fitted on the batch's non-cancer samples
    S_ik = - e_ik / SD_ref(e_k)

A batch with fewer than 10 non-cancer samples uses the pooled reference (b, SD) of the 422 non-cancer samples of the
paper's cohorts (`weights/reference/footprint_pooled_reference.npz`; the paper used the same pooled samples; among
the new cohorts this applies to Adalsteinsson, whose samples are all from cancer patients). A new deployment needs about
10-20 non-cancer plasma samples processed in the same batch (Methods, Supplementary Table 5).

**Copy-number template.** `CNA_ik = Pearson(a_i, T_k)` with T_k the TCGA arm-level gain-minus-loss frequency of cancer
type k (Taylor et al. 2018; `weights/reference/tcga_arm_templates.csv`).

**Prior.**

    PRIOR_ik = zs(S_i)_k + zs(CNA_i)_k

The package writes the footprint term as `zs(tf[:, None] * TS)` with `TS = S * tf`. Because zs() standardises
each sample's row, the per-sample factor TFx^2 cancels (up to the 1e-9 in the denominator), so the term equals zs(S_i):
the prior is independent of tumour fraction, as stated in the Methods. The prior is free of parameters fitted to cfDNA
labels and is defined for all eight classes, including liver and prostate cancer (the discovery cohort holds the other
six).

The 47 features of the label-free model are the 8 footprints S_i and the 39 arm log-ratios a_i; the score is an exact
additive function of them, which gives the per-feature evidence of Fig. 5e.

## 4. Single-cell module graph (`weights/reference/module_graph.npz`)

* Nodes: 764 modules. The sites of 17 site sets (7 adult epithelial lineages: breast_luminal, crc, gastric_foveolar,
  lung_AT2, liver, panc_ductal, ovary; 6 progenitor programmes: prog_breast, prog_crc, prog_gastric, prog_ovary,
  prog_panc, prog_bileduct; 4 blood lineages) were clustered within each set by their accessibility across adult CATLAS
  cell types and 18 sorted blood populations (k-means, 45 modules per set; 44 for prog_breast). 40,166 sites in total
  (`weights/reference/module_sites.npz`).
* Edges: single-cell co-accessibility. For two modules, the sum of the metacell co-accessibility r_epi >= 0.1 over their
  site pairs, divided by sqrt(n_i n_j); each module is linked to its 8 strongest partners; symmetrised, unweighted.
  2,775 edges, mean degree 7.3, 69 isolated modules.
* Normalised adjacency `A = D^-1/2 (A_sc + I) D^-1/2`; hop distance buckets 0, 1, 2, >= 3 for the attention bias.
* Lineage mask: class k may attend to the modules of its lineage (breast 89, colorectal 90, gastric 90, lung 45,
  ovary 90, pancreas incl. bile duct 135, liver 45; prostate 0, blood modules 0); the extra column (all 764
  modules) is a reserved detection slot.

## 5. Residual lineage GNN (`scfragtrace.gnn.ResidualLineageGNN`)

Node features are first referenced within the batch (z-score against the batch's non-cancer samples after a per-sample
standardisation over modules, clipped to +-5) and then standardised with the mean / SD of the seed's training samples
(stored in `too_node_standardisation.npz`), clipped to +-5.

    H0   = GELU( [A x, A^2 x] W_in + E ),   E = learned node embedding (764 x 32)         d = 32, dropout 0.2
    H    = H0 + Attn(LN(H0)),   Attn: 4 heads, logits q_u.k_v / sqrt(8) + b_h[hop(u,v)] + beta_h * A_uv * sqrt(764)
    H    = H + FFN(LN(H)),      FFN: 32 -> 64 -> 32, GELU
    (gamma, delta) = W_film [TFx_i, CNA_i] ;  H' = H * (1 + gamma) + delta                 FiLM conditioning
    alpha_uk = softmax over u in M_k of (W_att H'_u)_k                                    lineage-masked attention
    G_ik = (sum_u alpha_uk  H'_u . v_k + c_k) * hasmod_k                                  hasmod = 0 for prostate
    roo_ik = PRIOR_ik + G_ik                                                              training output

The input of the graph convolution is [A x, A^2 x]. A batch head (32 -> 32 -> n
batches, ReLU) on the mean token is attached through a gradient-reversal layer during training.

**Training (variant "wdat", `scfragtrace.gnn.train_wdat`).** omega_i = clip(TFx_i / 0.05, 0.05, 1).

    L = mean_i omega_i CE_w(roo_i, y_i)                         class weights from TFx-weighted class counts
      + 0.5 * mean_j omega_j KL(softmax(PRIOR_j) || softmax(roo_j))    j: 16 draws from the training patients
      + 0.3 * CE(batch head(GRL(mean_u H'_u)), batch)    (on both mini-batches)

AdamW (lr 1e-3, weight decay 1e-2), mini-batches of 16, at most 60 epochs, early stopping on the validation loss
(patience 12). Shipped bundle `cristiano_only`: the 243 Cristiano cancers with a class label (six of the eight classes),
three seeds, each with a stratified 20% validation split of that cohort (`train_test_split`, random_state = seed) for
early stopping and for the residual weight lambda_s in {0, 0.25, 0.5, 1} (balanced accuracy of PRIOR + lambda G^(s) on
the validation split; ties -> smaller lambda).

    final_ik = PRIOR_ik + (1/3) sum_s lambda_s G^(s)_ik          (mode "full")

**Use in new cohorts.** In new cohorts the paper uses the label-free model, which is defined for all eight classes
(the discovery cohort holds six of them); the `full` mode is the model within the discovery cohort. Extended Data
Fig. 8e of the paper shows `mode="full"` in new cohorts.

## 6. Global fragmentomic LR and late fusion (`scfragtrace.fusion`)

Global features are re-centred per batch on its non-cancer samples (`x - mean_ref(batch) + mean_ref(discovery)`;
missing values -> discovery reference mean), standardised, and fed to an L2 multinomial logistic regression
(C = 0.3, balanced class weights). Fusion:

    score_ik = final_ik + w * log p_global(k | i)

Within the discovery cohort, w was selected from a grid up to 100 by inner cross-validation within each training fold,
and per-class offsets were calibrated on the training folds; the cross-validated numbers of the paper use a
separate fit for each fold, and the shipped parameters are the full-cohort fit. In new cohorts w = 0. The shipped global LR (trained on the 243 Cristiano cancers) can be switched on
in full mode with `--w-global`.

## 7. Outputs

For each plasma genome the model returns three outputs (Supplementary Fig. 1 of the paper).

1. **Cancer score**:

        D_i = sum_m z_b(component m of sample i),   m = TFx, module LR, detection graph network, global LR, DELFI coverage LR

   z_b = z-score within the batch over all its samples (label-free). The five components carry equal, pre-specified
   weights. All trained components were fitted on the 539 genomes of the Cristiano cohort (275 cancer, 264 non-cancer):
   * module LR: L2 logistic regression (C = 0.01, balanced) on the 764 x 3 node features standardised within the batch;
   * detection graph network (`scfragtrace.gnn.DetectionGNN`): the Region branch of section 5, attention pooling over
     all modules, one logit; class-weighted binary cross-entropy; mean of three networks, each with a stratified 20%
     validation split of the Cristiano cohort for early stopping;
   * global fragmentomics LR: L2 logistic regression (C = 0.3, balanced) on the 1,005 global features;
   * DELFI coverage LR: the 1,063 DELFI features standardised within the batch, class-balanced L2 logistic regression
     with the penalty chosen by 5-fold cross-validation within the Cristiano cohort (probability output).

   The sum runs over the components whose input files are present (`global_features.npz` and `delfi_features.npz`
   are optional), and the column `detection_components` of the output lists the components used. D is a relative
   score within a batch; the paper reports a sample as positive when D exceeds the threshold that gives 95%
   specificity among the non-cancer samples of the batch.
2. **Tissue of origin**: the eight class scores, reported as a ranked list (`predicted_class`, `second_class`).
   `P_k = softmax_k(score_i)` is a relative confidence.
3. **Evidence** (LCC decomposition): terms
   footprint = zs(S), copy number = zs(CNA), graph network = (1/3) sum_s lambda_s G^(s), global = w log p_global (the
   last two are zero in label-free mode). For the predicted class k*, share_t = |t_k* - mean_k t_k| /
   sum_t' |t'_k* - mean_k t'_k|. (The paper reports the same quantity at the true class, averaged over patients.) The
   per-class terms are written to `evidence_terms.csv`.

Tumour fraction (ichorCNA) is reported alongside as an input.

## 8. Swapping the network weights

A bundle is a folder with `manifest.json` (architecture, file list with SHA-256, lambdas, fusion and detection
weights), `too_gnn_seed*.pt`, `too_node_standardisation.npz`, `det_gnn_seed*.pt`, `det_module_lr.npz`,
`det_global_lr.npz`, `det_delfi_lr.npz` and `global_lr.npz`; the graph and reference tables are shared in `weights/reference/`. To deploy other GNN weights with the
same architecture, put new `too_gnn_seed*.pt` files (state_dicts without the buffers r.A, r.D, mask, hasmod; see
`scfragtrace.gnn.state_dict_for_saving`) and their node standardisation into a copy of the bundle, update `lambdas`,
`gnn_weights` and the checksums in `manifest.json`, or retrain with `model/training/train_model.py`.
