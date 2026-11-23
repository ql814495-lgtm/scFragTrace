"""Label-free priors (no training labels): the single-cell lineage footprint S, the copy-number (CNA) template
correlation and their combination PRIOR = zs(TFx * S) + zs(CNA).
The arithmetic is that of the paper; the outputs were checked against the results reported in the paper
(model/weights/VERIFICATION.md)."""
import numpy as np

# lineage -> site sets pooled into its footprint (single-cell sets + same-lineage TCGA tumour-gained / tumour-lineage sets)
POOL = {"breast": ["breast_luminal", "tg_BRCA", "tl_BRCA"], "colorectal": ["crc", "tg_COAD", "tl_COAD"],
        "gastric": ["gastric_foveolar", "tg_STAD", "tl_STAD"], "lung": ["lung_AT2", "tg_LUAD", "tl_LUAD", "tg_LUSC", "tl_LUSC"],
        "ovary": ["ovary"], "pancreas": ["panc_ductal"], "liver": ["liver", "tg_LIHC", "tl_LIHC"], "prostate": ["tg_PRAD", "tl_PRAD"]}
CLASSES = list(POOL)
CONTROL_SETS = ["closed", "ubiq_open"]
FOOTPRINT_SETS = [s for v in POOL.values() for s in v] + CONTROL_SETS     # the 19 site sets the footprint needs
MIN_REF_FOOTPRINT = 10      # a batch needs >= 10 non-cancer samples to be its own footprint reference


def zs(A):
    """Row-wise z-score across classes."""
    A = A - A.mean(1, keepdims=True)
    return A / (A.std(1, keepdims=True) + 1e-9)


def corr_t(A, TT):
    """Pearson correlation of each sample's arm-level log2 ratios (rows of A) with each cancer-type template (rows of TT,
    TCGA arm gain-minus-loss frequencies, Taylor et al. 2018)."""
    A = np.asarray(A, float)
    Az = A - A.mean(1, keepdims=True); Az /= Az.std(1, keepdims=True) + 1e-9
    Tz = (TT - TT.mean(1, keepdims=True)) / TT.std(1, keepdims=True)
    return Az @ Tz.T / A.shape[1]


def footprint_ratios(C, F, set_names, pool=POOL):
    """R (samples x lineages) = log((sum C + 1) / (sum F + 1)) over each lineage's pooled sets, and the same for the
    closed and ubiquitously open control sets (samples x 2)."""
    nm = list(set_names)
    def g(ss): return np.log((sum(C[:, nm.index(s)] for s in ss) + 1) / (sum(F[:, nm.index(s)] for s in ss) + 1))
    R = np.stack([g(pool[c]) for c in pool], 1)
    Cc = np.stack([g(["closed"]), g(["ubiq_open"])], 1)
    return R, Cc


def footprint_reference(R, Cc, ref):
    """Regression of R on [1, closed, ubiq_open] over the reference (non-cancer) samples; returns (coefficients 3 x K,
    SD of the reference residuals per lineage)."""
    A_ = np.c_[np.ones(len(R)), Cc]
    b = np.linalg.lstsq(A_[ref], R[ref], rcond=None)[0]
    return b, (R - A_ @ b)[ref].std(0)


def footprint_apply(R, Cc, b, sd):
    """S = -(R - [1, Cc] b) / sd (more open = higher)."""
    return -(R - np.c_[np.ones(len(R)), Cc] @ b) / sd


def lineage_footprint(C, F, set_names, cohort, cancer, pool=POOL, min_ref=MIN_REF_FOOTPRINT, fallback=None):
    """Single-cell lineage footprint S (samples x lineages).
    C, F: fragment-midpoint counts per sample and site set in the site centre (|d| <= 75 bp) and flanks (1-1.5 kb).
    Within each cohort (batch) R is regressed on the closed and ubiquitously open control sets using the cohort's non-cancer
    samples, and S = -residual / SD(residual of the reference samples). A cohort with fewer than ``min_ref`` non-cancer
    samples uses all non-cancer samples given (as in the paper) or, when ``fallback=(b, sd)`` is given, these stored
    pooled-reference parameters (the model's default; computed from the non-cancer samples of the paper's cohorts)."""
    R, Cc = footprint_ratios(C, F, set_names, pool)
    S = np.full_like(R, np.nan)
    cohort = np.asarray(cohort); cancer = np.asarray(cancer, bool)
    for co in np.unique(cohort):
        idx = np.where(cohort == co)[0]; ref = idx[~cancer[idx]]
        if len(ref) >= min_ref:
            b, sd = footprint_reference(R, Cc, ref)
        elif fallback is not None:
            b, sd = fallback
        else:
            b, sd = footprint_reference(R, Cc, np.where(~cancer)[0])
        S[idx] = footprint_apply(R[idx], Cc[idx], b, sd)
    return S


def label_free_prior(tf, S, CP):
    """PRIOR = zs(TFx * TS) + zs(CNA template correlation), TS = TFx * S; tf = ichorCNA tumour fraction per sample.
    The footprint term is written as ``zs(tf[:, None] * TS)`` with ``TS = S * tf``. Because zs()
    standardises each row, the per-sample factor TFx^2 cancels (up to the 1e-9 in the denominator, which matters only
    when TFx^2 * S is of the order of 1e-9 or TFx = 0): numerically the footprint term is zs(S)."""
    tf = np.asarray(tf, float); TS = np.nan_to_num(S) * tf[:, None]
    return zs(tf[:, None] * TS) + zs(CP)


def footprint_term(tf, S):
    """The footprint term of the prior, zs(TFx * TS) (see label_free_prior)."""
    tf = np.asarray(tf, float); return zs(tf[:, None] * (np.nan_to_num(S) * tf[:, None]))


def prior_inputs(tf, S, CP):
    """Per-sample conditioning vector of the GNN: [zs(TFx*CNA), zs(TFx*TS), CNA r, TFx]
    (8+8+8+1 for 8 classes), TS = TFx * S."""
    tf = np.asarray(tf, float)
    TS = np.nan_to_num(S) * tf[:, None]
    return np.concatenate([zs(tf[:, None] * CP), zs(tf[:, None] * TS), CP, tf[:, None] * np.ones((len(tf), 1))], 1).astype(np.float32)
