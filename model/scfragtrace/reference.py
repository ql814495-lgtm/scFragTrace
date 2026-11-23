"""Single-cell module graph and node features: lineage index of the module site sets, normalised adjacency and
hop-distance buckets of the co-accessibility graph, and the 764 x 3 node features."""
import numpy as np

N_MODULES = 764
# module site set -> lineage index of the 7 classes that have single-cell modules (prostate has none)
LIN = {"breast_luminal": 0, "prog_breast": 0, "crc": 1, "prog_crc": 1, "gastric_foveolar": 2, "prog_gastric": 2, "lung_AT2": 3,
       "ovary": 4, "prog_ovary": 4, "panc_ductal": 5, "prog_panc": 5, "prog_bileduct": 5, "liver": 6}
N_LIN_WITH_MODULES = 7


def norm_adj(A):
    """Symmetric normalisation with self-loops: D^-1/2 (A + I) D^-1/2 (float32 array)."""
    A = np.asarray(A, np.float32) + np.eye(len(A), dtype=np.float32); d = A.sum(1); Dm = 1 / np.sqrt(d)
    return (A * Dm[:, None] * Dm[None, :]).astype(np.float32)


def hop_dist(A):
    """Shortest-path distance buckets 0, 1, 2, 3(+) on the unweighted graph (Graphormer spatial bias index)."""
    B = (A > 0).astype(np.int32); np.fill_diagonal(B, 0); D = np.full(A.shape, 3, np.int64); np.fill_diagonal(D, 0)
    R = np.eye(len(A), dtype=bool)
    for h in (1, 2):
        R2 = (R.astype(np.int32) @ B) > 0; D[R2 & (D == 3)] = h; R = R | R2
    return D


def graph_tensors(adj_sc):
    """Normalised adjacency and hop-distance matrix of the single-cell co-accessibility graph."""
    Ah = norm_adj(adj_sc)
    return Ah, hop_dist(Ah - np.diag(np.diag(Ah)))


def lineage_mask(module_set, n_classes=8):
    """M x (K+1) mask: column k = modules of class k (prostate, class 7, has none); last column = all modules.
    Returns (mask, hasmod)."""
    ML = np.array([LIN.get(str(x), -1) for x in module_set])
    cols = [ML == k for k in range(N_LIN_WITH_MODULES)] + [np.zeros(len(ML), bool)] * (n_classes - N_LIN_WITH_MODULES)
    mask = np.stack(cols + [np.ones(len(ML), bool)], 1)
    hasmod = np.array([1.0] * N_LIN_WITH_MODULES + [0.0] * (n_classes - N_LIN_WITH_MODULES), np.float32)
    return mask, hasmod


def module_node_features(counts, n_frag_total):
    """Node features of the M site modules from per-module raw counts.
    counts: (..., M, 6) with columns central, flank, UL, DL, UR, DR (GC-weighted fragment counts summed over the module's
    sites: midpoints within |d| <= 75 bp, midpoints 1000-1500 bp on both sides, and upstream/downstream fragment ends in
    the OCF windows [-70, -50) and [50, 70)); n_frag_total: total autosomal fragments of the sample.
    Features: central/flank ratio scaled by window widths (1002/151), log central counts per million fragments, and
    OCF = ((UR - DR) + (DL - UL)) / (UR + DR + DL + UL)."""
    counts = np.asarray(counts, np.float64); n = np.asarray(n_frag_total, np.float64)
    c, fl, UL, DL, UR, DR = (counts[..., j] for j in range(6))
    F = np.zeros(counts.shape[:-1] + (3,), np.float32)
    F[..., 0] = c / np.maximum(fl, 1e-9) * (1002 / 151)
    F[..., 1] = np.log1p(c / n[..., None] * 1e6)
    F[..., 2] = ((UR - DR) + (DL - UL)) / np.maximum(UR + DR + DL + UL, 1e-9)
    return F
