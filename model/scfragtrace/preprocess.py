"""Batch-wise referencing of the per-sample inputs, as in the paper.

too_node_features   per-sample standardisation of the 764 x 3 node features over modules (snorm), then
                    within each batch z-scored against its non-cancer samples (all samples when < 3), clipped to +-5
det_node_features   snorm, then z-scored within each batch over ALL its samples (label-free), clipped to +-5
global_recentre     missing values -> reference mean, then each batch re-centred on its non-cancer samples (all
                    samples when < 3): x - mean_ref(batch) + reference mean
batch_z             z-score of a score within each batch over all its samples
batch_standardise   column-wise z-score of the DELFI coverage features within each batch (label-free)"""
import numpy as np

MIN_REF_NODES = 3


def snorm(A):
    A = A.astype(np.float32)
    return (A - A.mean(-2, keepdims=True)) / (A.std(-2, keepdims=True) + 1e-6)


def too_node_features(node, batch, ref, min_ref=MIN_REF_NODES):
    arr = snorm(node); out = np.zeros_like(arr)
    for b in np.unique(batch):
        m = np.where(batch == b)[0]; r = ref[m].copy()
        if r.sum() < min_ref:
            r[:] = True
        mu = arr[m][r].mean(0); sd = arr[m][r].std(0) + 1e-6
        out[m] = np.clip((arr[m] - mu) / sd, -5, 5)
    return out.astype(np.float32)


def det_node_features(node, batch):
    Z = snorm(node).copy()
    for b in np.unique(batch):
        m = batch == b; Z[m] = (Z[m] - Z[m].mean(0)) / (Z[m].std(0) + 1e-6)
    return np.clip(Z, -5, 5).astype(np.float32)


def global_recentre(X, batch, ref, hmean, min_ref=MIN_REF_NODES):
    X = np.where(np.isnan(X), hmean, X).astype(float)
    for b in np.unique(batch):
        ix = np.where(batch == b)[0]; r = ix[ref[ix]]
        if len(r) < min_ref:
            r = ix
        X[ix] = X[ix] - X[r].mean(0) + hmean
    return X


def batch_z(v, batch):
    v = np.nan_to_num(np.asarray(v, float)); out = np.zeros_like(v)
    for b in np.unique(batch):
        m = batch == b; out[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-9)
    return out


def batch_standardise(X, batch):
    """Column-wise z-score of a feature matrix within each batch over all its samples (label-free); NaN are ignored."""
    X = np.asarray(X, float); X = np.where(np.isfinite(X), X, np.nan); out = np.full_like(X, np.nan)
    for b in np.unique(batch):
        m = batch == b
        if m.sum() > 1:
            out[m] = (X[m] - np.nanmean(X[m], 0)) / (np.nanstd(X[m], 0) + 1e-9)
        else:
            out[m] = 0.0
    return out
