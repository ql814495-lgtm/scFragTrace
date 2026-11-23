"""Smoke test on synthetic random data: imports the package, builds the label-free prior, fuses it with a global-LR
log-probability with weights chosen leave-one-cohort-out, scores it, and (if torch is installed) runs the residual GNN."""
import numpy as np
import pytest

import scfragtrace
from scfragtrace import priors, fusion, evaluation


def _synthetic(n=120, K=8, n_sets=30, seed=0):
    rng = np.random.default_rng(seed)
    cohort = np.array(["A", "B", "C", "D"])[rng.integers(0, 4, n)]
    cancer = rng.random(n) < 0.7
    y = rng.integers(0, K, n)
    tf = np.where(cancer, rng.uniform(0, 0.3, n), 0.0)
    names = sorted({s for v in priors.POOL.values() for s in v} | {"closed", "ubiq_open"})
    names += [f"extra{i}" for i in range(max(0, n_sets - len(names)))]
    F = rng.poisson(200, (n, len(names))).astype(float)
    C = rng.poisson(F * 0.15).astype(float)
    TT = rng.normal(0, 0.2, (K, 39)); ARM = rng.normal(0, 0.1, (n, 39))
    return dict(cohort=cohort, cancer=cancer, y=y, tf=tf, names=names, C=C, F=F, TT=TT, ARM=ARM, K=K)


def test_package_version():
    assert scfragtrace.__version__


def test_prior_fusion_scores():
    d = _synthetic()
    S = priors.lineage_footprint(d["C"], d["F"], d["names"], d["cohort"], d["cancer"])
    assert S.shape == (len(d["y"]), d["K"]) and np.isfinite(S).all()
    CP = priors.corr_t(d["ARM"], d["TT"])
    assert np.all(np.abs(CP) <= 1 + 1e-6)
    PRIOR = priors.label_free_prior(d["tf"], S, CP)
    assert np.allclose(PRIOR.mean(1), 0, atol=1e-6)
    rng = np.random.default_rng(1)
    glr = np.log(rng.dirichlet(np.ones(d["K"]), len(d["y"])))
    cohorts = ["A", "B", "C", "D"]
    S_u, W = fusion.unified_loco(PRIOR, glr, d["y"], d["cohort"], cohorts)
    assert S_u.shape == PRIOR.shape and set(W) == set(cohorts) and all(w in fusion.W_GRID for w in W.values())
    bal = evaluation.cohort_balanced_accuracy(d["y"], S_u, d["cohort"], cohorts)
    assert 0 <= bal <= 1
    assert 0 <= evaluation.topk(d["y"], S_u, 2) <= 1 and 0 <= evaluation.macro_f1(d["y"], S_u) <= 1
    lam, sc = fusion.choose_lambda(d["y"][:40], PRIOR[:40], rng.normal(size=(40, d["K"])))
    assert lam in fusion.LAMBDAS
    obs, lo, hi = evaluation.paired_bootstrap(d["y"], S_u, glr, d["cohort"], cohorts, n=50)
    assert lo <= hi
    assert np.allclose(fusion.fuse(PRIOR, glr, 0), PRIOR)


def test_gnn_forward_and_training():
    torch = pytest.importorskip("torch")
    from scfragtrace import gnn, reference
    rng = np.random.default_rng(0); M, K, n, fin = 40, 8, 64, 3
    A0 = (rng.random((M, M)) < 0.1).astype(np.float32); A0 = np.maximum(A0, A0.T); np.fill_diagonal(A0, 0)
    A = reference.norm_adj(A0); D = reference.hop_dist(A0)
    lin = rng.integers(-1, K - 1, M)                      # last class (e.g. prostate) has no modules
    mask = np.stack([lin == k for k in range(K - 1)] + [np.zeros(M, bool), np.ones(M, bool)], 1)
    hasmod = np.array([1.0] * (K - 1) + [0.0], np.float32)
    d = _synthetic(n=n, K=K)
    S = np.nan_to_num(priors.lineage_footprint(d["C"], d["F"], d["names"], d["cohort"], d["cancer"]))
    PRI = priors.prior_inputs(d["tf"], S, priors.corr_t(d["ARM"], d["TT"]))
    X = rng.normal(size=(n, M, fin)).astype(np.float32)
    net = gnn.ResidualLineageGNN(A, D, fin, mask, hasmod, nco=4)
    out = net(torch.tensor(X[:5]), torch.tensor(PRI[:5]))
    assert out["roo"].shape == (5, K) and torch.all(out["gnn"][:, -1] == 0)
    cid = np.unique(d["cohort"], return_inverse=True)[1]
    g, ep = gnn.train_wdat(net, X, PRI, d["y"], d["tf"], cid, np.arange(40), np.arange(40, 52), np.arange(52, 64), seed=0, epochs=2, patience=1)
    assert g.shape == (n, K) and ep >= 1
