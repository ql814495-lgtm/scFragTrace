"""Metrics used in the paper."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def cohort_balanced_accuracy(y, S, cohort, cohorts, mask=None):
    """Mean over cohorts of the within-cohort balanced accuracy (each cohort weighted equally)."""
    y = np.asarray(y); cohort = np.asarray(cohort); v = []
    for c in cohorts:
        m = (cohort == c) if mask is None else (cohort == c) & mask
        v.append(balanced_accuracy_score(y[m], np.asarray(S)[m].argmax(1)))
    return float(np.mean(v))


def topk(y, S, k=1):
    """Fraction of samples whose true class is among the k highest scores."""
    return float(np.mean((np.argsort(-np.asarray(S), 1)[:, :k] == np.asarray(y)[:, None]).any(1)))


def macro_f1(y, S):
    return float(f1_score(y, np.asarray(S).argmax(1), average="macro"))


def paired_bootstrap(y, S_a, S_b, cohort, cohorts, n=2000, seed=0, mask=None):
    """Difference in cohort-mean balanced accuracy (a - b), patients resampled within cohort; returns (obs, lo, hi)."""
    rng = np.random.default_rng(seed); cohort = np.asarray(cohort)
    base = {t: np.where((cohort == t) & (True if mask is None else mask))[0] for t in cohorts}
    def bal(S, idx): return np.mean([balanced_accuracy_score(y[ii], S[ii].argmax(1)) for ii in idx.values()])
    obs = bal(S_a, base) - bal(S_b, base)
    d = [bal(S_a, b) - bal(S_b, b) for b in ({t: rng.choice(ii, len(ii)) for t, ii in base.items()} for _ in range(n))]
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
