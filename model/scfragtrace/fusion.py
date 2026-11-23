"""Score fusion and weight selection.

final(lambda)  = PRIOR + lambda * GNN residual, lambda in {0, .25, .5, 1} chosen on a validation split of the training data
unified(w)     = final + w * log p(global fragmentomic LR), w chosen leave-one-cohort-out (or on the other CV folds)
No label of the test cohort is used to choose lambda or w."""
import numpy as np
from .evaluation import cohort_balanced_accuracy

LAMBDAS = [0, 0.25, 0.5, 1.0]
W_GRID = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 100]


def fuse(final_scores, global_logp, w):
    """Late fusion: S = final + w * log p_global (both samples x classes)."""
    return np.asarray(final_scores) + w * np.asarray(global_logp)


def choose_lambda(y_val, prior_val, gnn_val, lambdas=LAMBDAS):
    """lambda maximising validation balanced accuracy (rounded to 3 decimals; ties -> smaller lambda)."""
    from sklearn.metrics import balanced_accuracy_score
    sc = {l: balanced_accuracy_score(y_val, (prior_val + l * gnn_val).argmax(1)) for l in lambdas}
    return max(lambdas, key=lambda l: (round(sc[l], 3), -l)), sc


def choose_w_loco(final_scores, global_logp, y, cohort, test_cohort, cohorts, grid=W_GRID):
    """w for one test cohort = argmax over grid of the mean within-cohort balanced accuracy on the OTHER
    cohorts (first maximum in grid order)."""
    oth = [c for c in cohorts if c != test_cohort]
    return max(grid, key=lambda w: cohort_balanced_accuracy(y, fuse(final_scores, global_logp, w), cohort, oth))


def unified_loco(final_scores, global_logp, y, cohort, cohorts, grid=W_GRID):
    """Apply choose_w_loco to every new cohort; returns fused scores and the chosen w per cohort."""
    S = np.zeros_like(np.asarray(final_scores, float)); W = {}
    for t in cohorts:
        w = choose_w_loco(final_scores, global_logp, y, cohort, t, cohorts, grid); W[t] = w
        m = np.asarray(cohort) == t; S[m] = fuse(final_scores[m], global_logp[m], w)
    return S, W


def choose_w_folds(fold_metric, folds, k, grid=W_GRID):
    """Internal CV: w for test fold k = argmax over grid of the mean metric on the other folds; fold_metric(j, w)."""
    return max(grid, key=lambda w: np.mean([fold_metric(j, w) for j in folds if j != k]))
