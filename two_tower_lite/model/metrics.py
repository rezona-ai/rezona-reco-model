"""AUC and GAUC metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def gauc(y_true, y_score, groups):
    """Group AUC weighted by #impressions per group.
    Only groups containing both a positive and a negative contribute.
    Returns (gauc, coverage) where coverage = fraction of impressions in valid groups.
    """
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    g = groups[order]; yt = y_true[order]; ys = y_score[order]
    # boundaries of contiguous group blocks
    bounds = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    num = 0.0; den = 0.0; total = len(y_true)
    for i in range(len(bounds) - 1):
        a, bnd = bounds[i], bounds[i + 1]
        yt_g = yt[a:bnd]
        if yt_g.min() == yt_g.max():
            continue
        w = bnd - a
        num += w * roc_auc_score(yt_g, ys[a:bnd])
        den += w
    return (num / den if den > 0 else float("nan"), den / total if total else 0.0)
