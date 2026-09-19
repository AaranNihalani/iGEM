"""Cross-validation that respects near-duplicate structure, and its metrics.

Sequences within Hamming distance 2 (>= 6/8 identical residues) are merged into clusters; folds are
built over clusters (StratifiedGroupKFold) so that near-duplicates never straddle train and test.
Confidence intervals are cluster bootstrap intervals.
"""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import pssm as pssm_mod
from .constants import RANDOM_SEED
from .features import encode, index_matrix
from .model import GRID, _monotone, fit_logistic, select_C, sigmoid


def hamming_clusters(seqs, max_dist: int = 2) -> np.ndarray:
    """Connected components of the graph joining sequences with Hamming distance <= max_dist.

    Pigeonhole: with three blocks (3+3+2 positions) two sequences differing at <= 2 positions agree
    exactly on at least one block, so only sequences sharing a block need to be compared.
    """
    A = index_matrix(seqs)
    n = len(A)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for lo, hi in ((0, 3), (3, 6), (6, 8)):
        keys = {}
        for i, row in enumerate(A[:, lo:hi]):
            keys.setdefault(row.tobytes(), []).append(i)
        for members in keys.values():
            if len(members) < 2:
                continue
            m = np.array(members)
            sub = A[m]
            for a in range(len(m) - 1):
                d = (sub[a + 1:] != sub[a]).sum(axis=1)
                for b in np.nonzero(d <= max_dist)[0]:
                    ra, rb = find(m[a]), find(m[a + 1 + b])
                    if ra != rb:
                        parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def cross_validate(frame, positives_col="label", n_splits=5, seed=RANDOM_SEED, max_dist=2, tune=True,
                   n_null=5000):
    """Out-of-fold LR probability, PSSM score and LR+PSSM consensus percentile, with cluster ids.

    Within every outer fold the regularisation strength is chosen by an inner cluster-aware CV on the
    training part only (nested selection), and the background null used for percentiles is built from
    that fold's own models.
    """
    from .data import background_sequences
    seqs = frame["sequence"].tolist()
    y = frame[positives_col].values.astype(int)
    X = encode(seqs)
    groups = hamming_clusters(seqs, max_dist)
    oof_p, oof_s, oof_c = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    fold = np.zeros(len(y), dtype=int)
    chosen = []
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    null_seqs = background_sequences(n_null, np.random.default_rng(seed + 7))
    Xn = encode(null_seqs)
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        C = select_C(X[tr], y[tr], groups[tr], seed) if tune else 1.0
        chosen.append(C)
        mean, scale, coef, b = fit_logistic(X[tr], y[tr], C)
        oof_p[te] = sigmoid(b + ((X[te] - mean) / scale) @ coef)
        lo = pssm_mod.fit_log_odds([seqs[i] for i in tr if y[i] == 1])
        oof_s[te] = pssm_mod.score(lo, [seqs[i] for i in te])
        nq_lr = _monotone(np.quantile(sigmoid(b + ((Xn - mean) / scale) @ coef), GRID))
        nq_ps = _monotone(np.quantile(pssm_mod.score(lo, null_seqs), GRID))
        a, c = np.interp(oof_p[te], nq_lr, GRID), np.interp(oof_s[te], nq_ps, GRID)
        oof_c[te] = 2 * a * c / (a + c + 1e-12)
        fold[te] = k
    return {"y": y, "p": oof_p, "s": oof_s, "u": oof_c, "groups": groups, "fold": fold, "C": chosen}


def metrics(y, p) -> dict:
    return {"auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "mcc": float(matthews_corrcoef(y, p >= 0.5))}


def cluster_bootstrap_auroc(y, score, groups, n_boot=500, seed=RANDOM_SEED):
    """95% interval for AUROC, resampling clusters with replacement."""
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    uniq, start = np.unique(g_sorted, return_index=True)
    bounds = np.append(start, len(g_sorted))
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        if 0 < y[idx].sum() < len(idx):
            out.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def calibration_table(y, p, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    ece = sum(n * abs(a - o) for a, o, n in rows) / len(y)
    return rows, float(ece)


def competitor_auroc(frame, oof_score, target) -> dict:
    """AUROC of target positives against each competitor's positives (held out via cross-negatives)."""
    y = frame["label"].values
    src = frame["source"].values
    pos = y == 1
    out = {}
    for c in sorted(set(src[~pos]) - {"background", "shuffled"}):
        m = (src == c) & ~pos
        if m.sum() >= 10:
            yy = np.r_[np.ones(pos.sum()), np.zeros(m.sum())]
            out[c] = float(roc_auc_score(yy, np.r_[oof_score[pos], oof_score[m]]))
    return out
