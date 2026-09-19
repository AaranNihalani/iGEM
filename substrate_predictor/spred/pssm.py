"""Position-specific scoring matrix with BLOSUM62-derived pseudocounts.

For each of the 8 positions, observed residue frequencies f are smoothed toward a
substitution-weighted expectation q = f @ W, where W is the row-normalised positive part of
BLOSUM62, mixed with a small amount of background so that no residue has zero probability:

    Q      = (1 - eps) * q + eps * bg
    p      = (n f + beta Q) / (n + beta)
    S(x)   = sum_i log2( p[i, x_i] / bg[x_i] )

This is a heuristic pseudocount scheme (not the block-based Henikoff estimator).
"""
import numpy as np

from .constants import AA_INDEX, WINDOW
from .features import index_matrix
from .tables import AMINO_ACIDS, BLOSUM62, SWISSPROT_BACKGROUND

BACKGROUND = np.array([SWISSPROT_BACKGROUND[a] for a in AMINO_ACIDS])
_pos = np.maximum(np.array(BLOSUM62, dtype=float), 0.0)
BLOSUM_WEIGHTS = _pos / _pos.sum(axis=1, keepdims=True)


def fit_log_odds(positives, beta: float = 5.0, eps: float = 0.05) -> np.ndarray:
    idx = index_matrix(positives)
    counts = np.zeros((WINDOW, 20))
    for pos in range(WINDOW):
        counts[pos] = np.bincount(idx[:, pos], minlength=20)
    n = len(idx)
    f = counts / n
    Q = (1 - eps) * (f @ BLOSUM_WEIGHTS) + eps * BACKGROUND
    p = (counts + beta * Q) / (n + beta)
    return np.log2(p / BACKGROUND)


def score(log_odds: np.ndarray, seqs) -> np.ndarray:
    idx = index_matrix(seqs)
    return log_odds[np.arange(WINDOW), idx].sum(axis=1)


def information_content(log_odds: np.ndarray) -> np.ndarray:
    """Relative entropy (bits) of each position versus background."""
    p = BACKGROUND * 2.0 ** log_odds
    return (p * log_odds).sum(axis=1)
