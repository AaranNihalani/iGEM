"""External validation on measured substrates (Kridel et al. 2001, Table II).

Peptides are short, so some P4-P4' positions fall outside the peptide. The models are additive over
positions, hence a missing position contributes its training mean to the LR logit (z = 0) and its
background-expected log-odds to the PSSM score.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import AA_INDEX, WINDOW
from .features import VHSE_MATRIX
from .pssm import BACKGROUND
from .model import ProteaseModel, sigmoid

VALIDATION = Path(__file__).resolve().parents[1] / "data" / "validation"


def gapped_window(sequence: str, scissile_after: int) -> str:
    """P4..P4' around the scissile bond, '-' where the peptide does not extend."""
    out = []
    for k in range(-4, 4):                      # P4..P1 are k=-4..-1 ; P1'..P4' are k=0..3
        j = scissile_after + k
        out.append(sequence[j] if 0 <= j < len(sequence) else "-")
    return "".join(out)


def _z(model: ProteaseModel, gapped: str):
    x = np.zeros(WINDOW * 8)
    for p, a in enumerate(gapped):
        if a != "-":
            x[p * 8:(p + 1) * 8] = VHSE_MATRIX[AA_INDEX[a]]
    z = (x - model.mean) / model.scale
    for p, a in enumerate(gapped):
        if a == "-":
            z[p * 8:(p + 1) * 8] = 0.0
    return z


def partial_logit(model: ProteaseModel, gapped: str) -> float:
    return float(model.intercept + _z(model, gapped) @ model.coef)


def partial_pssm(model: ProteaseModel, gapped: str) -> float:
    s = 0.0
    for p, a in enumerate(gapped):
        s += model.log_odds[p, AA_INDEX[a]] if a != "-" else float((BACKGROUND * model.log_odds[p]).sum())
    return s


def matches(gapped: str, window: str) -> bool:
    return all(g == "-" or g == w for g, w in zip(gapped, window))


def load_kridel() -> pd.DataFrame:
    df = pd.read_csv(VALIDATION / "kridel2001_table_ii.csv")
    df["window"] = [gapped_window(s, i) for s, i in zip(df.sequence, df.scissile_after)]
    return df


def spearman_with_permutation(x, y, n_perm=200_000, seed=0):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm // 20)])
    p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        if len(set(ry[i])) > 1 and len(set(rx[i])) > 1:
            boots.append(np.corrcoef(rankdata(np.asarray(x)[i]), rankdata(np.asarray(y)[i]))[0, 1])
    return float(rho), float(p), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def masked_prob(model: ProteaseModel, seqs, missing) -> np.ndarray:
    """LR score with the given positions treated as unknown (feature at its training mean)."""
    from .features import encode
    z = (encode(seqs) - model.mean) / model.scale
    for p in missing:
        z[:, p * 8:(p + 1) * 8] = 0.0
    return sigmoid(model.intercept + z @ model.coef)


def masked_pssm(model: ProteaseModel, seqs, missing) -> np.ndarray:
    """PSSM score with unknown positions contributing their background-expected log-odds."""
    from .features import index_matrix
    lo = model.log_odds
    s = lo[np.arange(WINDOW), index_matrix(seqs)].copy()
    for p in missing:
        s[:, p] = float((BACKGROUND * lo[p]).sum())
    return s.sum(axis=1)


def fill_gaps(gapped: str) -> tuple[str, tuple]:
    """('-PRSLSGL') -> ('APRSLSGL', (0,)): placeholder residue plus the tuple of missing positions."""
    return gapped.replace("-", "A").replace("X", "A"), tuple(i for i, c in enumerate(gapped) if c in "-X")
