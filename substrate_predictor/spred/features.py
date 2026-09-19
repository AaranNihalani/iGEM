"""VHSE8 encoding of 8-residue P4-P4' windows."""
import numpy as np

from .constants import AA_INDEX, WINDOW, VHSE_NAMES
from .tables import AMINO_ACIDS, VHSE8

VHSE_MATRIX = np.array([VHSE8[a] for a in AMINO_ACIDS], dtype=np.float64)   # (20, 8)
N_FEATURES = WINDOW * VHSE_MATRIX.shape[1]                                    # 64


def validate(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != WINDOW:
        raise ValueError(f"expected {WINDOW} residues, got {len(seq)}: {seq!r}")
    bad = sorted(set(seq) - set(AMINO_ACIDS))
    if bad:
        raise ValueError(f"non-standard residues {bad} in {seq!r}")
    return seq


def index_matrix(seqs) -> np.ndarray:
    """(N, 8) integer residue indices."""
    return np.array([[AA_INDEX[a] for a in validate(s)] for s in seqs], dtype=np.int64)


def encode(seqs) -> np.ndarray:
    """(N, 64) feature matrix; position i occupies columns 8i..8i+7."""
    idx = index_matrix(seqs)
    return VHSE_MATRIX[idx].reshape(len(idx), -1)


def feature_names():
    from .constants import POSITIONS
    return [f"{p}:{v}" for p in POSITIONS for v in VHSE_NAMES]
