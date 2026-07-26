"""
VHSE8 encoding for substrate_design — self-contained, no dependency on
substrate_predictor/src to avoid sys.path package-name conflicts.

Source: Mei et al. 2005, J. Chem. Inf. Model. 45(3):856-862.
"""

import numpy as np

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}
WINDOW_LEN = 8

VHSE8_TABLE: dict[str, list[float]] = {
    'A': [ 0.15, -1.11, -1.35, -0.92,  0.02, -0.91,  0.36, -0.48],
    'R': [-1.47,  1.45,  1.24,  1.27,  1.05,  1.15,  2.16,  0.85],
    'N': [-0.99,  0.00, -0.37,  0.69, -0.55,  0.85,  0.73, -0.80],
    'D': [-1.15,  0.67, -0.41, -0.01, -2.68,  1.31,  0.03,  0.56],
    'C': [ 0.18, -1.67, -0.46, -0.21,  0.00,  1.20, -1.61, -0.19],
    'Q': [-0.96,  0.12,  0.18,  0.16,  0.09,  0.42, -0.20, -0.41],
    'E': [-1.18,  0.40,  0.10,  0.36, -2.16, -0.17,  0.91,  0.02],
    'G': [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
    'H': [-0.43, -0.25, -0.37,  0.19,  0.51,  1.28,  0.93,  0.65],
    'I': [ 1.27, -0.14,  0.30, -1.80,  0.30, -1.61, -0.16, -0.13],
    'L': [ 1.36,  0.07,  0.26, -0.80,  0.22, -1.37,  0.08, -0.62],
    'K': [-1.17,  0.70,  0.70,  0.80,  1.64,  0.67,  1.63,  0.13],
    'M': [ 1.01, -0.53,  0.43,  0.00,  0.23, -0.68,  0.11, -1.86],
    'F': [ 1.52,  0.61,  0.96, -0.16,  0.25, -1.36,  0.28, -0.60],
    'P': [ 0.22, -0.17, -0.50,  0.05, -0.01, -1.34, -0.19,  3.56],
    'S': [-0.67, -0.86, -1.07, -0.41,  0.39,  0.21, -0.43, -0.25],
    'T': [-0.34, -0.51, -0.55, -1.06,  0.01, -0.01, -0.44, -0.40],
    'W': [ 1.50,  2.06,  1.79,  0.75,  0.75, -0.13, -1.06, -0.60],
    'Y': [ 0.61,  1.60,  1.17,  0.47,  0.45,  0.45, -0.45, -0.41],
    'V': [ 0.76, -0.92,  0.17, -1.91,  0.22, -1.40, -0.24, -0.03],
}

_MATRIX = np.array([VHSE8_TABLE[aa] for aa in STANDARD_AAS], dtype=np.float32)


def encode_vhse8(seq: str) -> np.ndarray:
    """Encode an 8-mer as a 64-dim VHSE8 feature vector."""
    seq = seq.upper()
    indices = [AA_INDEX[aa] for aa in seq]
    return _MATRIX[indices].flatten()


def encode_vhse8_batch(sequences: list[str]) -> np.ndarray:
    """Encode a list of 8-mers; returns shape (N, 64)."""
    return np.stack([encode_vhse8(s) for s in sequences])
