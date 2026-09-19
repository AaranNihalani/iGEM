"""
Feature extraction for substrate peptide classification.

Primary: VHSE8 physicochemical encoding (64-dim, always available).
Optional: ESM2 mean-pooled embeddings (320-dim, requires transformers + torch).

VHSE (principal components score Vectors of Hydrophobic, Steric and Electronic
properties): Mei H, Liao ZH, Zhou Y, Li SZ (2005) Biopolymers 80(6):775-786,
doi:10.1002/bip.20296. Eight PCA score vectors derived from independent
families of 18 hydrophobic, 17 steric and 15 electronic properties.
  VHSE1-2: hydrophobic properties
  VHSE3-4: steric properties
  VHSE5-8: electronic properties
Glycine is NOT a zero vector. The numerical table is archived, with provenance,
in references/vhse8_table_peptides_0.5.0.json and is checked against that file
by tests/test_features.py.
"""

import logging
import numpy as np
from pathlib import Path

from src.utils import STANDARD_AAS, AA_INDEX, WINDOW_LEN

log = logging.getLogger(__name__)

# VHSE8 descriptor table: 20 amino acids x 8 descriptors (see module docstring)
VHSE8: dict[str, list[float]] = {
    'A': [ 0.15, -1.11, -1.35, -0.92,  0.02, -0.91,  0.36, -0.48],
    'R': [-1.47,  1.45,  1.24,  1.27,  1.55,  1.47,  1.30,  0.83],
    'N': [-0.99,  0.00, -0.37,  0.69, -0.55,  0.85,  0.73, -0.80],
    'D': [-1.15,  0.67, -0.41, -0.01, -2.68,  1.31,  0.03,  0.56],
    'C': [ 0.18, -1.67, -0.46, -0.21,  0.00,  1.20, -1.61, -0.19],
    'Q': [-0.96,  0.12,  0.18,  0.16,  0.09,  0.42, -0.20, -0.41],
    'E': [-1.18,  0.40,  0.10,  0.36, -2.16, -0.17,  0.91,  0.02],
    'G': [-0.20, -1.53, -2.63,  2.28, -0.53, -1.18,  2.01, -1.34],
    'H': [-0.43, -0.25,  0.37,  0.19,  0.51,  1.28,  0.93,  0.65],
    'I': [ 1.27, -0.14,  0.30, -1.80,  0.30, -1.61, -0.16, -0.13],
    'L': [ 1.36,  0.07,  0.36, -0.80,  0.22, -1.37,  0.08, -0.62],
    'K': [-1.17,  0.70,  0.70,  0.80,  1.64,  0.67,  1.63,  0.13],
    'M': [ 1.01, -0.53,  0.43,  0.00,  0.23,  0.10, -0.86, -0.68],
    'F': [ 1.52,  0.61,  0.96, -0.16,  0.25,  0.28, -1.33, -0.20],
    'P': [ 0.22, -0.17, -0.50,  0.05, -0.01, -1.34, -0.19,  3.56],
    'S': [-0.67, -0.86, -1.07, -0.41, -0.32,  0.27, -0.64,  0.11],
    'T': [-0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79,  0.39],
    'W': [ 1.50,  2.06,  1.79,  0.75,  0.75, -0.13, -1.01, -0.85],
    'Y': [ 0.61,  1.60,  1.17,  0.73,  0.53,  0.25, -0.96, -0.52],
    'V': [ 0.76, -0.92, -0.17, -1.91,  0.22, -1.40, -0.24, -0.03],
}

# Pre-build lookup matrix: shape (20, 8), rows indexed by AA_INDEX
_VHSE8_MATRIX = np.array([VHSE8[aa] for aa in STANDARD_AAS], dtype=np.float32)

VHSE8_DIM = WINDOW_LEN * 8        # 64
ESM2_DIM  = 320
COMBINED_DIM = VHSE8_DIM + ESM2_DIM  # 384


# ---------------------------------------------------------------------------
# VHSE8 encoding
# ---------------------------------------------------------------------------

def encode_vhse8(seq: str) -> np.ndarray:
    """
    Encode an 8-mer as a 64-dimensional VHSE8 feature vector.
    Position i occupies dimensions 8*i .. 8*i+7.
    """
    seq = seq.upper()
    if len(seq) != WINDOW_LEN:
        raise ValueError(f"Expected {WINDOW_LEN}-mer, got length {len(seq)}: '{seq}'")
    indices = []
    for aa in seq:
        if aa not in AA_INDEX:
            raise ValueError(f"Non-standard amino acid '{aa}' in '{seq}'")
        indices.append(AA_INDEX[aa])
    return _VHSE8_MATRIX[indices].flatten()


def encode_vhse8_batch(sequences: list[str]) -> np.ndarray:
    """Encode a list of 8-mers; returns shape (N, 64)."""
    return np.stack([encode_vhse8(s) for s in sequences])


# ---------------------------------------------------------------------------
# ESM2 embeddings (optional)
# ---------------------------------------------------------------------------

def esm2_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch         # noqa: F401
        return True
    except ImportError:
        return False


def encode_esm2_batch(sequences: list[str],
                      cache_path: Path | None = None) -> np.ndarray | None:
    """
    Encode sequences with ESM2 (esm2_t6_8M_UR50D, 320-dim).
    Returns shape (N, 320), or None if ESM2 is unavailable.
    Uses an on-disk cache keyed by sequence to avoid recomputation.
    """
    if not esm2_available():
        log.warning("ESM2 unavailable (transformers/torch not installed). "
                    "Falling back to VHSE8-only features.")
        return None

    import torch
    import pickle
    from transformers import AutoTokenizer, AutoModel

    # Load or initialise cache
    cache: dict[str, np.ndarray] = {}
    if cache_path and cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

    missing = [s for s in sequences if s not in cache]
    if missing:
        log.info(f"  Computing ESM2 embeddings for {len(missing)} sequences ...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        batch_size = 256
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt',
                               padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Mean-pool over residue tokens, excluding [CLS] and [EOS]
            hidden = outputs.last_hidden_state  # (B, L, 320)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            # Exclude first and last tokens
            hidden = hidden[:, 1:-1, :]
            mask = mask[:, 1:-1, :]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            for seq, emb in zip(batch, pooled.cpu().numpy()):
                cache[seq] = emb.astype(np.float32)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)

    return np.stack([cache[s] for s in sequences])


# ---------------------------------------------------------------------------
# Combined feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(sequences: list[str],
                         use_esm2: bool = False,
                         cache_path: Path | None = None) -> np.ndarray:
    """
    Return feature matrix of shape (N, D) where D is 64 (VHSE8 only)
    or 384 (VHSE8 + ESM2) depending on availability and use_esm2 flag.
    """
    X_phys = encode_vhse8_batch(sequences)

    if use_esm2 and esm2_available():
        X_esm = encode_esm2_batch(sequences, cache_path=cache_path)
        if X_esm is not None:
            return np.hstack([X_phys, X_esm])

    return X_phys


def feature_dim(use_esm2: bool = False) -> int:
    if use_esm2 and esm2_available():
        return COMBINED_DIM
    return VHSE8_DIM
