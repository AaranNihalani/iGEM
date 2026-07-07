"""
Selectivity scoring for candidate peptides across all panel proteases.

For a candidate 8-mer, all six per-protease LR classifiers return predicted
cleavage probabilities p_1 ... p_6.

Selectivity index for target protease i:
    S_i = p_i / max(p_j  for j != i)

Joint ranking score:
    score = p_i * log(S_i + 1)

The log term softens extreme selectivity; +1 avoids log(0).

PSSM fallback: if a trained LR model is absent for a protease (e.g. FAP),
we fall back to a PSSM log-odds score normalised to [0, 1] via a sigmoid.
"""

import logging
import math
import numpy as np
from pathlib import Path

import joblib

from src.utils import PROTEASE_ORDER, SWISSPROT_BG, AA_INDEX, STANDARD_AAS
from src.features import encode_vhse8

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load all models
# ---------------------------------------------------------------------------

def load_models(models_dir: Path) -> dict[str, object]:
    """Load all available LR .pkl models. Returns dict protease -> model."""
    models = {}
    for protease in PROTEASE_ORDER:
        path = models_dir / f"lr_{protease.replace('-','_')}.pkl"
        if path.exists():
            models[protease] = joblib.load(path)
        else:
            log.warning(f"No LR model found for {protease} at {path}. "
                        "Selectivity will use PSSM fallback for this protease.")
    return models


# ---------------------------------------------------------------------------
# PSSM fallback scorer
# ---------------------------------------------------------------------------

def _load_pssm(protease: str, pssm_search_dirs: list[Path]) -> np.ndarray | None:
    """Try to load a saved PSSM .npy from Task 6 processed directory."""
    fname = f"pssm_{protease.replace('-','_')}.npy"
    for d in pssm_search_dirs:
        p = d / fname
        if p.exists():
            return np.load(p)
    return None


def _pssm_prob(seq: str, pssm: np.ndarray) -> float:
    """Convert PSSM log-odds score to a pseudo-probability via sigmoid."""
    score = sum(pssm[pos, AA_INDEX[aa]] for pos, aa in enumerate(seq.upper()))
    return 1.0 / (1.0 + math.exp(-score / 4.0))  # temperature-scaled sigmoid


# ---------------------------------------------------------------------------
# Score one peptide
# ---------------------------------------------------------------------------

def score_peptide(seq: str,
                  target: str,
                  models: dict[str, object],
                  pssm_dirs: list[Path] | None = None) -> dict:
    """
    Score a single 8-mer candidate for a target protease.
    Returns a dict with p_target, selectivity, joint_score, and per-protease probs.
    """
    seq = seq.upper()
    x = encode_vhse8(seq).reshape(1, -1)

    probs = {}
    for protease in PROTEASE_ORDER:
        if protease in models:
            probs[protease] = float(models[protease].predict_proba(x)[0, 1])
        else:
            # PSSM fallback
            pssm = None
            if pssm_dirs:
                pssm = _load_pssm(protease, pssm_dirs)
            probs[protease] = _pssm_prob(seq, pssm) if pssm is not None else 0.5

    p_i = probs[target]
    competitors = [v for k, v in probs.items() if k != target]
    max_competitor = max(competitors) if competitors else 1e-9
    max_competitor = max(max_competitor, 1e-9)

    selectivity = p_i / max_competitor
    joint_score = p_i * math.log(selectivity + 1.0)

    return {
        'sequence':    seq,
        'p_cleave':    round(p_i, 4),
        'selectivity': round(selectivity, 3),
        'joint_score': round(joint_score, 4),
        'probs':       {k: round(v, 4) for k, v in probs.items()},
    }


# ---------------------------------------------------------------------------
# Score a batch of candidates
# ---------------------------------------------------------------------------

def score_candidates(candidates: list[str],
                     target: str,
                     models: dict[str, object],
                     pssm_dirs: list[Path] | None = None) -> list[dict]:
    """
    Score all candidates and return list of result dicts sorted by joint_score descending.
    Uses batched predict_proba for efficiency.
    """
    from src.features import encode_vhse8_batch

    X = encode_vhse8_batch(candidates)

    # Batch probabilities for all proteases
    all_probs: dict[str, np.ndarray] = {}
    for protease in PROTEASE_ORDER:
        if protease in models:
            all_probs[protease] = models[protease].predict_proba(X)[:, 1]
        else:
            pssm = None
            if pssm_dirs:
                pssm = _load_pssm(protease, pssm_dirs)
            if pssm is not None:
                scores = np.array([_pssm_prob(seq, pssm) for seq in candidates])
            else:
                scores = np.full(len(candidates), 0.5)
            all_probs[protease] = scores

    results = []
    for i, seq in enumerate(candidates):
        p_i = float(all_probs[target][i])
        competitors = [float(all_probs[p][i]) for p in PROTEASE_ORDER if p != target]
        max_comp = max(max(competitors), 1e-9)
        sel = p_i / max_comp
        joint = p_i * math.log(sel + 1.0)
        results.append({
            'sequence':    seq,
            'p_cleave':    round(p_i, 4),
            'selectivity': round(sel, 3),
            'joint_score': round(joint, 4),
            'probs':       {p: round(float(all_probs[p][i]), 4) for p in PROTEASE_ORDER},
        })

    results.sort(key=lambda r: r['joint_score'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Generate candidates from PSSM top-k amino acids per position
# ---------------------------------------------------------------------------

def generate_candidates_from_pssm(protease: str,
                                   pssm_dirs: list[Path],
                                   top_k_per_pos: int = 3) -> list[str]:
    """
    Enumerate peptide candidates using top-k AAs at each PSSM position.
    With top_k_per_pos=3 and 8 positions: 3^8 = 6,561 candidates.
    """
    import itertools

    pssm = _load_pssm(protease, pssm_dirs)
    if pssm is None:
        log.warning(f"No PSSM for {protease}; falling back to top-k from background.")
        top_aas_per_pos = [STANDARD_AAS[:top_k_per_pos]] * 8
    else:
        top_aas_per_pos = []
        for pos in range(8):
            ranked = sorted(range(len(STANDARD_AAS)),
                            key=lambda j: pssm[pos, j], reverse=True)
            top_aas_per_pos.append([STANDARD_AAS[j] for j in ranked[:top_k_per_pos]])

    candidates = [''.join(combo) for combo in itertools.product(*top_aas_per_pos)]
    return candidates
