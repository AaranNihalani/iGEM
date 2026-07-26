"""
Score the Task 8 substrate candidates through the 3-plex {MMP-2, MMP-9, NE}
classifiers, compute selectivity indices, and rank against the full PSSM
enumeration pool.

Stage 1 candidates (order for wet-lab):
  Candidate 1 — MMP-9-biased:      PRSLSGLK  (Kridel 2001 phage-display)
  Candidate 2 — MMP-2-leaning:     PLGLAGQS  (ACPP PLGLAG core, Whitney/Tsien)
  Candidate 4 — NE baseline:       AAPVSGIG  (canonical AAPV core, Kasperkiewicz 2014)

Stage 2 candidates (contingency / computational-only):
  Candidate 3 — MMP-2 alt (Nva→Val approx): PLGVALAA  (Murphy 1994; Nva not encodable)
  Candidate 5 — NE high-perf (PK101):        skipped   (4 unnatural residues)
"""

import logging
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).parent.parent.parent  # repo root

log = logging.getLogger(__name__)

# Local VHSE8 encoding — avoids sys.path collision with substrate_predictor/src
from src.features import encode_vhse8, encode_vhse8_batch  # noqa: E402

# 3-plex panel: only these three classifiers are used for Task 8
TRIPLEX = ['MMP-2', 'MMP-9', 'NE']

# Literature-seeded Stage 1 + Stage 2 candidates
CANDIDATES = {
    'Cand1_MMP9':  {'seq': 'PRSLSGLK', 'target': 'MMP-9', 'stage': 1,
                    'source': 'Kridel 2001 (phage display, P2=Arg/P1=Leu MMP-9 bias)'},
    'Cand2_MMP2':  {'seq': 'PLGLAGQS', 'target': 'MMP-2', 'stage': 1,
                    'source': 'Whitney/Tsien ACPP PLGLAG core; kcat/KM ≈ 2.8×10⁴ M⁻¹s⁻¹ (MMP-2)'},
    'Cand4_NE':    {'seq': 'AAPVSGIG', 'target': 'NE',    'stage': 1,
                    'source': 'Kasperkiewicz 2014 canonical AAPV core; NE Val-P1 preference'},
    'Cand3_MMP2b': {'seq': 'PLGVALAA', 'target': 'MMP-2', 'stage': 2,
                    'source': 'Murphy 1994; Nva→Val approx; kcat/KM = 3.97×10⁵ M⁻¹s⁻¹ (MMP-2)'},
}


def _encode_vhse8(seq: str) -> np.ndarray:
    return encode_vhse8(seq)


def _encode_vhse8_batch(seqs: list[str]) -> np.ndarray:
    return encode_vhse8_batch(seqs)


def load_triplex_models(models_dir: Path) -> dict:
    """Load LR models for MMP-2, MMP-9, NE only."""
    models = {}
    for protease in TRIPLEX:
        path = models_dir / f"lr_{protease.replace('-', '_')}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Run substrate_predictor/train_models.py first."
            )
        models[protease] = joblib.load(path)
        log.info(f"  Loaded {protease} LR model from {path.name}")
    return models


def score_one(seq: str, target: str, models: dict) -> dict:
    """Score a single 8-mer, computing selectivity within the 3-plex panel only."""
    x = _encode_vhse8(seq.upper()).reshape(1, -1)
    probs = {p: float(models[p].predict_proba(x)[0, 1]) for p in TRIPLEX}
    p_target = probs[target]
    competitors = [v for k, v in probs.items() if k != target]
    max_comp = max(max(competitors), 1e-9)
    selectivity = p_target / max_comp
    joint_score = p_target * math.log(selectivity + 1.0)
    return {
        'p_target': round(p_target, 4),
        'selectivity': round(selectivity, 3),
        'joint_score': round(joint_score, 4),
        **{f'p_{p.replace("-","_")}': round(v, 4) for p, v in probs.items()},
    }


def enumerate_pssm_pool(protease: str, pssm_dir: Path, top_k: int = 3) -> list[str]:
    """
    Generate up to top_k^8 candidates from the PSSM for a protease.
    Returns unique 8-mers as strings.
    """
    STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
    pssm_path = pssm_dir / f"pssm_{protease.replace('-', '_')}.npy"
    if not pssm_path.exists():
        log.warning(f"PSSM not found for {protease} at {pssm_path}; skipping enumeration.")
        return []
    pssm = np.load(pssm_path)
    top_aas_per_pos = []
    for pos in range(8):
        ranked = sorted(range(len(STANDARD_AAS)), key=lambda j: pssm[pos, j], reverse=True)
        top_aas_per_pos.append([STANDARD_AAS[j] for j in ranked[:top_k]])
    return [''.join(c) for c in itertools.product(*top_aas_per_pos)]


def percentile_rank(seq: str, target: str, pool_seqs: list[str], models: dict) -> float:
    """
    Return the percentile rank (0-100) of seq's joint_score within pool_seqs.
    A score of 95 means seq beats 95% of the enumerated pool.
    """
    if not pool_seqs:
        return float('nan')
    X = _encode_vhse8_batch(pool_seqs)
    pool_probs = {p: models[p].predict_proba(X)[:, 1] for p in TRIPLEX}
    p_targets = pool_probs[target]
    max_comps = np.maximum(
        np.max(np.stack([pool_probs[p] for p in TRIPLEX if p != target], axis=1), axis=1),
        1e-9
    )
    pool_joints = p_targets * np.log(p_targets / max_comps + 1.0)

    seq_result = score_one(seq, target, models)
    seq_joint = seq_result['joint_score']
    rank = float(np.mean(pool_joints < seq_joint) * 100)
    return round(rank, 1)


def run(models_dir: Path, pssm_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Score all candidates, compute percentile ranks, save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_triplex_models(models_dir)

    rows = []
    for name, info in CANDIDATES.items():
        seq, target, stage = info['seq'], info['target'], info['stage']
        log.info(f"Scoring {name} ({seq}) for target {target} ...")

        scores = score_one(seq, target, models)

        pool = enumerate_pssm_pool(target, pssm_dir, top_k=3)
        pctile = percentile_rank(seq, target, pool, models)

        rows.append({
            'name': name,
            'sequence': seq,
            'target': target,
            'stage': stage,
            **scores,
            'percentile_in_pssm_pool': pctile,
            'pool_size': len(pool),
            'source': info['source'],
        })

        log.info(f"  p_target={scores['p_target']:.3f}  "
                 f"selectivity={scores['selectivity']:.2f}  "
                 f"joint={scores['joint_score']:.4f}  "
                 f"percentile={pctile:.1f}%")

    df = pd.DataFrame(rows)
    out_path = output_dir / 'candidate_scores.csv'
    df.to_csv(out_path, index=False)
    log.info(f"Candidate scores saved to {out_path}")
    return df
