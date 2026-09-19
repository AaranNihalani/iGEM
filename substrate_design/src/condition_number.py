"""
3×3 specificity matrix and condition number for the Task 8 3-plex panel.

Builds PSSMs for MMP-2, MMP-9, and NE from the substrate_predictor training
CSVs (positives only), then scores the three Stage 1 candidate 8-mers using
the same log2-odds normalisation as Task 6.

Condition number decision rule (from plan.md):
  κ(A) < 10   → well-conditioned; 3-plex as designed
  κ(A) 10–30  → marginal; report with calibration note
  κ(A) > 30   → pivot to 2-plex {gelatinase + NE}; explain identifiability result
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from spred import data as spred_data, pssm as spred_pssm

log = logging.getLogger(__name__)

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}
FLAG_THRESHOLD = 0.20

TRIPLEX_PROTEASES = ['MMP-2', 'MMP-9', 'NE']
SPRED_ID = {'MMP-2': 'MMP2', 'MMP-9': 'MMP9', 'NE': 'NE'}

# Stage 1 candidate sequences, one per target protease
# P4-P4' windows with the scissile bond after residue 4; '-' = position outside the candidate core (see score_candidates.py)
STAGE1_SEQS = {
    'MMP-2': '-PLGLAGQ',   # Candidate 2 (PLG|LAGQS)
    'MMP-9': '-PRSLSGL',   # Candidate 1 (PRS|LSGLK)
    'NE':    'AAPVSGIG',   # Candidate 4 (AAPV|SGIG)
}


def load_triplex_pssms(_unused=None) -> dict[str, np.ndarray]:
    """PSSMs (8 x 20 log2-odds) for MMP-2, MMP-9 and NE from the predictor's positive windows."""
    pos = spred_data.load_positives()
    return {p: spred_pssm.fit_log_odds(pos[SPRED_ID[p]]) for p in TRIPLEX_PROTEASES}


def score_motif(motif: str, pssm: np.ndarray) -> float:
    """PSSM log-odds sum for a P4-P4' window; '-' positions contribute their background-expected log-odds."""
    bg = spred_pssm.BACKGROUND
    return sum(float((bg * pssm[pos]).sum()) if aa == '-' else pssm[pos, AA_INDEX[aa]] for pos, aa in enumerate(motif.upper()))


def build_specificity_matrix(pssms: dict[str, np.ndarray],
                              candidates: dict[str, str]) -> pd.DataFrame:
    """
    Build normalised 3×3 specificity matrix.
    Rows = proteases (PSSM used for scoring).
    Cols = candidates (sequenced scored against that row's PSSM).
    Normalisation: divide each row by |self-score| (diagonal entry).
    Column order follows candidate target order.
    """
    col_order = TRIPLEX_PROTEASES  # MMP-2 col = Cand2, MMP-9 col = Cand1, NE col = Cand4
    raw = {}
    for protease in TRIPLEX_PROTEASES:
        pssm = pssms[protease]
        row = {p: score_motif(candidates[p], pssm) for p in col_order}
        raw[protease] = row

    raw_df = pd.DataFrame(raw, index=col_order).T  # rows=proteases, cols=candidates

    normed_df = raw_df.copy()
    for protease in TRIPLEX_PROTEASES:
        self_score = raw_df.loc[protease, protease]
        denom = abs(self_score) if self_score != 0 else 1.0
        if self_score < 0:
            log.warning(f"{protease}: negative self-score ({self_score:.3f}); "
                        "using |self-score| for normalisation.")
        normed_df.loc[protease] = raw_df.loc[protease] / denom

    return raw_df, normed_df


def condition_number(normed_df: pd.DataFrame) -> float:
    """κ(A) = σ_max / σ_min from SVD of the normalised 3×3 matrix."""
    A = normed_df.values.astype(float)
    _, singular_values, _ = np.linalg.svd(A)
    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]
    if sigma_min < 1e-12:
        return float('inf')
    return float(sigma_max / sigma_min)


def interpret_kappa(kappa: float) -> tuple[str, str]:
    """Return (verdict, action) for the condition number."""
    if kappa < 10:
        return ('well-conditioned', '3-plex as designed — MMP-2/MMP-9/NE separable')
    elif kappa < 30:
        return ('marginal', '3-plex with calibration note; partial separation achievable')
    else:
        return ('poorly-conditioned',
                'Pivot to 2-plex {gelatinase + NE}; '
                'MMP-2/MMP-9 not separable by substrate alone. '
                'Bayesian deconvolution (Task 9) is the correct route to partial A/B resolution.')


def flag_pairs(normed_df: pd.DataFrame) -> list[tuple[str, str, float]]:
    flagged = []
    for row_protease in TRIPLEX_PROTEASES:
        for col_protease in TRIPLEX_PROTEASES:
            if row_protease == col_protease:
                continue
            val = normed_df.loc[row_protease, col_protease]
            if val >= FLAG_THRESHOLD:
                flagged.append((row_protease, col_protease, round(float(val), 3)))
    return sorted(flagged, key=lambda x: -x[2])


def run(output_dir: Path) -> dict:
    """
    Full 3×3 specificity matrix + condition number pipeline.
    Returns dict with matrix DataFrames and kappa.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building 3-plex PSSMs ...")
    pssms = load_triplex_pssms()

    log.info("Building 3×3 specificity matrix ...")
    raw_df, normed_df = build_specificity_matrix(pssms, STAGE1_SEQS)

    kappa = condition_number(normed_df)
    verdict, action = interpret_kappa(kappa)

    log.info(f"\n3×3 Specificity Matrix (normalised):\n{normed_df.round(3).to_string()}")
    log.info(f"\nCondition number κ(A) = {kappa:.2f}")
    log.info(f"Verdict: {verdict}")
    log.info(f"Action:  {action}")

    flagged = flag_pairs(normed_df)
    if flagged:
        log.warning(f"\n{len(flagged)} off-diagonal pair(s) flagged (≥ {FLAG_THRESHOLD}):")
        for row, col, val in flagged:
            log.warning(f"  {row} × {STAGE1_SEQS[col]}: {val:.3f}")
    else:
        log.info("No off-diagonal pairs flagged — all < 0.20.")

    raw_df.to_csv(output_dir / 'specificity_matrix_3plex_raw.csv')
    normed_df.to_csv(output_dir / 'specificity_matrix_3plex_normalised.csv')

    summary = {
        'raw': raw_df,
        'normalised': normed_df,
        'kappa': kappa,
        'verdict': verdict,
        'action': action,
        'flagged_pairs': flagged,
        'stage1_sequences': STAGE1_SEQS,
    }
    log.info(f"Matrix saved to {output_dir}/specificity_matrix_3plex_*.csv")
    return summary
