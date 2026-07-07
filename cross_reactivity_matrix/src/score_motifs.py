"""
Score panel motifs against all PSSMs, normalise so each diagonal = 1.0,
compute selectivity ratios, and save the specificity matrix CSVs.

Normalisation:
    normalised[i][j] = raw[i][j] / raw[i][i]

If a diagonal raw score is <= 0 (possible for log-odds sums when the motif
is atypical for its own protease), we use abs(diagonal) for normalisation
and log a warning -- the heatmap will still render correctly but the value
should be interpreted cautiously.

Selectivity ratio for off-diagonal cell (i, j):
    SR[i][j] = normalised[i][j]   (since diagonal is already 1.0)

Cells with SR >= 0.20 are flagged.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import STANDARD_AAS, AA_INDEX, PANEL_MOTIFS, PROTEASE_ORDER, WINDOW_LEN
from src.build_pssm import load_pssms

log = logging.getLogger(__name__)

FLAG_THRESHOLD = 0.20
ORTHOGONAL_THRESHOLD = 0.05


def score_motif(motif: str, pssm: np.ndarray) -> float:
    """Compute the PSSM log-odds sum score for an 8-mer motif."""
    if len(motif) != WINDOW_LEN:
        raise ValueError(f"Motif '{motif}' must be {WINDOW_LEN} residues.")
    total = 0.0
    for pos, aa in enumerate(motif.upper()):
        if aa not in AA_INDEX:
            raise ValueError(f"Non-standard amino acid '{aa}' in motif '{motif}'.")
        total += pssm[pos, AA_INDEX[aa]]
    return total


def build_raw_matrix(pssms: dict[str, np.ndarray],
                     motifs: dict[str, str]) -> pd.DataFrame:
    """
    Build the raw (unnormalised) score matrix.
    Rows = proteases, columns = motif names, values = PSSM log-odds scores.
    """
    rows = {}
    for protease in PROTEASE_ORDER:
        if protease not in pssms:
            log.warning(f"No PSSM for {protease}; row will be NaN.")
            rows[protease] = {m: np.nan for m in PROTEASE_ORDER}
            continue
        pssm = pssms[protease]
        row = {}
        for motif_owner in PROTEASE_ORDER:
            motif = motifs.get(motif_owner)
            if motif is None:
                row[motif_owner] = np.nan
            else:
                row[motif_owner] = score_motif(motif, pssm)
        rows[protease] = row

    return pd.DataFrame(rows, index=PROTEASE_ORDER).T  # rows=proteases, cols=motifs


def normalise_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise each row by its diagonal (self-score) so diagonal = 1.0.
    """
    normed = raw.copy()
    for protease in PROTEASE_ORDER:
        if protease not in raw.index:
            continue
        self_score = raw.loc[protease, protease]
        if np.isnan(self_score):
            log.warning(f"{protease}: self-score is NaN; row cannot be normalised.")
            continue
        if self_score == 0:
            log.warning(f"{protease}: self-score is exactly 0; using 1.0 as denominator.")
            self_score = 1.0
        elif self_score < 0:
            log.warning(
                f"{protease}: self-score is negative ({self_score:.3f}); "
                "using |self_score| for normalisation. Interpret with caution."
            )
            self_score = abs(self_score)
        normed.loc[protease] = raw.loc[protease] / self_score

    return normed


def flag_pairs(normed: pd.DataFrame) -> list[tuple[str, str, float]]:
    """
    Return list of (protease, motif_owner, score) for off-diagonal cells >= FLAG_THRESHOLD.
    """
    flagged = []
    for protease in PROTEASE_ORDER:
        for motif_owner in PROTEASE_ORDER:
            if protease == motif_owner:
                continue
            val = normed.loc[protease, motif_owner]
            if not np.isnan(val) and val >= FLAG_THRESHOLD:
                flagged.append((protease, motif_owner, val))
    return sorted(flagged, key=lambda x: -x[2])


def run_scoring(pssm_dir: Path, output_dir: Path,
                motifs: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full scoring pipeline: load PSSMs, score motifs, normalise, save CSVs.

    Returns (raw_df, normalised_df).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if motifs is None:
        motifs = PANEL_MOTIFS

    pssms = load_pssms(pssm_dir)
    if not pssms:
        raise RuntimeError("No PSSMs found. Run build_pssm.py first.")

    raw = build_raw_matrix(pssms, motifs)
    normed = normalise_matrix(raw)

    raw_path = output_dir / 'specificity_matrix_raw.csv'
    normed_path = output_dir / 'specificity_matrix_normalised.csv'
    raw.to_csv(raw_path)
    normed.to_csv(normed_path)

    log.info(f"Raw matrix saved to {raw_path}")
    log.info(f"Normalised matrix saved to {normed_path}")

    flagged = flag_pairs(normed)
    if flagged:
        log.warning(f"{len(flagged)} off-diagonal pair(s) flagged (score >= {FLAG_THRESHOLD}):")
        for protease, motif_owner, val in flagged:
            log.warning(f"    {protease} x {motif_owner}-motif: {val:.3f}")
    else:
        log.info("No off-diagonal pairs exceed the flag threshold (all < 0.20). Panel is orthogonal.")

    orthogonal = [
        (p, m) for p in PROTEASE_ORDER for m in PROTEASE_ORDER
        if p != m and not np.isnan(normed.loc[p, m]) and normed.loc[p, m] < ORTHOGONAL_THRESHOLD
    ]
    log.info(f"{len(orthogonal)} off-diagonal pairs are orthogonal (score < {ORTHOGONAL_THRESHOLD}).")

    return raw, normed


if __name__ == '__main__':
    from pathlib import Path
    base = Path(__file__).parent.parent
    run_scoring(
        pssm_dir=base / 'data' / 'processed',
        output_dir=base / 'data' / 'outputs',
    )
