"""
Build per-protease Position-Specific Scoring Matrices (PSSMs) from extracted
cleavage windows.

For each protease:
  1. Count amino acid occurrences at each of the 8 positions (P4-P4')
  2. Add Laplace pseudocount of 0.5 to avoid zero entries
  3. Convert to frequency matrix
  4. Compute log2 odds vs Swiss-Prot background frequencies

Output: one (8, 20) numpy array per protease saved as .npy and .csv
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import (
    STANDARD_AAS, AA_INDEX, SWISSPROT_BG, MEROPS_IDS,
    PROTEASE_ORDER, WINDOW_LEN,
)

log = logging.getLogger(__name__)
PSEUDOCOUNT = 0.5


def build_pssm(sequences: list[str]) -> np.ndarray:
    """
    Build an (8, 20) log-odds PSSM from a list of 8-mer sequences.

    Returns log2(f / bg) where f is the pseudocount-smoothed frequency and
    bg is the Swiss-Prot background frequency.
    """
    counts = np.zeros((WINDOW_LEN, len(STANDARD_AAS)), dtype=float)

    for seq in sequences:
        for pos, aa in enumerate(seq.upper()):
            if aa in AA_INDEX:
                counts[pos, AA_INDEX[aa]] += 1.0

    # Laplace pseudocount
    counts += PSEUDOCOUNT

    # Row-normalise to frequency
    freq = counts / counts.sum(axis=1, keepdims=True)

    # Log2 odds vs background
    bg = np.array([SWISSPROT_BG[aa] for aa in STANDARD_AAS])
    pssm = np.log2(freq / bg)

    return pssm


def pssm_to_dataframe(pssm: np.ndarray) -> pd.DataFrame:
    """Convert an (8, 20) PSSM array to a labelled DataFrame."""
    positions = [f'P{i}' if i <= 4 else f"P{i-4}'" for i in range(1, 9)]
    # Relabel as P4 P3 P2 P1 P1' P2' P3' P4'
    positions = ["P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'"]
    return pd.DataFrame(pssm, index=positions, columns=STANDARD_AAS)


def build_all_pssms(processed_dir: Path, output_dir: Path) -> dict[str, np.ndarray]:
    """
    Load cleavage window CSVs and build one PSSM per protease.
    Saves .npy and .csv files to output_dir.
    Returns dict protease -> (8, 20) ndarray.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pssms = {}

    for protease in PROTEASE_ORDER:
        csv_path = processed_dir / f"cleavage_windows_{protease.replace('-', '_')}.csv"
        if not csv_path.exists():
            log.error(f"Missing window file: {csv_path}. Run parse_merops.py first.")
            continue

        df = pd.read_csv(csv_path)
        sequences = df['sequence'].dropna().tolist()
        n = len(sequences)

        if n == 0:
            log.warning(f"{protease}: no sequences found; skipping PSSM.")
            continue

        pssm = build_pssm(sequences)
        pssms[protease] = pssm

        npy_path = output_dir / f"pssm_{protease.replace('-', '_')}.npy"
        np.save(npy_path, pssm)

        csv_out = output_dir / f"pssm_{protease.replace('-', '_')}.csv"
        pssm_to_dataframe(pssm).to_csv(csv_out)

        log.info(f"  {protease:12s}: PSSM built from {n} sequences -> {npy_path.name}")

    return pssms


def load_pssms(processed_dir: Path) -> dict[str, np.ndarray]:
    """Load previously saved PSSM .npy files."""
    pssms = {}
    for protease in PROTEASE_ORDER:
        npy_path = processed_dir / f"pssm_{protease.replace('-', '_')}.npy"
        if npy_path.exists():
            pssms[protease] = np.load(npy_path)
        else:
            log.warning(f"PSSM not found for {protease}: {npy_path}")
    return pssms


if __name__ == '__main__':
    base = Path(__file__).parent.parent
    build_all_pssms(
        processed_dir=base / 'data' / 'processed',
        output_dir=base / 'data' / 'processed',
    )
