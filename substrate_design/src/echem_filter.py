"""
Electrochemical device constraints filter for 8-mer peptide candidates.

Rules (all applied to the P4-P4' 8-mer core only):
  1. No Cys  — must be reserved for the Au-thiol anchor in the flanking construct
  2. No Trp  — electrochemically active at ~0.6 V vs Ag/AgCl, overlaps MB redox window
  3. Net charge ∈ [-1, +2] at pH 7.4  — prevents surface repulsion or NiNTA aggregation
  4. No poly-Pro (PP) — rigid β-turn occludes active site
  5. MW 600–1100 Da  — 8-mer core range; outside this: weak signal or steric surface interference
                       (the full construct including Cys anchor, linker, and Lys-MB will be ~1.5–2× heavier)
  6. Met warning (soft) — H₂O₂ blocking steps oxidise Met to Met-sulfoxide
"""

import pandas as pd

# Average residue masses (Da), as would appear in a synthetic peptide
RESIDUE_MASS = {
    'A': 71.08,  'R': 156.19, 'N': 114.10, 'D': 115.09, 'C': 103.14,
    'Q': 128.13, 'E': 129.12, 'G': 57.05,  'H': 137.14, 'I': 113.16,
    'L': 113.16, 'K': 128.17, 'M': 131.20, 'F': 147.18, 'P': 97.12,
    'S': 87.08,  'T': 101.10, 'W': 186.21, 'Y': 163.18, 'V': 99.13,
}
WATER_MASS = 18.02


def net_charge(seq: str) -> int:
    """Net charge at pH 7.4: +1 per Arg/Lys, -1 per Asp/Glu. His treated as neutral."""
    seq = seq.upper()
    return seq.count('R') + seq.count('K') - seq.count('D') - seq.count('E')


def peptide_mw(seq: str) -> float:
    """Average MW of the 8-mer core as a free peptide (sum of residue masses + water)."""
    return sum(RESIDUE_MASS.get(aa, 111.1) for aa in seq.upper()) + WATER_MASS


def check_candidate(seq: str) -> dict:
    """
    Apply all electrochemical filter rules to a single 8-mer.
    Returns a dict with per-rule results and an overall 'passed' flag.
    """
    seq = seq.upper()
    charge = net_charge(seq)
    mw = peptide_mw(seq)

    rules = {
        'no_cys':       'C' not in seq,
        'no_trp':       'W' not in seq,
        'charge_ok':    -1 <= charge <= 2,
        'no_poly_pro':  'PP' not in seq,
        'mw_ok':        600 <= mw <= 1100,
    }
    warnings = {
        'met_warning': 'M' in seq,
    }

    passed = all(rules.values())

    return {
        'sequence': seq,
        'passed': passed,
        'charge': charge,
        'mw_da': round(mw, 1),
        **rules,
        **warnings,
    }


def filter_candidates(sequences: list[str]) -> pd.DataFrame:
    """
    Filter a list of 8-mer sequences against all electrochemical rules.
    Returns a DataFrame with one row per sequence, sorted pass-first.
    """
    rows = [check_candidate(seq) for seq in sequences]
    df = pd.DataFrame(rows)
    df = df.sort_values('passed', ascending=False).reset_index(drop=True)
    return df


def pass_rate_report(df: pd.DataFrame) -> None:
    """Print per-rule pass rates."""
    rules = ['no_cys', 'no_trp', 'charge_ok', 'no_poly_pro', 'mw_ok']
    n = len(df)
    print(f"\nElectrochemical filter: {n} candidates")
    print(f"  Overall pass:  {df['passed'].sum()} / {n} "
          f"({100 * df['passed'].mean():.1f}%)")
    for rule in rules:
        passes = df[rule].sum()
        print(f"  {rule:15s}: {passes} / {n} ({100 * passes / n:.1f}%)")
    warnings = df['met_warning'].sum()
    if warnings:
        print(f"  Met warning (soft): {warnings} sequences contain Met")
