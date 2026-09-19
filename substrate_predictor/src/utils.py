"""Shared constants for the Task 7 substrate predictor."""

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}

WINDOW_LEN = 8  # P4..P4'

# UniProtKB/Swiss-Prot release 2026_03 amino-acid composition (Expasy relstat, section 6.1),
# normalised to sum to 1. Archived in references/swissprot_composition_2026_03.json.
SWISSPROT_BG = {
    'A': 0.082591, 'R': 0.055261, 'N': 0.040645, 'D': 0.054660, 'C': 0.013815,
    'Q': 0.039343, 'E': 0.067174, 'G': 0.070778, 'H': 0.022725, 'I': 0.059065,
    'L': 0.096506, 'K': 0.057964, 'M': 0.024127, 'F': 0.038643, 'P': 0.047552,
    'S': 0.066673, 'T': 0.053659, 'W': 0.011012, 'Y': 0.029232, 'V': 0.068575,
}

# MEROPS Substrate_search.txt identifiers (column 1, family.member format)
MEROPS_IDS = {
    'MMP14': ['M10.014'],
    'MMP-8': ['M10.002'],
    'MMP-9': ['M10.004'],
    'MMP-2': ['M10.003'],
    'NE':    ['S01.131'],
    'FAP':   ['S09.007'],
    'Fecal': ['S01.151', 'S01.152', 'S01.153'],
    # LasB (M04.005) not in current Substrate_search.txt snapshot;
    # training set is built from literature-curated cleavage sites (see build_lasb.py)
    'LasB':  ['M04.005'],
}

PROTEASE_ORDER = ['MMP14', 'MMP-8', 'MMP-9', 'MMP-2', 'NE', 'Fecal', 'FAP', 'LasB']

# Minimum positives for full-confidence training
MIN_POSITIVES = {'high': 200, 'medium': 50, 'low': 0}

def confidence_tier(n_positives: int) -> str:
    if n_positives >= 200:
        return 'high'
    elif n_positives >= 50:
        return 'medium'
    return 'low'


THREE_TO_ONE = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}

def is_valid_window(seq: str) -> bool:
    return len(seq) == WINDOW_LEN and all(aa in AA_INDEX for aa in seq.upper())
