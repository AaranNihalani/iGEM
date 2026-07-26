"""Shared constants for the Task 7 substrate predictor."""

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}

WINDOW_LEN = 8  # P4..P4'

SWISSPROT_BG = {
    'A': 0.0777, 'R': 0.0526, 'N': 0.0427, 'D': 0.0530, 'C': 0.0157,
    'Q': 0.0391, 'E': 0.0629, 'G': 0.0691, 'H': 0.0228, 'I': 0.0591,
    'L': 0.0960, 'K': 0.0573, 'M': 0.0237, 'F': 0.0405, 'P': 0.0492,
    'S': 0.0694, 'T': 0.0550, 'W': 0.0130, 'Y': 0.0322, 'V': 0.0667,
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
}

PROTEASE_ORDER = ['MMP14', 'MMP-8', 'MMP-9', 'MMP-2', 'NE', 'Fecal', 'FAP']

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
