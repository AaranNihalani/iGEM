"""Shared constants and helpers for the Task 6 specificity matrix pipeline."""

import numpy as np

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}

# Swiss-Prot amino acid background frequencies (UniProt 2023, Robinson & Robinson 1991)
SWISSPROT_BG = {
    'A': 0.0777, 'R': 0.0526, 'N': 0.0427, 'D': 0.0530, 'C': 0.0157,
    'Q': 0.0391, 'E': 0.0629, 'G': 0.0691, 'H': 0.0228, 'I': 0.0591,
    'L': 0.0960, 'K': 0.0573, 'M': 0.0237, 'F': 0.0405, 'P': 0.0492,
    'S': 0.0694, 'T': 0.0550, 'W': 0.0130, 'Y': 0.0322, 'V': 0.0667,
}

# MEROPS identifiers from Substrate_search.txt (column 1, family.member format)
# MMP14 = membrane-type matrix metallopeptidase-1 (M10.014): 136 entries
# MMP-8 = matrix metallopeptidase-8 (M10.002): 114 entries
# MMP-9 = matrix metallopeptidase-9 (M10.004): 370 entries
# NE    = elastase-2 / neutrophil elastase (S01.131): 488 entries
# FAP   = fibroblast activation protein alpha (S09.007): 6 entries (low-data)
# Fecal = trypsin-1 (S01.151) + chymotrypsin B (S01.152) + elastase-1 (S01.153): ~23k entries
MEROPS_IDS = {
    'MMP14': ['M10.014'],
    'MMP-8': ['M10.002'],
    'MMP-9': ['M10.004'],
    'NE':    ['S01.131'],
    'FAP':   ['S09.007'],
    'Fecal': ['S01.151', 'S01.152', 'S01.153'],
}

# Final panel motifs.
# MMP14  - SGRSENIR: validated electrochemical sensor motif (Ohuchi et al. 1997; 8-mer core of SGRSENIRTA)
# MMP-8  - KSQFLQGT: Task 7 ML top-1 (LR AUROC 0.773, selectivity 2.02)
# MMP-9  - GAPPVQAP: Task 7 ML top-1 (LR AUROC 0.774, selectivity 2.39; Pro-rich collagen motif)
# NE     - PETISLPP: Task 7 ML top-1 (LR AUROC 0.825, selectivity 17.9; highest confidence)
# Fecal  - DLPRALDQ: Task 7 ML top-1 (LR AUROC 0.938, selectivity 1.98; Arg at P1 tryptic)
# FAP    - GPRFPIGS: literature (Edosada et al. 2006); ML unreliable (n=2 positives)
PANEL_MOTIFS = {
    'MMP14': 'SGRSENIR',
    'MMP-8': 'KSQFLQGT',
    'MMP-9': 'GAPPVQAP',
    'NE':    'PETISLPP',
    'Fecal': 'DLPRALDQ',
    'FAP':   'GPRFPIGS',
}

PROTEASE_ORDER = ['MMP14', 'MMP-8', 'MMP-9', 'NE', 'Fecal', 'FAP']
WINDOW_LEN = 8  # P4..P4'


def is_valid_window(seq: str) -> bool:
    """Return True if seq is exactly WINDOW_LEN standard amino acids."""
    return len(seq) == WINDOW_LEN and all(aa in AA_INDEX for aa in seq.upper())


def validate_motifs() -> None:
    for name, motif in PANEL_MOTIFS.items():
        if not is_valid_window(motif):
            raise ValueError(f"Panel motif for {name} is invalid: '{motif}'")
