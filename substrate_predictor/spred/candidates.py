"""Candidate peptide generators."""
import itertools
from pathlib import Path

import numpy as np

from . import pssm as pssm_mod
from .constants import RANDOM_SEED, WINDOW
from .data import background_sequences
from .features import validate
from .tables import AMINO_ACIDS


def single_mutants(seq: str):
    """All 8 x 19 single substitutions as (position_index, new_residue, mutant)."""
    seq = validate(seq)
    return [(i, a, seq[:i] + a + seq[i + 1:]) for i in range(WINDOW) for a in AMINO_ACIDS if a != seq[i]]


def random_pool(n: int, seed: int = RANDOM_SEED):
    return background_sequences(n, np.random.default_rng(seed))


def pssm_top_k(log_odds: np.ndarray, k: int = 3):
    """k^8 combinations of the k highest-scoring residues at each position."""
    tops = [[AMINO_ACIDS[j] for j in np.argsort(log_odds[i])[::-1][:k]] for i in range(WINDOW)]
    return ["".join(c) for c in itertools.product(*tops)]


def windows(sequence: str):
    sequence = sequence.upper()
    return [(i + 1, sequence[i:i + WINDOW]) for i in range(len(sequence) - WINDOW + 1)
            if set(sequence[i:i + WINDOW]) <= set(AMINO_ACIDS)]


def fasta_windows(path):
    out, name, buf = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name:
                out += [(name, p, w) for p, w in windows("".join(buf))]
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name:
        out += [(name, p, w) for p, w in windows("".join(buf))]
    return out


_MASS = {"A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14, "Q": 128.13, "E": 129.12, "G": 57.05, "H": 137.14,
         "I": 113.16, "L": 113.16, "K": 128.17, "M": 131.20, "F": 147.18, "P": 97.12, "S": 87.08, "T": 101.10, "W": 186.21,
         "Y": 163.18, "V": 99.13}


def sensor_ok(seq: str) -> bool:
    """Electrochemical-sensor constraints from the Task 8 design filter (team device rules, not model output):
    no Cys (reserved for the Au-thiol anchor), no Trp (electroactive near the methylene-blue window), net charge
    in [-1, +2], no Pro-Pro, core mass 600-1100 Da."""
    charge = seq.count("R") + seq.count("K") - seq.count("D") - seq.count("E")
    mass = sum(_MASS[a] for a in seq) + 18.02
    return "C" not in seq and "W" not in seq and "PP" not in seq and -1 <= charge <= 2 and 600 <= mass <= 1100
