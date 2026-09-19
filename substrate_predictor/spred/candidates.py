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
