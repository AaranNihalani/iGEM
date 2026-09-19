"""Reference tables must equal the archived, provenance-documented files."""
import json
from pathlib import Path

import numpy as np

from spred import tables

REF = Path(__file__).resolve().parents[1] / "references"


def test_vhse_matches_archive():
    ref = json.load(open(REF / "vhse8_table_peptides_0.5.0.json"))["table"]
    for aa in tables.AMINO_ACIDS:
        assert np.allclose(tables.VHSE8[aa], ref[aa], atol=1e-9), aa


def test_glycine_is_not_zero():
    assert np.abs(np.array(tables.VHSE8["G"])).sum() > 5


def test_blosum62_matches_archive_and_is_symmetric():
    ref = json.load(open(REF / "blosum62_biopython.json"))["table"]
    M = np.array(tables.BLOSUM62, dtype=float)
    for i, a in enumerate(tables.AMINO_ACIDS):
        for j, b in enumerate(tables.AMINO_ACIDS):
            assert M[i, j] == ref[a][b]
    assert (M == M.T).all()


def test_background_matches_archive_and_sums_to_one():
    ref = json.load(open(REF / "swissprot_composition_2026_03.json"))["normalised_fraction"]
    assert abs(sum(tables.SWISSPROT_BACKGROUND.values()) - 1) < 1e-5
    for a in tables.AMINO_ACIDS:
        assert abs(tables.SWISSPROT_BACKGROUND[a] - ref[a]) < 1e-6
