"""Tests for VHSE8 feature extraction."""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import encode_vhse8, encode_vhse8_batch, VHSE8_DIM
from src.utils import WINDOW_LEN


def test_encode_shape():
    vec = encode_vhse8('SGRSENIR')
    assert vec.shape == (VHSE8_DIM,), f"Expected ({VHSE8_DIM},), got {vec.shape}"


def test_encode_gly_zero():
    """Gly is the VHSE8 reference point: all descriptors should be 0."""
    vec = encode_vhse8('GGGGGGGG')
    np.testing.assert_array_equal(vec, np.zeros(VHSE8_DIM))


def test_encode_known_value():
    """Ala at position 0 should have VHSE1=0.15 at index 0."""
    vec = encode_vhse8('AGGGGAGG')
    assert pytest.approx(vec[0], abs=1e-6) == 0.15   # Ala VHSE1
    assert pytest.approx(vec[1], abs=1e-6) == -1.11  # Ala VHSE2


def test_invalid_aa_raises():
    with pytest.raises(ValueError, match="Non-standard"):
        encode_vhse8('SGRSENXR')


def test_wrong_length_raises():
    with pytest.raises(ValueError, match="Expected"):
        encode_vhse8('SGRS')


def test_batch_shape():
    seqs = ['SGRSENIR', 'GPLGIAGQ', 'AAPVAALS']
    X = encode_vhse8_batch(seqs)
    assert X.shape == (3, VHSE8_DIM)


def test_batch_matches_single():
    seqs = ['SGRSENIR', 'VPLSLYSG']
    X = encode_vhse8_batch(seqs)
    for i, s in enumerate(seqs):
        np.testing.assert_array_almost_equal(X[i], encode_vhse8(s))
