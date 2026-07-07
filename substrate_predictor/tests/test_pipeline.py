"""Tests for the data pipeline."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import (
    shuffled_negatives, random_background_negatives, build_training_set,
)
from src.utils import WINDOW_LEN, is_valid_window


POSITIVES = ['SGRSENIR', 'GPLGIAGQ', 'AAPVAALS', 'VPLSLYSG',
             'KQLRAIRM', 'GPRFPIGS', 'LPEVAAGL', 'IEALGRFE']


def test_shuffled_negatives_length():
    negs = shuffled_negatives(POSITIVES, n_per_positive=2)
    assert len(negs) <= len(POSITIVES) * 2


def test_shuffled_negatives_composition():
    """Each shuffled sequence should have the same AA composition as its positive."""
    for seq in POSITIVES:
        shuffled = shuffled_negatives([seq], n_per_positive=1)
        if shuffled:
            assert sorted(shuffled[0]) == sorted(seq)


def test_shuffled_negatives_valid():
    negs = shuffled_negatives(POSITIVES, n_per_positive=2)
    for seq in negs:
        assert is_valid_window(seq), f"Invalid window: {seq}"


def test_random_background_length():
    negs = random_background_negatives(100)
    assert len(negs) == 100


def test_random_background_valid():
    negs = random_background_negatives(50)
    for seq in negs:
        assert is_valid_window(seq), f"Invalid window: {seq}"
        assert len(seq) == WINDOW_LEN


def test_training_set_columns():
    df = build_training_set(POSITIVES)
    assert set(df.columns) >= {'sequence', 'label', 'source'}


def test_training_set_labels():
    df = build_training_set(POSITIVES)
    assert set(df['label'].unique()) <= {0, 1}
    assert df['label'].sum() == len(POSITIVES)


def test_training_set_ratio():
    """Negative:positive ratio should be approximately 3:1."""
    df = build_training_set(POSITIVES, target_ratio=3.0)
    n_pos = df['label'].sum()
    n_neg = len(df) - n_pos
    assert n_neg / n_pos >= 2.0, f"Ratio too low: {n_neg}/{n_pos}"


def test_training_set_no_duplicates():
    df = build_training_set(POSITIVES)
    assert df['sequence'].nunique() == len(df), "Duplicate sequences found"
