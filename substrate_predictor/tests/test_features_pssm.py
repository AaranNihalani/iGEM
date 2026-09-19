import numpy as np
import pytest

from spred import features, pssm
from spred.tables import VHSE8


def test_encode_layout():
    x = features.encode(["AGGGGGGG"])
    assert x.shape == (1, 64)
    assert np.allclose(x[0, :8], VHSE8["A"]) and np.allclose(x[0, 8:16], VHSE8["G"])


@pytest.mark.parametrize("bad", ["AAAA", "AAAAAAAX", "AAAAAAAAA"])
def test_validation_rejects(bad):
    with pytest.raises(ValueError):
        features.encode([bad])


def test_pssm_scores_training_sequence_above_random():
    pos = ["GPLGIAGQ", "GPLGVAGQ", "GPQGIAGQ", "PPLGLLGQ", "GPLGIFGQ"]
    lo = pssm.fit_log_odds(pos)
    assert np.isfinite(lo).all() and lo.shape == (8, 20)
    assert pssm.score(lo, ["GPLGIAGQ"])[0] > pssm.score(lo, ["WWWWWWWW"])[0]


def test_pssm_score_is_sum_of_positions():
    lo = pssm.fit_log_odds(["GPLGIAGQ", "GPLGVAGQ"])
    s = "GPLGIAGQ"
    from spred.constants import AA_INDEX
    assert np.isclose(pssm.score(lo, [s])[0], sum(lo[i, AA_INDEX[a]] for i, a in enumerate(s)))


def test_information_content_nonnegative():
    lo = pssm.fit_log_odds(["GPLGIAGQ"] * 20 + ["AAAAAAAA"] * 5)
    assert (pssm.information_content(lo) >= -1e-9).all()
