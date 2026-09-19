import numpy as np
import pytest

from spred import candidates
from spred.model import ProteaseModel, load_all
from spred.panel import PRESETS, Panel


@pytest.fixture(scope="module")
def models():
    return load_all()


def test_json_roundtrip(models, tmp_path):
    m = models["MMP9"]
    m.save(tmp_path)
    m2 = ProteaseModel.load("MMP9", tmp_path)
    s = ["GPLGIAGQ", "IPRTLTAA", "AAPVSGIG"]
    assert np.allclose(m.prob(s), m2.prob(s), atol=1e-6)
    assert np.allclose(m.pssm(s), m2.pssm(s), atol=1e-4)


def test_contributions_sum_to_logit(models):
    m = models["NE"]
    s = "AAPVSGIG"
    assert np.isclose(m.contributions(s).sum() + m.intercept, m.logit([s])[0])


def test_percentiles_in_unit_interval_and_monotone_in_score(models):
    m = models["MMP2"]
    seqs = candidates.random_pool(2000, seed=7)
    u = m.lr_percentile(seqs)
    assert u.min() >= 0 and u.max() <= 1
    order = np.argsort(m.prob(seqs))
    assert (np.diff(u[order]) >= -1e-9).all()


def test_percentile_of_background_is_roughly_uniform(models):
    m = models["MMP9"]
    u = m.lr_percentile(candidates.random_pool(20000, seed=99))
    assert abs(u.mean() - 0.5) < 0.02


def test_margin_definition(models):
    p = Panel(PRESETS["wound"], models)
    sc = p.score(["GPLGIAGQ", "AAPVSGIG"])
    mg = p.margin(sc, "NE")
    others = [f"u_{m}" for m in p.members if m != "NE"]
    assert np.allclose(mg, sc["u_NE"] - sc[others].max(axis=1))


def test_single_mutants_count():
    assert len(candidates.single_mutants("GPLGIAGQ")) == 8 * 19


def test_pssm_top_k_size():
    m = load_all()["MMP9"]
    assert len(candidates.pssm_top_k(m.log_odds, 2)) == 2 ** 8
