"""Tests for the SWV forward model. Run: python -m pytest test_swv_model.py -q"""
import numpy as np
from scipy.integrate import solve_ivp, quad

import swv_forward_model as m
import report_lib as L

E = m.E_SCAN


def test_closed_form_matches_ode_solver():
    for k0 in (0.5, 20.0, 400.0):
        kf, kb = m._bv_rates(-0.22, -0.25, k0, 0.37, 2)
        K, geq, tau = kf + kb, kf / (kf + kb), 0.05
        sol = solve_ivp(lambda t, y: [kf * (1 - y[0]) - kb * y[0]], (0, tau), [0.3],
                        rtol=1e-12, atol=1e-14, method="LSODA")
        assert abs(sol.y[0, -1] - (geq + (0.3 - geq) * np.exp(-K * tau))) < 1e-9


def test_current_identity():
    kf, kb = m._bv_rates(-0.2, -0.25, 30.0, 0.37, 2)
    g_end, I = m._half_cycle_end(0.4, 1.0, 30.0, 0.37, 2, -0.2, -0.25, 0.05)
    two_term = 2 * m.F_const * (kb * g_end - kf * (1 - g_end))
    assert abs(I - two_term) < 1e-9 * abs(two_term)


def test_stable_at_large_kappa_and_finite():
    y = m.swv_voltammogram(E, 5e4, E0=-0.25, f_swv=100.0)
    assert np.all(np.isfinite(y)) and abs(y).max() < 1e-30


def test_vectorised_simulator_matches_model():
    for k0 in (0.5, 10.0, 127.0):
        a = L.swv_multi(E, [k0], f=10.0)[0]
        b = m.swv_voltammogram(E, k0, E0=-0.25, f_swv=10.0)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12 * abs(b).max())


def test_pd_normalised():
    val, _ = quad(lambda d: float(L.pd_norm(np.array([d]))[0]), m.D_PEG_NM, m.D_MAX_NM)
    assert abs(val - 1) < 1e-3


def test_distance_average_matches_model_function():
    a = L.dist_avg(E, 10.0, n=80)
    b = m.dist_averaged_voltammogram(E, 0.0, n_dist=80)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-10 * b.max())


def test_peak_proportional_to_intact_probe():
    base = L.dist_avg(E, 10.0, n=200)
    for fk in (0.2, 0.5, 0.9):
        assert np.allclose(L.dist_avg(E, 10.0, fk=fk, n=200), (1 - fk) * base, rtol=1e-12)


def test_quasi_reversible_maximum():
    kap = np.logspace(-2, 1.5, 200)
    g = L.g_of_kappa(kap)
    assert 0.2 < kap[g.argmax()] < 1.0


def test_defaults_use_confirmed_area():
    a = m.dist_averaged_voltammogram(E, 0.0, n_dist=40)
    b = m.dist_averaged_voltammogram(E, 0.0, Gamma_0=m.GAMMA_0, A_cm2=m.A_ELEC, n_dist=40)
    assert np.allclose(a, b)
    assert m.A_ELEC == 0.017


def test_cleavage_kinetics():
    assert abs(m.concentration_to_Fk(6, 30) - 0.136) < 0.002
    assert abs(m.concentration_to_Fk(30, 30) - 0.518) < 0.002
    assert m.concentration_to_Fk(0, 30) == 0.0
