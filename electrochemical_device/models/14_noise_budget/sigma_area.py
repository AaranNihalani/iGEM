"""
sigma_area.py  --  Layer 14 -> Layer 12/13 bridge
==================================================
Propagates the per-point net-SWV current noise sigma_dI (noise_budget.py) through
the Layer 12 peak-extraction pipeline to obtain sigma_area, the 1-sigma noise on
the fitted MB (and Fc) peak area.

Four estimates are compared:

  (1) direct integration   sigma_area = sigma_dI * sqrt(N) * dE       (contents.md 14.8)
  (2) Cramer-Rao, free shape   3-parameter Gaussian (A, E0, sigma_E), Layer 12 Eq. IV.3
  (3) Cramer-Rao, fixed shape  amplitude only (E0, sigma_E locked from the R(0) scan)
  (4) Monte Carlo of the FULL Layer 12 pipeline:
        white noise -> Savitzky-Golay (11, 3) -> cubic baseline on anchor regions
        -> Gaussian fit in the +-3 sigma_E window   (and a windowed-sum estimator)

Because every term of the Gaussian-area variance scales as sigma_dI^2 and is
independent of the amplitude A, sigma_area does not depend on Gamma_intact; the
SNR = A_area / sigma_area therefore scales linearly with Gamma (see snr_vs_gamma.py).

Run:  python sigma_area.py
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_coeffs, savgol_filter

import noise_budget as nb

warnings.simplefilter("ignore", OptimizeWarning)
SQ2PI = np.sqrt(2.0 * np.pi)

# Scan window from Layer 11 (Baker 2006): -0.46 -> -0.05 V; MB peak at -0.29 V.
E_START, E_END, E_MB = -0.46, -0.05, -0.29


# --------------------------------------------------------------------------
# Peak model and analytic covariance
# --------------------------------------------------------------------------
def gaussian(E, A, E0, s):
    return A * np.exp(-0.5 * ((E - E0) / s) ** 2)


def window_grid(E0: float, s: float, dE: float, n_sigma: float = 3.0) -> np.ndarray:
    """Potentials of the steps inside E0 +- n_sigma * s (Layer 12 Eq. III.1)."""
    n = int(np.floor(n_sigma * s / dE))
    return E0 + dE * np.arange(-n, n + 1)


def crb_covariance(A: float, E0: float, s: float, E: np.ndarray, sigma_n: float) -> np.ndarray:
    """Cramer-Rao covariance of (A, E0, sigma_E) for white noise sigma_n."""
    g = np.exp(-0.5 * ((E - E0) / s) ** 2)
    J = np.column_stack([g, A * g * (E - E0) / s ** 2, A * g * (E - E0) ** 2 / s ** 3])
    return sigma_n ** 2 * np.linalg.inv(J.T @ J)


def sigma_area_layer12(cov: np.ndarray, A: float, s: float) -> float:
    """Layer 12 Eq. IV.3 (includes the negative A-sigma covariance term)."""
    var = 2 * np.pi * (s ** 2 * cov[0, 0] + A ** 2 * cov[2, 2] + 2 * A * s * cov[0, 2])
    return float(np.sqrt(var))


def sigma_area_fixed_shape(s: float, E: np.ndarray, E0: float, sigma_n: float) -> float:
    """Amplitude-only fit: var(A) = sigma_n^2 / sum(g^2);  area = A s sqrt(2 pi)."""
    g = np.exp(-0.5 * ((E - E0) / s) ** 2)
    return float(sigma_n / np.sqrt(np.sum(g ** 2)) * s * SQ2PI)


def sigma_area_integration(sigma_n: float, N: int, dE: float) -> float:
    return sigma_n * np.sqrt(N) * dE


# --------------------------------------------------------------------------
# Monte Carlo of the Layer 12 pipeline
# --------------------------------------------------------------------------
def sg_noise_factor(window: int = 11, poly: int = 3) -> float:
    return float(np.sqrt(np.sum(savgol_coeffs(window, poly) ** 2)))


def pipeline_once(E, dI, E0, s_pred, n_sigma=3.0, sg=(11, 3)):
    """Layer 12 chain on one noisy trace.  Returns (fit_area, window_sum_area)."""
    y = savgol_filter(dI, sg[0], sg[1], mode="interp")
    anchor = np.abs(E - E0) > n_sigma * s_pred
    coef = np.polyfit(E[anchor] - E0, y[anchor], 3)
    yc = y - np.polyval(coef, E - E0)
    win = ~anchor
    dE = E[1] - E[0]
    a0 = max(float(yc[win].max()), 1e-15)
    try:
        popt, _ = curve_fit(gaussian, E[win], yc[win], p0=[a0, E0, s_pred],
                            bounds=([0, E0 - 0.075, 0.5 * s_pred], [2 * a0 + 1e-12, E0 + 0.075, 5 * s_pred]),
                            maxfev=2000)
        fit_area = popt[0] * popt[2] * SQ2PI
    except RuntimeError:
        fit_area = np.nan
    return fit_area, float(yc[win].sum() * dE)


def monte_carlo(sigma_dI: float, A_area: float, s: float = nb.SIGMA_E, dE: float = 2e-3,
                n_trials: int = 3000, seed: int = 1, E0: float = E_MB) -> dict:
    rng = np.random.default_rng(seed)
    E = np.arange(E_START, E_END + dE / 2, dE)
    A_peak = A_area / (s * SQ2PI)
    clean = gaussian(E, A_peak, E0, s)
    fit, wsum = np.empty(n_trials), np.empty(n_trials)
    for i in range(n_trials):
        fit[i], wsum[i] = pipeline_once(E, clean + rng.normal(0, sigma_dI, E.size), E0, s)
    ok = np.isfinite(fit)
    return {"fit": fit[ok], "wsum": wsum, "n_points": E.size, "A_peak": A_peak,
            "fit_std": float(np.std(fit[ok], ddof=1)), "fit_mean": float(np.mean(fit[ok])),
            "wsum_std": float(np.std(wsum, ddof=1)), "wsum_mean": float(np.mean(wsum))}


# --------------------------------------------------------------------------
# Top-level calculation
# --------------------------------------------------------------------------
def compute(fe=nb.DSTAT, cell=nb.Cell(), A_area=nb.A_MB_L13, n_trials=3000) -> dict:
    b = nb.budget(fe, cell)
    sdI = b["sigma_dI_A"]
    dE = cell.dE_step
    s = nb.SIGMA_E
    Ew = window_grid(nb.E_MB if hasattr(nb, "E_MB") else E_MB, s, dE)
    N = Ew.size
    A_peak = A_area / (s * SQ2PI)

    cov = crb_covariance(A_peak, E_MB, s, Ew, sdI)
    out = {
        "sigma_dI_pA": sdI * nb.PA,
        "N_window": int(N),
        "A_peak_nA": A_peak * 1e9,
        "A_area_nAV": A_area * 1e9,
        "sg_factor": sg_noise_factor(),
        "sigma_area_int_AV": sigma_area_integration(sdI, N, dE),
        "sigma_area_crb_AV": sigma_area_layer12(cov, A_peak, s),
        "sigma_area_fixed_AV": sigma_area_fixed_shape(s, Ew, E_MB, sdI),
        "sigma_A_peak_pA": float(np.sqrt(cov[0, 0])) * nb.PA,
        "corr_A_sigma": float(cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2])),
    }
    mc = monte_carlo(sdI, A_area, s, dE, n_trials)
    out["sigma_area_mc_fit_AV"] = mc["fit_std"]
    out["sigma_area_mc_wsum_AV"] = mc["wsum_std"]
    out["mc_fit_bias_pct"] = 100 * (mc["fit_mean"] / A_area - 1)
    out["mc_n_points"] = mc["n_points"]
    for k in ("int", "crb", "fixed", "mc_fit", "mc_wsum"):
        out[f"snr_{k}"] = A_area / out[f"sigma_area_{k}_AV"]
        out[f"rel_{k}_pct"] = 100 * out[f"sigma_area_{k}_AV"] / A_area

    # Fc reference peak (n_e = 1, sigma_E = 40 mV, window +-3 sigma = 240 mV)
    sF = nb.SIGMA_E_FC
    EwF = window_grid(0.40, sF, dE)
    A_pk_F = nb.A_FC_REF / (sF * SQ2PI)
    covF = crb_covariance(A_pk_F, 0.40, sF, EwF, sdI)
    out["Fc"] = {"N_window": int(EwF.size), "A_area_nAV": nb.A_FC_REF * 1e9,
                 "sigma_area_crb_AV": sigma_area_layer12(covF, A_pk_F, sF)}
    out["Fc"]["rel_crb_pct"] = 100 * out["Fc"]["sigma_area_crb_AV"] / nb.A_FC_REF

    # contents.md 14.8 reproduced for reference
    md_i = np.hypot(0.41, 0.15) * np.sqrt(30) * 1e-12
    out["planning_spec"] = {"i_total_pA": md_i * 1e12,
                            "sigma_area_AV": md_i * np.sqrt(33) * 2e-3,
                            "SNR": 8.1e-10 / (md_i * np.sqrt(33) * 2e-3)}
    out["_mc"] = mc
    out["_Ew"] = Ew
    return out


def scenario_table(cell=nb.Cell()) -> list[dict]:
    """SNR for the three MB peak amplitudes that appear in the pipeline."""
    rows = []
    sdI = nb.budget(nb.DSTAT, cell)["sigma_dI_A"]
    Ew = window_grid(E_MB, nb.SIGMA_E, cell.dE_step)
    for label, A_area in [("Layer 13 (A_MB = 0.3 nA·V)", 0.3e-9),
                          ("contents.md 14.8 (0.81 nA·V)", 8.1e-10),
                          ("Layer 11 Eq. V.7 (peak 0.27 µA)", 0.27e-6 * nb.SIGMA_E * SQ2PI)]:
        A_pk = A_area / (nb.SIGMA_E * SQ2PI)
        sa = sigma_area_layer12(crb_covariance(A_pk, E_MB, nb.SIGMA_E, Ew, sdI), A_pk, nb.SIGMA_E)
        rows.append({"label": label, "A_area_nAV": A_area * 1e9, "sigma_area_fAV": sa * 1e15,
                     "SNR": A_area / sa, "rel_pct": 100 * sa / A_area})
    return rows


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(res: dict, cell=nb.Cell()) -> str:
    mc = res["_mc"]
    A_area = nb.A_MB_L13
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    x = mc["fit"] * 1e12
    ax.hist(x, bins=50, density=True, color="#1565c0", alpha=0.6, label="Monte Carlo (SG → baseline → fit)")
    xs = np.linspace(x.min(), x.max(), 300)
    sig = res["sigma_area_crb_AV"] * 1e12
    mu = mc["fit_mean"] * 1e12
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * SQ2PI), "r-", lw=1.8,
            label=f"Cramér–Rao width (Layer 12 Eq. IV.3): σ = {sig:.3f} pA·V")
    ax.axvline(A_area * 1e12, color="k", ls=":", lw=1, label=f"true area (MC bias {res['mc_fit_bias_pct']:+.2f}%)")
    ax.set_xlim(x.min() - 0.1, A_area * 1e12 + 1.6)
    ax.set_xlabel("Fitted MB peak area (pA·V)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"A — Area scatter at A_MB = 0.3 nA·V\nMC σ = {res['sigma_area_mc_fit_AV']*1e12:.3f} pA·V "
                 f"= {res['rel_mc_fit_pct']:.3f}% of area")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    labels = ["Direct\nintegration", "Free-shape\nfit (CRB)", "Full pipeline\nfit (MC)", "Windowed\nsum (MC)",
              "Fixed-shape\nfit (CRB)"]
    keys = ["int", "crb", "mc_fit", "mc_wsum", "fixed"]
    vals = [res[f"sigma_area_{k}_AV"] * 1e15 for k in keys]
    cols = ["#9e9e9e", "#c62828", "#1565c0", "#ef6c00", "#2e7d32"]
    ax.bar(labels, vals, color=cols, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("σ_area (fA·V)")
    ax.set_title("B — Estimator comparison\n(same 4.6 pA per-point noise, 61-point window)")
    ax.tick_params(axis="x", labelsize=7)

    ax = axes[2]
    dEs = np.array([0.5, 1, 2, 3, 4, 5]) * 1e-3
    sdI = res["sigma_dI_pA"] * 1e-12
    sig_free, sig_fixed = [], []
    for dE in dEs:
        Ew = window_grid(E_MB, nb.SIGMA_E, dE)
        A_pk = A_area / (nb.SIGMA_E * SQ2PI)
        sig_free.append(sigma_area_layer12(crb_covariance(A_pk, E_MB, nb.SIGMA_E, Ew, sdI), A_pk, nb.SIGMA_E))
        sig_fixed.append(sigma_area_fixed_shape(nb.SIGMA_E, Ew, E_MB, sdI))
    ax.plot(dEs * 1e3, np.array(sig_free) * 1e15, "o-", color="#c62828", label="Free-shape fit (CRB)")
    ax.plot(dEs * 1e3, np.array(sig_fixed) * 1e15, "s-", color="#2e7d32", label="Fixed-shape fit (CRB)")
    ax.axvline(2, color="k", ls=":", lw=1)
    ax.set_xlabel("Staircase step ΔE_step (mV)")
    ax.set_ylabel("σ_area (fA·V)")
    ax.set_title("C — σ_area ∝ √ΔE_step at fixed per-point noise\n(smaller step: lower σ, longer scan)")
    ax.legend(fontsize=8)

    fig.suptitle("Layer 14 → 12: propagation of per-point electronic noise to the MB peak area", fontsize=10)
    fig.tight_layout()
    out = nb.figures_dir() / "fig_L14_sigma_area.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)
    return str(out)


def main() -> dict:
    cell = nb.Cell()
    res = compute(cell=cell)
    print("=" * 72)
    print("Layer 14: sigma_area propagation (DStat, R_f = 1 MOhm, f_SWV = 60 Hz)")
    print("=" * 72)
    print(f"per-point net-SWV noise sigma_dI      : {res['sigma_dI_pA']:.2f} pA")
    print(f"SG(11,3) noise factor sqrt(sum c^2)   : {res['sg_factor']:.3f}  (sum c^2 = {res['sg_factor']**2:.4f}; Layer 12 quotes 0.233)")
    print(f"window points N                       : {res['N_window']}  (+-3 sigma_E at {cell.dE_step*1e3:g} mV)")
    print(f"MB peak: area {res['A_area_nAV']:.2f} nA.V  ->  peak {res['A_peak_nA']:.2f} nA")
    print(f"corr(A, sigma_E) in the fit           : {res['corr_A_sigma']:+.2f}")
    print(f"{'estimator':<34}{'sigma_area (fA.V)':>20}{'SNR':>12}{'sigma/A (%)':>14}")
    for k, lab in [("int", "direct integration"), ("crb", "free-shape fit (CRB, Eq. IV.3)"),
                   ("fixed", "fixed-shape fit (CRB)"), ("mc_fit", "full pipeline fit (MC)"),
                   ("mc_wsum", "windowed sum (MC)")]:
        print(f"{lab:<34}{res[f'sigma_area_{k}_AV']*1e15:>20.2f}{res[f'snr_{k}']:>12.0f}{res[f'rel_{k}_pct']:>14.4f}")
    print(f"MC fit bias: {res['mc_fit_bias_pct']:+.3f}%")
    print(f"Fc peak (A = 1 nA.V, N = {res['Fc']['N_window']}): sigma_area = {res['Fc']['sigma_area_crb_AV']*1e15:.1f} fA.V "
          f"({res['Fc']['rel_crb_pct']:.4f}%)")
    ps = res["planning_spec"]
    print(f"contents.md 14.8 as written: i_total = {ps['i_total_pA']:.2f} pA, sigma_area = {ps['sigma_area_AV']*1e15:.1f} fA.V, SNR = {ps['SNR']:.0f}")
    print("\nSNR for the three MB amplitudes in the pipeline:")
    rows = scenario_table(cell)
    for r in rows:
        print(f"  {r['label']:<36} area {r['A_area_nAV']:7.2f} nA.V  sigma {r['sigma_area_fAV']:6.1f} fA.V  SNR {r['SNR']:9.0f}  sigma/A {r['rel_pct']:.4f}%")
    print("Saved", make_figure(res, cell))
    slim = {k: v for k, v in res.items() if not k.startswith("_")}
    slim["scenarios"] = rows
    (nb.results_dir() / "sigma_area.json").write_text(json.dumps(slim, indent=2, default=float))
    return res


if __name__ == "__main__":
    main()
