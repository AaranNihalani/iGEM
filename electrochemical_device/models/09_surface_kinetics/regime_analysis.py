"""
regime_analysis.py -- Layer 9 (sensitivity analysis)
======================================================
Explores how the unknown surface rate constant keff_surf governs
detection limit and calibration linearity.

Since both the steady-state and irreversible limits reduce to:
  Fk(t) = 1 - exp(-keff_surf * [E] * t)
at clinical concentrations, keff_surf is the single parameter that
must be measured on the fabricated electrode.

The solution value kcat/Km_sol = 188,000 M-1 s-1 is an upper bound.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

MW_MMP9     = 92e3
kcat_Km_sol = 188e3          # M-1s-1, solution upper bound
E_lo_M      = 17e-6  / MW_MMP9
E_hi_M      = 200e-6 / MW_MMP9

# Range of plausible surface keff values
keff_vals = [2000, 5000, 10000, 20000, 56400, 100000, 188000]
keff_labs = ["2,000", "5,000", "10,000", "20,000",
             "56,400 (prev.)", "100,000", "188,000 (sol. UB)"]
colors    = plt.cm.plasma(np.linspace(0.05, 0.85, len(keff_vals)))


def Fk(E_M, t_s, keff):
    return 1 - np.exp(-keff * E_M * t_s)


def Fk_linear(E_M, t_s, keff):
    return keff * E_M * t_s


def linearity_deviation(t_s, keff, E_check=E_hi_M):
    fe = Fk(E_check, t_s, keff)
    fl = Fk_linear(E_check, t_s, keff)
    return np.abs(fe - fl) / (fe + 1e-12)


def T_detect(keff, threshold=0.05, E_M=E_lo_M):
    """Incubation time (s) for Fk to exceed threshold at E_M."""
    t_arr = np.linspace(60, 14400, 10000)
    fk    = Fk(E_M, t_arr, keff)
    idx   = np.searchsorted(fk, threshold)
    return t_arr[min(idx, len(t_arr)-1)]


def T_linear(keff, max_dev=0.10, E_check=E_hi_M):
    """Last incubation time (s) where calibration is linear to within max_dev."""
    t_arr = np.linspace(60, 14400, 10000)
    devs  = linearity_deviation(t_arr, keff, E_check)
    idx   = np.searchsorted(devs, max_dev)
    return t_arr[min(idx, len(t_arr)-1)]


def plot_regime(save=True):
    t_arr = np.linspace(60, 7200, 3000)   # 1 min to 120 min

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: LOD vs keff (key plot)
    ax = axes[0, 0]
    sigma_Fk = 0.005   # conservative 0.5% total noise estimate
    keff_cont = np.logspace(3, 5.3, 200)
    # LOD = 3*sigma / slope; slope = keff * T * (1e-6/MW) at small Fk
    for T_min, col, ls in [(30, "#1565C0", "-"), (60, "#C62828", "--"),
                            (90, "#00897B", ":")]:
        T_s    = T_min * 60
        slope  = keff_cont * T_s * (1e-6 / MW_MMP9)   # per (ng/mL)
        LOD_ng = 3 * sigma_Fk / slope
        ax.loglog(keff_cont, LOD_ng, color=col, lw=2, ls=ls, label=f"T = {T_min} min")
    ax.axhline(17,  color="gray", ls=":", lw=1.5, label="17 ng/mL (clinical LB)")
    ax.axhline(200, color="gray", ls="-.", lw=1, label="200 ng/mL (clinical UB)")
    ax.axvline(kcat_Km_sol, color="orange", ls=":", lw=1.5,
               label="Solution upper bound")
    ax.set_xlabel("Surface keff (M$^{-1}$s$^{-1}$)")
    ax.set_ylabel("LOD (ng/mL)")
    ax.set_title("A  --  LOD vs surface keff at T = 30, 60, 90 min\n"
                 "(sigma_Fk = 0.5%; clinical range shaded)")
    ax.axhspan(17, 200, alpha=0.07, color="green")
    ax.legend(fontsize=8)
    ax.set_xlim(1e3, 2e5)
    ax.set_ylim(0.1, 1e4)

    # Panel B: linearity window vs keff
    ax = axes[0, 1]
    T_lin_arr = np.array([T_linear(k, 0.10) for k in keff_cont])
    T_det_arr = np.array([T_detect(k, 0.05) for k in keff_cont])
    ax.loglog(keff_cont, T_lin_arr/60, "b-",  lw=2, label="Linear to 10% (T_max)")
    ax.loglog(keff_cont, T_det_arr/60, "r--", lw=2, label="Detectable at 17 ng/mL (T_min)")
    ax.fill_between(keff_cont,
                    np.where(T_det_arr < T_lin_arr, T_det_arr/60, np.nan),
                    np.where(T_det_arr < T_lin_arr, T_lin_arr/60, np.nan),
                    alpha=0.2, color="green", label="Valid linear window")
    ax.axvline(kcat_Km_sol, color="orange", ls=":", lw=1.5, label="Solution UB")
    ax.set_xlabel("Surface keff (M$^{-1}$s$^{-1}$)")
    ax.set_ylabel("Incubation time (min)")
    ax.set_title("B  --  Linear calibration window vs keff\n"
                 "(green = valid window; red above blue = no linear window)")
    ax.legend(fontsize=8)
    ax.set_xlim(1e3, 2e5)

    # Panel C: calibration curves at T=60 min for different keff
    ax = axes[1, 0]
    E_ng = np.linspace(0, 250, 300)
    E_M  = E_ng * 1e-6 / MW_MMP9
    T_s  = 60 * 60
    for keff, lbl, col in zip(keff_vals, keff_labs, colors):
        ax.plot(E_ng, Fk(E_M, T_s, keff)*100, color=col, lw=2, label=lbl)
    ax.axvline(17,  color="gray", ls=":",  lw=1)
    ax.axvline(200, color="gray", ls="-.", lw=1)
    ax.text(18, 0.5, "17 ng/mL", fontsize=8, color="gray")
    ax.text(180, 0.5, "200 ng/mL", fontsize=8, color="gray")
    ax.set_xlabel("[MMP-9] (ng/mL)")
    ax.set_ylabel("$F_k$ (%)")
    ax.set_title("C  --  Calibration curves at T = 60 min\n"
                 "for range of surface keff values (M$^{-1}$s$^{-1}$)")
    ax.legend(fontsize=7, title="keff")
    ax.set_xlim(0, 250)

    # Panel D: Fk at 17 ng/mL vs T for each keff
    ax = axes[1, 1]
    t_120 = np.linspace(0, 7200, 2000)
    for keff, lbl, col in zip(keff_vals, keff_labs, colors):
        ax.plot(t_120/60, Fk(E_lo_M, t_120, keff)*100, color=col, lw=2, label=lbl)
    ax.axhline(5, color="gray", ls=":", lw=1.5, label="5% detection threshold")
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("$F_k$ at 17 ng/mL (%)")
    ax.set_title("D  --  Detectability of 17 ng/mL vs T\n"
                 "for range of surface keff values")
    ax.legend(fontsize=7, title="keff (M$^{-1}$s$^{-1}$)")
    ax.set_xlim(0, 120)

    fig.suptitle(
        "Layer 9: Sensitivity analysis -- predictions span 2 orders of magnitude in keff\n"
        "All quantitative outputs (LOD, operating window) are conditional on measuring "
        "keff_surf on the fabricated electrode",
        fontsize=10
    )
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B9_regime_analysis.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    print("\nOperating window summary:")
    for keff, lbl in zip(keff_vals, keff_labs):
        T_d = T_detect(keff) / 60
        T_l = T_linear(keff) / 60
        window = f"{T_d:.0f}--{T_l:.0f} min" if T_d < T_l else "NO LINEAR WINDOW"
        print(f"  keff = {keff:7.0f}: T_detect = {T_d:.0f} min, "
              f"T_linear = {T_l:.0f} min  --> {window}")


if __name__ == "__main__":
    plot_regime()
