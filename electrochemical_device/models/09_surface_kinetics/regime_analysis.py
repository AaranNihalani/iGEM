"""
regime_analysis.py  —  Layer 9
==================================
Identify the incubation time T where:
  (a) The F_k vs [E] calibration curve is still linear (F_k < 0.20)
  (b) F_k at the lower clinical bound (17 ng/mL) is detectable (> 5%)

This defines the OPTIMAL OPERATING TIME for live experiments.

Key insight:
  First-order F_k = 1 - exp(-k_obs * T)  where k_obs = (kcat/Km_surf) * [E]
  Linear regime: k_obs * T << 1  →  F_k ≈ k_obs * T  (calibration is linear in [E])
  Linearity holds while F_k < ~0.2 (exp deviates <10% from linear below ~0.2)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Parameters (from surface_kinetics.py) ────────────────────────────────────
kcat_sol       = 703.0       # s-1  (C15)
Km_sol         = 3.7e-3      # M
delta          = 5.5e-9      # m
f_steric       = 0.6
Km_surf_molcm2 = Km_sol * 1e-3 * (delta * 1e2) / f_steric   # mol/cm²
alpha_base     = 0.3

kcat_Km_surf   = alpha_base * (kcat_sol / Km_surf_molcm2)   # effective M-1s-1 (vol units)
# Note: kcat/Km_surf has units of (s-1)/(mol/cm²) = cm²/mol/s
# To use with [E] in mol/L: k_obs = (kcat/Km_surf) [cm²/mol/s] * [E] [mol/L]
# = kcat_sol/Km_surf_molcm2 * 1e-3 [L/cm³] * alpha  [s-1]
kcat_Km_surf_vol = alpha_base * kcat_sol / (Km_surf_molcm2 * 1e3 / 1e-4)
# Simpler: use the calibrated value from mmp9_substrate_selection.py
# kcat/Km_solution (M-1s-1) * alpha → effective surface rate
kcat_Km_eff = 188e3 * alpha_base   # M-1s-1 (solution units; alpha accounts for surface)

# Clinical [E] range
MW_MMP9 = 92e3
E_lo    = 17e-6  / MW_MMP9   # 0.185 nM
E_hi    = 200e-6 / MW_MMP9   # 2.17 nM
E_range = np.linspace(E_lo * 0.5, E_hi * 1.5, 300)  # mol/L

T_values_min = [5, 10, 15, 20, 30, 45, 60]
T_values_s   = [t * 60 for t in T_values_min]


def Fk_first_order(E_M, T_s, kcat_Km=kcat_Km_eff):
    k_obs = kcat_Km * E_M
    return 1 - np.exp(-k_obs * T_s)

def Fk_linear_approx(E_M, T_s, kcat_Km=kcat_Km_eff):
    return kcat_Km * E_M * T_s

def linearity_deviation(E_M, T_s, kcat_Km=kcat_Km_eff):
    """Fractional deviation of exponential from linear approximation."""
    Fk_exp = Fk_first_order(E_M, T_s, kcat_Km)
    Fk_lin = Fk_linear_approx(E_M, T_s, kcat_Km)
    return np.abs(Fk_exp - Fk_lin) / (Fk_exp + 1e-10)


def find_linear_regime_T(max_deviation=0.10, E_check=E_hi):
    """
    Find maximum T where deviation from linearity < max_deviation at E_hi.
    Returns T in seconds.
    """
    T_arr = np.linspace(60, 7200, 5000)
    devs  = linearity_deviation(E_check, T_arr)
    idx   = np.searchsorted(devs, max_deviation)
    if idx >= len(T_arr):
        return T_arr[-1]
    return T_arr[idx]

T_linear_10pct = find_linear_regime_T(0.10, E_hi)
T_linear_20pct = find_linear_regime_T(0.20, E_hi)


def plot_regime(save=True):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(T_values_min)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    # Panel A: F_k vs [E] calibration curves at different T
    ax = axes[0]
    for T_s, T_min, col in zip(T_values_s, T_values_min, colors):
        Fk = Fk_first_order(E_range, T_s) * 100
        ax.plot(E_range * 1e9, Fk, color=col, lw=2, label=f"T={T_min} min")
    # Linear approx at 60 min
    ax.plot(E_range*1e9, Fk_linear_approx(E_range, 3600)*100, "k--",
            lw=1.5, label="T=60 min (linear)")
    ax.axhspan(5, 90, alpha=0.07, color="green", label="Operating window")
    ax.axvline(E_lo*1e9, color="gray", ls=":", lw=1.2, label="17 ng/mL")
    ax.axvline(E_hi*1e9, color="gray", ls="-.", lw=1.2, label="200 ng/mL")
    ax.set_xlabel("[MMP-9] (nM)")
    ax.set_ylabel("F_k (%)")
    ax.set_title("A  —  Calibration curve F_k vs [E] at different T")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, E_hi*1e9*1.5)

    # Panel B: Linearity deviation at E_hi vs T
    ax2 = axes[1]
    T_arr = np.linspace(60, 7200, 1000)
    dev   = linearity_deviation(E_hi, T_arr) * 100
    ax2.plot(T_arr/60, dev, "b-", lw=2)
    ax2.axhline(10, color="orange", ls="--", lw=1.5, label="10% deviation limit")
    ax2.axhline(20, color="red",    ls="--", lw=1.5, label="20% deviation limit")
    ax2.axvline(T_linear_10pct/60, color="orange", ls=":", lw=1.5,
                label=f"T_max(10%) = {T_linear_10pct/60:.0f} min")
    ax2.axvline(T_linear_20pct/60, color="red",    ls=":", lw=1.5,
                label=f"T_max(20%) = {T_linear_20pct/60:.0f} min")
    ax2.set_xlabel("Incubation time T (min)")
    ax2.set_ylabel("Linearity deviation at 200 ng/mL (%)")
    ax2.set_title("B  —  Departure from linear calibration vs T")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 50)

    # Panel C: F_k at lower bound (17 ng/mL) vs T — detectability
    ax3 = axes[2]
    Fk_lo_arr = Fk_first_order(E_lo, T_arr) * 100
    ax3.plot(T_arr/60, Fk_lo_arr, "g-", lw=2)
    ax3.axhline(5, color="red", ls="--", lw=1.5, label="LOD threshold 5%")
    # Time when F_k(E_lo) > 5%:
    idx_det = np.searchsorted(Fk_lo_arr, 5)
    T_detect = T_arr[min(idx_det, len(T_arr)-1)]
    ax3.axvline(T_detect/60, color="green", ls=":", lw=1.5,
                label=f"T_detect = {T_detect/60:.0f} min")
    ax3.set_xlabel("Incubation time T (min)")
    ax3.set_ylabel("F_k at 17 ng/mL (%)")
    ax3.set_title("C  —  Detectability of lower clinical bound vs T")
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 120)

    # Panel D: Operating window — recommended T with annotations
    ax4 = axes[3]
    T_op = np.linspace(0, 120, 500)
    Fk_lo_op = Fk_first_order(E_lo, np.array(T_op)*60) * 100
    Fk_hi_op = Fk_first_order(E_hi, np.array(T_op)*60) * 100
    ax4.fill_between(T_op, Fk_lo_op, Fk_hi_op, alpha=0.25, color="blue",
                     label="F_k range (17→200 ng/mL)")
    ax4.plot(T_op, Fk_lo_op, "b-",  lw=1.5, label="17 ng/mL")
    ax4.plot(T_op, Fk_hi_op, "r-",  lw=1.5, label="200 ng/mL")
    ax4.axhspan(5, 90, alpha=0.08, color="green")
    ax4.axhline(5,  color="gray", ls=":", lw=1)
    ax4.axhline(90, color="gray", ls=":", lw=1)
    # Recommended window: T_detect → T_linear_20pct
    T_rec_lo = T_detect / 60
    T_rec_hi = T_linear_20pct / 60
    if T_rec_lo < T_rec_hi:
        ax4.axvspan(T_rec_lo, T_rec_hi, alpha=0.15, color="gold",
                    label=f"Linear window: {T_rec_lo:.0f}–{T_rec_hi:.0f} min")
    else:
        ax4.text(0.5, 0.5,
                 f"No valid linear window at α={alpha_base}\n"
                 f"Detectable at T>{T_rec_lo:.0f} min, but\n"
                 f"linear regime ends at T<{T_rec_hi:.0f} min\n"
                 f"→ increase α or use 4PL calibration",
                 transform=ax4.transAxes, ha="center", va="center",
                 color="red", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    ax4.set_xlabel("Incubation time T (min)")
    ax4.set_ylabel("F_k (%)")
    ax4.set_title("D  —  Operating window for live experiments")
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, 120)

    fig.suptitle("Layer 9: Regime analysis — optimal incubation time for first-order linear calibration\n"
                 f"C15 | kcat/Km_eff = {kcat_Km_eff:.0f} M⁻¹s⁻¹  "
                 f"(kcat/Km_sol × α = 188,000 × {alpha_base})", fontsize=10)

    if save:
        out = FIGURES / "fig_B9_regime_analysis.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    print(f"\nRecommended incubation time for first-order linear calibration:")
    print(f"  Detectable at 17 ng/mL:    T > {T_detect/60:.0f} min")
    print(f"  Linear to within 10%:      T < {T_linear_10pct/60:.0f} min")
    print(f"  Linear to within 20%:      T < {T_linear_20pct/60:.0f} min")
    if T_detect/60 < T_linear_20pct/60:
        print(f"  → RECOMMENDED WINDOW: {T_detect/60:.0f}–{T_linear_20pct/60:.0f} min (linear range)")
    else:
        print(f"  → NO VALID LINEAR WINDOW at α={alpha_base}: detectable T>{T_detect/60:.0f} min "
              f"but linear only T<{T_linear_20pct/60:.0f} min")
        print(f"     → Use 4PL calibration (nonlinear), or increase α by optimising SAM chemistry")


if __name__ == "__main__":
    plot_regime()
