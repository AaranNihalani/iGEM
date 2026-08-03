"""
swv_current_model.py  —  Layer 11
=====================================
Normalised SWV peak current vs incubation time.

i_peak(t) = i_peak_0 * (1 - F_k(t))

where F_k(t) = 1 - exp(-k_obs * t)  and  k_obs = (kcat/Km_surf) * [E]

This is the KEY OBSERVABLE linking wound biomarker concentration to
electrochemical signal. Plotted for the full clinical [E] range.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "09_surface_kinetics"))

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Parameters ────────────────────────────────────────────────────────────────
kcat_Km_eff = 188e3 * 0.3   # M-1s-1  (C15, alpha=0.3)
MW_MMP9     = 92e3

E_clinical_ngml = [17, 50, 100, 150, 200]
E_labels        = [f"{c} ng/mL" for c in E_clinical_ngml]
E_clinical_M    = [c * 1e-6 / MW_MMP9 for c in E_clinical_ngml]

colors = ["#1a237e", "#1565c0", "#0288d1", "#00897b", "#c62828"]

T_arr = np.linspace(0, 3600, 1000)   # 0–60 min in seconds


def Fk(E_M, T_s):
    k_obs = kcat_Km_eff * E_M
    return 1 - np.exp(-k_obs * T_s)


def i_peak_norm(E_M, T_s):
    """Normalised peak current (100% at T=0, drops to (1-F_k)×100%)."""
    return (1 - Fk(E_M, T_s)) * 100


def plot_current_decay(save=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: normalised peak current vs time for each [E]
    ax = axes[0]
    for E_M, lbl, col in zip(E_clinical_M, E_labels, colors):
        i_norm = i_peak_norm(E_M, T_arr)
        ax.plot(T_arr/60, i_norm, color=col, lw=2.2, label=lbl)
    ax.axvline(30, color="gray", ls=":", lw=1.2, label="T=30 min")
    ax.axvline(60, color="gray", ls="-.", lw=1.2, label="T=60 min")
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("Normalised SWV peak current (%)")
    ax.set_title("A  —  Peak current decay across clinical [MMP-9] range")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)

    # Panel B: current at T=30 and T=60 min vs [E] — calibration readout
    ax2 = axes[1]
    E_range = np.linspace(0.05e-9, 3e-9, 300)
    E_ngml  = np.array([e * MW_MMP9 * 1e6 for e in E_range])  # ng/mL
    for T_min, ls, col in [(30, "-", "#1565c0"), (60, "--", "#c62828")]:
        i_at_T = i_peak_norm(E_range, T_min*60)
        ax2.plot(E_ngml, i_at_T, color=col, ls=ls, lw=2,
                 label=f"T={T_min} min")
    ax2.axvspan(17, 200, alpha=0.08, color="green", label="Clinical range")
    ax2.axhline(10, color="gray", ls=":", lw=1, label="10% remaining (90% cleaved)")
    ax2.set_xlabel("[MMP-9] (ng/mL)")
    ax2.set_ylabel("Normalised peak current (%)")
    ax2.set_title("B  —  Calibration readout: i_peak vs [MMP-9]")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 250)

    # Panel C: ΔF_k between 17 and 200 ng/mL vs T — discriminability
    ax3 = axes[2]
    T_plot = np.linspace(60, 3600, 500)
    Fk_lo = Fk(E_clinical_M[0], T_plot)
    Fk_hi = Fk(E_clinical_M[-1], T_plot)
    delta_i = (Fk_hi - Fk_lo) * 100   # in current drop units
    ax3.plot(T_plot/60, delta_i, "k-", lw=2.5)
    ax3.axhline(15, color="orange", ls="--", lw=1.5,
                label="15% ΔF_k target")
    idx_15 = np.searchsorted(delta_i, 15)
    if idx_15 < len(T_plot):
        T_15pct = T_plot[idx_15]
        ax3.axvline(T_15pct/60, color="orange", ls=":", lw=1.5,
                    label=f"Achieves 15% at T={T_15pct/60:.0f} min")
    else:
        ax3.text(0.5, 0.5, "15% ΔF_k not reached\nin 60 min window",
                 transform=ax3.transAxes, ha="center", va="center", color="orange")
    ax3.axvline(30, color="gray", ls=":", lw=1.2)
    ax3.axvline(60, color="gray", ls="-.", lw=1.2)
    ax3.set_xlabel("Incubation time (min)")
    ax3.set_ylabel("Δi_peak between 17 and 200 ng/mL (%)")
    ax3.set_title("C  —  Discriminability: Δi_peak vs T")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 60)

    fig.suptitle("Layer 11: Normalised SWV peak current vs incubation time\n"
                 r"$i_\mathrm{peak}(t) = i_0 \times (1 - F_k(t))$, "
                 r"$F_k = 1 - e^{-k_\mathrm{obs}\,t}$, "
                 f"C15 | α = 0.3", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B10_swv_current.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    print("\nKey readouts:")
    for E_M, lbl in zip(E_clinical_M, E_labels):
        print(f"  {lbl}: i_peak(30min) = {i_peak_norm(E_M,1800):.1f}%  "
              f"i_peak(60min) = {i_peak_norm(E_M,3600):.1f}%")


if __name__ == "__main__":
    plot_current_decay()
