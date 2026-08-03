"""
sam_stability.py  —  Layer 7
==============================
SAM degradation over the 60-minute assay window.

Three mechanisms:
  1. Oxidative desorption by H2O2 (Au-S → Au + RSOH)
  2. Thiol exchange with wound-fluid Cys/GSH
  3. Electrochemical desorption (scan potential limit check)

Outputs: % SAM lost per mechanism over 30 and 60 min.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Assay parameters ────────────────────────────────────────────────────────
t_assay_s = 3600   # 60 min assay

# ── 1. Oxidative desorption by H2O2 ─────────────────────────────────────────
# Rate: k_ox_thiol * [H2O2]
# k_ox_thiol ~ 1 M-1s-1 for alkanethiol SAMs (Schoenfisch & Pemberton 1998,
# JACS — CONFIDENT that this is the correct order of magnitude)
k_ox_thiol = 1.0   # M-1 s-1  (CONFIDENT order of magnitude)
# FLAG: exact k_ox for MCH-SAM with peptide probe in wound-fluid matrix
# (peptide headgroup may alter susceptibility)

# [H2O2] in wound fluid: 10–100 µM (inflamed wound, Bucala et al. 1993)
H2O2_range = np.array([10, 50, 100]) * 1e-6   # M

def frac_oxidised(H2O2_M, t_s):
    """Fraction of SAM oxidised at time t."""
    k_pseudo = k_ox_thiol * H2O2_M   # s-1
    return 1 - np.exp(-k_pseudo * t_s)

# ── 2. Thiol exchange with free thiols in wound fluid ─────────────────────────
# Cys in wound fluid: ~5–20 µM (Biswas et al.)
# GSH in wound fluid (chamber): ~25–250 µM after dilution
# Exchange rate k_exchange ~ 10-100 M-1s-1 (Pertsin & Grunze 2000 — order of magnitude)
# FLAG: k_exchange for this SAM chemistry
k_exchange = 50.0   # M-1 s-1  ← FLAG

free_thiol_range = np.array([5, 50, 250]) * 1e-6   # M (Cys + GSH combined)

def frac_exchanged(thiol_M, t_s):
    k_pseudo = k_exchange * thiol_M
    return 1 - np.exp(-k_pseudo * t_s)

# ── 3. Electrochemical desorption ─────────────────────────────────────────────
# E_des (reductive) ≈ -0.9 V vs Ag/AgCl for alkanethiol SAMs
# Our scan window: -0.65 V to +0.1 V → safely above E_des → NO electrochemical loss
E_scan_min = -0.65    # V vs Ag/AgCl (from Layer 6)
E_des      = -0.90    # V vs Ag/AgCl (literature for alkanethiol — CONFIDENT)
safe_margin = E_des - E_scan_min  # should be negative (we never reach E_des)


def plot_stability(save=True):
    t = np.linspace(0, t_assay_s, 500)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors_ox  = ["#FFB300", "#FF6D00", "#D50000"]
    colors_ex  = ["#1565C0", "#0288D1", "#4FC3F7"]

    ax = axes[0]
    for H2, col in zip(H2O2_range, colors_ox):
        frac = frac_oxidised(H2, t) * 100
        ax.plot(t/60, frac, color=col, lw=2,
                label=f"[H₂O₂] = {H2*1e6:.0f} µM")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SAM oxidised (%)")
    ax.set_title("A  —  Oxidative desorption by H₂O₂")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 0.3)

    ax2 = axes[1]
    for th, col in zip(free_thiol_range, colors_ex):
        frac = frac_exchanged(th, t) * 100
        ax2.plot(t/60, frac, color=col, lw=2,
                 label=f"[free thiol] = {th*1e6:.0f} µM")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("SAM exchanged (%)")
    ax2.set_title("B  —  Thiol exchange with wound-fluid Cys/GSH")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 60)

    fig.suptitle(f"Layer 7: SAM stability over 60-min assay\n"
                 f"Electrochemical desorption: NOT REACHED  "
                 f"(E_scan_min={E_scan_min} V, E_des={E_des} V)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L7b_sam_stability.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    print(f"Electrochemical desorption: {'SAFE' if E_scan_min > E_des else 'RISK'} "
          f"(margin = {abs(safe_margin)*1e3:.0f} mV)")
    for H2 in H2O2_range:
        print(f"  Oxidative loss at [H2O2]={H2*1e6:.0f} µM, 60 min: "
              f"{frac_oxidised(H2, 3600)*100:.4f}%")


if __name__ == "__main__":
    plot_stability()
