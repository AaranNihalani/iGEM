"""
inhibitor_kinetics.py  —  Layer 3 (supplementary)
===================================================
In-device inhibitor binding kinetics: TIMP-1 and A1AT co-elute into the chamber
alongside MMP-9 and HNE and continue to deplete free enzyme during incubation.

Simplified ODE model (well-mixed chamber, no diffusion):
  d[E_free]/dt = -k_on,I * [E_free] * [I_free]
  d[I_free]/dt = -k_on,I * [E_free] * [I_free]
  d[EI]/dt    = +k_on,I * [E_free] * [I_free]

(k_off neglected for A1AT/HNE — near-irreversible; included for TIMP-1/MMP-9)

Outputs (to Layer 9):
  [E_free](t) — free enzyme available for probe cleavage
  Fraction of enzyme inhibited vs time
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Rate constants ─────────────────────────────────────────────────────────
k_on_A1AT   = 2e7    # M-1 s-1  (Beatty et al. 1980 — CONFIDENT)
k_off_A1AT  = 1e-6   # s-1      (near-irreversible)

# FLAG: k_on_TIMP1 — see wound_biology.py
k_on_TIMP1  = 2e5    # M-1 s-1  ← FLAG
k_off_TIMP1 = 2e-5   # s-1

# ── Initial concentrations in chamber (from Layer 1 output) ─────────────────
# Representative mid-range; in forward chain these come from sample_collection.py

# MMP-9 channel
C_MMP9_ch  = 0.33e-9   # mol/L  (~0.33 nM mid-range active MMP-9 after elution)
C_TIMP1_ch = 7e-9 * 0.65 / 2   # ~2.3 nM TIMP-1 in chamber after dilution

# HNE channel
C_HNE_ch   = 100e-9 * 0.75 / 2  # ~37.5 nM HNE in chamber after dilution
C_A1AT_ch  = 1.9e-6 * 0.60 / 2  # ~0.57 µM A1AT in chamber after dilution


# ── ODE systems ──────────────────────────────────────────────────────────────

def mmp9_inhibition_ode(t, y):
    E, I, EI = y
    v_on  = k_on_TIMP1  * E * I
    v_off = k_off_TIMP1 * EI
    return [-v_on + v_off, -v_on + v_off, v_on - v_off]


def hne_inhibition_ode(t, y):
    E, I, EI = y
    v = k_on_A1AT * E * I
    return [-v, -v, v]


def run_inhibition(t_max_s=3600):
    t_eval = np.linspace(0, t_max_s, 2000)

    sol_mmp9 = solve_ivp(mmp9_inhibition_ode, (0, t_max_s),
                         [C_MMP9_ch, C_TIMP1_ch, 0.0],
                         t_eval=t_eval, method="Radau",
                         rtol=1e-9, atol=1e-13)

    sol_hne  = solve_ivp(hne_inhibition_ode, (0, t_max_s),
                         [C_HNE_ch, C_A1AT_ch, 0.0],
                         t_eval=t_eval, method="Radau",
                         rtol=1e-9, atol=1e-13)
    return sol_mmp9, sol_hne


def plot_inhibition(save=True):
    sol_mmp9, sol_hne = run_inhibition()
    t_min = sol_mmp9.t / 60

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: MMP-9 / TIMP-1
    ax = axes[0]
    E_norm_mmp9 = sol_mmp9.y[0] / C_MMP9_ch * 100
    ax.plot(t_min, E_norm_mmp9, "b-", lw=2, label="[MMP-9]$_{\\rm free}$")
    ax.fill_between(t_min, E_norm_mmp9, 100, alpha=0.15, color="blue",
                    label="Inhibited by TIMP-1")
    ax.axvline(30, color="gray", ls=":", lw=1.2, label="T=30 min")
    ax.axvline(60, color="gray", ls="-.", lw=1.2, label="T=60 min")
    ax.set_xlabel("Time in device (min)")
    ax.set_ylabel("% free MMP-9 remaining")
    ax.set_title("A  —  MMP-9 inhibition by TIMP-1 (in-device)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 60)

    # Panel B: HNE / A1AT
    ax2 = axes[1]
    E_norm_hne = sol_hne.y[0] / C_HNE_ch * 100
    ax2.plot(t_min, E_norm_hne, "r-", lw=2, label="[HNE]$_{\\rm free}$")
    ax2.fill_between(t_min, E_norm_hne, 100, alpha=0.15, color="red",
                     label="Inhibited by A1AT")
    ax2.axvline(30, color="gray", ls=":", lw=1.2, label="T=30 min")
    ax2.axvline(60, color="gray", ls="-.", lw=1.2, label="T=60 min")
    ax2.set_xlabel("Time in device (min)")
    ax2.set_ylabel("% free HNE remaining")
    ax2.set_title("B  —  HNE inhibition by A1AT (in-device, near-irreversible)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.set_xlim(0, 60)

    fig.suptitle("Layer 3: In-device inhibitor binding kinetics\n"
                 "(TIMP-1 k_on flagged — see inhibitor_kinetics.py)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B5_inhibitor_kinetics.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    for t_check in [5, 15, 30, 60]:
        idx = np.argmin(np.abs(sol_mmp9.t - t_check*60))
        print(f"t={t_check:2d} min | MMP-9 free: {sol_mmp9.y[0,idx]/C_MMP9_ch*100:.1f}% "
              f"| HNE free: {sol_hne.y[0,idx]/C_HNE_ch*100:.1f}%")


if __name__ == "__main__":
    plot_inhibition()
