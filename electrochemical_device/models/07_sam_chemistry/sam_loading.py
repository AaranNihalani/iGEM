"""
sam_loading.py  —  Layer 7
============================
Competitive Langmuir adsorption model for mixed probe-thiol / MCH SAM.

Equilibrium surface composition:
  Gamma_p / Gamma_max = (Ka_p * c_p) / (1 + Ka_p * c_p + Ka_d * c_d)
  Gamma_d / Gamma_max = (Ka_d * c_d) / (1 + Ka_p * c_p + Ka_d * c_d)

Target: Gamma_p = 2–5 pmol/cm²  (from Layer 8 steric accessibility requirement)

Outputs → Layer 8 (steric hindrance), Layer 9 (surface kinetics)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── SAM packing parameters ──────────────────────────────────────────────────
Gamma_max = 1000e-12   # mol/cm²  (close-packed thiol monolayer ~1 nmol/cm²)
# CONFIDENT: alkanethiol close-packed monolayer density
# (Love et al. 2005, Chem Rev — standard reference)

# Adsorption equilibrium constants:
# Ka = k_ad / k_des  (M-1)
# For MCH (6-mercaptohexanol, small diluent): Ka_d ~ 1e6 M-1
# For probe-thiol (large peptide headgroup): Ka_p ~ 1e4–1e5 M-1
# (Ka_p/Ka_d ratio ~ 0.01–0.1 — CONFIDENT that probe adsorbs less strongly,
#  but exact ratio is substrate- and sequence-dependent)
# FLAG: Ka_p and Ka_d for YOUR specific probe sequence + MCH on Au in ethanol.
Ka_p = 5e4    # M-1  ← FLAG: measure by solution-depletion or SPR
Ka_d = 1e6    # M-1  ← FLAG: literature MCH/Au, but ethanol vs aqueous matters


def gamma_equilibrium(c_p_M, c_d_M):
    """Equilibrium probe and diluent surface densities (mol/cm²)."""
    denom     = 1 + Ka_p * c_p_M + Ka_d * c_d_M
    Gamma_p   = Gamma_max * Ka_p * c_p_M / denom
    Gamma_d   = Gamma_max * Ka_d * c_d_M / denom
    return Gamma_p, Gamma_d


def target_c_p(Gamma_p_target, c_d_M):
    """Back-calculate c_p needed to achieve target Gamma_p (mol/cm²)."""
    # Gamma_p_target = Gamma_max * Ka_p * c_p / (1 + Ka_p*c_p + Ka_d*c_d)
    # Solve for c_p:
    A = Gamma_max * Ka_p - Gamma_p_target * Ka_p
    B = Gamma_p_target * (1 + Ka_d * c_d_M)
    # A*c_p - B = 0  →  c_p = B/A
    return B / A


def plot_sam_loading(save=True):
    c_d_fixed  = 100e-6    # 100 µM MCH (typical deposition concentration)
    c_p_range  = np.logspace(-8, -4, 300)   # 10 nM to 100 µM

    Gp_vals = np.array([gamma_equilibrium(cp, c_d_fixed)[0] * 1e12
                        for cp in c_p_range])   # pmol/cm²

    # Ratio sweep at fixed c_p
    c_p_fixed = 1e-6   # 1 µM
    ratio_range = np.logspace(-3, 1, 200)   # c_p/c_d
    c_d_from_ratio = c_p_fixed / ratio_range
    Gp_ratio = np.array([gamma_equilibrium(c_p_fixed, cd)[0] * 1e12
                         for cd in c_d_from_ratio])

    # Optimal deposition conditions
    Gamma_target_lo = 2e-12   # 2 pmol/cm²
    Gamma_target_hi = 5e-12   # 5 pmol/cm²
    c_p_lo = target_c_p(Gamma_target_lo, c_d_fixed)
    c_p_hi = target_c_p(Gamma_target_hi, c_d_fixed)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.semilogx(c_p_range * 1e6, Gp_vals, "b-", lw=2)
    ax.axhspan(2, 5, alpha=0.15, color="green", label="Target Γ_p window (2–5 pmol/cm²)")
    ax.axvline(c_p_lo * 1e6, color="g", ls="--", lw=1.5,
               label=f"c_p for 2 pmol/cm² = {c_p_lo*1e6:.2f} µM")
    ax.axvline(c_p_hi * 1e6, color="darkgreen", ls="--", lw=1.5,
               label=f"c_p for 5 pmol/cm² = {c_p_hi*1e6:.2f} µM")
    ax.set_xlabel("Probe-thiol concentration c_p (µM)")
    ax.set_ylabel("Γ_p at equilibrium (pmol/cm²)")
    ax.set_title(f"A  —  Probe loading vs c_p  (c_d = {c_d_fixed*1e6:.0f} µM MCH)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 15)

    ax2 = axes[1]
    ax2.semilogx(ratio_range, Gp_ratio, "r-", lw=2)
    ax2.axhspan(2, 5, alpha=0.15, color="green", label="Target window")
    ax2.set_xlabel("Deposition ratio c_p / c_d")
    ax2.set_ylabel("Γ_p at equilibrium (pmol/cm²)")
    ax2.set_title("B  —  Probe loading vs c_p/c_d ratio  (c_p = 1 µM)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 15)

    fig.suptitle("Layer 7: SAM competitive Langmuir adsorption\n"
                 "(Ka_p, Ka_d flagged — Ka_p/Ka_d ratio assumed 0.05)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B6_sam_loading.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    Gp, Gd = gamma_equilibrium(1e-6, 100e-6)
    print(f"At c_p=1 µM, c_d=100 µM: Γ_p = {Gp*1e12:.2f} pmol/cm², "
          f"Γ_d = {Gd*1e12:.1f} pmol/cm²")
    c_p_opt = target_c_p(3e-12, 100e-6)
    print(f"To achieve Γ_p = 3 pmol/cm² with c_d=100 µM MCH: "
          f"c_p = {c_p_opt*1e6:.3f} µM  (FLAG: Ka values uncertain)")
    plot_sam_loading()
