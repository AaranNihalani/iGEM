"""
steric_hindrance.py  —  Layer 8
==================================
Steric correction to surface kcat/Km as a function of probe surface density.

At high Gamma_p, probes crowd each other, blocking enzyme access.
Steric factor: f_steric = 1 - Gamma_p / Gamma_steric_max

Effective surface kcat/Km:
  (kcat/Km)_surf = alpha * (kcat/Km)_sol * f_steric

where alpha is the base surface correction (geometry, orientation effects).

Sensitivity loss curve:
  S(Gamma_p) = f_steric(Gamma_p) / f_steric(Gamma_opt)

Outputs → Layer 9, Layer 13
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Steric packing parameters ─────────────────────────────────────────────────
# Maximum surface density before enzyme is fully sterically blocked:
# Estimate from enzyme size: MMP-9 r ~ 4 nm (from Stokes-Einstein + crystal structure)
# A_enzyme = pi * r^2 ~ 50 nm²  →  max accessible probes ~ 1/A_enzyme per enzyme footprint
# At Gamma_p = 1/(A_enzyme * N_A) per unit area, enzymes can no longer fit between probes.
# Gamma_steric_max ~ 1/(50e-18 m² * 6.02e23) = 3.3e-11 mol/m² = 3.3 pmol/cm²
# Consistent with literature: Peterson et al. 2002 (JACS) measured optimum
# at ~2–3 pmol/cm² for similar probe/enzyme systems.
r_MMP9_nm      = 4.0            # nm  (from crystal structure 1L6J — CONFIDENT)
A_enzyme_nm2   = np.pi * r_MMP9_nm**2
Gamma_steric_max = 1.0 / (A_enzyme_nm2 * 1e-18 * 6.022e23) * 1e-4   # mol/cm²
Gamma_steric_max_pmol = Gamma_steric_max * 1e12   # pmol/cm²

# Base surface correction alpha (geometry + orientation averaging):
# Literature: alpha ranges 0.1–1.0 for peptide substrates on SAMs
# (Squires et al. 2008 Nature Biotech; Perez et al. 2012 JACS)
# Use alpha = 0.3 as conservative central estimate
# FLAG: alpha should be measured empirically by comparing solution vs surface kobs
alpha_base = 0.3    # ← FLAG

# Solution kcat/Km for C15 (MMP-9 channel) from Max's data:
kcat_Km_sol = 188e3   # M-1 s-1

def f_steric(Gamma_p_pmol):
    """Steric accessibility factor [0,1]."""
    return np.maximum(0, 1 - Gamma_p_pmol / Gamma_steric_max_pmol)

def kcat_Km_surf(Gamma_p_pmol):
    """Effective surface kcat/Km (M-1s-1) vs probe density."""
    return alpha_base * kcat_Km_sol * f_steric(Gamma_p_pmol)

def sensitivity_loss(Gamma_p_pmol, Gamma_opt_pmol=2.0):
    """Fractional sensitivity relative to optimal Gamma_p."""
    S_opt = kcat_Km_surf(Gamma_opt_pmol)
    return kcat_Km_surf(Gamma_p_pmol) / S_opt

def optimal_Gamma():
    """
    Gamma_p that maximises F_k dynamic range.
    Trade-off: higher Gamma_p → more MB signal but more steric loss.
    Optimal: Gamma_p ~ Gamma_steric_max / 2 for maximum product Gamma * f_steric.
    d/dGamma [Gamma * f_steric] = d/dGamma [Gamma * (1 - Gamma/Gmax)]
                                 = Gmax/2
    """
    return Gamma_steric_max_pmol / 2


def plot_steric(save=True):
    Gp = np.linspace(0.1, Gamma_steric_max_pmol * 1.5, 400)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: f_steric vs Gamma_p
    ax = axes[0]
    ax.plot(Gp, f_steric(Gp), "b-", lw=2)
    ax.axvline(optimal_Gamma(), color="g", ls="--", lw=1.5,
               label=f"Optimal Γ_p = {optimal_Gamma():.1f} pmol/cm²")
    ax.axvspan(2, 5, alpha=0.15, color="green", label="Target window (2–5 pmol/cm²)")
    ax.axvline(Gamma_steric_max_pmol, color="r", ls=":", lw=1.5,
               label=f"Γ_steric_max = {Gamma_steric_max_pmol:.1f} pmol/cm²")
    ax.set_xlabel("Probe density Γ_p (pmol/cm²)")
    ax.set_ylabel("Steric factor f_steric")
    ax.set_title("A  —  Steric accessibility vs probe density")
    ax.legend(fontsize=8)
    ax.set_xlim(0, Gamma_steric_max_pmol * 1.5)

    # Panel B: effective kcat/Km_surf
    ax2 = axes[1]
    kk_surf = kcat_Km_surf(Gp)
    ax2.plot(Gp, kk_surf / 1e3, "r-", lw=2)   # ×10³ M-1s-1
    ax2.axvspan(2, 5, alpha=0.15, color="green")
    ax2.axvline(optimal_Gamma(), color="g", ls="--", lw=1.5)
    ax2.set_xlabel("Probe density Γ_p (pmol/cm²)")
    ax2.set_ylabel("(k_cat/K_m)_surf (×10³ M⁻¹s⁻¹)")
    ax2.set_title(f"B  —  Effective surface kcat/Km  (α = {alpha_base})")
    ax2.set_xlim(0, Gamma_steric_max_pmol * 1.5)

    # Panel C: sensitivity loss vs Gamma_p (normalised to optimum)
    ax3 = axes[2]
    sens = sensitivity_loss(Gp, optimal_Gamma())
    ax3.plot(Gp, sens * 100, "k-", lw=2)
    ax3.axhline(100, color="g", ls="--", lw=1.5, label="Optimal sensitivity")
    ax3.axhline(50, color="orange", ls=":", lw=1.2, label="50% sensitivity loss")
    ax3.axvspan(2, 5, alpha=0.15, color="green", label="Target window")
    ax3.set_xlabel("Probe density Γ_p (pmol/cm²)")
    ax3.set_ylabel("Relative sensitivity (%)")
    ax3.set_title("C  —  Sensitivity loss vs probe packing")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, Gamma_steric_max_pmol * 1.5)
    ax3.set_ylim(0, 110)

    fig.suptitle("Layer 8: Steric hindrance — enzyme accessibility vs probe density\n"
                 f"Γ_steric_max = {Gamma_steric_max_pmol:.1f} pmol/cm²  "
                 f"(MMP-9 r = {r_MMP9_nm} nm)  |  α = {alpha_base} (FLAG)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B7_steric_hindrance.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Γ_steric_max = {Gamma_steric_max_pmol:.2f} pmol/cm²")
    print(f"Optimal Γ_p  = {optimal_Gamma():.2f} pmol/cm²")
    for Gp_test in [1, 2, 3, 5, 8]:
        print(f"  Γ_p = {Gp_test} pmol/cm²: f_steric = {f_steric(Gp_test):.3f}, "
              f"kcat/Km_surf = {kcat_Km_surf(Gp_test):.0f} M-1s-1, "
              f"rel. sensitivity = {sensitivity_loss(Gp_test)*100:.1f}%")
    plot_steric()
