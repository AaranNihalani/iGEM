"""
probe_conformation.py  —  Layer 8
====================================
PEG spacer end-to-end distance and effective MB–gold tunnelling distance
as a function of probe surface density.

Model:
  PEG in dilute brush regime (Gamma_p < Gamma*):
    L_PEG = N^(3/5) * l_0   (Flory scaling, good solvent)

  PEG in dense brush regime (Gamma_p > Gamma*):
    L_PEG = N * l_0 * (Gamma_p / Gamma*)^(1/3)  (Alexander-de Gennes)

  Effective tunnelling distance:
    d_eff = L_PEG + L_peptide + L_MB

  Gamma* = 1 / (pi * R_F^2) where R_F = N^(3/5) * l_0 is the Flory radius
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── PEG spacer parameters ─────────────────────────────────────────────────────
N_EO     = 6          # number of ethylene oxide units (PEG6 spacer)
l_0      = 0.35e-9    # m  (EO monomer contour length — CONFIDENT, textbook polymer physics)
l_kuhn   = 0.76e-9    # m  (Kuhn length for PEG in water — Mark 1999 polymer handbook)
# Equivalent Kuhn segments: n_k = N_EO * l_0 / l_kuhn
n_k      = N_EO * l_0 / l_kuhn

# Flory radius (good solvent, water):
R_F      = (n_k ** (3/5)) * l_kuhn   # m

# Cross-over surface density from dilute to brush regime:
Gamma_star = 1.0 / (np.pi * R_F**2 * 6.022e23) * 1e-4  # mol/cm²  (= 1/pi*R_F^2 per molecule)
Gamma_star_pmol = Gamma_star * 1e12

# Peptide and MB contribution to d_eff:
L_peptide = 8 * 0.38e-9    # m  (8-residue peptide, ~0.38 nm per residue backbone)
L_MB      = 0.5e-9         # m  (methylene blue molecular length ~ 0.5 nm)

def L_PEG(Gamma_p_pmol):
    """PEG end-to-end distance vs surface density (m)."""
    Gp = Gamma_p_pmol * 1e-12    # mol/cm²
    if np.isscalar(Gp):
        if Gp < Gamma_star:
            return R_F
        else:
            return N_EO * l_0 * (Gp / Gamma_star) ** (1/3)
    else:
        L = np.where(Gp < Gamma_star,
                     R_F,
                     N_EO * l_0 * (Gp / Gamma_star) ** (1/3))
        return L

def d_eff(Gamma_p_pmol):
    """Total tunnelling distance MB → gold (m)."""
    return L_PEG(Gamma_p_pmol) + L_peptide + L_MB


def plot_conformation(save=True):
    Gp = np.linspace(0.1, 20, 400)

    L_peg_nm  = L_PEG(Gp) * 1e9
    d_eff_nm  = d_eff(Gp) * 1e9

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.plot(Gp, L_peg_nm, "b-", lw=2, label="L_PEG (end-to-end)")
    ax.plot(Gp, d_eff_nm, "r-", lw=2, label="d_eff (MB→Au)")
    ax.axvline(Gamma_star_pmol, color="gray", ls="--", lw=1.5,
               label=f"Γ* = {Gamma_star_pmol:.2f} pmol/cm² (dilute→brush)")
    ax.axvspan(2, 5, alpha=0.12, color="green", label="Target Γ_p window")
    ax.set_xlabel("Probe density Γ_p (pmol/cm²)")
    ax.set_ylabel("Distance (nm)")
    ax.set_title("A  —  PEG extension and d_eff vs probe density")
    ax.legend(fontsize=8)

    # Panel B: regimes labelled
    ax2 = axes[1]
    Gp_lo = Gp[Gp <= Gamma_star_pmol]
    Gp_hi = Gp[Gp >  Gamma_star_pmol]
    ax2.plot(Gp_lo, L_PEG(Gp_lo)*1e9, "b-", lw=2.5, label="Dilute: Flory scaling")
    ax2.plot(Gp_hi, L_PEG(Gp_hi)*1e9, "r-", lw=2.5, label="Brush: Alexander-de Gennes")
    ax2.axvline(Gamma_star_pmol, color="gray", ls="--", lw=1.5)
    ax2.set_xlabel("Probe density Γ_p (pmol/cm²)")
    ax2.set_ylabel("L_PEG (nm)")
    ax2.set_title("B  —  Conformation regimes")
    ax2.legend(fontsize=9)

    fig.suptitle(f"Layer 8: Probe conformation — PEG6 spacer (N={N_EO} EO units)\n"
                 f"Γ* = {Gamma_star_pmol:.2f} pmol/cm², "
                 f"R_F = {R_F*1e9:.2f} nm, "
                 f"L_peptide = {L_peptide*1e9:.1f} nm", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L8_probe_conformation.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"PEG6 spacer: N={N_EO}, R_F={R_F*1e9:.2f} nm, Γ*={Gamma_star_pmol:.2f} pmol/cm²")
    for Gp_test in [1, 2, 3, 5]:
        print(f"  Γ_p={Gp_test} pmol/cm²: L_PEG={L_PEG(Gp_test)*1e9:.2f} nm, "
              f"d_eff={d_eff(Gp_test)*1e9:.2f} nm")
    plot_conformation()
