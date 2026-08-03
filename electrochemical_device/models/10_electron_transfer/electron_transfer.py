"""
electron_transfer.py  —  Layer 10
=====================================
Marcus outer-sphere electron transfer rate and Butler-Volmer current.

Marcus (tunnelling through-space / through-bond):
  k_ET = k_ET_0 * exp(-beta * (d - d_0))

where:
  k_ET_0 = pre-exponential at contact (d = d_0 ~ 0.5 nm)
  beta    = tunnelling decay constant (~1.0 Å-1 for aliphatic chain; CONFIDENT)
  d       = effective MB-gold distance (from Layer 8)

Peak current in SWV (normalised model):
  i_peak(t) = i_peak_0 * (1 - F_k(t))   (proportional to remaining MB)

Butler-Volmer for MB redox at the electrode (qualitative):
  i = i_0 * [exp(alpha_BV * F * eta / RT) - exp(-(1-alpha_BV) * F * eta / RT)]
  where eta = E - E°_MB  (overpotential)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Constants ────────────────────────────────────────────────────────────────
F_  = 96485.0
R_  = 8.314
T_  = 298.15
RT_F = R_ * T_ / F_   # 25.7 mV

# ── Marcus parameters ────────────────────────────────────────────────────────
# beta for aliphatic / PEG chain: ~1.0 per Å = 1e10 m-1
# (Smalley et al. 2003, JACS — CONFIDENT for alkyl chain; PEG may differ slightly)
beta      = 1.0e10    # m-1  (CONFIDENT)
d_0       = 0.5e-9   # m  (contact distance, ~0.5 nm)

# k_ET_0 for MB on SAM: literature ~1e3 – 1e5 s-1 depending on chain length
# FLAG: k_ET_0 for our specific probe architecture (PEG6 + 8-mer peptide)
# Placeholder: 1e4 s-1 at d_0 (close-contact limit)
k_ET_0    = 1e4      # s-1  ← FLAG

# MB formal potential vs Ag/AgCl (pH 7.4, phosphate buffer):
# E°_MB ≈ -0.25 V vs Ag/AgCl (well established, Bernabi et al. 2019 — CONFIDENT)
E0_MB     = -0.250   # V vs Ag/AgCl  (CONFIDENT)

# Butler-Volmer transfer coefficient for MB:
# alpha_BV ≈ 0.5 (symmetric, typical for outer-sphere ET — CONFIDENT)
alpha_BV  = 0.5      # (CONFIDENT)


def k_ET(d_m):
    """Electron transfer rate constant vs distance (s-1)."""
    return k_ET_0 * np.exp(-beta * (d_m - d_0))


def i_bv(eta_V, k_ET_val, A_cm2=0.0177, Gamma_MB_mol_cm2=3e-12, n=2):
    """
    Butler-Volmer peak current (qualitative) for surface-confined MB.
    Surface-confined: i = n*F*A*Gamma_MB * k_ET * [exp(alpha*F*eta/RT) - exp(-(1-alpha)*F*eta/RT)]
    """
    exp_f = np.exp( alpha_BV     * eta_V / RT_F)
    exp_b = np.exp(-(1-alpha_BV) * eta_V / RT_F)
    return n * F_ * A_cm2 * Gamma_MB_mol_cm2 * k_ET_val * (exp_f - exp_b)


def plot_ET(save=True):
    # d_eff range from Layer 8: probe conformation outputs ~4–8 nm
    d_range_nm = np.linspace(0.5, 15, 300)
    kET_vals   = k_ET(d_range_nm * 1e-9)

    # d_eff at target Gamma_p = 3 pmol/cm² from probe_conformation.py
    d_eff_nm_target = 4.9 + 3.04 + 0.5   # L_PEG + L_peptide + L_MB
    kET_target = k_ET(d_eff_nm_target * 1e-9)

    # SWV eta range: ±0.1 V around E°_MB
    eta_range = np.linspace(-0.15, 0.15, 400)
    Gamma_MB_remaining = [1.0, 0.75, 0.50, 0.25]   # fractions (F_k = 0, 0.25, 0.5, 0.75)
    colors = ["#1a237e", "#1565c0", "#0288d1", "#80d8ff"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: k_ET vs d
    ax = axes[0]
    ax.semilogy(d_range_nm, kET_vals, "b-", lw=2)
    ax.axvline(d_eff_nm_target, color="r", ls="--", lw=1.5,
               label=f"d_eff = {d_eff_nm_target:.1f} nm\nk_ET = {kET_target:.1f} s⁻¹")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="1 s⁻¹ threshold")
    ax.set_xlabel("MB–Au distance d (nm)")
    ax.set_ylabel("k_ET (s⁻¹)")
    ax.set_title("A  —  Marcus k_ET vs distance")
    ax.legend(fontsize=8)

    # Panel B: Butler-Volmer i-E at different F_k
    ax2 = axes[1]
    for frac, col in zip(Gamma_MB_remaining, colors):
        Gamma_MB = frac * 3e-12   # mol/cm²
        i_nA = i_bv(eta_range, kET_target, Gamma_MB_mol_cm2=Gamma_MB) * 1e9
        ax2.plot(eta_range * 1e3, i_nA, color=col, lw=2,
                 label=f"F_k = {(1-frac)*100:.0f}%  (Γ = {frac*100:.0f}% intact)")
    ax2.axhline(0, color="gray", ls="-", lw=0.8)
    ax2.axvline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("Overpotential η (mV)")
    ax2.set_ylabel("Current i (nA)")
    ax2.set_title("B  —  Butler-Volmer i-E at different F_k values")
    ax2.legend(fontsize=8)

    # Panel C: peak current vs F_k (normalised, key for SWV model)
    ax3 = axes[2]
    Fk_arr = np.linspace(0, 1, 200)
    i_peak_norm = (1 - Fk_arr) * 100   # normalised to 100% at F_k=0
    ax3.plot(Fk_arr * 100, i_peak_norm, "k-", lw=2.5)
    ax3.axvline(0, color="gray", ls=":", lw=1)
    ax3.axhline(0, color="gray", ls=":", lw=1)
    ax3.set_xlabel("F_k (% cleavage)")
    ax3.set_ylabel("Normalised SWV peak current (%)")
    ax3.set_title("C  —  i_peak vs F_k (linear, normalised)")
    ax3.text(50, 60, "i_peak(t) = i_0 × (1 − F_k(t))", fontsize=10,
             ha="center", bbox=dict(boxstyle="round", facecolor="lightyellow"))

    fig.suptitle("Layer 10: Electron transfer — Marcus model + Butler-Volmer\n"
                 f"β = {beta/1e10:.1f} Å⁻¹ (CONFIDENT), k_ET_0 = {k_ET_0:.0e} s⁻¹ (FLAG), "
                 f"E°_MB = {E0_MB*1e3:.0f} mV vs Ag/AgCl", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L10_electron_transfer.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    d_target = 8.44e-9   # m (from probe_conformation at Gamma_p=3 pmol/cm²)
    print(f"k_ET at d_eff = {d_target*1e9:.1f} nm: {k_ET(d_target):.2f} s-1")
    print(f"SWV regime: {'quasi-reversible' if k_ET(d_target) > 0.1 else 'irreversible'}")
    plot_ET()
