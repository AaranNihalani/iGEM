"""
calcium_artefact.py  —  Layer 7
==================================
Edinburgh Ca²⁺ artefact: PEG-Ca²⁺ coordination contracts the spacer,
bringing MB closer to the gold surface → higher k_ET → artefactual peak increase.

Model:
  Theta_Ca = [Ca²⁺] / (Kd + [Ca²⁺])         (fractional PEG chelation)
  delta_d   = Theta_Ca * delta_d_max           (PEG contraction, nm)
  k_ET_Ca   = k_ET_0 * exp(beta * delta_d)    (tunnelling rate increase)

Ratiometric cancellation:
  F_k = 1 - R_active/R_control
  Ca²⁺ multiplies BOTH numerator and denominator by same factor IF
  Gamma_p is identical on both electrodes.

  Residual F_k error from inter-electrode Gamma_p mismatch delta_Gamma:
  error_Fk ≈ (delta_Gamma / Gamma_p) * (1 - k_ET_Ca/k_ET_0) * (peak shape factor)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── PEG-Ca²⁺ binding ────────────────────────────────────────────────────────
# Kd for PEG-Ca²⁺ crown-ether-like chelation:
# FLAG: no direct measurement for PEG4/PEG6 spacer; using analogy from
# polyethylene glycol / Ca²⁺ isothermal calorimetry (Branca et al. 2002).
Kd_Ca = 2e-3      # M  (2 mM)  ← FLAG

# Maximum PEG contraction upon full Ca²⁺ chelation:
# Estimated from persistence length reduction in chelated PEG4 (4-unit EO chain).
# PEG4 end-to-end ~ 1.5 nm extended; contraction ~ 0.1–0.3 nm.
# FLAG: no direct measurement for this probe architecture.
delta_d_max = 0.2e-9   # m = 0.2 nm  ← FLAG

# Tunnelling decay constant for MB through alkanethiol-like spacer:
# beta ~ 1.0 per Å = 1e10 m-1 (well established for saturated carbon chain,
# Smalley et al. 2003 JACS — CONFIDENT as order of magnitude)
beta = 1.0e10   # m-1  (CONFIDENT)

# Ca²⁺ in wound fluid:
# Ionised Ca²⁺ in wound fluid ~ 1.0–1.3 mM (similar to serum — CONFIDENT)
Ca_wound   = 1.2e-3   # M
Ca_chamber = Ca_wound * 30e-6 / (30e-6 + 60e-6)   # after dilution ~ 0.4 mM


def theta_Ca(Ca_M):
    return Ca_M / (Kd_Ca + Ca_M)

def kET_ratio(Ca_M):
    """k_ET_Ca / k_ET_0 — fold increase in ET rate."""
    th = theta_Ca(Ca_M)
    dd = th * delta_d_max
    return np.exp(beta * dd)

def residual_Fk_error(Ca_M, delta_Gamma_frac):
    """
    Approximate F_k artefact from Ca²⁺ when active and control electrodes
    have Gamma_p differing by delta_Gamma_frac (fractional mismatch).
    """
    ratio = kET_ratio(Ca_M)
    # Peak height ~ proportional to k_ET in the quasi-reversible SWV regime
    # Artefact: R_active inflated by ratio*(1 + delta_Gamma_frac)
    #           R_control inflated by ratio
    # F_k = 1 - R_active/R_control ≈ -delta_Gamma_frac (from Taylor expansion)
    # The Ca²⁺ contribution to error is second-order in delta_Gamma × (ratio-1)
    return delta_Gamma_frac * (ratio - 1) / ratio


def plot_ca_artefact(save=True):
    Ca_range = np.linspace(0, 5e-3, 300)    # 0–5 mM

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.plot(Ca_range*1e3, theta_Ca(Ca_range)*100, "b-", lw=2)
    ax.axvline(Ca_chamber*1e3, color="r", ls="--", lw=1.5,
               label=f"Chamber [Ca²⁺] = {Ca_chamber*1e3:.1f} mM")
    ax.set_xlabel("[Ca²⁺] (mM)")
    ax.set_ylabel("PEG chelation Θ_Ca (%)")
    ax.set_title("A  —  PEG-Ca²⁺ chelation")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(Ca_range*1e3, kET_ratio(Ca_range), "g-", lw=2)
    ax2.axvline(Ca_chamber*1e3, color="r", ls="--", lw=1.5,
                label=f"Fold increase = {kET_ratio(Ca_chamber):.3f}×")
    ax2.axhline(1, color="gray", ls=":", lw=1)
    ax2.set_xlabel("[Ca²⁺] (mM)")
    ax2.set_ylabel("k_ET fold increase")
    ax2.set_title("B  —  k_ET increase from PEG contraction")
    ax2.legend(fontsize=9)

    ax3 = axes[2]
    mismatch_range = np.linspace(0, 0.20, 100)   # 0–20% Gamma_p mismatch
    for Ca_val, col, ls in [(Ca_chamber, "r", "-"), (1.2e-3, "b", "--")]:
        err = residual_Fk_error(Ca_val, mismatch_range) * 100
        ax3.plot(mismatch_range*100, err, color=col, ls=ls, lw=2,
                 label=f"[Ca²⁺] = {Ca_val*1e3:.1f} mM")
    ax3.set_xlabel("Γ_p inter-electrode mismatch (%)")
    ax3.set_ylabel("Residual F_k error (%)")
    ax3.set_title("C  —  Ratiometric cancellation residual")
    ax3.legend(fontsize=9)
    ax3.axhline(1, color="gray", ls=":", lw=1, label="1% F_k threshold")

    fig.suptitle("Layer 7: Edinburgh Ca²⁺ artefact — PEG chelation model\n"
                 "(Kd_Ca, delta_d_max flagged — see calcium_artefact.py)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L7c_calcium.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    print(f"Ca²⁺ in chamber: {Ca_chamber*1e3:.2f} mM")
    print(f"k_ET fold increase at chamber [Ca²⁺]: {kET_ratio(Ca_chamber):.4f}×")
    print(f"Residual F_k error at 5% Gamma_p mismatch: "
          f"{residual_Fk_error(Ca_chamber, 0.05)*100:.3f}%")


if __name__ == "__main__":
    plot_ca_artefact()
