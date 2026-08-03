"""
gold_electrode.py  —  Layer 4
===============================
Gold thin-film electrode characterisation: roughness factor and effective area.

Roughness factor rf = A_real / A_geometric
  Measured by: (1) Au oxide stripping peak area in H2SO4 cyclic voltammetry
                   (charge density for Au oxide monolayer = 390 µC/cm² — CONFIDENT,
                    standard value from Trasatti & Petrii 1992, Pure Appl Chem)
               (2) Pb underpotential deposition (UPD) — alternative method

Effective area A_eff = rf * A_geo

The roughness factor propagates to Layer 7 (Gamma_p per geometric area)
and Layer 11 (absolute peak current scaling).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Geometric electrode parameters ───────────────────────────────────────────
# Working electrode: circular disk, diameter 1.5 mm (typical screen-printed)
# FLAG: confirm WE diameter with device fabrication team
d_geo  = 1.5e-3          # m  ← FLAG: check with fabrication
A_geo  = np.pi * (d_geo/2)**2   # m²  = 1.77e-6 m² = 0.0177 cm²

# ── Roughness factor model ────────────────────────────────────────────────────
# Electrodeposited Au films typically rf = 1.2–2.5 depending on deposition
# conditions (current density, temperature, electrolyte).
# Sputtered Au: rf ≈ 1.05–1.3 (smoother).
# FLAG: measure rf by Au oxide stripping in your specific device.
# Use rf = 1.5 as representative for sputter-deposited then annealed film.
rf_nominal = 1.5    # ← FLAG: replace with measured value
rf_range   = np.linspace(1.0, 3.0, 100)

A_eff = rf_nominal * A_geo   # m²  (effective electroactive area)

# ── Au oxide stripping: back-calculation of rf ────────────────────────────────
# Measured charge in Au-oxide stripping peak:
Q_oxide_ref_uC_per_cm2 = 390.0   # µC/cm²  (Trasatti & Petrii 1992 — CONFIDENT)

def rf_from_Q_oxide(Q_measured_uC, A_geo_cm2):
    """Compute roughness factor from measured oxide-stripping charge."""
    return Q_measured_uC / (Q_oxide_ref_uC_per_cm2 * A_geo_cm2)


def effective_Gamma(Gamma_geo_pmol_cm2, rf):
    """
    Probe surface density per real area given geometric loading.
    Gamma_real = Gamma_geo / rf  (same amount spread over more real area → lower local density)
    This feeds into steric correction (Layer 8).
    """
    return Gamma_geo_pmol_cm2 / rf


def plot_roughness(save=True):
    A_geo_cm2 = A_geo * 1e4
    A_eff_range = rf_range * A_geo_cm2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.plot(rf_range, A_eff_range * 1e3, "b-", lw=2)   # cm² × 1e3 = mm²
    ax.axvline(rf_nominal, color="r", ls="--", lw=1.5,
               label=f"Nominal rf = {rf_nominal} (FLAG)")
    ax.axhline(A_geo_cm2 * 1e3, color="gray", ls=":", lw=1, label="Geometric area")
    ax.set_xlabel("Roughness factor rf")
    ax.set_ylabel("Effective area A_eff (mm²)")
    ax.set_title("A  —  Effective area vs roughness factor")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    Gamma_geo = 3.0   # pmol/cm² (target probe loading)
    Gamma_real = effective_Gamma(Gamma_geo, rf_range)
    ax2.plot(rf_range, Gamma_real, "g-", lw=2)
    ax2.axvline(rf_nominal, color="r", ls="--", lw=1.5, label=f"rf = {rf_nominal}")
    ax2.axhline(Gamma_geo, color="gray", ls=":", lw=1, label="Geometric Γ_p")
    ax2.set_xlabel("Roughness factor rf")
    ax2.set_ylabel("Real-area probe density Γ_p (pmol/cm²)")
    ax2.set_title("B  —  Local probe density vs roughness")
    ax2.legend(fontsize=9)

    fig.suptitle("Layer 4: Gold electrode roughness factor\n"
                 "(rf and WE diameter flagged — measure by Au-oxide stripping)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L4_gold_electrode.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"WE geometric area A_geo = {A_geo*1e4:.4f} cm²  (FLAG: verify diameter)")
    print(f"Nominal rf = {rf_nominal}  (FLAG: measure by oxide stripping)")
    print(f"Effective area A_eff = {A_eff*1e4:.4f} cm²")
    print(f"\nIf you measure Q_oxide = 12 µC, rf = "
          f"{rf_from_Q_oxide(12, A_geo*1e4):.2f}")
    plot_roughness()
