"""
reference_counter.py  —  Layer 5
===================================
On-chip Ag/AgCl reference electrode potential and counter electrode sizing.

Reference electrode:
  E_ref = E°_AgCl + (RT/F) * ln([Cl-])
  E°_AgCl = +0.222 V vs SHE at 25°C  (CONFIDENT — standard textbook value)

Counter electrode:
  Must pass total anodic current without becoming potential-limiting.
  Sizing: A_CE > i_peak * t_pulse / (i_lim_CE)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Constants ────────────────────────────────────────────────────────────────
R  = 8.314    # J/mol/K
F  = 96485    # C/mol
T  = 298.15   # K

# ── Ag/AgCl reference potential ──────────────────────────────────────────────
E0_AgCl = 0.2224   # V vs SHE at 25°C  (Bard & Faulkner, Electrochemical Methods — CONFIDENT)

def E_ref(Cl_M):
    """Ag/AgCl potential vs SHE as a function of [Cl-]."""
    return E0_AgCl - (R * T / F) * np.log(Cl_M)

# Wound fluid [Cl-]: physiological saline ~ 150 mM (CONFIDENT — clinical chemistry)
Cl_wound   = 0.150   # M
Cl_PBS     = 0.137   # M  (PBS pH 7.4)
# After dilution in device (~2×): ~75–68 mM
Cl_chamber = (Cl_wound * 30e-6 + Cl_PBS * 60e-6) / (30e-6 + 60e-6)

E_ref_wound   = E_ref(Cl_wound)
E_ref_chamber = E_ref(Cl_chamber)
E_ref_PBS     = E_ref(Cl_PBS)

# ── Potential drift with [Cl-] uncertainty ───────────────────────────────────
# If [Cl-] varies ±20% around chamber value:
Cl_range = np.linspace(0.05, 0.25, 200)
E_ref_range = E_ref(Cl_range)

# ── Counter electrode sizing ─────────────────────────────────────────────────
# Peak SWV current ~ 1–10 nA (estimated from Layer 11 / noise budget)
# Counter electrode must source this without dropping below water oxidation onset
i_peak_est = 10e-9     # A (conservative upper estimate)
t_SWV_scan = 5.0       # s (total SWV scan duration)
# FLAG: confirm with instrument specs
j_lim_CE   = 1e-3      # A/cm²  (limiting current density for carbon CE — conservative)
A_CE_min   = i_peak_est / j_lim_CE   # cm²

def plot_reference(save=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.plot(Cl_range * 1e3, (E_ref_range - E_ref_PBS) * 1e3, "b-", lw=2)
    ax.axvline(Cl_chamber * 1e3, color="r", ls="--", lw=1.5,
               label=f"Chamber [Cl⁻] = {Cl_chamber*1e3:.0f} mM")
    ax.axhline(0, color="gray", ls=":", lw=1, label="PBS reference")
    ax.set_xlabel("[Cl⁻] (mM)")
    ax.set_ylabel("ΔE_ref vs PBS (mV)")
    ax.set_title("A  —  Reference potential drift with [Cl⁻]")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    # Nernst shift per decade of [Cl-]:  dE/d(log[Cl-]) = -RT/F * ln(10) = -59.2 mV/decade
    log_Cl = np.log10(Cl_range)
    nernst_slope = -(R * T / F) * np.log(10) * 1e3   # mV/decade
    ax2.plot(log_Cl, E_ref_range * 1e3, "k-", lw=2, label="E_ref vs SHE")
    ax2.axvline(np.log10(Cl_chamber), color="r", ls="--", lw=1.5)
    ax2.set_xlabel("log₁₀[Cl⁻]")
    ax2.set_ylabel("E_ref vs SHE (mV)")
    ax2.set_title(f"B  —  Nernst response ({nernst_slope:.0f} mV/decade)")
    ax2.legend(fontsize=9)

    fig.suptitle("Layer 5: On-chip Ag/AgCl reference electrode\n"
                 "E°(AgCl) = 222.4 mV vs SHE  (Bard & Faulkner 2001)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L5_reference.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"E_ref (wound fluid):  {E_ref_wound*1e3:.1f} mV vs SHE")
    print(f"E_ref (chamber):      {E_ref_chamber*1e3:.1f} mV vs SHE")
    print(f"E_ref (PBS):          {E_ref_PBS*1e3:.1f} mV vs SHE")
    print(f"Shift wound→chamber:  {(E_ref_chamber-E_ref_wound)*1e3:.1f} mV")
    print(f"Min CE area needed:   {A_CE_min*1e4:.2e} mm²  (trivially small)")
    plot_reference()
