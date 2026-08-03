"""
double_layer.py  —  Layer 6
=============================
Gouy-Chapman-Stern (GCS) model for the electrochemical double layer.

C_total^{-1} = C_H^{-1} + C_GC^{-1}

where:
  C_H   = Helmholtz (inner) layer capacitance  [F/m²]
  C_GC  = Gouy-Chapman (diffuse) layer capacitance [F/m²]

C_GC = eps * kappa * cosh(F*phi_2 / (2*R*T))

kappa = Debye-Hückel parameter = sqrt(2*F²*I / (eps*R*T))
I     = ionic strength [mol/m³]

Impact on SWV: double-layer charging current i_c = C_dl * A * dE/dt
must be << Faradaic signal current i_F (condition for clean SWV).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Constants ────────────────────────────────────────────────────────────────
eps_0  = 8.854e-12    # F/m
eps_r  = 78.4         # water at 25°C  (CONFIDENT)
eps    = eps_0 * eps_r
F_     = 96485.0      # C/mol
R_     = 8.314        # J/mol/K
T_     = 298.15       # K

# ── Ionic strength ────────────────────────────────────────────────────────────
# PBS pH 7.4: ~150 mM NaCl + phosphate → I ≈ 163 mM
# Wound fluid chamber after dilution: ~80–100 mM
I_PBS    = 0.163      # M  (CONFIDENT — standard PBS composition)
I_chamber= 0.090      # M  (estimated after dilution with elution buffer)

def debye_length(I_M):
    """Debye screening length λ_D in nm."""
    I_SI = I_M * 1e3   # mol/m³
    kappa = np.sqrt(2 * F_**2 * I_SI / (eps * R_ * T_))
    return 1.0 / kappa * 1e9   # nm

def C_GC(I_M, phi_2_V=0.0):
    """Gouy-Chapman diffuse layer capacitance [µF/cm²]."""
    I_SI = I_M * 1e3
    kappa = np.sqrt(2 * F_**2 * I_SI / (eps * R_ * T_))
    C = eps * kappa * np.cosh(F_ * phi_2_V / (2 * R_ * T_))
    return C * 1e4 * 1e-6   # F/m² → µF/cm²

# Helmholtz layer capacitance: typically 20–50 µF/cm² for SAM-modified Au
# FLAG: C_H for MCH-SAM on Au in physiological buffer — literature range wide.
C_H = 30.0   # µF/cm²  ← FLAG

def C_total(I_M, phi_2_V=0.0):
    Cgc = C_GC(I_M, phi_2_V)
    return 1.0 / (1.0/C_H + 1.0/Cgc)   # µF/cm²

# ── Double-layer charging current ─────────────────────────────────────────────
# SWV pulse: dE/dt = Delta_E / tau_pulse
# Typical: Delta_E = 25 mV, tau = 5 ms → dE/dt = 5 V/s
Delta_E  = 25e-3    # V
tau_pulse= 5e-3     # s  (FLAG: confirm SWV pulse width with potentiostat spec)
dEdt     = Delta_E / tau_pulse   # V/s

A_geo_cm2= np.pi * (0.075)**2   # cm²  (1.5 mm dia WE)
C_total_chamber = C_total(I_chamber)   # µF/cm²
i_charging = C_total_chamber * 1e-6 * A_geo_cm2 * dEdt * 1e9   # nA

def plot_double_layer(save=True):
    I_range = np.linspace(0.01, 0.3, 200)
    phi_range = np.linspace(-0.2, 0.2, 200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: C_total vs ionic strength
    ax = axes[0]
    C_vals = [C_total(I) for I in I_range]
    ax.plot(I_range * 1e3, C_vals, "b-", lw=2, label="C_total (GCS)")
    ax.axvline(I_PBS*1e3,     color="g", ls="--", lw=1.5, label=f"PBS I={I_PBS*1e3:.0f} mM")
    ax.axvline(I_chamber*1e3, color="r", ls="--", lw=1.5, label=f"Chamber I={I_chamber*1e3:.0f} mM")
    ax.set_xlabel("Ionic strength I (mM)")
    ax.set_ylabel("C_total (µF/cm²)")
    ax.set_title("A  —  Double-layer capacitance vs ionic strength")
    ax.legend(fontsize=9)

    # Panel B: Debye length vs I
    ax2 = axes[1]
    ld_vals = [debye_length(I) for I in I_range]
    ax2.plot(I_range*1e3, ld_vals, "k-", lw=2)
    ax2.axvline(I_chamber*1e3, color="r", ls="--", lw=1.5,
                label=f"Chamber: λ_D = {debye_length(I_chamber):.1f} nm")
    ax2.set_xlabel("Ionic strength I (mM)")
    ax2.set_ylabel("Debye length λ_D (nm)")
    ax2.set_title("B  —  Debye screening length")
    ax2.legend(fontsize=9)

    fig.suptitle("Layer 6: Gouy-Chapman-Stern double layer\n"
                 f"C_H = {C_H} µF/cm² (FLAG), i_charging ≈ {i_charging:.2f} nA", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L6_double_layer.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Debye length (PBS):     {debye_length(I_PBS):.2f} nm")
    print(f"Debye length (chamber): {debye_length(I_chamber):.2f} nm")
    print(f"C_GC (chamber, phi=0):  {C_GC(I_chamber):.1f} µF/cm²")
    print(f"C_total (chamber):      {C_total(I_chamber):.1f} µF/cm²")
    print(f"DL charging current:    {i_charging:.3f} nA  (SWV pulse dE/dt = {dEdt:.0f} V/s)")
    plot_double_layer()
