"""
swv_forward.py  —  Layer 11
==============================
Osteryoung square-wave voltammetry (SWV) forward model.

Net SWV current:  i_net(E) = i_forward(E) - i_backward(E)

For a surface-confined redox couple (MB, n=2):
  i_forward  = n*F*A*Gamma * k_ET * [exp(alpha*F*(E-E°)/RT) / Z]
  i_backward = n*F*A*Gamma * k_ET * [exp(-(1-alpha)*F*(E-E°)/RT) / Z]

Z = exp(alpha*F*(E-E°)/RT) + exp(-(1-alpha)*F*(E-E°)/RT)

In the quasi-reversible SWV regime (k_ET comparable to scan frequency),
the peak position is at E° and the peak height scales with k_ET.
At high k_ET (reversible): peak is symmetric, max height = n²F²A*Gamma*delta_E / (4RT)
At low k_ET (irreversible): peak broadens and height drops.

This model generates synthetic SWV voltammograms at different F_k values
(to simulate how the peak shrinks as MB is cleaved off).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── SWV parameters ────────────────────────────────────────────────────────────
n         = 2          # electrons transferred by MB (CONFIDENT — MB is 2e- redox)
F_        = 96485.0
R_        = 8.314
T_        = 298.15
RT_F      = R_ * T_ / F_
E0_MB     = -0.250     # V vs Ag/AgCl  (CONFIDENT)
alpha_BV  = 0.5

# SWV scan parameters:
# FLAG: confirm with instrument (potentiostat) specifications
delta_E   = 0.025      # V  (pulse amplitude, 25 mV — FLAG)
dE_step   = 0.005      # V  (step potential, 5 mV — FLAG)
f_SWV     = 15.0       # Hz  (frequency — FLAG)
tau_pulse = 1 / (2 * f_SWV)   # s (half-period)

# Electrode parameters
A_cm2     = 0.0177     # cm²  (1.5 mm dia WE, geometric)
Gamma_0   = 3e-12      # mol/cm²  (initial probe density)
k_ET_ref  = 50.0       # s-1  (estimated from Layer 10 at d_eff ~ 8 nm)
# FLAG: k_ET_ref — depends on probe architecture, measure by AC voltammetry


def swv_current_surface(E_arr, Gamma_MB, k_ET, n=2):
    """
    Net SWV current for surface-confined redox couple.
    Uses quasi-reversible model (Gulaboski & Mirceski, Electrochimica Acta 2015).
    """
    theta = n * F_ * (E_arr - E0_MB) / (R_ * T_)

    # Forward and backward pulse potentials
    E_fwd  = E_arr + delta_E
    E_bwd  = E_arr - delta_E
    th_fwd = n * F_ * (E_fwd - E0_MB) / (R_ * T_)
    th_bwd = n * F_ * (E_bwd - E0_MB) / (R_ * T_)

    # Dimensionless kinetic parameter
    kappa = k_ET * tau_pulse

    def i_pulse(theta_p):
        # Simplified: reversible limit scaled by kappa/(1+kappa)
        exp_f = np.exp( alpha_BV   * theta_p)
        exp_b = np.exp(-(1-alpha_BV) * theta_p)
        denom = exp_f + exp_b
        current = n * F_ * A_cm2 * Gamma_MB * k_ET * (exp_f - exp_b) / denom
        return current * kappa / (1 + kappa)

    i_net = i_pulse(th_fwd) - i_pulse(th_bwd)
    return i_net


def plot_swv(save=True):
    E_scan = np.linspace(-0.5, 0.0, 500)   # V vs Ag/AgCl
    Fk_vals   = [0.0, 0.25, 0.50, 0.75]
    colors    = ["#1a237e", "#1565c0", "#0288d1", "#80d8ff"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: SWV voltammogram at different F_k
    ax = axes[0]
    for Fk, col in zip(Fk_vals, colors):
        Gamma_MB = Gamma_0 * (1 - Fk)
        i_net_nA = swv_current_surface(E_scan, Gamma_MB, k_ET_ref) * 1e9
        ax.plot(E_scan*1e3, i_net_nA, color=col, lw=2,
                label=f"F_k = {Fk*100:.0f}%")
    ax.axvline(E0_MB*1e3, color="gray", ls="--", lw=1, label="E°_MB")
    ax.set_xlabel("E (mV vs Ag/AgCl)")
    ax.set_ylabel("Net SWV current i_net (nA)")
    ax.set_title("A  —  SWV voltammogram at different F_k values")
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # Panel B: Peak height vs k_ET (quasi-reversible effect)
    ax2 = axes[1]
    kET_range = np.logspace(-1, 4, 200)
    i_peak_kET = np.array([
        np.max(np.abs(swv_current_surface(E_scan, Gamma_0, k))) * 1e9
        for k in kET_range
    ])
    ax2.semilogx(kET_range, i_peak_kET, "r-", lw=2)
    ax2.axvline(k_ET_ref, color="b", ls="--", lw=1.5,
                label=f"k_ET = {k_ET_ref} s⁻¹ (FLAG)")
    ax2.set_xlabel("k_ET (s⁻¹)")
    ax2.set_ylabel("SWV peak current (nA)")
    ax2.set_title("B  —  Peak height vs k_ET (quasi-reversible effect)")
    ax2.legend(fontsize=9)

    # Panel C: Peak area vs F_k  (area ∝ charge = n*F*Gamma remaining)
    ax3 = axes[2]
    Fk_cont  = np.linspace(0, 1, 200)
    i_peak_Fk = np.array([
        np.max(np.abs(swv_current_surface(E_scan, Gamma_0*(1-Fk), k_ET_ref))) * 1e9
        for Fk in Fk_cont
    ])
    ax3.plot(Fk_cont*100, i_peak_Fk, "k-", lw=2.5)
    ax3.axhline(0, color="gray", ls=":", lw=1)
    ax3.set_xlabel("F_k (% cleavage)")
    ax3.set_ylabel("SWV peak current (nA)")
    ax3.set_title("C  —  Peak current drops linearly with F_k")
    ax3.text(50, i_peak_Fk[0]*0.6, "i_peak = i_0 × (1 − F_k)", fontsize=10,
             ha="center", bbox=dict(boxstyle="round", facecolor="lightyellow"))

    fig.suptitle("Layer 11: SWV forward model — Osteryoung SWV (surface-confined MB)\n"
                 f"E°_MB = {E0_MB*1e3:.0f} mV, n=2, f={f_SWV} Hz (FLAG), "
                 f"ΔE={delta_E*1e3:.0f} mV (FLAG)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B11_swv_forward.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_swv()
