"""
ratiometric.py  —  Layer 13
==============================
Ratiometric F_k readout and error propagation.

F_k = 1 - (i_active(T) / i_control(T)) / (i_active(0) / i_control(0))
    = 1 - R(T) / R(0)

where R(t) = i_active(t) / i_control(t)

Error propagation:
  sigma_Fk² = (dF_k/di_a)² sigma_ia² + (dF_k/di_c)² sigma_ic²
             + cross terms (if correlated noise)

Also models:
  - Cancellation of Ca²⁺ artefact (both electrodes affected equally)
  - Partial cancellation of fouling (if both electrodes foul equally)
  - Residual error from inter-electrode Gamma_p mismatch
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Signal model ───────────────────────────────────────────────────────────
# i_active(t)  = i_0 * (1 - F_k(t))  (cleaved probe loses MB signal)
# i_control(t) = i_0_ctrl            (scrambled probe, not cleaved → stable)
# Ratiometric ratio R(t) = i_active(t) / i_control(t)
# R(0) = i_0 / i_0_ctrl = 1 (if Gamma_p identical on both electrodes)

kcat_Km_eff = 188e3 * 0.3   # M-1s-1 (C15, alpha=0.3)
MW_MMP9     = 92e3
E_clinical  = [17e-9/MW_MMP9, 50e-9/MW_MMP9, 100e-9/MW_MMP9,
               150e-9/MW_MMP9, 200e-9/MW_MMP9]
E_labels    = ["17", "50", "100", "150", "200"]
colors_cl   = ["#1a237e","#1565c0","#0288d1","#00897b","#c62828"]

# Noise parameters (from Layer 14)
sigma_i_nA  = 0.15    # nA  (total RMS noise per measurement)
i_0_nA      = 5.0     # nA  (typical peak current at T=0)

T_arr = np.linspace(0, 3600, 500)


def Fk(E_M, T_s):
    return 1 - np.exp(-kcat_Km_eff * E_M * T_s)

def R_ratio(E_M, T_s, delta_Gamma_frac=0.0):
    """
    Ratiometric signal R(t) = i_active(t)/i_control(t).
    delta_Gamma_frac: fractional mismatch between active and control Gamma_p.
    """
    i_active  = i_0_nA * (1 - Fk(E_M, T_s)) * (1 + delta_Gamma_frac)
    i_control = i_0_nA
    return i_active / i_control

def Fk_measured(E_M, T_s, delta_Gamma_frac=0.0):
    return 1 - R_ratio(E_M, T_s, delta_Gamma_frac) / R_ratio(E_M, 0, delta_Gamma_frac)

def sigma_Fk(E_M, T_s, sigma_i=sigma_i_nA, i_0=i_0_nA):
    """Error propagation through ratiometric F_k."""
    # Partial derivatives:
    # dF_k/d(i_active)  = -1/(i_0_ctrl * R_0) = -1/i_0
    # dF_k/d(i_control) = R(T)/R_0/i_control = -(1-Fk)/i_0
    ia   = i_0 * (1 - Fk(E_M, T_s))
    ic   = i_0
    dFk_dia = -1.0 / (ic)        # 1/nA
    dFk_dic =  ia   / (ic**2)    # 1/nA
    return np.sqrt((dFk_dia * sigma_i)**2 + (dFk_dic * sigma_i)**2)


def plot_ratiometric(save=True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    # Panel A: F_k(t) with error bands
    ax = axes[0]
    for E_M, lbl, col in zip(E_clinical, E_labels, colors_cl):
        fk = np.array([Fk_measured(E_M, T) for T in T_arr]) * 100
        sg = np.array([sigma_Fk(E_M, T) for T in T_arr]) * 100
        ax.plot(T_arr/60, fk, color=col, lw=2, label=f"{lbl} ng/mL")
        ax.fill_between(T_arr/60, fk-2*sg, fk+2*sg, alpha=0.15, color=col)
    ax.axhspan(5, 90, alpha=0.07, color="green")
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("F_k (%)")
    ax.set_title("A  —  Ratiometric F_k(t) with 2σ error bands")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 60)

    # Panel B: sigma_Fk vs [E] at T=30 and 60 min
    ax2 = axes[1]
    E_range = np.linspace(E_clinical[0]*0.5, E_clinical[-1]*1.5, 200)
    for T_min, ls, col in [(30,"-","b"), (60,"--","r")]:
        sg = np.array([sigma_Fk(E, T_min*60) for E in E_range]) * 100
        ax2.plot(E_range*MW_MMP9*1e9, sg, color=col, ls=ls, lw=2,
                 label=f"T={T_min} min")
    ax2.set_xlabel("[MMP-9] (ng/mL)")
    ax2.set_ylabel("σ(F_k) (%)")
    ax2.set_title("B  —  Measurement uncertainty vs [E]")
    ax2.legend(fontsize=9)

    # Panel C: Gamma_p mismatch effect on F_k error
    ax3 = axes[2]
    mismatch = np.linspace(-0.2, 0.2, 200)
    T_60 = 3600
    for E_M, lbl, col in zip([E_clinical[0], E_clinical[-1]],
                               ["17 ng/mL", "200 ng/mL"],
                               ["b", "r"]):
        fk_err = np.array([Fk_measured(E_M, T_60, dg) for dg in mismatch]) * 100
        fk_true = Fk_measured(E_M, T_60, 0.0) * 100
        ax3.plot(mismatch*100, fk_err - fk_true, color=col, lw=2, label=lbl)
    ax3.axhline(1, color="gray", ls="--", lw=1, label="1% error threshold")
    ax3.axhline(-1, color="gray", ls="--", lw=1)
    ax3.set_xlabel("Γ_p inter-electrode mismatch (%)")
    ax3.set_ylabel("F_k error from mismatch (%)")
    ax3.set_title("C  —  Ratiometric error from Γ_p mismatch")
    ax3.legend(fontsize=9)

    # Panel D: Ca²⁺ cancellation check
    ax4 = axes[3]
    beta_MB = 1e10   # m-1
    Kd_Ca   = 2e-3   # M
    delta_d_max = 0.2e-9   # m
    Ca_arr  = np.linspace(0, 5e-3, 200)
    theta   = Ca_arr / (Kd_Ca + Ca_arr)
    k_ratio = np.exp(beta_MB * theta * delta_d_max)

    # If both electrodes identical: F_k artefact ≈ 0
    # If 5% Gamma_p mismatch: residual error scales with (k_ratio-1)*mismatch
    residual_5pct = 0.05 * (k_ratio - 1) / k_ratio * 100
    ax4.plot(Ca_arr*1e3, residual_5pct, "purple", lw=2, label="5% Γ_p mismatch")
    ax4.axvline(0.4, color="gray", ls="--", lw=1.5, label="Chamber [Ca²⁺] ≈ 0.4 mM")
    ax4.set_xlabel("[Ca²⁺] (mM)")
    ax4.set_ylabel("Residual F_k artefact (%)")
    ax4.set_title("D  —  Ca²⁺ artefact after ratiometric cancellation")
    ax4.legend(fontsize=9)

    fig.suptitle("Layer 13: Ratiometric readout — F_k computation and error propagation\n"
                 r"$F_k = 1 - R(T)/R(0)$ where $R = i_\mathrm{active}/i_\mathrm{control}$",
                 fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L13_ratiometric.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_ratiometric()
