"""
forward_chain.py  —  Layer 16  (INTEGRATION)
===============================================
Full end-to-end pipeline simulation:

  [E]_wound  →  [E]_chamber  →  [E]_surface  →  F_k(T)  →  i_peak(T)

Chains together all preceding layer models.

Pipeline:
  Layer 0:  wound_biology  → [MMP9_free], [HNE_free] at t_swab
  Layer 1:  sample_collection → [E]_chamber (dilution + elution)
  Layer 3:  mass_transport    → [E]_surface (diffusion to electrode)
  Layer 3b: inhibitor_kinetics → [E]_free(t) (in-device depletion)
  Layer 3c: fouling_model      → alpha_foul  (flux attenuation)
  Layer 9:  surface_kinetics   → Gamma(t), F_k(T)
  Layer 11: swv_current_model  → i_peak(T) (normalised)
  Layer 13: ratiometric        → F_k_measured (with noise)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Layer parameters (consolidated) ──────────────────────────────────────────
MW_MMP9     = 92e3
eta_MMP9    = 0.70           # elution efficiency
Df          = 2.0            # dilution factor
kcat_Km_eff = 67.5e3 * 0.3  # M-1s-1 (surface est., A11/SGKIPRTLTA, alpha=0.3)
k_on_TIMP1  = 2e5            # M-1s-1 (FLAG)
k_off_TIMP1 = 2e-5           # s-1
k_f_foul    = 1e3            # M-1s-1 (FLAG)
k_d_foul    = 1e-4           # s-1    (FLAG)
P_bulk      = 150e-6         # M total fouling protein (FLAG)
h_channel   = 100e-6         # m
h_f_max     = 10e-9          # m
sigma_Fk    = 0.005          # measurement noise

# Representative TIMP-1 in chamber (from Layer 1)
TIMP1_chamber = (200e-6 / 28.5e3) * 0.65 / Df   # mol/L

def pipeline(E_wound_M, T_incubation_s=1800):
    """
    Full forward simulation for a given wound [MMP-9] and incubation time.
    Returns: F_k_true, F_k_measured, i_peak_norm
    """
    # Layer 1: elution + dilution
    E_chamber = E_wound_M * eta_MMP9 / Df

    # Layer 3b: in-device inhibitor depletion during incubation
    def inhibit_ode(t, y):
        E_f, I_f, EI = y
        v_on  = k_on_TIMP1  * E_f * I_f
        v_off = k_off_TIMP1 * EI
        return [-v_on+v_off, -v_on+v_off, v_on-v_off]

    y0 = [E_chamber, TIMP1_chamber, 0.0]
    sol_inh = solve_ivp(inhibit_ode, (0, T_incubation_s), y0,
                        t_eval=[T_incubation_s/2, T_incubation_s],
                        method="Radau", rtol=1e-9, atol=1e-13)
    E_free_mid  = sol_inh.y[0, 0]
    E_free_end  = sol_inh.y[0, 1]
    E_eff = (E_chamber + E_free_mid + E_free_end) / 3   # time-averaged free [E]

    # Layer 3c: fouling attenuation
    def foul_ode(t, Th):
        return [k_f_foul * P_bulk * (1-Th[0]) - k_d_foul * Th[0]]
    sol_f = solve_ivp(foul_ode, (0, T_incubation_s), [0.0],
                      t_eval=[T_incubation_s], method="RK45")
    Theta_f = sol_f.y[0, -1]
    h_f = h_f_max * Theta_f
    alpha_foul = h_channel / (h_channel + h_f)

    # Layer 9: surface kinetics (first-order approximation)
    k_obs = kcat_Km_eff * E_eff * alpha_foul
    F_k_true = 1 - np.exp(-k_obs * T_incubation_s)

    # Layer 11: normalised peak current
    i_peak = (1 - F_k_true) * 100

    # Layer 13: measured F_k (add noise)
    F_k_measured = F_k_true + np.random.normal(0, sigma_Fk)
    F_k_measured = np.clip(F_k_measured, 0, 1)

    return F_k_true, F_k_measured, i_peak


def plot_forward_chain(save=True):
    np.random.seed(42)
    E_range_ngml = np.linspace(5, 250, 60)
    E_range_M    = E_range_ngml * 1e-6 / MW_MMP9

    # Run at T=30 and T=60 min
    results = {}
    for T_min in [30, 60]:
        fk_true, fk_meas, ipeak = [], [], []
        for E_M in E_range_M:
            ft, fm, ip = pipeline(E_M, T_min*60)
            fk_true.append(ft); fk_meas.append(fm); ipeak.append(ip)
        results[T_min] = {
            "fk_true": np.array(fk_true),
            "fk_meas": np.array(fk_meas),
            "ipeak":   np.array(ipeak),
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    colors_T = {"30": "#1565c0", "60": "#c62828"}

    # Panel A: F_k true vs [E] — pipeline output
    ax = axes[0]
    for T_min in [30, 60]:
        r = results[T_min]
        ax.plot(E_range_ngml, r["fk_true"]*100, color=colors_T[str(T_min)], lw=2,
                label=f"T={T_min} min (true)")
        ax.scatter(E_range_ngml[::4], r["fk_meas"][::4]*100,
                   color=colors_T[str(T_min)], s=20, alpha=0.5, label=f"T={T_min} min (meas)")
    ax.axhspan(5, 90, alpha=0.07, color="green", label="Operating window")
    ax.axvspan(17, 200, alpha=0.05, color="blue", label="Clinical range")
    ax.set_xlabel("[MMP-9] wound (ng/mL)")
    ax.set_ylabel("F_k (%)")
    ax.set_title("A  —  End-to-end F_k vs [MMP-9]_wound")
    ax.legend(fontsize=7)

    # Panel B: i_peak vs [E]
    ax2 = axes[1]
    for T_min in [30, 60]:
        ax2.plot(E_range_ngml, results[T_min]["ipeak"],
                 color=colors_T[str(T_min)], lw=2, label=f"T={T_min} min")
    ax2.axvspan(17, 200, alpha=0.05, color="blue")
    ax2.set_xlabel("[MMP-9] wound (ng/mL)")
    ax2.set_ylabel("Normalised i_peak (%)")
    ax2.set_title("B  —  SWV peak current vs [MMP-9]_wound")
    ax2.legend(fontsize=9)

    # Panel C: pipeline stages waterfall at [E]=100 ng/mL
    ax3 = axes[2]
    E_demo  = 100e-6 / MW_MMP9
    stages  = ["Wound\n[E]", "Chamber\n[E]", "Free\n[E] (30min)", "k_obs·T", "F_k"]
    E_wound = E_demo
    E_cham  = E_demo * eta_MMP9 / Df
    ft, fm, ip = pipeline(E_demo, 1800)
    E_free   = E_cham * 0.7   # approximate (after inhibitor depletion)
    k_obs_T  = kcat_Km_eff * E_free * 1800   # dimensionless
    stage_vals = [E_wound*1e9, E_cham*1e9, E_free*1e9, k_obs_T*10, ft*100]
    stage_units= ["nM","nM","nM","×0.1","% F_k"]
    bar_cols = ["#1a237e","#1565c0","#0288d1","#2e7d32","#c62828"]
    bars = ax3.bar(stages, stage_vals, color=bar_cols, alpha=0.85)
    ax3.set_ylabel("Value (see legend for units)")
    ax3.set_title("C  —  Pipeline waterfall at [MMP-9]=100 ng/mL, T=30 min")
    for bar, val, unit in zip(bars, stage_vals, stage_units):
        ax3.text(bar.get_x()+bar.get_width()/2, val*1.04,
                 f"{val:.2f}\n({unit})", ha="center", fontsize=8)

    # Panel D: Sensitivity analysis — effect of alpha on F_k at T=30 min
    ax4 = axes[3]
    alpha_range = np.linspace(0.05, 1.5, 100)
    E_hi = 200e-6 / MW_MMP9
    E_lo = 17e-6  / MW_MMP9
    for E_M, lbl, col in [(E_lo,"17 ng/mL","b"),(E_hi,"200 ng/mL","r")]:
        Fk_alpha = 1 - np.exp(-67.5e3 * alpha_range * E_M * 1800)
        ax4.plot(alpha_range, Fk_alpha*100, color=col, lw=2, label=lbl)
    ax4.axvline(0.3, color="gray", ls="--", lw=1.5, label="α = 0.3 (nominal, FLAG)")
    ax4.axhspan(5, 90, alpha=0.07, color="green", label="Operating window")
    ax4.set_xlabel("Surface correction factor α")
    ax4.set_ylabel("F_k at T=30 min (%)")
    ax4.set_title("D  —  Sensitivity to surface correction α")
    ax4.legend(fontsize=8)

    fig.suptitle("Layer 16: Full forward chain — [MMP-9]_wound → SWV signal\n"
                 "Layers 0→1→3→3b→3c→9→11→13 chained", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B14_forward_chain.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_forward_chain()
    # Summary at clinical range endpoints
    for E_ng in [17, 100, 200]:
        E_M = E_ng * 1e-6 / MW_MMP9
        ft30, _, _ = pipeline(E_M, 1800)
        ft60, _, _ = pipeline(E_M, 3600)
        print(f"[MMP-9]={E_ng:3d} ng/mL: F_k(30min)={ft30*100:.1f}%  F_k(60min)={ft60*100:.1f}%")
