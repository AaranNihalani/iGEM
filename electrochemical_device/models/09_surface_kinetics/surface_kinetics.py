"""
surface_kinetics.py  —  Layer 9  (CORE MODEL)
================================================
Surface Michaelis-Menten ODE in both regimes:

  dΓ/dt = -kcat * [E] * Γ / (Km_surf + Γ)

ZEROTH-ORDER regime (Γ >> Km_surf):
  dΓ/dt ≈ -kcat * [E]          (rate independent of Γ)
  Γ(t) = Γ_0 - kcat * [E] * t
  F_k(t) = kcat * [E] * t / Γ_0    ← LINEAR in time

FIRST-ORDER regime (Γ << Km_surf):
  dΓ/dt ≈ -(kcat/Km_surf) * [E] * Γ
  Γ(t) = Γ_0 * exp(-k_obs * t)
  F_k(t) = 1 - exp(-k_obs * t)     ← EXPONENTIAL in time
  where k_obs = (kcat/Km_surf) * [E]

The CROSSOVER occurs at Γ = Km_surf.
We WANT first-order: Γ << Km_surf throughout the assay.

This is the critical model for interpreting experimental data.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Parameters ────────────────────────────────────────────────────────────────
# C15 substrate (MMP-9 channel), solution values from Max's assay
kcat_sol  = 703.0          # s-1  (C15, from experimental data)
Km_sol_mM = 3.7            # mM
Km_sol    = Km_sol_mM * 1e-3   # M

# Surface correction factor alpha (empirical; from Layer 8 steric model)
alpha_base = 0.3           # FLAG: must be measured (Plaxco group best practice)

# Effective surface 2nd-order rate constant: kcat/Km × alpha
# Units: M⁻¹s⁻¹  — this is the correctly-calibrated constant used in the ODE
kcat_Km_eff = 188e3 * alpha_base   # M⁻¹s⁻¹  (solution assay value × alpha)

# Surface crossover concentration Km_surf [mol/cm²]
# This is the effective surface Km, derived from solution Km × probe-layer thickness / f_steric
# It sets the boundary between zeroth-order (Γ >> Km_surf) and first-order (Γ << Km_surf) regime
delta     = 5.5e-9         # m (L_PEG + L_peptide; from Layer 8)
f_steric  = 0.6            # steric accessibility factor (from steric_hindrance.py)
Km_surf_molcm2 = Km_sol * 1e-3 * (delta * 1e2) / f_steric   # mol/cm²
# = Km_sol [mol/L] × 1e-3 [L/cm³ → mol/cm³] × delta [cm] / f_steric ≈ 3.39 pmol/cm²

# Initial probe density
Gamma_0_pmol = 3.0         # pmol/cm²
Gamma_0      = Gamma_0_pmol * 1e-12   # mol/cm²

# Clinical [E] range (C15 MMP-9 channel)
MW_MMP9     = 92e3
E_lo_M      = 17e-6 / MW_MMP9    # 17 ng/mL → 0.185 nM
E_hi_M      = 200e-6 / MW_MMP9   # 200 ng/mL → 2.17 nM
E_clinical  = np.array([E_lo_M, 0.5e-9, 1e-9, E_hi_M])
E_labels    = ["17 ng/mL (0.18 nM)", "0.5 nM", "1.0 nM", "200 ng/mL (2.17 nM)"]

print(f"Km_sol     = {Km_sol*1e3:.1f} mM")
print(f"Km_surf    = {Km_surf_molcm2*1e12:.4f} pmol/cm²")
print(f"Gamma_0    = {Gamma_0_pmol:.1f} pmol/cm²")
print(f"Gamma_0/Km_surf = {Gamma_0/Km_surf_molcm2:.2f}  "
      f"(>> 1 → zeroth-order; << 1 → first-order)")

# ── ODE solver ────────────────────────────────────────────────────────────────
# Full surface MM ODE (dimensionally consistent):
#   dΓ/dt = -kcat_Km_eff [M⁻¹s⁻¹] × E [M] × Γ [mol/cm²] × Km_surf / (Km_surf + Γ)
# Units: (L/mol/s) × (mol/L) × (mol/cm²) × (mol/cm²)/(mol/cm²) = mol/cm²/s ✓
# Zeroth-order limit (Γ >> Km_surf): rate → kcat_Km_eff × E × Km_surf = const
# First-order limit  (Γ << Km_surf): rate → kcat_Km_eff × E × Γ       (linear in Γ)
def surface_kinetics_ode(t, Gamma, E_M, kcat_Km, Km_surf):
    v = kcat_Km * E_M * Km_surf * Gamma[0] / (Km_surf + Gamma[0])
    return [-v]

def solve_Gamma(E_M, t_end_s=3600, Gamma_0=Gamma_0, kcat_Km=kcat_Km_eff,
                Km_surf=Km_surf_molcm2, N=1000):
    t_eval = np.linspace(0, t_end_s, N)
    sol = solve_ivp(surface_kinetics_ode, (0, t_end_s), [Gamma_0],
                    args=(E_M, kcat_Km, Km_surf),
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-15)
    Fk = 1 - sol.y[0] / Gamma_0
    return sol.t, sol.y[0], Fk

# ── Analytic approximations ────────────────────────────────────────────────────
def Fk_zeroth_order(E_M, t_arr, kcat_Km=kcat_Km_eff, Km_surf=Km_surf_molcm2,
                    Gamma_0_val=None):
    """Zeroth-order analytic: valid when Γ >> Km_surf (constant cleavage rate)."""
    if Gamma_0_val is None:
        Gamma_0_val = Gamma_0
    V_max = kcat_Km * E_M * Km_surf   # mol/cm²/s (const rate at saturation)
    Gamma_t = np.maximum(0, Gamma_0_val - V_max * t_arr)
    return 1 - Gamma_t / Gamma_0_val

def Fk_first_order(E_M, t_arr, kcat_Km=kcat_Km_eff):
    """First-order analytic: valid when Γ << Km_surf (exponential decay)."""
    k_obs = kcat_Km * E_M   # s⁻¹
    return 1 - np.exp(-k_obs * t_arr)


def plot_surface_kinetics(save=True):
    t_max = 3600   # 60 min
    t_arr = np.linspace(0, t_max, 1000)
    E_demo = E_hi_M   # upper clinical [E] for regime demonstration

    # Full ODE
    t_ode, Gamma_ode, Fk_ode = solve_Gamma(E_demo, t_end_s=t_max)

    # Analytic limits
    Fk_zero  = Fk_zeroth_order(E_demo, t_arr)
    Fk_first = Fk_first_order(E_demo, t_arr)

    # Crossover time: when Gamma(t) = Km_surf
    # Zeroth-order approximation: Gamma_0 - kcat*E*t_cross = Km_surf
    Gamma_ratio = Gamma_0 / Km_surf_molcm2
    regime_label = "zeroth-order" if Gamma_ratio > 10 else \
                   "first-order"  if Gamma_ratio < 0.1 else "intermediate"

    fig = plt.figure(figsize=(14, 9))
    gs  = plt.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Panel A: F_k(t) — all three curves on one plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t_ode/60, Fk_ode*100,   "k-",   lw=2.5, label="Full ODE (exact)", zorder=5)
    ax1.plot(t_arr/60, Fk_zero*100,  "r--",  lw=2,   label="Zeroth-order approx (Γ >> Km): LINEAR")
    ax1.plot(t_arr/60, Fk_first*100, "b--",  lw=2,   label="First-order approx (Γ << Km): EXPONENTIAL")
    ax1.axvline(30, color="gray", ls=":", lw=1.2, label="T=30 min")
    ax1.axvline(60, color="gray", ls="-.", lw=1.2, label="T=60 min")
    ax1.set_xlabel("Incubation time (min)")
    ax1.set_ylabel("F_k (%)")
    ax1.set_title(f"A  —  F_k(t) — zeroth vs first order + full ODE  "
                  f"([E] = {E_demo*1e9:.2f} nM)\n"
                  f"Γ_0/Km_surf = {Gamma_ratio:.1f}  →  regime: {regime_label}")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_xlim(0, 60)

    # Panel B: Γ(t) showing crossover point
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t_ode/60, Gamma_ode*1e12, "k-", lw=2)
    ax2.axhline(Km_surf_molcm2*1e12, color="purple", ls="--", lw=1.5,
                label=f"Km_surf = {Km_surf_molcm2*1e12:.4f} pmol/cm²\n(crossover)")
    ax2.axhline(Gamma_0_pmol, color="gray", ls=":", lw=1, label="Γ_0")
    ax2.set_xlabel("Incubation time (min)")
    ax2.set_ylabel("Γ (pmol/cm²)")
    ax2.set_title("B  —  Surface probe density Γ(t)")
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, 60)

    # Panel C: F_k(t) for clinical [E] range
    ax3 = fig.add_subplot(gs[1, :2])
    colors = ["#1565C0", "#0288D1", "#00897B", "#C62828"]
    for E_M, lbl, col in zip(E_clinical, E_labels, colors):
        _, _, Fk_c = solve_Gamma(E_M, t_end_s=t_max)
        ax3.plot(t_ode/60, Fk_c*100, color=col, lw=2, label=lbl)
    ax3.axvline(30, color="gray", ls=":", lw=1.2)
    ax3.axvline(60, color="gray", ls="-.", lw=1.2)
    ax3.axhspan(5, 90, alpha=0.07, color="green", label="Operating window (5–90%)")
    ax3.set_xlabel("Incubation time (min)")
    ax3.set_ylabel("F_k (%)")
    ax3.set_title("C  —  F_k(t) across clinical MMP-9 concentration range")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 60)

    # Panel D: F_k vs [E] — linear calibration check
    ax4 = fig.add_subplot(gs[1, 2])
    E_range = np.linspace(0.05e-9, 3e-9, 200)   # 0.05–3 nM
    for T_min, col, ls in [(15,"b","-"), (30,"g","-"), (60,"r","-")]:
        Fk_fo = Fk_first_order(E_range, T_min * 60)
        ax4.plot(E_range*1e9, Fk_fo*100, color=col, lw=2, ls=ls,
                 label=f"T={T_min} min (exp)")
        # Linear approximation: valid when k_obs × T << 1
        Fk_lin = kcat_Km_eff * E_range * T_min * 60 * 100
        ax4.plot(E_range*1e9, np.minimum(Fk_lin, 100), color=col, lw=1.2, ls="--",
                 label=f"T={T_min} min (linear approx)")
    ax4.axhspan(5, 90, alpha=0.07, color="green")
    ax4.set_xlabel("[E] (nM)")
    ax4.set_ylabel("F_k (%)")
    ax4.set_title("D  —  First-order calibration: exp vs linear approx")
    ax4.legend(fontsize=6.5)
    ax4.set_xlim(0, 3)

    fig.suptitle("Layer 9: Surface kinetics — zeroth vs first order + crossover\n"
                 f"C15 substrate | kcat/Km_eff={kcat_Km_eff:.0f} M⁻¹s⁻¹ | "
                 f"α={alpha_base}, f_steric={f_steric}", fontsize=10)

    if save:
        out = FIGURES / "fig_B8_surface_kinetics.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_surface_kinetics()
    # Key readouts at T=30 and 60 min
    for E_M, lbl in zip(E_clinical, E_labels):
        _, _, Fk = solve_Gamma(E_M, t_end_s=3600)
        idx30 = 500; idx60 = 999
        print(f"[E]={lbl}: F_k(30min)={Fk[idx30]*100:.1f}%  F_k(60min)={Fk[idx60]*100:.1f}%")
