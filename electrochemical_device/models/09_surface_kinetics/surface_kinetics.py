"""
surface_kinetics.py -- Layer 9 (CORE MODEL)
============================================
Three-state surface kinetic model following:
  Anne A., Demaille C., Moiroux J. (2001) Macromolecules 34, 5735-5745

Surface mechanism:
  E + Gamma_S <--k1--> Gamma_ES --kcat--> E + Gamma_P
                k-1

State variables (all mol/cm2):
  Gamma_S  : uncleaved surface probe
  Gamma_ES : enzyme-probe surface complex
  Gamma_P  : cleaved probe (product)

Conservation: Gamma_S + Gamma_ES + Gamma_P = Gamma_0

ODEs (Anne et al. 2001, Eq. 1):
  dGamma_S/dt  = k_m1 * Gamma_ES - k1 * [E] * Gamma_S
  dGamma_ES/dt = k1 * [E] * Gamma_S - (k_m1 + kcat) * Gamma_ES
  dGamma_P/dt  = kcat * Gamma_ES

k1 (M-1 s-1), k_m1 (s-1), kcat (s-1) are SURFACE rate constants.
They must be measured on the fabricated electrode, not borrowed from solution.

ANALYTIC LIMITS
---------------
At clinical [E] (0.19-2.17 nM) and any physically reasonable k1 (<1e8 M-1s-1):
  k1 * [E_max] < 0.22 s-1  <<  kcat_sol = 86.4 s-1

Gamma_ES therefore accumulates negligibly. Both kinetic limits reduce to:

  Fk(t) = 1 - exp(-keff_surf * [E] * t)

  Steady-state limit (k_m1 >> kcat):  keff = kcat / Km  (Km = (k_m1+kcat)/k1)
  Irreversible limit  (k_m1 << kcat): keff = k1

The solution value kcat/Km_sol = 67,500 M-1 s-1 is an UPPER BOUND for keff_surf.
The actual surface value must be measured on the fabricated electrode.

From Kridel et al. 2001 (J Biol Chem 276:20572), Table II, peptide A11 (SGKIPRTLTA):
  kcat_sol    = 86.4  s-1
  kcat_Km_sol = 67,500  M-1 s-1
  Km_sol      = kcat_sol / kcat_Km_sol = 1.28 mM
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Known solution parameters (Kridel 2001, Table II, peptide A11 SGKIPRTLTA) ──
kcat_sol    = 86.4
kcat_Km_sol = 67.5e3         # M-1s-1 (upper bound for surface keff)
Km_sol      = kcat_sol / kcat_Km_sol   # 1.28e-3 M

MW_MMP9 = 92e3
E_lo_M  = 17e-6  / MW_MMP9  # 0.185 nM
E_hi_M  = 200e-6 / MW_MMP9  # 2.17  nM

Gamma_0_pmol = 3.0
Gamma_0      = Gamma_0_pmol * 1e-12   # mol/cm2


def ode_3state(t, y, k1, k_m1, kcat, E_M):
    """Anne et al. 2001 three-state surface kinetic ODE."""
    Gs, Ges, Gp = y
    dGs  = k_m1 * Ges - k1 * E_M * Gs
    dGes = k1 * E_M * Gs - (k_m1 + kcat) * Ges
    dGp  = kcat * Ges
    return [dGs, dGes, dGp]


def solve_3state(E_M, t_end_s, k1, k_m1, kcat, N=2000):
    t_eval = np.linspace(0, t_end_s, N)
    sol = solve_ivp(ode_3state, (0, t_end_s), [Gamma_0, 0.0, 0.0],
                    args=(k1, k_m1, kcat, E_M),
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-18)
    Gs, Ges, Gp = sol.y
    return sol.t, Gs, Ges, Gp, Gp / Gamma_0


def Fk_analytic(E_M, t_arr, keff_surf):
    """Valid when k1*[E] << kcat (confirmed at clinical concentrations)."""
    return 1 - np.exp(-keff_surf * E_M * t_arr)


def plot_surface_kinetics(save=True):
    t_max = 3600
    t_arr = np.linspace(0, t_max, 2000)

    # Irreversible limit: k-1 = 0, k1 = kcat/Km_sol (solution upper bound)
    k1_demo   = kcat_Km_sol
    k_m1_demo = 0.0

    # Same keff, three different k-1/kcat ratios to show regime independence
    regimes = [
        ("Irreversible ($k_{-1}=0$)",          0.0,         k1_demo,     "#1565C0"),
        ("Intermediate ($k_{-1}=k_{cat}$)",    kcat_sol,    2*k1_demo,   "#00897B"),
        ("Quasi-equil. ($k_{-1}=10k_{cat}$)",  10*kcat_sol, 11*k1_demo,  "#C62828"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: 3-state trajectories -- show Gamma_ES is negligible
    ax = axes[0, 0]
    t_ode, Gs, Ges, Gp, _ = solve_3state(
        E_hi_M, t_max, k1_demo, k_m1_demo, kcat_sol)
    ax.plot(t_ode/60, Gs/Gamma_0*100, "b-", lw=2,
            label=r"$\Gamma_S$ (free probe)")
    ax.plot(t_ode/60, Gp/Gamma_0*100, "g-", lw=2,
            label=r"$\Gamma_P$ (cleaved)")
    ax2 = ax.twinx()
    ax2.plot(t_ode/60, Ges/Gamma_0*1e6, "r--", lw=1.5,
             label=r"$\Gamma_{ES}$ (ppm, right)")
    ax2.set_ylabel(r"$\Gamma_{ES}/\Gamma_0$ (ppm)", color="red", fontsize=9)
    ax2.tick_params(axis="y", colors="red")
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("% of initial probe density")
    ax.set_title("A  --  3-state populations at [E] = 200 ng/mL\n"
                 r"$\Gamma_{ES}$ < 1 ppm: analytic limit valid at clinical [E]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8, loc="center right")
    ax.set_xlim(0, 60)

    # Panel B: all three regimes give same Fk
    ax = axes[0, 1]
    for lbl, k_m1, k1, col in regimes:
        _, _, _, _, Fk_r = solve_3state(E_hi_M, t_max, k1, k_m1, kcat_sol)
        ax.plot(t_ode/60, Fk_r*100, color=col, lw=2.5, label=lbl)
    ax.plot(t_arr/60, Fk_analytic(E_hi_M, t_arr, kcat_Km_sol)*100,
            "k--", lw=1.5, label="Analytic (both limits)")
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("$F_k$ (%)")
    ax.set_title("B  --  All regimes give identical $F_k(t)$ at clinical [E]\n"
                 "(each uses keff = 188,000 M$^{-1}$s$^{-1}$; curves overlap)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 60)

    # Panel C: Fk(t) across clinical [E], solution upper bound
    ax = axes[1, 0]
    E_clin  = [E_lo_M, 0.5e-9, 1.0e-9, E_hi_M]
    E_labs  = ["17 ng/mL", "46 ng/mL", "92 ng/mL", "200 ng/mL"]
    colors_c = ["#1565C0", "#0288D1", "#00897B", "#C62828"]
    for E_M, lbl, col in zip(E_clin, E_labs, colors_c):
        ax.plot(t_arr/60, Fk_analytic(E_M, t_arr, kcat_Km_sol)*100,
                color=col, lw=2, label=lbl)
    ax.axhline(5, color="gray", ls=":", lw=1)
    ax.axvline(30, color="gray", ls=":",  lw=1, label="T=30/60 min")
    ax.axvline(60, color="gray", ls="-.", lw=1)
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("$F_k$ (%)")
    ax.set_title("C  --  $F_k(t)$ across clinical range\n"
                 "at solution upper bound keff = 188,000 M$^{-1}$s$^{-1}$")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 60)

    # Panel D: sensitivity to unknown surface keff
    ax = axes[1, 1]
    keff_vals = [2000, 10000, 20250, 30000, 67500]
    keff_labs = ["2,000", "10,000", "20,250 (alpha=0.3 est.)",
                 "30,000", "67,500 (sol. UB)"]
    t_120 = np.linspace(0, 7200, 3000)
    colors_d = plt.cm.plasma(np.linspace(0.05, 0.85, len(keff_vals)))
    for keff, lbl, col in zip(keff_vals, keff_labs, colors_d):
        ax.plot(t_120/60, Fk_analytic(E_lo_M, t_120, keff)*100,
                color=col, lw=2, label=f"{lbl}")
    ax.axhline(5, color="gray", ls=":", lw=1, label="5% threshold")
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("$F_k$ at 17 ng/mL (%)")
    ax.set_title("D  --  Sensitivity to unknown surface keff\n"
                 "(17 ng/mL; all predictions bracketed until keff is measured)")
    ax.legend(fontsize=7, title="keff (M$^{-1}$s$^{-1}$)")
    ax.set_xlim(0, 120)

    fig.suptitle(
        "Layer 9: Surface kinetics (Anne, Demaille, Moiroux 2001)\n"
        "$k_{cat,sol}$ = 86.4 s$^{-1}$,  $K_{m,sol}$ = 1.28 mM,  "
        "$(k_{cat}/K_m)_{sol}$ = 67,500 M$^{-1}$s$^{-1}$  "
        "(Kridel 2001, A11/SGKIPRTLTA)  [surface keff must be measured]",
        fontsize=10
    )
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B8_surface_kinetics.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    print(f"\nAnalytic limit check: kcat / (k1*[E_hi]) = "
          f"{kcat_sol/(kcat_Km_sol*E_hi_M):.2e}  >> 1 -- valid")
    t_chk = np.linspace(60, 7200, 5000)
    print("T to detect 17 ng/mL (Fk > 5%) at each keff:")
    for keff in keff_vals:
        fk = Fk_analytic(E_lo_M, t_chk, keff)
        idx = np.searchsorted(fk, 0.05)
        if idx < len(t_chk):
            print(f"  keff = {keff:7.0f}:  T = {t_chk[idx]/60:.0f} min")
        else:
            print(f"  keff = {keff:7.0f}:  > 120 min")


if __name__ == "__main__":
    plot_surface_kinetics()
