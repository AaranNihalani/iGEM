"""
damkohler.py  —  Layer 3, Section 3.5
========================================
Damköhler-number analysis: determines whether the assay operates in the
reaction-limited regime (signal reports kcat/Km_surf, the desired outcome)
or the diffusion-limited regime (signal reports D_E instead), and how that
balance depends on the two design/operating variables Gamma_0 (probe
coverage) and [E]_0 (bulk enzyme concentration).

Two related but distinct quantities are computed:

  Da_0 (analytic, "instantaneous")
      Da_0 = (kcat/Km_surf) * Gamma_0 * h / D_E
      Evaluated at t=0 from Gamma_0 alone. In the strict linear
      (pseudo-first-order, Gamma << Km_surf) surface-kinetics regime this
      is independent of [E]_0: both the maximum surface reaction flux and
      the diffusive supply flux are linear in [E], so [E]_0 cancels from
      their ratio. This makes Da_0 a fast, closed-form screening tool for
      probe-coverage design, but it cannot by itself capture how much
      *cumulative* signal is lost to transport over a finite incubation.

  Da_apparent(Gamma_0, [E]_0) (numerical, "time-integrated")
      Obtained from the full coupled PDE-ODE solve (diffusion_pde.py) as
      the fractional shortfall between the actual (diffusion-limited)
      cleavage achieved over the incubation and the cleavage that would be
      achieved if the chamber were perfectly mixed (Da -> 0) at the same
      Gamma_0 and [E]_0:
          Da_apparent = -ln(1 - f_actual) / (-ln(1 - f_ideal)) - 1
      This *does* depend on [E]_0, because Gamma(t) and [E](0,t) are
      coupled multiplicatively (kappa(t) = (kcat/Km_surf) Gamma(t)): raising
      [E]_0 accelerates probe depletion, which lowers kappa(t) over the
      course of the incubation and changes the time-integrated balance
      between reaction and diffusion, even though the initial balance
      (Da_0) is unchanged. Da_apparent -> 0 recovers reaction-limited
      behaviour; Da_apparent >> 0 flags a diffusion-limited operating point.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from diffusion_pde import (D_MMP9_25C, M_to_SI, M1s1_to_SI,
                            solve_surface_depletion, damkohler_initial)

FIGURES = Path(__file__).parents[2] / "reports" / "figures"


def apparent_damkohler(D_E, h, E_bulk0, Gamma0, kcat_over_Km, t_end,
                        Nz=40, Nt=600):
    """
    Runs the coupled PDE-ODE solver at one (Gamma0, E_bulk0) operating
    point and returns (Da_0, Da_apparent, f_actual, f_ideal).
    """
    t, z, E, Gamma = solve_surface_depletion(
        D_E, h, E_bulk0, Gamma0, kcat_over_Km, t_end, Nz=Nz, Nt=Nt)

    f_actual = 1.0 - Gamma[-1] / Gamma0
    k_obs_wellmixed = kcat_over_Km * E_bulk0
    f_ideal = 1.0 - np.exp(-k_obs_wellmixed * t_end)

    Da0 = damkohler_initial(kcat_over_Km, Gamma0, h, D_E)

    # Guard against the (numerically well-behaved but log-singular) edges
    # of the domain where f_actual or f_ideal -> 0.
    eps = 1e-12
    rate_actual = -np.log(max(1 - f_actual, eps))
    rate_ideal = -np.log(max(1 - f_ideal, eps))
    Da_apparent = (rate_ideal / max(rate_actual, eps)) - 1.0

    return Da0, Da_apparent, f_actual, f_ideal


def sweep_grid(D_E=D_MMP9_25C, h=375e-6, kcat_over_Km=M1s1_to_SI(188e3 / 10.0),
                t_end=1800.0,
                Gamma0_range_pmol_cm2=(0.5, 15.0),
                E0_range_nM=(0.1, 10.0),
                n_gamma=25, n_E=25):
    """
    Sweeps Gamma_0 in [Gamma0_range] pmol/cm^2 and [E]_0 in [E0_range] nM,
    returning the grids and both Da_0 and Da_apparent surfaces.
    """
    Gamma0_pmol = np.linspace(*Gamma0_range_pmol_cm2, n_gamma)
    E0_nM = np.geomspace(*E0_range_nM, n_E)

    Gamma0_SI = Gamma0_pmol * 1e-12 * 1e4      # pmol/cm^2 -> mol/m^2
    E0_SI = M_to_SI(E0_nM * 1e-9)              # nM -> mol/m^3

    Da0_grid = np.zeros((n_E, n_gamma))
    Da_app_grid = np.zeros((n_E, n_gamma))

    for i, E0 in enumerate(E0_SI):
        for j, G0 in enumerate(Gamma0_SI):
            Da0, Da_app, _, _ = apparent_damkohler(
                D_E, h, E0, G0, kcat_over_Km, t_end, Nz=30, Nt=300)
            Da0_grid[i, j] = Da0
            Da_app_grid[i, j] = Da_app

    return Gamma0_pmol, E0_nM, Da0_grid, Da_app_grid


def plot_damkohler(save=True):
    Gamma0_pmol, E0_nM, Da0_grid, Da_app_grid = sweep_grid()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.pcolormesh(Gamma0_pmol, E0_nM, Da0_grid, shading="auto",
                        cmap="viridis")
    cs = ax.contour(Gamma0_pmol, E0_nM, Da0_grid, levels=[0.1, 1, 10],
                     colors="white", linewidths=1.2)
    ax.clabel(cs, fmt="Da$_0$=%.1f", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Gamma_0$ (pmol/cm$^2$)")
    ax.set_ylabel(r"$[E]_0$ (nM)")
    ax.set_title(r"A — Analytic Da$_0$ (initial, $[E]_0$-independent)")
    fig.colorbar(im, ax=ax, label=r"Da$_0$")

    ax2 = axes[1]
    im2 = ax2.pcolormesh(Gamma0_pmol, E0_nM, Da_app_grid, shading="auto",
                          cmap="magma")
    cs2 = ax2.contour(Gamma0_pmol, E0_nM, Da_app_grid, levels=[0.1, 1, 10],
                       colors="white", linewidths=1.2)
    ax2.clabel(cs2, fmt="Da$_{app}$=%.1f", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\Gamma_0$ (pmol/cm$^2$)")
    ax2.set_ylabel(r"$[E]_0$ (nM)")
    ax2.set_title(r"B — Time-integrated Da$_{apparent}$ (30 min incubation)")
    fig.colorbar(im2, ax=ax2, label=r"Da$_{apparent}$")

    fig.suptitle("Layer 3: Damköhler-number design map (MMP-9 channel)\n"
                  r"$\kappa_{surf}$ = C15-analogue $k_{cat}/K_{m,surf}$"
                  " (188,000 M$^{-1}$s$^{-1}$ / 10 steric correction)",
                  fontsize=10)
    plt.tight_layout()

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        out = FIGURES / "fig_B_damkohler_map.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    return Gamma0_pmol, E0_nM, Da0_grid, Da_app_grid


if __name__ == "__main__":
    # Single-point screening estimate at the design target (Gamma0 = 5
    # pmol/cm^2, upper-clinical [E]_0 = 2.17 nM).
    D_E = D_MMP9_25C
    h = 375e-6
    kcat_over_Km = M1s1_to_SI(188e3 / 10.0)
    Gamma0 = 5e-12 * 1e4
    E0 = M_to_SI(2.17e-9)
    Da0, Da_app, f_actual, f_ideal = apparent_damkohler(
        D_E, h, E0, Gamma0, kcat_over_Km, 1800.0)
    print(f"Design point: Gamma0 = 5 pmol/cm^2, [E]_0 = 2.17 nM")
    print(f"  Da_0        = {Da0:.2f}")
    print(f"  Da_apparent = {Da_app:.2f}")
    print(f"  Well-mixed ideal cleavage fraction  = {f_ideal*100:.2f}%")
    print(f"  Diffusion-limited actual cleavage   = {f_actual*100:.2f}%")
    print(f"  Signal recovered vs. ideal           = {f_actual/f_ideal*100:.1f}%")
    print()
    print("Generating (Gamma0, [E]_0) Damköhler design map "
          "(this involves ~1250 coupled PDE-ODE solves; a few seconds)...")
    plot_damkohler()
