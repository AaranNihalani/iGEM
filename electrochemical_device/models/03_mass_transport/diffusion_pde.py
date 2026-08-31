"""
diffusion_pde.py  —  Layer 3
==============================
Coupled 1-D diffusion-reaction solver for enzyme transport from the bulk
incubation chamber to the electrode surface, linked to the Layer 9 surface
depletion ODE, plus the extended three-species reaction-diffusion system
that tracks in-device inhibitor binding (contents.md, Section 3.9 — "the
most important addition to this layer identified in lifecycle analysis").

Unit convention (SI throughout; see Section II of the Layer 3 report for the
full derivation and the reasoning behind it):

    length          m
    time            s
    [E], [I]        mol/m^3   (volumetric concentration; 1 mM = 1 mol/m^3)
    Gamma           mol/m^2   (areal probe/substrate density)
    D               m^2/s
    kcat            1/s
    kcat/Km_surf    m^3/(mol s)   i.e. a standard M^-1 s^-1 specificity
                                  constant expressed in SI
    k_on            m^3/(mol s)   i.e. M^-1 s^-1 x 1e-3

Governing system (validated linear / pseudo-first-order regime, Gamma <<
Km_surf — the operating regime the rest of the pipeline, Layers 9/13/15,
assumes and is designed around):

    d[E]/dt = D_E d^2[E]/dz^2,                 0 < z < h
    d[E]/dz|_{z=h} = 0                          (no-flux at the PDMS lid)
    D_E d[E]/dz|_{z=0} = kappa(t) [E](0,t)      (Robin BC at the electrode)
    kappa(t) = (kcat/Km_surf) * Gamma(t)        (m/s — an areal reaction
                                                  velocity, see report Sec. II)
    dGamma/dt = -kappa(t) [E](0,t)              (Layer 9 surface ODE)

kappa(t) and [E](0,t) are coupled through Gamma(t); the two are advanced
with the lagged (semi-implicit) scheme specified in contents.md Section 3.4:
at each step, kappa is evaluated from the Gamma already known at that step,
the Crank-Nicolson diffusion half-step is solved treating that kappa as a
fixed linear Robin coefficient, and Gamma is then advanced using the freshly
solved [E](0, t_{n+1}).

Section 3.9 extends this to three coupled species (free enzyme, free
inhibitor, and bound complex), each diffusing with its own D, with a bulk
second-order binding reaction -k_on [E][I] superimposed on the diffusion.
Only the free-enzyme field is subject to the Robin sink at the electrode;
the inhibitor and complex do not react at the surface and satisfy no-flux
conditions at both boundaries.
"""

import numpy as np
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Physical constants ───────────────────────────────────────────────────────
kB = 1.380649e-23  # J/K


def water_viscosity(T_K):
    """
    Dynamic viscosity of water (Pa s) via the Vogel-Fulcher-Tammann fit,
    valid 0-100 degC (CRC Handbook of Chemistry and Physics). Cross-checked
    against tabulated values: eta(20C)=1.002, eta(25C)=0.890, eta(37C)=0.690
    mPa s -- all reproduced to <0.2% by this formula.
    """
    return 2.414e-5 * 10 ** (247.8 / (T_K - 140.0))


def r_h_from_MW(MW_Da):
    """Hydrodynamic radius (m) from molecular weight via the empirical
    globular-protein scaling law r_h[nm] = 0.066 * MW[Da]^(1/3)
    (Tyn & Gusek 1990)."""
    return 0.066e-9 * MW_Da ** (1.0 / 3.0)


def stokes_einstein_D(r_h_m, T_K=298.15, eta=None):
    """Stokes-Einstein diffusion coefficient (m^2/s)."""
    if eta is None:
        eta = water_viscosity(T_K)
    return kB * T_K / (6 * np.pi * eta * r_h_m)


def scale_D_to_T(D_ref, T_ref_K, T_K):
    """Temperature-correct a diffusion coefficient via Stokes-Einstein,
    D(T) = D(Tref) * (T/Tref) * (eta(Tref)/eta(T))."""
    return D_ref * (T_K / T_ref_K) * (water_viscosity(T_ref_K) / water_viscosity(T_K))


# ── Unit-conversion helpers (literature values are usually quoted in M, M^-1 s^-1)
def M_to_SI(c_M):
    """mol/L -> mol/m^3."""
    return c_M * 1e3


def M1s1_to_SI(k):
    """M^-1 s^-1 -> m^3/(mol s)."""
    return k * 1e-3


# ── Literature / target diffusion coefficients at 25 degC ──────────────────
# MMP-9 and HNE use literature hydrodynamic radii (README Layer 0) rather
# than the pure MW-scaling law: MMP-9 is a multidomain (catalytic +
# hemopexin) protein and is measurably less compact than an equivalent-mass
# globular sphere, so the scaling law (r_h ~ 2.94 nm, giving D ~ 7-8e-11
# m^2/s) underestimates its true hydrodynamic drag. HNE is a compact
# single-domain serine protease for which the scaling law agrees with
# literature to within ~5%, so either estimate is usable.
D_MMP9_25C = 5.0e-11        # m^2/s  (r_h,eff ~ 3.5 nm; README Layer 0)
D_MMP9_DIMER_25C = D_MMP9_25C / 2 ** (1.0 / 3.0)   # ~4.0e-11 m^2/s
D_HNE_25C = 9.2e-11         # m^2/s  (r_h ~ 2.06 nm; scaling law + lit. agree)
D_TIMP1_25C = stokes_einstein_D(r_h_from_MW(21e3))   # TIMP-1 ~21 kDa
D_A1AT_25C = stokes_einstein_D(r_h_from_MW(52e3))    # A1AT ~52 kDa
D_FRAG_25C = 3.2e-10        # m^2/s  (SGIG-PEG2-Lys(MB) fragment, ~1 kDa)


# ── Core Crank-Nicolson solver: single diffusing species, Robin sink at z=0,
#    no-flux at z=h, optional fixed timestep bulk reaction term ─────────────

def _build_cn_matrix(Nz, r, kappa, dz, D):
    """
    Banded (tridiagonal) Crank-Nicolson matrix for interior nodes, with the
    Robin condition at z=0 folded in implicitly (mirror-node elimination)
    and the no-flux condition at z=h folded in via mirror reflection.
    kappa is the (fixed, for this step) Robin surface velocity (m/s); set
    kappa=0 to recover a pure no-flux condition at z=0 as well.
    """
    N = Nz + 1
    ab = np.zeros((3, N))
    ab[0, 1:] = -r                       # upper diagonal
    ab[1, :] = 1 + 2 * r                 # main diagonal
    ab[2, :-1] = -r                      # lower diagonal
    # z = 0: Robin BC, mirror node E[-1] = E[1] - 2*dz*kappa/D * E[0]
    ab[1, 0] = 1 + 2 * r + 2 * r * kappa * dz / D
    ab[0, 1] = -2 * r
    # z = h: no-flux, mirror node E[Nz+1] = E[Nz-1]
    ab[1, -1] = 1 + 2 * r
    ab[2, -2] = -2 * r
    return ab


def solve_surface_depletion(D_E, h, E_bulk0, Gamma0, kcat_over_Km,
                             t_end, Nz=50, Nt=2000):
    """
    Layer 3 core solver: single-species diffusion PDE coupled to the Layer 9
    surface ODE via a Robin boundary condition, using the lagged
    Crank-Nicolson scheme described in the module docstring.

    Parameters
    ----------
    D_E : float            enzyme diffusion coefficient, m^2/s
    h : float               channel height, m
    E_bulk0 : float         initial (uniform) bulk enzyme concentration, mol/m^3
    Gamma0 : float          initial probe/substrate areal density, mol/m^2
    kcat_over_Km : float    surface specificity constant, m^3/(mol s)
    t_end : float           incubation time, s
    Nz, Nt : int             grid / timestep resolution

    Returns
    -------
    t : (Nt+1,) array        time grid, s
    z : (Nz+1,) array        spatial grid, m
    E : (Nt+1, Nz+1) array   [E](z,t), mol/m^3
    Gamma : (Nt+1,) array    Gamma(t), mol/m^2
    """
    dz = h / Nz
    dt = t_end / Nt
    r = D_E * dt / (2 * dz ** 2)

    z = np.linspace(0, h, Nz + 1)
    E = np.full(Nz + 1, E_bulk0)
    Gamma = Gamma0

    t_arr = np.linspace(0, t_end, Nt + 1)
    E_store = np.zeros((Nt + 1, Nz + 1))
    Gamma_store = np.zeros(Nt + 1)
    E_store[0] = E
    Gamma_store[0] = Gamma

    for n in range(Nt):
        kappa = kcat_over_Km * Gamma      # frozen (lagged) surface velocity, m/s
        ab = _build_cn_matrix(Nz, r, kappa, dz, D_E)

        rhs = np.empty(Nz + 1)
        # interior explicit half
        rhs[1:-1] = r * E[:-2] + (1 - 2 * r) * E[1:-1] + r * E[2:]
        # z = 0 explicit half (mirror node E[-1] = E[1] - 2*dz*kappa/D*E[0])
        rhs[0] = (r * (E[1] - 2 * dz * kappa / D_E * E[0])
                  + (1 - 2 * r) * E[0] + r * E[1])
        # z = h explicit half (mirror node E[Nz+1] = E[Nz-1])
        rhs[-1] = 2 * r * E[-2] + (1 - 2 * r) * E[-1]

        E = solve_banded((1, 1), ab, rhs)
        E = np.clip(E, 0.0, None)

        J = kappa * E[0]                  # mol/(m^2 s), post-step surface flux
        Gamma = max(Gamma - J * dt, 0.0)

        E_store[n + 1] = E
        Gamma_store[n + 1] = Gamma

    return t_arr, z, E_store, Gamma_store


def damkohler_initial(kcat_over_Km, Gamma0, h, D_E):
    """
    Closed-form (linear-regime) Damköhler number evaluated at t=0:
        Da_0 = kappa(0) / (D_E/h) = (kcat/Km_surf) * Gamma0 * h / D_E
    This is the ratio of the surface reaction velocity to the diffusive
    supply velocity across the channel height, and is independent of the
    bulk enzyme concentration in the linear-kinetics regime (see report
    Section IV): both the surface flux and the diffusive flux are linear in
    [E], so [E]_0 cancels from the ratio. It should be treated as a
    screening estimate; damkohler.py computes the time-integrated apparent
    Da from the full numerical solution, which does depend on [E]_0 through
    the bilinear Gamma-E coupling.
    """
    return kcat_over_Km * Gamma0 * h / D_E


# ── Three-species reaction-diffusion extension (Section 3.9) ───────────────

def solve_inhibitor_diffusion_reaction(D_E, D_I, D_C, h,
                                        E_bulk0, I_bulk0,
                                        Gamma0, kcat_over_Km,
                                        k_on, k_off,
                                        t_end, Nz=50, Nt=4000):
    """
    Extended Layer 3 solver: tracks free enzyme, free inhibitor, and
    enzyme-inhibitor complex as three coupled diffusing species, with a
    bulk second-order binding reaction superimposed on diffusion (Strang-
    style operator splitting: a reaction sub-step followed by an
    independent Crank-Nicolson diffusion sub-step for each species, each of
    length dt/2, applied symmetrically). Only [E_free] is consumed at the
    electrode surface (Robin BC as in solve_surface_depletion); [I_free]
    and [complex] satisfy no-flux at both boundaries, since only the free
    enzyme reacts with the immobilised probe.

    This is "the critical correction to the naive single-species diffusion
    model" (contents.md 3.9): TIMP-1 and A1AT continue to deplete free
    enzyme throughout the incubation, independently of surface cleavage,
    and the device's ratiometric signal Fk reflects only the FREE enzyme
    that survives this competition -- which is the clinically meaningful
    quantity.

    Returns
    -------
    t : (Nt+1,) array
    z : (Nz+1,) array
    E : (Nt+1, Nz+1) array    free enzyme, mol/m^3
    I : (Nt+1, Nz+1) array    free inhibitor, mol/m^3
    C : (Nt+1, Nz+1) array    complex, mol/m^3
    Gamma : (Nt+1,) array     surface probe density, mol/m^2
    """
    dz = h / Nz
    dt = t_end / Nt
    z = np.linspace(0, h, Nz + 1)

    E = np.full(Nz + 1, E_bulk0)
    I = np.full(Nz + 1, I_bulk0)
    C = np.zeros(Nz + 1)
    Gamma = Gamma0

    t_arr = np.linspace(0, t_end, Nt + 1)
    E_store = np.zeros((Nt + 1, Nz + 1))
    I_store = np.zeros((Nt + 1, Nz + 1))
    C_store = np.zeros((Nt + 1, Nz + 1))
    Gamma_store = np.zeros(Nt + 1)
    E_store[0], I_store[0], C_store[0], Gamma_store[0] = E, I, C, Gamma

    def react(E, I, C, sub_dt):
        """Explicit bulk-reaction half-step (pointwise in z)."""
        v_on = k_on * E * I
        v_off = k_off * C
        dE = -v_on + v_off
        dI = -v_on + v_off
        dC = v_on - v_off
        return (np.clip(E + dE * sub_dt, 0, None),
                np.clip(I + dI * sub_dt, 0, None),
                np.clip(C + dC * sub_dt, 0, None))

    def diffuse_noflux(field, D, sub_dt):
        """No-flux-at-both-ends CN half-step, no surface reaction."""
        r = D * sub_dt / (2 * dz ** 2)
        ab = _build_cn_matrix(Nz, r, 0.0, dz, D)   # kappa=0 -> no-flux at z=0
        rhs = np.empty(Nz + 1)
        rhs[1:-1] = r * field[:-2] + (1 - 2 * r) * field[1:-1] + r * field[2:]
        rhs[0] = r * field[1] + (1 - 2 * r) * field[0] + r * field[1]
        rhs[-1] = 2 * r * field[-2] + (1 - 2 * r) * field[-1]
        return np.clip(solve_banded((1, 1), ab, rhs), 0, None)

    def diffuse_E_with_sink(E, kappa, sub_dt):
        r = D_E * sub_dt / (2 * dz ** 2)
        ab = _build_cn_matrix(Nz, r, kappa, dz, D_E)
        rhs = np.empty(Nz + 1)
        rhs[1:-1] = r * E[:-2] + (1 - 2 * r) * E[1:-1] + r * E[2:]
        rhs[0] = (r * (E[1] - 2 * dz * kappa / D_E * E[0])
                  + (1 - 2 * r) * E[0] + r * E[1])
        rhs[-1] = 2 * r * E[-2] + (1 - 2 * r) * E[-1]
        return np.clip(solve_banded((1, 1), ab, rhs), 0, None)

    for n in range(Nt):
        # half reaction step
        E, I, C = react(E, I, C, dt / 2)
        # full diffusion step (E has the electrode sink; I, C are no-flux)
        kappa = kcat_over_Km * Gamma
        E = diffuse_E_with_sink(E, kappa, dt)
        I = diffuse_noflux(I, D_I, dt)
        C = diffuse_noflux(C, D_C, dt)
        # half reaction step
        E, I, C = react(E, I, C, dt / 2)

        J = kappa * E[0]
        Gamma = max(Gamma - J * dt, 0.0)

        E_store[n + 1], I_store[n + 1] = E, I
        C_store[n + 1], Gamma_store[n + 1] = C, Gamma

    return t_arr, z, E_store, I_store, C_store, Gamma_store


# ── Demonstration / figure generation ───────────────────────────────────────

def _demo_parameters():
    """
    Representative upper-clinical operating point (MMP-9 / TIMP-1 channel),
    chosen to give an illustrative, non-trivial Damköhler number. kcat/Km_surf
    uses the C15-substrate literature specificity constant (188,000 M^-1 s^-1;
    fouling_model.py) with a x10 steric correction (Layer 9) as the working
    estimate pending wet-lab surface kinetics (pES-01..08).
    """
    h = 375e-6                    # m, midpoint channel height
    Gamma0 = 5e-12 * 1e4          # mol/m^2  (5 pmol/cm^2 -> 5e-8 mol/m^2)
    E_bulk0 = M_to_SI(2.17e-9)    # mol/m^3  (~2.17 nM, upper clinical MMP-9)
    I_bulk0 = M_to_SI(2.3e-9)     # mol/m^3  (~2.3 nM TIMP-1 in chamber)
    kcat_over_Km = M1s1_to_SI(188e3 / 10.0)
    k_on = M1s1_to_SI(2e5)        # TIMP-1 on-rate
    k_off = 2e-5                  # TIMP-1 off-rate, 1/s
    t_end = 1800.0                # 30 min incubation
    return dict(D_E=D_MMP9_25C, D_I=D_TIMP1_25C, D_C=D_MMP9_25C, h=h,
                E_bulk0=E_bulk0, I_bulk0=I_bulk0, Gamma0=Gamma0,
                kcat_over_Km=kcat_over_Km, k_on=k_on, k_off=k_off, t_end=t_end)


def plot_comparison(save=True):
    """Compare the naive single-species depletion model against the
    inhibitor-corrected three-species model for the MMP-9/TIMP-1 channel."""
    p = _demo_parameters()

    t1, z1, E1, G1 = solve_surface_depletion(
        p["D_E"], p["h"], p["E_bulk0"], p["Gamma0"], p["kcat_over_Km"],
        p["t_end"], Nz=50, Nt=1800)

    t2, z2, E2, I2, C2, G2 = solve_inhibitor_diffusion_reaction(
        p["D_E"], p["D_I"], p["D_C"], p["h"], p["E_bulk0"], p["I_bulk0"],
        p["Gamma0"], p["kcat_over_Km"], p["k_on"], p["k_off"],
        p["t_end"], Nz=50, Nt=1800)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    ax.plot(t1 / 60, E1[:, 0] / p["E_bulk0"] * 100, "b-", lw=2,
            label="Naive (diffusion only)")
    ax.plot(t2 / 60, E2[:, 0] / p["E_bulk0"] * 100, "r-", lw=2,
            label="Corrected (+ TIMP-1 binding)")
    ax.axhline(100, color="gray", ls="--", lw=1)
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("[MMP-9]$_{free}$ at electrode / bulk (%)")
    ax.set_title("A — Surface enzyme availability")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    ax2 = axes[1]
    ax2.plot(t1 / 60, G1 / p["Gamma0"] * 100, "b-", lw=2,
             label="Naive (diffusion only)")
    ax2.plot(t2 / 60, G2 / p["Gamma0"] * 100, "r-", lw=2,
             label="Corrected (+ TIMP-1 binding)")
    ax2.set_xlabel("Incubation time (min)")
    ax2.set_ylabel("Intact probe $\\Gamma / \\Gamma_0$ (%)")
    ax2.set_title("B — Probe cleavage (naive vs. corrected)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)

    fig.suptitle("Layer 3: Coupled diffusion-reaction model\n"
                  "Naive single-species vs. inhibitor-corrected three-species system",
                  fontsize=10)
    plt.tight_layout()

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        out = FIGURES / "fig_B_diffusion_pde.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    Da0 = damkohler_initial(p["kcat_over_Km"], p["Gamma0"], p["h"], p["D_E"])
    k_obs_wellmixed = p["kcat_over_Km"] * p["E_bulk0"]
    ideal_cleavage = 1 - np.exp(-k_obs_wellmixed * p["t_end"])
    actual_cleavage_naive = 1 - G1[-1] / p["Gamma0"]
    print(f"Da_0 (initial, linear-regime estimate) = {Da0:.3f}")
    print(f"Well-mixed (Da->0) ideal cleavage fraction at T = {ideal_cleavage*100:.1f}%")
    print(f"Naive diffusion-limited model: actual cleavage fraction at T = "
          f"{actual_cleavage_naive*100:.1f}%  (diffusion penalty visible when Da_0 is not << 1)")
    print(f"Corrected model: Gamma(T)/Gamma0 = {G2[-1] / p['Gamma0'] * 100:.1f}%")
    print(f"Corrected model: [E]_free(T)/[E]_0 (bulk) "
          f"= {E2[-1, -1] / p['E_bulk0'] * 100:.1f}% (should stay near bulk far from electrode)")


if __name__ == "__main__":
    print(f"D_MMP9        = {D_MMP9_25C:.2e} m^2/s")
    print(f"D_MMP9(dimer) = {D_MMP9_DIMER_25C:.2e} m^2/s")
    print(f"D_HNE         = {D_HNE_25C:.2e} m^2/s")
    print(f"D_TIMP1       = {D_TIMP1_25C:.2e} m^2/s")
    print(f"D_A1AT        = {D_A1AT_25C:.2e} m^2/s")
    print(f"D_frag        = {D_FRAG_25C:.2e} m^2/s")
    plot_comparison()
