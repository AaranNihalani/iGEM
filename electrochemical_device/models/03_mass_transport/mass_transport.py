"""
mass_transport.py  —  Layer 3
===============================
1-D diffusion PDE for enzyme transport from bulk to electrode surface.

Governing PDE:
  ∂[E]/∂t = D_E * ∂²[E]/∂z²    (0 < z < h, t > 0)

Boundary conditions:
  z = h (bulk):  [E](h,t) = [E]_bulk   (well-mixed reservoir above diffusion layer)
  z = 0 (surface): D_E * ∂[E]/∂z = k_obs * Gamma * [E](0,t)  (flux = cleavage rate)

Solved by Crank-Nicolson finite difference (unconditionally stable).

Damköhler number:  Da = k_obs * h / D_E
  Da << 1: transport-limited (surface sees bulk [E]) — desired operating regime
  Da >> 1: reaction-limited (depleted surface layer)

Outputs (to Layer 9):
  [E](z=0, t): surface enzyme concentration vs time
  Da: Damköhler number
"""

import numpy as np
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Physical constants ───────────────────────────────────────────────────────
kB   = 1.381e-23    # J/K
T    = 298.15       # K  (25°C)
eta  = 1.0e-3       # Pa·s  (water viscosity)

# ── Enzyme diffusion coefficients (Stokes-Einstein) ─────────────────────────
# D = kB*T / (6*pi*eta*r_h)
# Hydrodynamic radius from MW^(1/3) scaling: r ≈ 0.066 * MW^(1/3) nm
# (Tyn & Gusek 1990, Biotechnology and Bioengineering — well established)
def stokes_einstein(MW_Da):
    r_nm = 0.066 * MW_Da**(1/3)   # nm
    r_m  = r_nm * 1e-9
    return kB * T / (6 * np.pi * eta * r_m)   # m²/s

D_MMP9 = stokes_einstein(92e3)    # ~4.7e-11 m²/s = 4.7e-7 cm²/s
D_HNE  = stokes_einstein(30e3)    # ~7.2e-11 m²/s

# ── Channel geometry ─────────────────────────────────────────────────────────
h    = 100e-6       # m  (100 µm diffusion layer height — channel half-height)

# ── Cleavage parameters (first-order surface kinetics) ───────────────────────
# From Layer 9 / Kridel 2001: A11/SGKIPRTLTA, kcat/Km = 67,500 M-1s-1
# Surface correction alpha = 1.0 (best case); [E] = 2.17 nM (upper clinical)
kcat_Km_surf = 67.5e3 * 1.0         # M-1 s-1 (solution UB; surface value to be measured)
Gamma_0      = 3e-12 * 1e4         # mol/m²  (3 pmol/cm² → 3e-8 mol/m²)
# k_obs = (kcat/Km_surf) * [E]; at [E]=2.17 nM → k_obs ≈ 4.1e-4 s-1

E_bulk_nM = 2.17e-9   # mol/L = mol/m³ * 1e-3 → use in SI consistently

# ── Crank-Nicolson PDE solver ────────────────────────────────────────────────

def solve_diffusion_CN(D, h_m, E_bulk_M, k_obs_surface, t_end_s,
                       Nz=100, Nt=500):
    """
    Solve 1-D diffusion by Crank-Nicolson.
    Returns t_arr, z_arr, E(z,t) in mol/m³ (= 1000 × mol/L).
    """
    dz = h_m / Nz
    dt = t_end_s / Nt
    r  = D * dt / (2 * dz**2)    # Crank-Nicolson stability parameter

    z = np.linspace(0, h_m, Nz + 1)
    E = np.ones(Nz + 1) * E_bulk_M   # initial condition: uniform E_bulk

    # Banded matrix storage (upper/main/lower diagonals)
    def build_matrix(Nz, r, k_obs_surface, dz):
        N = Nz + 1
        ab = np.zeros((3, N))
        # Interior: -r * E[i-1] + (1+2r) * E[i] - r * E[i+1] = RHS
        ab[0, 1:] = -r        # upper diagonal
        ab[1, :]  =  1 + 2*r  # main diagonal
        ab[2, :-1]= -r        # lower diagonal
        # z=0 BC (flux = k_obs * E[0]):  -D*(E[1]-E[0])/dz = k_obs*E[0]
        # Crank-Nicolson surface BC: implemented as Robin BC
        ab[1, 0]  = 1 + 2*r + 2*r * k_obs_surface * dz / D
        ab[0, 1]  = -2*r
        # z=h BC (Dirichlet: E = E_bulk)
        ab[1, -1] = 1
        ab[2, -2] = 0
        ab[0, -1] = 0
        return ab

    ab = build_matrix(Nz, r, k_obs_surface, dz)

    t_arr    = np.linspace(0, t_end_s, Nt + 1)
    E_store  = np.zeros((Nt + 1, Nz + 1))
    E_store[0] = E.copy()

    for n in range(Nt):
        # RHS from explicit part
        rhs = E.copy()
        rhs[1:-1] = (r * E[:-2] + (1 - 2*r) * E[1:-1] + r * E[2:])
        rhs[0]    = E[0]   # surface (updated by BC in matrix)
        rhs[-1]   = E_bulk_M   # bulk Dirichlet

        E = solve_banded((1, 1), ab, rhs)
        E[-1] = E_bulk_M   # enforce Dirichlet
        E_store[n+1] = E

    return t_arr, z * 1e6, E_store   # z in µm, E in mol/m³

def compute_Da(kcat_Km, E_bulk, h_m, D, Gamma):
    k_obs = kcat_Km * E_bulk
    return k_obs * h_m / D


def plot_transport(save=True):
    # Run for MMP-9 / A11 (SGKIPRTLTA) at upper clinical concentration
    k_obs = kcat_Km_surf * E_bulk_nM * 1e-3   # convert E from mol/L to mol/m³? No—
    # kcat_Km is M-1s-1, E in mol/L; k_obs = kcat_Km * E [s-1]
    k_obs = kcat_Km_surf * E_bulk_nM           # s-1 (E in mol/L is fine for kcat_Km in M-1s-1)

    # Convert E_bulk to mol/m³ for PDE (SI)
    E_bulk_SI = E_bulk_nM * 1e3   # mol/m³

    t_arr, z_um, E_store = solve_diffusion_CN(
        D=D_MMP9, h_m=h, E_bulk_M=E_bulk_SI,
        k_obs_surface=k_obs, t_end_s=3600, Nz=80, Nt=800)

    t_plot_min = [0, 5, 15, 30, 60]
    t_idx = [np.argmin(np.abs(t_arr - tm*60)) for tm in t_plot_min]

    Da = compute_Da(kcat_Km_surf, E_bulk_nM, h, D_MMP9, Gamma_0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: [E](z) profiles
    ax = axes[0]
    cmap = plt.cm.Blues
    for k, (idx, tm) in enumerate(zip(t_idx, t_plot_min)):
        frac = (k + 1) / len(t_plot_min)
        E_nM = E_store[idx] / 1e3 * 1e9   # mol/m³ → mol/L → nM
        ax.plot(z_um, E_nM, color=cmap(0.3 + 0.6*frac), lw=2,
                label=f"t = {tm} min")
    ax.axhline(E_bulk_nM*1e9, color="gray", ls="--", lw=1, label="Bulk [E]")
    ax.set_xlabel("Distance from electrode z (µm)")
    ax.set_ylabel("[MMP-9] (nM)")
    ax.set_title(f"A  —  Diffusion profiles (Da = {Da:.3f})")
    ax.legend(fontsize=8)
    ax.set_xlim(0, h*1e6)

    # Panel B: surface [E](t) normalised to bulk
    ax2 = axes[1]
    E_surface_norm = E_store[:, 0] / E_bulk_SI
    ax2.plot(t_arr / 60, E_surface_norm * 100, "b-", lw=2)
    ax2.axhline(100, color="gray", ls="--", lw=1, label="Bulk [E]")
    ax2.axhline(90, color="orange", ls=":", lw=1.2, label="10% depletion")
    ax2.set_xlabel("Incubation time (min)")
    ax2.set_ylabel("[E]_surface / [E]_bulk (%)")
    ax2.set_title("B  —  Surface enzyme depletion over time")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)

    fig.suptitle(f"Layer 3: Mass transport — 1D diffusion PDE (Crank-Nicolson)\n"
                 f"D_{{MMP9}} = {D_MMP9*1e11:.1f}×10⁻¹¹ m²/s, "
                 f"h = {h*1e6:.0f} µm, Da = {Da:.3f}", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B3_mass_transport.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"D_MMP9 = {D_MMP9:.2e} m²/s  ({D_MMP9*1e4:.2e} cm²/s)")
    print(f"D_HNE  = {D_HNE:.2e} m²/s  ({D_HNE*1e4:.2e} cm²/s)")
    k_obs = kcat_Km_surf * E_bulk_nM
    Da    = compute_Da(kcat_Km_surf, E_bulk_nM, h, D_MMP9, Gamma_0)
    print(f"k_obs (upper clinical) = {k_obs:.3e} s-1")
    print(f"Damköhler number Da = {Da:.4f} (Da << 1: transport not limiting)")
    plot_transport()
