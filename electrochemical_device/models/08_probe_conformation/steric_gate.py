"""
steric_gate.py -- Layer 8, Sections IV-V (Sec. 8.4-8.5 of the planning notes)
===============================================================================
Steric gating of enzyme access to the surface-immobilised SGKIPRTLTA probe,
and the SAM-density dependence of the probe's brush/mushroom conformation.

Enzyme footprint: Layer 7 (Sec. "Probe spacing constraint from enzyme size")
already estimates r_MMP-9 ~ 3-4 nm (from MW ~ 92 kDa assuming globular
geometry; NOT independently confirmed from a paper in papers/layer_07/) and
derives a spacing criterion l >~ 2 r_MMP-9 from a simple nearest-neighbour
argument. This script uses the same r_MMP-9 range but the area-exclusion
criterion from the planning notes (Sec. 8.5), A_probe = 1/Gamma_p > pi
r_enzyme^2, and cross-checks the result against Layer 7's own spacing-based
number as an internal consistency check between the two independent
geometric arguments.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import probe_conformation as pc

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

N_A = 6.02214076e23  # mol^-1

# ── Enzyme footprint (Layer 7 estimate; unconfirmed from any paper) ───────
r_MMP9_lo_nm = 3.0
r_MMP9_hi_nm = 4.0
r_MMP9_mid_nm = 3.5


def Gamma_steric_max_pmol(r_enzyme_nm):
    """
    Area-exclusion criterion (planning notes, Sec. 8.5):
        A_probe = 1/Gamma_p > pi r_enzyme^2  =>  Gamma_p < 1/(pi r_enzyme^2)
    Returns Gamma_steric_max in pmol/cm^2.
    """
    r_cm = r_enzyme_nm * 1e-7
    Gamma_per_area = 1.0 / (np.pi * r_cm**2)           # molecules / cm^2 (per-molecule count basis... )
    # the criterion above is a *molecule count* areal density; convert to a
    # molar areal density (mol/cm^2) then to pmol/cm^2:
    Gamma_molar = Gamma_per_area / N_A                  # mol/cm^2
    return Gamma_molar * 1e12                           # pmol/cm^2


def Gamma_spacing_max_pmol(r_enzyme_nm):
    """
    Layer 7's independent nearest-neighbour spacing criterion:
        l >= 2 r_enzyme,   l = (Gamma_p * N_A)^(-1/2)
        => Gamma_p <= 1 / (N_A * (2 r_enzyme)^2)
    Returns pmol/cm^2, for cross-checking against the area-exclusion number.
    """
    l_min_cm = 2 * r_enzyme_nm * 1e-7
    Gamma_molar = 1.0 / (N_A * l_min_cm**2)
    return Gamma_molar * 1e12


def f_steric(Gamma_p_pmol, Gamma_max_pmol):
    """Steric accessibility factor [0, 1]."""
    return np.maximum(0.0, 1.0 - Gamma_p_pmol / Gamma_max_pmol)


# ── Brush/mushroom conformation vs. SAM density (Sec. 8.4) ───────────────
def Gamma_crossover_pmol(l_p_m):
    """
    Mushroom -> brush crossover density (planning notes, Sec. 8.4):
        Gamma_crossover ~ 1/(4 pi l_p^2)
    l_p in metres; returns pmol/cm^2.
    """
    l_p_cm = l_p_m * 1e2
    Gamma_per_area = 1.0 / (4 * np.pi * l_p_cm**2)       # molecules/cm^2
    return (Gamma_per_area / N_A) * 1e12                 # pmol/cm^2


def d_eff_brush(Gamma_p_pmol, r_g_m, Gamma_crossover_pmol_val):
    """
    Brush-to-mushroom interpolation (planning notes, Sec. 8.4):
        d_eff = r_g * (1 + Gamma_p/Gamma_crossover)^(1/3)
    """
    return r_g_m * (1 + Gamma_p_pmol / Gamma_crossover_pmol_val) ** (1 / 3)


def run():
    # radius of gyration proxy for the full tether, from the additive-variance
    # WLC <r^2> computed in probe_conformation.py (Gaussian-chain relation
    # r_g^2 = <R^2>/6; used here only to set the mushroom-regime length scale
    # for the brush model, not as a substitute for the full MC P(r))
    r2_full = pc.wlc_mean_r2(pc.L_c_total, pc.lp_eff)
    r_g = np.sqrt(r2_full / 6)

    Gamma_max_area = {
        "lo (r=3.0 nm)": Gamma_steric_max_pmol(r_MMP9_lo_nm),
        "mid (r=3.5 nm)": Gamma_steric_max_pmol(r_MMP9_mid_nm),
        "hi (r=4.0 nm)": Gamma_steric_max_pmol(r_MMP9_hi_nm),
    }
    Gamma_max_spacing = {
        "lo (r=3.0 nm)": Gamma_spacing_max_pmol(r_MMP9_lo_nm),
        "mid (r=3.5 nm)": Gamma_spacing_max_pmol(r_MMP9_mid_nm),
        "hi (r=4.0 nm)": Gamma_spacing_max_pmol(r_MMP9_hi_nm),
    }

    Gamma_star = Gamma_crossover_pmol(pc.lp_eff)

    print("=" * 70)
    print("Layer 8.5: Steric exclusion of MMP-9 by probe crowding")
    print("=" * 70)
    print(f"MB-tether radius of gyration proxy r_g = {r_g*1e9:.2f} nm  "
          f"(from l_p,eff = {pc.lp_eff*1e9:.2f} nm, L_c = {pc.L_c_total*1e9:.2f} nm)")
    print()
    print("Gamma_steric_max (area-exclusion, pi r^2):")
    for k, v in Gamma_max_area.items():
        print(f"  r_MMP-9 {k:<16s}: {v:6.2f} pmol/cm^2")
    print()
    print("Gamma_steric_max (Layer 7 spacing criterion, cross-check):")
    for k, v in Gamma_max_spacing.items():
        print(f"  r_MMP-9 {k:<16s}: {v:6.2f} pmol/cm^2")
    print()
    print(f"Layer 7 target Gamma_p = 2-5 pmol/cm^2 (device-planned)")
    for Gp in [1, 2, 3, 5, 8]:
        fs_mid = f_steric(Gp, Gamma_max_area["mid (r=3.5 nm)"])
        print(f"  Gamma_p = {Gp} pmol/cm^2: f_steric(mid r) = {fs_mid:.3f}")
    print()
    print(f"Mushroom->brush crossover Gamma* = 1/(4 pi l_p,eff^2) = {Gamma_star:.1f} pmol/cm^2")
    print("(far above the device's target 2-5 pmol/cm^2 window: the tether is")
    print(" in the mushroom regime at all steric-window-compatible densities)")

    # d_eff(Gamma_p) sweep, and the delta_eff estimate that Layer 9 needs
    Gp_sweep = np.linspace(0.1, 20, 400)
    d_eff_vals = d_eff_brush(Gp_sweep, r_g, Gamma_star) * 1e9  # nm

    d_eff_at_3 = d_eff_brush(3.0, r_g, Gamma_star) * 1e9
    print()
    print(f"d_eff at Gamma_p = 3 pmol/cm^2 (Layer 7 target midpoint): {d_eff_at_3:.2f} nm")
    print(f"  -> proposed delta_eff for Layer 9 Eq. IV.2, replacing their")
    print(f"     5 nm placeholder: delta_eff ~ {d_eff_at_3:.1f} nm (mushroom-regime r_g-based)")

    plot_steric(Gamma_max_area, Gamma_max_spacing, Gp_sweep, d_eff_vals, Gamma_star, r_g)

    return dict(Gamma_max_area=Gamma_max_area, Gamma_max_spacing=Gamma_max_spacing,
                Gamma_star=Gamma_star, r_g_nm=r_g * 1e9, d_eff_at_3nm=d_eff_at_3)


def plot_steric(Gamma_max_area, Gamma_max_spacing, Gp_sweep, d_eff_vals, Gamma_star, r_g):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    # Panel A: f_steric vs Gamma_p, for the r_MMP-9 range
    ax = axes[0]
    colors = {"lo (r=3.0 nm)": "#8fbfe0", "mid (r=3.5 nm)": "#3b6fa0", "hi (r=4.0 nm)": "#1a3a5c"}
    Gp = np.linspace(0.1, 12, 300)
    for k, Gmax in Gamma_max_area.items():
        ax.plot(Gp, f_steric(Gp, Gmax), color=colors[k], lw=2, label=f"$r_{{MMP9}}$ {k}")
    ax.axvspan(2, 5, color="green", alpha=0.12, label="Layer 7 target (2-5 pmol/cm$^2$)")
    ax.set_xlabel("Probe density $\\Gamma_p$ (pmol/cm$^2$)")
    ax.set_ylabel("$f_{steric}$")
    ax.set_title("A — Steric accessibility vs. $\\Gamma_p$")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)

    # Panel B: area-exclusion vs spacing-criterion cross-check
    ax2 = axes[1]
    labels = list(Gamma_max_area.keys())
    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(x - w/2, [Gamma_max_area[k] for k in labels], width=w,
            color="#3b6fa0", label="Area-exclusion (Eq. 8.5, this layer)")
    ax2.bar(x + w/2, [Gamma_max_spacing[k] for k in labels], width=w,
            color="#c0703b", label="Spacing criterion (Layer 7)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("$\\Gamma_{steric}^{max}$ (pmol/cm$^2$)")
    ax2.set_title("B — Two independent geometric estimates")
    ax2.legend(fontsize=7)

    # Panel C: d_eff(Gamma_p) brush model
    ax3 = axes[2]
    ax3.plot(Gp_sweep, d_eff_vals, "b-", lw=2, label="$d_{eff}(\\Gamma_p)$, brush model")
    ax3.axhline(r_g * 1e9, color="gray", ls=":", lw=1.2, label=f"$r_g$ = {r_g*1e9:.2f} nm (mushroom limit)")
    ax3.axvline(Gamma_star, color="r", ls="--", lw=1.2, label=f"$\\Gamma^*$ = {Gamma_star:.0f} pmol/cm$^2$")
    ax3.axvspan(2, 5, color="green", alpha=0.12, label="Layer 7 target")
    ax3.set_xlabel("Probe density $\\Gamma_p$ (pmol/cm$^2$)")
    ax3.set_ylabel("$d_{eff}$ (nm)")
    ax3.set_title("C — Brush/mushroom $d_{eff}(\\Gamma_p)$")
    ax3.legend(fontsize=7)
    ax3.set_xlim(0, 20)

    fig.suptitle("Layer 8.4/8.5: Steric gate and SAM-density conformation "
                 "(SGKIPRTLTA probe, MMP-9 footprint)", fontsize=10)
    plt.tight_layout()
    out = FIGURES / "fig_L8_steric_gate.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    run()
