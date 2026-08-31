"""
fragment_clearance.py  —  Layer 3, Sections 3.6-3.7
=======================================================
After AAPV|SGIG cleavage, the MB-labelled fragment (SGIG-PEG2-Lys(MB),
~1 kDa, D_frag ~ 3.2e-10 m^2/s) is released from the surface and diffuses
into the bulk. Two questions matter for the SWV measurement downstream
(Layer 11):

  1. Nernst-layer clearance (3.6): does the fragment leave the thin
     diffusion layer immediately adjacent to the electrode quickly enough
     that it does not itself contribute a diffusion-controlled MB peak
     superimposed on the surface-confined peak of interest?

  2. Bulk accumulation and cross-talk (3.6-3.7): over a full incubation,
     does the released fragment build up in the bulk chamber solution to a
     concentration that could be detected as background signal, and can it
     reach a neighbouring electrode (different probe channel) and
     contaminate that measurement?

All quantities here are downstream outputs of the coupled Gamma(t) computed
in diffusion_pde.py: Gamma(t) sets how much MB has been released as a
function of time, which is what feeds the calculations below.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from diffusion_pde import D_FRAG_25C

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

NA = 6.02214076e23  # Avogadro's number


def nernst_clearance_time(delta, D_frag=D_FRAG_25C):
    """
    Time for the fragment to diffuse beyond a Nernst diffusion layer of
    thickness delta (m): t_clear = delta^2 / D_frag.
    """
    return delta ** 2 / D_frag


def bulk_MB_concentration(Gamma0, F_k, A_electrode, V_chamber):
    """
    Bulk MB concentration (mol/m^3) after a fraction F_k of the probes on
    one electrode of area A_electrode (m^2) have been cleaved, assuming
    complete, well-mixed distribution through the chamber volume
    V_chamber (m^3).

        n_MB = Gamma0 * A_electrode * F_k
        [MB]_bulk = n_MB / V_chamber
    """
    n_MB = Gamma0 * A_electrode * F_k
    return n_MB / V_chamber


def cross_talk_time(d_elec, D_frag=D_FRAG_25C):
    """Characteristic lateral diffusion time to travel a separation
    d_elec (m) between neighbouring electrodes: t = d_elec^2/(2 D_frag)."""
    return d_elec ** 2 / (2 * D_frag)


def point_source_concentration(n_mol, d_elec, t, h_channel, D_frag=D_FRAG_25C):
    """
    Order-of-magnitude estimate of the concentration a neighbouring
    electrode sees due to lateral diffusion from a point-like release of
    n_mol moles of fragment at the source electrode, using the 2-D
    (in-plane) diffusion Green's function appropriate for a source
    confined to the channel height h_channel:

        C(d, t) = n_mol / (4 pi D_frag t h_channel) * exp(-d^2/(4 D_frag t))

    This is an idealised instantaneous point release, not the true
    continuous, spatially-extended release; it is used only to confirm
    that even a generous upper-bound estimate falls far below the ~100 nM
    solution-phase SWV detection floor at electrode-relevant separations
    and incubation times.
    """
    return (n_mol / (4 * np.pi * D_frag * t * h_channel)
            * np.exp(-d_elec ** 2 / (4 * D_frag * t)))


def plot_clearance_and_crosstalk(save=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: Nernst-layer clearance time vs. delta
    delta_um = np.linspace(10, 150, 200)
    t_clear = nernst_clearance_time(delta_um * 1e-6)
    ax = axes[0]
    ax.plot(delta_um, t_clear, "m-", lw=2)
    ax.axvspan(50, 100, color="gray", alpha=0.15, label="Typical quiescent range")
    ax.set_xlabel(r"Nernst diffusion layer thickness $\delta$ (µm)")
    ax.set_ylabel(r"Clearance time $t_{clear}$ (s)")
    ax.set_title("A — Fragment clearance from the electrode")
    ax.legend(fontsize=8)

    # Panel B: bulk [MB] buildup vs incubation time, for a range of F_k(T)
    A_electrode = 1e-6           # m^2 (1 mm^2)
    V_chamber = 37.5e-9          # m^3 (37.5 µL)
    Gamma0 = 5e-12 * 1e4         # mol/m^2 (5 pmol/cm^2)
    T_arr_min = np.linspace(0, 60, 200)
    T_arr_s = T_arr_min * 60

    k_obs_lo = 188e3 * 1e-3 * 1e3 * 0.185e-9      # illustrative low [E]
    k_obs_hi = 188e3 * 1e-3 * 1e3 * 2.17e-9       # illustrative high [E]
    # (k_obs in SI here; equivalent to kcat_over_Km[SI] * E0[SI])
    Fk_lo = 1 - np.exp(-k_obs_lo * T_arr_s)
    Fk_hi = 1 - np.exp(-k_obs_hi * T_arr_s)

    MB_lo_nM = bulk_MB_concentration(Gamma0, Fk_lo, A_electrode, V_chamber) * 1e6
    MB_hi_nM = bulk_MB_concentration(Gamma0, Fk_hi, A_electrode, V_chamber) * 1e6
    # (mol/m^3 -> nM: x1e6)

    ax2 = axes[1]
    ax2.plot(T_arr_min, MB_lo_nM, "b-", lw=2, label="Low [E] (17 ng/mL-equiv.)")
    ax2.plot(T_arr_min, MB_hi_nM, "r-", lw=2, label="High [E] (200 ng/mL-equiv.)")
    ax2.axhline(100, color="gray", ls="--", lw=1.2,
                label="Typical solution-phase SWV detection floor (~100 nM)")
    ax2.axvline(30, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Incubation time (min)")
    ax2.set_ylabel("Bulk [MB] (nM)")
    ax2.set_title("B — Bulk MB fragment accumulation")
    ax2.legend(fontsize=7)

    # Panel C: cross-talk — characteristic lateral diffusion time vs.
    # separation, compared against the incubation timescale
    d_mm = np.linspace(0.2, 5, 200)
    t_cross_min = cross_talk_time(d_mm * 1e-3) / 60
    ax3 = axes[2]
    ax3.plot(d_mm, t_cross_min, "g-", lw=2)
    ax3.axhline(30, color="gray", ls="--", lw=1.2, label="T = 30 min incubation")
    ax3.axvline(1.0, color="gray", ls=":", lw=1, label="1 mm design minimum")
    ax3.set_xlabel("Electrode-to-electrode separation (mm)")
    ax3.set_ylabel(r"$t_{cross} = d_{elec}^2/(2D_{frag})$ (min)")
    ax3.set_title("C — Lateral cross-talk timescale vs. separation")
    ax3.set_yscale("log")
    ax3.legend(fontsize=8)

    fig.suptitle("Layer 3: Released MB-fragment clearance, bulk accumulation, "
                 "and electrode cross-talk", fontsize=10)
    plt.tight_layout()

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        out = FIGURES / "fig_B_fragment_clearance.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    # ── Console summary ──────────────────────────────────────────────────
    print(f"t_clear (delta=50 µm)  = {nernst_clearance_time(50e-6):.2f} s")
    print(f"t_clear (delta=100 µm) = {nernst_clearance_time(100e-6):.2f} s")

    MB_50pct = bulk_MB_concentration(Gamma0, 0.5, A_electrode, V_chamber)
    print(f"Bulk [MB] at F_k=0.5, T=30 min, single 1 mm^2 electrode "
          f"= {MB_50pct*1e6:.2f} nM  (vs. ~100 nM solution-phase SWV floor)")

    h_channel = 375e-6
    T_incubation = 1800.0
    n_mol_released = Gamma0 * A_electrode * 0.5   # F_k=0.5 on a 1 mm^2 electrode
    for d_mm_val in [1, 2, 5]:
        t_c = cross_talk_time(d_mm_val * 1e-3)
        C_neighbour = point_source_concentration(
            n_mol_released, d_mm_val * 1e-3, T_incubation, h_channel)
        print(f"d_elec = {d_mm_val} mm: t_cross = {t_c:.0f} s ({t_c/60:.1f} min, "
              f"{t_c/T_incubation:.1f}x the incubation time); "
              f"point-source upper-bound conc. at neighbour, T=30 min "
              f"= {C_neighbour*1e6:.2e} nM")


if __name__ == "__main__":
    plot_clearance_and_crosstalk()
