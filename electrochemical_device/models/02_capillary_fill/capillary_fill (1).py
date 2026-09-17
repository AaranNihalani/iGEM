"""
capillary_fill.py  --  Layer 2 (Cartridge Inlet and Capillary Fill)
=====================================================================
Eton iGEM 2026, Dry Lab.

Implements Sections 2.1-2.3, 2.5 and 2.7 of contents.md:

  2.1  Capillary driving pressure, dP_cap = gamma*cos(theta)*(1/h + 1/w)
  2.2  Lucas-Washburn free-fill dynamics, L(t) and t_fill(h)
  2.3  Air venting: no-vent stall length L_stall, and a coupled
       "liquid resistance in series with vent resistance" ODE that
       gives the *actual*, vent-limited fill time and the minimum
       vent membrane area needed to hit a fill-time target
  2.5  Electrode wet-time differential t_wet,i (dead-volume timing)
  2.7  Coupling to the Layer 1 inlet filter (pressure-budget check)

Governing equations are taken verbatim from contents.md Sec. 2.1-2.3
(the L(t)=sqrt(h^2 dP t/(3 eta)) form, factor 3, not the classical
parallel-plate Poiseuille factor of 12 -- see the LIMITATIONS note at
the bottom of this file and Section II of the report).

Run directly to print the full numeric summary used in the Layer 2
report and regenerate figures/fig_L2_1_fill_dynamics.pdf and
figures/fig_L2_2_vent_sizing.pdf.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 0. Chamber geometry (contents.md Sec. "Chamber geometry")
# =====================================================================
L_chamber = 20e-3          # m   chamber length
w_chamber = 5e-3           # m   chamber width
h_values  = np.array([0.25, 0.30, 0.375, 0.50]) * 1e-3   # m, design sweep
h_design  = 0.375e-3       # m   design midpoint (matches Layer 3's own
                            #     "design midpoint h = 375 um" reference point)
P_atm = 101_325.0          # Pa

# Electrode layout (contents.md Sec. 2.5 worked example; 8 electrodes,
# x1 = 2 mm, x8 = 18 mm, evenly spaced -> pitch 2.29 mm, within the
# 2-3 mm spec range)
n_electrodes = 8
x_electrodes = np.linspace(2e-3, 18e-3, n_electrodes)      # m

# =====================================================================
# 1. Fluid properties
# =====================================================================
# Surface tension:
gamma_water = 0.072                    # N/m, water, 25 C -- CONFIRMED (standard)
gamma_wound_lo, gamma_wound_hi = 0.050, 0.065   # N/m, wound fluid w/ surfactant
                                        # proteins -- contents.md placeholder, FLAG
gamma_wound = 0.5 * (gamma_wound_lo + gamma_wound_hi)   # midpoint used for headline #s

# Viscosity:
eta_water_25C = 1.0e-3                 # Pa s -- CONFIRMED (standard)
eta_contents_placeholder = 1.2e-3      # Pa s, contents.md's wound-fluid placeholder
                                        # (37 C) -- FLAG, see below
# Literature-corrected wound EXUDATE viscosity (real seropurulent samples,
# shear-thinning, 37 C): 6.5-30.8 mPa.s across the shear-rate range tested.
# This is 5-25x the contents.md placeholder -- see report Sec. II.
eta_exudate_lo, eta_exudate_hi = 6.5e-3, 30.8e-3   # Pa s -- CONFIRMED (cited)

# Contact angles (contents.md Sec. 2.1, 2.4). Design target after O2-plasma
# clean is theta < 30 deg on all three surface types (gold/probe-SAM floor,
# polymer lid, polymer side walls); native/unclean polymer and gold are far
# more hydrophobic (theta ~ 70-110 deg literature range) -- FLAG, stack-specific
# values are unconfirmed pending contact-angle goniometry on the fabricated
# cartridge.
theta_design_deg = 30.0                # deg, post-plasma target, all surfaces
theta_native_deg = 100.0               # deg, representative unclean/native surface

# =====================================================================
# 2. Vent membrane properties (contents.md Sec. 2.3)
# =====================================================================
# contents.md's own placeholder range for ePTFE gas permeance:
Pg_lit_lo, Pg_lit_hi = 1e-8, 1e-7      # m^3/(s m^2 Pa) -- FLAG, generic literature range
# A representative commercial 0.22 um pore ePTFE membrane datasheet gives
# breathability >= 0.4 L/min/cm^2 at 7 kPa, i.e.:
Pg_datasheet = (0.4e-3 / 60) / 1e-4 / 7000     # m^3/(s m^2 Pa)
A_vent_design = 5e-6                   # m^2 (5 mm^2), candidate design vent area

# =====================================================================
# Core physics (contents.md Sec. 2.1-2.3, formulas used verbatim)
# =====================================================================
def dP_cap(h, gamma=gamma_wound, theta_deg=theta_design_deg, w=w_chamber):
    """Capillary driving pressure for a rectangular channel, h << w."""
    theta = np.radians(theta_deg)
    return gamma * np.cos(theta) * (1.0 / h + 1.0 / w)


def t_fill_free(x, h, eta, gamma=gamma_wound, theta_deg=theta_design_deg):
    """Free (unvented, no-resistance-added) Lucas-Washburn time to reach
    position x (contents.md Eq., factor of 3)."""
    dP = dP_cap(h, gamma, theta_deg)
    return 3.0 * eta * x**2 / (h**2 * dP)


def L_of_t_free(t, h, eta, gamma=gamma_wound, theta_deg=theta_design_deg):
    dP = dP_cap(h, gamma, theta_deg)
    return np.sqrt(h**2 * dP * t / (3.0 * eta))


def L_stall(h, gamma=gamma_wound, theta_deg=theta_design_deg, P_atm=P_atm,
            L=L_chamber):
    """Maximum fill length with NO vent (contents.md Sec. 2.3): fill stalls
    when the trapped-air back-pressure equals the capillary driving
    pressure."""
    dP = dP_cap(h, gamma, theta_deg)
    return L * dP / (dP + P_atm)


def R_vent(h, Pg, A_vent, w=w_chamber):
    """Series flow resistance (Pa s / m) presented by the vent membrane,
    referred to the liquid front velocity dL/dt (contents.md Sec. 2.3:
    Q = A_cross * dL/dt with A_cross = w*h, the liquid flow cross-section)."""
    return (w * h) / (Pg * A_vent)


def t_fill_vented(x, h, eta, Pg, A_vent, gamma=gamma_wound,
                   theta_deg=theta_design_deg, w=w_chamber):
    """Time to reach position x when the vent membrane's finite gas
    permeance is included as a series resistance to the liquid-viscous
    resistance (derivation: report Sec. III). Reduces exactly to
    t_fill_free when A_vent -> infinity (R_vent -> 0)."""
    dP = dP_cap(h, gamma, theta_deg)
    Rv = R_vent(h, Pg, A_vent, w)
    return (3.0 * eta * x**2 / h**2 + Rv * x) / dP


def min_vent_area(h, eta, Pg, t_target, gamma=gamma_wound,
                   theta_deg=theta_design_deg, w=w_chamber, L=L_chamber):
    """Minimum vent area to reach t_fill(L) = t_target. Returns None if the
    free-fill time already meets the target (vent is not the bottleneck at
    any positive area)."""
    dP = dP_cap(h, gamma, theta_deg)
    free = 3.0 * eta * L**2 / (h**2 * dP)
    if free >= t_target:
        return None
    Rv_needed = (t_target * dP - 3.0 * eta * L**2 / h**2) / L
    return (w * h) / (Pg * Rv_needed)


def dP_vent_at_Q(h, Pg, A_vent, dLdt, w=w_chamber):
    """Back-pressure the vent membrane presents at a given front velocity."""
    Q = w * h * dLdt
    return Q / (Pg * A_vent)


# =====================================================================
# 3. Numeric summary (printed; drives the report's tables)
# =====================================================================
def summarise():
    print("=" * 78)
    print("LAYER 2 -- Capillary fill: numeric summary")
    print("=" * 78)

    print(f"\nChamber: L={L_chamber*1e3:.0f} mm, w={w_chamber*1e3:.0f} mm, "
          f"h in [{h_values.min()*1e3:.2f}, {h_values.max()*1e3:.2f}] mm")
    for h in h_values:
        V_uL = L_chamber * w_chamber * h * 1e9
        print(f"  h={h*1e3:5.3f} mm  ->  V = {V_uL:5.1f} uL")

    print("\n--- 2.1 Capillary driving pressure (design theta=30 deg) ---")
    for h in h_values:
        dP = dP_cap(h)
        print(f"  h={h*1e3:5.3f} mm  ->  dP_cap = {dP:7.1f} Pa "
              f"(water: {dP_cap(h, gamma_water):7.1f} Pa)")

    print("\n--- 2.2 Free Lucas-Washburn fill time to L_chamber ---")
    print("  (contents.md Eq., factor-3 form; see LIMITATIONS)")
    header = f"  {'h (mm)':>7} {'t_fill water (s)':>18} " \
             f"{'t_fill contents-eta (s)':>25} {'t_fill exudate-lo (s)':>22} " \
             f"{'t_fill exudate-hi (s)':>22}"
    print(header)
    for h in h_values:
        tw = t_fill_free(L_chamber, h, eta_water_25C)
        tc = t_fill_free(L_chamber, h, eta_contents_placeholder)
        tl = t_fill_free(L_chamber, h, eta_exudate_lo)
        th = t_fill_free(L_chamber, h, eta_exudate_hi)
        print(f"  {h*1e3:7.3f} {tw:18.4f} {tc:25.4f} {tl:22.4f} {th:22.4f}")

    print("\n--- 2.3a No-vent stall length ---")
    for h in h_values:
        Ls = L_stall(h)
        print(f"  h={h*1e3:5.3f} mm  ->  L_stall = {Ls*1e6:7.1f} um "
              f"({Ls/L_chamber*100:.3f}% of chamber length) -- fill essentially "
              f"does not start without a vent")

    print(f"\n--- 2.3b Vent membrane permeance: literature placeholder vs. datasheet ---")
    print(f"  contents.md placeholder range: {Pg_lit_lo:.1e} - {Pg_lit_hi:.1e} "
          f"m^3/(s m^2 Pa)")
    print(f"  0.22 um ePTFE datasheet (>=0.4 L/min/cm^2 @ 7 kPa): "
          f"Pg = {Pg_datasheet:.2e} m^3/(s m^2 Pa)")

    print(f"\n--- 2.3c Vent-limited fill time at design h={h_design*1e3:.3f} mm, "
          f"A_vent={A_vent_design*1e6:.0f} mm^2, worst-case exudate "
          f"eta={eta_exudate_hi*1e3:.1f} mPa.s ---")
    for label, Pg in [("literature-lo", Pg_lit_lo), ("literature-hi", Pg_lit_hi),
                       ("datasheet", Pg_datasheet)]:
        t = t_fill_vented(L_chamber, h_design, eta_exudate_hi, Pg, A_vent_design)
        print(f"  Pg={label:14s} ({Pg:.2e}) -> t_fill = {t:8.3f} s")

    print(f"\n--- 2.3d Minimum vent area to hit t_fill(L_chamber) = 30 s, "
          f"h={h_design*1e3:.3f} mm, worst-case exudate viscosity ---")
    for label, Pg in [("literature-lo", Pg_lit_lo), ("literature-hi", Pg_lit_hi),
                       ("datasheet", Pg_datasheet)]:
        A = min_vent_area(h_design, eta_exudate_hi, Pg, 30.0)
        if A is None:
            print(f"  Pg={label:14s}: free-fill already < 30 s, vent not the "
                  f"bottleneck at any area")
        else:
            print(f"  Pg={label:14s} ({Pg:.2e}) -> A_vent,min = {A*1e6:8.2f} mm^2")

    print("\n--- 2.5 Electrode wet-time differential (x1=2mm, x8=18mm), "
          f"h={h_design*1e3:.3f} mm, worst-case exudate viscosity ---")
    x1, x8 = x_electrodes[0], x_electrodes[-1]
    t1_free = t_fill_free(x1, h_design, eta_exudate_hi)
    t8_free = t_fill_free(x8, h_design, eta_exudate_hi)
    print(f"  Free fill:   t_wet,1 = {t1_free:.4f} s   t_wet,8 = {t8_free:.4f} s   "
          f"dt = {t8_free-t1_free:.4f} s  ({(t8_free-t1_free)/1800*100:.3f}% of 30-min incubation)")
    t1_vent = t_fill_vented(x1, h_design, eta_exudate_hi, Pg_datasheet, A_vent_design)
    t8_vent = t_fill_vented(x8, h_design, eta_exudate_hi, Pg_datasheet, A_vent_design)
    print(f"  Vented ({A_vent_design*1e6:.0f} mm^2, datasheet Pg): "
          f"t_wet,1 = {t1_vent:.4f} s   t_wet,8 = {t8_vent:.4f} s   "
          f"dt = {t8_vent-t1_vent:.4f} s  "
          f"({(t8_vent-t1_vent)/1800*100:.3f}% of 30-min incubation)")

    print("\n--- Handoff to Layer 3: Peclet number for fill-phase convection ---")
    D_MMP9 = 5.0e-11    # m^2/s, from README / Layer 0
    t_fill_report = t_fill_vented(L_chamber, h_design, eta_exudate_hi,
                                   Pg_datasheet, A_vent_design)
    u = L_chamber / t_fill_report
    Pe = u * L_chamber / D_MMP9
    print(f"  This report's t_fill (vented, worst-case eta) = {t_fill_report:.3f} s")
    print(f"  u = L/t_fill = {u*1e3:.2f} mm/s")
    print(f"  Pe = u*L/D_MMP9 = {Pe:.2e}  (Layer 3 assumed t_fill~60s -> "
          f"Pe~1.3e5; this is {Pe/1.3e5:.1f}x higher, i.e. an even safer margin)")

    return {
        "t_fill_report": t_fill_report,
    }


# =====================================================================
# 4. Figures
# =====================================================================
def make_figure_1():
    """Fill dynamics: L(t) curves, t_fill(h) sweep, stall length."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel A: L(t) at design h, free vs vented, water vs exudate viscosity
    ax = axes[0]
    t_max = 3.0 * eta_exudate_hi * L_chamber**2 / (h_design**2 * dP_cap(h_design)) * 1.3
    t = np.linspace(1e-4, t_max, 800)
    L_water = np.minimum(L_of_t_free(t, h_design, eta_water_25C), L_chamber) * 1e3
    L_exlo = np.minimum(L_of_t_free(t, h_design, eta_exudate_lo), L_chamber) * 1e3
    L_exhi = np.minimum(L_of_t_free(t, h_design, eta_exudate_hi), L_chamber) * 1e3
    ax.plot(t, L_water, "g--", lw=1.6, label=r"free fill, water $\eta$=1.0 mPa$\cdot$s")
    ax.plot(t, L_exlo, "b-", lw=1.8, label=r"free fill, exudate $\eta$=6.5 mPa$\cdot$s")
    ax.plot(t, L_exhi, "b-.", lw=1.8, label=r"free fill, exudate $\eta$=30.8 mPa$\cdot$s")
    # vented curve (numeric): invert t_fill_vented(x) on a grid of x
    x_grid = np.linspace(1e-6, L_chamber, 400)
    t_vent_grid = t_fill_vented(x_grid, h_design, eta_exudate_hi, Pg_datasheet, A_vent_design)
    ax.plot(t_vent_grid, x_grid * 1e3, "r-", lw=2.0,
            label=r"vented ($A_v$=5 mm$^2$, datasheet $P_g$), exudate hi-$\eta$")
    ax.axhline(L_chamber * 1e3, color="k", ls=":", lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fill front position (mm)")
    ax.set_title(f"A  Fill front, h = {h_design*1e3:.3f} mm")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_xlim(0, t_max)

    # Panel B: t_fill(h) sweep, free vs vented, log y
    ax = axes[1]
    h_fine = np.linspace(h_values.min(), h_values.max(), 100)
    ax.plot(h_fine * 1e3, t_fill_free(L_chamber, h_fine, eta_water_25C),
            "g--", lw=1.6, label=r"free, water")
    ax.plot(h_fine * 1e3, t_fill_free(L_chamber, h_fine, eta_exudate_lo),
            "b-", lw=1.8, label=r"free, exudate lo-$\eta$")
    ax.plot(h_fine * 1e3, t_fill_free(L_chamber, h_fine, eta_exudate_hi),
            "b-.", lw=1.8, label=r"free, exudate hi-$\eta$")
    ax.plot(h_fine * 1e3, t_fill_vented(L_chamber, h_fine, eta_exudate_hi,
                                        Pg_datasheet, A_vent_design),
            "r-", lw=2.0, label=r"vented, datasheet $P_g$, exudate hi-$\eta$")
    ax.axhline(30, color="k", ls=":", lw=1.2, label="30 s spec")
    ax.set_yscale("log")
    ax.set_xlabel("Chamber height h (mm)")
    ax.set_ylabel("$t_{fill}$ (s)")
    ax.set_title("B  Fill time vs. chamber height")
    ax.legend(fontsize=6.5)

    # Panel C: stall length vs h (log y, no vent)
    ax = axes[2]
    Ls = np.array([L_stall(h) for h in h_fine]) * 1e6
    ax.plot(h_fine * 1e3, Ls, "purple", lw=2)
    ax.axhline(L_chamber * 1e6, color="k", ls=":", lw=1, label="chamber length (20 mm)")
    ax.set_yscale("log")
    ax.set_xlabel("Chamber height h (mm)")
    ax.set_ylabel(r"$L_{stall}$ without a vent ($\mu$m)")
    ax.set_title("C  No-vent stall length")
    ax.legend(fontsize=7)

    fig.suptitle("Layer 2: Capillary fill dynamics (capillary_fill.py)", fontsize=11)
    plt.tight_layout()
    out = FIGDIR / "fig_L2_1_fill_dynamics.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def make_figure_2():
    """Vent sizing and filter coupling."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel A: minimum vent area vs. assumed Pg, to hit 30s target
    ax = axes[0]
    Pg_grid = np.logspace(-8.3, -4.7, 200)
    A_min = np.array([min_vent_area(h_design, eta_exudate_hi, pg, 30.0) or np.nan
                       for pg in Pg_grid]) * 1e6
    ax.plot(Pg_grid, A_min, "k-", lw=2)
    ax.axvspan(Pg_lit_lo, Pg_lit_hi, color="orange", alpha=0.25,
               label="contents.md placeholder range")
    ax.axvline(Pg_datasheet, color="b", ls="--", lw=1.5,
               label=f"0.22 $\\mu$m ePTFE datasheet\n($P_g$={Pg_datasheet:.1e})")
    ax.axhline(100, color="r", ls=":", lw=1, label="chamber footprint (100 mm$^2$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Vent permeance $P_g$ (m$^3$ s$^{-1}$ m$^{-2}$ Pa$^{-1}$)")
    ax.set_ylabel(r"Min. vent area for $t_{fill}$<30 s (mm$^2$)")
    ax.set_title("A  Vent sizing vs. membrane permeance")
    ax.legend(fontsize=6.5)

    # Panel B: filter coupling -- t_fill vs dP_filter (Layer 1 handoff)
    ax = axes[1]
    dP = dP_cap(h_design)
    Rv = R_vent(h_design, Pg_datasheet, A_vent_design)
    dP_filter_grid = np.linspace(0, dP * 0.97, 300)
    # effective driving pressure after filter loss; solve t at reduced dP
    dP_eff = dP - dP_filter_grid
    t_eff = (3 * eta_exudate_hi * L_chamber**2 / h_design**2
             + Rv * L_chamber) / dP_eff
    ax.plot(dP_filter_grid, t_eff, "darkgreen", lw=2)
    ax.axhline(30, color="k", ls=":", lw=1.2, label="30 s spec")
    ax.axvline(dP * 0.3, color="orange", ls="--", lw=1.2,
               label=r"30% of $\Delta P_{cap}$ (suggested filter budget)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Filter pressure drop $\Delta P_{filter}$ (Pa)  [Layer 1]")
    ax.set_ylabel("$t_{fill}$ (s)")
    ax.set_title(f"B  Coupling to Layer 1 filter, h={h_design*1e3:.3f} mm")
    ax.legend(fontsize=7)

    fig.suptitle("Layer 2: Vent sizing and filter-pressure-budget coupling "
                 "(capillary_fill.py)", fontsize=11)
    plt.tight_layout()
    out = FIGDIR / "fig_L2_2_vent_sizing.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    summarise()
    make_figure_1()
    make_figure_2()

# =====================================================================
# LIMITATIONS (see report Sec. X for full discussion)
# =====================================================================
# 1. The Lucas-Washburn coefficient is taken verbatim from contents.md
#    (factor of 3 in t_fill = 3*eta*L^2/(h^2*dP_cap)). The classical
#    parallel-plate Poiseuille derivation for h << w gives a coefficient
#    of 12, not 3, which would make every free-fill time in this file
#    4x LONGER than reported. This has NOT been re-derived or corrected
#    here; contents.md's Sec. 2.2 formula is used as the authoritative
#    device-document specification, as Layer 3 and Layer 15 use formulas
#    handed down from earlier layers. All conclusions in this report
#    that depend on the venting/viscosity correction being the dominant
#    correction (rather than free-fill time alone) are robust to this
#    factor either way, since even a 4x free-fill correction keeps
#    every free-fill number in Sec. 2.2 far below the 30 s spec.
# 2. The vent-resistance model (Sec. 2.3, R_vent) is a first-order
#    "series resistance" construction, not a rigorously solved coupled
#    liquid/gas free-boundary problem. It correctly recovers L_stall in
#    the R_vent -> infinity limit is NOT enforced (that is a separate,
#    already-validated closed form); it is intended for the *finite*,
#    well-vented regime only.
# 3. Contact angles (theta_design=30 deg) are a target, not a measured
#    value for the fabricated gold/probe-SAM floor, PDMS/PETG/PMMA lid,
#    or the specific vent-membrane bonding process. See fill_uniformity.py
#    and report Sec. IV for the shelf-life / degradation treatment.
