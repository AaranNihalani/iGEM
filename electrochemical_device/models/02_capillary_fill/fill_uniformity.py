"""
fill_uniformity.py  --  Layer 2 (Cartridge Inlet and Capillary Fill)
=======================================================================
Eton iGEM 2026, Dry Lab.

Implements contents.md Sec. 2.6: a lateral (width-wise) model of the
fill front to test whether a floor wettability pattern (e.g. a gold
electrode trace narrower than the full 5 mm chamber width, bordered by
less-clean polymer/PCB margins) causes the fill front to tilt and trap
an air pocket against a side wall or in the wake of a "fast" region.

Modelling approach ("independent parallel capillaries", stated as a
limitation in Sec. X of the report): the 5 mm width is discretised into
Ny thin lateral strips. Each interior strip is treated as its own
1D Lucas-Washburn capillary whose driving pressure comes from the
FLOOR and LID surfaces only (no side-wall term for an interior strip --
its "walls" are the neighbouring liquid, not a solid surface):

    dP_cap(y) = gamma * (cos[theta_floor(y)] + cos[theta_lid]) / h

Only the two true edge strips (y=0, y=w) pick up the full side-wall
term from contents.md's Sec. 2.1 formula. This lateral decomposition
deliberately ignores lateral pressure equalisation between strips
(no cross-flow), which makes it a *worst-case* (conservative) estimate
of non-uniformity -- a real front would partially self-heal via lateral
flow, which this model cannot capture.

Two floor wettability scenarios are compared:
  (a) uniform, well-cleaned floor (theta_floor = theta_design everywhere)
  (b) manufacturing-defect scenario: a central clean gold/SAM band
      (fraction f_gold of the width) flanked by two less-clean polymer
      margins (theta_margin, representing solder-mask/PCB laminate that
      missed full O2-plasma exposure near the panel edge)

Run directly to print the numeric summary and regenerate
figures/fig_L2_3_fill_uniformity.pdf.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Geometry / fluid properties (kept consistent with capillary_fill.py)
# ----------------------------------------------------------------------
L_chamber = 20e-3      # m
w_chamber = 5e-3       # m
h_design = 0.375e-3    # m

gamma_wound = 0.0575       # N/m, midpoint of 50-65 mN/m contents.md range
eta_exudate_hi = 30.8e-3   # Pa s, worst-case (slowest) wound-exudate viscosity
                           # -- conservative choice for a defect-risk analysis

theta_design_deg = 30.0    # deg, well-cleaned surfaces (design target)
theta_lid_deg = 30.0       # deg, lid held at design value throughout

Ny = 200                   # number of lateral strips across the width

# ----------------------------------------------------------------------
# Local capillary pressure
# ----------------------------------------------------------------------
def dP_interior(theta_floor_deg, theta_lid_deg=theta_lid_deg, h=h_design,
                 gamma=gamma_wound):
    """Interior-strip driving pressure: floor + lid contributions only."""
    tf = np.radians(theta_floor_deg)
    tl = np.radians(theta_lid_deg)
    return gamma * (np.cos(tf) + np.cos(tl)) / h


def dP_edge(theta_floor_deg, theta_lid_deg=theta_lid_deg, theta_wall_deg=theta_design_deg,
            h=h_design, w=w_chamber, gamma=gamma_wound):
    """True side-wall strip: full 4-surface contents.md Sec. 2.1 formula,
    approximated here as floor+lid (as interior) plus one wall term."""
    interior = dP_interior(theta_floor_deg, theta_lid_deg, h, gamma)
    return interior + gamma * np.cos(np.radians(theta_wall_deg)) / w


def theta_floor_profile(y, w=w_chamber, f_gold=0.6, theta_gold=25.0,
                         theta_margin=75.0):
    """Manufacturing-defect scenario: clean gold/SAM band of fractional
    width f_gold centred in the channel, flanked by less-clean polymer
    margins. theta_margin < 90 deg (not a total wetting failure) but
    well above the design target."""
    margin = (1 - f_gold) / 2 * w
    is_margin = (y < margin) | (y > w - margin)
    return np.where(is_margin, theta_margin, theta_gold)


def front_position(t, dP, h=h_design, eta=eta_exudate_hi):
    """Per-strip free Lucas-Washburn front position (contents.md Sec. 2.2
    factor-3 form), clipped at zero for dP<=0 (locally stalled strip)."""
    dP = np.maximum(dP, 0.0)
    return np.sqrt(h**2 * dP * t / (3.0 * eta))


def simulate(theta_floor_of_y, w=w_chamber, Ny=Ny, h=h_design, eta=eta_exudate_hi,
             gamma=gamma_wound):
    """Return y-grid, local dP_cap(y), and a front_position(t) callable
    giving L(y, t) for the whole strip array."""
    y = (np.arange(Ny) + 0.5) / Ny * w
    theta_floor = theta_floor_of_y(y)
    dP = dP_interior(theta_floor, theta_lid_deg, h, gamma)
    # true edge strips (first/last) get the extra wall term
    dP[0] = dP_edge(theta_floor[0], theta_lid_deg, theta_design_deg, h, w, gamma)
    dP[-1] = dP_edge(theta_floor[-1], theta_lid_deg, theta_design_deg, h, w, gamma)

    def L_of_t(t):
        return front_position(t, dP, h, eta)

    return y, theta_floor, dP, L_of_t


def uniformity_index(L_row):
    """(L_max - L_min) / mean(L) across the width at a fixed time."""
    if np.all(L_row <= 0):
        return np.nan
    return (L_row.max() - L_row.min()) / L_row.mean()


def summarise():
    print("=" * 78)
    print("LAYER 2 -- Fill uniformity across chamber width: numeric summary")
    print("=" * 78)

    # Scenario A: uniform, well-cleaned floor
    y, theta_u, dP_u, Lfun_u = simulate(
        lambda y: np.full_like(y, theta_design_deg))
    # Scenario B: manufacturing-defect floor (60% clean gold band, 75 deg margins)
    y, theta_d, dP_d, Lfun_d = simulate(
        lambda y: theta_floor_profile(y, f_gold=0.6, theta_gold=25.0, theta_margin=75.0))

    print(f"\nLateral discretisation: Ny={Ny} strips across w={w_chamber*1e3:.1f} mm")
    print(f"\nScenario A (uniform, theta_floor={theta_design_deg} deg everywhere):")
    print(f"  dP_cap range across width: [{dP_u.min():.1f}, {dP_u.max():.1f}] Pa "
          f"(edge strips slightly higher: wall term)")

    print(f"\nScenario B (60% clean gold band theta=25 deg, "
          f"20% margins each side theta=75 deg):")
    print(f"  dP_cap range across width: [{dP_d.min():.1f}, {dP_d.max():.1f}] Pa")
    print(f"  Gold-band dP_cap = {dP_interior(25.0):.1f} Pa   "
          f"Margin dP_cap = {dP_interior(75.0):.1f} Pa   "
          f"ratio = {dP_interior(25.0)/dP_interior(75.0):.2f}x")

    # time to fill the chamber length for fastest vs slowest strip
    t_chamber_fast = 3 * eta_exudate_hi * L_chamber**2 / (h_design**2 * dP_d.max())
    t_chamber_slow = 3 * eta_exudate_hi * L_chamber**2 / (h_design**2 * dP_d.min())
    print(f"\n  Time for FASTEST strip (gold band) to reach x=20 mm: "
          f"{t_chamber_fast:.3f} s")
    print(f"  Time for SLOWEST strip (margin)    to reach x=20 mm: "
          f"{t_chamber_slow:.3f} s")
    print(f"  Lag = {t_chamber_slow - t_chamber_fast:.3f} s "
          f"({(t_chamber_slow-t_chamber_fast)/t_chamber_fast*100:.0f}% slower) "
          f"-- the margin strips are still advancing well after the gold-band "
          f"front has already reached the far wall, i.e. the front is "
          f"'V-shaped' / concave, which is the classic geometry for trapping "
          f"an air pocket against the two side walls if a lid or end-wall "
          f"seals before the margins catch up.")

    # uniformity index at several snapshot times (using scenario B)
    print(f"\n  Uniformity index (Lmax-Lmin)/Lmean at fixed times (scenario B):")
    for tt in [0.05, 0.1, 0.3, 0.6, 1.0, 1.8]:
        row = Lfun_d(tt)
        ui = uniformity_index(np.minimum(row, L_chamber))
        print(f"    t={tt:4.2f} s  ->  uniformity index = {ui:.3f}  "
              f"(Lmin={row.min()*1e3:.2f} mm, Lmax={min(row.max(),L_chamber)*1e3:.2f} mm)")

    stall_risk = np.any(dP_d <= 0)
    print(f"\n  Local wetting-failure stall risk (any dP_cap<=0 across width)? "
          f"{'YES -- FLAG' if stall_risk else 'No, all strips have theta<90 deg '
           'so all strips advance (slowly, not stalled)'}")

    return y, theta_u, dP_u, Lfun_u, theta_d, dP_d, Lfun_d


def make_figure():
    y, theta_u, dP_u, Lfun_u, theta_d, dP_d, Lfun_d = summarise()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel A: contact-angle / dP_cap profile across width, both scenarios
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(y * 1e3, theta_u, "g--", lw=1.5, label=r"$\theta_{floor}$, scenario A (uniform)")
    ax.plot(y * 1e3, theta_d, "r-", lw=1.8, label=r"$\theta_{floor}$, scenario B (defect)")
    ax.axhline(90, color="gray", ls=":", lw=1)
    ax.set_xlabel("Lateral position y (mm)")
    ax.set_ylabel(r"Local floor contact angle $\theta_{floor}$ (deg)")
    ax2.plot(y * 1e3, dP_d, "b-.", lw=1.2, alpha=0.7, label=r"$\Delta P_{cap}(y)$, scenario B")
    ax2.set_ylabel(r"$\Delta P_{cap}(y)$ (Pa)", color="b")
    ax.set_title("A  Lateral wettability pattern")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="center right")

    # Panel B: fill-front shape (x vs y) at a few snapshot times, scenario B
    ax = axes[1]
    for tt, style in zip([0.05, 0.2, 0.6, 1.2, 1.8],
                          ["b:", "b--", "b-.", "b-", "navy"]):
        row = np.minimum(Lfun_d(tt), L_chamber) * 1e3
        ax.plot(row, y * 1e3, style, lw=1.6, label=f"t={tt:.2f} s")
    ax.axvline(L_chamber * 1e3, color="k", ls=":", lw=1)
    ax.set_xlabel("Fill front position x (mm)")
    ax.set_ylabel("Lateral position y (mm)")
    ax.set_title("B  Front shape over time (scenario B)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.invert_yaxis()

    # Panel C: uniformity index vs time, both scenarios
    ax = axes[2]
    t_grid = np.linspace(0.01, 2.2, 300)
    ui_u = [uniformity_index(np.minimum(Lfun_u(tt), L_chamber)) for tt in t_grid]
    ui_d = [uniformity_index(np.minimum(Lfun_d(tt), L_chamber)) for tt in t_grid]
    ax.plot(t_grid, ui_u, "g--", lw=1.8, label="scenario A (uniform)")
    ax.plot(t_grid, ui_d, "r-", lw=2, label="scenario B (defect)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Uniformity index $(L_{max}-L_{min})/L_{mean}$")
    ax.set_title("C  Front non-uniformity vs. time")
    ax.legend(fontsize=8)

    fig.suptitle("Layer 2: Fill uniformity across chamber width (fill_uniformity.py)",
                 fontsize=11)
    plt.tight_layout()
    out = FIGDIR / "fig_L2_3_fill_uniformity.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()

# =====================================================================
# LIMITATIONS (see report Sec. X)
# =====================================================================
# 1. No lateral pressure equalisation / cross-flow between strips is
#    modelled ("independent parallel capillaries"). Real fluid would
#    partially redistribute laterally once a pressure gradient exists
#    between a fast and a slow strip, which would flatten the front
#    relative to what is shown here. The model is therefore a
#    conservative (worst-case) estimate of non-uniformity, useful for
#    flagging manufacturing risk, not for predicting the exact
#    trapped-bubble volume.
# 2. The floor-pattern geometry (60% gold band, 75 deg margins, sharp
#    step at the band edge) is illustrative of the *type* of defect
#    contents.md Sec. 2.6 describes, not a measurement of the actual
#    fabricated electrode layout or its as-manufactured contact angles.
# 3. Vent resistance is not included in this lateral model (it only
#    matters for the total fill *time*, not the *shape* of the front,
#    to leading order); see capillary_fill.py for the vent-limited time.
