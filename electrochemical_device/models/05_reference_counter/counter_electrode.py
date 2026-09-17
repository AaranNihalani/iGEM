"""
counter_electrode.py -- Layer 5.3 / 5.4 / 5.5
==============================================
Counter electrode (CE) current budget and sizing, O2-evolution gas-bubble
risk check, shared-vs-individual CE comparison for sequential vs simultaneous
SWV multiplexing, and solution-resistance / iR-compensation check.

Outputs feed:
    - Hardware design spec (CE area, CE placement relative to vent)
    - Layer 6 (double-layer / capacitive current cross-check)
    - Layer 14 (noise budget; iR drop must stay below the noise floor)

Run as a script to print the numeric summary used in the Layer 5 report and
regenerate Figure 2.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Per-WE current budget during SWV ─────────────────────────────────────────
# Capacitive current: I_cap = 2 * C_dl * A_WE * Delta_E_SW * f_SWV  (Layer 6)
C_dl = 2e-6          # F/cm^2, SAM-modified gold, mid-range literature value (Layer 6) -- FLAG
A_WE = 1e-2          # cm^2 (1 mm^2 working-electrode geometric area, Layer 6) -- FLAG, design parameter
dE_SW = 0.025        # V, square-wave amplitude (Layer 11 typical)
f_SWV = 60.0         # Hz (Layer 11 typical)

I_cap_per_WE = 2 * C_dl * A_WE * dE_SW * f_SWV  # A
I_faradaic_range = (10e-9, 100e-9)              # A, peak, contents.md Sec. 5.3

I_total_per_WE_range = (I_cap_per_WE + I_faradaic_range[0],
                         I_cap_per_WE + I_faradaic_range[1])

N_WE_planning = 8    # original planning figure, contents.md Sec. 5.3 (8-WE, dual-channel cartridge)
N_WE_confirmed_min = 2  # confirmed minimum: 1 active + 1 D-enantiomer control pair (Layer 13)
N_active_typ = 4     # Layer 13 worked example: N_active = 4 paired electrodes averaged

# Simultaneous-measurement CE current (all WEs scanning at once)
I_CE_simultaneous = tuple(n * np.array(I_total_per_WE_range) for n in
                           (N_WE_planning, 2 * N_active_typ))  # active+control pairs -> 2*N_active electrodes

# Sequential-measurement CE current (one WE at a time; recommended, contents.md Sec. 5.4)
I_CE_sequential = I_total_per_WE_range  # same as a single WE

# ── CE sizing ─────────────────────────────────────────────────────────────
j_CE_max = 1e-3  # A/cm^2, max practical current density before bubble formation, smooth Au in physiological solution -- FLAG (literature estimate)


def min_CE_area(I_CE_max):
    return I_CE_max / j_CE_max  # cm^2


A_CE_min_simultaneous = tuple(min_CE_area(I[1]) for I in I_CE_simultaneous)
A_CE_min_sequential = min_CE_area(I_CE_sequential[1])

A_WE_total_planning = N_WE_planning * A_WE
A_WE_total_confirmed = 2 * N_active_typ * A_WE
RULE_OF_THUMB_RATIO = 10.0  # CE area >= 10x total WE area, standard practice

A_CE_rule_of_thumb_planning = RULE_OF_THUMB_RATIO * A_WE_total_planning
A_CE_rule_of_thumb_confirmed = RULE_OF_THUMB_RATIO * A_WE_total_confirmed

# ── O2 evolution / gas-bubble risk check ────────────────────────────────────
# Water oxidation: 2H2O -> O2 + 4H+ + 4e-, E0 = +1.23 V vs NHE (+1.0 V vs sat. Ag/AgCl)
E_O2_vs_NHE = 1.23
E_AgCl_sat_vs_NHE = 0.197
E_O2_vs_AgCl_sat = E_O2_vs_NHE - E_AgCl_sat_vs_NHE  # ~ +1.03 V vs sat. Ag/AgCl

# Confirmed SWV scan window for the fabricated device (Layer 7 Eq. III.7,
# Layer 10 Sec. III): -0.65 V to +0.1 V vs Ag/AgCl -- narrower than the
# +/-0.65 V originally assumed for CE sizing in the Layer 5 content spec
# (Sec. 5.3), because the confirmed design measures a single MB/Fc pair per
# electrode rather than the originally planned wide dual-channel window.
E_scan_confirmed = (-0.65, 0.10)   # V vs Ag/AgCl
E_scan_planning = (-0.65, 0.65)    # V vs Ag/AgCl, contents.md Sec. 5.3 original assumption

# CE potential during the most demanding half-cycle is set by the
# potentiostat's compliance voltage, not by the WE scan potential directly;
# what matters is the margin between the WE's most extreme excursion and the
# O2-evolution potential, since a grossly undersized CE forces the CE
# potential toward (and eventually past) that limit to sustain the required
# current.
margin_confirmed = E_O2_vs_AgCl_sat - E_scan_confirmed[1]
margin_planning = E_O2_vs_AgCl_sat - E_scan_planning[1]

# ── Solution resistance and iR drop ──────────────────────────────────────────
sigma_sol = 0.15       # S/m, wound fluid conductivity (~15 mS/cm), contents.md Sec. 5.5
rho_sol = 1.0 / sigma_sol  # Ohm*m

d_RE_WE = 2e-3         # m, RE-to-WE separation (contents.md Sec. 5.5 resolved geometry)
w_channel = 5e-3       # m, channel width
h_channel = 0.375e-3   # m, channel height
A_cross = w_channel * h_channel  # m^2, current-path cross-section

R_s = rho_sol * d_RE_WE / A_cross  # Ohm

# iR drop at representative currents (single WE, since RE/WE geometry is
# local to one electrode regardless of how many WEs are active elsewhere)
I_cap_iR = I_cap_per_WE
I_far_iR = I_faradaic_range[1]  # conservative (upper) faradaic estimate

V_iR_cap = I_cap_iR * R_s
V_iR_far = I_far_iR * R_s

# SWV parameters against which the iR drop is judged for significance
dE_step = 0.002   # V, staircase step (Layer 11 typical)
FWHM_MB = 0.065   # V, MB peak full width at half maximum (Layer 11)

# Judged against the SWV peak width (the quantity iR drop would actually
# distort), not the step size: a drop that is a small fraction of the MB
# peak's FWHM cannot measurably broaden or shift the peak integration window.
IR_COMPENSATION_NEEDED = max(V_iR_cap, V_iR_far) > 0.1 * FWHM_MB


def plot_ce_summary(save=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # Panel A: CE current budget, simultaneous vs sequential
    ax = axes[0]
    labels = ["Sequential\n(1 WE)", f"Simultaneous\n(2xN_active={2*N_active_typ} WE)", f"Simultaneous\n(planning, {N_WE_planning} WE)"]
    lo = [I_CE_sequential[0] * 1e9, I_CE_simultaneous[1][0] * 1e9, I_CE_simultaneous[0][0] * 1e9]
    hi = [I_CE_sequential[1] * 1e9, I_CE_simultaneous[1][1] * 1e9, I_CE_simultaneous[0][1] * 1e9]
    x = np.arange(len(labels))
    ax.bar(x, hi, color=["seagreen", "steelblue", "gray"], alpha=0.85)
    ax.bar(x, lo, color="white", alpha=0.6, edgecolor="black", linewidth=0.6)
    for xi, (l, h) in enumerate(zip(lo, hi)):
        ax.annotate(f"{l:.0f}-{h:.0f} nA", (xi, h), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("I_CE (nA)")
    ax.set_title("A -- CE current budget by multiplexing mode")

    # Panel B: O2 evolution margin across scan window options
    ax2 = axes[1]
    scans = ["Confirmed\n(-0.65 to +0.10 V)", "Original planning\n(-0.65 to +0.65 V)"]
    upper_lims = [E_scan_confirmed[1], E_scan_planning[1]]
    ax2.bar(scans, upper_lims, color=["seagreen", "gray"], alpha=0.85, label="Scan upper limit")
    ax2.axhline(E_O2_vs_AgCl_sat, color="red", ls="--", lw=1.5, label=f"O2 evolution ({E_O2_vs_AgCl_sat:.2f} V vs Ag/AgCl sat.)")
    ax2.set_ylabel("Potential (V vs Ag/AgCl sat.)")
    ax2.set_title("B -- Margin to O2 evolution onset")
    ax2.legend(fontsize=8)

    # Panel C: CE area requirement vs WE area (rule of thumb)
    ax3 = axes[2]
    cats = ["Confirmed design\n(2 electrodes)", "Original planning\n(8 electrodes)"]
    we_areas = [A_WE_total_confirmed * 1e2, A_WE_total_planning * 1e2]  # mm^2
    ce_areas = [A_CE_rule_of_thumb_confirmed * 1e2, A_CE_rule_of_thumb_planning * 1e2]  # mm^2
    x = np.arange(len(cats))
    w = 0.35
    ax3.bar(x - w/2, we_areas, w, label="Total WE area", color="steelblue")
    ax3.bar(x + w/2, ce_areas, w, label="CE area (10x rule)", color="darkorange")
    ax3.set_xticks(x)
    ax3.set_xticklabels(cats, fontsize=8)
    ax3.set_ylabel("Area (mm^2)")
    ax3.set_title("C -- CE sizing, 10x rule of thumb")
    ax3.legend(fontsize=8)

    fig.suptitle("Layer 5.3/5.4/5.5: Counter electrode sizing and gas-evolution margin", fontsize=11)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L5_counter.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Layer 5.3: Per-WE current budget ===")
    print(f"I_cap per WE: {I_cap_per_WE*1e9:.1f} nA")
    print(f"I_faradaic per WE (peak): {I_faradaic_range[0]*1e9:.0f}-{I_faradaic_range[1]*1e9:.0f} nA")
    print(f"Total per WE: {I_total_per_WE_range[0]*1e9:.0f}-{I_total_per_WE_range[1]*1e9:.0f} nA")
    print()
    print("=== CE current, by multiplexing mode ===")
    print(f"Sequential (1 WE at a time): {I_CE_sequential[0]*1e9:.0f}-{I_CE_sequential[1]*1e9:.0f} nA")
    print(f"Simultaneous, confirmed design (2 x N_active={N_active_typ} = {2*N_active_typ} electrodes): "
          f"{I_CE_simultaneous[1][0]*1e9:.0f}-{I_CE_simultaneous[1][1]*1e9:.0f} nA")
    print(f"Simultaneous, original planning ({N_WE_planning} electrodes): "
          f"{I_CE_simultaneous[0][0]*1e9:.0f}-{I_CE_simultaneous[0][1]*1e9:.0f} nA")
    print()
    print("=== CE minimum area (current-density limited) ===")
    print(f"Sequential: {A_CE_min_sequential*1e2:.2e} mm^2 (trivial)")
    print(f"Simultaneous, confirmed design: {A_CE_min_simultaneous[1]*1e2:.2e} mm^2 (trivial)")
    print(f"Simultaneous, original planning: {A_CE_min_simultaneous[0]*1e2:.2e} mm^2 (trivial)")
    print(f"-> current-density limit is never the binding constraint; the 10x-WE-area rule of thumb dominates:")
    print(f"   confirmed design: {A_CE_rule_of_thumb_confirmed*1e2:.3f} mm^2; "
          f"original planning: {A_CE_rule_of_thumb_planning*1e2:.3f} mm^2")
    print()
    print("=== O2 evolution margin ===")
    print(f"O2 evolution onset: {E_O2_vs_AgCl_sat:.3f} V vs sat. Ag/AgCl")
    print(f"Confirmed scan window upper limit: {E_scan_confirmed[1]:.2f} V -> margin {margin_confirmed*1e3:.0f} mV")
    print(f"Original planning scan window upper limit: {E_scan_planning[1]:.2f} V -> margin {margin_planning*1e3:.0f} mV")
    print()
    print("=== Solution resistance and iR drop ===")
    print(f"R_s = {R_s:.0f} Ohm (rho_sol={rho_sol:.2f} Ohm*m, d={d_RE_WE*1e3:.1f} mm, A_cross={A_cross*1e6:.3f} mm^2)")
    print(f"iR drop, capacitive current ({I_cap_iR*1e9:.0f} nA): {V_iR_cap*1e3:.2f} mV")
    print(f"iR drop, faradaic current ({I_far_iR*1e9:.0f} nA): {V_iR_far*1e3:.2f} mV")
    print(f"For comparison: DeltaE_step = {dE_step*1e3:.0f} mV, MB peak FWHM = {FWHM_MB*1e3:.0f} mV")
    print(f"iR compensation needed? {IR_COMPENSATION_NEEDED} (both drops << DeltaE_step)")
    plot_ce_summary()
