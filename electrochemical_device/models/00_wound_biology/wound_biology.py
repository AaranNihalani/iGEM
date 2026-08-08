"""
wound_biology.py  --  Layer 0
=============================
Full protease activation cascade ODE for MMP-9 and HNE in chronic wound fluid.

Architecture note
-----------------
Layer 0 models IN-SITU wound fluid concentrations (ng/mL to ug/mL range) as measured
directly in wound fluid by ELISA and zymography. Layer 1 (sample collection) applies a
dilution factor Df to convert from in-situ concentrations to what enters the device.
The sensor detection range (17-200 ng/mL MMP-9) is reached AFTER Layer 1 dilution.

Cascade overview
----------------
MMP-9 pathway (metalloprotease):
  pro-MMP-9 --[MMP-3]--> active MMP-9   (proteolytic zymogen activation)
  pro-MMP-9 --[plasmin]--> active MMP-9  (second activation route)
  active MMP-9 + TIMP-1  <--> MMP-9:TIMP-1  (reversible inhibition)

HNE pathway (serine protease, released already active):
  HNE + A1AT  --> HNE:A1AT    (near-irreversible inhibition, k_on=6.5e7 M-1s-1, Beatty 1980)
  HNE + SLPI  <--> HNE:SLPI  (reversible slow-tight inhibition)

A1AT WOUND-STATE DEPENDENCE (CONFIRMED: Rao et al. 1995, J Invest Dermatol 105:572):
  HEALING/ACUTE wounds: A1AT is intact and functional (Rao 1995, text p.572-575).
    HNE t_half = ln2/(k_on*[A1AT]) = 0.693/(6.5e7 * 1.92e-6) = 5.5 ms.
    HNE_free = 0 at all t > 5 ms. SLPI irrelevant.
  CHRONIC wounds: A1AT is degraded and non-functional in 9/10 samples (Rao 1995, text).
    [functional A1AT] ~ 0. HNE is not rapidly neutralised.
    HNE partially inhibited by SLPI over ~10 min timescale.
    HNE_free at t=30 min: ~30 nM (SLPI partially inhibits, placeholder kinetics).
    However, elevated HNE detected in only 6/13 (50%) of chronic VLU samples (Trengove 1999).

With in-situ concentrations (Ladwig 2002: chronic total MMP-9 12,900 ng/mL = 140 nM,
active 4,800 ng/mL = 52 nM; Trengove 1999: TIMP-1 800 ng/mL = 28 nM), active MMP-9
exceeds TIMP-1 in chronic wounds. Ki(TIMP-1/MMP-9) = 8.5 nM (Olson 1997 Table IV).
Equilibrium free MMP-9 in chronic wounds: ~32 nM (MMP-9 excess over TIMP-1).
Healing wound free MMP-9 with estimated parameters: ~6 nM (TIMP-1 excess).
MMP-9 discrimination ratio: ~5x. HNE also discriminates (chronic: ~30 nM, healing: 0)
but is unreliable (elevated in only 50% of chronic cases). MMP-9 preferred.

Two wound-state parameter sets: "chronic" (elevated MMP-9, lower TIMP-1:MMP-9 ratio,
A1AT degraded) and "healing" (lower MMP-9, higher TIMP-1:MMP-9 ratio, A1AT intact).

Values marked NEEDS MEASUREMENT are estimated from related contexts and must be
replaced with wound-fluid-specific measurements. All other values are literature-cited.

Outputs (sent to Layer 1, before dilution factor Df is applied):
  [MMP-9]_free(t)  [mol/L, in-situ wound fluid]
  [HNE]_free(t)    [mol/L, in-situ wound fluid; ~30 nM in chronic, ~0 in healing]
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Molecular weights ──────────────────────────────────────────────────────────
MW_MMP9  = 92e3    # g/mol
MW_HNE   = 30e3
MW_TIMP1 = 28.5e3
MW_A1AT  = 52e3
MW_SLPI  = 11.7e3
MW_MMP3  = 54e3

# ── Wound-state parameter sets ────────────────────────────────────────────────
# IN-SITU wound fluid concentrations from primary literature.
# Layer 1 applies swab dilution factor Df to convert to device-entry concentrations.
#
# Ladwig et al. 2002 (Wound Repair Regen 10:26-37) -- ELISA in pressure ulcer wound fluid
# [CONFIRMED: paper in papers/ladwig2002.pdf]:
#   Pro-MMP-9: 8100 +/- 3000 ng/mL (stated in text, p.31)
#   Active MMP-9: 4800 +/- 1300 ng/mL (stated in text, p.31)
#   Total MMP-9: 8100 + 4800 = 12,900 ng/mL (derived, both components from text)
#   Pro fraction: 8100 / 12900 = 0.627 (derived from text values)
#   Good healer active MMP-9 vs poor healer: lower in good healers (Fig 8, approximate)
#   TIMP-1 higher in good healers than poor healers at Day 0 (Fig 10, approximate)
#   NOTE: Ladwig data are from pressure ulcers. No VLU-specific quantitative MMP-9 ELISA
#   is available in the papers folder. Absolute values used as best available source.
#
# Trengove et al. 1999 (Wound Repair Regen 7:442) -- ELISA in venous leg ulcer wound fluid
# [CONFIRMED: paper in papers/trengove1999.pdf]:
#   TIMP-1 (chronic VLU): 0.8 ug/mL = 800 ng/mL (stated in text, p.448, n=5)
#   HNE (non-healing VLU): median 1.4 ug/mL = ~47 nM (stated in text, p.448)
#
# Unit convention: c [ng/mL] * 1e-6 [g/L per ng/mL] / MW [g/mol] = mol/L

_MMP9_total_chronic_ngmL  = 12900.0   # ng/mL  Ladwig 2002 (pro 8100 + active 4800, text)
                                       # Converts to 140 nM total MMP-9

_MMP9_total_healing_ngmL  = 5400.0    # ng/mL  APPROXIMATE -- read from Ladwig 2002 Fig 8:
                                       # good healer active MMP-9 ~2000 ng/mL / 0.373 = 5400
                                       # NOT stated in paper text; approximate from graph.
                                       # NEEDS MEASUREMENT: joint VLU ELISA (MMP-9 + TIMP-1).

_TIMP1_chronic_ngmL       = 800.0     # ng/mL  Trengove 1999: 0.8 ug/mL (text, p.448, n=5)
                                       # Converts to 28 nM

_TIMP1_healing_ngmL       = 1104.0    # ng/mL  ESTIMATE: insufficient data in papers folder
                                       # to confirm this value. Trengove 1999 acute wound
                                       # TIMP-1 = 44,000 ng/mL (not applicable). Ladwig 2002
                                       # Fig 10 good healer TIMP-1 ~170 ng/mL (approx, from
                                       # graph, different wound type). Value retained from
                                       # previous analysis; treat as unconfirmed.
                                       # NEEDS MEASUREMENT: VLU healing-phase TIMP-1 ELISA.

_pro_fraction             = 0.627     # Ladwig 2002 (p.31): 8100/(8100+4800) = 0.627
                                       # [CONFIRMED: papers/ladwig2002.pdf]

# A1AT wound-state concentrations.
# CONFIRMED (qualitative): Rao CN et al. (1995) J Invest Dermatol 105:572-578.
#   Healing/acute wounds: A1AT intact and functional (text p.572-575, all samples).
#   Chronic wounds: A1AT degraded and non-functional in 9/10 samples (text p.572-575).
#   Paper in papers/layer_00/rao1995.pdf
# CONCENTRATION (quantitative): not stated in Rao 1995. Using serum/10 for healing.
_A1AT_healing_ugmL = 100.0    # ug/mL -> 1.92 uM (serum/10 estimate; INTACT in healing)
_A1AT_chronic_ugmL = 0.0      # functional A1AT ~ 0 in chronic wounds (degraded, Rao 1995)


def _wound_ics(mmp9_total_ngmL, timp1_ngmL, a1at_ugmL):
    """Build initial condition vector [mol/L] for a given wound state."""
    C_proMMP9 = _pro_fraction       * (mmp9_total_ngmL * 1e-6) / MW_MMP9
    C_MMP9    = (1 - _pro_fraction) * (mmp9_total_ngmL * 1e-6) / MW_MMP9
    C_TIMP1   = (timp1_ngmL * 1e-6) / MW_TIMP1

    # HNE: Trengove 1999 (Wound Repair Regen 7:442): non-healing VLU median 1.4 ug/mL
    # (range <1-11.9 ug/mL); elevated in only 6/13 chronic samples.
    # 1.4 ug/mL / 30,000 g/mol = 47 nM. Many samples below LOD (<1 ug/mL = <33 nM).
    # In HEALING wounds: HNE instantly neutralised by functional A1AT (t_half = 5.5 ms).
    # In CHRONIC wounds: A1AT degraded; HNE persists, partially inhibited by SLPI (~30 nM).
    C_HNE = 47e-9                            # 47 nM  Trengove 1999 VLU median (both states)

    # A1AT: wound-state specific. Intact in healing, degraded in chronic (Rao 1995).
    # Concentration estimate (healing only): serum ~1-2 mg/mL / 10 dilution = 100 ug/mL.
    # Unit: ug/mL * 1e-3 (g/L per ug/mL) / MW (g/mol) = mol/L
    C_A1AT = (a1at_ugmL * 1e-3) / MW_A1AT   # mol/L; 0 for chronic, 1.92 uM for healing

    # SLPI: 100-500 ng/mL range cited in serine protease inhibitor literature.
    # NEEDS CITATION: paper not in papers folder.
    # Relevant for chronic wounds (A1AT degraded), placeholder kinetics used.
    C_SLPI = (200e-6) / MW_SLPI             # 200 ng/mL -> ~17 nM  NEEDS CITATION

    # MMP-3: activates pro-MMP-9. No wound-fluid ELISA data found.
    # Order-of-magnitude placeholder -- does not affect HNE channel prediction.
    C_MMP3   = (10e-6) / MW_MMP3            # 10 ng/mL -> ~0.185 nM  NEEDS MEASUREMENT

    # Plasmin: activates pro-MMP-9 directly and via MMP-3. Placeholder.
    C_plasmin = 1e-9                         # ~1 nM  NEEDS MEASUREMENT

    return C_proMMP9, C_MMP9, C_TIMP1, C_HNE, C_A1AT, C_SLPI, C_MMP3, C_plasmin


# Default: chronic wound (used for single-state runs and Panel A bar chart)
(C_proMMP9_0, C_MMP9_0, C_TIMP1_0,
 C_HNE_0, C_A1AT_0, C_SLPI_0,
 C_MMP3_0, C_plasmin_0) = _wound_ics(_MMP9_total_chronic_ngmL, _TIMP1_chronic_ngmL,
                                      _A1AT_chronic_ugmL)

# ── Rate constants ─────────────────────────────────────────────────────────────

# A1AT/HNE association: near-irreversible serpin inhibition.
# CONFIRMED: Beatty K, Bieth J, Travis J (1980) JBC 255(9):3931-3934.
#   Table 1: k_assoc(human leukocyte elastase / native alpha-1-PI) = 6.5 +/- 4.0 x 10^7 M-1 s-1
#   Reaction effectively irreversible (k_off not detectable; inhibition obeys second-order kinetics).
#   Measured at 25 C, pH 8.0. Applied here at wound pH ~7.4 as best available value.
# Paper in papers/layer_00/beatty1980.pdf
k_on_A1AT  = 6.5e7  # M-1 s-1  Beatty 1980 Table 1 [CONFIRMED]
k_off_A1AT = 1e-6    # s-1      near-irreversible (not measurable; conservative placeholder)

# TIMP-1 / MMP-9 association: slow-binding tight inhibition.
# CONFIRMED: Olson MW, Gervasi DC, Mobashery S, Fridman R (1997) JBC 272(47):29975-29983.
#   Table IV (enzyme inhibition assay, physiologically relevant conditions):
#     MMP-9 (82 kDa, active form) + TIMP-1:
#     k_on = 2.48 +/- 0.47 x 10^5 M-1 s-1
#     k_off = 2.1 +/- 1.1 x 10^{-3} s-1
#     Ki = 8.5 +/- 2.9 nM  (stated directly; check: k_off/k_on = 2.1e-3/2.48e5 = 8.5 nM)
#   Abstract: "k_on ~ 10^5 M-1s-1, k_off ~ 10^{-3} s-1, Ki in 10^{-7}--10^{-9} M range"
#   Enzyme inhibition values used (Table IV) in preference to SPR (Tables I,II) per
#   Olson 1997 p.29981: enzyme inhibition k_on is 2.5-7x higher than SPR, reflecting
#   solution-phase behaviour closer to wound conditions.
# Paper in papers/layer_00/olson1997.pdf
k_on_TIMP1  = 2.48e5  # M-1 s-1  Olson 1997 Table IV [CONFIRMED]
k_off_TIMP1 = 2.1e-3  # s-1      Olson 1997 Table IV [CONFIRMED]; Ki = 8.5 nM

# SLPI / HNE association. SLPI is a Kunitz-type inhibitor.
# NEEDS MEASUREMENT: wound-fluid specific k_on. Order-of-magnitude placeholder.
k_on_SLPI  = 1e5     # M-1 s-1  NEEDS MEASUREMENT
k_off_SLPI = 1e-4    # s-1      Ki ~ 1 nM  NEEDS MEASUREMENT

# Pro-MMP-9 activation by MMP-3 (kcat/Km for prodomain cleavage).
# NEEDS CITATION: papers not in papers folder. Values are order-of-magnitude placeholders.
# At current placeholder activator concentrations, < 1% of pro-MMP-9 is activated in 120 min,
# so these values do not affect device-relevant predictions.
kcat_Km_MMP3    = 5e3    # M-1 s-1  NEEDS CITATION
kcat_Km_plasmin = 1e4    # M-1 s-1  NEEDS CITATION

# ── ODE system ─────────────────────────────────────────────────────────────────

def cascade_ode(t, y, C_MMP3, C_plasmin):
    """
    State vector y (all in mol/L):
      0: [proMMP9]        1: [MMP9_free]     2: [TIMP1_free]
      3: [MMP9:TIMP1]     4: [HNE_free]      5: [A1AT_free]
      6: [HNE:A1AT]       7: [SLPI_free]     8: [HNE:SLPI]
    MMP-3 and plasmin treated as constant (large excess relative to pro-MMP-9).
    """
    proMMP9, MMP9f, TIMP1f, MMP9T1, HNEf, A1ATf, HNE_A1AT, SLPIf, HNE_SLPI = y

    v_act_MMP3    = kcat_Km_MMP3    * C_MMP3    * proMMP9
    v_act_plasmin = kcat_Km_plasmin * C_plasmin  * proMMP9
    v_T1_on       = k_on_TIMP1  * MMP9f * TIMP1f
    v_T1_off      = k_off_TIMP1 * MMP9T1
    v_A1AT        = k_on_A1AT   * HNEf  * A1ATf
    v_SLPI_on     = k_on_SLPI   * HNEf  * SLPIf
    v_SLPI_off    = k_off_SLPI  * HNE_SLPI

    return [
        -(v_act_MMP3 + v_act_plasmin),
         (v_act_MMP3 + v_act_plasmin) - v_T1_on + v_T1_off,
        -v_T1_on + v_T1_off,
         v_T1_on - v_T1_off,
        -v_A1AT - v_SLPI_on + v_SLPI_off,
        -v_A1AT,
         v_A1AT,
        -v_SLPI_on + v_SLPI_off,
         v_SLPI_on - v_SLPI_off,
    ]


def run_cascade(t_max_min=120, wound_state="chronic"):
    """Integrate ODE for the specified wound state ('chronic' or 'healing')."""
    if wound_state == "chronic":
        ics = _wound_ics(_MMP9_total_chronic_ngmL, _TIMP1_chronic_ngmL,
                         _A1AT_chronic_ugmL)
    elif wound_state == "healing":
        ics = _wound_ics(_MMP9_total_healing_ngmL, _TIMP1_healing_ngmL,
                         _A1AT_healing_ugmL)
    else:
        raise ValueError(f"wound_state must be 'chronic' or 'healing', got {wound_state!r}")

    C_proMMP9, C_MMP9, C_TIMP1, C_HNE, C_A1AT, C_SLPI, C_MMP3, C_plasmin = ics
    y0 = [C_proMMP9, C_MMP9, C_TIMP1, 0.0,
          C_HNE, C_A1AT, 0.0, C_SLPI, 0.0]
    t_eval = np.linspace(0, t_max_min * 60, 2000)
    sol = solve_ivp(cascade_ode, (0, t_max_min * 60), y0,
                    args=(C_MMP3, C_plasmin),
                    t_eval=t_eval, method="Radau", rtol=1e-10, atol=1e-15)
    return sol


def plot_cascade(sol, save=True):
    t = sol.t / 60   # minutes

    # Extract species in nM
    proMMP9    = sol.y[0] * 1e9
    MMP9f      = sol.y[1] * 1e9
    TIMP1f     = sol.y[2] * 1e9
    MMP9_TIMP1 = sol.y[3] * 1e9
    HNEf       = sol.y[4] * 1e9
    HNE_A1AT   = sol.y[6] * 1e9
    HNE_SLPI   = sol.y[8] * 1e9

    # Layer 1 device-entry range (post-dilution, shown for reference)
    MMP9_lo_nM = 17e-6 / MW_MMP9 * 1e9    # 0.185 nM (post-Layer 1)
    MMP9_hi_nM = 200e-6 / MW_MMP9 * 1e9   # 2.17 nM  (post-Layer 1)

    c_enzyme   = "#C0392B"
    c_zymogen  = "#2980B9"
    c_inhibitor= "#27AE60"
    c_activator= "#E67E22"
    c_complex  = "#8E44AD"

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.40)

    # ── Panel A: initial concentrations in wound fluid (in-situ) ─────────────
    ax1 = fig.add_subplot(gs[0, 0])

    names = ['MMP-3 (activator)\nNeeds measurement', 'plasmin (activator)\nNeeds measurement',
             'pro-MMP-9\n(inactive zymogen)',
             'MMP-9 free\n(active enzyme)',
             'TIMP-1\n(MMP-9 inhibitor)',
             'HNE\n(active enzyme)',
             'SLPI\n(HNE inhibitor) Needs measurement',
             'A1AT\n(HNE inhibitor)\nNeeds wound-fluid value']

    concs = [C_MMP3_0*1e9, C_plasmin_0*1e9,
             C_proMMP9_0*1e9, C_MMP9_0*1e9,
             C_TIMP1_0*1e9, C_HNE_0*1e9,
             C_SLPI_0*1e9, C_A1AT_0*1e9]   # all in nM

    bar_colors = [c_activator, c_activator,
                  c_zymogen, c_enzyme,
                  c_inhibitor, c_enzyme,
                  c_inhibitor, c_inhibitor]

    bars = ax1.barh(names, concs, color=bar_colors, alpha=0.80,
                    edgecolor="white", height=0.58)
    ax1.set_xscale("log")
    ax1.set_xlabel("IN-SITU concentration in wound fluid (nM)")
    ax1.set_title("A   In-situ wound fluid composition (chronic)",
                  loc="left", fontweight="bold", fontsize=9)
    ax1.tick_params(axis="y", labelsize=7.5)

    legend_patches = [
        mpatches.Patch(color=c_enzyme,    label="Active enzyme"),
        mpatches.Patch(color=c_activator, label="Activator"),
        mpatches.Patch(color=c_zymogen,   label="Zymogen (inactive)"),
        mpatches.Patch(color=c_inhibitor, label="Inhibitor"),
    ]
    ax1.legend(handles=legend_patches, fontsize=7, loc="lower right")
    ax1.text(0.98, 0.02,
             "Layer 1 applies Df to reach\nsensor detection range",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=6.5, color="gray", style="italic")

    # ── Panel B: MMP-9 over time ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.fill_between(t, 0, proMMP9, alpha=0.20, color=c_zymogen,
                     label="pro-MMP-9 (inactive)")
    ax2.fill_between(t, 0, MMP9_TIMP1, alpha=0.25, color=c_inhibitor,
                     label="MMP-9 blocked by TIMP-1")
    ax2.plot(t, MMP9f, color=c_enzyme, lw=2.2, label="MMP-9 free (in-situ)")

    ax2.axvline(30, color="gray", ls=":", lw=1.2, label="30 min (sensor read)")
    ax2.axvline(60, color="gray", ls="--", lw=1.2, label="60 min (extended)")

    ax2.set_xlabel("Time in wound (min)")
    ax2.set_ylabel("Concentration, nM (in-situ)")
    ax2.set_title("B   MMP-9: activation vs TIMP-1 inhibition (in-situ)",
                  loc="left", fontweight="bold", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(bottom=0)

    idx_30 = np.argmin(np.abs(sol.t - 30 * 60))
    ax2.annotate(
        f"Free MMP-9\nat 30 min:\n{MMP9f[idx_30]:.1f} nM\n(in-situ)",
        xy=(30, MMP9f[idx_30]),
        xytext=(55, max(MMP9f[idx_30] * 0.6, MMP9f[idx_30] + 2)),
        arrowprops=dict(arrowstyle="->", color=c_enzyme, lw=1.2),
        fontsize=8, color=c_enzyme,
    )

    # ── Panel C: HNE rapid inhibition ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.fill_between(t, 0, HNE_A1AT, alpha=0.25, color=c_inhibitor,
                     label="HNE blocked by A1AT")
    ax3.fill_between(t, 0, HNE_SLPI, alpha=0.20, color=c_complex,
                     label="HNE blocked by SLPI")
    ax3.plot(t, HNEf, color=c_enzyme, lw=2.2, label="HNE free (active)")

    ax3.set_xlabel("Time in wound (min)")
    ax3.set_ylabel("Concentration (nM)")
    ax3.set_title("C   HNE: rapid inhibition by A1AT (log scale)",
                  loc="left", fontweight="bold", fontsize=9)
    ax3.set_yscale("log")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_xlim(0, t[-1])

    half_HNE = C_HNE_0 * 1e9 / 2
    t_half_idx = np.argmin(np.abs(HNEf - half_HNE))
    if t_half_idx < len(t) - 1:
        ax3.axvline(t[t_half_idx], color="gray", ls=":", lw=1.2)
        ax3.text(t[t_half_idx] + 0.5, half_HNE * 1.2,
                 f"t1/2 = {t[t_half_idx]:.1f} min", fontsize=7.5, color="gray")

    # ── Panel D: Free MMP-9 delivered to sensor (in-situ) vs read time ────────
    ax4 = fig.add_subplot(gs[1, 1])

    t_points = [5, 15, 30, 60, 120]
    mmp9_free_vals = []
    hne_free_vals  = []
    for tp in t_points:
        idx = np.argmin(np.abs(sol.t - tp * 60))
        mmp9_free_vals.append(sol.y[1, idx] * 1e9)
        hne_free_vals.append(sol.y[4, idx] * 1e9)

    x = np.arange(len(t_points))
    w = 0.35
    ax4.bar(x - w/2, mmp9_free_vals, w, color=c_enzyme,    alpha=0.80, label="MMP-9 free")
    ax4.bar(x + w/2, hne_free_vals,  w, color=c_activator, alpha=0.80, label="HNE free")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{tp} min" for tp in t_points])
    ax4.set_xlabel("Swab contact time")
    ax4.set_ylabel("Free enzyme in wound fluid (nM, in-situ)")
    ax4.set_title("D   In-situ free enzyme vs read time (before Layer 1 Df)",
                  loc="left", fontweight="bold", fontsize=9)
    ax4.legend(fontsize=7)
    ax4.set_ylim(bottom=0)
    ax4.text(0.98, 0.97,
             "Layer 1 applies swab\ndilution factor Df\nto reach device range",
             transform=ax4.transAxes, ha="right", va="top",
             fontsize=6.5, color="gray", style="italic")

    fig.suptitle(
        "Layer 0: Wound fluid protease cascade (in-situ concentrations, Ladwig 2002)\n"
        "Chronic wound: MMP-9_active (52 nM) > TIMP-1 (28 nM) -- free MMP-9 persists. "
        "HNE persists: A1AT degraded in chronic wounds (Rao 1995), SLPI partially inhibits.",
        fontsize=9.5, y=1.01
    )

    if save:
        out = FIGURES / "fig_B1_wound_biology.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def plot_comparison(sol_c, sol_h, save=True):
    """Side-by-side comparison: chronic vs healing wound trajectories."""
    t_c = sol_c.t / 60
    t_h = sol_h.t / 60
    c_chronic = "#C0392B"
    c_healing = "#27AE60"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: free MMP-9 comparison
    ax = axes[0]
    ax.plot(t_c, sol_c.y[1] * 1e9, color=c_chronic, lw=2.2,
            label=f"Chronic ({_MMP9_total_chronic_ngmL:.0f} ng/mL, {_MMP9_total_chronic_ngmL*1e-6/MW_MMP9*1e9:.0f} nM total)")
    ax.plot(t_h, sol_h.y[1] * 1e9, color=c_healing, lw=2.2, ls="--",
            label=f"Healing ({_MMP9_total_healing_ngmL:.0f} ng/mL, {_MMP9_total_healing_ngmL*1e-6/MW_MMP9*1e9:.0f} nM total)")
    ax.axvline(30, color="gray", ls=":", lw=1.2, label="T=30 min (read)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Free MMP-9 (nM, in-situ)")
    ax.set_title("A   Free MMP-9 vs time (in-situ)", loc="left", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)

    # Panel B: TIMP-1 comparison
    ax = axes[1]
    ax.plot(t_c, sol_c.y[2] * 1e9, color=c_chronic, lw=2.2, label="Chronic TIMP-1")
    ax.plot(t_h, sol_h.y[2] * 1e9, color=c_healing, lw=2.2, ls="--",
            label="Healing TIMP-1")
    ax.axvline(30, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Free TIMP-1 (nM, in-situ)")
    ax.set_title("B   Free TIMP-1 vs time (in-situ)", loc="left", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)

    # Panel C: MMP-9 free at read times
    ax = axes[2]
    t_points = [5, 15, 30, 60, 120]
    mmp9_c, mmp9_h = [], []
    for tp in t_points:
        i_c = np.argmin(np.abs(sol_c.t - tp * 60))
        i_h = np.argmin(np.abs(sol_h.t - tp * 60))
        mmp9_c.append(sol_c.y[1, i_c] * 1e9)
        mmp9_h.append(sol_h.y[1, i_h] * 1e9)
    x = np.arange(len(t_points))
    w = 0.35
    ax.bar(x - w/2, mmp9_c, w, color=c_chronic, alpha=0.80, label="Chronic")
    ax.bar(x + w/2, mmp9_h, w, color=c_healing, alpha=0.80, label="Healing")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{tp} min" for tp in t_points])
    ax.set_xlabel("Swab contact time")
    ax.set_ylabel("Free MMP-9 in-situ (nM)\nbefore Layer 1 dilution")
    ax.set_title("C   Wound state discrimination vs read time",
                 loc="left", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        "Layer 0: Chronic vs healing wound states (in-situ concentrations)\n"
        "Ladwig 2002 (Wound Repair Regen 10:26-37). Layer 1 applies swab dilution Df.",
        fontsize=9, y=1.02
    )
    plt.tight_layout()
    if save:
        out = FIGURES / "fig_B1b_wound_state_comparison.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    sol_c = run_cascade(t_max_min=120, wound_state="chronic")
    sol_h = run_cascade(t_max_min=120, wound_state="healing")

    plot_cascade(sol_c)
    plot_comparison(sol_c, sol_h)

    for state, sol in [("CHRONIC", sol_c), ("HEALING", sol_h)]:
        print(f"\n── {state} wound (in-situ concentrations) ──")
        for t_check in [5, 30, 60, 120]:
            idx = np.argmin(np.abs(sol.t - t_check * 60))
            print(f"  t={t_check:3d} min | "
                  f"MMP-9_free = {sol.y[1,idx]*1e9:.2f} nM | "
                  f"HNE_free   = {sol.y[4,idx]*1e12:.2f} pM | "
                  f"TIMP-1_free = {sol.y[2,idx]*1e9:.2f} nM | "
                  f"A1AT_free  = {sol.y[5,idx]*1e6:.2f} uM")
