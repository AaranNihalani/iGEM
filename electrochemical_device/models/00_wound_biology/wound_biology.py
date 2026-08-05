"""
wound_biology.py  --  Layer 0
=============================
Full protease activation cascade ODE for MMP-9 and HNE in chronic wound fluid.

Cascade overview
----------------
MMP-9 pathway (metalloprotease):
  pro-MMP-9 --[MMP-3]--> active MMP-9   (proteolytic zymogen activation)
  pro-MMP-9 --[plasmin]--> active MMP-9  (second activation route)
  active MMP-9 + TIMP-1  <--> MMP-9:TIMP-1  (reversible inhibition)

HNE pathway (serine protease, released already active):
  HNE + A1AT  --> HNE:A1AT    (near-irreversible inhibition, k_on=2e7 M-1s-1)
  HNE + SLPI  --> HNE:SLPI   (reversible slow-tight inhibition)

Values marked NEEDS MEASUREMENT are estimated from related contexts (serum, other
wound types) and must be replaced with wound-fluid-specific measurements.
All other values have primary literature citations.

Outputs (sent to Layer 1):
  [MMP-9]_free(t), [HNE]_free(t)
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

# ── Initial wound-fluid concentrations ────────────────────────────────────────
# Unit convention: c [ng/mL] * 1e-6 [g/L per ng/mL] / MW [g/mol] = mol/L
# For µg/mL: c [µg/mL] * 1e-3 [g/L per µg/mL] / MW [g/mol] = mol/L

# MMP-9: 17-200 ng/mL clinical range; 100 ng/mL used as representative mid-point.
# Yager et al. 1996 (J Invest Dermatol): ~70% is pro-form in chronic wound.
C_proMMP9_0 = 0.70 * (100e-6) / MW_MMP9   # 70% of 100 ng/mL  -> ~0.76 nM
C_MMP9_0    = 0.30 * (100e-6) / MW_MMP9   # 30% already active -> ~0.33 nM

# HNE: released from neutrophil granules as active enzyme.
# Trengove et al. 1999 (Wound Repair Regen): ~50-200 nM in chronic wound fluid.
C_HNE_0 = 100e-9                           # 100 nM  (CONFIRMED -- Trengove 1999)

# TIMP-1: ~100-400 ng/mL in chronic wounds (Trengove 1999 reports 200 ng/mL).
# NEEDS MEASUREMENT: confirm whether Trengove reports free or total TIMP-1.
C_TIMP1_0 = (200e-6) / MW_TIMP1            # 200 ng/mL -> ~7 nM

# A1AT (alpha-1-antitrypsin): dominant HNE inhibitor.
# Serum concentration ~1-2 mg/mL; wound fluid estimated ~10x diluted -> ~100-200 ug/mL.
# NEEDS MEASUREMENT: wound-fluid specific value (not serum-derived).
# Correct unit: 100 ug/mL = 0.1 g/L -> 0.1 / 52e3 = 1.92 uM
C_A1AT_0 = (100e-3) / MW_A1AT              # 100 ug/mL -> ~1.92 uM

# SLPI: 100-500 ng/mL in wound fluid (Ashcroft et al. 2000, JCI).
# NEEDS MEASUREMENT: Ashcroft used acute wounds; verify for chronic context.
C_SLPI_0 = (200e-6) / MW_SLPI             # 200 ng/mL -> ~17 nM

# MMP-3 (activates pro-MMP-9): concentration uncertain in chronic wounds.
# NEEDS MEASUREMENT: [MMP-3] from a chronic wound MMP panel study.
# Placeholder: 10 ng/mL (general MMP panel order-of-magnitude estimate).
C_MMP3_0 = (10e-6) / MW_MMP3              # 10 ng/mL -> ~0.185 nM  NEEDS MEASUREMENT

# Plasmin: rapidly inhibited by alpha-2-antiplasmin in vivo.
# NEEDS MEASUREMENT: wound-fluid specific plasmin concentration.
C_plasmin_0 = 1e-9                         # ~1 nM order-of-magnitude  NEEDS MEASUREMENT

# ── Rate constants ─────────────────────────────────────────────────────────────

# A1AT/HNE association: fastest among serine-protease inhibitors.
# Beatty et al. 1980 (Biochemistry); confirmed Bieth 1984 (CONFIRMED).
k_on_A1AT  = 2e7     # M-1 s-1  CONFIRMED
k_off_A1AT = 1e-6    # s-1  (Ki = k_off/k_on ~ 0.05 pM, near-irreversible -- CONFIRMED)

# TIMP-1 / MMP-9 association.
# Langton et al. 1998 (Biochemistry) reports ~2e5 M-1s-1.
# NEEDS MEASUREMENT: verify this is TIMP-1, not TIMP-2.
k_on_TIMP1  = 2e5    # M-1 s-1  NEEDS MEASUREMENT
k_off_TIMP1 = 2e-5   # s-1  (Ki = 0.1 nM -- Nagase & Woessner 1999 CONFIRMED)

# SLPI / HNE association. SLPI is a slow tight-binding inhibitor.
# NEEDS MEASUREMENT: k_on for SLPI/HNE in wound fluid.
k_on_SLPI  = 1e5     # M-1 s-1  NEEDS MEASUREMENT
k_off_SLPI = 1e-4    # s-1  (Ki ~ 1 nM order of magnitude  NEEDS MEASUREMENT)

# Pro-MMP-9 activation by MMP-3 (kcat/Km for latency peptide cleavage).
# NEEDS MEASUREMENT: find from MEROPS database or MMP activation kinetics literature.
kcat_Km_MMP3    = 5e3    # M-1 s-1  NEEDS MEASUREMENT
kcat_Km_plasmin = 1e4    # M-1 s-1  NEEDS MEASUREMENT

# ── ODE system ─────────────────────────────────────────────────────────────────

def cascade_ode(t, y):
    """
    State vector y (all in mol/L):
      0: [proMMP9]        1: [MMP9_free]     2: [TIMP1_free]
      3: [MMP9:TIMP1]     4: [HNE_free]      5: [A1AT_free]
      6: [HNE:A1AT]       7: [SLPI_free]     8: [HNE:SLPI]
    MMP-3 and plasmin treated as constant (large excess relative to pro-MMP-9).
    """
    proMMP9, MMP9f, TIMP1f, MMP9T1, HNEf, A1ATf, HNE_A1AT, SLPIf, HNE_SLPI = y

    v_act_MMP3    = kcat_Km_MMP3   * C_MMP3_0   * proMMP9
    v_act_plasmin = kcat_Km_plasmin * C_plasmin_0 * proMMP9
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


def run_cascade(t_max_min=120):
    y0 = [C_proMMP9_0, C_MMP9_0, C_TIMP1_0, 0.0,
          C_HNE_0, C_A1AT_0, 0.0, C_SLPI_0, 0.0]
    t_eval = np.linspace(0, t_max_min * 60, 2000)
    sol = solve_ivp(cascade_ode, (0, t_max_min * 60), y0, t_eval=t_eval,
                    method="Radau", rtol=1e-10, atol=1e-15)
    return sol


def plot_cascade(sol, save=True):
    t = sol.t / 60   # minutes

    # Extract species in nM for unified display
    proMMP9    = sol.y[0] * 1e9
    MMP9f      = sol.y[1] * 1e9
    TIMP1f     = sol.y[2] * 1e9
    MMP9_TIMP1 = sol.y[3] * 1e9
    HNEf       = sol.y[4] * 1e9
    HNE_A1AT   = sol.y[6] * 1e9
    HNE_SLPI   = sol.y[8] * 1e9

    # Clinical range in nM
    MMP9_lo_nM = 17e-6 / MW_MMP9 * 1e9    # 0.185 nM
    MMP9_hi_nM = 200e-6 / MW_MMP9 * 1e9   # 2.17 nM

    # Colors
    c_enzyme   = "#C0392B"   # red
    c_zymogen  = "#2980B9"   # blue
    c_inhibitor= "#27AE60"   # green
    c_activator= "#E67E22"   # orange
    c_complex  = "#8E44AD"   # purple

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.40)

    # ── Panel A: "Who is in wound fluid?" -- initial concentrations ───────────
    ax1 = fig.add_subplot(gs[0, 0])

    # Species ordered from smallest to largest (log scale makes this sensible)
    names = ['MMP-3 (activator)\nFlag', 'plasmin (activator)\nFlag',
             'pro-MMP-9\n(inactive zymogen)',
             'MMP-9 free\n(active enzyme)',
             'TIMP-1\n(MMP-9 inhibitor) Flag',
             'HNE\n(active enzyme)',
             'SLPI\n(HNE inhibitor) Flag',
             'A1AT\n(HNE inhibitor)  Flag']

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
    ax1.set_xlabel("Concentration in wound fluid (nM)")
    ax1.set_title("A   What is in wound fluid at time of sampling?",
                  loc="left", fontweight="bold", fontsize=9)
    ax1.tick_params(axis="y", labelsize=7.5)

    # Shade the MMP-9 clinical detection range
    ax1.axvspan(MMP9_lo_nM, MMP9_hi_nM, alpha=0.12, color=c_enzyme,
                label=f"MMP-9 clinical range\n({MMP9_lo_nM:.2f}-{MMP9_hi_nM:.1f} nM)")
    ax1.legend(fontsize=7, loc="lower right")

    # Add value labels on bars
    for bar, val in zip(bars, concs):
        ax1.text(val * 1.8, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}" if val < 100 else f"{val:.0f}",
                 va="center", fontsize=7)

    legend_patches = [
        mpatches.Patch(color=c_enzyme,    label="Active enzyme"),
        mpatches.Patch(color=c_activator, label="Activator"),
        mpatches.Patch(color=c_zymogen,   label="Zymogen (inactive)"),
        mpatches.Patch(color=c_inhibitor, label="Inhibitor"),
    ]
    ax1.legend(handles=legend_patches, fontsize=7, loc="lower right")

    # ── Panel B: MMP-9 over time ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.fill_between(t, 0, proMMP9, alpha=0.20, color=c_zymogen,
                     label="pro-MMP-9 (inactive)")
    ax2.fill_between(t, 0, MMP9_TIMP1, alpha=0.25, color=c_inhibitor,
                     label="MMP-9 blocked by TIMP-1")
    ax2.plot(t, MMP9f, color=c_enzyme, lw=2.2, label="MMP-9 free (reaches probe)")

    ax2.axhspan(MMP9_lo_nM, MMP9_hi_nM, alpha=0.12, color=c_enzyme,
                label="Clinical range (17-200 ng/mL)")
    ax2.axvline(30, color="gray", ls=":", lw=1.2, label="30 min (sensor read)")
    ax2.axvline(60, color="gray", ls="--", lw=1.2, label="60 min (extended)")

    ax2.set_xlabel("Time in wound (min)")
    ax2.set_ylabel("Concentration (nM)")
    ax2.set_title("B   MMP-9: activation vs TIMP-1 inhibition",
                  loc="left", fontweight="bold", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(bottom=0)

    # Annotate key feature
    idx_30 = np.argmin(np.abs(sol.t - 30 * 60))
    ax2.annotate(
        f"Free MMP-9\nat 30 min:\n{MMP9f[idx_30]:.2f} nM",
        xy=(30, MMP9f[idx_30]),
        xytext=(55, MMP9f[idx_30] + 0.05),
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

    # Half-life annotation
    half_HNE = C_HNE_0 * 1e9 / 2
    t_half_idx = np.argmin(np.abs(HNEf - half_HNE))
    if t_half_idx < len(t) - 1:
        ax3.axvline(t[t_half_idx], color="gray", ls=":", lw=1.2)
        ax3.text(t[t_half_idx] + 0.5, half_HNE * 1.2,
                 f"t1/2 = {t[t_half_idx]:.1f} min", fontsize=7.5, color="gray")

    # ── Panel D: What reaches the sensor at different read times ──────────────
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
    ax4.axhspan(MMP9_lo_nM, MMP9_hi_nM, alpha=0.10, color=c_enzyme,
                label=f"MMP-9 clinical range")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{tp} min" for tp in t_points])
    ax4.set_xlabel("Swab contact time")
    ax4.set_ylabel("Free enzyme concentration (nM)")
    ax4.set_title("D   Free enzyme delivered to sensor vs read time",
                  loc="left", fontweight="bold", fontsize=9)
    ax4.legend(fontsize=7)
    ax4.set_ylim(bottom=0)

    fig.suptitle(
        "Layer 0: Wound fluid protease cascade\n"
        "MMP-9 is progressively blocked by TIMP-1; HNE is rapidly neutralised by A1AT. "
        "Items marked 'Flag' are estimated -- see literature gaps table.",
        fontsize=9.5, y=1.01
    )

    if save:
        out = FIGURES / "fig_B1_wound_biology.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    sol = run_cascade(t_max_min=120)
    plot_cascade(sol)
    for t_check in [5, 30, 60, 120]:
        idx = np.argmin(np.abs(sol.t - t_check * 60))
        print(f"t={t_check:3d} min | "
              f"MMP-9_free = {sol.y[1,idx]*1e9:.3f} nM | "
              f"HNE_free   = {sol.y[4,idx]*1e12:.2f} pM | "
              f"TIMP-1_free = {sol.y[2,idx]*1e9:.2f} nM | "
              f"A1AT_free  = {sol.y[5,idx]*1e6:.2f} uM")
