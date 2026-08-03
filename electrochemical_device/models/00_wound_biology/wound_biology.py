"""
wound_biology.py  —  Layer 0
=============================
Full protease activation cascade ODE for MMP-9 and HNE in chronic wound fluid.

Cascade overview
----------------
MMP-9 pathway (metalloprotease):
  pro-MMP-9 ──[MMP-3]──> active MMP-9   (proteolytic zymogen activation)
  pro-MMP-9 ──[plasmin]──> active MMP-9  (second activation route)
  active MMP-9 + TIMP-1  <──> MMP-9:TIMP-1  (reversible inhibition)

HNE pathway (serine protease, released already active):
  HNE + A1AT  ──> HNE:A1AT    (near-irreversible inhibition, k_on=2e7 M-1s-1)
  HNE + SLPI  ──> HNE:SLPI   (reversible slow-tight inhibition)

Literature values flagged with # FLAG are uncertain — see inline comments.
All other values are cited and defensible.

Outputs (sent to Layer 1):
  [MMP-9]_free(t), [HNE]_free(t)
  [TIMP-1]_free(t), [A1AT]_free(t), [SLPI]_free(t)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Initial conditions (wound fluid concentrations at t=0) ─────────────────
# All in mol/L (M).

# MMP-9: 17–200 ng/mL clinical range; mid-point ~100 ng/mL used here.
# Assume 70% is pro-form (zymogen) at time of wounding (Yager et al. 1996,
# J Invest Dermatol — chronic wound fluid has majority pro-MMP-9).
MW_MMP9   = 92e3    # g/mol
MW_HNE    = 30e3    # g/mol
MW_TIMP1  = 28.5e3  # g/mol
MW_A1AT   = 52e3    # g/mol
MW_SLPI   = 11.7e3  # g/mol
MW_MMP3   = 54e3    # g/mol  (stromelysin-1)

C_proMMP9_0  = 0.70 * (100e-6) / MW_MMP9    # 70% of 100 ng/mL  → ~0.76 nM
C_MMP9_0     = 0.30 * (100e-6) / MW_MMP9    # 30% already active → ~0.33 nM

# HNE: released from neutrophil granules as active enzyme.
# Concentration in chronic wound fluid: ~50–200 nM (Trengove et al. 1999,
# Wound Repair Regen).  Use 100 nM as representative.
C_HNE_0  = 100e-9    # 100 nM  (CONFIDENT — Trengove et al. 1999)

# Inhibitors — FLAG: please verify wound-fluid concentrations
# TIMP-1: ~100–400 ng/mL in chronic wounds (Trengove 1999 reports 200 ng/mL typical)
C_TIMP1_0 = (200e-6) / MW_TIMP1   # 200 ng/mL → ~7 nM
# FLAG: Trengove 1999 is the most cited source but check if they report
# free vs total TIMP-1.  If total includes bound forms this overestimates free.

# A1AT: dominant HNE inhibitor in wound fluid.  Serum A1AT ~1–2 mg/mL;
# wound fluid is diluted ~10×, so ~0.1–0.2 mg/mL expected.
# Use 0.1 mg/mL = 100 µg/mL.
C_A1AT_0  = (100e-6) / MW_A1AT    # 100 µg/mL → ~1.9 µM  (CONFIDENT order of magnitude)
# FLAG: need a wound-fluid specific measurement, not serum-derived estimate.

# SLPI: ~100–500 ng/mL reported in wound fluid (Ashcroft et al. 2000, JCI).
C_SLPI_0  = (200e-6) / MW_SLPI    # 200 ng/mL → ~17 nM
# FLAG: Ashcroft 2000 is the reference but check their patient population matches
# chronic wound context (they used acute).

# MMP-3 (activates pro-MMP-9): present in wound fluid but concentration uncertain.
# FLAG: [MMP-3] in chronic wound fluid — please find a defensible number.
# Placeholder: 10 ng/mL based on general MMP panel data in wounds.
C_MMP3_0  = (10e-6) / MW_MMP3     # 10 ng/mL → ~0.19 nM   ← FLAG

# Plasmin: active plasmin in wound fluid difficult to measure (rapidly inhibited
# by alpha-2-antiplasmin).  Use ~1 nM as order-of-magnitude estimate.
# FLAG: this value is poorly constrained — please find wound-fluid specific data.
C_plasmin_0 = 1e-9    # 1 nM  ← FLAG

# ── Rate constants ──────────────────────────────────────────────────────────

# A1AT/HNE association: k_on = 2×10^7 M-1s-1 (Beatty et al. 1980, Biochemistry;
# confirmed by Bieth 1984, Bull Eur Physiopathol Respir — CONFIDENT)
k_on_A1AT  = 2e7     # M-1 s-1  (CONFIDENT)
k_off_A1AT = 1e-6    # s-1  (near-irreversible; Ki = k_off/k_on ~ 0.05 pM — CONFIDENT)

# TIMP-1/MMP-9:
# FLAG: k_on — I have seen 2×10^5 M-1s-1 cited (Langton et al. 1998 Biochemistry)
# but please verify this is for TIMP-1 specifically (not TIMP-2).
k_on_TIMP1  = 2e5    # M-1 s-1  ← FLAG
k_off_TIMP1 = 2e-5   # s-1  (Ki = k_off/k_on = 0.1 nM — consistent with
                     # Nagase & Woessner 1999 review — CONFIDENT that Ki ~ 0.1 nM)

# SLPI/HNE:
# FLAG: k_on for SLPI/HNE.  SLPI is a slow, tight-binding inhibitor.
# Literature suggests Ki ~ 1 nM and association is slower than A1AT.
# Placeholder k_on ~ 1×10^5 M-1s-1  ← FLAG
k_on_SLPI  = 1e5     # M-1 s-1  ← FLAG
k_off_SLPI = 1e-4    # s-1  (Ki = k_off/k_on = 1 nM  ← FLAG order of magnitude)

# Pro-MMP-9 activation by MMP-3:
# Proteolytic activation — treated as pseudo-first-order in [proMMP9] with
# effective rate k_act_MMP3 = kcat_MMP3 * [MMP3] / Km_MMP3.
# FLAG: kcat and Km for MMP-3 acting on pro-MMP-9 latency peptide.
# Placeholder: k_act_MMP3 ≈ 5×10^3 M-1s-1 × [MMP3]   ← FLAG
kcat_Km_MMP3  = 5e3  # M-1 s-1  ← FLAG — need literature kcat/Km for MMP-3/proMMP-9

# Pro-MMP-9 activation by plasmin:
# FLAG: same uncertainty — proteolytic rate constant poorly defined.
# Placeholder: kcat/Km ~ 1×10^4 M-1s-1  ← FLAG
kcat_Km_plasmin = 1e4  # M-1 s-1  ← FLAG

# ── ODE system ──────────────────────────────────────────────────────────────

def cascade_ode(t, y):
    """
    State vector y:
      0: [proMMP9]
      1: [MMP9_free]
      2: [TIMP1_free]
      3: [MMP9_TIMP1]
      4: [HNE_free]
      5: [A1AT_free]
      6: [HNE_A1AT]
      7: [SLPI_free]
      8: [HNE_SLPI]
    MMP-3 and plasmin treated as constant (in large excess relative to pro-MMP-9).
    """
    proMMP9, MMP9f, TIMP1f, MMP9T1, HNEf, A1ATf, HNE_A1AT, SLPIf, HNE_SLPI = y

    # Activation fluxes
    v_act_MMP3   = kcat_Km_MMP3   * C_MMP3_0   * proMMP9
    v_act_plasmin= kcat_Km_plasmin * C_plasmin_0 * proMMP9

    # TIMP-1 / MMP-9 association–dissociation
    v_T1_on  = k_on_TIMP1  * MMP9f * TIMP1f
    v_T1_off = k_off_TIMP1 * MMP9T1

    # A1AT / HNE (near-irreversible)
    v_A1AT   = k_on_A1AT * HNEf * A1ATf

    # SLPI / HNE
    v_SLPI_on  = k_on_SLPI  * HNEf * SLPIf
    v_SLPI_off = k_off_SLPI * HNE_SLPI

    d_proMMP9  = -(v_act_MMP3 + v_act_plasmin)
    d_MMP9f    =  (v_act_MMP3 + v_act_plasmin) - v_T1_on + v_T1_off
    d_TIMP1f   = -v_T1_on + v_T1_off
    d_MMP9T1   =  v_T1_on - v_T1_off
    d_HNEf     = -v_A1AT - v_SLPI_on + v_SLPI_off
    d_A1ATf    = -v_A1AT
    d_HNE_A1AT =  v_A1AT
    d_SLPIf    = -v_SLPI_on + v_SLPI_off
    d_HNE_SLPI =  v_SLPI_on - v_SLPI_off

    return [d_proMMP9, d_MMP9f, d_TIMP1f, d_MMP9T1,
            d_HNEf, d_A1ATf, d_HNE_A1AT, d_SLPIf, d_HNE_SLPI]


def run_cascade(t_max_min=60):
    y0 = [C_proMMP9_0, C_MMP9_0, C_TIMP1_0, 0.0,
          C_HNE_0, C_A1AT_0, 0.0, C_SLPI_0, 0.0]

    t_span = (0, t_max_min * 60)
    t_eval = np.linspace(0, t_max_min * 60, 1000)

    sol = solve_ivp(cascade_ode, t_span, y0, t_eval=t_eval,
                    method="Radau", rtol=1e-9, atol=1e-12)
    return sol


def plot_cascade(sol, save=True):
    t_min = sol.t / 60

    proMMP9  = sol.y[0] * 1e9   # nM
    MMP9f    = sol.y[1] * 1e9
    TIMP1f   = sol.y[2] * 1e9
    HNEf     = sol.y[4] * 1e9
    A1ATf    = sol.y[5] * 1e6   # µM
    SLPIf    = sol.y[7] * 1e9

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: MMP-9 species
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_min, proMMP9, "b--",  lw=1.8, label="pro-MMP-9 (zymogen)")
    ax1.plot(t_min, MMP9f,   "b-",   lw=1.8, label="MMP-9 free (active)")
    ax1.plot(t_min, TIMP1f,  "g-",   lw=1.8, label="TIMP-1 free")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Concentration (nM)")
    ax1.set_title("A  —  MMP-9 cascade")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, t_min[-1])

    # Panel B: HNE inhibition
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_min, HNEf,  "r-",  lw=1.8, label="HNE free (active)")
    ax2.plot(t_min, SLPIf, "m-",  lw=1.8, label="SLPI free")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Concentration (nM)")
    ax2.set_title("B  —  HNE inhibition")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, t_min[-1])

    # Panel C: A1AT depletion (µM scale)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_min, A1ATf,         "k-",  lw=1.8, label="A1AT free")
    ax3.plot(t_min, sol.y[6]*1e6,  "k--", lw=1.8, label="HNE:A1AT complex")
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("Concentration (µM)")
    ax3.set_title("C  —  A1AT depletion")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, t_min[-1])

    # Panel D: outputs to Layer 1 summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_min, MMP9f, "b-", lw=2.0, label="[MMP-9]$_\\mathrm{free}$")
    ax4.plot(t_min, HNEf,  "r-", lw=2.0, label="[HNE]$_\\mathrm{free}$")
    ax4.axvspan(0, 5,   alpha=0.08, color="gray", label="Swab contact window")
    ax4.set_xlabel("Time post-wounding (min)")
    ax4.set_ylabel("Concentration (nM)")
    ax4.set_title("D  —  Free enzyme outputs to Layer 1")
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, t_min[-1])

    fig.suptitle("Layer 0: Wound-fluid protease activation cascade\n"
                 "(dashed rates flagged — see wound_biology.py)", fontsize=11)

    if save:
        out = FIGURES / "fig_B1_wound_biology.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    sol = run_cascade(t_max_min=120)
    plot_cascade(sol)
    # Print summary at t=5 min (swab contact point) and t=60 min
    for t_check in [5, 30, 60, 120]:
        idx = np.argmin(np.abs(sol.t - t_check * 60))
        print(f"t={t_check:3d} min | "
              f"[MMP-9]_free={sol.y[1,idx]*1e9:.3f} nM | "
              f"[HNE]_free={sol.y[4,idx]*1e9:.2f} nM | "
              f"[TIMP-1]_free={sol.y[2,idx]*1e9:.2f} nM | "
              f"[A1AT]_free={sol.y[5,idx]*1e6:.2f} µM")
