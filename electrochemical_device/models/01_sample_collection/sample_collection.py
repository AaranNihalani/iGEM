"""
sample_collection.py  —  Layer 1
==================================
Models the swab-absorption → elution → chamber-fill sequence.

Inputs  (from Layer 0):
  [E]_wound, [I]_wound  for each species at t_swab

Outputs (to Layer 2 / 3):
  [E]_chamber, [I]_chamber  (concentrations entering the microfluidic channel)
  pH_chamber
  [GSH]_chamber, [ascorbate]_chamber  (wound redox co-elutants)

Key sub-models
--------------
1. Swab absorption volume V_abs
2. Elution efficiency η per species (size-dependent)
3. Dilution factor D_f = V_elution / V_abs
4. Inhibitor co-elution (SLPI elutes more efficiently than HNE → enriched ratio)
5. pH mixing: wound pH + buffer pH → chamber pH
6. Wound redox agents: GSH, ascorbate co-elute
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Device geometry ─────────────────────────────────────────────────────────
V_swab_max   = 30e-6    # L  (30 µL max swab absorption volume — manufacturer spec)
V_elution    = 60e-6    # L  (60 µL elution buffer added to swab)
D_f          = V_elution / V_swab_max   # dilution factor ≈ 2×

# ── Elution efficiencies (fraction of absorbed analyte that elutes) ─────────
# Small molecules / peptides elute near-completely; large proteins less so.
# Values estimated from published swab elution recovery studies (Magerusan et al.
# 2020 Biosens Bioelectron; Rogers et al. 2019 PLOS ONE).
# FLAG: species-specific η for TIMP-1, A1AT, SLPI in this exact swab/buffer
# system not measured — use size-based approximation until wet lab validates.

eta = {
    "MMP9":     0.70,   # 92 kDa large protein — moderate recovery    ← FLAG
    "HNE":      0.75,   # 30 kDa — slightly better recovery            ← FLAG
    "TIMP1":    0.65,   # 28.5 kDa — similar to HNE                   ← FLAG
    "A1AT":     0.60,   # 52 kDa — moderate                           ← FLAG
    "SLPI":     0.85,   # 11.7 kDa small — high recovery (Ashcroft 2000)
    "GSH":      0.90,   # small molecule — high recovery
    "ascorbate":0.90,   # small molecule — high recovery
}

# ── Wound-fluid concentrations (from Layer 0 at t_swab) ───────────────────
# Representative mid-range wound values.  In forward chain, these come from
# wound_biology.py output at the time of swab contact.
wound_conc = {
    "MMP9":      100e-6 / 92e3,    # 100 ng/mL in mol/L
    "HNE":       100e-9,           # 100 nM
    "TIMP1":     200e-6 / 28.5e3,  # 200 ng/mL
    "A1AT":      1.9e-6,           # ~1.9 µM
    "SLPI":      200e-6 / 11.7e3,  # 200 ng/mL
    "GSH":       0.25e-3,          # 0.25 mM  (midpoint of 0.1–0.5 mM range)
    "ascorbate": 0.1e-3,           # 0.1 mM
}

# ── pH mixing ───────────────────────────────────────────────────────────────
# Wound pH in chronic wounds: typically 7.4–8.9 (Gethin 2007, J Wound Care).
# Elution buffer pH: 7.4 (PBS or HEPES).
# FLAG: exact wound pH range to use — Gethin reports wide variation.
pH_wound  = 8.0    # representative chronic wound pH  ← FLAG (use your target patient population)
pH_buffer = 7.4    # elution buffer  (CONFIDENT)

# Simple volume-weighted pH mixing (valid when buffering capacity >> pH swing).
# FLAG: if wound fluid has significant buffering capacity (bicarbonate, proteins)
# the buffer addition may not fully bring pH to 7.4.  Need buffering capacity data.
V_wound_in_chamber = V_swab_max   # volume of wound fluid absorbed into swab
pH_chamber = (pH_wound * V_wound_in_chamber + pH_buffer * V_elution) / \
             (V_wound_in_chamber + V_elution)

# ── Chamber concentrations ───────────────────────────────────────────────────
def elute(species):
    c_wound = wound_conc[species]
    eta_s   = eta[species]
    # Absorbed amount = c_wound × V_swab_max × eta_s  (mol)
    # Eluted into V_elution volume:
    return c_wound * V_swab_max * eta_s / V_elution

chamber_conc = {s: elute(s) for s in wound_conc}

# ── SLPI/HNE enrichment ratio ────────────────────────────────────────────────
# In wound fluid: [SLPI]/[HNE] ratio.  After elution, SLPI is enriched
# relative to HNE because eta_SLPI > eta_HNE.
ratio_wound   = wound_conc["SLPI"]   / wound_conc["HNE"]
ratio_chamber = chamber_conc["SLPI"] / chamber_conc["HNE"]
slpi_enrichment = ratio_chamber / ratio_wound


def report():
    print("=" * 60)
    print("Layer 1  —  Sample collection summary")
    print(f"  Swab volume:   {V_swab_max*1e6:.0f} µL")
    print(f"  Elution vol:   {V_elution*1e6:.0f} µL")
    print(f"  Dilution D_f:  {D_f:.1f}×")
    print(f"  pH_wound:      {pH_wound}")
    print(f"  pH_chamber:    {pH_chamber:.2f}")
    print()
    print(f"  {'Species':<12} {'[wound] (nM)':>14} {'[chamber] (nM)':>16} {'η':>6}")
    print(f"  {'-'*52}")
    for s in wound_conc:
        cw = wound_conc[s] * 1e9
        cc = chamber_conc[s] * 1e9
        print(f"  {s:<12} {cw:>14.3f} {cc:>16.3f} {eta[s]:>6.2f}")
    print()
    print(f"  SLPI/HNE ratio — wound: {ratio_wound:.2f}  chamber: {ratio_chamber:.2f}"
          f"  (enrichment: {slpi_enrichment:.2f}×)")


def plot_elution(save=True):
    species_list = list(wound_conc.keys())
    cw = np.array([wound_conc[s] * 1e9 for s in species_list])
    cc = np.array([chamber_conc[s] * 1e9 for s in species_list])
    etas = np.array([eta[s] for s in species_list])

    x = np.arange(len(species_list))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(x - width/2, cw, width, label="Wound fluid", color="#2196F3", alpha=0.85)
    axes[0].bar(x + width/2, cc, width, label="Chamber",     color="#FF5722", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(species_list, rotation=30, ha="right")
    axes[0].set_ylabel("Concentration (nM)")
    axes[0].set_title("A  —  Wound vs chamber concentrations")
    axes[0].legend()

    axes[1].bar(x, etas * 100, color="#4CAF50", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(species_list, rotation=30, ha="right")
    axes[1].set_ylabel("Elution efficiency η (%)")
    axes[1].set_title("B  —  Species elution efficiency")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(100, color="gray", lw=0.8, ls="--")

    fig.suptitle("Layer 1: Sample collection — elution and dilution\n"
                 "(η values flagged — see sample_collection.py)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B1b_sample_collection.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    report()
    plot_elution()
