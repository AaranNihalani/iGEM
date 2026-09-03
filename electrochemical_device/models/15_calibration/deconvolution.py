"""
deconvolution.py  —  HNE interference bias check (rewritten)
================================================================
The device is a SINGLE MMP-9 channel (SGKIPRTLTA active + D-enantiomer
control, Layer 13 Sec III), not a two-probe MMP-9/HNE system. There is
no second channel to deconvolve against, so this module is no longer a
2x2 matrix inversion. Instead it implements Layer 9 Sec X's actual
competition model: in the first-order regime, contributions to the
cleavage rate from different proteases on the SAME probe add
independently:

    kobs_total = sum_j k_surf_eff_j * [E_j]           (Layer 9 Eq X.1)
    xi_j = k_surf_eff_j * [E_j] / kobs_total            (Eq X.2)

If HNE cross-cleaves SGKIPRTLTA at all (Layer 9 Table 3: "Unconfirmed,
Moderate risk"), it inflates F_k and biases the inferred [MMP-9]
upward, since the single-channel inversion (calibration_curve.E_inverse)
has no way to separate the two contributions.

This module quantifies that bias:
  - as a function of the (currently unconfirmed) HNE cross-reactivity
    fraction, and
  - at the literature-reported clinical HNE levels (Layer 0 / Trengove
    1999: ~30 nM in chronic wounds, ~0 in healing wounds),
and checks whether the resulting bias threatens the 5x chronic/healing
MMP-9 discrimination ratio that Layer 9 identifies as robust in the
*absence* of interference (Sec VIII, Eq VIII.3).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from layer15_common import (
    T_INCUBATION, RHO, KCAT_KM_MMP9_SURFACE_DEMO, HNE_ON_MMP9_PROBE_FRACTION,
    WOUND_MMP9_CHRONIC_NM, WOUND_MMP9_HEALING_NM, nM_to_M, M_to_nM,
)
from calibration_curve import Fk_forward, Fk_from_target, E_inverse

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# Literature-consistent clinical HNE levels (layer0_wound_biology.pdf):
# HNE is elevated (~30 nM) in roughly half of chronic wound samples, and
# essentially absent (~0) in healing wounds.
HNE_CHRONIC_NM = 30.0    # TBD (only ~50% of chronic samples show elevated HNE)
HNE_HEALING_NM = 0.0


def kobs_contributions(c_mmp9_nM, c_hne_nM, k_mmp9=KCAT_KM_MMP9_SURFACE_DEMO,
                        hne_fraction=HNE_ON_MMP9_PROBE_FRACTION):
    """
    Per Layer 9 Eq X.1: pseudo-first-order rate contributions from MMP-9
    and (cross-reacting) HNE on the same SGKIPRTLTA probe.
    k_hne_on_probe = hne_fraction * k_mmp9 (crude placeholder scaling;
    replace with a measured k_surf_eff for HNE on SGKIPRTLTA once
    available — Layer 9 Table 3 confirms no such measurement exists yet).
    """
    k_hne_on_probe = hne_fraction * k_mmp9
    kobs_mmp9 = k_mmp9 * nM_to_M(c_mmp9_nM)
    kobs_hne = k_hne_on_probe * nM_to_M(c_hne_nM)
    return kobs_mmp9, kobs_hne


def biased_mmp9_estimate(c_mmp9_true_nM, c_hne_nM, T=T_INCUBATION, rho=RHO,
                          k_mmp9=KCAT_KM_MMP9_SURFACE_DEMO,
                          hne_fraction=HNE_ON_MMP9_PROBE_FRACTION):
    """
    Simulate the single-channel readout in the presence of HNE cross-
    reactivity, then invert it AS IF it were pure MMP-9 signal (which is
    all the device can do — there is no second channel to separate the
    two). Returns (c_mmp9_apparent_nM, xi_hne, Fk_observed).
    """
    kobs_mmp9, kobs_hne = kobs_contributions(c_mmp9_true_nM, c_hne_nM, k_mmp9, hne_fraction)
    kobs_total = kobs_mmp9 + kobs_hne
    xi_hne = kobs_hne / kobs_total if kobs_total > 0 else 0.0

    Fk_observed = Fk_from_target(kobs_total * T, rho)
    c_mmp9_apparent_M = E_inverse(Fk_observed, T, k_mmp9, rho)
    return M_to_nM(c_mmp9_apparent_M), xi_hne, Fk_observed


def plot_bias_sweep(save=True):
    """Sweep the unconfirmed HNE cross-reactivity fraction and show the
    resulting bias in inferred [MMP-9] at clinical concentrations."""
    fractions = np.logspace(-3, -0.3, 60)   # 0.1% to ~50% cross-talk

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for c_mmp9, c_hne, label, color in [
        (WOUND_MMP9_CHRONIC_NM, HNE_CHRONIC_NM, "Chronic (30 nM MMP-9, 30 nM HNE)", "red"),
        (WOUND_MMP9_HEALING_NM, HNE_HEALING_NM, "Healing (6 nM MMP-9, 0 nM HNE)", "green"),
    ]:
        bias_pct = []
        for frac in fractions:
            c_app, _, _ = biased_mmp9_estimate(c_mmp9, c_hne, hne_fraction=frac)
            bias_pct.append((c_app - c_mmp9) / c_mmp9 * 100 if c_mmp9 > 0 else 0)
        ax1.semilogx(fractions * 100, bias_pct, "-", color=color, lw=2, label=label)

    ax1.axvline(HNE_ON_MMP9_PROBE_FRACTION * 100, color="gray", ls=":", lw=1.2,
                label=f"Assumed placeholder ({HNE_ON_MMP9_PROBE_FRACTION*100:.0f}%)")
    ax1.axhline(20, color="orange", ls="--", lw=1, label="20% bias (LLOQ-scale error)")
    ax1.set_xlabel("HNE cross-reactivity fraction on SGKIPRTLTA (%)")
    ax1.set_ylabel("Bias in inferred [MMP-9] (%)")
    ax1.set_title("Layer 9 Sec X: HNE interference bias vs cross-reactivity")
    ax1.legend(fontsize=7.5)

    # Discrimination ratio under worst-case bias (chronic, HNE present) vs
    # clean healing baseline (no HNE reported in healing wounds).
    ratios = []
    for frac in fractions:
        c_chronic_app, _, _ = biased_mmp9_estimate(WOUND_MMP9_CHRONIC_NM, HNE_CHRONIC_NM, hne_fraction=frac)
        c_healing_app, _, _ = biased_mmp9_estimate(WOUND_MMP9_HEALING_NM, HNE_HEALING_NM, hne_fraction=frac)
        ratios.append(c_chronic_app / c_healing_app)

    ax2.semilogx(fractions * 100, ratios, "b-", lw=2, label="Apparent chronic/healing ratio")
    ax2.axhline(5, color="k", ls="--", lw=1.5, label="Clean ratio (no interference) = 5x")
    ax2.axvline(HNE_ON_MMP9_PROBE_FRACTION * 100, color="gray", ls=":", lw=1.2,
                label=f"Assumed placeholder ({HNE_ON_MMP9_PROBE_FRACTION*100:.0f}%)")
    ax2.set_xlabel("HNE cross-reactivity fraction on SGKIPRTLTA (%)")
    ax2.set_ylabel("Apparent chronic/healing MMP-9 ratio")
    ax2.set_title("Discrimination ratio robustness to HNE interference")
    ax2.legend(fontsize=8)

    fig.suptitle("HNE interference on the single MMP-9 channel (Layer 9 Sec X)", fontsize=11)
    plt.tight_layout()
    if save:
        out = FIGDIR / "fig_hne_interference_bias.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def main():
    print("HNE interference bias check (Layer 9 Sec X) — single MMP-9 channel\n")
    print("NOTE: HNE cross-reactivity on SGKIPRTLTA is UNCONFIRMED (Layer 9 Table 3).")
    print(f"All numbers below use a {HNE_ON_MMP9_PROBE_FRACTION*100:.0f}% placeholder "
          "fraction pending a real cross-reactivity measurement.\n")

    for label, c_mmp9, c_hne in [
        ("Chronic wound", WOUND_MMP9_CHRONIC_NM, HNE_CHRONIC_NM),
        ("Healing wound", WOUND_MMP9_HEALING_NM, HNE_HEALING_NM),
    ]:
        c_app, xi_hne, Fk_obs = biased_mmp9_estimate(c_mmp9, c_hne)
        bias_pct = (c_app - c_mmp9) / c_mmp9 * 100 if c_mmp9 > 0 else float("nan")
        print(f"{label}: true [MMP-9]={c_mmp9:.1f} nM, [HNE]={c_hne:.1f} nM")
        print(f"  F_k observed          : {Fk_obs*100:.2f}%")
        print(f"  HNE fraction of signal (xi_HNE): {xi_hne*100:.2f}%")
        print(f"  Apparent [MMP-9]      : {c_app:.2f} nM  (bias {bias_pct:+.1f}%)\n")

    c_chronic_app, _, _ = biased_mmp9_estimate(WOUND_MMP9_CHRONIC_NM, HNE_CHRONIC_NM)
    c_healing_app, _, _ = biased_mmp9_estimate(WOUND_MMP9_HEALING_NM, HNE_HEALING_NM)
    apparent_ratio = c_chronic_app / c_healing_app
    print(f"Apparent chronic/healing ratio with interference: {apparent_ratio:.2f}x "
          f"(clean/no-interference value: 5.0x, Layer 9 Eq VIII.3)")

    plot_bias_sweep()
    return apparent_ratio


if __name__ == "__main__":
    main()
