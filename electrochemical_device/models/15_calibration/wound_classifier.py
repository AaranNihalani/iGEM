"""
wound_classifier.py  —  Layer 15.7 (rewritten against Layer 0/9/13)
========================================================================
Bayesian wound-state classifier, updated against the actual confirmed
data in layer0_wound_biology.pdf and the real single-channel device
architecture (Layers 9, 13).

Two changes from the original version:

1. TWO states, not three. Layer 0 only ever analyses Healing vs. Chronic
   wounds (no "Inflammatory" intermediate state is defined or supported
   by any cited literature); a third state was an invented middle
   category with no grounding. Removed.

2. PRIMARY classifier uses [MMP-9] alone. The device has a single
   MMP-9 channel (Layers 9/13) — there is no HNE channel to measure
   clinically. Layer 0 itself concludes HNE is "not a reliable
   standalone discriminator" (elevated in only 6/13 chronic samples —
   i.e. ~50% of chronic wounds are HNE-low regardless of state) and
   explicitly frames it as "a secondary reference channel, not a
   primary diagnostic output." So classify_mmp9() is what the real
   device can actually compute; classify_mmp9_hne() is kept only as an
   illustrative secondary model for a possible future dual-channel
   device, clearly flagged as such.

Confirmed wound-state values (layer0_wound_biology.pdf, Table 2 / Sec
II-III, full ODE at t = 30 min):
    [MMP-9]_chronic  = 31.9 nM   (TIMP-1/MMP-9 equilibrium + activation)
    [MMP-9]_healing  =  6.2 nM
    [HNE]_chronic    = ~30 nM    (ONLY when A1AT is degraded AND SLPI
                                   placeholder kinetics hold; not
                                   quantitatively reliable — Sec V)
    [HNE]_healing    = ~0 nM     (A1AT-mediated clearance, t1/2 ~ 5-55 ms,
                                   robust conclusion)

No paper reports a directly propagated uncertainty on [MMP-9]free
itself. Layer 0 does note the healing estimate should be treated as
"order-of-magnitude" (Sec II sensitivity discussion). Rather than invent
an asymmetric split not backed by their analysis, both states use the
same conservative log10-sigma below — flagged as approximate, not
literature-confirmed. TBD (wet lab / a real propagated Layer 0 error
analysis).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from layer15_common import WOUND_MMP9_CHRONIC_NM, WOUND_MMP9_HEALING_NM

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ── Confirmed / best-estimate wound-state parameters (log10 nM space) ─────
# Sigmas are APPROXIMATE, derived from Layer 0's reported Ki-sensitivity
# (chronic ~12% relative, healing much more sensitive) — not a formally
# propagated uncertainty. TBD (wet lab / a real propagated Layer 0 error
# analysis).
# Sigmas are APPROXIMATE. Layer 0 itself only characterises the healing
# estimate as "order-of-magnitude" (Sec II) and gives no formally
# propagated uncertainty for either state, so a single conservative
# symmetric log10-sigma is used for both rather than inventing an
# asymmetric split that isn't backed by their sensitivity analysis.
WOUND_MMP9_SIGMA_LOG10 = 0.15   # ~40% relative uncertainty, both states  # TBD (wet lab / real propagation)

WOUND_STATES_MMP9 = {
    "Healing": {"mean_log10": np.log10(WOUND_MMP9_HEALING_NM), "sigma_log10": WOUND_MMP9_SIGMA_LOG10, "prior": 0.5},
    "Chronic": {"mean_log10": np.log10(WOUND_MMP9_CHRONIC_NM), "sigma_log10": WOUND_MMP9_SIGMA_LOG10, "prior": 0.5},
}

# Illustrative-only 2D extension (NOT implemented on the real device).
# Chronic HNE is bimodal (elevated in only ~50% of chronic samples per
# Trengove 1999), which a single Gaussian can't really capture; the wide
# sigma below is a crude stand-in for that bimodality, not a real fit.
WOUND_STATES_MMP9_HNE = {
    "Healing": {"mean": np.log10([WOUND_MMP9_HEALING_NM, 1.0]),   # HNE ~0, floored at 1 nM for log-space
                "cov": np.diag([WOUND_MMP9_SIGMA_LOG10**2, 0.30**2]), "prior": 0.5},
    "Chronic": {"mean": np.log10([WOUND_MMP9_CHRONIC_NM, 15.0]),  # HNE mean ~15 nM (crude 50/50 elevated/not average)
                "cov": np.diag([WOUND_MMP9_SIGMA_LOG10**2, 0.55**2]), "prior": 0.5},
}


def _gaussian_1d_logpdf(x_log10, mean_log10, sigma_log10):
    return -0.5 * ((x_log10 - mean_log10) / sigma_log10) ** 2 - np.log(sigma_log10 * np.sqrt(2 * np.pi))


def _gaussian_2d_logpdf(x, mean, cov):
    d = x - mean
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    exponent = -0.5 * d @ inv_cov @ d
    norm_const = -0.5 * (2 * np.log(2 * np.pi) + np.log(det_cov))
    return norm_const + exponent


def classify_mmp9(c_mmp9_nM, states=WOUND_STATES_MMP9):
    """
    PRIMARY classifier — matches what the real (single-channel) device
    can actually compute: P(state | [MMP-9]) via Bayes' rule with
    log-normal likelihoods.
    """
    x = np.log10(max(c_mmp9_nM, 1e-3))
    log_joint = {s: _gaussian_1d_logpdf(x, p["mean_log10"], p["sigma_log10"]) + np.log(p["prior"])
                 for s, p in states.items()}
    max_log = max(log_joint.values())
    unnorm = {s: np.exp(v - max_log) for s, v in log_joint.items()}
    total = sum(unnorm.values())
    return {s: v / total for s, v in unnorm.items()}


def classify_mmp9_hne(c_mmp9_nM, c_hne_nM, states=WOUND_STATES_MMP9_HNE):
    """
    ILLUSTRATIVE ONLY — the real device has no HNE channel. Kept to show
    what a future dual-channel device could add, and how little the HNE
    axis actually helps given Layer 0's own conclusion that HNE is
    unreliable in ~50% of chronic samples.
    """
    x = np.log10([max(c_mmp9_nM, 1e-3), max(c_hne_nM, 1e-3)])
    log_joint = {s: _gaussian_2d_logpdf(x, p["mean"], p["cov"]) + np.log(p["prior"])
                 for s, p in states.items()}
    max_log = max(log_joint.values())
    unnorm = {s: np.exp(v - max_log) for s, v in log_joint.items()}
    total = sum(unnorm.values())
    return {s: v / total for s, v in unnorm.items()}


def find_decision_boundary_mmp9(states=WOUND_STATES_MMP9, lo=0.5, hi=200):
    """[MMP-9] at which P(Healing) = P(Chronic) = 0.5, via bisection."""
    from scipy.optimize import brentq

    def f(c):
        p = classify_mmp9(c, states)
        return p["Healing"] - p["Chronic"]

    return brentq(f, lo, hi)


def plot_mmp9_classifier(save=True, calibration_LOD_nM=None, calibration_ULOQ_nM=None):
    c_range = np.logspace(np.log10(0.5), np.log10(200), 400)
    p_healing = np.array([classify_mmp9(c)["Healing"] for c in c_range])
    p_chronic = 1 - p_healing
    boundary = find_decision_boundary_mmp9()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogx(c_range, p_healing * 100, "g-", lw=2, label="P(Healing)")
    ax.semilogx(c_range, p_chronic * 100, "r-", lw=2, label="P(Chronic)")
    ax.axvline(boundary, color="k", ls="--", lw=1.5, label=f"Decision boundary = {boundary:.1f} nM")
    ax.axvline(WOUND_MMP9_HEALING_NM, color="green", ls=":", lw=1.2, label=f"Healing mean = {WOUND_MMP9_HEALING_NM:.1f} nM")
    ax.axvline(WOUND_MMP9_CHRONIC_NM, color="red", ls=":", lw=1.2, label=f"Chronic mean = {WOUND_MMP9_CHRONIC_NM:.1f} nM")

    if calibration_LOD_nM is not None and calibration_ULOQ_nM is not None:
        ax.axvspan(calibration_LOD_nM, calibration_ULOQ_nM, alpha=0.08, color="blue",
                   label=f"Device measurable range ({calibration_LOD_nM:.0f}-{calibration_ULOQ_nM:.0f} nM)")

    ax.set_xlabel("[MMP-9] (nM)")
    ax.set_ylabel("Posterior probability (%)")
    ax.set_title("Layer 15.7: MMP-9-only wound-state classifier (single-channel device)")
    ax.legend(fontsize=8, loc="center left")
    plt.tight_layout()
    if save:
        out = FIGDIR / "fig_wound_classifier_mmp9.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)
    return boundary


def main():
    print("Layer 15.7: Wound-state classification (rewritten against Layer 0)\n")
    print("PRIMARY (device-realistic): [MMP-9] alone, single-channel device.\n")

    examples_mmp9 = {
        "Confirmed chronic mean": WOUND_MMP9_CHRONIC_NM,
        "Confirmed healing mean": WOUND_MMP9_HEALING_NM,
        "Ambiguous mid-range": 14.0,
    }
    for name, c in examples_mmp9.items():
        p = classify_mmp9(c)
        best = max(p, key=p.get)
        print(f"{name}: [MMP-9]={c:.1f} nM")
        for state, prob in sorted(p.items(), key=lambda kv: -kv[1]):
            marker = "  <-- MAP" if state == best else ""
            print(f"    P({state:8s}) = {prob:.3f}{marker}")
        print()

    boundary = find_decision_boundary_mmp9()
    print(f"Decision boundary (P(Healing)=P(Chronic)): {boundary:.2f} nM")

    # Integration check against the actual fitted calibration curve
    # (Layer 15.2/15.3): is the decision boundary inside the device's
    # measurable range?
    try:
        import csv
        with open(Path(__file__).parent / "calibration_summary.csv") as f:
            row = next(csv.DictReader(f))
        LOD_nM, ULOQ_nM = float(row["LOD_nM"]), float(row["ULOQ_nM"])
        print(f"\nDevice measurable range (from calibration_summary.csv): "
              f"[{LOD_nM:.1f}, {ULOQ_nM:.1f}] nM")
        if LOD_nM <= boundary <= ULOQ_nM:
            print("Decision boundary falls WITHIN the device's measurable range — good.")
        else:
            print("WARNING: decision boundary falls OUTSIDE the device's measurable range "
                  "at the current (placeholder) k_surf_eff — classification would be unreliable "
                  "near the clinically relevant threshold until the electrode is calibrated.")
        plot_mmp9_classifier(calibration_LOD_nM=LOD_nM, calibration_ULOQ_nM=ULOQ_nM)
    except (FileNotFoundError, StopIteration):
        print("\n(calibration_summary.csv not found — run calibration_curve.py first "
              "for the device-range integration check.)")
        plot_mmp9_classifier()

    # Illustrative secondary model
    print("\n[ILLUSTRATIVE ONLY — no HNE channel exists on the real device]")
    print("Layer 0's own conclusion: HNE is elevated in only ~50% of chronic samples,")
    print("so it is 'not a reliable standalone discriminator'. Shown only to demonstrate")
    print("how little a hypothetical second channel would actually add:\n")
    for name, (c9, chne) in {"Chronic, HNE elevated": (31.9, 30.0),
                              "Chronic, HNE normal": (31.9, 1.0)}.items():
        p = classify_mmp9_hne(c9, chne)
        print(f"  {name}: [MMP-9]={c9} nM, [HNE]={chne} nM -> "
              f"P(Chronic)={p['Chronic']*100:.1f}%, P(Healing)={p['Healing']*100:.1f}%")

    return boundary


if __name__ == "__main__":
    main()
