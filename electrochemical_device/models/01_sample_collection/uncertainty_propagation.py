"""
uncertainty_propagation.py
Layer 1 of 17 -- Eton iGEM 2026 Dry Lab.

Monte Carlo propagation of pre-analytical uncertainty (V_abs, eta_elut,
k_deg, t_transit, D_f) through to [E]_chamber, for both confirmed
clinical wound states (Layer 0 ODE output, t = 30 min: chronic 31.9 nM,
healing 6.2 nM). Also produces a first-order (binned conditional-variance)
sensitivity decomposition -- a single-layer preview of the full Sobol
analysis planned for Layer 16 (contents.md Sec. 1.7 / README Layer 16).

Produces:
  figures/fig1_chamber_concentration_mc.png
  figures/fig2_sensitivity_and_transit.png

and prints the summary statistics quoted in the Layer 1 report.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from preanalytical_model import (
    v_abs_lognormal_params, chamber_concentration, composite_recovery_factor,
    ETA_ELUT_POINT, ratio_enrichment_factor, ph_chamber, filter_pressure_drop,
)

RNG = np.random.default_rng(20260917)
N = 300_000

# ----------------------------------------------------------------------
# Confirmed Layer 0 wound-fluid concentrations (t = 30 min read; Ladwig
# 2002 parametrisation, as used identically in Layer 9/13/15)
# ----------------------------------------------------------------------
E_WOUND_CHRONIC = 31.9   # nM
E_WOUND_HEALING = 6.2    # nM

# Layer 13/15's own device detection floor, in [E]-space, at the
# synthetic placeholder k_surf = 2.0e4 M^-1 s^-1 (Layer 15 Sec. III):
LOD_NM = 23.9
ULOQ_NM = 168.2


def sample_inputs(n, t_transit_range):
    """Draw n Monte Carlo samples of the five pre-analytical inputs."""
    mu, sigma = v_abs_lognormal_params(20.0, 200.0, ci=0.90)
    V_abs = RNG.lognormal(mean=mu, sigma=sigma, size=n)

    # eta_elut: literature band is a range, not a distribution shape, so a
    # triangular distribution (mode at the contents.md-quoted central
    # value) is used as a conservative, weakly-informative choice.
    lo, hi = 0.4, 0.7
    mode = ETA_ELUT_POINT["MMP9"]
    eta_elut = RNG.triangular(lo, mode, hi, size=n)

    # k_deg: contents.md gives 0.001-0.01 /min with no further shape
    # information -> uniform (maximum-entropy choice given only bounds).
    k_deg = RNG.uniform(0.001, 0.01, size=n)

    t_transit = RNG.uniform(t_transit_range[0], t_transit_range[1], size=n)

    # D_f: contents.md gives a single worked value (0.8); +-15% relative
    # variability is assumed here for loading-volume pipetting tolerance
    # (not itself sourced from contents.md -- flagged in Limitations).
    D_f = np.clip(RNG.normal(0.8, 0.8 * 0.15, size=n), 1e-3, None)

    return V_abs, eta_elut, k_deg, t_transit, D_f


def run_scenario(E_wound, t_transit_range):
    V_abs, eta_elut, k_deg, t_transit, D_f = sample_inputs(N, t_transit_range)
    E_chamber = chamber_concentration(E_wound, eta_elut, V_abs, k_deg, t_transit, D_f)
    return E_chamber, (V_abs, eta_elut, k_deg, t_transit, D_f)


def first_order_sensitivity(E_chamber, inputs, names, n_bins=25):
    """
    Binned estimator of the first-order Sobol index S_i = Var(E[Y|Xi]) / Var(Y),
    computed directly from one shared Monte Carlo sample (no extra model
    evaluations needed). This is an approximation (accuracy improves with
    n_bins and N) but is enough to rank the dominant uncertainty sources,
    matching the qualitative question contents.md Sec. 1.7 raises ahead of
    the full Layer 16 Sobol analysis.
    """
    var_total = np.var(E_chamber)
    indices = {}
    for name, x in zip(names, inputs):
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = E_chamber[order]
        bin_edges = np.quantile(x_sorted, np.linspace(0, 1, n_bins + 1))
        bin_idx = np.clip(np.digitize(x_sorted, bin_edges[1:-1]), 0, n_bins - 1)
        cond_means = np.array([y_sorted[bin_idx == b].mean() for b in range(n_bins)])
        cond_counts = np.array([(bin_idx == b).sum() for b in range(n_bins)])
        var_of_cond_mean = np.average((cond_means - y_sorted.mean()) ** 2, weights=cond_counts)
        indices[name] = var_of_cond_mean / var_total
    return indices


def main():
    names = ["V_abs", "eta_elut", "k_deg", "t_transit", "D_f"]

    # --- Scenario A: clinic-typical transit (0-15 min) ---
    E_chr_clinic, inputs_chr_clinic = run_scenario(E_WOUND_CHRONIC, (0.0, 15.0))
    E_heal_clinic, inputs_heal_clinic = run_scenario(E_WOUND_HEALING, (0.0, 15.0))

    # --- Scenario B: field-worst-case transit (0-120 min) ---
    E_chr_field, _ = run_scenario(E_WOUND_CHRONIC, (0.0, 120.0))
    E_heal_field, _ = run_scenario(E_WOUND_HEALING, (0.0, 120.0))

    def summarise(x, label):
        gm = np.exp(np.mean(np.log(x)))
        gsd = np.exp(np.std(np.log(x)))
        rel_sd = np.std(x) / np.mean(x)
        p05, p50, p95 = np.percentile(x, [5, 50, 95])
        p_below_lod = np.mean(x < LOD_NM)
        print(f"{label:28s} mean={np.mean(x):7.3f} nM  geomean={gm:7.3f} nM  "
              f"median={p50:7.3f}  [P5,P95]=[{p05:6.3f},{p95:6.3f}]  "
              f"rel.SD={rel_sd*100:5.1f}%   P([E]<LOD)={p_below_lod*100:5.1f}%")
        return dict(mean=np.mean(x), geomean=gm, gsd=gsd, rel_sd=rel_sd,
                    p05=p05, p50=p50, p95=p95, p_below_lod=p_below_lod)

    print("\n--- Scenario A: clinic-typical transit (0-15 min) ---")
    s_chr_clinic = summarise(E_chr_clinic, "[MMP-9]_chamber, chronic")
    s_heal_clinic = summarise(E_heal_clinic, "[MMP-9]_chamber, healing")

    print("\n--- Scenario B: field-worst-case transit (0-120 min) ---")
    s_chr_field = summarise(E_chr_field, "[MMP-9]_chamber, chronic")
    s_heal_field = summarise(E_heal_field, "[MMP-9]_chamber, healing")

    # --- Sensitivity decomposition (Scenario A, chronic state) ---
    sens = first_order_sensitivity(E_chr_clinic, inputs_chr_clinic, names)
    print("\nFirst-order sensitivity indices (chronic, clinic-transit scenario):")
    for k, v in sorted(sens.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s}: S1 = {v:.3f}")

    # Also report R_total distribution stats (composite recovery factor)
    V_abs, eta_elut, k_deg, t_transit, D_f = sample_inputs(N, (0.0, 15.0))
    R_total = composite_recovery_factor(eta_elut, V_abs, k_deg, t_transit, D_f)
    print(f"\nComposite recovery factor R_total: mean={np.mean(R_total)*100:.2f}%, "
          f"median={np.median(R_total)*100:.2f}%, "
          f"[P5,P95]=[{np.percentile(R_total,5)*100:.2f}%,{np.percentile(R_total,95)*100:.2f}%]")

    # ------------------------------------------------------------------
    # Figure 1: [E]_chamber distributions vs. LOD/ULOQ (clinic scenario)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.logspace(np.log10(0.02), np.log10(60), 70)
    ax = axes[0]
    ax.hist(E_chr_clinic, bins=bins, alpha=0.55, color="#b5443c", label="Chronic (wound 31.9 nM)")
    ax.hist(E_heal_clinic, bins=bins, alpha=0.55, color="#3c7ab5", label="Healing (wound 6.2 nM)")
    ax.axvline(LOD_NM, color="k", ls="--", lw=1.3, label=f"Device LOD = {LOD_NM} nM")
    ax.set_xscale("log")
    ax.set_xlabel("[MMP-9]$_{chamber}$ (nM)")
    ax.set_ylabel("Monte Carlo count")
    ax.set_title("Layer 1.7/1.8: [E]$_{chamber}$ after full\npre-analytical chain (0-15 min transit)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for label, arr, color in [("Chronic, 0-15 min transit", E_chr_clinic, "#b5443c"),
                               ("Chronic, 0-120 min transit", E_chr_field, "#e8a24c"),
                               ("Healing, 0-15 min transit", E_heal_clinic, "#3c7ab5"),
                               ("Healing, 0-120 min transit", E_heal_field, "#6cc0c4")]:
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf, label=label, color=color, lw=1.6)
    ax.axvline(LOD_NM, color="k", ls="--", lw=1.3, label=f"LOD = {LOD_NM} nM")
    ax.set_xscale("log")
    ax.set_xlabel("[MMP-9]$_{chamber}$ (nM)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Effect of transit-time scenario on\n[E]$_{chamber}$ (CDF)")
    ax.legend(fontsize=7.3, loc="lower right")

    fig.suptitle("Layer 1: Monte Carlo pre-analytical chain (V_abs, $\\eta_{elut}$, $k_{deg}$, $t_{transit}$, $D_f$)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("../figures/fig1_chamber_concentration_mc.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: sensitivity bar chart + recovery-factor vs transit time
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    order = sorted(sens.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in order]
    vals = [v for _, v in order]
    label_map = {"V_abs": r"$V_{abs}$ (swab volume)", "eta_elut": r"$\eta_{elut}$ (elution eff.)",
                 "k_deg": r"$k_{deg}$ (transit decay)", "t_transit": r"$t_{transit}$ (transit time)",
                 "D_f": r"$D_f$ (chamber loading)"}
    ax.barh([label_map[l] for l in labels], vals, color="#5a7fa6")
    ax.set_xlabel("First-order sensitivity index $S_1$")
    ax.set_title("Layer 1.7: variance contribution to\n[E]$_{chamber}$ (chronic, clinic-transit)")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.25)

    ax = axes[1]
    t_grid = np.linspace(0, 120, 200)
    k_lo, k_mid, k_hi = 0.001, 0.0055, 0.01
    for k_deg_val, style in [(k_lo, "--"), (k_mid, "-"), (k_hi, ":")]:
        decay = np.exp(-k_deg_val * t_grid)
        ax.plot(t_grid, decay * 100, style, color="#b5443c",
                label=f"$k_{{deg}}$={k_deg_val:.3f} min$^{{-1}}$")
    ax.axvspan(0, 15, color="grey", alpha=0.15, label="Typical clinic transit")
    ax.set_xlabel("Transit time (min)")
    ax.set_ylabel("Analyte surviving transit (%)")
    ax.set_title("Layer 1.5: first-order transit decay\nacross the literature $k_{deg}$ range")
    ax.legend(fontsize=8)

    fig.suptitle("Layer 1: sensitivity ranking and transit-decay sweep", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("../figures/fig2_sensitivity_and_transit.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: inhibitor ratio-shift bar chart + pH-mixing curve
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    pairs = [("SLPI / HNE", ETA_ELUT_POINT["SLPI"], ETA_ELUT_POINT["HNE"]),
             ("TIMP-1 / MMP-9", ETA_ELUT_POINT["TIMP1"], ETA_ELUT_POINT["MMP9"]),
             ("A1AT / HNE", ETA_ELUT_POINT["A1AT"], ETA_ELUT_POINT["HNE"])]
    labels = [p[0] for p in pairs]
    enrich = [ratio_enrichment_factor(p[1], p[2]) for p in pairs]
    colors = ["#b5443c" if e > 1 else "#3c7ab5" for e in enrich]
    ax.bar(labels, enrich, color=colors)
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_ylabel("Enrichment in chamber vs.\nwound-fluid ratio ($\\eta_I/\\eta_E$)")
    ax.set_title("Layer 1.3: inhibitor/enzyme ratio shift\nfrom differential elution efficiency")
    for i, v in enumerate(enrich):
        ax.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9)

    ax = axes[1]
    pH_wound_grid = np.linspace(6.5, 8.2, 60)
    for V_wound, style in [(20, ":"), (50, "-"), (100, "--")]:
        pH_out = [ph_chamber(pw, 7.4, V_wound, 200.0) for pw in pH_wound_grid]
        ax.plot(pH_wound_grid, pH_out, style, color="#5a7fa6",
                label=f"$V_{{wound}}$={V_wound} uL into 200 uL PBS")
    ax.plot(pH_wound_grid, pH_wound_grid, color="grey", lw=0.8, label="No buffering (pH$_{chamber}$=pH$_{wound}$)")
    ax.axhline(7.4, color="k", lw=0.8, ls=":", label="Full buffering (pH$_{chamber}$=7.4)")
    ax.set_xlabel("Wound-fluid pH")
    ax.set_ylabel("pH$_{chamber}$ (mixed eluate)")
    ax.set_title("Layer 1.4: exact proton mass-balance\npH vs. complete-buffering assumption")
    ax.legend(fontsize=7.3, loc="upper left")

    fig.suptitle("Layer 1: inhibitor co-elution ratio shift and elution-buffer pH mixing", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("../figures/fig3_inhibitor_and_ph.png", dpi=200)
    plt.close(fig)

    print("\nFigures written to ../figures/")


if __name__ == "__main__":
    main()
