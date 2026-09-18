"""
average_ket.py -- Layer 8, Section III (Sec. 8.3 of the planning notes)
========================================================================
Propagates the Monte Carlo P(r) from probe_conformation.py through Layer
10's Marcus/Smalley distance-decay law for k_ET(r), and quantifies how much
the conformational distribution boosts the effective ET rate relative to
evaluating k_ET at the mean distance <r> -- the central physical point of
this sub-layer: <k_ET> != k_ET(<r>) because k_ET(r) is convex (exponential)
in r, so short-r excursions of the flexible tether contribute
disproportionately (Jensen's inequality).

k_ET(r) parameters are taken directly from Layer 10 (Sec. II, Eq. II.5),
NOT re-derived here, so that Layer 8's output is consistent with what
Layer 10 already assumes:
    k_ET(r) = k0 * exp(-beta * r),   k0 = 6.0e8 s^-1,  beta = 0.95 Ang^-1
                                                       (Smalley et al. 1995)
Layer 10 Table 3 evaluates this at a single placeholder <d>; this script
supplies the P(r)-averaged value that should replace that placeholder.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import probe_conformation as pc

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Layer 10 Marcus/Smalley parameters (Eq. II.5 of the Layer 10 report) ──
k0_ET = 6.0e8          # s^-1  (Smalley 1995, kT,s,n=0)
beta  = 0.95e10        # m^-1  (0.95 Ang^-1 = 9.5 nm^-1 = 0.95e10 m^-1)


def k_ET(r_m):
    """Marcus/Smalley distance-decay ET rate constant, r in metres."""
    return k0_ET * np.exp(-beta * r_m)


def average_kET_montecarlo(r_samples):
    """<k_ET> = k0 <exp(-beta r)>, from MC ensemble of r (metres)."""
    return k0_ET * np.mean(np.exp(-beta * r_samples))


def kET_at_mean_r(r_samples):
    """Naive estimate: k_ET evaluated at the mean distance <r>."""
    return k_ET(np.mean(r_samples))


def bootstrap_ci(r_samples, func, n_boot=2000, seed=1):
    """Simple percentile bootstrap CI for a scalar functional of r_samples."""
    rng = np.random.default_rng(seed)
    n = len(r_samples)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = func(r_samples[idx])
    return np.percentile(vals, [2.5, 50, 97.5])


def integrate_analytic_wlc_Pr(r_grid, L_c, l_p):
    """
    Semi-analytic radial WLC distribution (Kratky-Porod / Gaussian-chain
    limit), used only as a numerical-integration cross-check against the
    Monte Carlo estimate. For short chains (L_c/l_p ~ 10-20, our regime)
    the true WLC P(r) is non-Gaussian; the standard closed-form
    approximation used here is the Gaussian-chain radial density built
    from the same WLC <r^2>(L_c, l_p), i.e.
        P(r) = 4 pi r^2 (3/(2 pi <r^2>))^(3/2) exp(-3 r^2 / (2<r^2>))
    This is the textbook substitute for the exact Kratky-Porod transfer-
    matrix P(r) referenced in the planning notes (Sec. 8.3); it is not
    exact at this L_c/l_p, which is exactly why the Monte Carlo estimate
    above -- not this analytic cross-check -- is used as Layer 8's primary
    <k_ET> output.
    """
    mean_r2 = wlc_mean_r2_total(L_c, l_p)
    pref = (3 / (2 * np.pi * mean_r2)) ** 1.5
    return 4 * np.pi * r_grid**2 * pref * np.exp(-3 * r_grid**2 / (2 * mean_r2))


def wlc_mean_r2_total(L_c, l_p):
    x = L_c / l_p
    return 2 * l_p * L_c * (1 - (l_p / L_c) * (1 - np.exp(-x)))


def run(n_samples=12000):
    r_samples = pc.monte_carlo_P_r(n_samples=n_samples)

    kmc = average_kET_montecarlo(r_samples)
    knaive = kET_at_mean_r(r_samples)
    ratio = kmc / knaive

    lo, med, hi = bootstrap_ci(r_samples, average_kET_montecarlo, n_boot=1000)

    # analytic cross-check using the additive-variance Gaussian-chain P(r)
    r_grid = np.linspace(1e-12, pc.L_c_total, 4000)
    P_gauss = integrate_analytic_wlc_Pr(r_grid, pc.L_c_total, pc.lp_eff)
    P_gauss /= np.trapezoid(P_gauss, r_grid)  # renormalise on the truncated grid
    k_analytic = k0_ET * np.trapezoid(np.exp(-beta * r_grid) * P_gauss, r_grid)

    print("=" * 70)
    print("Layer 8.3: <k_ET> from the WLC conformational ensemble")
    print("=" * 70)
    print(f"k0 (Smalley 1995, Layer 10)      = {k0_ET:.2e} s^-1")
    print(f"beta (Smalley 1995, Layer 10)    = {beta*1e-9:.2f} nm^-1  "
          f"(= 0.95 A^-1)")
    print()
    print(f"<r>  (Monte Carlo, n={n_samples})       = {r_samples.mean()*1e9:.3f} nm")
    print(f"k_ET(<r>)   [naive point estimate]     = {knaive:.4e} s^-1")
    print(f"<k_ET>      [MC average over P(r)]     = {kmc:.4e} s^-1  "
          f"(95% CI [{lo:.3e}, {hi:.3e}])")
    print(f"<k_ET> / k_ET(<r>)  ratio               = {ratio:.1f}x")
    print()
    print(f"Cross-check, analytic Gaussian-chain P(r) with l_p,eff:")
    print(f"  <k_ET>_analytic                        = {k_analytic:.4e} s^-1  "
          f"({100*(k_analytic-kmc)/kmc:+.1f}% vs Monte Carlo)")
    print()
    print("Interpretation: the flexible tether's short-r excursions dominate")
    print("the ET average; a naive estimate using only <r> underestimates the")
    print("true <k_ET> by roughly the ratio above. This is the Layer 8 value")
    print("that should replace the single-<d> placeholder in Layer 10 Table 3.")

    # sensitivity of the ratio to beta (does the qualitative conclusion survive
    # uncertainty in beta, e.g. if the true PEG/peptide beta differs from the
    # alkanethiol-calibrated Smalley value used here; Layer 10 Limitation 2)
    beta_scan = np.linspace(0.3e10, 1.3e10, 40)
    ratio_scan = np.array([
        np.mean(np.exp(-b * r_samples)) / np.exp(-b * r_samples.mean())
        for b in beta_scan
    ])

    plot_average_ket(r_samples, beta_scan, ratio_scan, ratio)

    return dict(k_naive=knaive, k_mc=kmc, ratio=ratio, r_mean_nm=r_samples.mean()*1e9,
                k_mc_ci=(lo, hi), k_analytic=k_analytic)


def plot_average_ket(r_samples, beta_scan, ratio_scan, ratio_at_layer10_beta):
    r_nm = r_samples * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax_twin = ax.twinx()
    counts, bins, _ = ax.hist(r_nm, bins=80, density=True, color="#3b6fa0",
                                alpha=0.55, label="$P(r)$ (Monte Carlo)")
    r_fine = np.linspace(1e-3, r_nm.max(), 400)
    ax_twin.plot(r_fine, k_ET(r_fine * 1e-9), "r-", lw=2, label="$k_{ET}(r)$")
    ax_twin.set_yscale("log")
    ax.set_xlabel("MB-to-Au distance $r$ (nm)")
    ax.set_ylabel("Probability density", color="#3b6fa0")
    ax_twin.set_ylabel("$k_{ET}(r)$ (s$^{-1}$)", color="r")
    ax.set_title("A — $P(r)$ vs. the exponentially-decaying $k_{ET}(r)$")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    ax2 = axes[1]
    ax2.plot(beta_scan * 1e-9, ratio_scan, "b-", lw=2)
    ax2.axvline(0.95, color="k", ls="--", lw=1.2,
                label=r"Layer 10 $\beta$ = 0.95 nm$^{-1}$ (Smalley 1995)")
    ax2.axhline(ratio_at_layer10_beta, color="gray", ls=":", lw=1)
    ax2.set_xlabel(r"$\beta$ (nm$^{-1}$)")
    ax2.set_ylabel(r"$\langle k_{ET} \rangle \, / \, k_{ET}(\langle r \rangle)$")
    ax2.set_title("B — Boost ratio vs. tunnelling decay constant $\\beta$")
    ax2.legend(fontsize=8)
    ax2.set_yscale("log")

    fig.suptitle("Layer 8.3: Conformationally-averaged $k_{ET}$ vs. the naive "
                 "point estimate at $\\langle r \\rangle$", fontsize=10)
    plt.tight_layout()
    out = FIGURES / "fig_L8_average_ket.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    run()
