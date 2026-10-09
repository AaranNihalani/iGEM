"""
wiki_figs.py -- figures for the Surface-Enzyme Kinetics wiki page.

Reproduces the six figures from surface_enzyme_kinetics_writeup.tex exactly
(progress curves vs lambda; the four compare_curves panels at lambda = 1.5,
3, 10, infinity; the Phi(x) lemma plot). Colours come from the same
site-anchored, validated palette as the SWV and substrate-predictor report
figures (report_lib.py / fig_common.py); the report's own tikz figures used
unvalidated colours, which the site's colour-scheme pass intentionally
replaces here, matching the dry-lab figures elsewhere on the wiki.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parents[2] / "reports" / "figures"
OUT.mkdir(exist_ok=True)

# Validated categorical slots (node scripts/validate_palette.js, dataviz skill):
# {green, gold, red, purple, orange} passes lightness/chroma/CVD/normal-vision
# checks as a 5-set; {blue, green} is excluded (fails the normal-vision floor,
# as already found fixing the SWV and substrate-predictor figures).
GREEN, GOLD, RED, PURPLE, ORANGE = "#15876A", "#C79A2E", "#e34948", "#4a3aa7", "#eb6834"
INK, MUTE, GRID, PAPER = "#173B34", "#8A9892", "#E3E8E1", "#F5EADA"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8,
    "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": MUTE,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- the maths
def theta_exact(tau, lam):
    """Eq. 9: the exact biexponential solution, theta_P(tau; lambda)."""
    tau = np.asarray(tau, dtype=float)
    if lam == 1.0:
        return 1.0 - np.exp(-tau / 2.0)
    if np.isinf(lam):
        return 1.0 - (1.0 + tau / 2.0) * np.exp(-tau / 2.0)
    return 1.0 - 0.5 * ((1 + lam) * np.exp(-tau / 2.0)
                         + (1 - lam) * np.exp(-((lam + 1) / (lam - 1)) * (tau / 2.0)))


def theta_steady(tau, lam):
    """Eq. 6, rewritten in terms of (tau, lambda) per Section 6.1."""
    return 1.0 - np.exp(-(1.0 + 1.0 / lam) * (tau / 4.0))


def phi(x):
    """Section 6.2: Phi(x) = 4x / (1+x)^2."""
    return 4.0 * x / (1.0 + x) ** 2


# ---------------------------------------------------------------- Figure 1
def fig_progress_curves():
    tau = np.linspace(0, 10, 400)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    series = [
        (1.0, "$\\lambda = 1$", GREEN, "-"),
        (1.5, "$\\lambda = 1.5$", GOLD, "-"),
        (3.0, "$\\lambda = 3$", RED, "-"),
        (10.0, "$\\lambda = 10$", PURPLE, "-"),
        (np.inf, "$\\lambda \\to \\infty$", ORANGE, "--"),
    ]
    for lam, label, col, ls in series:
        ax.plot(tau, theta_exact(tau, lam), color=col, ls=ls, lw=2.0, label=label)
    ax.set_xlabel(r"Dimensionless time, $\tau$")
    ax.set_ylabel(r"Fractional product coverage, $\theta_{\mathrm{P}}$")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Exact progress curves become more S-shaped as $\\lambda$ grows")
    fig.tight_layout()
    save(fig, "swvr_sek_01_progress_curves")


# ---------------------------------------------------------------- Figures 2-5
def fig_comparison_one(lam, label, tag, fname):
    """One compare_curvesN panel from the report: exact (solid) vs steady-state
    (dashed), both rewritten in terms of the same lambda, at a single lambda."""
    tau = np.linspace(0, 10, 400)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(tau, theta_exact(tau, lam), color=GREEN, lw=2.0, label="Exact solution")
    ax.plot(tau, theta_steady(tau, lam), color=INK, lw=1.5, ls="--", label="Steady-state solution")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\theta_{\mathrm{P}}$")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"Progress curves when {label}")
    fig.tight_layout()
    save(fig, fname)


def fig_comparisons():
    fig_comparison_one(1.5, "$\\lambda = 1.5$", "b", "swvr_sek_02_compare_curves1")
    fig_comparison_one(3.0, "$\\lambda = 3$", "c", "swvr_sek_03_compare_curves2")
    fig_comparison_one(10.0, "$\\lambda = 10$", "d", "swvr_sek_04_compare_curves3")
    fig_comparison_one(np.inf, "$\\lambda \\to \\infty$", "e", "swvr_sek_05_compare_curves4")


# ---------------------------------------------------------------- Figure 6
def fig_phi():
    x = np.linspace(1e-6, 15, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(x, phi(x), color=GREEN, lw=2.0)
    ax.axhline(1.0, color=MUTE, lw=0.8, ls=":")
    ax.plot([1.0], [1.0], marker="o", ms=5, color=RED, zorder=3)
    ax.annotate("max at $x=1$", xy=(1.0, 1.0), xytext=(2.3, 0.9), fontsize=8, color=INK)
    ax.set_xlabel(r"$x = [\mathrm{E}]_t / K_M$")
    ax.set_ylabel(r"$\Phi(x) = 4x / (1+x)^2$")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.15)
    ax.set_title(r"Graph of $f(x) = 4x / (1 + x)^2$")
    fig.tight_layout()
    save(fig, "swvr_sek_06_phi_function")


if __name__ == "__main__":
    fig_progress_curves()
    fig_comparisons()
    fig_phi()
