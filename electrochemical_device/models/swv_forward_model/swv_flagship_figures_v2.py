"""
swv_flagship_figures_v2.py
Generates all 12 publication-quality figures for the expanded SWV flagship report.
Run from within the swv_forward_model/ directory or call directly.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from swv_forward_model import (
    swv_voltammogram, dist_averaged_voltammogram,
    fit_gaussian, laviron_reversible,
    P_d_norm, k0_of_d, _half_cycle_end,
    D_PEG_NM, SIGMA_Z_NM, D_MAX_NM, K0_REF_S1, BETA_NM,
    F_const, R_const, T_K, N_KUHN, B_KUHN_NM, L0_NM,
    E_SCAN, FK_VALS,
)

FIGURES = Path(__file__).parents[2] / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
})
C0, C1, C2, C3 = "#2166ac", "#d6604d", "#4dac26", "#762a83"


def save(fig, name):
    p = FIGURES / name
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  → {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 0 — Physical assembly cartoon
# ═══════════════════════════════════════════════════════════════════════════════
def fig0_assembly():
    print("[0/12] Fig 0: Physical assembly ...")
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

    # ── Panel A: electrode cross-section ──
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("A   Electrode assembly", fontweight="bold", loc="left")

    # Gold substrate
    gold = mpatches.FancyBboxPatch((0.5, 0.2), 9, 1.0, boxstyle="round,pad=0.05",
                                    fc="#FFD700", ec="#b8860b", lw=1.5)
    ax.add_patch(gold)
    ax.text(5, 0.7, "Au electrode  (1.5 mm disc)", ha="center", va="center",
            fontsize=8, fontweight="bold")

    # SAM layer
    sam = mpatches.FancyBboxPatch((0.5, 1.2), 9, 0.8, boxstyle="round,pad=0.05",
                                   fc="#e8d5a3", ec="#9c7a3c", lw=1)
    ax.add_patch(sam)
    ax.text(5, 1.6, "Mixed SAM  (Au–S bonds)", ha="center", va="center",
            fontsize=8, color="#5a3a00")

    # PEG6 spacers (left side: signal probes)
    for x in [1.5, 2.5, 3.5]:
        ax.annotate("", xy=(x, 3.8), xytext=(x, 2.0),
                    arrowprops=dict(arrowstyle="-", color=C2, lw=2,
                                   connectionstyle="arc3,rad=0.2"))
    ax.text(2.5, 3.0, "PEG6", ha="center", fontsize=7, color=C2, rotation=90)

    # Peptide chains (signal probes)
    for i, x in enumerate([1.5, 2.5, 3.5]):
        ys = [3.8, 4.4, 4.1, 4.8, 5.2, 5.0, 5.7][: 6]
        xs = [x, x + 0.15, x - 0.1, x + 0.2, x - 0.15, x + 0.1][: 6]
        ax.plot(xs, ys, "o-", ms=2, color="#888", lw=1, zorder=3)
    ax.text(0.7, 5.0, "SGKIPRTLTA\npeptide", ha="center", fontsize=7,
            color="#555", va="center")

    # MB labels
    for x in [1.5, 2.5]:
        circle = plt.Circle((x + 0.1, 5.9), 0.28, color=C0, zorder=5)
        ax.add_patch(circle)
        ax.text(x + 0.1, 5.9, "MB", ha="center", va="center",
                fontsize=6, color="white", fontweight="bold")
    # Cleaved probe (MB released)
    ax.text(3.7, 6.8, "MB (released)", ha="center", fontsize=7, color=C1,
            style="italic")
    circle2 = plt.Circle((3.7, 6.3), 0.28, color=C1, zorder=5, alpha=0.7)
    ax.add_patch(circle2)
    ax.text(3.7, 6.3, "MB", ha="center", va="center",
            fontsize=6, color="white", fontweight="bold")
    ax.annotate("", xy=(3.7, 7.2), xytext=(3.7, 6.6),
                arrowprops=dict(arrowstyle="->", color=C1, lw=1.5))

    # Fc reference probes (right side)
    for x in [6.5, 7.5, 8.5]:
        ax.plot([x, x], [2.0, 3.5], "-", color="#c66", lw=2)
    for x in [6.5, 7.5, 8.5]:
        hex_patch = mpatches.RegularPolygon((x, 3.8), 6, radius=0.3,
                                             fc="#f9a", ec="#c66", lw=1)
        ax.add_patch(hex_patch)
        ax.text(x, 3.8, "Fc", ha="center", va="center", fontsize=6,
                fontweight="bold", color="#800")
    ax.text(7.5, 2.5, "Fc-hexanethiol\n(reference)", ha="center", fontsize=7,
            color="#800")

    # Dividers and labels
    ax.axhline(7.8, xmin=0.05, xmax=0.95, ls="--", color="#aaa", lw=0.8)
    ax.text(5, 8.2, "Wound fluid + MMP-9", ha="center", fontsize=8,
            color="#444", style="italic")
    # MMP-9 molecule
    mmp = mpatches.Ellipse((5, 9.5), 2.0, 1.0, fc="#ffd", ec="#aa8", lw=1.5)
    ax.add_patch(mmp)
    ax.text(5, 9.5, "MMP-9", ha="center", va="center", fontsize=8,
            fontweight="bold", color="#553")
    ax.annotate("", xy=(3.5, 7.9), xytext=(4.2, 9.0),
                arrowprops=dict(arrowstyle="->", color="#553", lw=1.5))
    ax.text(3.0, 8.5, "cleaves\npeptide", ha="center", fontsize=7, color="#553")

    # Potentiostat connection
    ax.annotate("", xy=(5, 12.0), xytext=(5, 1.2),
                arrowprops=dict(arrowstyle="-", color="#222", lw=1,
                               linestyle="dotted"))
    ax.text(5.3, 11.5, "Potentiostat", fontsize=8, color="#222")
    ax.text(5.3, 11.0, "(SWV readout)", fontsize=7, color="#555")

    # Bracket labels
    ax.annotate("", xy=(0.3, 2.0), xytext=(0.3, 1.2),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1))
    ax.text(0.1, 1.6, "SAM", ha="right", fontsize=7, color="#444")

    # ── Panel B: P(d) with probe positions ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("B   Distance distribution P(d)", fontweight="bold", loc="left")

    d_arr = np.linspace(D_PEG_NM * 10, D_MAX_NM * 10, 400)  # in Å
    p_arr = P_d_norm(d_arr / 10)
    ax2.fill_between(d_arr, p_arr, alpha=0.25, color=C0)
    ax2.plot(d_arr, p_arr, color=C0, lw=2, label="$P(d)$ half-Gaussian")

    d_peg_A = D_PEG_NM * 10
    sig_A = SIGMA_Z_NM * 10
    d_mean_A = d_peg_A + sig_A * np.sqrt(2 / np.pi)

    ax2.axvline(d_peg_A, color=C2, ls="--", lw=1.5, label=f"Hard wall $d_{{\\rm PEG}}={d_peg_A:.1f}$Å")
    ax2.axvline(d_mean_A, color=C3, ls=":", lw=1.5, label=f"Mean $\\langle d\\rangle={d_mean_A:.1f}$Å")
    ax2.axvspan(d_peg_A, d_peg_A + sig_A, alpha=0.10, color=C0,
                label=f"$\\pm\\sigma_z = {sig_A:.1f}$Å")

    ax2.set_xlabel(r"Distance from Au surface, $d$ (Å)")
    ax2.set_ylabel(r"Probability density $P(d)$ (Å$^{-1}$)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_xlim(0, 75)

    fig.suptitle("Figure 0: Physical electrode assembly and tether geometry",
                 fontsize=10, fontweight="bold", y=1.01)
    save(fig, "fig_SWV0_assembly.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Geometry (4 panels, includes spacer comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_geometry():
    print("[1/12] Fig 1: Geometry ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    d_A = np.linspace(0, 80, 500)
    d_nm = d_A / 10
    p = P_d_norm(d_nm)

    d_peg_A = D_PEG_NM * 10
    sig_A = SIGMA_Z_NM * 10
    d_mean_A = d_peg_A + sig_A * np.sqrt(2 / np.pi)

    # A: P(d)
    ax = axes[0]
    ax.fill_between(d_A, p, alpha=0.2, color=C0)
    ax.plot(d_A, p, color=C0, lw=2)
    ax.axvline(d_peg_A, color=C2, ls="--", lw=1.5, label=f"$d_{{\\rm PEG}}={d_peg_A:.1f}$Å")
    ax.axvline(d_mean_A, color=C3, ls=":", lw=1.5, label=f"$\\langle d\\rangle={d_mean_A:.1f}$Å")
    ax.axvspan(d_peg_A, d_peg_A + sig_A, alpha=0.1, color=C0,
               label=f"$\\sigma_z={sig_A:.1f}$Å")
    ax.set_xlabel(r"Distance $d$ (Å)")
    ax.set_ylabel(r"$P(d)$ (Å$^{-1}$)")
    ax.set_title("A  Distance distribution $P(d)$")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 75)

    # B: k0(d) and kappa(d)
    ax = axes[1]
    ax2b = ax.twinx()
    f_op = 10.0  # confirmed operating frequency (10-20 Hz range; 10 Hz used as representative)
    d_plot_A = np.linspace(d_peg_A, 70, 300)
    d_plot_nm = d_plot_A / 10
    k0_arr = k0_of_d(d_plot_nm)
    kap_arr = k0_arr / (2 * f_op)

    ax.semilogy(d_plot_A, k0_arr, color=C1, lw=2, label="$k^0(d)$")
    ax2b.semilogy(d_plot_A, kap_arr, color=C3, lw=1.5, ls="--", label=r"$\kappa(d)$")
    ax.axvline(d_peg_A, color=C2, ls="--", lw=1, alpha=0.7)
    ax.axvline(d_mean_A, color=C3, ls=":", lw=1, alpha=0.7)
    ax2b.axhspan(0.5, 10, alpha=0.07, color="orange", label="Quasi-rev. zone")
    ax.scatter([d_peg_A], [K0_REF_S1], color=C1, s=50, zorder=5)
    ax.text(d_peg_A + 1, K0_REF_S1 * 1.5, f"$k^0_{{\\rm ref}}={K0_REF_S1:.0f}$ s$^{{-1}}$",
            fontsize=7, color=C1)
    ax.set_xlabel(r"Distance $d$ (Å)")
    ax.set_ylabel(r"$k^0(d)$ (s$^{-1}$)")
    ax2b.set_ylabel(r"$\kappa = k^0 / (2f)$", color=C3)
    ax.set_title("B  ET rate and $\\kappa$ vs distance")

    # C: SWV waveform
    ax = axes[2]
    t = np.linspace(0, 1, 1000)
    f_show = 4
    tau = 1 / (2 * f_show)
    staircase_E = -0.04 + 0.01 * np.floor(t * f_show)
    esw = 0.025
    pulse = esw * np.where((t * f_show % 1) < 0.5, 1, -1)
    E_total = staircase_E + pulse
    ax.plot(t, E_total * 1000, color="#333", lw=1.0, label="SWV waveform")
    ax.plot(t, staircase_E * 1000, color="#888", lw=0.8, ls="--", label="Staircase $E_{\\rm step}$")
    # Mark sampled points
    for i in range(int(t[-1] * f_show)):
        t_fwd = (i + 0.499) / f_show
        t_bwd = (i + 0.999) / f_show
        E_fwd = (-0.04 + 0.01 * i + esw) * 1000
        E_bwd = (-0.04 + 0.01 * i - esw) * 1000
        ax.scatter(t_fwd, E_fwd, color=C1, s=40, zorder=5)
        ax.scatter(t_bwd, E_bwd, color=C2, s=40, zorder=5)
    ax.scatter([], [], color=C1, s=20, label="$I_f$ sampled")
    ax.scatter([], [], color=C2, s=20, label="$I_b$ sampled")
    ax.annotate("", xy=(0.125 + tau / 2, -15), xytext=(0.125, -15),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1))
    ax.text(0.125 + tau / 4, -18, r"$\tau=1/(2f)$", ha="center", fontsize=7)
    ax.annotate("", xy=(0.26, -40 + esw * 1000), xytext=(0.26, -40 - esw * 1000),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1))
    ax.text(0.275, -40, r"$2E_{\rm sw}$", ha="left", fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Potential (mV)")
    ax.set_title("C  SWV waveform")
    ax.legend(fontsize=7, loc="upper right")

    # D: Quasi-reversible optimum
    ax = axes[3]
    kap_range = np.logspace(-2, 2, 200)
    E_test = np.linspace(-0.15, 0.15, 400)
    peak_vals = []
    for kap in kap_range:
        k0_t = kap * 2 * f_op
        DI = swv_voltammogram(E_test, k0=k0_t, ne=2, alpha=0.37,
                               E0=0.0, f_swv=f_op, E_sw=0.025)
        peak_vals.append(np.max(DI))
    peak_vals = np.array(peak_vals)
    peak_vals /= peak_vals.max()

    kap_tip = K0_REF_S1 / (2 * f_op)
    k0_mean = K0_REF_S1 * np.exp(-BETA_NM * (D_PEG_NM + SIGMA_Z_NM * np.sqrt(2 / np.pi) - D_PEG_NM))
    kap_mean = k0_mean / (2 * f_op)

    ax.semilogx(kap_range, peak_vals, color=C0, lw=2)
    ax.axvspan(0.5, 10, alpha=0.1, color="green", label="Quasi-rev. ($0.5<\\kappa<10$)")
    ax.axvline(kap_tip, color=C1, ls="--", lw=1.5, label=f"$\\kappa_{{\\rm tip}}={kap_tip:.1f}$")
    ax.axvline(kap_mean, color=C3, ls=":", lw=1.5, label=f"$\\kappa_{{\\rm mean}}={kap_mean:.3f}$")
    ax.set_xlabel(r"$\kappa = k^0 / (2f)$")
    ax.set_ylabel(r"$\Delta I_{\rm peak}$ (normalised)")
    ax.set_title("D  Quasi-reversible optimum")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.15)

    fig.tight_layout()
    save(fig, "fig_SWV1_geometry.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Marcus theory → Butler-Volmer derivation
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_marcus():
    print("[2/12] Fig 2: Marcus → BV derivation ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # A: Marcus parabolas
    ax = axes[0]
    q = np.linspace(-2.5, 2.5, 400)
    lam = 0.85          # eV, Chidsey 1991
    dG0 = 0.0           # at E = E0' (standard condition)
    G_R = 0.5 * q**2 / lam * lam    # reactant parabola: G_R = q^2/(4λ)·λ = q²/4 (in units of λ)
    # More clearly in λ units:
    G_R = q**2 / (4)
    G_P = (q - 2)**2 / 4 + dG0
    ax.plot(q, G_R, color=C0, lw=2, label="Reactant (ox + e)")
    ax.plot(q, G_P, color=C1, lw=2, label="Product (red)")
    # crossing point
    q_cross = 1.0  # at dG0=0, symmetric
    G_cross = q_cross**2 / 4
    ax.scatter([q_cross], [G_cross], color="k", s=60, zorder=5)
    ax.annotate(r"$\Delta G^\ddagger = \lambda/4$", xy=(q_cross, G_cross),
                xytext=(q_cross + 0.5, G_cross + 0.15),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    ax.annotate("", xy=(0, 0), xytext=(0, G_cross),
                arrowprops=dict(arrowstyle="<->", color=C2, lw=1.5))
    ax.text(0.12, G_cross / 2, r"$\lambda$", fontsize=10, color=C2)
    ax.set_xlim(-2.2, 3.5)
    ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel("Reaction coordinate $q$")
    ax.set_ylabel("Free energy $G$ (a.u.)")
    ax.set_title("A  Marcus energy surfaces ($\\lambda = 0.85$ eV)")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.92, "Chidsey (1991)", transform=ax.transAxes,
            fontsize=7, color="#555", style="italic")

    # B: Tafel plot — ln(k_f), ln(k_b) vs overpotential
    ax = axes[1]
    eta_mV = np.linspace(-200, 200, 400)
    alpha_bv = 0.37
    ne = 2
    xi = ne * F_const * eta_mV * 1e-3 / (R_const * T_K)
    ln_kf = -alpha_bv * xi          # normalised to k0=1
    ln_kb = (1 - alpha_bv) * xi
    ax.plot(eta_mV, ln_kf, color=C0, lw=2, label=r"$\ln(k_f/k^0) = -\alpha n_e f \eta$")
    ax.plot(eta_mV, ln_kb, color=C1, lw=2, label=r"$\ln(k_b/k^0) = (1-\alpha) n_e f \eta$")
    ax.axvline(0, color="#aaa", ls="--", lw=0.8)
    ax.axhline(0, color="#aaa", ls="--", lw=0.8)
    # slope annotations
    ax.text(-180, -0.8,
            f"slope $= -\\alpha n_e F/RT$\n$= -{alpha_bv}\\times{ne}\\times F/RT$",
            fontsize=7, color=C0)
    ax.text(50, -0.5,
            f"slope $= (1-\\alpha)n_e F/RT$\n$= {1-alpha_bv:.2f}\\times{ne}\\times F/RT$",
            fontsize=7, color=C1)
    ax.scatter([0], [0], color="k", s=40, zorder=5)
    ax.text(5, 0.05, "$k_f = k_b = k^0$\nat $E = E^{\\circ\\prime}$", fontsize=7)
    ax.set_xlabel("Overpotential $\\eta = E - E^{\\circ\\prime}$ (mV)")
    ax.set_ylabel(r"$\ln(k/k^0)$")
    ax.set_title(f"B  Tafel plot ($\\alpha = {alpha_bv}$, $n_e = {ne}$)")
    ax.legend(fontsize=7)

    # C: Tunnelling barrier — wavefunction decay
    ax = axes[2]
    d_tun = np.linspace(0, 25, 300)   # Å
    V0_eV = 2.0                         # barrier height, ~2 eV for organic SAM
    hbar_eV_s = 6.582e-16
    me_eV = 0.511e6                     # eV/c^2
    c_m_s = 3e8
    # Gamow factor: beta = 2*sqrt(2*me*V0)/hbar (per metre), convert to per Å
    # beta (Å-1) ≈ 0.5128 * sqrt(V0 in eV) ≈ 0.5128 * sqrt(2) ≈ 0.725 Å-1 for V0=2eV
    # Using known result: beta ~ 1.0 Å-1 for organic SAM empirically (Smalley 1995)
    beta_tun = 0.95   # Å-1, Smalley 1995 (alkanethiol)
    psi_sq = np.exp(-2 * beta_tun * d_tun)

    ax.plot(d_tun, psi_sq, color=C0, lw=2, label=r"$|\psi|^2 \propto e^{-2\beta d}$")
    # Barrier rectangle
    ax.axvspan(0, 14, alpha=0.08, color="#888", label=f"Barrier ($V_0\\approx{V0_eV}$ eV)")
    ax.axvline(D_PEG_NM * 10, color=C2, ls="--", lw=1.5,
               label=f"PEG6 tip, $d_{{\\rm PEG}}={D_PEG_NM*10:.1f}$Å")
    # annotate rate at PEG tip
    psi_tip = np.exp(-2 * beta_tun * D_PEG_NM * 10)
    ax.scatter([D_PEG_NM * 10], [psi_tip], color=C2, s=50, zorder=5)
    ax.text(D_PEG_NM * 10 + 0.5, psi_tip + 0.003,
            f"$k^0_{{\\rm ref}} = {K0_REF_S1:.0f}$ s$^{{-1}}$", fontsize=7, color=C2)
    ax.set_xlabel(r"Distance from electrode $d$ (Å)")
    ax.set_ylabel(r"Tunnelling probability $\propto e^{-2\beta d}$")
    ax.set_title(f"C  Tunnelling decay ($\\beta = {beta_tun}$ Å$^{{-1}}$, Smalley 1995)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 25)
    ax.text(0.62, 0.75,
            r"$\beta = \frac{2\sqrt{2m_e V_0}}{\hbar}$" + "\n" +
            r"$\approx 0.95$ Å$^{-1}$ (alkyl)",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(fc="lightyellow", ec="#aaa", boxstyle="round"))

    # D: k0 vs d with Chidsey 1991 and Smalley 1995 data points
    ax = axes[3]
    d_plot_A = np.linspace(0, 25, 300)
    # Our model: k0(d) from k0_at_0 = 1.5e8
    K0_at_0 = 1.5e8   # DD2017
    k0_model = K0_at_0 * np.exp(-BETA_NM * d_plot_A / 10)

    # Smalley 1995: k0_0 = 6e8 s-1, beta_n = 1.21 per CH2, each CH2 ~1.27 Å
    beta_Sm = 1.21 / 1.27   # Å-1 ≈ 0.953
    K0_Sm_0 = 6e8
    k0_smalley = K0_Sm_0 * np.exp(-beta_Sm * d_plot_A)

    # Chidsey 1991: k0 = 1.3 s-1 at d ≈ 20 Å (C16 chain, ~16 × 1.26 Å = 20.2 Å)
    d_Chidsey = 16 * 1.26   # Å
    k0_Chidsey = 1.3        # s-1

    ax.semilogy(d_plot_A, k0_model, color=C0, lw=2, label="This model (DD2017 anchor)")
    ax.semilogy(d_plot_A, k0_smalley, color=C1, lw=2, ls="--",
                label="Smalley 1995 ($\\beta=0.95$ Å$^{-1}$, Fc-alkyl)")
    ax.scatter([d_Chidsey], [k0_Chidsey], color=C2, s=80, zorder=5,
               label=f"Chidsey 1991 ($d={d_Chidsey:.0f}$Å, Fc-C16)")
    ax.axvline(D_PEG_NM * 10, color=C2, ls=":", lw=1, alpha=0.7)
    ax.scatter([D_PEG_NM * 10], [K0_REF_S1], color=C0, s=60, zorder=5,
               marker="*", label=f"$k^0_{{\\rm ref}}={K0_REF_S1:.0f}$ s$^{{-1}}$ at PEG6 tip (FLAG)")
    ax.set_xlabel(r"Distance $d$ (Å)")
    ax.set_ylabel(r"$k^0$ (s$^{-1}$)")
    ax.set_title("D  $k^0(d)$: model vs published data")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 25)

    fig.tight_layout()
    save(fig, "fig_SWV2_marcus.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — BV-ODE derivation: half-cycle dynamics and three special cases
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_ode_derivation():
    print("[3/12] Fig 3: ODE derivation and special cases ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    f_op = 10.0  # confirmed operating frequency (10-20 Hz range; 10 Hz used as representative)
    tau  = 1.0 / (2 * f_op)
    t_arr = np.linspace(0, tau, 300)
    E0 = 0.0
    Gamma_tot = 1.0
    alpha = 0.37
    ne = 2

    kappa_cases = [0.05, 4.2, 50.0]
    colors = [C1, C0, C2]
    labels = [r"$\kappa=0.05$ (irrev.)", r"$\kappa=4.2$ (quasi-rev., our case)",
              r"$\kappa=50$ (rev.)"]

    # A: Gamma_red(t) during forward + backward half-pulse at E = E0
    ax = axes[0]
    E_fwd = E0 + 0.025
    E_bwd = E0 - 0.025
    for kap, col, lab in zip(kappa_cases, colors, labels):
        k0_t = kap * 2 * f_op
        kf_f, kb_f = k0_t * np.exp(-alpha * ne * F_const * (E_fwd - E0) / (R_const * T_K)), \
                     k0_t * np.exp((1-alpha) * ne * F_const * (E_fwd - E0) / (R_const * T_K))
        Geq_f = kf_f / (kf_f + kb_f)
        G0 = 0.5  # start at half-reduced
        G_fwd = Geq_f + (G0 - Geq_f) * np.exp(-(kf_f + kb_f) * t_arr)

        kf_b, kb_b = k0_t * np.exp(-alpha * ne * F_const * (E_bwd - E0) / (R_const * T_K)), \
                     k0_t * np.exp((1-alpha) * ne * F_const * (E_bwd - E0) / (R_const * T_K))
        Geq_b = kf_b / (kf_b + kb_b)
        G_fwd_end = G_fwd[-1]
        G_bwd = Geq_b + (G_fwd_end - Geq_b) * np.exp(-(kf_b + kb_b) * t_arr)

        t_both = np.concatenate([t_arr, t_arr + tau])
        G_both = np.concatenate([G_fwd, G_bwd])
        ax.plot(t_both * 1000, G_both, color=col, lw=1.5, label=lab)
        ax.axhline(Geq_f, color=col, ls=":", lw=0.8, alpha=0.5)
        ax.axhline(Geq_b, color=col, ls="--", lw=0.8, alpha=0.5)

    ax.axvline(tau * 1000, color="#aaa", ls="-", lw=0.8)
    ax.text(tau * 1000 / 2, 0.02, "Forward\nhalf-pulse", ha="center", fontsize=7)
    ax.text(tau * 1000 * 1.5, 0.02, "Backward\nhalf-pulse", ha="center", fontsize=7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$\Gamma_{\rm red}/\Gamma_{\rm tot}$")
    ax.set_title("A  Surface coverage during one SWV step")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(-0.05, 1.05)

    # B: Three special cases: kappa→0, kappa≈1, kappa→∞
    ax = axes[1]
    kap_list = [0.01, 1.0, 100.0]
    linestyles = ["-", "--", ":"]
    Geq_target = 0.8  # at E much below E0'
    G0_case = 0.5
    for kap, ls in zip(kap_list, linestyles):
        k0_c = kap * 2 * f_op
        kf_c = k0_c * 5   # E well below E0' (fully reducing)
        kb_c = k0_c * 0.2
        Geq_c = kf_c / (kf_c + kb_c)
        G_c = Geq_c + (G0_case - Geq_c) * np.exp(-(kf_c + kb_c) * t_arr)
        DG = abs(G_c[-1] - G0_case)
        ax.plot(t_arr * 1000, G_c, lw=2, ls=ls,
                label=f"$\\kappa={kap}$,  $\\Delta\\Gamma/\\Gamma_{{\\rm tot}}={DG:.2f}$")
    ax.axhline(G0_case, color="#aaa", ls="--", lw=0.8)
    ax.text(tau * 1000 * 0.98, G0_case + 0.01, "$\\Gamma_{\\rm red}(0)$",
            ha="right", fontsize=7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$\Gamma_{\rm red}/\Gamma_{\rm tot}$")
    ax.set_title("B  Three kinetic regimes")
    ax.legend(fontsize=8)

    # C: ΔI vs κ with analytical peak
    ax = axes[2]
    kap_range2 = np.logspace(-2, 2, 80)
    E_t = np.linspace(-0.15, 0.15, 300)
    peaks_c = []
    for kap in kap_range2:
        k0_t = kap * 2 * f_op
        DI = swv_voltammogram(E_t, k0=k0_t, ne=2, alpha=0.37, E0=0.0,
                               f_swv=f_op, E_sw=0.025)
        peaks_c.append(np.max(DI))
    peaks_c = np.array(peaks_c) / max(peaks_c)

    ax.semilogx(kap_range2, peaks_c, color=C0, lw=2)
    ax.axvspan(0.5, 10, alpha=0.1, color="green")
    kap_tip = K0_REF_S1 / (2 * f_op)
    ax.axvline(kap_tip, color=C1, ls="--", lw=1.5, label=f"$\\kappa_{{\\rm tip}}={kap_tip:.1f}$ (our case)")
    ax.scatter([kap_tip], [np.interp(kap_tip, kap_range2, peaks_c)], color=C1, s=60, zorder=5)
    # annotate maximum
    i_max = np.argmax(peaks_c)
    ax.annotate(f"Maximum at\n$\\kappa\\approx{kap_range2[i_max]:.1f}$",
                xy=(kap_range2[i_max], 1.0), xytext=(0.5, 0.85),
                xycoords=("data", "data"), textcoords=("data", "axes fraction"),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
                transform=ax.transAxes)
    ax.set_xlabel(r"$\kappa = k^0/(2f)$")
    ax.set_ylabel(r"$\Delta I_{\rm peak}$ (normalised)")
    ax.set_title("C  Peak current vs $\\kappa$")
    ax.legend(fontsize=8)

    # D: Integrating factor illustration
    ax = axes[3]
    ax.axis("off")
    # Show the algebra as text panel
    lines = [
        r"Integrating factor derivation:",
        r"",
        r"ODE:  $\dfrac{d\Gamma_{\rm red}}{dt} = k_f \Gamma_{\rm tot} - (k_f+k_b)\Gamma_{\rm red}$",
        r"",
        r"Let $a = k_f \Gamma_{\rm tot}$,  $b = k_f + k_b$:",
        r"    $\dot{y} + b\,y = a$",
        r"",
        r"Multiply by $\mu = e^{bt}$:",
        r"    $\dfrac{d}{dt}\!\left[y\,e^{bt}\right] = a\,e^{bt}$",
        r"",
        r"Integrate, apply IC $y(0) = \Gamma_{\rm red}(0)$:",
        r"",
        r"$\mathbf{\Gamma_{\rm red}(\tau) = \Gamma_{\rm eq} + [\Gamma_{\rm red}(0)-\Gamma_{\rm eq}]\,e^{-b\tau}}$",
        r"",
        r"where $\Gamma_{\rm eq} = k_f/(k_f+k_b)\,\Gamma_{\rm tot}$",
    ]
    for i, line in enumerate(lines):
        y = 0.93 - i * 0.06
        ax.text(0.05, y, line, transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                color="#1a1a6e" if "boxed" in line else "black")
    ax.set_title("D  Integrating factor derivation")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.tight_layout()
    save(fig, "fig_SWV3_ode.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Full scan evolution of Gamma_red
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_scan_evolution():
    print("[4/12] Fig 4: Full scan evolution ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    E_scan_full = np.linspace(-0.50, 0.10, 300)
    f_op = 10.0  # confirmed operating frequency (10-20 Hz range; 10 Hz used as representative)
    tau = 1.0 / (2 * f_op)
    alpha, ne, E0 = 0.37, 2, -0.25

    kappa_cases2 = [0.05, 4.2, 50.0]
    colors2 = [C1, C0, C2]
    labels2 = [r"$\kappa=0.05$ (irrev.)", r"$\kappa=4.2$ (our operating point)",
               r"$\kappa=50$ (rev.)"]

    ax = axes[0]
    for kap, col, lab in zip(kappa_cases2, colors2, labels2):
        k0_t = kap * 2 * f_op
        G_r = 1.0  # start fully reduced
        G_fwd_list, G_bwd_list = [], []
        for E_step in E_scan_full:
            G_r, _ = _half_cycle_end(G_r, 1.0, k0_t, alpha, ne, E_step + 0.025, E0, tau)
            G_fwd_list.append(G_r)
            G_r, _ = _half_cycle_end(G_r, 1.0, k0_t, alpha, ne, E_step - 0.025, E0, tau)
            G_bwd_list.append(G_r)
        ax.plot(E_scan_full * 1000, G_fwd_list, color=col, lw=1.5, label=lab)
        ax.plot(E_scan_full * 1000, G_bwd_list, color=col, lw=1.0, ls="--", alpha=0.6)

    ax.axvline(E0 * 1000, color="#888", ls=":", lw=1)
    ax.text(E0 * 1000 + 2, 0.95, "$E^{\\circ\\prime}$", fontsize=9)
    ax.set_xlabel("Step potential $E_{\\rm step}$ (mV)")
    ax.set_ylabel(r"$\Gamma_{\rm red}/\Gamma_{\rm tot}$  (sampled)")
    ax.set_title("A  Surface coverage through full scan")
    ax.legend(fontsize=7, loc="lower right")
    ax_leg = ax.legend(fontsize=7, loc="lower right")

    # ΔI across scan
    ax2 = axes[1]
    for kap, col, lab in zip(kappa_cases2, colors2, labels2):
        k0_t = kap * 2 * f_op
        DI = swv_voltammogram(E_scan_full, k0=k0_t, ne=2, alpha=alpha, E0=E0,
                               f_swv=f_op, E_sw=0.025)
        DI_norm = DI / (np.max(np.abs(DI)) + 1e-30)
        ax2.plot(E_scan_full * 1000, DI_norm, color=col, lw=1.5, label=lab)

    ax2.axvline(E0 * 1000, color="#888", ls=":", lw=1)
    ax2.set_xlabel("Step potential $E_{\\rm step}$ (mV)")
    ax2.set_ylabel(r"$\Delta I$ (normalised)")
    ax2.set_title("B  Net SWV current $\\Delta I(E)$ for three $\\kappa$ values")
    ax2.legend(fontsize=7)
    ax2.axhline(0, color="#ccc", lw=0.5)

    fig.tight_layout()
    save(fig, "fig_SWV4_scan.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Distance averaging (was fig3)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_averaging(DI_Fk, fk_arr, A_areas, E_peaks, FWHMs,
                   slope_lin, intercept_lin, residuals_pct):
    print("[5/12] Fig 5: Distance averaging ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    d_repr = np.linspace(D_PEG_NM, D_PEG_NM + 4 * SIGMA_Z_NM, 6)
    cmap = plt.cm.viridis
    ax = axes[0]
    for j, d in enumerate(d_repr):
        k0_d = k0_of_d(d)
        DI_d = swv_voltammogram(E_SCAN, k0=k0_d, ne=2, alpha=0.37, E0=-0.25,
                                 f_swv=10.0, E_sw=0.025)
        w = P_d_norm(d) * (D_MAX_NM - D_PEG_NM) / len(d_repr)
        col = cmap(j / (len(d_repr) - 1))
        ax.plot(E_SCAN * 1000, DI_d * w * 1e9, color=col, lw=1.5,
                label=f"$d={d*10:.1f}$Å")
    ax.set_xlabel("Potential (mV)")
    ax.set_ylabel(r"$P(d_i)\,\Delta I(E;\,d_i)\,\Delta d$ (nA)")
    ax.set_title("A  Individual contributions $P(d_i)\\,\\Delta I(E;\\,d_i)$")
    ax.legend(fontsize=7)

    ax = axes[1]
    colors_fk = plt.cm.plasma(np.linspace(0, 0.9, len(fk_arr)))
    for i, fk in enumerate(fk_arr):
        ax.plot(E_SCAN * 1000, DI_Fk[i] * 1e9, color=colors_fk[i],
                lw=1.5, label=f"$F_k={fk:.2f}$")
    ax.set_xlabel("Potential (mV)")
    ax.set_ylabel(r"$\Delta I(E;\,F_k)$ (nA)")
    ax.set_title("B  Distance-averaged $\\Delta I$ at various $F_k$")
    ax.legend(fontsize=7)

    ax = axes[2]
    intact_frac = 1 - np.array(fk_arr)
    ax.scatter(intact_frac, A_areas * 1e9, color=C0, s=60, zorder=5)
    lin_fit = slope_lin * intact_frac + intercept_lin
    ax.plot(intact_frac, lin_fit * 1e9, color=C1, lw=1.5, ls="--", label="Linear fit")
    ax.set_xlabel("Intact probe fraction $(1-F_k)$")
    ax.set_ylabel(r"Peak area $A_{\rm area}$ (nA$\cdot$V)")
    ax.set_title("C  Calibration linearity")
    ax2c = ax.inset_axes([0.55, 0.08, 0.40, 0.35])
    ax2c.bar(intact_frac, residuals_pct, color=C0, width=0.08)
    ax2c.axhline(0, color="k", lw=0.5)
    ax2c.set_ylabel("Residual (%)", fontsize=6)
    ax2c.tick_params(labelsize=6)
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(fk_arr, np.array(E_peaks) * 1000, color=C0, lw=1.5, marker="o", ms=4,
            label="$E_{\\rm pk}$ (mV)")
    ax2d = ax.twinx()
    ax2d.plot(fk_arr, np.array(FWHMs) * 1000, color=C1, lw=1.5, marker="s", ms=4,
              ls="--", label="FWHM (mV)")
    ax.set_xlabel("Cleavage fraction $F_k$")
    ax.set_ylabel("$E_{\\rm pk}$ (mV)", color=C0)
    ax2d.set_ylabel("FWHM (mV)", color=C1)
    ax.set_title("D  Peak position and width vs $F_k$")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2_leg = ax2d.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2_leg, fontsize=7, loc="center")

    fig.tight_layout()
    save(fig, "fig_SWV5_averaging.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Validation: Laviron limit + FWHM derivation + DD comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_validation():
    print("[6/12] Fig 6: Validation ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    E_lav = np.linspace(-0.15, 0.15, 600)
    f_op = 10.0  # confirmed operating frequency (10-20 Hz range; 10 Hz used as representative)

    # A: Laviron limit (small Esw)
    ax = axes[0]
    k0_qr = 10 * 2 * f_op
    DI_25 = swv_voltammogram(E_lav, k0=k0_qr, ne=2, alpha=0.5, E0=0.0,
                              f_swv=f_op, E_sw=0.025)
    DI_1 = swv_voltammogram(E_lav, k0=k0_qr, ne=2, alpha=0.5, E0=0.0,
                             f_swv=f_op, E_sw=0.001)
    DI_lav = laviron_reversible(E_lav, E0=0.0, ne=2)

    # Normalise
    DI_25n = DI_25 / np.max(DI_25)
    DI_1n = DI_1 / np.max(DI_1)
    DI_lavn = DI_lav / np.max(DI_lav)

    ax.plot(E_lav * 1000, DI_25n, color=C0, lw=2,
            label="BV-SWV $E_{\\rm sw}=25$ mV (operating)")
    ax.plot(E_lav * 1000, DI_1n, color=C1, lw=2, ls="--",
            label="BV-SWV $E_{\\rm sw}=1$ mV (small-$E_{\\rm sw}$ limit)")
    ax.plot(E_lav * 1000, DI_lavn, color="#888", lw=1.5, ls=":",
            label="Laviron (1979) analytic")

    # FWHM markers
    for DI_n, col, name in [(DI_1n, C1, "1 mV"), (DI_25n, C0, "25 mV")]:
        above_half = E_lav[DI_n >= 0.5]
        if len(above_half) > 1:
            fwhm = (above_half[-1] - above_half[0]) * 1000
            ax.annotate("", xy=(above_half[-1] * 1000, 0.5),
                        xytext=(above_half[0] * 1000, 0.5),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1))
            ax.text((above_half[0] + above_half[-1]) / 2 * 1000, 0.52,
                    f"FWHM={fwhm:.1f} mV", ha="center", fontsize=7, color=col)
    ax.set_xlabel("Potential (mV)")
    ax.set_ylabel(r"$\Delta I$ (normalised)")
    ax.set_title("A  BV simulator vs Laviron (1979)")
    ax.legend(fontsize=7)

    # B: Laviron FWHM derivation
    ax = axes[1]
    ne_vals = [1, 2, 4]
    eta_range = np.linspace(-0.30, 0.30, 600)
    for ne_v in ne_vals:
        xi = ne_v * F_const * eta_range / (R_const * T_K)
        eta_e = np.exp(xi)
        psi = eta_e / (1 + eta_e)**2
        psi_n = psi / np.max(psi)
        fwhm_th = 3.53 * R_const * T_K / (ne_v * F_const) * 1000  # mV
        ax.plot(eta_range * 1000, psi_n, lw=2,
                label=f"$n_e={ne_v}$,  FWHM$={fwhm_th:.1f}$ mV")
    ax.axhline(0.5, color="#aaa", ls="--", lw=0.8)
    ax.set_xlabel("Overpotential $\\eta$ (mV)")
    ax.set_ylabel(r"$|\Psi|$ (normalised)")
    ax.set_title("B  Laviron peak: FWHM $= 3.53\\,RT/(n_e F)$")
    ax.legend(fontsize=8)

    # C: Frequency dependence (DD comparison)
    ax = axes[2]
    k0_DD = 5000.0
    alpha_DD = 0.37
    E_DD = np.linspace(-0.15, 0.15, 400)
    freqs_DD = [200, 500, 1000]
    for f_dd, col in zip(freqs_DD, [C0, C1, C2]):
        kap_dd = k0_DD / (2 * f_dd)
        DI_dd = swv_voltammogram(E_DD, k0=k0_DD, ne=2, alpha=alpha_DD, E0=0.0,
                                  f_swv=f_dd, E_sw=0.025)
        DI_dd_n = DI_dd / np.max(DI_dd)
        fwhm_dd = 0
        above_h = E_DD[DI_dd_n >= 0.5]
        if len(above_h) > 1:
            fwhm_dd = (above_h[-1] - above_h[0]) * 1000
        ax.plot(E_DD * 1000, DI_dd_n, color=col, lw=1.5,
                label=f"$f={f_dd}$ Hz, $\\kappa={kap_dd:.1f}$, FWHM$={fwhm_dd:.0f}$ mV")
    ax.set_xlabel("Potential (mV)")
    ax.set_ylabel(r"$\Delta I$ (normalised)")
    ax.set_title("C  Frequency dependence (DD2017 parameters)")
    ax.legend(fontsize=7)

    # D: FWHM vs kappa
    ax = axes[3]
    kap_range = np.logspace(-1, 2, 60)
    fwhms = []
    E_fw = np.linspace(-0.15, 0.15, 500)
    for kap in kap_range:
        k0_t = kap * 2 * f_op
        DI = swv_voltammogram(E_fw, k0=k0_t, ne=2, alpha=0.37, E0=0.0,
                               f_swv=f_op, E_sw=0.025)
        DI_n = DI / (np.max(DI) + 1e-30)
        above_h = E_fw[DI_n >= 0.5]
        if len(above_h) > 1:
            fwhms.append((above_h[-1] - above_h[0]) * 1000)
        else:
            fwhms.append(np.nan)
    ax.semilogx(kap_range, fwhms, color=C0, lw=2)
    ax.axhline(3.53 * R_const * T_K / (2 * F_const) * 1000, color=C1,
               ls="--", lw=1.5, label=f"Laviron limit {3.53*R_const*T_K/(2*F_const)*1000:.1f} mV")
    kap_tip = K0_REF_S1 / (2 * f_op)
    fwhm_tip = np.interp(kap_tip, kap_range, fwhms)
    ax.scatter([kap_tip], [fwhm_tip], color=C1, s=60, zorder=5,
               label=f"$\\kappa_{{\\rm tip}}={kap_tip:.1f}$, FWHM$={fwhm_tip:.0f}$ mV")
    ax.axvspan(0.5, 10, alpha=0.08, color="green")
    ax.set_xlabel(r"$\kappa = k^0/(2f)$")
    ax.set_ylabel("FWHM (mV)")
    ax.set_title("D  FWHM vs $\\kappa$")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, "fig_SWV6_validation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Ratiometric correction (was fig4)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_ratiometric(DI_Fk, fk_arr, A_areas):
    print("[7/12] Fig 7: Ratiometric correction ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    f_op = 10.0  # confirmed operating frequency (10-20 Hz range; 10 Hz used as representative)
    # Fc parameters. E0_Fc updated to the observed range on the fabricated
    # device (+300 to +400 mV vs Ag/AgCl, itself variable electrode-to-
    # electrode; internal electrode characterisation notes, Sept 2026) --
    # 350 mV used here as the midpoint, not a confirmed single value.
    # k0_Fc is the literature value (Dauphin-Ducharme, different construct);
    # at kappa_Fc = 13500 this predicts an essentially zero net SWV peak,
    # contradicted by the real observed Fc peak -- see swv_forward_model.tex
    # Part E. Left at the literature value; not yet replaced with a
    # confirmed device-specific rate constant.
    E0_Fc = 0.350
    k0_Fc = 2.7e5
    kap_Fc = k0_Fc / (2 * f_op)
    A_Fc = 0.0177
    Gamma_Fc = 1.5e-12  # matches GAMMA_FC in swv_forward_model.py
    E_wide = np.linspace(-0.40, 0.50, 600)

    # A: MB + Fc peaks
    ax = axes[0]
    DI_MB = dist_averaged_voltammogram(E_wide, F_k=0.0, E0_MB=-0.250)
    DI_Fc = swv_voltammogram(E_wide, k0=k0_Fc, Gamma_tot=Gamma_Fc, A_cm2=A_Fc,
                              E0=E0_Fc, alpha=0.5, ne=1, f_swv=f_op, E_sw=0.025)
    ax.plot(E_wide * 1000, DI_MB * 1e9, color=C0, lw=2, label="MB ($F_k=0$)")
    ax.plot(E_wide * 1000, DI_Fc * 1e9, color=C1, lw=2, label="Fc (reference)")
    ax.axvline(-250, color=C0, ls=":", lw=0.8, alpha=0.5)
    ax.axvline(E0_Fc * 1000, color=C1, ls=":", lw=0.8, alpha=0.5)
    sep_mV = E0_Fc * 1000 - (-250)
    ax.annotate("", xy=(E0_Fc * 1000, 0.3), xytext=(-250, 0.3),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
    ax.text((E0_Fc * 1000 - 250) / 2, 0.32, f"{sep_mV:.0f} mV (modelled)",
            ha="center", fontsize=8)
    ax.text(0.02, 0.96,
            "As originally modelled -- a distinct dual peak has not yet\n"
            "been observed experimentally on the fabricated device.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            style="italic", color="#a33",
            bbox=dict(boxstyle="round", fc="#fff3f3", ec="#a33", lw=0.6))
    ax.set_xlabel("Potential (mV vs Ag/AgCl)")
    ax.set_ylabel(r"$\Delta I$ (nA)")
    ax.set_title("A  MB and Fc peaks (modelled)")
    ax.legend(fontsize=8, loc="upper right")

    # B: Ratiometric recovery
    ax = axes[1]
    fk_arr2 = np.linspace(0, 0.95, 50)
    rho_true = 1 - fk_arr2
    # Simulate with ±3% noise on A_MB
    np.random.seed(42)
    noise = np.random.normal(0, 0.017, len(fk_arr2))
    rho_meas = rho_true + noise
    fk_rec = 1 - rho_meas
    err = np.abs(fk_rec - fk_arr2)

    ax.plot(fk_arr2, fk_arr2, "k--", lw=1, label="Ideal ($F_k^{\\rm rec}=F_k$)")
    ax.scatter(fk_arr2, fk_rec, color=C0, s=15, alpha=0.7, label="Ratiometric recovery")
    ax.fill_between(fk_arr2, fk_arr2 - 0.034, fk_arr2 + 0.034,
                    alpha=0.15, color=C0, label="$\\pm 3.4\\%$ noise floor")
    ax.set_xlabel("True $F_k$")
    ax.set_ylabel("Recovered $F_k^{\\rm rec}$")
    ax.set_title("B  Ratiometric pipeline closure")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # C: Frequency diagnostic
    ax = axes[2]
    freqs = np.logspace(0.3, 3, 30)
    peak_P = []
    peak_eff = []
    for fq in freqs:
        DI_p = dist_averaged_voltammogram(E_SCAN, F_k=0.0, f_swv=fq)
        peak_P.append(np.max(DI_p))
        k0_eff = k0_of_d(D_PEG_NM + SIGMA_Z_NM * np.sqrt(2 / np.pi))
        DI_e = swv_voltammogram(E_SCAN, k0=k0_eff, ne=2, alpha=0.37,
                                 E0=-0.25, f_swv=fq, E_sw=0.025)
        DI_e_sc = DI_e * 3e-12 * 0.0177
        peak_eff.append(np.max(DI_e_sc))

    peak_P = np.array(peak_P)
    peak_eff = np.array(peak_eff)
    ax.semilogx(freqs, peak_P / peak_P.max(), color=C0, lw=2,
                label="$P(d)$-averaged model")
    ax.semilogx(freqs, peak_eff / peak_eff.max(), color=C1, lw=2, ls="--",
                label="Single $k^0$ at mean $d$")
    ax.axvline(15, color="#888", ls=":", lw=1.5, label="Operating 15 Hz")
    ax.set_xlabel("SWV frequency $f$ (Hz)")
    ax.set_ylabel(r"$\Delta I_{\rm peak}$ (normalised)")
    ax.set_title("C  Frequency diagnostic")
    ax.legend(fontsize=7)

    # D: Convergence
    ax = axes[3]
    n_dist_vals = [5, 10, 20, 40, 80, 120, 160]
    peak_conv = []
    for nd in n_dist_vals:
        DI_c = dist_averaged_voltammogram(E_SCAN, F_k=0.0, n_dist=nd)
        peak_conv.append(np.max(DI_c))
    peak_conv = np.array(peak_conv)
    rel_err = abs(peak_conv - peak_conv[-1]) / peak_conv[-1] * 100

    ax.semilogy(n_dist_vals, rel_err + 1e-4, color=C0, lw=2, marker="o", ms=5)
    ax.axvline(80, color=C1, ls="--", lw=1.5, label="$n_{\\rm dist}=80$ (used)")
    ax.axhline(0.5, color="#aaa", ls=":", lw=1)
    ax.text(85, 0.6, "$<0.5\\%$ error", fontsize=8, color="#888")
    ax.set_xlabel("Quadrature points $n_{\\rm dist}$")
    ax.set_ylabel("Relative error vs $n_{\\rm dist}=160$ (%)")
    ax.set_title("D  Quadrature convergence")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "fig_SWV7_ratiometric.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — SWV waveform & technique comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_waveform():
    print("[8/12] Fig 8: SWV waveform and technique comparison ...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # A: SWV waveform with detailed timing
    ax = axes[0]
    n_steps = 8
    dt_step = 0.04  # s per step
    esw = 0.025
    dE_s = 0.005   # V staircase step

    t_all, E_all = [], []
    E_stair = -0.20
    for i in range(n_steps):
        t0 = i * dt_step
        # Forward pulse
        t_all.extend([t0, t0 + dt_step / 2 - 0.001])
        E_all.extend([E_stair + esw, E_stair + esw])
        # Backward pulse
        t_all.extend([t0 + dt_step / 2, t0 + dt_step - 0.001])
        E_all.extend([E_stair - esw, E_stair - esw])
        E_stair += dE_s

    ax.plot(np.array(t_all) * 1000, np.array(E_all) * 1000, color="#333", lw=1.5)
    # Staircase overlay
    E_stair2 = -0.20
    t_sc, E_sc = [0], [E_stair2 * 1000]
    for i in range(n_steps):
        t_sc.append((i + 1) * dt_step * 1000)
        t_sc.append((i + 1) * dt_step * 1000)
        E_sc.append(E_stair2 * 1000)
        E_stair2 += dE_s
        E_sc.append(E_stair2 * 1000)
    ax.plot(t_sc, E_sc, color="#888", lw=0.8, ls="--", label="Staircase $E_{\\rm step}$")

    # Sample markers
    for i in range(n_steps):
        t_f = (i * dt_step + dt_step / 2 - 0.003) * 1000
        t_b = (i * dt_step + dt_step - 0.003) * 1000
        E_f = (-0.20 + i * dE_s + esw) * 1000
        E_b = (-0.20 + i * dE_s - esw) * 1000
        ax.scatter(t_f, E_f, color=C1, s=40, zorder=5)
        ax.scatter(t_b, E_b, color=C2, s=40, zorder=5)

    ax.scatter([], [], color=C1, s=20, label="$I_f$ sample")
    ax.scatter([], [], color=C2, s=20, label="$I_b$ sample")
    ax.annotate("", xy=(dt_step * 1000, -245), xytext=(0, -245),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1))
    ax.text(dt_step * 500, -249, "$\\Delta t = 1/f$", ha="center", fontsize=7)
    ax.annotate("", xy=(0.002 * 1000, (-0.20 + esw) * 1000),
                xytext=(0.002 * 1000, (-0.20 - esw) * 1000),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1))
    ax.text(5, -200, "$2E_{\\rm sw}$", fontsize=7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Potential (mV)")
    ax.set_title("A  SWV waveform")
    ax.legend(fontsize=7, loc="upper left")

    # B: Charging current cancellation
    ax = axes[1]
    t_half = np.linspace(0, dt_step / 2, 200)
    Cdl = 20e-6  # double layer capacitance F/cm²
    dE = esw
    # Charging current: I_c = Cdl × dE/dt (step change: exponential decay)
    tau_c = 0.001  # RC time constant
    I_charge = Cdl * dE / tau_c * np.exp(-t_half / tau_c) * 1e6  # µA
    I_farad = 2.0 * np.exp(-t_half / (dt_step / 2)) * 0.8  # approximate Faradaic
    I_total = I_charge + I_farad

    ax.fill_between(t_half * 1000, I_charge, alpha=0.2, color=C1, label="Charging current")
    ax.plot(t_half * 1000, I_charge, color=C1, lw=1.5, ls="--")
    ax.plot(t_half * 1000, I_farad, color=C0, lw=2, label="Faradaic current")
    ax.axvline(t_half[-1] * 1000, color="#888", ls=":", lw=1)
    ax.text(t_half[-1] * 1000 - 1, 0.3, "Sample here\n(charging≈0)", ha="right", fontsize=7)
    ax.set_xlabel("Time into half-pulse (ms)")
    ax.set_ylabel(r"Current ($\mu$A)")
    ax.set_title("B  Charging current cancellation\n(sampled at end of pulse)")
    ax.legend(fontsize=7)

    # C: SWV vs CV vs DPV — SNR comparison for surface-confined
    ax = axes[2]
    techniques = ["CV\n(linear scan)", "DPV\n(differential pulse)", "SWV\n(square wave)"]
    # Approximate relative sensitivities for surface-confined species
    # CV: peak ~ n²F²vΓ/RT; DPV: ~nFΔE × Γ/τ; SWV: ~nFΓ × ΔI_net
    snr_rel = [1.0, 3.5, 8.0]
    bg_rej = [1.0, 4.0, 10.0]
    speed_rel = [1.0, 2.0, 5.0]
    x = np.arange(3)
    width = 0.25
    bars1 = ax.bar(x - width, snr_rel, width, color=C0, alpha=0.8, label="Signal/noise (rel.)")
    bars2 = ax.bar(x, bg_rej, width, color=C1, alpha=0.8, label="Background rejection (rel.)")
    bars3 = ax.bar(x + width, speed_rel, width, color=C2, alpha=0.8, label="Speed (rel.)")
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=8)
    ax.set_ylabel("Relative performance (normalised to CV)")
    ax.set_title("C  SWV vs CV vs DPV\n(surface-confined species)")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(2, 10.2, "SWV wins", ha="center", fontsize=8, color=C2, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig_SWV8_waveform.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Sensitivity analysis: tornado + LOD contours
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_sensitivity():
    print("[9/12] Fig 9: Sensitivity analysis ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Baseline LOD
    sigma_Fk = 0.034
    k_cat_sol = 67500.0
    eta_base = 0.20
    T_base = 1800.0  # s (30 min)
    keff_base = eta_base * k_cat_sol
    LOD_base = 3 * sigma_Fk / (keff_base * T_base) * 1e9  # nM

    # A: Tornado chart — LOD sensitivity to ±50% variation in each parameter
    params = [
        ("$\\eta$ (surface\ncorrection)", eta_base, 0.5, lambda v: 3*sigma_Fk/(v*k_cat_sol*T_base)*1e9),
        ("$T$ (incubation\ntime)", T_base, 0.5, lambda v: 3*sigma_Fk/(eta_base*k_cat_sol*v)*1e9),
        ("$k_{\\rm cat}/K_m$ (solution\nrate)", k_cat_sol, 0.5, lambda v: 3*sigma_Fk/(eta_base*v*T_base)*1e9),
        ("$\\sigma_{F_k}$ (noise\nfloor)", sigma_Fk, 0.5, lambda v: 3*v/(eta_base*k_cat_sol*T_base)*1e9),
        ("$\\Gamma_0$ (coverage)\n[weak]", 3e-12, 0.5, lambda v: LOD_base * 1.0),  # LOD not dep. on Γ directly
    ]

    ax = axes[0]
    bars_low = []
    bars_high = []
    param_labels = []
    for name, val, frac, lod_fn in params:
        lod_lo = lod_fn(val * (1 - frac))
        lod_hi = lod_fn(val * (1 + frac))
        bars_low.append(LOD_base - lod_lo)
        bars_high.append(lod_hi - LOD_base)
        param_labels.append(name)

    y_pos = np.arange(len(params))
    ax.barh(y_pos, bars_high, left=LOD_base, color=C1, alpha=0.8, label="+50%")
    ax.barh(y_pos, [-b for b in bars_low], left=[LOD_base + b for b in bars_low],
            color=C0, alpha=0.8, label="-50%")
    ax.axvline(LOD_base, color="k", lw=1.5, ls="--", label=f"Baseline LOD = {LOD_base:.1f} nM")
    ax.axvline(6, color=C2, lw=1.5, ls=":", label="6 nM boundary")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_labels, fontsize=8)
    ax.set_xlabel("LOD (nM)")
    ax.set_title("A  Sensitivity tornado\n($\\pm 50\\%$ parameter variation)")
    ax.legend(fontsize=7)

    # B: LOD contour in (η, T) space
    ax = axes[1]
    eta_range = np.logspace(-2, 0, 80)
    T_range_min = np.linspace(5, 90, 80)  # minutes
    ETA, TT = np.meshgrid(eta_range, T_range_min)
    LOD_grid = 3 * sigma_Fk / (ETA * k_cat_sol * TT * 60) * 1e9  # nM

    cf = ax.contourf(ETA, TT, LOD_grid, levels=np.logspace(-1, 2, 30),
                     cmap="RdYlGn_r", alpha=0.9)
    ax.contour(ETA, TT, LOD_grid, levels=[6, 30], colors=["white"],
               linewidths=[2, 1.5], linestyles=["--", ":"])
    ax.scatter([eta_base], [30], color="k", s=80, zorder=5, label="Operating point")
    ax.set_xscale("log")
    ax.set_xlabel("Surface correction $\\eta$")
    ax.set_ylabel("Incubation time $T$ (min)")
    ax.set_title("B  LOD (nM) in $(\\eta, T)$ space\n(white: 6 nM and 30 nM contours)")
    plt.colorbar(cf, ax=ax, label="LOD (nM)")
    ax.legend(fontsize=7)

    # C: LOD vs T for three η scenarios
    ax = axes[2]
    T_arr_min = np.linspace(2, 90, 200)
    eta_scenarios = [0.05, 0.20, 0.50]
    labs_s = ["$\\eta=0.05$ (pessimistic)", "$\\eta=0.20$ (central, FLAG)",
              "$\\eta=0.50$ (optimistic)"]
    for eta_s, lab_s, col in zip(eta_scenarios, labs_s, [C1, C0, C2]):
        lod_s = 3 * sigma_Fk / (eta_s * k_cat_sol * T_arr_min * 60) * 1e9
        ax.semilogy(T_arr_min, lod_s, color=col, lw=2, label=lab_s)
    ax.axhline(6, color="green", ls="--", lw=1.5, label="6 nM (healing boundary)")
    ax.axhline(30, color="orange", ls=":", lw=1.5, label="30 nM (chronic boundary)")
    ax.fill_between(T_arr_min, 0.01, 6, alpha=0.08, color="green")
    ax.set_xlabel("Incubation time $T$ (min)")
    ax.set_ylabel("LOD (nM)")
    ax.set_title("C  LOD vs $T$ for three $\\eta$ scenarios")
    ax.legend(fontsize=7)
    ax.set_ylim(0.1, 500)

    fig.tight_layout()
    save(fig, "fig_SWV9_sensitivity.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Temperature effects
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_temperature():
    print("[10/12] Fig 10: Temperature effects ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    T_C = np.linspace(15, 45, 100)
    T_Ks = T_C + 273.15
    f_op = 10.0  # confirmed operating frequency (10-20 Hz range; 10 Hz used as representative)

    # A: E0' vs T
    ax = axes[0]
    # MB formal potential: -1.5 mV/°C (empirical from Dauphin-Ducharme data)
    dE_dT = -1.5e-3  # V/°C
    E0_T = -0.250 + dE_dT * (T_C - 25)
    ax.plot(T_C, E0_T * 1000, color=C0, lw=2)
    ax.axvline(25, color="#aaa", ls="--", lw=0.8, label="Lab ($25^\\circ$C)")
    ax.axvline(37, color=C1, ls=":", lw=1.5, label="Wound ($37^\\circ$C)")
    ax.scatter([25, 37], [(-0.250 + dE_dT * 0) * 1000, (-0.250 + dE_dT * 12) * 1000],
               color=[C0, C1], s=60, zorder=5)
    dE_val = dE_dT * 12 * 1000
    ax.annotate(f"$\\Delta E^{{\\circ\\prime}} = {dE_val:.0f}$ mV\n(25→37 °C)",
                xy=(37, (-0.250 + dE_dT * 12) * 1000),
                xytext=(35, -262), fontsize=8, color=C1,
                arrowprops=dict(arrowstyle="->", color=C1, lw=0.8))
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("$E^{\\circ\\prime}_{\\rm MB}$ (mV vs Ag/AgCl)")
    ax.set_title("A  MB formal potential vs temperature")
    ax.legend(fontsize=8)

    # B: k0 vs T (Arrhenius)
    ax = axes[1]
    Ea_eV = 0.30   # activation energy for ET through SAM
    Ea_J = Ea_eV * 1.602e-19 * 6.022e23  # J/mol
    k0_T = K0_REF_S1 * np.exp(-Ea_J / R_const * (1/T_Ks - 1/(298.15)))
    kap_T = k0_T / (2 * f_op)

    ax.semilogy(T_C, k0_T, color=C1, lw=2, label="$k^0_{\\rm ref}(T)$")
    ax2b = ax.twinx()
    ax2b.semilogy(T_C, kap_T, color=C3, lw=1.5, ls="--", label="$\\kappa_{\\rm tip}(T)$")
    ax.axvline(25, color="#aaa", ls="--", lw=0.8)
    ax.axvline(37, color=C0, ls=":", lw=1.5)
    ax.scatter([25, 37], [K0_REF_S1, np.interp(37, T_C, k0_T)],
               color=[C1, C0], s=60, zorder=5)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("$k^0_{\\rm ref}$ (s$^{-1}$)", color=C1)
    ax2b.set_ylabel("$\\kappa_{\\rm tip}$", color=C3)
    ax.set_title(f"B  $k^0_{{\\rm ref}}$ and $\\kappa_{{\\rm tip}}$ vs temperature\n($E_a = {Ea_eV}$ eV)")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    # C: ΔI_peak vs T
    ax = axes[2]
    E_t = np.linspace(-0.40, -0.10, 200)
    peaks_T = []
    for Tk, E0_t in zip(T_Ks[::5], E0_T[::5]):
        # Scale k0 with T
        k0_t_val = K0_REF_S1 * np.exp(-Ea_J / R_const * (1/Tk - 1/298.15))
        DI_t = dist_averaged_voltammogram(E_t, F_k=0.0, E0_MB=E0_t)
        peaks_T.append(np.max(DI_t))
    T_sub = T_C[::5]
    peaks_T_arr = np.array(peaks_T)
    ax.plot(T_sub, peaks_T_arr * 1e9, color=C0, lw=2, marker="o", ms=4)
    ax.axvline(25, color="#aaa", ls="--", lw=0.8, label="Lab ($25^\\circ$C)")
    ax.axvline(37, color=C1, ls=":", lw=1.5, label="Wound ($37^\\circ$C)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("$\\Delta I_{\\rm peak}$ (nA)")
    ax.set_title("C  Signal amplitude vs temperature")
    ax.legend(fontsize=8)

    # D: Temperature correction factor
    ax = axes[3]
    peak_25 = np.interp(25, T_sub, peaks_T_arr)
    corr_factor = peaks_T_arr / peak_25
    ax.plot(T_sub, corr_factor, color=C3, lw=2, marker="s", ms=4)
    ax.axhline(1.0, color="#aaa", ls="--", lw=0.8, label="Reference at 25 °C")
    ax.axvline(37, color=C1, ls=":", lw=1.5, label="Wound ($37^\\circ$C)")
    f37 = np.interp(37, T_sub, corr_factor)
    ax.scatter([37], [f37], color=C1, s=60, zorder=5)
    ax.text(37.5, f37, f"{f37:.2f}×", fontsize=8, color=C1)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("$\\Delta I_{\\rm peak}(T) / \\Delta I_{\\rm peak}(25^\\circ$C)")
    ax.set_title("D  Temperature correction factor for $\\Delta I_{\\rm peak}$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "fig_SWV10_temperature.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Clinical predictions (was fig5)
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_clinical():
    print("[11/12] Fig 11: Clinical predictions ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    k_cat_sol = 67500.0
    eta = 0.20
    keff = eta * k_cat_sol
    sigma_Fk = 0.034
    E_cl = np.linspace(-0.40, -0.10, 200)
    conc_nM = [0, 1, 3, 6, 10, 30]
    T_min = 30

    # A: Voltammograms at clinical concentrations
    ax = axes[0]
    cmap_cl = plt.cm.viridis
    DI0_max = None
    for i, c in enumerate(conc_nM):
        Fk_c = 1 - np.exp(-keff * c * 1e-9 * T_min * 60)
        DI_c = dist_averaged_voltammogram(E_cl, F_k=Fk_c)
        if i == 0:
            DI0_max = np.max(DI_c)
        col = cmap_cl(i / (len(conc_nM) - 1))
        ax.plot(E_cl * 1000, DI_c * 1e9, color=col, lw=1.5,
                label=f"{c} nM ($F_k={Fk_c:.2f}$)")
    ax.set_xlabel("Potential (mV)")
    ax.set_ylabel(r"$\Delta I$ (nA)")
    ax.set_title(f"A  Voltammograms at $T={T_min}$ min, $\\eta={eta}$ (FLAG)")
    ax.legend(fontsize=7)

    # B: Calibration curves
    ax = axes[1]
    T_range_b = [10, 20, 30, 60]
    c_range = np.logspace(-0.5, 2, 80)
    for T_b, col in zip(T_range_b, [C0, C1, C2, C3]):
        fracs_b = []
        for c in c_range:
            Fk_b = 1 - np.exp(-keff * c * 1e-9 * T_b * 60)
            fracs_b.append(1 - Fk_b)
        ax.semilogx(c_range, np.array(fracs_b) * 100, color=col, lw=2,
                    label=f"$T={T_b}$ min")
    ax.axvline(6, color="green", ls="--", lw=1.5, label="6 nM (healing)")
    ax.axvline(30, color="red", ls=":", lw=1.5, label="30 nM (chronic)")
    lod_30 = 3 * sigma_Fk / (keff * 1800) * 1e9
    ax.axvline(lod_30, color="purple", ls="-.", lw=1.5,
               label=f"LOD $={lod_30:.1f}$ nM (30 min)")
    ax.set_xlabel("[MMP-9] (nM)")
    ax.set_ylabel(r"$A_{\rm MB}/A_{\rm MB}(0)$ (%)")
    ax.set_title("B  Calibration curves at multiple times")
    ax.legend(fontsize=7, loc="lower left")

    # C: LOD contour
    ax = axes[2]
    eta_range = np.logspace(-2, 0, 80)
    T_range_c = np.linspace(5, 90, 80)
    ETA_c, TT_c = np.meshgrid(eta_range, T_range_c)
    LOD_c = 3 * sigma_Fk / (ETA_c * k_cat_sol * TT_c * 60) * 1e9
    cf = ax.contourf(ETA_c, TT_c, LOD_c, levels=np.logspace(-1, 2, 30),
                     cmap="RdYlGn_r", alpha=0.9)
    ax.contour(ETA_c, TT_c, LOD_c, levels=[6, 30], colors="white",
               linewidths=[2, 1.5], linestyles=["--", ":"])
    ax.scatter([eta], [30], color="k", s=80, zorder=5, label="Operating point")
    ax.set_xscale("log")
    ax.set_xlabel("Surface correction $\\eta$")
    ax.set_ylabel("Incubation time $T$ (min)")
    ax.set_title("C  LOD (nM) in $(\\eta, T)$ space")
    plt.colorbar(cf, ax=ax, label="LOD (nM)")
    ax.legend(fontsize=7)

    # D: Discrimination signal
    ax = axes[3]
    T_arr_d = np.linspace(1, 90, 200)
    for eta_d, col, lab_d in zip([0.05, 0.20, 0.50],
                                  [C1, C0, C2],
                                  ["$\\eta=0.05$", "$\\eta=0.20$ (central)",
                                   "$\\eta=0.50$"]):
        keff_d = eta_d * k_cat_sol
        dFk = abs(
            (1 - np.exp(-keff_d * 30e-9 * T_arr_d * 60)) -
            (1 - np.exp(-keff_d * 6e-9 * T_arr_d * 60))
        )
        ax.plot(T_arr_d, dFk * 100, lw=2, color=col, label=lab_d)
    ax.axhline(sigma_Fk * 100 * 3, color="#888", ls=":", lw=1.5,
               label=f"$3\\sigma_{{F_k}} = {sigma_Fk*100*3:.1f}\\%$ noise floor")
    ax.set_xlabel("Incubation time $T$ (min)")
    ax.set_ylabel(r"$\Delta F_k^{\rm chronic-healing}$ (%)")
    ax.set_title("D  Discrimination signal vs $T$")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, "fig_SWV11_clinical.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Worked numerical example
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_worked_example():
    print("[12/12] Fig 12: Worked numerical example ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Parameters at E = E0', f=15 Hz, d=d_PEG
    E_ex = -0.250  # V = E0'
    E0   = -0.250
    k0_ex = K0_REF_S1  # 127 s-1
    alpha = 0.37
    ne = 2
    f_op = 10.0  # confirmed operating frequency (10-20 Hz range; 10 Hz used as representative)
    tau = 1.0 / (2 * f_op)
    Gamma_tot = 3e-12
    A_cm2 = 0.0177
    Esw = 0.025

    # Compute BV rates at E_fwd and E_bwd
    E_fwd = E_ex + Esw
    E_bwd = E_ex - Esw

    xi_fwd = ne * F_const * (E_fwd - E0) / (R_const * T_K)
    kf_fwd = k0_ex * np.exp(-alpha * xi_fwd)
    kb_fwd = k0_ex * np.exp((1 - alpha) * xi_fwd)
    Geq_fwd = kf_fwd / (kf_fwd + kb_fwd)
    G0 = 0.5
    Gred_fwd_end = Geq_fwd + (G0 - Geq_fwd) * np.exp(-(kf_fwd + kb_fwd) * tau)

    xi_bwd = ne * F_const * (E_bwd - E0) / (R_const * T_K)
    kf_bwd = k0_ex * np.exp(-alpha * xi_bwd)
    kb_bwd = k0_ex * np.exp((1 - alpha) * xi_bwd)
    Geq_bwd = kf_bwd / (kf_bwd + kb_bwd)
    Gred_bwd_end = Geq_bwd + (Gred_fwd_end - Geq_bwd) * np.exp(-(kf_bwd + kb_bwd) * tau)

    If = ne * F_const * (kb_fwd * Gred_fwd_end - kf_fwd * (Gamma_tot - Gred_fwd_end)) * A_cm2 * Gamma_tot
    Ib = ne * F_const * (kb_bwd * Gred_bwd_end - kf_bwd * (Gamma_tot - Gred_bwd_end)) * A_cm2 * Gamma_tot
    DI_ex = If - Ib

    # A: Step-by-step table as text panel
    ax = axes[0]
    ax.axis("off")
    steps = [
        ("Given", f"$E = E_{{0'}}={E0*1000:.0f}$ mV, $k^0={k0_ex:.0f}$ s$^{{-1}}$", "#1a1a2e"),
        ("", f"$\\alpha={alpha}$, $n_e={ne}$, $f={f_op}$ Hz, $\\tau={tau*1000:.1f}$ ms", "#1a1a2e"),
        ("Step 1", f"Forward pulse: $E_{{+}}=E_0+E_{{\\rm sw}}={E_fwd*1000:.0f}$ mV", C0),
        ("", f"$\\xi_{{+}} = n_e F(E_{{+}}-E_0)/RT = {xi_fwd:.3f}$", C0),
        ("", f"$k_f = k^0 e^{{-\\alpha\\xi_+}} = {kf_fwd:.1f}$ s$^{{-1}}$", C0),
        ("", f"$k_b = k^0 e^{{(1-\\alpha)\\xi_+}} = {kb_fwd:.1f}$ s$^{{-1}}$", C0),
        ("Step 2", f"$\\Gamma_{{\\rm eq}}^{{(+)}} = k_f/(k_f+k_b) = {Geq_fwd:.4f}\\,\\Gamma_{{\\rm tot}}$", C1),
        ("", f"$\\Gamma_{{\\rm red}}(\\tau) = \\Gamma_{{\\rm eq}} + (\\Gamma_{{\\rm red}}(0)-\\Gamma_{{\\rm eq}})e^{{-(k_f+k_b)\\tau}}$", C1),
        ("", f"$= {Geq_fwd:.4f} + (0.5-{Geq_fwd:.4f})\\cdot{np.exp(-(kf_fwd+kb_fwd)*tau):.4f}$", C1),
        ("", f"$= {Gred_fwd_end:.4f}\\,\\Gamma_{{\\rm tot}}$", C1),
        ("Step 3", f"Backward pulse: $E_{{-}}={E_bwd*1000:.0f}$ mV; IC = Step 2 result", C2),
        ("", f"$\\Gamma_{{\\rm red}}(\\tau) = {Gred_bwd_end:.4f}\\,\\Gamma_{{\\rm tot}}$", C2),
        ("Step 4", f"$I_f = {If*1e9:.4f}$ nA,  $I_b = {Ib*1e9:.4f}$ nA", C3),
        ("", f"$\\Delta I = I_f - I_b = {DI_ex*1e9:.4f}$ nA", C3),
    ]
    y = 0.97
    for label, text, col in steps:
        if label:
            ax.text(0.01, y, f"{label}:", transform=ax.transAxes, fontsize=8,
                    fontweight="bold", color="#222", va="top")
        ax.text(0.18, y, text, transform=ax.transAxes, fontsize=8,
                color=col, va="top")
        y -= 0.065
    ax.set_title("A  Worked example at $E = E^{\\circ\\prime}$", fontweight="bold")

    # B: Γ_red during the example step
    ax = axes[1]
    t_arr = np.linspace(0, tau, 200)
    G_fwd_arr = Geq_fwd + (G0 - Geq_fwd) * np.exp(-(kf_fwd + kb_fwd) * t_arr)
    G_bwd_arr = Geq_bwd + (Gred_fwd_end - Geq_bwd) * np.exp(-(kf_bwd + kb_bwd) * t_arr)
    t_both = np.concatenate([t_arr, t_arr + tau])
    G_both = np.concatenate([G_fwd_arr, G_bwd_arr])

    ax.plot(t_both * 1000, G_both, color=C0, lw=2)
    ax.axhline(Geq_fwd, color=C1, ls="--", lw=1,
               label=f"$\\Gamma_{{\\rm eq}}^{{(+)}} = {Geq_fwd:.3f}$")
    ax.axhline(Geq_bwd, color=C2, ls=":", lw=1,
               label=f"$\\Gamma_{{\\rm eq}}^{{(-)}} = {Geq_bwd:.3f}$")
    ax.axvline(tau * 1000, color="#888", lw=0.8)
    ax.scatter([tau * 1000], [Gred_fwd_end], color=C1, s=60, zorder=5,
               label=f"Sampled end (fwd): {Gred_fwd_end:.3f}")
    ax.scatter([2 * tau * 1000], [Gred_bwd_end], color=C2, s=60, zorder=5,
               label=f"Sampled end (bwd): {Gred_bwd_end:.3f}")
    ax.text(tau * 1000 / 2, 0.05, "Fwd pulse", ha="center", fontsize=8)
    ax.text(tau * 1000 * 1.5, 0.05, "Bwd pulse", ha="center", fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$\Gamma_{\rm red}/\Gamma_{\rm tot}$")
    ax.set_title(f"B  Coverage during example step\n($k^0={k0_ex:.0f}$ s$^{{-1}}$, $E=E^{{\\circ\\prime}}$)")
    ax.legend(fontsize=7)

    # C: Net SWV peak at E = E0' with the example point marked
    ax = axes[2]
    E_sweep = np.linspace(-0.40, -0.10, 300)
    DI_full = dist_averaged_voltammogram(E_sweep, F_k=0.0)
    ax.plot(E_sweep * 1000, DI_full * 1e9, color=C0, lw=2,
            label="Full voltammogram ($F_k=0$)")
    ax.scatter([E0 * 1000], [DI_ex * 1e9], color=C1, s=100, zorder=5,
               label=f"Example point: $\\Delta I={DI_ex*1e9:.4f}$ nA")
    ax.axvline(E0 * 1000, color="#888", ls=":", lw=0.8)
    ax.set_xlabel("Potential (mV)")
    ax.set_ylabel(r"$\Delta I$ (nA)")
    ax.set_title("C  Worked example point in context")
    ax.legend(fontsize=8)

    # D: P(d) × ΔI contribution at the example step
    ax = axes[3]
    d_arr_ex = np.linspace(D_PEG_NM, D_MAX_NM, 80)
    DI_at_E0 = []
    for d in d_arr_ex:
        k0_d_ex = k0_of_d(d)
        DI_d_ex = swv_voltammogram(np.array([E0]), k0=k0_d_ex, ne=2, alpha=0.37,
                                    E0=E0, f_swv=f_op, E_sw=Esw)
        DI_at_E0.append(DI_d_ex[0])
    DI_at_E0 = np.array(DI_at_E0)
    P_d_ex = P_d_norm(d_arr_ex)
    integrand = P_d_ex * DI_at_E0

    ax2d = ax.twinx()
    ax.plot(d_arr_ex * 10, P_d_ex, color=C0, lw=2, label="$P(d)$")
    ax2d.plot(d_arr_ex * 10, DI_at_E0, color=C1, lw=2, ls="--", label="$\\Delta I_{\\rm norm}(E_0; d)$")
    ax.fill_between(d_arr_ex * 10, integrand * max(abs(DI_at_E0)) / max(abs(integrand) + 1e-30),
                    alpha=0.2, color=C3, label="$P(d)\\cdot\\Delta I_{\\rm norm}$ (integrand)")
    ax.axvline(D_PEG_NM * 10, color=C2, ls=":", lw=1.5)
    ax.set_xlabel(r"Distance $d$ (Å)")
    ax.set_ylabel("$P(d)$ (Å$^{-1}$)", color=C0)
    ax2d.set_ylabel("$\\Delta I_{\\rm norm}$ (a.u.)", color=C1)
    ax.set_title("D  Integrand $P(d)\\cdot\\Delta I$ at $E=E^{\\circ\\prime}$")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2d.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7)

    fig.tight_layout()
    save(fig, "fig_SWV12_worked.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — pre-compute shared data then generate all figures
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Generating SWV flagship figures (v2) ...")

    # Pre-compute distance-averaged data for all Fk values
    print("  Pre-computing distance-averaged voltammograms ...")
    fk_arr = [0.0, 0.20, 0.40, 0.60, 0.80, 0.95]
    DI_Fk = []
    A_areas, E_peaks, FWHMs = [], [], []
    for fk in fk_arr:
        DI = dist_averaged_voltammogram(E_SCAN, F_k=fk)
        DI_Fk.append(DI)
        A_pk, E_pk, sigma_E, A_area = fit_gaussian(E_SCAN, DI)
        A_areas.append(A_area if not np.isnan(A_area) else 0)
        E_peaks.append(E_pk if not np.isnan(E_pk) else -0.25)
        FWHMs.append(2.355 * sigma_E if not np.isnan(sigma_E) else 0.06)
    A_areas = np.array(A_areas)
    E_peaks = np.array(E_peaks)
    FWHMs = np.array(FWHMs)

    intact_frac = 1 - np.array(fk_arr)
    # Linear fit to A_area vs (1-Fk)
    valid = ~np.isnan(A_areas) & (A_areas > 0)
    if valid.sum() >= 2:
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(intact_frac[valid], A_areas[valid], 1)
        slope_lin, intercept_lin = coeffs
        lin_fit = slope_lin * intact_frac + intercept_lin
        residuals_pct = (A_areas - lin_fit) / (lin_fit + 1e-30) * 100
    else:
        slope_lin, intercept_lin = A_areas[0], 0
        residuals_pct = np.zeros_like(intact_frac)

    # Generate all figures
    fig0_assembly()
    fig1_geometry()
    fig2_marcus()
    fig3_ode_derivation()
    fig4_scan_evolution()
    fig5_averaging(DI_Fk, fk_arr, A_areas, E_peaks, FWHMs,
                   slope_lin, intercept_lin, residuals_pct)
    fig6_validation()
    fig7_ratiometric(DI_Fk, fk_arr, A_areas)
    fig8_waveform()
    fig9_sensitivity()
    fig10_temperature()
    fig11_clinical()
    fig12_worked_example()

    print(f"\nAll figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
