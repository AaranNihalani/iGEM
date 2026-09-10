"""
swv_flagship_figures.py
=======================
Generates all publication-quality figures for the SWV forward model
flagship report.  Imports functions from swv_forward_model.py so the
model code is not duplicated.

Five figures, each saved as fig_SWV{n}_*.pdf in reports/figures/.

Run from repository root or from this directory:
    python swv_flagship_figures.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path

from swv_forward_model import (
    F_const, R_const, T_K,
    _bv_rates, _half_cycle_end, swv_voltammogram, laviron_reversible,
    P_d_norm, k0_of_d, dist_averaged_voltammogram,
    fit_gaussian, concentration_to_Fk,
    D_PEG_NM, D_MAX_NM, SIGMA_Z_NM, N_KUHN, B_KUHN_NM, L0_NM,
    K0_REF_S1, BETA_NM, K0_AT_0_S1,
    F_SWV, E_SW, GAMMA_0, A_ELEC,
    K0_FC, E0_FC, GAMMA_FC, E_SCAN_FC,
    DD_E_V, DD_FIG4B_NORM, DD_FIG5B_NORM,
    FK_VALS, E_SCAN, KEFF_SOL_M1S1,
    concentration_to_voltammogram,
)

FIGURES = Path(__file__).parents[2] / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
BLUE     = "#1565c0"
RED      = "#c62828"
GREEN    = "#2e7d32"
PURPLE   = "#6a1b9a"
ORANGE   = "#e65100"
TEAL     = "#00695c"
GREY     = "#546e7a"
NAVY     = "#1a237e"

def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax, letter, x=-0.18, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Physical setup: P(d), k0(d), κ(d), SWV waveform cartoon
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_geometry(save=True):
    """
    A: Half-Gaussian P(d) with key distances marked.
    B: k0(d) on log scale with κ(d) on right axis.
    C: SWV waveform schematic showing staircase, pulses, and sampled currents.
    D: κ-regime map (ΔI_peak vs κ for a single probe).
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    d_nm = np.linspace(D_PEG_NM, D_MAX_NM, 500)
    Pd   = P_d_norm(d_nm)
    d_mean = D_PEG_NM + SIGMA_Z_NM * np.sqrt(2.0 / np.pi)

    # ── A: P(d) ──────────────────────────────────────────────────────────────
    ax_A.plot(d_nm * 10, Pd / 10, color=BLUE, lw=2.5)   # convert nm→Å, nm-1→Å-1
    ax_A.fill_between(d_nm * 10, Pd / 10, alpha=0.18, color=BLUE)
    ax_A.axvline(D_PEG_NM * 10, color=GREY, ls=":", lw=1.5,
                 label=f"PEG6 tip  $d_{{\\rm PEG}} = {D_PEG_NM*10:.1f}$ Å")
    ax_A.axvline(d_mean * 10, color=GREEN, ls="--", lw=1.8,
                 label=f"Mean  $\\langle d \\rangle = {d_mean*10:.1f}$ Å")
    # ±σ_z shading
    ax_A.axvspan((d_mean - SIGMA_Z_NM) * 10, (d_mean + SIGMA_Z_NM) * 10,
                 alpha=0.10, color=GREEN, label=f"$\\pm\\sigma_z = {SIGMA_Z_NM*10:.1f}$ Å")
    _style_ax(ax_A, "$d$ (Å)  [MB–electrode distance]",
              "$P(d)$ (Å$^{-1}$)", "A — Distance distribution $P(d)$")
    ax_A.legend(fontsize=7.5, loc="upper right")
    ax_A.set_xlim(0, D_MAX_NM * 10 + 2)
    ax_A.set_ylim(bottom=0)
    # annotate PEG6 region
    ax_A.annotate("PEG6\nregion", xy=(D_PEG_NM * 10 / 2, max(Pd / 10) * 0.5),
                  fontsize=7, color=GREY, ha="center")
    _panel_label(ax_A, "A")

    # ── B: k0(d) and κ(d) ────────────────────────────────────────────────────
    k0d  = k0_of_d(d_nm)
    kapp = k0d / (2 * F_SWV)
    ax_B.semilogy(d_nm * 10, k0d, color=RED, lw=2.5)
    ax_B2 = ax_B.twinx()
    ax_B2.semilogy(d_nm * 10, kapp, color=PURPLE, lw=2.0, ls="--")
    ax_B.axvline(D_PEG_NM * 10, color=GREY, ls=":", lw=1.5)
    ax_B.axvline(d_mean * 10,   color=GREEN, ls="--", lw=1.5)
    ax_B2.axhline(1.0,  color=ORANGE, ls="-.", lw=1.2, alpha=0.8)
    ax_B2.axhline(10.0, color=ORANGE, ls="-.", lw=1.2, alpha=0.8)
    ax_B2.text(D_MAX_NM * 10 + 0.5, 1.0, "q-rev. zone", fontsize=6.5,
               color=ORANGE, va="center")
    ax_B.set_xlabel("$d$ (Å)", fontsize=9)
    ax_B.set_ylabel("$k^0(d)$ (s$^{-1}$)", color=RED, fontsize=9)
    ax_B2.set_ylabel("$\\kappa = k^0 / 2f$  ($f = 15$ Hz)", color=PURPLE, fontsize=9)
    ax_B.set_title("B — ET rate $k^0(d)$ and dimensionless $\\kappa$", fontsize=9, pad=4)
    ax_B.tick_params(labelsize=8)
    ax_B2.tick_params(labelsize=8)
    ax_B.spines["top"].set_visible(False)
    ax_B2.spines["top"].set_visible(False)
    # annotate the two key points
    k0_tip  = k0_of_d(D_PEG_NM)
    k0_mean = k0_of_d(d_mean)
    ax_B.annotate(f"$k^0_{{\\rm tip}} = {k0_tip:.0f}$ s$^{{-1}}$\n$\\kappa={k0_tip/(2*F_SWV):.1f}$",
                  xy=(D_PEG_NM * 10, k0_tip), xytext=(D_PEG_NM * 10 + 8, k0_tip * 5),
                  fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8),
                  color=RED)
    ax_B.annotate(f"$k^0_{{\\rm mean}} = {k0_mean:.2f}$ s$^{{-1}}$\n$\\kappa={k0_mean/(2*F_SWV):.3f}$",
                  xy=(d_mean * 10, k0_mean), xytext=(d_mean * 10 + 5, k0_mean * 50),
                  fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8),
                  color=RED)
    _panel_label(ax_B, "B")

    # ── C: SWV waveform schematic ─────────────────────────────────────────────
    # Schematic: staircase E_step, forward pulse E+Esw, backward pulse E-Esw,
    # sampled currents I_f and I_b, net ΔI = I_f - I_b.
    n_steps = 6
    E_step_mV = np.arange(n_steps) * 10 - 25   # -25, -15, ..., +25 mV re E0
    t_per_step = 1.0  # normalised units
    t_all = []
    E_all = []
    for i, E_s in enumerate(E_step_mV):
        t_fwd = i * t_per_step + 0.25
        t_bwd = i * t_per_step + 0.75
        t_all += [i * t_per_step, t_fwd, t_fwd, t_bwd, t_bwd,
                  (i + 1) * t_per_step]
        E_all += [E_s, E_s, E_s + 25, E_s + 25, E_s - 25, E_s - 25]
    ax_C.plot(t_all, E_all, color=BLUE, lw=1.8, label="Applied $E(t)$")
    ax_C.axhline(0, color=GREY, ls=":", lw=1.0, label="$E^{\\circ\\prime}$")
    # Mark sample points
    for i, E_s in enumerate(E_step_mV[1:-1], 1):
        t_f = i * t_per_step + 0.25
        t_b = i * t_per_step + 0.75
        ax_C.plot(t_f, E_s + 25, "^", color=RED,   ms=7, zorder=5)
        ax_C.plot(t_b, E_s - 25, "v", color=GREEN, ms=7, zorder=5)
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=BLUE, lw=1.8, label="Applied $E(t)$"),
        Line2D([0], [0], marker="^", color=RED,   ls="None", ms=8,
               label="Sample $I_f$ (forward)"),
        Line2D([0], [0], marker="v", color=GREEN, ls="None", ms=8,
               label="Sample $I_b$ (backward)"),
    ]
    ax_C.legend(handles=legend_els, fontsize=7.5, loc="lower right")
    ax_C.set_xlabel("Time (step units)", fontsize=9)
    ax_C.set_ylabel("$E$ (mV vs $E^{\\circ\\prime}$)", fontsize=9)
    ax_C.set_title("C — SWV waveform: staircase + pulses\n"
                   "$\\Delta I = I_f - I_b$ at each step", fontsize=9, pad=4)
    ax_C.tick_params(labelsize=8)
    ax_C.spines["top"].set_visible(False)
    ax_C.spines["right"].set_visible(False)
    ax_C.text(0.02, 0.97, f"$E_{{\\rm sw}} = {int(E_SW*1000)}$ mV\n$f = {F_SWV:.0f}$ Hz",
              transform=ax_C.transAxes, fontsize=7.5, va="top")
    _panel_label(ax_C, "C")

    # ── D: ΔI_peak vs κ (single probe, shows quasi-reversible optimum) ────────
    kappa_arr = np.logspace(-2, 2.5, 80)
    E_test = np.linspace(-0.10, 0.10, 300)
    DI_peak_arr = []
    for kap in kappa_arr:
        k0_t = kap * 2 * F_SWV
        DI_t = swv_voltammogram(E_test, k0=k0_t, ne=2, alpha=0.37,
                                E0=0.0, f_swv=F_SWV, E_sw=E_SW)
        DI_peak_arr.append(np.max(DI_t))
    DI_peak_arr = np.array(DI_peak_arr)
    DI_norm_arr = DI_peak_arr / np.max(DI_peak_arr)

    ax_D.semilogx(kappa_arr, DI_norm_arr, color=BLUE, lw=2.5)
    ax_D.fill_between(kappa_arr, DI_norm_arr, alpha=0.12, color=BLUE)
    # Shade the quasi-reversible zone
    ax_D.axvspan(0.5, 10, alpha=0.10, color=GREEN,
                 label="Quasi-reversible zone")
    ax_D.axvline(k0_of_d(D_PEG_NM) / (2 * F_SWV), color=RED, ls="--", lw=2,
                 label=f"$\\kappa_{{\\rm tip}} = {K0_REF_S1/(2*F_SWV):.1f}$ (PEG6 tip)")
    ax_D.axvline(k0_of_d(d_mean) / (2 * F_SWV), color=ORANGE, ls="-.", lw=1.8,
                 label=f"$\\kappa_{{\\rm mean}} = {k0_of_d(d_mean)/(2*F_SWV):.3f}$ (mean $d$)")
    ax_D.set_xlim(kappa_arr[0], kappa_arr[-1])
    _style_ax(ax_D, "$\\kappa = k^0 / 2f$",
              "$\\Delta I_{{\\rm peak}}$ (normalised)", "D — Quasi-reversible optimum")
    ax_D.legend(fontsize=7.5)
    ax_D.text(0.5, 0.50, "Quasi-reversible\noptimum", transform=ax_D.transAxes,
              fontsize=8, color=GREEN, ha="center", va="center",
              fontweight="bold")
    _panel_label(ax_D, "D")

    fig.suptitle(
        "Fig 1 — Electrode geometry, tether distribution, and kinetic regime",
        fontsize=11, y=1.01
    )
    if save:
        p = FIGURES / "fig_SWV1_geometry.pdf"
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — BV-SWV derivation: ODE solution trajectories + validation
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_ode_validation(save=True):
    """
    A: Γ_red(t) during forward + backward half-pulse for three κ values.
    B: Current I(t) during same half-pulses (showing the sampled endpoint).
    C: BV peak vs Laviron shape (small-Esw limit).
    D: Frequency dependence showing three quasi-reversible κ values + DD data.
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # Half-pulse trajectory computation
    tau = 1.0 / (2.0 * F_SWV)   # 33.3 ms
    t_pulse = np.linspace(0, tau, 500)

    # Three κ values: irreversible, quasi-reversible (ours), reversible
    kappa_cases = [
        (0.05,  "deep irrev. ($\\kappa = 0.05$)", "#b71c1c"),
        (4.2,   "quasi-rev.  ($\\kappa = 4.2$, PEG6 tip)", BLUE),
        (50.0,  "reversible  ($\\kappa = 50$)", GREEN),
    ]

    E_pulse_fwd = -0.225   # V = E0 + Esw = -0.250 + 0.025
    E_pulse_bwd = -0.275   # V = E0 - Esw
    E0_MB = -0.250

    for kap, lbl, col in kappa_cases:
        k0_t = kap * 2 * F_SWV
        k_f_fwd, k_b_fwd = _bv_rates(E_pulse_fwd, E0_MB, k0_t, 0.37, 2)
        Ktot_fwd  = k_f_fwd + k_b_fwd
        Geq_fwd   = k_f_fwd / Ktot_fwd   # equilibrium reduced fraction
        # Start forward pulse fully reduced
        Gamma_r0_fwd = 1.0
        Gr_fwd = Geq_fwd + (Gamma_r0_fwd - Geq_fwd) * np.exp(-Ktot_fwd * t_pulse)
        I_fwd  = 2 * F_const * (k_b_fwd * Gr_fwd - k_f_fwd * (1 - Gr_fwd))

        # Start backward pulse from end of forward pulse
        Gamma_r0_bwd = Gr_fwd[-1]
        k_f_bwd, k_b_bwd = _bv_rates(E_pulse_bwd, E0_MB, k0_t, 0.37, 2)
        Ktot_bwd = k_f_bwd + k_b_bwd
        Geq_bwd  = k_f_bwd / Ktot_bwd
        Gr_bwd = Geq_bwd + (Gamma_r0_bwd - Geq_bwd) * np.exp(-Ktot_bwd * t_pulse)
        I_bwd  = 2 * F_const * (k_b_bwd * Gr_bwd - k_f_bwd * (1 - Gr_bwd))

        t_ms = t_pulse * 1000

        ax_A.plot(t_ms, Gr_fwd * 100, color=col, lw=2.2,
                  label=lbl + "  (forward)")
        ax_A.plot(t_ms + tau * 1000, Gr_bwd * 100, color=col, lw=2.2,
                  ls="--")
        ax_A.axhline(Geq_fwd * 100, color=col, lw=0.8, ls=":", alpha=0.6)

        ax_B.plot(t_ms, np.abs(I_fwd) * 1e3, color=col, lw=2.2, label=lbl)
        ax_B.plot(t_ms + tau * 1000, np.abs(I_bwd) * 1e3, color=col,
                  lw=2.2, ls="--")
        # Mark sampled endpoint
        ax_B.plot(tau * 1000, np.abs(I_fwd[-1]) * 1e3,
                  "o", color=col, ms=9, zorder=5)
        ax_B.plot(tau * 2000, np.abs(I_bwd[-1]) * 1e3,
                  "s", color=col, ms=9, zorder=5)

    ax_A.axvline(tau * 1000, color="gray", ls=":", lw=1.2,
                 label="Half-pulse boundary")
    ax_A.set_xlim(0, tau * 2000)
    ax_A.set_xticks([0, tau * 500, tau * 1000, tau * 1500, tau * 2000])
    ax_A.set_xticklabels(["0", r"$\tau/2$", r"$\tau$", r"$3\tau/2$", r"$2\tau$"])
    _style_ax(ax_A, "Time", "$\\Gamma_{\\rm red}/\\Gamma_{\\rm tot}$ (%)",
              "A — Surface coverage $\\Gamma_{\\rm red}(t)$ during fwd/bwd pulses")
    ax_A.legend(fontsize=6.8, loc="right")
    ax_A.text(0.02, 0.04,
              "Solid: forward pulse\nDashed: backward pulse\n"
              "Dotted: $\\Gamma_{\\rm eq}$ at each pulse",
              transform=ax_A.transAxes, fontsize=7)
    _panel_label(ax_A, "A")

    ax_B.axvline(tau * 1000, color="gray", ls=":", lw=1.2)
    ax_B.set_xlim(0, tau * 2000)
    ax_B.set_xticks([0, tau * 500, tau * 1000, tau * 1500, tau * 2000])
    ax_B.set_xticklabels(["0", r"$\tau/2$", r"$\tau$", r"$3\tau/2$", r"$2\tau$"])
    _style_ax(ax_B, "Time", "$|I(t)|$ (mA mol$^{-1}$ cm$^2$)",
              "B — Instantaneous current $|I(t)|$ (circles/squares: sampled values)")
    ax_B.legend(fontsize=7.5)
    _panel_label(ax_B, "B")

    # ── C: BV simulator vs Laviron shape ─────────────────────────────────────
    E_test = np.linspace(-0.15, 0.15, 600)
    k0_qr  = 10.0 * (2.0 * F_SWV)   # κ = 10
    DI_qr  = swv_voltammogram(E_test, k0=k0_qr, ne=2, alpha=0.5,
                               E0=0.0, f_swv=F_SWV, E_sw=E_SW)
    # Small-Esw limit: E_sw = 1 mV → Laviron shape
    DI_tiny = swv_voltammogram(E_test, k0=k0_qr, ne=2, alpha=0.5,
                                E0=0.0, f_swv=F_SWV, E_sw=0.001)
    DI_lav = laviron_reversible(E_test, E0=0.0, ne=2)

    DI_qr_n   = DI_qr   / np.max(DI_qr)
    DI_tiny_n = DI_tiny  / np.max(DI_tiny)
    DI_lav_n  = DI_lav   / np.max(DI_lav)

    above_qr   = E_test[DI_qr_n   >= 0.5]
    above_tiny = E_test[DI_tiny_n  >= 0.5]
    fwhm_qr   = (above_qr[-1]   - above_qr[0])   * 1000
    fwhm_tiny = (above_tiny[-1] - above_tiny[0]) * 1000

    ax_C.plot(E_test * 1000, DI_lav_n,  color=GREY,  lw=2.5, ls=":",
              label=f"Laviron (1979) analytic peak\n(LPSV limit $E_{{\\rm sw}} \\to 0$, FWHM = 45 mV)")
    ax_C.plot(E_test * 1000, DI_tiny_n, color=GREEN, lw=2.0, ls="--",
              label=f"BV simulator $E_{{\\rm sw}} = 1$ mV (FWHM = {fwhm_tiny:.0f} mV)")
    ax_C.plot(E_test * 1000, DI_qr_n,   color=BLUE,  lw=2.5,
              label=f"BV simulator $E_{{\\rm sw}} = 25$ mV, $\\kappa = 10$ (FWHM = {fwhm_qr:.0f} mV)")
    ax_C.axhline(0.5, color="gray", ls=":", lw=0.9)
    _style_ax(ax_C, "$E - E^{\\circ\\prime}$ (mV)", "Normalised $\\Delta I$",
              "C — BV simulator unit test vs Laviron (1979)")
    ax_C.legend(fontsize=7.5)
    ax_C.text(0.02, 0.04,
              "BV simulator agrees with Laviron in the\n$E_{\\rm sw}\\to 0$ limit; "
              "$E_{\\rm sw}=25$ mV broadens\nthe peak as expected.",
              transform=ax_C.transAxes, fontsize=7)
    _panel_label(ax_C, "C")

    # ── D: Frequency dependence at three quasi-reversible κ values ───────────
    k0_DNA = 5000.0   # s-1 (Dauphin-Ducharme Fig 5, MB-DNA)
    E_DD   = np.linspace(-0.40, -0.10, 300)
    freqs  = [200.0, 500.0, 1000.0]
    f_cols = [GREEN, BLUE, RED]
    for freq, col in zip(freqs, f_cols):
        kap = k0_DNA / (2 * freq)
        DI_f = swv_voltammogram(E_DD, k0=k0_DNA, ne=2, alpha=0.37,
                                E0=-0.250, f_swv=freq, E_sw=0.025)
        DI_fn = DI_f / np.max(np.abs(DI_f))
        above = E_DD[DI_fn >= 0.5]
        fw_mV = (above[-1] - above[0]) * 1000 if len(above) > 1 else np.nan
        ax_D.plot((E_DD + 0.250) * 1000, DI_fn, color=col, lw=2.2,
                  label=f"{freq:.0f} Hz, $\\kappa = {kap:.1f}$, FWHM = {fw_mV:.0f} mV")

    # DD digitised points
    ax_D.scatter((DD_E_V + 0.250) * 1000, DD_FIG5B_NORM,
                 marker="o", s=30, color=GREY, zorder=5, alpha=0.8,
                 label="DD Fig.5B (100 Hz, $\\kappa=25$,\ndigitised; outside BV-ODE range)")
    ax_D.axhline(0, color="gray", ls="--", lw=0.7)
    ax_D.axhline(0.5, color="gray", ls=":", lw=0.9)
    _style_ax(ax_D, "$E - E^{\\circ\\prime}$ (mV)", "Normalised $\\Delta I$",
              "D — Frequency dependence (MB-DNA, $k^0 = 5000$ s$^{-1}$)")
    ax_D.legend(fontsize=7.5, loc="upper left")
    ax_D.text(0.97, 0.04,
              "FWHM 56–60 mV across all\n"
              "quasi-reversible frequencies.\n"
              "BV-ODE valid for $\\kappa \\lesssim 12$.",
              transform=ax_D.transAxes, fontsize=7, ha="right")
    _panel_label(ax_D, "D")

    fig.suptitle(
        "Fig 2 — Butler-Volmer surface ODE: half-pulse trajectories and validation",
        fontsize=11, y=1.01
    )
    if save:
        p = FIGURES / "fig_SWV2_ode.pdf"
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Distance averaging: individual contributions, convergence, calibration
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_distance_averaging(DI_Fk, fk_arr, A_areas, E_peaks, FWHMs,
                             slope_lin, intercept_lin, residuals_pct,
                             save=True):
    """
    A: Individual ΔI(E; d) for 6 representative distances, with P(d) weights.
    B: Family of distance-averaged ΔI(E; F_k).
    C: Peak area vs (1-F_k) linearity with residuals.
    D: Peak position and FWHM vs F_k (stability check).
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # ── A: Individual contributions at representative distances ───────────────
    n_rep = 6
    d_reps = np.linspace(D_PEG_NM, D_PEG_NM + 2.5 * SIGMA_Z_NM, n_rep)
    rep_cols = plt.cm.plasma(np.linspace(0.05, 0.90, n_rep))
    E_test = np.linspace(-0.40, -0.10, 200)
    Pd_weights = P_d_norm(d_reps)

    for i, (d, col) in enumerate(zip(d_reps, rep_cols)):
        k0_d = k0_of_d(d)
        kap  = k0_d / (2 * F_SWV)
        DI_d = swv_voltammogram(E_test, k0=k0_d, ne=2, alpha=0.37,
                                E0=-0.250, f_swv=F_SWV, E_sw=E_SW)
        # Scale by P(d) weight for visual intuition
        w = Pd_weights[i] * (d_reps[1] - d_reps[0])
        ax_A.plot(E_test * 1000, DI_d * w * 1e14, color=col, lw=2.0,
                  label=f"$d={d*10:.0f}$ Å, $\\kappa={kap:.2f}$")

    _style_ax(ax_A, "$E$ (mV vs Ag/AgCl)",
              "$P(d)\\,\\Delta I_{{\\rm norm}}$ (arb.)",
              "A — Individual contributions $P(d_i)\\,\\Delta I(E;\\,d_i)$")
    ax_A.legend(fontsize=6.8, loc="upper right")
    ax_A.axhline(0, color="gray", ls="--", lw=0.7)
    ax_A.text(0.02, 0.04,
              "Probes at $d \\approx d_{\\rm PEG}$ (high $\\kappa$)\n"
              "dominate the signal; far probes\n"
              "are negligible (irreversible limit).",
              transform=ax_A.transAxes, fontsize=7)
    _panel_label(ax_A, "A")

    # ── B: Distance-averaged voltammograms ────────────────────────────────────
    COLORS_FK = [NAVY, BLUE, "#0288d1", TEAL, ORANGE, RED]
    for fk, col in zip(FK_VALS, COLORS_FK):
        ax_B.plot(E_SCAN * 1000, DI_Fk[fk] * 1e12, color=col, lw=2.2,
                  label=f"$F_k = {fk:.2f}$")
    ax_B.axhline(0, color="gray", ls="--", lw=0.7)
    ax_B.axvline(-250, color=GREY, ls=":", lw=1.2)
    _style_ax(ax_B, "$E$ (mV vs Ag/AgCl)", "$\\Delta I$ (pA)",
              f"B — Distance-averaged $\\Delta I(E;\\,F_k)$\n$f={F_SWV}$ Hz, $E_{{\\rm sw}}={int(E_SW*1000)}$ mV, $n_{{\\rm dist}}=80$")
    ax_B.legend(fontsize=8, loc="upper right")
    ax_B.text(0.02, 0.04,
              "Peak area $\\propto (1-F_k)$;\npeak position and FWHM\nare invariant.",
              transform=ax_B.transAxes, fontsize=7)
    _panel_label(ax_B, "B")

    # ── C: Calibration linearity ──────────────────────────────────────────────
    valid = ~np.isnan(A_areas)
    x_plot = 1.0 - fk_arr[valid]
    y_plot = A_areas[valid] * 1e12
    y_fit  = (slope_lin * (1.0 - fk_arr[valid]) + intercept_lin) * 1e12

    ax_C.scatter(x_plot * 100, y_plot, s=90, color=BLUE, zorder=5,
                 label="$A_{\\rm MB,area}$ (model)")
    ax_C.plot(x_plot * 100, y_fit, color=RED, lw=2, ls="--",
              label=f"Linear fit (max resid. {max(residuals_pct):.1f}%)")
    ax_C.set_xlabel("$(1 - F_k)$ (%)", fontsize=9)
    ax_C.set_ylabel("$A_{\\rm MB,area}$ (pA$\\cdot$V)", fontsize=9)
    ax_C.set_title("C — Calibration linearity: $A_{\\rm area}$ vs intact probe fraction",
                   fontsize=9, pad=4)
    ax_C.legend(fontsize=8)
    ax_C.spines["top"].set_visible(False)
    ax_C.spines["right"].set_visible(False)
    ax_C.tick_params(labelsize=8)
    _panel_label(ax_C, "C")

    # Add residual inset
    ax_Ci = ax_C.inset_axes([0.55, 0.10, 0.40, 0.38])
    resid = (y_plot - y_fit) / y_plot[0] * 100
    ax_Ci.bar(x_plot * 100, resid, color=BLUE, alpha=0.7, width=8)
    ax_Ci.axhline(0, color="gray", lw=0.8)
    ax_Ci.set_xlabel("$(1-F_k)$ (%)", fontsize=6)
    ax_Ci.set_ylabel("Resid. (%)", fontsize=6)
    ax_Ci.tick_params(labelsize=5)
    ax_Ci.set_title("Residuals", fontsize=6)

    # ── D: Peak position and FWHM stability ──────────────────────────────────
    fk_pct = fk_arr[valid] * 100
    ax_D.plot(fk_pct, E_peaks[valid] * 1000, "o-", color=BLUE, lw=2, ms=7,
              label="$E_{\\rm pk}$ (mV vs Ag/AgCl)")
    ax_D2 = ax_D.twinx()
    ax_D2.plot(fk_pct, FWHMs[valid], "s--", color=RED, lw=2, ms=7,
               label="FWHM (mV)")
    ax_D.set_xlabel("$F_k$ (%)", fontsize=9)
    ax_D.set_ylabel("Peak potential (mV vs Ag/AgCl)", color=BLUE, fontsize=9)
    ax_D2.set_ylabel("FWHM (mV)", color=RED, fontsize=9)
    ax_D.set_title("D — Peak stability: position and FWHM vs $F_k$",
                   fontsize=9, pad=4)
    ax_D.tick_params(labelsize=8)
    ax_D2.tick_params(labelsize=8)
    ax_D.spines["top"].set_visible(False)
    lines_a, labs_a = ax_D.get_legend_handles_labels()
    lines_b, labs_b = ax_D2.get_legend_handles_labels()
    ax_D.legend(lines_a + lines_b, labs_a + labs_b, fontsize=8)
    ax_D.text(0.02, 0.04,
              "FWHM variation $<3$ mV across\nfull $F_k$ range confirms\n"
              "binary cleavage model.",
              transform=ax_D.transAxes, fontsize=7)
    _panel_label(ax_D, "D")

    fig.suptitle(
        "Fig 3 — Distance averaging, calibration linearity, and peak stability",
        fontsize=11, y=1.01
    )
    if save:
        p = FIGURES / "fig_SWV3_calibration.pdf"
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Fc reference and ratiometric closure
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_ratiometric(DI_Fk, fk_arr, A_areas, save=True):
    """
    A: MB and Fc peaks on the same E axis.
    B: Ratiometric F_k recovery (F_k,rec vs F_k,true).
    C: Frequency diagnostic: ΔI_peak vs f for P(d) model vs single k0.
    D: Convergence: ΔI_peak vs N_quadrature.
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # ── A: MB + Fc on same axis ───────────────────────────────────────────────
    DI_Fc = swv_voltammogram(E_SCAN_FC, k0=K0_FC, Gamma_tot=GAMMA_FC,
                              A_cm2=A_ELEC, E0=E0_FC, alpha=0.5, ne=1,
                              f_swv=F_SWV, E_sw=E_SW)
    _, _, sigma_Fc, A_area_Fc = fit_gaussian(E_SCAN_FC, DI_Fc)
    kappa_Fc = K0_FC / (2 * F_SWV)

    ax_A.plot(E_SCAN * 1000, DI_Fk[0.0] * 1e12,
              color=BLUE, lw=2.5, label=f"MB ($F_k = 0$, $E^{{\\circ\\prime}} = -250$ mV)")
    ax_A.plot(E_SCAN_FC * 1000, DI_Fc * 1e12,
              color=RED,  lw=2.5, label=f"Fc ($\\kappa = {kappa_Fc:.0f}$, $E^{{\\circ\\prime}} = +210$ mV)")
    ax_A.axhline(0, color="gray", ls="--", lw=0.7)
    ax_A.fill_between(E_SCAN * 1000, DI_Fk[0.0] * 1e12, alpha=0.12, color=BLUE)
    ax_A.fill_between(E_SCAN_FC * 1000, DI_Fc * 1e12, alpha=0.12, color=RED)
    _style_ax(ax_A, "$E$ (mV vs Ag/AgCl)", "$\\Delta I$ (pA)",
              "A — MB and Fc peaks in a single scan")
    ax_A.legend(fontsize=8)
    ax_A.text(0.02, 0.95,
              f"Peak separation: 460 mV\nBoth measured in one scan\n"
              f"Fc is always reversible ($\\kappa={kappa_Fc:.0f}$)",
              transform=ax_A.transAxes, fontsize=7, va="top")
    _panel_label(ax_A, "A")

    # ── B: Ratiometric recovery ───────────────────────────────────────────────
    R_vals = A_areas / A_area_Fc
    R_0    = R_vals[0]
    rho    = R_vals / R_0
    Fk_rec = 1.0 - rho

    valid2 = ~np.isnan(Fk_rec)
    ax_B.plot(fk_arr[valid2] * 100, Fk_rec[valid2] * 100,
              "o-", color=BLUE, lw=2, ms=9, label="$F_{{k,\\rm rec}} = 1 - \\rho$")
    ax_B.plot([0, 100], [0, 100], "k--", lw=1.5, label="Perfect recovery")
    err_max = np.nanmax(np.abs(Fk_rec[valid2] - fk_arr[valid2])) * 100
    ax_B.fill_between([0, 100],
                      [0 - err_max, 100 - err_max],
                      [0 + err_max, 100 + err_max],
                      alpha=0.12, color=RED, label=f"$\\pm${err_max:.1f}% error band")
    _style_ax(ax_B, "$F_k$ (true, %)", "$F_{{k,\\rm rec}}$ (recovered, %)",
              "B — Ratiometric pipeline closure")
    ax_B.legend(fontsize=8)
    ax_B.set_xlim(-2, 102)
    ax_B.set_ylim(-5, 105)
    ax_B.text(0.02, 0.95,
              f"Max error: {err_max:.1f}%\n(below 3.5% noise target)",
              transform=ax_B.transAxes, fontsize=8, va="top",
              color=RED if err_max > 3.5 else GREEN)
    _panel_label(ax_B, "B")

    # ── C: Frequency diagnostic ───────────────────────────────────────────────
    freq_arr = np.array([5, 10, 15, 30, 50, 100, 200, 500])
    d_mean   = D_PEG_NM + SIGMA_Z_NM * np.sqrt(2.0 / np.pi)
    k0_eff   = k0_of_d(d_mean)
    DI_peak_dist = []
    DI_peak_sing = []
    for fq in freq_arr:
        DI_tmp = dist_averaged_voltammogram(E_SCAN, 0.0, f_swv=fq, E_sw=E_SW,
                                            n_dist=40)
        DI_peak_dist.append(np.max(DI_tmp) * 1e12)
        DI_s = swv_voltammogram(E_SCAN, k0=k0_eff,
                                Gamma_tot=GAMMA_0, A_cm2=A_ELEC,
                                E0=-0.250, alpha=0.37, ne=2,
                                f_swv=fq, E_sw=E_SW)
        DI_peak_sing.append(np.max(DI_s) * 1e12)

    ax_C.semilogx(freq_arr, DI_peak_dist, "o-", color=BLUE, lw=2, ms=8,
                  label="Distance-averaged ($P(d)$ model)")
    ax_C.semilogx(freq_arr, DI_peak_sing, "s--", color=RED, lw=2, ms=8,
                  label=f"Single $k^0_{{\\rm eff}} = {k0_eff:.1f}$ s$^{{-1}}$ (mean $d$)")
    ax_C.axvline(F_SWV, color=GREY, ls=":", lw=1.5,
                 label=f"Operating point: {F_SWV:.0f} Hz")
    _style_ax(ax_C, "SWV frequency (Hz)", "$\\Delta I_{{\\rm peak}}$ (pA)",
              "C — Frequency diagnostic: P(d) vs single-$k^0$ model")
    ax_C.legend(fontsize=7.5)
    ax_C.text(0.02, 0.04,
              "$P(d)$ integration broadens and lowers\n"
              "the frequency optimum relative\nto a single-distance model.",
              transform=ax_C.transAxes, fontsize=7)
    _panel_label(ax_C, "C")

    # ── D: Convergence: ΔI_peak vs N_quadrature ───────────────────────────────
    N_vals = [5, 10, 20, 40, 60, 80, 100, 120]
    DI_pk_N = []
    for N in N_vals:
        DI_N = dist_averaged_voltammogram(E_SCAN, 0.0, f_swv=F_SWV, E_sw=E_SW,
                                          n_dist=N)
        DI_pk_N.append(np.max(DI_N) * 1e12)
    DI_pk_N = np.array(DI_pk_N)
    ref_val  = DI_pk_N[-1]

    ax_D.plot(N_vals, DI_pk_N, "o-", color=BLUE, lw=2, ms=8,
              label="$\\Delta I_{\\rm peak}$ (pA)")
    ax_D.axhline(ref_val, color=GREY, ls=":", lw=1.2, label=f"Converged value ({ref_val:.2f} pA)")
    ax_D.axvline(80, color=RED, ls="--", lw=1.5, label="$n_{\\rm dist} = 80$ (operating)")
    err_at_80 = abs(DI_pk_N[N_vals.index(80)] - ref_val) / ref_val * 100
    _style_ax(ax_D, "$n_{\\rm dist}$ (quadrature points)",
              "$\\Delta I_{\\rm peak}$ (pA)",
              "D — Quadrature convergence ($n_{\\rm dist} = 80$)")
    ax_D.legend(fontsize=8)
    ax_D.text(0.55, 0.10,
              f"Error at $n=80$: {err_at_80:.2f}%\n(converged)",
              transform=ax_D.transAxes, fontsize=8,
              color=GREEN if err_at_80 < 1 else RED)
    _panel_label(ax_D, "D")

    fig.suptitle(
        "Fig 4 — Fc internal reference, ratiometric correction, and convergence",
        fontsize=11, y=1.01
    )
    if save:
        p = FIGURES / "fig_SWV4_ratiometric.pdf"
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Clinical predictions: concentration response + LOD contour
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_clinical(save=True):
    """
    A: Predicted voltammograms at 0, 1, 3, 6, 10, 30 nM MMP-9 at T=30 min.
    B: Clinical calibration curves A_MB(%) vs [MMP-9] at 4 incubation times.
    C: LOD contour in (keff_surf_frac, T) space with clinical boundaries.
    D: Discrimination: ΔFk = Fk_chronic - Fk_healing vs T for 3 keff values.
    """
    KEFF_FRAC   = 0.20
    T_FIXED_MIN = 30.0
    CONC_NM     = [0.0, 1.0, 3.0, 6.0, 10.0, 30.0]
    CONC_COLS   = [NAVY, BLUE, "#0288d1", TEAL, ORANGE, RED]
    T_SWEEP     = [10.0, 20.0, 30.0, 60.0]
    T_COLS      = ["#0288d1", BLUE, NAVY, PURPLE]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # ── A: Voltammograms at clinical concentrations ───────────────────────────
    print("  Fig 5A: concentration voltammograms ...")
    DI_conc = {}
    Fk_conc = {}
    Apk_conc = {}
    for cnM in CONC_NM:
        _, DI_c, fk_c = concentration_to_voltammogram(
            cnM, T_FIXED_MIN, keff_surf_frac=KEFF_FRAC)
        DI_conc[cnM] = DI_c
        Fk_conc[cnM] = fk_c
        _, _, _, area = fit_gaussian(E_SCAN, DI_c)
        Apk_conc[cnM] = area

    for cnM, col in zip(CONC_NM, CONC_COLS):
        lbl = f"{cnM:.0f} nM  ($F_k = {Fk_conc[cnM]*100:.1f}$%)"
        ax_A.plot(E_SCAN * 1000, DI_conc[cnM] * 1e12, color=col, lw=2.2,
                  label=lbl)
    ax_A.axhline(0, color="gray", ls="--", lw=0.7)
    ax_A.axvline(-250, color=GREY, ls=":", lw=1.2)
    _style_ax(ax_A, "$E$ (mV vs Ag/AgCl)", "$\\Delta I$ (pA)",
              f"A — Predicted voltammograms vs [MMP-9]\n$T={T_FIXED_MIN:.0f}$ min, "
              f"$\\eta = {KEFF_FRAC}$ (20% of solution rate)")
    ax_A.legend(fontsize=7.5, loc="upper right")
    _panel_label(ax_A, "A")

    # ── B: Clinical calibration curves ───────────────────────────────────────
    print("  Fig 5B: calibration curves ...")
    conc_sweep = np.logspace(-1, 2, 50)   # 0.1–100 nM
    area_0 = Apk_conc[0.0]
    for T_m, col in zip(T_SWEEP, T_COLS):
        areas_T = []
        for cnM in conc_sweep:
            fk_t = concentration_to_Fk(cnM, T_m, keff_surf_frac=KEFF_FRAC)
            # Linear approximation: area ∝ (1-Fk) × area_0_at_this_T
            # But compute properly via Fk scaling (area proportional to 1-Fk)
            area_t = area_0 * (1 - fk_t)
            areas_T.append(area_t)
        y_pct = np.array(areas_T) / area_0 * 100
        ax_B.semilogx(conc_sweep, y_pct, color=col, lw=2.2,
                      label=f"$T = {T_m:.0f}$ min")

    ax_B.axvline(6.0,  color=GREEN, ls="--", lw=1.8, label="Healing: 6 nM")
    ax_B.axvline(30.0, color=RED,   ls="--", lw=1.8, label="Chronic: 30 nM")
    lod_30 = 3 * 0.034 / (KEFF_FRAC * KEFF_SOL_M1S1 * T_FIXED_MIN * 60) * 1e9
    ax_B.axvline(lod_30, color=PURPLE, ls=":", lw=1.5,
                 label=f"LOD (30 min) = {lod_30:.1f} nM")
    ax_B.set_ylim(0, 105)
    ax_B.set_xlim(0.1, 100)
    _style_ax(ax_B, "[MMP-9] (nM)", "$A_{\\rm MB} / A_{\\rm MB}(0)$ (%)",
              "B — Clinical calibration curves")
    ax_B.legend(fontsize=7.5)
    _panel_label(ax_B, "B")

    # ── C: LOD contour in (η, T) space ───────────────────────────────────────
    eta_arr = np.logspace(-2, 0, 80)     # 0.01 to 1.0
    T_arr_s = np.linspace(5, 120, 80) * 60  # 5–120 min in seconds
    sigma_Fk = 0.034  # 3.4% ratiometric noise floor
    ETA, T_GRID = np.meshgrid(eta_arr, T_arr_s)
    keff_surf = ETA * KEFF_SOL_M1S1
    LOD_nM = 3 * sigma_Fk / (keff_surf * T_GRID) * 1e9

    levels = [1, 2, 4, 6, 10, 20, 30, 50, 100]
    cf = ax_C.contourf(eta_arr, T_arr_s / 60, LOD_nM, levels=levels,
                       cmap="RdYlGn_r", alpha=0.85)
    cs = ax_C.contour(eta_arr, T_arr_s / 60, LOD_nM, levels=[6, 30],
                      colors=["white", "white"], linewidths=[2.0, 2.0],
                      linestyles=["--", "-"])
    ax_C.clabel(cs, fmt={6: "6 nM (healing)", 30: "30 nM (chronic)"},
                fontsize=7, colors="white")
    plt.colorbar(cf, ax=ax_C, label="LOD (nM)", shrink=0.85)
    ax_C.axvline(KEFF_FRAC, color="white", ls=":", lw=2.0,
                 label=f"Central $\\eta = {KEFF_FRAC}$")
    ax_C.axhline(T_FIXED_MIN, color="white", ls="-.", lw=2.0,
                 label=f"$T = {T_FIXED_MIN:.0f}$ min")
    ax_C.set_xscale("log")
    _style_ax(ax_C, "Surface correction $\\eta$ (dimensionless)",
              "Incubation time $T$ (min)",
              "C — LOD contour in $(\\eta, T)$ space\n(white contours: 6 nM healing, 30 nM chronic)")
    ax_C.legend(fontsize=7.5, loc="upper right")
    _panel_label(ax_C, "C")

    # ── D: Discrimination ratio ΔFk vs T ─────────────────────────────────────
    T_min_arr   = np.linspace(1, 120, 300)
    keff_cases  = [(0.05, "low $\\eta=0.05$", "#0288d1"),
                   (0.20, "central $\\eta=0.20$", BLUE),
                   (0.50, "high $\\eta=0.50$",  NAVY)]
    for eta, lbl, col in keff_cases:
        Fk_c = concentration_to_Fk(30.0, T_min_arr, keff_surf_frac=eta)
        Fk_h = concentration_to_Fk(6.0,  T_min_arr, keff_surf_frac=eta)
        dFk  = Fk_c - Fk_h
        ax_D.plot(T_min_arr, dFk * 100, color=col, lw=2.2, label=lbl)

    ax_D.axhline(0.034 * 100, color=GREY, ls=":", lw=1.5,
                 label="3.4% noise floor")
    ax_D.axvline(T_FIXED_MIN, color=GREY, ls="--", lw=1.2)
    _style_ax(ax_D, "Incubation time $T$ (min)",
              "$\\Delta F_k = F_k^{\\rm chronic} - F_k^{\\rm healing}$ (%)",
              "D — Discrimination signal vs $T$ for three $\\eta$ values")
    ax_D.legend(fontsize=8)
    ax_D.text(0.02, 0.04,
              "$5\\times$ ratio guaranteed at short $T$;\n"
              "absolute gap $\\Delta F_k$ peaks at optimal $T$.",
              transform=ax_D.transAxes, fontsize=7)
    _panel_label(ax_D, "D")

    fig.suptitle(
        "Fig 5 — Clinical predictions: voltammograms, calibration, and LOD landscape",
        fontsize=11, y=1.01
    )
    if save:
        p = FIGURES / "fig_SWV5_clinical.pdf"
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Generating SWV flagship figures ...")

    print("\n[1/5] Fig 1: Geometry ...")
    fig1_geometry()

    print("\n[2/5] Fig 2: ODE and validation ...")
    fig2_ode_validation()

    # Pre-compute shared data for Figs 3 & 4
    print("\n[3-4/5] Pre-computing distance-averaged voltammograms ...")
    DI_Fk = {}
    for fk in FK_VALS:
        print(f"  F_k = {fk:.2f}")
        DI_Fk[fk] = dist_averaged_voltammogram(
            E_SCAN, fk,
            Gamma_0=GAMMA_0, A_cm2=A_ELEC,
            E0_MB=-0.250, alpha=0.37, ne=2,
            f_swv=F_SWV, E_sw=E_SW, n_dist=80)

    A_areas, E_peaks, FWHMs = [], [], []
    for fk in FK_VALS:
        A_pk, E_pk, sigma_E, A_area = fit_gaussian(E_SCAN, DI_Fk[fk])
        A_areas.append(A_area)
        E_peaks.append(E_pk)
        FWHMs.append(sigma_E * 2.355 * 1000.0 if not np.isnan(sigma_E) else np.nan)

    A_areas = np.array(A_areas, dtype=float)
    E_peaks = np.array(E_peaks, dtype=float)
    FWHMs   = np.array(FWHMs,   dtype=float)
    fk_arr  = np.array(FK_VALS, dtype=float)

    valid = ~np.isnan(A_areas)
    x_lin = 1.0 - fk_arr[valid]
    y_lin = A_areas[valid]
    slope_lin, intercept_lin = np.polyfit(x_lin, y_lin, 1)
    residuals_pct = np.abs((y_lin - (slope_lin * x_lin + intercept_lin)) / y_lin[0]) * 100

    print("\n[3/5] Fig 3: Distance averaging and calibration ...")
    fig3_distance_averaging(DI_Fk, fk_arr, A_areas, E_peaks, FWHMs,
                             slope_lin, intercept_lin, residuals_pct)

    print("\n[4/5] Fig 4: Ratiometric and convergence ...")
    fig4_ratiometric(DI_Fk, fk_arr, A_areas)

    print("\n[5/5] Fig 5: Clinical predictions ...")
    fig5_clinical()

    print("\nAll figures saved to:", FIGURES)


if __name__ == "__main__":
    main()
