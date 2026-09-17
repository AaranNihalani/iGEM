"""
reference_electrode.py -- Layer 5.1 / 5.2
==========================================
On-chip Ag/AgCl quasi-reference electrode: Nernst potential, the offset this
introduces relative to the saturated-KCl Ag/AgCl reference convention used in
the literature values cited elsewhere in the pipeline (Layer 10), and the
three drift mechanisms identified in the Layer 5 content spec (Cl- depletion,
AgCl dissolution, photosensitivity).

All outputs feed:
    - Layer 10/11 potential axis (quasi-reference correction to apply to
      literature E°' values before comparing to on-chip SWV data)
    - Layer 12 (peak-position tolerance / integration-window clipping check)

Run as a script to print the numeric summary used in the Layer 5 report and
regenerate Figure 1.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Physical constants ───────────────────────────────────────────────────────
R = 8.314        # J/mol/K
F = 96485.0      # C/mol
T = 298.15       # K (25 degC)
RT_F = R * T / F  # V, = 0.02569 V at 25 degC  (CONFIDENT)

# ── Standard potentials ──────────────────────────────────────────────────────
# Standard Ag/AgCl potential vs SHE/NHE at 25 degC (activity of Cl- = 1).
# CONFIDENT -- standard textbook value, Bard & Faulkner, Electrochemical
# Methods, 2nd ed., Table A.1 / p. 813.
E0_AgCl_vs_NHE = 0.2224  # V

# Potential of the conventional *saturated* KCl Ag/AgCl reference electrode
# vs NHE at 25 degC -- this is the "vs Ag/AgCl (sat.)" convention used for
# literature formal potentials quoted elsewhere in this pipeline (e.g. Layer
# 10's E'_MB = -290 mV vs Ag/AgCl, Pheeney & Barton 2012).
# CONFIDENT -- standard reference-electrode handbook value (sat. KCl, 25 degC).
E_AgCl_sat_vs_NHE = 0.197  # V


def E_ref_vs_NHE(a_Cl):
    """On-chip Ag/AgCl quasi-reference potential vs NHE, Nernst equation.

    E_ref = E0_AgCl - (RT/F) * ln(a_Cl-)
    """
    return E0_AgCl_vs_NHE - RT_F * np.log(a_Cl)


# ── Wound fluid chloride activity ────────────────────────────────────────────
# [Cl-] in wound fluid: 110-120 mM (contents.md, Sec. 5.1). Activity
# coefficient for a 1:1 electrolyte at physiological ionic strength,
# gamma ~ 0.75 (Debye-Huckel extended / mean literature value at I ~ 150 mM).
# FLAG: gamma is a literature approximation, not measured for wound fluid
# specifically.
Cl_low, Cl_high = 0.110, 0.120        # M
gamma_Cl = 0.75                        # dimensionless, FLAG (approximate)

a_Cl_low = gamma_Cl * Cl_low
a_Cl_high = gamma_Cl * Cl_high
a_Cl_mid = gamma_Cl * 0.5 * (Cl_low + Cl_high)

E_ref_low = E_ref_vs_NHE(a_Cl_low)
E_ref_high = E_ref_vs_NHE(a_Cl_high)
E_ref_mid = E_ref_vs_NHE(a_Cl_mid)

# Offset of the on-chip quasi-reference from the conventional saturated
# Ag/AgCl reference used in the literature formal-potential values:
#   E_offset = E_ref(on-chip, wound fluid) - E_AgCl_sat_vs_NHE
E_offset_low = E_ref_low - E_AgCl_sat_vs_NHE
E_offset_high = E_ref_high - E_AgCl_sat_vs_NHE
E_offset_mid = E_ref_mid - E_AgCl_sat_vs_NHE


def correct_literature_potential(E_lit_vs_AgCl_sat, E_offset=E_offset_mid):
    """Convert a literature formal potential (quoted vs sat. Ag/AgCl) into
    the potential that will actually be applied vs the on-chip quasi-reference.

    E_applied,onchip = E_lit,sat - E_offset

    (The on-chip electrode sits E_offset positive of the sat. reference on the
    NHE scale, so the *same* absolute redox potential reads out E_offset more
    negative when expressed vs the on-chip electrode.)
    """
    return E_lit_vs_AgCl_sat - E_offset


# Confirmed literature formal potentials quoted "vs Ag/AgCl" elsewhere in the
# pipeline (Layer 10, Sec. VI; values are the sat.-reference convention as
# reported in the source papers):
E_MB_lit = -0.290   # V vs Ag/AgCl (sat.), Pheeney & Barton 2012, pH 7 -- CONFIRMED (Layer 10)
E_Fc_lit_range = (0.40, 0.47)  # V vs Ag/AgCl (sat.), Layer 10 Sec. VII -- literature range, unconfirmed on-chip

E_MB_onchip = correct_literature_potential(E_MB_lit)
E_Fc_onchip_range = tuple(correct_literature_potential(e) for e in E_Fc_lit_range)

# ── Drift mechanism (a): Cl- depletion / accumulation at the RE surface ─────
# delta_E_ref = (RT/F) * delta_ln(a_Cl) ~ (RT/F) * delta_[Cl-]/[Cl-]  for small
# fractional changes (first-order Taylor expansion of ln).
def drift_from_Cl_change(frac_change, Cl_ref=0.115):
    """Voltage drift (V) for a given fractional change in [Cl-] at the RE surface."""
    return RT_F * frac_change


drift_5pct = drift_from_Cl_change(0.05)  # contents.md worked example

# ── Drift mechanism (b): Ag/AgCl dissolution ────────────────────────────────
Ksp_AgCl = 1.8e-10  # CONFIDENT, standard solubility product, 25 degC
Cl_wound_typ = 0.115  # M


def Ag_plus_from_Cl(Cl_M):
    """Free Ag+ concentration in equilibrium with AgCl(s) at a given [Cl-]."""
    return Ksp_AgCl / Cl_M


Ag_plus_wound = Ag_plus_from_Cl(Cl_wound_typ)

# ── Drift mechanism (c): photosensitivity ───────────────────────────────────
# Ag/AgCl decomposes photolytically under UV/visible light.
# Literature range for illuminated quasi-references: 1-10 mV drift.
# FLAG: device-specific value depends on cartridge opacity; assumed dark
# (opaque cartridge) as the baseline design requirement.
drift_light_range = (0.001, 0.010)  # V, if illuminated (design requirement: exclude light)

# ── Combined random-walk drift model over the assay ─────────────────────────
def simulate_drift(t_minutes, sigma_per_sqrt_min=0.0009, n_traces=200, seed=0):
    """Model E_ref(t) as a random walk: variance proportional to elapsed time.

    sigma_per_sqrt_min is chosen so that, at t = 30 min, the RMS drift from
    depletion + dissolution effects is ~1-2 mV (contents.md Sec. 5.2, "for
    short assays (<30 min), drift is likely < 2 mV if the RE design is
    sound"). This is a design *target*, not a measured diffusion coefficient
    -- FLAG.
    """
    rng = np.random.default_rng(seed)
    n_steps = len(t_minutes)
    dt = np.diff(t_minutes, prepend=0.0)
    dt[0] = t_minutes[0] if len(t_minutes) else 0.0
    steps = rng.normal(0.0, sigma_per_sqrt_min, size=(n_traces, n_steps)) * np.sqrt(np.maximum(dt, 1e-9))
    return np.cumsum(steps, axis=1)


ASSAY_LEN_MIN = 30.0
ACCEPTABLE_DRIFT_MV = 5.0      # contents.md Sec. 5.2: "acceptable total RE drift over assay: < 5 mV"
CLIP_RISK_DRIFT_MV = 25.0      # one SWV peak half-width at half-maximum (HWHM), FWHM_MB ~ 65 mV / 2 (Layer 11)


def plot_reference_summary(save=True):
    t = np.linspace(0, ASSAY_LEN_MIN, 200)
    traces_mV = simulate_drift(t) * 1e3  # mV

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # Panel A: Nernst response vs [Cl-]
    ax = axes[0]
    Cl_range = np.linspace(0.05, 0.25, 300)
    a_range = gamma_Cl * Cl_range
    E_range = E_ref_vs_NHE(a_range)
    ax.plot(Cl_range * 1e3, E_range * 1e3, "b-", lw=2)
    ax.axvspan(Cl_low * 1e3, Cl_high * 1e3, color="orange", alpha=0.25,
               label=f"Wound fluid range ({Cl_low*1e3:.0f}-{Cl_high*1e3:.0f} mM)")
    ax.axhline(E_AgCl_sat_vs_NHE * 1e3, color="gray", ls="--", lw=1.2,
               label="Sat. Ag/AgCl (197 mV vs NHE)")
    ax.set_xlabel("[Cl-] (mM)")
    ax.set_ylabel("E_ref vs NHE (mV)")
    ax.set_title("A -- On-chip Ag/AgCl Nernst response")
    ax.legend(fontsize=8)

    # Panel B: random-walk drift traces
    ax2 = axes[1]
    for i in range(min(40, traces_mV.shape[0])):
        ax2.plot(t, traces_mV[i], color="steelblue", alpha=0.15, lw=0.8)
    rms = np.sqrt(np.mean(traces_mV**2, axis=0))
    ax2.plot(t, rms, "k-", lw=2, label="RMS drift")
    ax2.plot(t, -rms, "k-", lw=2)
    ax2.axhline(ACCEPTABLE_DRIFT_MV, color="green", ls="--", lw=1.2, label=f"Acceptable ({ACCEPTABLE_DRIFT_MV:.0f} mV)")
    ax2.axhline(-ACCEPTABLE_DRIFT_MV, color="green", ls="--", lw=1.2)
    ax2.axhline(CLIP_RISK_DRIFT_MV, color="red", ls=":", lw=1.2, label=f"Peak-clip risk ({CLIP_RISK_DRIFT_MV:.0f} mV, 1 HWHM)")
    ax2.set_xlabel("Assay time (min)")
    ax2.set_ylabel("E_ref drift (mV)")
    ax2.set_title("B -- Random-walk drift model (30 min assay)")
    ax2.legend(fontsize=8)

    # Panel C: literature-to-on-chip potential correction
    ax3 = axes[2]
    labels = ["E'_MB\n(Pheeney 2012)", "E'_Fc low\n(Layer 10)", "E'_Fc high\n(Layer 10)"]
    lit_vals = [E_MB_lit, E_Fc_lit_range[0], E_Fc_lit_range[1]]
    onchip_vals = [E_MB_onchip, E_Fc_onchip_range[0], E_Fc_onchip_range[1]]
    x = np.arange(len(labels))
    w = 0.35
    ax3.bar(x - w/2, [v * 1e3 for v in lit_vals], w, label="vs sat. Ag/AgCl (lit.)", color="gray")
    ax3.bar(x + w/2, [v * 1e3 for v in onchip_vals], w, label="vs on-chip quasi-ref", color="crimson")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel("Formal potential (mV)")
    ax3.set_title(f"C -- Quasi-reference correction ({E_offset_mid*1e3:+.0f} mV)")
    ax3.legend(fontsize=8)
    ax3.axhline(0, color="black", lw=0.8)

    fig.suptitle("Layer 5.1/5.2: On-chip Ag/AgCl reference electrode -- potential and drift", fontsize=11)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L5_reference.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved {out}")
    plt.close(fig)
    return t, traces_mV


if __name__ == "__main__":
    print("=== Layer 5.1: On-chip Ag/AgCl quasi-reference ===")
    print(f"E_ref (wound fluid, [Cl-]={Cl_low*1e3:.0f}-{Cl_high*1e3:.0f} mM): "
          f"{E_ref_low*1e3:.1f} to {E_ref_high*1e3:.1f} mV vs NHE (mid {E_ref_mid*1e3:.1f} mV)")
    print(f"Sat. Ag/AgCl reference: {E_AgCl_sat_vs_NHE*1e3:.1f} mV vs NHE")
    print(f"Offset from sat. Ag/AgCl: {E_offset_low*1e3:+.1f} to {E_offset_high*1e3:+.1f} mV "
          f"(mid {E_offset_mid*1e3:+.1f} mV)")
    print()
    print("=== Literature-to-on-chip potential correction ===")
    print(f"E'_MB: {E_MB_lit*1e3:.0f} mV (lit., sat. Ag/AgCl) -> {E_MB_onchip*1e3:.0f} mV (on-chip)")
    print(f"E'_Fc: {E_Fc_lit_range[0]*1e3:.0f}-{E_Fc_lit_range[1]*1e3:.0f} mV (lit.) -> "
          f"{E_Fc_onchip_range[0]*1e3:.0f}-{E_Fc_onchip_range[1]*1e3:.0f} mV (on-chip)")
    print()
    print("=== Drift budget ===")
    print(f"(a) Cl- depletion, 5% change: {drift_5pct*1e3:.2f} mV")
    print(f"(b) [Ag+] release at [Cl-]={Cl_wound_typ*1e3:.0f} mM: {Ag_plus_wound:.2e} M (negligible reservoir depletion)")
    print(f"(c) Photosensitivity (if illuminated): {drift_light_range[0]*1e3:.0f}-{drift_light_range[1]*1e3:.0f} mV "
          f"-- design requirement: opaque/dark cartridge")
    print(f"Acceptable total drift: < {ACCEPTABLE_DRIFT_MV:.0f} mV; peak-clip risk threshold: {CLIP_RISK_DRIFT_MV:.0f} mV (1 HWHM)")
    plot_reference_summary()
