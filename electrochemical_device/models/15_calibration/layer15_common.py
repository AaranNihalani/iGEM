"""
layer15_common.py  —  Layer 15 shared constants and helpers
=============================================================
Shared numbers and small utility functions used by the Layer 15 modules
(calibration_curve.py, deconvolution.py, matrix_correction.py,
wound_classifier.py), so the modules stay numerically consistent with
each other and with the rest of the device model chain (Layers 0-14).

Updated against the actual Layer 9 (surface_kinetics) and Layer 13
(ratiometric) reports:

  - The device is a SINGLE MMP-9 channel: active electrode carries
    SGKIPRTLTA, negative control is the D-enantiomer of the same
    peptide (stereospecific control), not a second HNE-specific probe.
    HNE appears only as an unconfirmed *interferent* on the MMP-9
    channel (Layer 9 Sec X), not as its own measurable channel.
  - kcat/Km for MMP-9 on SGKIPRTLTA is CONFIRMED (solution-phase,
    Kridel 2001): 67,500 M^-1 s^-1. This is an *upper bound* on the
    surface rate constant k_surf_eff; surface immobilisation is
    expected to reduce it 2-100x (Layer 9 Sec V, XIII).
  - The correct progress-curve model is the implicit integrated MM
    equation (Layer 9 Eq IV.5/IV.6), not a pure exponential — the
    probe-saturation ratio rho = Gamma0/KGm_surf is estimated at
    0.3-3.1 (Layer 9 Table 1), well outside the rho << 1 regime where
    the monoexponential would be valid.
  - The Layer 13 noise model gives sigma_Fk analytically from the
    ratiometric error chain, rather than a flat blank-noise placeholder.

Anything still not backed by a wet-lab measurement is flagged
`# TBD (wet lab)`.
"""

import numpy as np

# ── Molecular weights ─────────────────────────────────────────────────────
MW_MMP9 = 92e3   # g/mol, active MMP-9 (README Layer 0)
MW_HNE  = 30e3   # g/mol, active HNE  (README Layer 3)

# ── Assay timing (Layers 9, 13) ────────────────────────────────────────────
T_INCUBATION = 30 * 60   # s, standard incubation time

# Pre-analytical dilution (Layer 1, not yet supplied as a report). Calibration
# cartridges are spiked directly into matrix at known CHAMBER concentrations,
# so D_f/eta_elut do NOT enter the calibration model itself (Layer 9's [E]_t
# is used directly as the chamber/electrode-local concentration). They are
# kept here only for later clinical (wound -> chamber) conversion.
D_F      = 0.8     # TBD (Layer 1)
ETA_ELUT = 0.7     # TBD (Layer 1)

# ── Confirmed wound-state free MMP-9 (Layer 0 ODE, restated in Layer 9) ────
WOUND_MMP9_CHRONIC_NM  = 30.0   # nM, free MMP-9 at T=30 min, chronic wound
WOUND_MMP9_HEALING_NM  = 6.0    # nM, free MMP-9 at T=30 min, healing wound

# ── MMP-9 surface kinetics on SGKIPRTLTA (Layer 9) ─────────────────────────
KCAT_KM_MMP9_SOLUTION_UB = 67_500.0          # M^-1 s^-1, CONFIRMED (Kridel 2001, Table II)
KCAT_KM_MMP9_SURFACE_RANGE  = (2_000.0, 67_500.0)   # plausible surface range, Layer 9 Sec XIII.1
KCAT_KM_MMP9_SURFACE_TARGET = (13_000.0, 27_000.0)  # range consistent with a 30-60 min read (Layer 9 Table 2)
# Synthetic "true" surface rate constant used ONLY to generate demo
# calibration data until the fabricated electrode is actually measured.
KCAT_KM_MMP9_SURFACE_DEMO = 20_000.0         # M^-1 s^-1   # TBD (Layer 15 electrode calibration)

# Surface Michaelis constant and probe density (Layer 9 Sec IV, Table 1).
# Default row: fsteric = 1.0 (no Layer 8 steric correction applied) ->
# rho = Gamma0 / KGm_surf ~= 3.1 (Layer 9's own worked example).
KGM_SURF = 6.4e-13    # mol/cm^2  # TBD (Layer 8 delta_eff unconfirmed, Eq IV.2-IV.3)
GAMMA0   = 2.0e-12    # mol/cm^2  # TBD (Pheeney 2012 midpoint; device-planned value ~5e-12)
RHO = GAMMA0 / KGM_SURF   # probe-saturation ratio (dimensionless)

# ── HNE cross-reactivity on the MMP-9 (SGKIPRTLTA) channel (Layer 9 Sec X) ─
# No paper reports k_eff for HNE on SGKIPRTLTA. Placeholder assumes HNE is
# ~3% as efficient as MMP-9 on the same probe; used only for an illustrative
# interference-bias check, not a real second channel.
HNE_ON_MMP9_PROBE_FRACTION = 0.03            # TBD (wet lab cross-reactivity panel)
KCAT_KM_HNE_ON_MMP9_PROBE = HNE_ON_MMP9_PROBE_FRACTION * KCAT_KM_MMP9_SURFACE_DEMO   # TBD

# Legacy alias for deconvolution.py, which still assumes a (now believed
# non-existent) second HNE-specific probe/channel. Kept only so that
# script doesn't crash on import; deconvolution.py itself has NOT been
# updated to reflect the real single-channel architecture yet — see the
# note in layer15_common.py's module docstring.
KCAT_KM_MMP9_ON_HNE_PROBE = KCAT_KM_HNE_ON_MMP9_PROBE   # DEPRECATED, placeholder only

# Legacy aliases for matrix_correction.py (still uses the old flat-noise,
# monoexponential-slope framing). NOT updated to the Layer 9/13 model yet.
SIGMA_F_BLANK = sigma_Fk_placeholder = 0.068    # DEPRECATED: sigma_Fk(0), single electrode (see sigma_Fk() below)
KCAT_KM_MMP9 = KCAT_KM_MMP9_SURFACE_DEMO         # DEPRECATED alias, old name

# ── Ratiometric noise model (Layer 13, Sec I-III) ──────────────────────────
SIGMA_RHO_OVER_RHO_SINGLE = 0.048   # per active-control electrode pair, single scan (Eq II.6)
N_ACTIVE_ELECTRODES_DEFAULT = 1     # averaging reduces sigma_rho/rho by 1/sqrt(N)

# Layer 13's own detection/quantification thresholds on Fk itself (Sec IV):
F_MIN = 0.20   # minimum reliably detectable cleavage (~3 sigma_Fk at Fk=0, single electrode)
F_MAX = 0.95   # beyond this, MB signal is indistinguishable from noise / inversion diverges

# ── Calibration experiment design (Sec 15.1, contents.md) ─────────────────
MATRIX_COMPOSITION = {
    "buffer": "PBS",
    "BSA_mg_per_mL": 4,
    "fibrinogen_uM": 50,
    "Ca2+_mM": 1,
    "IgG_mg_per_mL": 0.1,
}
PH_LEVELS = [6.5, 7.0, 7.4, 8.0]
CAL_CONC_NM = np.array([0, 0.1, 0.3, 1, 3, 10, 30, 100])   # nM, per Sec 15.1
N_REPLICATES = 3


def sigma_Fk(Fk, sigma_rho_over_rho=SIGMA_RHO_OVER_RHO_SINGLE, n_active=N_ACTIVE_ELECTRODES_DEFAULT):
    """
    Analytic Fk uncertainty from the Layer 13 ratiometric error chain
    (Eq III.6): sigma_Fk = (1 - Fk) * sqrt(2) * (sigma_rho/rho), where
    sigma_rho/rho itself shrinks by 1/sqrt(n_active) when averaging over
    n_active paired active/control electrodes.
    """
    Fk = np.asarray(Fk, dtype=float)
    return (1 - Fk) * np.sqrt(2) * (sigma_rho_over_rho / np.sqrt(n_active))


def slope_from_kcat_km(kcat_km, T=T_INCUBATION, D_f=D_F, eta=ETA_ELUT):
    """
    Legacy short-time linear-regime slope (M^-1), kept for modules that
    still use the simple monoexponential approximation (deconvolution.py,
    matrix_correction.py). NOT used by the updated calibration_curve.py,
    which uses the full implicit integrated-MM model instead.
    """
    return kcat_km * D_f * eta * T


def pH_correction(pH, pH_ref=7.4, sensitivity=0.15):
    """
    Empirical pH correction factor applied to the calibration slope
    (Sec 15.6): peaks at pH_ref and falls off roughly linearly away
    from it (enzyme kcat pH-dependence + SWV peak-fitting error).
    Placeholder pending a measured pH-activity profile.
    """
    return np.clip(1 - sensitivity * np.abs(np.asarray(pH) - pH_ref), 0.3, 1.0)


def nM_to_M(c_nM):
    return np.asarray(c_nM) * 1e-9


def M_to_nM(c_M):
    return np.asarray(c_M) * 1e9
