"""
preanalytical_model.py
Layer 1 of 17 -- Eton iGEM 2026 Dry Lab, electrochemical wound-protease device.

Deterministic pre-analytical model: converts an in-situ wound-fluid enzyme
concentration [E]_wound (Layer 0 output) into the effective enzyme
concentration seen by the electrode, [E]_chamber (Layer 3 input), via the
swab -> elution -> transit -> cartridge-loading chain.

IMPORTANT CORRECTION vs. contents.md Sec. 1.2
----------------------------------------------
contents.md Eq. (informal) states:

    [E]_eluate = [E]_wound * eta_elut

This is dimensionally fine (a concentration times a dimensionless
efficiency), but it silently assumes the eluate volume equals the
*absorbed* volume V_abs, i.e. that none of the analyte is diluted on
mixing with the elution buffer. In the actual protocol (Sec. 1.2/1.4 of
contents.md) the swab -- having absorbed V_abs of wound fluid -- is placed
into V_buffer (typically 200 uL PBS) and the analyte is released into
that buffer volume. Mass balance requires:

    moles released  = eta_elut * [E]_wound * V_abs
    [E]_eluate       = moles released / V_eluate_total
                     ~ eta_elut * [E]_wound * V_abs / V_buffer      (V_buffer >> retained volume)

i.e. there is an *additional* dilution factor V_abs / V_buffer (typically
0.1-1.0 for the literature V_abs range of 20-200 uL into a 200 uL buffer)
that Sec. 1.2's formula omits. This module implements the corrected form
and is used throughout the Layer 1 report (Sec. I-II).
"""

from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------
# I. Swab absorption
# ----------------------------------------------------------------------

def v_abs_lognormal_params(low=20.0, high=200.0, ci=0.90):
    """
    Parameterise V_abs (uL) as log-normal from a literature range
    [low, high] interpreted as the central `ci` interval (default: the
    90% interval, i.e. the 5th-95th percentile band quoted in
    contents.md Sec. 1.1, "V_abs ~ 20-200 uL per swab pass on a moderate
    exudate wound").

    Returns (mu, sigma) of the underlying normal on ln(V_abs).
    """
    from scipy.stats import norm
    z = norm.ppf(0.5 + ci / 2.0)  # e.g. 1.645 for the 90% interval
    mu = 0.5 * (np.log(low) + np.log(high))
    sigma = (np.log(high) - np.log(low)) / (2 * z)
    return mu, sigma


# ----------------------------------------------------------------------
# II. Elution efficiency (size-dependent, Sec. 1.2) and corrected eluate
#     concentration (mass-balance fix, see module docstring)
# ----------------------------------------------------------------------

# Literature bands from contents.md Sec. 1.2/1.3, by nominal molecular
# weight class. These are UNCONFIRMED placeholders (no wet-lab or
# swab-specific source has been located for the fabricated device;
# see Report Sec. II and Limitations).
ETA_ELUT_BANDS = {
    "small": (0.7, 0.9),     # < 10 kDa
    "medium": (0.4, 0.7),    # 30-100 kDa (MMP-9, HNE, TIMP-1, A1AT sit here or nearby)
    "large": (0.2, 0.5),     # large proteins / complexes
}

# Molecular weights (kDa), standard biochemistry values (not wound-fluid
# specific, no citation needed beyond the protein's own literature, e.g.
# Kridel 2001 for MMP-9's catalytic domain / Trengove 1999 for HNE).
MW_KDA = {
    "MMP9": 92.0,
    "HNE": 30.0,
    "TIMP1": 28.0,
    "A1AT": 52.0,
    "SLPI": 11.7,
}

# Point-estimate elution efficiencies used for the central-value
# calculations in the report (midpoints of the contents.md bands, except
# where contents.md Sec. 1.3 gives inhibitor-specific point values).
ETA_ELUT_POINT = {
    "MMP9": 0.55,   # medium-band midpoint (92 kDa; upper end of "medium")
    "HNE": 0.55,    # medium-band midpoint (30 kDa)
    "TIMP1": 0.6,   # contents.md Sec. 1.3
    "A1AT": 0.5,    # contents.md Sec. 1.3
    "SLPI": 0.8,    # contents.md Sec. 1.3
}


def eluate_concentration(E_wound, eta_elut, V_abs_uL, V_buffer_uL=200.0):
    """
    Corrected eluate concentration (see module docstring):

        [E]_eluate = [E]_wound * eta_elut * (V_abs / V_buffer)

    E_wound   : wound-fluid concentration (any consistent unit, e.g. nM)
    eta_elut  : dimensionless elution efficiency, 0-1
    V_abs_uL  : swab-absorbed volume (uL)
    V_buffer_uL : elution buffer volume (uL); default 200 uL (contents.md 1.2)
    """
    return E_wound * eta_elut * (V_abs_uL / V_buffer_uL)


# ----------------------------------------------------------------------
# V. Analyte stability during transit (Sec. 1.5 first instance)
# ----------------------------------------------------------------------

def transit_decay_factor(k_deg_per_min, t_transit_min):
    """First-order survival fraction after transit: exp(-k_deg * t)."""
    return np.exp(-k_deg_per_min * t_transit_min)


# ----------------------------------------------------------------------
# VII. Net chamber concentration (contents.md Sec. 1.6, corrected)
# ----------------------------------------------------------------------

def chamber_concentration(E_wound, eta_elut, V_abs_uL, k_deg_per_min,
                           t_transit_min, D_f, V_buffer_uL=200.0):
    """
    Full pre-analytical chain, wound -> chamber:

        [E]_chamber = [E]_wound * eta_elut * (V_abs/V_buffer)
                      * exp(-k_deg * t_transit) * D_f

    D_f = V_eluate_loaded / V_chamber is the cartridge-loading factor
    (contents.md Sec. 1.6; worked example D_f = 30/37.5 = 0.8).
    """
    eluate = eluate_concentration(E_wound, eta_elut, V_abs_uL, V_buffer_uL)
    decayed = eluate * transit_decay_factor(k_deg_per_min, t_transit_min)
    return decayed * D_f


def composite_recovery_factor(eta_elut, V_abs_uL, k_deg_per_min,
                               t_transit_min, D_f, V_buffer_uL=200.0):
    """R_total = [E]_chamber / [E]_wound, the net pre-analytical recovery."""
    return chamber_concentration(1.0, eta_elut, V_abs_uL, k_deg_per_min,
                                  t_transit_min, D_f, V_buffer_uL)


# ----------------------------------------------------------------------
# III. Inhibitor co-elution and enzyme/inhibitor ratio shift (Sec. 1.3)
# ----------------------------------------------------------------------

def inhibitor_chamber_concentration(I_wound, eta_elut_I, V_abs_uL,
                                     D_f, V_buffer_uL=200.0,
                                     k_deg_per_min=0.0, t_transit_min=0.0):
    """Same chain as chamber_concentration, applied to an inhibitor."""
    return chamber_concentration(I_wound, eta_elut_I, V_abs_uL,
                                  k_deg_per_min, t_transit_min, D_f,
                                  V_buffer_uL)


def ratio_enrichment_factor(eta_elut_inhibitor, eta_elut_enzyme):
    """
    Enrichment of an inhibitor relative to the target enzyme in the
    eluate/chamber, purely from differential elution efficiency (the
    V_abs/V_buffer, decay, and D_f factors are common to both species and
    cancel in the ratio):

        enrichment = eta_elut,I / eta_elut,E

    enrichment > 1 means the inhibitor is proportionally *more*
    concentrated in the chamber than the wound-fluid ratio would predict
    (contents.md Sec. 1.3: e.g. SLPI vs HNE).
    """
    return eta_elut_inhibitor / eta_elut_enzyme


# ----------------------------------------------------------------------
# IV. Elution buffer effect on matrix: pH mixing (Sec. 1.4)
# ----------------------------------------------------------------------

def ph_chamber(pH_wound, pH_buffer, V_wound_uL, V_buffer_uL):
    """
    Exact proton mass-balance pH of the mixed eluate (contents.md Sec.
    1.4), rather than assuming complete buffering to pH_buffer:

        pH = -log10( (V_w*10^-pHw + V_b*10^-pHb) / (V_w + V_b) )
    """
    h_wound = 10.0 ** (-pH_wound)
    h_buffer = 10.0 ** (-pH_buffer)
    h_mix = (V_wound_uL * h_wound + V_buffer_uL * h_buffer) / (V_wound_uL + V_buffer_uL)
    return -np.log10(h_mix)


# ----------------------------------------------------------------------
# VI. Particulate filtration and inlet pressure drop (Sec. 1.6 filter)
# ----------------------------------------------------------------------

def filter_pressure_drop(eta_Pa_s, L_m, Q_m3_s, r_pore_m, n_pores):
    """
    Hagen-Poiseuille pressure drop across a parallel-pore membrane filter
    (contents.md Sec. 1.6):

        dP = 8 * eta * L * Q / (pi * r_pore^4 * N_pores)

    Returns dP in Pa.
    """
    return 8.0 * eta_Pa_s * L_m * Q_m3_s / (np.pi * r_pore_m ** 4 * n_pores)


# ----------------------------------------------------------------------
# Worked example (reproduces the central-value numbers quoted in the
# Layer 1 report)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # --- Layer 0 confirmed wound-fluid concentrations, t = 30 min read ---
    E_wound_chronic = 31.9   # nM, Layer 0 ODE (Ladwig 2002 parametrisation)
    E_wound_healing = 6.2    # nM, Layer 0 ODE

    # --- Central-value pre-analytical parameters ---
    eta_elut_MMP9 = ETA_ELUT_POINT["MMP9"]
    V_abs_central = np.exp(v_abs_lognormal_params()[0])   # geometric mean, uL
    k_deg_central = 0.0055   # /min, midpoint of 0.001-0.01 /min (Sec. 1.5)
    t_transit_clinic = 15.0  # min
    t_transit_field = 120.0  # min
    D_f = 0.8                # contents.md Sec. 1.6 worked example
    V_buffer = 200.0         # uL

    print(f"V_abs geometric-mean central estimate: {V_abs_central:.1f} uL")

    R_total_clinic = composite_recovery_factor(
        eta_elut_MMP9, V_abs_central, k_deg_central, t_transit_clinic, D_f, V_buffer)
    R_total_field = composite_recovery_factor(
        eta_elut_MMP9, V_abs_central, k_deg_central, t_transit_field, D_f, V_buffer)

    print(f"Composite recovery factor R_total (15-min transit): {R_total_clinic:.4f}"
          f"  ({R_total_clinic*100:.1f}%)")
    print(f"Composite recovery factor R_total (120-min transit): {R_total_field:.4f}"
          f"  ({R_total_field*100:.1f}%)")

    for label, E_w in [("chronic", E_wound_chronic), ("healing", E_wound_healing)]:
        E_c_clinic = chamber_concentration(E_w, eta_elut_MMP9, V_abs_central,
                                            k_deg_central, t_transit_clinic, D_f, V_buffer)
        E_c_field = chamber_concentration(E_w, eta_elut_MMP9, V_abs_central,
                                           k_deg_central, t_transit_field, D_f, V_buffer)
        print(f"[MMP-9]_wound {label} = {E_w:.1f} nM -> "
              f"[MMP-9]_chamber = {E_c_clinic:.3f} nM (15-min transit), "
              f"{E_c_field:.3f} nM (120-min transit)")

    # --- Inhibitor ratio shift, SLPI vs HNE (contents.md Sec. 1.3) ---
    enrich_SLPI_HNE = ratio_enrichment_factor(ETA_ELUT_POINT["SLPI"], ETA_ELUT_POINT["HNE"])
    enrich_TIMP1_MMP9 = ratio_enrichment_factor(ETA_ELUT_POINT["TIMP1"], ETA_ELUT_POINT["MMP9"])
    enrich_A1AT_HNE = ratio_enrichment_factor(ETA_ELUT_POINT["A1AT"], ETA_ELUT_POINT["HNE"])
    print(f"SLPI/HNE enrichment in chamber vs. wound-fluid ratio: {enrich_SLPI_HNE:.2f}x")
    print(f"TIMP-1/MMP-9 enrichment in chamber vs. wound-fluid ratio: {enrich_TIMP1_MMP9:.2f}x")
    print(f"A1AT/HNE enrichment in chamber vs. wound-fluid ratio: {enrich_A1AT_HNE:.2f}x")

    # --- pH mixing (contents.md Sec. 1.4 worked example) ---
    pH_mix = ph_chamber(pH_wound=6.5, pH_buffer=7.4, V_wound_uL=50.0, V_buffer_uL=200.0)
    print(f"pH_chamber (pH_wound=6.5, 50 uL wound + 200 uL PBS pH 7.4): {pH_mix:.2f}")

    # --- Filter pressure drop, representative worked example ---
    # Assumed inputs (flagged as illustrative pending Layer 2 fill-time data):
    eta_fluid = 1.5e-3       # Pa.s, wound fluid ~1.5x water viscosity
    L_filter = 20e-6         # m, membrane thickness (20 um)
    Q_fill = 5e-9            # m^3/s, ~5 uL/s representative fill rate
    r_pore = 0.75e-6         # m, mid-range pore radius (0.45-2 um diam range)
    n_pores = 2.0e6          # pores across a ~5x5 mm filter patch at this density
    dP = filter_pressure_drop(eta_fluid, L_filter, Q_fill, r_pore, n_pores)
    print(f"Illustrative filter pressure drop: {dP:.2f} Pa ({dP/1000:.4f} kPa)")
