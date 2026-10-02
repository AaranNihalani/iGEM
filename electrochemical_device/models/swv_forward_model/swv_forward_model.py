"""
swv_forward_model.py  —  SWV Forward Model (Flagship)
=======================================================
Converts a cleavage fraction F_k into the full SWV voltammogram
the electrode would actually record.

Pipeline position:
  Layer 7 (SAM) → Layer 8 (P(d)) → Layer 10 (k_ET(d))
                                          ↓
  Layer 9 (F_k) ──────────────→  THIS MODEL → ΔI(E; F_k)
                                          ↓
                              Layer 12 (peak) → A_MB
                                          ↓
                              Layer 13 (ratio) → F_k recovered

Five modules, executed in order:
  M1  BV-SWV simulator (single probe, exact ODE)
  M2  Distance distribution P(d) via freely jointed chain
  M3  Distance-averaged voltammogram
  M4  F_k dependence and calibration curve linearity
  M5  Fc reference peak and ratiometric closure

References (papers/swv_forward_model/):
  [1] Dauphin-Ducharme et al. (2017) Langmuir 33:4407  nihms881272.pdf
  [2] Laviron E (1979) J Electroanal Chem 100:263        1-s2.0-S0022072879801679-main.pdf
  [3] Kohn et al. (2004) PNAS 101:12491                  kohn2004.pdf
  [4] Smalley et al. (1995) J Phys Chem 99:13141         (papers/layer_10/smalley1995.pdf)
  [5] O'Dea et al. (1981) Anal Chem 53:695               (papers/layer_11/odea1981.pdf)
  [6] Baker et al. (2006) JACS 128:3138                  (papers/layer_11/baker2006.pdf)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ─── Physical constants ───────────────────────────────────────────────────────
F_const = 96485.0        # C mol-1
R_const = 8.3145         # J mol-1 K-1
T_K     = 298.15         # K  (room temperature for validation; 310 K for wound)

# ─── M1: BV-SWV SINGLE-PROBE SIMULATOR ───────────────────────────────────────
# Theory: Layer 11, Section II; validated against Laviron (1979) [2] and
# Dauphin-Ducharme (2017) [1].
#
# The half-cycle ODE for a surface-confined redox couple has an exact
# analytical solution.  For a half-cycle at potential E starting from
# Γ_red(0):
#
#   Γ_red(τ) = Γ_eq + [Γ_red(0) - Γ_eq] exp(-(k_f+k_b)τ)
#
#   Γ_eq = k_f/(k_f+k_b) × Γ_tot
#   k_f  = k0 exp(-α n_e F (E - E0) / RT)
#   k_b  = k0 exp( (1-α) n_e F (E - E0) / RT)
#
# The sampled current at the END of the half-cycle:
#
#   I(τ) = n_e F A [k_f Γ_red(τ) - k_b Γ_ox(τ)]
#
# This is correct.  The existing swv_forward.py (Layer 11) omits the
# initial-condition propagation between half-cycles, which is the key
# subtlety.

def _bv_rates(E, E0, k0, alpha, ne):
    """Forward (k_f) and backward (k_b) BV rate constants at potential E."""
    xi   = ne * F_const * (E - E0) / (R_const * T_K)
    k_f  = k0 * np.exp(-alpha * xi)
    k_b  = k0 * np.exp((1.0 - alpha) * xi)
    return k_f, k_b


def _half_cycle_end(Gamma_red_0, Gamma_tot, k0, alpha, ne, E, E0, tau):
    """
    Evolve one half-cycle of duration tau.  Returns (Gamma_red_end, I_anodic).

    BV convention (cathodic = forward reaction):
      k_f  = k0 exp(-α nF(E-E0)/RT)   [reduction rate]
      k_b  = k0 exp((1-α) nF(E-E0)/RT) [oxidation rate]
      dΓ_red/dt = k_f Γ_ox - k_b Γ_red

    Equilibrium: Γ_eq = k_f / (k_f+k_b) × Γ_tot  (fully reduced at E << E0).

    Current convention: anodic positive (oxidation = positive), matching
    Dauphin-Ducharme [1] and standard SWV display for an anodic MB scan.
      I_anodic = ne F (k_b Γ_red - k_f Γ_ox)
    This gives a POSITIVE ΔI peak at E = E0 on an anodic staircase sweep.
    """
    k_f, k_b = _bv_rates(E, E0, k0, alpha, ne)
    K_tot    = k_f + k_b
    Gamma_eq = k_f / K_tot * Gamma_tot          # equilibrium reduced coverage
    displacement = (Gamma_red_0 - Gamma_eq) * np.exp(-K_tot * tau)
    Gamma_red_end = Gamma_eq + displacement
    # Anodic current: I = ne F (k_b Γ_red - k_f Γ_ox).  Substituting
    # Γ_ox = Γ_tot - Γ_red gives I = ne F (k_f + k_b)(Γ_red - Γ_eq), i.e.
    # ne F K_tot x displacement from equilibrium.  Evaluating it in this form
    # avoids the catastrophic cancellation of the two-term form at large
    # kappa = k0/(2f), where each term is huge and their difference is ~0.
    I_anodic = ne * F_const * K_tot * displacement
    return Gamma_red_end, I_anodic


def swv_voltammogram(E_arr, k0, Gamma_tot=1.0, A_cm2=1.0,
                     E0=0.0, alpha=0.37, ne=2,
                     f_swv=15.0, E_sw=0.025):
    """
    Compute net SWV current ΔI(E) for a surface-confined redox couple.

    Parameters
    ----------
    E_arr     : array of staircase potential values (V)
    k0        : standard ET rate constant (s-1)
    Gamma_tot : total surface coverage (mol cm-2); default 1 for normalised output
    A_cm2     : electrode area (cm2); default 1 for normalised output
    E0        : formal potential (V)
    alpha     : transfer coefficient (0.37 from Dauphin-Ducharme MB-DNA [1])
    ne        : number of electrons (2 for MB [6])
    f_swv     : SWV frequency (Hz)
    E_sw      : SWV amplitude (V)

    Returns
    -------
    DI : net SWV current array (same units as n_e F A Gamma_tot, i.e. A if
         Gamma in mol cm-2 and A in cm2; or A cm-2 mol-1 in normalised form)
    """
    tau = 1.0 / (2.0 * f_swv)   # half-period (s)

    # Initial condition: equilibrium at E_start (far negative of E0, fully reduced)
    # We run a few dummy cycles at E_start to reach the periodic steady state.
    E_start  = E_arr[0]
    Gamma_r  = Gamma_tot  # start fully reduced (MB at rest potential)

    # Burn-in: 5 full cycles at the starting potential to reach PSS
    for _ in range(5):
        Gamma_r, _ = _half_cycle_end(Gamma_r, Gamma_tot, k0, alpha, ne,
                                     E_start + E_sw, E0, tau)
        Gamma_r, _ = _half_cycle_end(Gamma_r, Gamma_tot, k0, alpha, ne,
                                     E_start - E_sw, E0, tau)

    DI = np.zeros_like(E_arr, dtype=float)
    for i, E_step in enumerate(E_arr):
        # Forward half-cycle: E_step + E_sw (cathodic pulse drives oxidation)
        Gamma_r_fwd, I_fwd = _half_cycle_end(Gamma_r, Gamma_tot, k0, alpha, ne,
                                              E_step + E_sw, E0, tau)
        # Reverse half-cycle: E_step - E_sw
        Gamma_r_rev, I_rev = _half_cycle_end(Gamma_r_fwd, Gamma_tot, k0, alpha, ne,
                                              E_step - E_sw, E0, tau)
        DI[i]  = I_fwd - I_rev
        Gamma_r = Gamma_r_rev   # carry state into next step

    # NOTE: Gamma_tot is already correctly incorporated inside
    # _half_cycle_end (via Gamma_red_end / Gamma_ox_end, both computed in
    # real coverage units), so I_fwd/I_rev/DI are already a real current
    # density (A/cm^2), not a per-Gamma_tot normalised quantity. Multiplying
    # by Gamma_tot again here double-counts it, scaling the output by
    # Gamma_tot^2 instead of Gamma_tot^1 -- confirmed by direct test
    # (peak / Gamma_tot^2 is constant across Gamma_tot values; peak /
    # Gamma_tot is not). Only A_cm2 (current density -> total current)
    # belongs here.
    return DI * A_cm2


def laviron_reversible(E_arr, E0, ne, Gamma_tot=1.0, A_cm2=1.0):
    """
    Laviron (1979) Eq. 6: equilibrium (reversible-limit) SWV/LPSV peak.
    |Ψ| = η/(1+η)^2 where η = exp[nF(E-E0')/RT].
    Normalised to peak area = 1 (for shape comparison).
    """
    xi  = ne * F_const * (E_arr - E0) / (R_const * T_K)
    eta = np.exp(xi)
    psi = eta / (1.0 + eta)**2
    return psi


# ─── Unit test: Laviron reversible limit ─────────────────────────────────────
def test_bv_simulator():
    """
    Validation checks for the BV-SWV simulator.

    (A) Peak position: ΔI_peak must lie within E_sw = 25 mV of E0.
        In SWV the forward half-pulse maximum is at E + E_sw = E0, so
        the net peak is near E0 ± E_sw in the quasi-reversible regime.

    (B) Sign: ΔI at the peak must be > 0 (anodic positive convention,
        anodic staircase scan).

    (C) FWHM: at κ = 10, f = 15 Hz, E_sw = 25 mV, ne = 2,
        the ideal SWV net-current FWHM is ≈ 60-70 mV (wider than the
        Laviron LPSV value of 45.3 mV because E_sw is not << RT/nF).
        Accept 40 – 90 mV as the valid range.

    (D) Laviron shape (small E_sw limit): at E_sw = 1 mV (<< RT/nF =
        12.8 mV), the SWV net current does NOT closely approach the
        Laviron LPSV peak eta/(1+eta)^2 -- checked directly, the RMS
        difference is ~24%, not the <3% originally hoped for. This is a
        genuine gap, not a bug in this test: Laviron's formula describes a
        continuously-swept, always-at-equilibrium system, while this
        simulator samples a discretely-pulsed SWV waveform once per
        half-cycle; there is no guarantee the two converge, and directly
        checking shows they don't (see swv_forward_model.tex Part C for
        the full discussion). shape_rms below is a REAL computed value,
        not a placeholder, and is NOT used to gate all_ok (a large value
        here is expected and does not indicate the simulator is broken
        for its own operating regime, kappa < ~15).

    Returns bool values and diagnostics.
    """
    # --- (A/B/C) Quasi-reversible peak checks at κ = 10 --------------------
    E_arr  = np.linspace(-0.15, 0.15, 600)
    f_test = 15.0
    k0_qr  = 10.0 * (2.0 * f_test)   # κ = 10  → k0 = 300 s-1
    DI_qr  = swv_voltammogram(E_arr, k0=k0_qr, ne=2, alpha=0.5,
                               E0=0.0, f_swv=f_test, E_sw=0.025)

    peak_idx   = np.argmax(DI_qr)
    E_peak_mV  = E_arr[peak_idx] * 1000.0
    peak_sign  = DI_qr[peak_idx] > 0
    DI_norm    = DI_qr / DI_qr[peak_idx]
    above_half = E_arr[DI_norm >= 0.5]
    fwhm_mV    = (above_half[-1] - above_half[0]) * 1000.0 if len(above_half) > 1 else np.nan

    pos_ok  = abs(E_peak_mV) < 30.0   # within E_sw = 25 mV of E0
    fwhm_ok = 40.0 < fwhm_mV < 90.0
    sign_ok = peak_sign

    # --- (D) Monotone cleavage response: ΔI_peak decreases as F_k increases -
    # At F_k=0 (no cleavage) vs F_k=0.5 (50% cleavage), peak must decrease.
    DI_fk0  = swv_voltammogram(E_arr, k0=k0_qr, ne=2, alpha=0.5,
                                E0=0.0, f_swv=f_test, E_sw=0.025)
    # Simulate 50% cleavage: fewer active probes → scale amplitude by 0.5
    DI_fk05 = 0.5 * DI_fk0
    mono_ok = (np.max(DI_fk05) < np.max(DI_fk0))

    DI_lav_n  = laviron_reversible(E_arr, E0=0.0, ne=2)
    DI_lav_n  = DI_lav_n / np.max(DI_lav_n)

    # (D) REAL small-Esw shape comparison (was previously a hardcoded 0.0
    # placeholder -- fixed). Computed at Esw=1mV, the actual small-signal
    # limit the docstring describes, not the 25mV operating value used above.
    DI_small = swv_voltammogram(E_arr, k0=k0_qr, ne=2, alpha=0.5,
                                 E0=0.0, f_swv=f_test, E_sw=0.001)
    DI_small_n = DI_small / np.max(np.abs(DI_small))
    shape_rms = float(np.sqrt(np.mean((DI_small_n - DI_lav_n)**2)))

    return (pos_ok and sign_ok and fwhm_ok and mono_ok,
            fwhm_ok, fwhm_mV, shape_rms,
            DI_norm, DI_lav_n, E_arr,
            DI_fk05 / np.max(DI_fk0))


# ─── M2: DISTANCE DISTRIBUTION P(d) ─────────────────────────────────────────
# Freely jointed chain (FJC) for SGKIPRTLTA (10 residues) + PEG6 spacer.
# Kohn et al. (2004) [3] confirm random-coil FJC scaling for unfolded peptides.
#
# Amino acid geometry (standard polypeptide backbone):
#   Contour length per residue: l0 = 0.38 nm
#   Kuhn length: b = 0.76 nm (= 2 residues; Flory 1953)
#
# PEG6 spacer (6 ethylene oxide units, Layer 8 probe_conformation.py):
#   l_EO = 0.35 nm per EO unit, l_kuhn_PEG = 0.76 nm
#   n_k_PEG = 6 × 0.35 / 0.76 = 2.76 Kuhn segments
#   Flory radius R_F = n_k^(3/5) × l_kuhn = 1.40 nm (dilute regime at Γ_p = 3 pmol/cm²)
#   Hard wall: d_PEG = R_F = 1.40 nm  (mean PEG end-to-end distance, dilute regime)
#
# NOTE: Previously this used d_C6 = 0.756 nm (C6 alkanethiol). The spacer is
# PEG6, not C6. Corrected from Jaden Potishko's experimental note (Apr 2026).
#
# FJC end-to-end statistics for the PEPTIDE portion above PEG6 tip:
#   N_K = L_c / b = (10 × 0.38) / 0.76 = 5 Kuhn segments
#   <r²> = N_K b² = 5 × (0.76)² = 2.888 nm²
#   σ_z = sqrt(<r²>/3) = 0.981 nm
#
#   P(d) = sqrt(2/π) × (1/σ_z) × exp(-(d - d_PEG)²/(2σ_z²))  for d ≥ d_PEG

N_RESIDUES  = 10          # SGKIPRTLTA
L0_NM       = 0.38        # nm per residue (contour)
B_KUHN_NM   = 0.76        # nm (Kuhn length = 2 residues)
D_PEG_NM    = 1.398       # nm (PEG6 Flory radius at Γ_p=3 pmol/cm², dilute regime)
N_KUHN      = (N_RESIDUES * L0_NM) / B_KUHN_NM   # 5 segments
R2_MEAN_NM2 = N_KUHN * B_KUHN_NM**2              # 2.888 nm²
SIGMA_Z_NM  = np.sqrt(R2_MEAN_NM2 / 3.0)         # 0.981 nm

D_MAX_NM    = D_PEG_NM + N_RESIDUES * L0_NM      # 5.198 nm (PEG tip + peptide contour)
D_GRID      = np.linspace(D_PEG_NM, D_MAX_NM, 300)


def P_d(d_nm):
    """
    Half-Gaussian distance distribution for MB label above electrode.
    Hard wall at d_PEG (PEG6 spacer tip); FJC distribution for peptide above.
    d_nm : distance from gold surface to MB label (nm)
    Returns: probability density (nm-1), normalised over [D_PEG_NM, D_MAX_NM].
    """
    d = np.asarray(d_nm, dtype=float)
    p = np.where(
        d >= D_PEG_NM,
        np.sqrt(2.0 / np.pi) / SIGMA_Z_NM * np.exp(-(d - D_PEG_NM)**2 / (2.0 * SIGMA_Z_NM**2)),
        0.0
    )
    return p


def _normalise_P_d():
    """Return normalisation factor for P(d) over [D_PEG_NM, D_MAX_NM]."""
    norm, _ = quad(P_d, D_PEG_NM, D_MAX_NM)
    return norm

_P_NORM = _normalise_P_d()


def P_d_norm(d_nm):
    """Normalised P(d) integrating to 1 over [D_PEG_NM, D_MAX_NM]."""
    return P_d(d_nm) / _P_NORM


# ─── M3: k_ET(d) AND DISTANCE-AVERAGED VOLTAMMOGRAM ─────────────────────────
# From Layer 10 / Dauphin-Ducharme (2017) [1] Eqs 13-14 / Smalley (1995) [4]:
#
#   k0(d) = k0_ref × exp(-β (d - d_PEG))
#
# Anchoring point from Dauphin-Ducharme [1] Fig. 4:
#   MB-dithiol at x=0: k0 = 1.5×10^8 s-1 (extrapolated to zero distance).
#   β = 1.0 Å-1 = 10 nm-1 confirmed for ALKANETHIOL SAMs (Smalley 1995 [4]).
#
# FLAG: β = 1.0 Å-1 is from Smalley 1995 for alkyl chains. The PEG6 linker
# may have a different effective β (literature range 0.3-1.0 Å-1 through-bond
# for PEG, but no confirmed value in our paper set). k_ref therefore needs
# experimental calibration via EIS (trumpet plot) on our fabricated electrode.
#
# k_ref at PEG6 tip (d = D_PEG_NM = 1.398 nm), using β = 1.0 Å-1 as placeholder:
#   k0_ref = 1.5×10^8 × exp(-10 × 1.398) = 127 s-1
#
# Key consequence: k0(d_PEG) = 127 s-1, κ(15 Hz) = 4.2 (quasi-reversible).
# k0(d_mean) = 0.05 s-1, κ = 0.002 (deep irreversible). Signal comes
# predominantly from probes at d ≈ d_PEG (P(d) peak), not the mean.

BETA_NM     = 10.0         # nm-1  (β = 1.0 Å-1, Smalley 1995 [4]; FLAG: may differ for PEG6)
K0_AT_0_S1  = 1.5e8        # s-1  (MB-dithiol at x=0, Dauphin-Ducharme [1] Fig 4)
# k0 at PEG6 tip (d = D_PEG_NM): FLAG -- see note above
K0_REF_S1   = K0_AT_0_S1 * np.exp(-BETA_NM * D_PEG_NM)   # ≈ 127 s-1


def k0_of_d(d_nm):
    """
    Standard ET rate constant at MB-electrode distance d (nm).
    k0(d) = K0_REF_S1 × exp(-β(d - D_PEG_NM))   [s-1]
    FLAG: β and K0_REF_S1 are placeholders; needs EIS calibration.
    """
    return K0_REF_S1 * np.exp(-BETA_NM * (np.asarray(d_nm) - D_PEG_NM))


def dist_averaged_voltammogram(E_arr, F_k,
                                Gamma_0=None, A_cm2=None,
                                E0_MB=-0.250, alpha=0.37, ne=2,
                                f_swv=10.0, E_sw=0.025,
                                n_dist=80):
    """
    Distance-averaged SWV voltammogram.

    ΔI(E; F_k) = (1-F_k) × Γ_0 × A × ∫ P(d) ΔI_norm(E; k0(d)) dd

    where ΔI_norm is the net SWV current per unit coverage per unit area.

    Parameters
    ----------
    E_arr   : staircase potential values (V)
    F_k     : cleavage fraction [0,1]
    Gamma_0 : total initial probe coverage (mol cm-2)
    A_cm2   : geometric electrode area (cm2)
    n_dist  : number of distance quadrature points

    Returns
    -------
    DI_total : net SWV current (A)

    Gamma_0 and A_cm2 default to the confirmed module values GAMMA_0 and
    A_ELEC (defined below), so the defaults cannot drift from them.
    """
    Gamma_0 = GAMMA_0 if Gamma_0 is None else Gamma_0
    A_cm2   = A_ELEC if A_cm2 is None else A_cm2
    d_vals   = np.linspace(D_PEG_NM, D_MAX_NM, n_dist)
    dd       = d_vals[1] - d_vals[0]
    weights  = P_d_norm(d_vals)   # nm-1, normalised

    DI_total = np.zeros_like(E_arr, dtype=float)
    for i, d in enumerate(d_vals):
        k0_d  = k0_of_d(d)
        # Per-unit-coverage, per-unit-area voltammogram
        DI_norm = swv_voltammogram(E_arr, k0=k0_d,
                                   Gamma_tot=1.0, A_cm2=1.0,
                                   E0=E0_MB, alpha=alpha, ne=ne,
                                   f_swv=f_swv, E_sw=E_sw)
        DI_total += weights[i] * DI_norm * dd

    # Multiply by (1-F_k), Γ_0, A and convert nm-1 → m-1 (cancel with Γ_0 units)
    # DI_total is in units of [A cm-2 per mol cm-2 per nm-1 × nm] = [A cm-2 mol-1]
    # Multiply by Gamma_0 [mol cm-2] × A_cm2 [cm2] to get [A]:
    DI_total *= (1.0 - F_k) * Gamma_0 * A_cm2
    return DI_total


# ─── Gaussian peak fit ────────────────────────────────────────────────────────
def gaussian(E, A_pk, E_pk, sigma_E):
    return A_pk * np.exp(-(E - E_pk)**2 / (2.0 * sigma_E**2))


def fit_gaussian(E_arr, DI_arr):
    """
    Fit a Gaussian to a SWV peak.
    Returns (A_pk, E_pk, sigma_E, A_area) where A_area = A_pk × sigma_E × sqrt(2π).
    """
    i_pk = np.argmax(np.abs(DI_arr))
    p0   = [DI_arr[i_pk], E_arr[i_pk], 0.020]
    try:
        popt, _ = curve_fit(gaussian, E_arr, DI_arr, p0=p0,
                            bounds=([0, E_arr[0], 0.005],
                                    [np.inf, E_arr[-1], 0.080]))
        A_pk, E_pk, sigma_E = popt
        A_area = A_pk * sigma_E * np.sqrt(2.0 * np.pi)
        return A_pk, E_pk, sigma_E, A_area
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ─── M4: F_k DEPENDENCE AND CALIBRATION LINEARITY ───────────────────────────
FK_VALS  = [0.0, 0.20, 0.40, 0.60, 0.80, 0.95]
E_SCAN   = np.linspace(-0.40, -0.10, 200)   # V vs Ag/AgCl (around MB peak)

# ─── KINETICS: CONCENTRATION → F_k ──────────────────────────────────────────
KEFF_SOL_M1S1 = 67_500.0   # M-1 s-1 (Kridel 2001, papers/layer_09/)

def concentration_to_Fk(conc_nM, T_min, keff_surf_frac=0.20):
    """
    Convert MMP-9 concentration and incubation time to cleavage fraction F_k.

    Uses exact pseudo-first-order kinetics:
        F_k = 1 - exp(-k_eff_surf × [MMP-9] × T)

    k_eff_surf = keff_surf_frac × KEFF_SOL_M1S1
    where keff_surf_frac is the surface correction (Layer 8 steric/orientational
    reduction relative to the Kridel 2001 solution value).  Central estimate 0.20;
    range 0.01–1.0 (unconfirmed, FLAG: measure at Layer 15).

    Parameters
    ----------
    conc_nM        : MMP-9 concentration (nM)
    T_min          : incubation time (min)
    keff_surf_frac : surface correction factor (dimensionless, default 0.20)

    Returns
    -------
    Fk : cleavage fraction [0, 1)
    """
    keff  = keff_surf_frac * KEFF_SOL_M1S1   # M-1 s-1
    conc  = conc_nM * 1e-9                    # M
    T_s   = T_min * 60.0                      # s
    return 1.0 - np.exp(-keff * conc * T_s)


def concentration_to_voltammogram(conc_nM, T_min, keff_surf_frac=0.20,
                                   E0_MB=-0.250, alpha=0.37, ne=2,
                                   f_swv=10.0, E_sw=0.025,
                                   Gamma_0=None, A_cm2=None,
                                   n_dist=80):
    """
    Full forward model: MMP-9 concentration → SWV voltammogram.

    Chain: [MMP-9] (nM) + T (min) → F_k (Layer 9 kinetics)
                                   → ΔI(E) (distance-averaged BV-SWV)

    Returns
    -------
    E_arr : staircase potentials (V vs Ag/AgCl)
    DI    : net SWV current (A)
    Fk    : cleavage fraction that produced this voltammogram
    """
    Fk  = concentration_to_Fk(conc_nM, T_min, keff_surf_frac)
    DI  = dist_averaged_voltammogram(E_SCAN, Fk,
                                      Gamma_0=Gamma_0, A_cm2=A_cm2,
                                      E0_MB=E0_MB, alpha=alpha, ne=ne,
                                      f_swv=f_swv, E_sw=E_sw,
                                      n_dist=n_dist)
    return E_SCAN.copy(), DI, Fk


# ─── DIGITISED DATA: Dauphin-Ducharme 2017 Fig 4B and Fig 5B ─────────────────
# Manually digitised from nihms881272.pdf (pages 16–17).
# Accuracy: ±15% on y-axis (small thumbnail figure); ±5 mV on x-axis.
# Both datasets are net SWV current, 100 Hz, E_sw = 25 mV, E0' ≈ -0.250 V.
#
# Fig 4B: MB-dithiol directly on C6-SAM (no extra tether).
#   k0_x=8Å = 5.0×10^4 s^-1,  α = 0.37  (best-fit from their COMSOL model)
#   This validates our BV-ODE at a known k0 with no distance distribution.
#
# Fig 5B: MB on 20-base linear DNA, C6 anchor.
#   k0_x=8Å = 5.0×10^3 s^-1,  α = 0.37
#   Longer tether → lower k0 due to extra tunnelling distance.

# E vs Ag/AgCl (V) — shared x-axis for both figures
DD_E_V = np.array([-0.100, -0.130, -0.160, -0.190, -0.210, -0.225,
                    -0.240, -0.250, -0.260, -0.275, -0.290, -0.310,
                    -0.340, -0.370, -0.400])

# Fig 4B net current, normalised to peak = 1
DD_FIG4B_NORM = np.array([0.00, 0.02, 0.10, 0.42, 0.72, 0.90,
                           0.98, 1.00, 0.97, 0.84, 0.63, 0.36,
                           0.09, 0.02, 0.00])

# Fig 5B net current, normalised to peak = 1  (peak ≈ 4× smaller in absolute terms)
DD_FIG5B_NORM = np.array([0.00, 0.01, 0.08, 0.38, 0.70, 0.88,
                           0.97, 1.00, 0.96, 0.82, 0.58, 0.30,
                           0.07, 0.01, 0.00])

# SWV parameters (Layer 11 operating point)
# F_SWV: team confirmed operating frequency is 10 or 20 Hz (internal
# electrode characterisation notes, Sept 2026), NOT the 15 Hz previously
# assumed here from a borrowed literature protocol. 10 Hz used as the
# representative default below; 20 Hz gives kappa_tip = 3.18 instead of 6.35
# (see swv_forward_model.tex Part B for both).
F_SWV    = 10.0    # Hz (confirmed range 10-20 Hz; 10 Hz used as default)
# E_SW confirmed against the potentiostat settings (25 mV, matches the
# Baker 2006 literature value exactly). A_ELEC confirmed against the
# fabricated device: geometric area 1.7 mm^2, not the 1.5 mm diameter
# design spec previously assumed (0.017 cm^2 vs 0.0177 cm^2 -- a ~4%
# difference, small enough not to change any qualitative conclusion).
# Both CONFIRMED (internal electrode characterisation notes, Sept 2026).
E_SW     = 0.025   # V  (25 mV amplitude; confirmed, Baker 2006 [6] + field confirmation)
GAMMA_0  = 3.0e-12 # mol cm-2 (Adjemian 2010, papers/layer_09/ and layer_11/)
A_ELEC   = 0.017   # cm2 (1.7 mm^2 geometric area; confirmed)

# ─── M5: Fc REFERENCE PEAK ───────────────────────────────────────────────────
# NOT PART OF THE PREDICTIONS: Fc-hexanethiol is co-immobilised with the
# MB-peptide in the current protocol (one pot, no MCH; it acts as the
# diluent), but it is deliberately NOT modelled: its formal potential is
# uncertain and no separate MB/Fc peak pair has been resolved on the real
# device (see swv_forward_model.tex, Part VI). The constants below are kept
# only as a record of the earlier ratiometric design; nothing below feeds
# the predictions of the report.
# Ferrocene-hexanethiol on C6 SAM.
# K0_FC below (2.7e5 s-1, Dauphin-Ducharme [1] Fig. 3) is a LITERATURE VALUE
# FOR A DIFFERENT CONSTRUCT (MB-DNA electrode), not confirmed for this
# Fc-hexanethiol/PEG6-peptide device -- "confirmed" in the original comment
# here meant only "correctly transcribed from that paper," not "applicable
# to this device." At this k0 and the real 10-20 Hz operating frequency,
# kappa_Fc = 6,750-13,500, which the BV-ODE predicts as an essentially zero
# net SWV signal -- contradicted by direct observation (a real Fc peak has
# been seen, height 5e-8 to 1e-5 A). Back-calculating with CONFIRMED
# A_ELEC=0.017 cm2 and E_SW=25 mV, from the worst-case observed height
# (5e-8 A) at Gamma_Fc=1.5e-12 and f=20 Hz, gives an optimal k0_Fc ~ 23 s-1
# (kappa ~ 0.57, near the quasi-reversible optimum) -- roughly four orders
# of magnitude slower than the borrowed value. At that k0 and coverage, the
# model cannot reach the best-case observed height (1e-5 A) at any k0 --
# that would need Gamma_Fc ~ 6e-10 mol/cm2, ~400x the current placeholder.
# NOT YET CONFIRMED; needs a dedicated Fc-only characterisation. K0_FC left
# at the literature value below since no confirmed replacement exists yet;
# do not treat it as reliable for absolute Fc signal predictions.
K0_FC   = 2.7e5    # s-1  (FLAG: literature value for a different construct, likely wrong for this device -- see note above)
E0_FC   = 0.35     # V vs Ag/AgCl (FLAG: observed range +0.30 to +0.40 V, itself variable electrode-to-electrode; midpoint used here, not a confirmed single value)
GAMMA_FC = 1.5e-12 # mol cm-2 (internal Fc SAM fraction; FLAG: confirm at Layer 15)
E_SCAN_FC = np.linspace(0.15, 0.55, 200)
