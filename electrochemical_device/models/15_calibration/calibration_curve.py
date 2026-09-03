"""
calibration_curve.py  —  Layer 15.2-15.3 (rewritten against Layers 9 & 13)
=============================================================================
Single-channel MMP-9 calibration: active electrode = SGKIPRTLTA,
negative control = D-enantiomer (stereospecific control, Layer 13 Sec III).
There is no separate HNE channel in the fabricated device — HNE only
enters as a possible interferent on this one channel (Layer 9 Sec X);
see `hne_interference_check()` at the bottom of this file.

Forward model — NOTE on a units fix applied here:
Layer 9's Eq IV.5, taken literally, is dimensionally inconsistent:
    k_cat[E]t*t = KGm_surf * ln(1/(1-Fk)) + Gamma0*Fk
The left side is dimensionless (k_surf_eff [M^-1 s^-1] x [E] [M] x t [s]),
but the right side carries units of mol/cm^2 (KGm_surf, Gamma0 ~1e-13 to
1e-12) — off by ~12 orders of magnitude. Eq IV.2 converts the solution
Km into areal units via a probe-layer thickness delta_eff, but that
conversion isn't threaded consistently into Eq IV.5/IV.8.

Dividing Eq IV.5 through by KGm_surf gives a dimensionally consistent
equation that exactly recovers Layer 9's own stated probe-dilute limit
(rho -> 0: Fk = 1 - exp(-k_surf_eff*[E]*T), Eq IV.7) and uses their own
dimensionless probe-saturation ratio rho = Gamma0/KGm_surf (Eq IV.4):

    k_surf_eff * [E] * T = ln(1/(1 - Fk)) + rho * Fk

This is the form implemented below. It reduces to the simple
monoexponential only when rho << 1; Layer 9's own estimate puts rho in
[0.3, 3.1] (Table 1) — not in that limit — so the implicit form is used
throughout.

Fk -> [E] is closed-form (units-corrected analogue of Layer 13 Eq IV.6):
    [E] = (ln(1/(1-Fk)) + rho*Fk) / (k_surf_eff * T)

[E] -> Fk requires solving the implicit equation numerically (bisection),
exactly as Layer 9 Sec IV specifies.

Noise model (Layer 13 Eq III.6):
    sigma_Fk(Fk) = (1 - Fk) * sqrt(2) * (sigma_rho/rho)
a Fk-DEPENDENT (heteroscedastic) uncertainty, not a flat blank-noise
constant. Layer 13 itself defines the detection/quantification window on
Fk directly: F_min = 0.20 (~3 sigma_Fk at Fk=0, single electrode),
F_max = 0.95 (inversion becomes unreliable beyond this).

Only ONE parameter is fit from calibration data: the surface rate
constant k_surf_eff (M^-1 s^-1). KGm_surf and Gamma0 are treated as
known inputs from Layer 9/Layer 8 (both still flagged unconfirmed
there); T is fixed by protocol.
"""

from dataclasses import dataclass, asdict
import csv
import numpy as np
from scipy.optimize import brentq, curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from layer15_common import (
    T_INCUBATION, KGM_SURF, GAMMA0, RHO,
    KCAT_KM_MMP9_SOLUTION_UB, KCAT_KM_MMP9_SURFACE_RANGE, KCAT_KM_MMP9_SURFACE_TARGET,
    KCAT_KM_MMP9_SURFACE_DEMO, KCAT_KM_HNE_ON_MMP9_PROBE,
    WOUND_MMP9_CHRONIC_NM, WOUND_MMP9_HEALING_NM,
    SIGMA_RHO_OVER_RHO_SINGLE, N_ACTIVE_ELECTRODES_DEFAULT, F_MIN, F_MAX,
    CAL_CONC_NM, N_REPLICATES, sigma_Fk, nM_to_M, M_to_nM,
)

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)


# ── Forward / inverse model (units-corrected Layer 9 Eq IV.5, see header) ──
def _implicit_residual(Fk, target, rho):
    return np.log(1.0 / (1.0 - Fk)) + rho * Fk - target


def Fk_from_target(target, rho=RHO):
    """
    Solve ln(1/(1-Fk)) + rho*Fk = target for a single dimensionless
    target (e.g. kobs_total * T, summing multiple contributions per
    Layer 9 Sec X Eq X.1). Used directly by deconvolution.py /
    matrix_correction.py so they don't have to re-derive the equation.
    """
    if target <= 0:
        return 0.0
    return brentq(_implicit_residual, 0.0, 1 - 1e-12, args=(target, rho))


def Fk_forward(E_M, T, k_surf_eff, rho=RHO):
    """
    Solve the (units-corrected) implicit integrated MM equation for Fk
    given [E]: k_surf_eff*[E]*T = ln(1/(1-Fk)) + rho*Fk.
    Vectorised over E_M via per-point bisection (brentq); each point is
    monotonic in Fk on [0, 1), so this is robust and fast.
    """
    E_M = np.atleast_1d(np.asarray(E_M, dtype=float))
    Fk_out = np.zeros_like(E_M)
    for i, E in enumerate(E_M):
        Fk_out[i] = Fk_from_target(k_surf_eff * E * T, rho)
    return Fk_out if Fk_out.size > 1 else Fk_out[0]


def E_inverse(Fk, T, k_surf_eff, rho=RHO):
    """Closed-form Fk -> [E] inversion (units-corrected analogue of Layer 13 Eq IV.6)."""
    Fk = np.asarray(Fk, dtype=float)
    return (np.log(1.0 / (1.0 - Fk)) + rho * Fk) / (k_surf_eff * T)


def apparent_rate(Fk, T):
    """kappa_k = -ln(1-Fk)/T (Layer 13 Eq IV.4), the preferred T-normalised readout."""
    return -np.log(1.0 - np.asarray(Fk, dtype=float)) / T


# ── Calibration result ──────────────────────────────────────────────────
@dataclass
class CalibrationResult:
    k_surf_eff_M: float        # fitted surface rate constant, M^-1 s^-1
    k_surf_eff_err_M: float
    rho: float                  # Gamma0/KGm_surf, probe-saturation ratio (Layer 9 Eq IV.4)
    LOD_nM: float               # [E] at Fk = F_MIN (Layer 13's detection threshold)
    ULOQ_nM: float               # [E] at Fk = F_MAX
    dynamic_range_decades: float
    covers_clinical_range: bool  # does [LOD, ULOQ] bracket 6-30 nM?
    # backward-compat field for deconvolution.py / matrix_correction.py,
    # which still use the short-time linear approximation:
    # F_k ~= k_surf_eff*[E]*T / (1 + rho)   (Taylor-expand ln(1/(1-Fk))+rho*Fk near Fk=0)
    slope_M: float


def simulate_calibration_data(k_surf_eff=KCAT_KM_MMP9_SURFACE_DEMO,
                               conc_nM=CAL_CONC_NM, n_rep=N_REPLICATES,
                               T=T_INCUBATION, rho=RHO,
                               sigma_rho_over_rho=SIGMA_RHO_OVER_RHO_SINGLE,
                               n_active=N_ACTIVE_ELECTRODES_DEFAULT, seed=0):
    """
    Simulate a Sec 15.1 calibration experiment using the REAL forward
    model and REAL (heteroscedastic) Layer 13 noise model, rather than a
    flat blank-noise placeholder.
    k_surf_eff defaults to a synthetic placeholder (KCAT_KM_MMP9_SURFACE_DEMO)
    standing in for the as-yet-unmeasured fabricated-electrode value.
    """
    rng = np.random.default_rng(seed)
    E_M = nM_to_M(conc_nM)
    Fk_true = Fk_forward(E_M, T, k_surf_eff, rho)
    Fk_true = np.atleast_1d(Fk_true)

    Fk_mean, Fk_std = [], []
    for Fk_t in Fk_true:
        sigma = sigma_Fk(Fk_t, sigma_rho_over_rho, n_active)
        reps = np.clip(Fk_t + rng.normal(0, sigma, n_rep), 0, 0.999)
        Fk_mean.append(reps.mean())
        Fk_std.append(reps.std(ddof=1) if n_rep > 1 else sigma)
    return conc_nM, np.array(Fk_mean), np.array(Fk_std), Fk_true


def fit_calibration(conc_nM, Fk_mean, Fk_std, T=T_INCUBATION, rho=RHO):
    """
    Fit the single free parameter k_surf_eff (M^-1 s^-1) to calibration
    data using the implicit forward model, weighted by the measured
    (Fk-dependent) uncertainties. rho is held fixed at the Layer 9
    estimate (itself unconfirmed pending Layer 8 delta_eff).
    """
    E_M = nM_to_M(conc_nM)

    def model(E_arr, k_surf_eff):
        return Fk_forward(E_arr, T, k_surf_eff, rho)

    sigma_w = np.clip(Fk_std, 1e-4, None)
    popt, pcov = curve_fit(model, E_M, Fk_mean, p0=[KCAT_KM_MMP9_SURFACE_DEMO],
                            sigma=sigma_w, absolute_sigma=True, maxfev=10000)
    k_fit, k_err = popt[0], np.sqrt(pcov[0, 0])

    LOD_M = E_inverse(F_MIN, T, k_fit, rho)
    ULOQ_M = E_inverse(F_MAX, T, k_fit, rho)
    LOD_nM, ULOQ_nM = M_to_nM(LOD_M), M_to_nM(ULOQ_M)
    dyn_range_decades = np.log10(ULOQ_nM / LOD_nM)
    covers_clinical = (LOD_nM <= WOUND_MMP9_HEALING_NM) and (ULOQ_nM >= WOUND_MMP9_CHRONIC_NM)

    # Short-time linear slope, kept for backward compatibility with
    # modules that still assume F_k ~= slope * [E] (Taylor-expand the
    # implicit equation near Fk=0: ln(1/(1-Fk)) + rho*Fk ~= (1+rho)*Fk).
    slope_M = k_fit * T / (1 + rho)

    return CalibrationResult(
        k_surf_eff_M=k_fit, k_surf_eff_err_M=k_err, rho=rho,
        LOD_nM=LOD_nM, ULOQ_nM=ULOQ_nM, dynamic_range_decades=dyn_range_decades,
        covers_clinical_range=covers_clinical, slope_M=slope_M,
    )


def plot_calibration(res, conc_nM, Fk_mean, Fk_std, T=T_INCUBATION, save=True):
    E_plot_nM = np.linspace(0, max(conc_nM) * 1.1, 300)
    Fk_plot = Fk_forward(nM_to_M(E_plot_nM), T, res.k_surf_eff_M, res.rho) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(conc_nM, Fk_mean * 100, yerr=Fk_std * 100, fmt="ko",
                 capsize=4, markersize=6, label="Calibration data")
    ax1.plot(E_plot_nM, Fk_plot, "b-", lw=2, label="Implicit integrated-MM fit")
    ax1.axvline(res.LOD_nM, color="orange", ls=":", lw=1.5, label=f"LOD (F_min=20%) = {res.LOD_nM:.2f} nM")
    ax1.axvline(res.ULOQ_nM, color="red", ls=":", lw=1.5, label=f"ULOQ (F_max=95%) = {res.ULOQ_nM:.1f} nM")
    ax1.axvspan(WOUND_MMP9_HEALING_NM, WOUND_MMP9_CHRONIC_NM, alpha=0.12, color="green",
                label=f"Clinical range ({WOUND_MMP9_HEALING_NM:.0f}-{WOUND_MMP9_CHRONIC_NM:.0f} nM)")
    ax1.set_xlabel("[MMP-9] (nM)")
    ax1.set_ylabel("F_k (%)")
    ax1.set_title(f"Calibration curve (rho = {res.rho:.2f})")
    ax1.legend(fontsize=7.5)

    # Noise model panel: sigma_Fk(Fk) is heteroscedastic (Layer 13 Eq III.6)
    Fk_range = np.linspace(0, 0.99, 200)
    ax2.plot(Fk_range * 100, sigma_Fk(Fk_range) * 100, "b-", lw=2, label="sigma_Fk (1 electrode)")
    ax2.plot(Fk_range * 100, sigma_Fk(Fk_range, n_active=4) * 100, "g-", lw=2, label="sigma_Fk (4 electrodes)")
    ax2.axvline(F_MIN * 100, color="orange", ls=":", lw=1.5, label=f"F_min = {F_MIN*100:.0f}%")
    ax2.axvline(F_MAX * 100, color="red", ls=":", lw=1.5, label=f"F_max = {F_MAX*100:.0f}%")
    ax2.set_xlabel("F_k (%)")
    ax2.set_ylabel("sigma_Fk (%)")
    ax2.set_title("Layer 13 heteroscedastic noise model")
    ax2.legend(fontsize=8)

    fig.suptitle("Layer 15.2/15.3: MMP-9 channel calibration (Layers 9/13 model)", fontsize=11)
    plt.tight_layout()
    if save:
        out = FIGDIR / "fig_calibration_curves.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def hne_interference_check(c_HNE_nM=30.0, c_MMP9_nM=30.0, k_surf_eff_mmp9=KCAT_KM_MMP9_SURFACE_DEMO,
                            k_surf_eff_hne_on_mmp9_probe=KCAT_KM_HNE_ON_MMP9_PROBE):
    """
    Layer 9 Sec X: in the linear (first-order) regime, contributions to
    the cleavage rate add independently:
        kobs_total = sum_j k_surf_eff_j * [E_j]
    This reports the fractional contribution of HNE to the MMP-9 channel
    signal (xi_HNE), NOT a second deconvolvable channel — there is no HNE
    probe on this device. Entirely dependent on the unconfirmed HNE
    cross-reactivity placeholder; treat as illustrative only.
    """
    k_mmp9 = k_surf_eff_mmp9 * nM_to_M(c_MMP9_nM)
    k_hne = k_surf_eff_hne_on_mmp9_probe * nM_to_M(c_HNE_nM)
    xi_hne = k_hne / (k_mmp9 + k_hne)
    return xi_hne, k_mmp9, k_hne


def write_summary_csv(res, fname="calibration_summary.csv"):
    out = FIGDIR.parent / fname
    row = asdict(res)
    row["channel"] = "MMP9"   # kept for backward compatibility with older consumers
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved {out}")
    return out


def main():
    conc_nM, Fk_mean, Fk_std, Fk_true = simulate_calibration_data()
    res = fit_calibration(conc_nM, Fk_mean, Fk_std)

    print("[MMP-9 channel — single-channel device, D-enantiomer control]")
    print(f"  Synthetic 'true' k_surf_eff (demo input) : {KCAT_KM_MMP9_SURFACE_DEMO:.3e} M^-1 s^-1")
    print(f"  Fitted k_surf_eff                        : {res.k_surf_eff_M:.3e} +/- {res.k_surf_eff_err_M:.2e} M^-1 s^-1")
    print(f"  (Solution-phase upper bound, confirmed)  : {KCAT_KM_MMP9_SOLUTION_UB:.0f} M^-1 s^-1  (Kridel 2001)")
    print(f"  Plausible surface range (Layer 9 Sec XIII): {KCAT_KM_MMP9_SURFACE_RANGE} M^-1 s^-1")
    print(f"  Target range for 30-60 min read (Table 2) : {KCAT_KM_MMP9_SURFACE_TARGET} M^-1 s^-1")
    print(f"  rho = Gamma0/KGm_surf                     : {res.rho:.2f}  (Layer 9 estimates 0.3-3.1)")
    print(f"  LOD  (F_k = {F_MIN*100:.0f}%)                        : {res.LOD_nM:.3f} nM")
    print(f"  ULOQ (F_k = {F_MAX*100:.0f}%)                        : {res.ULOQ_nM:.1f} nM")
    print(f"  Dynamic range                             : {res.dynamic_range_decades:.2f} decades")
    print(f"  Covers clinical range ({WOUND_MMP9_HEALING_NM:.0f}-{WOUND_MMP9_CHRONIC_NM:.0f} nM)?         : "
          f"{'YES' if res.covers_clinical_range else 'NO'}")

    # Cross-check against Layer 13's own closed-form kappa_k dynamic range
    kappa_min, kappa_max = apparent_rate(F_MIN, T_INCUBATION), apparent_rate(F_MAX, T_INCUBATION)
    print(f"\n  Layer 13 kappa_k window: [{kappa_min:.4f}, {kappa_max:.3f}] s^-1 "
          f"({kappa_max/kappa_min:.1f}x, {np.log10(kappa_max/kappa_min):.2f} decades) — "
          "cross-check only, not used in the [E]-based LOD/ULOQ above.")

    xi_hne, k_mmp9, k_hne = hne_interference_check()
    print(f"\n  [Illustrative only — HNE cross-reactivity is UNCONFIRMED, Layer 9 Sec X]")
    print(f"  At 30 nM HNE + 30 nM MMP-9 (both chronic-level): "
          f"HNE contributes xi_HNE = {xi_hne*100:.1f}% of the channel signal")

    plot_calibration(res, conc_nM, Fk_mean, Fk_std)
    write_summary_csv(res)
    return res


if __name__ == "__main__":
    main()
