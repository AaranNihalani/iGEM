"""
matrix_correction.py  —  Layer 15.4 (rewritten against Layers 9 & 13)
=========================================================================
Matrix-effect characterisation via the standard addition (spike-and-
recovery) method, updated to use the real implicit integrated-MM model
(calibration_curve.py) and the real Layer 13 noise model, and to reflect
what actually survives the D-enantiomer double-ratio correction.

Layer 13 shows the ratiometric double ratio F_k = 1 - rho_active/rho_control
(Sec III) ALREADY cancels, exactly, any non-stereospecific background
cleavage (nucleases, ROS, thiol-bond hydrolysis) AND any UNIFORM
electrode fouling (Sec II, "concrete fouling example"). So the matrix
effects that can still bias a calibration built in buffer vs. real
wound-fluid-matched matrix are narrower than a naive picture suggests:

  1. A genuine change in the enzyme's catalytic efficiency (kcat/Km) in
     the matrix vs. buffer — ionic strength, Ca2+, BSA/fibrinogen
     crowding, etc. — modelled as a multiplicative attenuation on
     k_surf_eff.
  2. pH shift (Sec 15.6 / Layer 10), via the existing pH_correction().
  3. DIFFERENTIAL fouling between the active (SGKIPRTLTA) and control
     (D-SGKIPRTLTA) electrodes — the one fouling scenario Layer 13
     explicitly flags as NOT cancelled (Sec VII, Limitation 2) — modelled
     as a small residual additive bias on F_k.

Recovery(%) = (F_k,spiked - F_k,blank) / F_k,calibration * 100
Target acceptance band: 80-120% (contents.md Sec 15.4).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from layer15_common import (
    T_INCUBATION, RHO, KCAT_KM_MMP9_SURFACE_DEMO, PH_LEVELS,
    pH_correction, sigma_Fk, nM_to_M, M_to_nM,
)
from calibration_curve import Fk_forward, E_inverse

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# Matrix-effect parameters for this simulation (placeholders pending
# wet-lab spike-and-recovery data on real wound fluid samples).
MATRIX_KCAT_ATTENUATION = 0.85       # matrix reduces effective k_surf_eff by ~15%   # TBD (wet lab)
DIFFERENTIAL_FOULING_BIAS_FK = 0.015  # residual F_k bias NOT cancelled by the double
                                       # ratio (Layer 13 Sec VII Limitation 2)         # TBD (wet lab)


def simulate_spike_recovery(conc_nM, k_surf_eff_nominal=KCAT_KM_MMP9_SURFACE_DEMO,
                             T=T_INCUBATION, rho=RHO, pH=7.4, n_rep=3, seed=0):
    """
    Simulate spiking known [MMP-9] into real wound-fluid-matched matrix
    (matrix effects applied) vs. the buffer-only calibration curve
    (nominal k_surf_eff, pH 7.4, no matrix attenuation).
    """
    rng = np.random.default_rng(seed)
    matrix_k_surf_eff = k_surf_eff_nominal * MATRIX_KCAT_ATTENUATION * pH_correction(pH)

    E_M = nM_to_M(conc_nM)
    Fk_spiked_true = np.atleast_1d(Fk_forward(E_M, T, matrix_k_surf_eff, rho)) + DIFFERENTIAL_FOULING_BIAS_FK
    Fk_blank_true = DIFFERENTIAL_FOULING_BIAS_FK   # matrix-only cartridge, no spiked enzyme

    Fk_spiked = np.clip(
        Fk_spiked_true + rng.normal(0, sigma_Fk(Fk_spiked_true), len(conc_nM)), 0, 0.999)
    Fk_blank = np.clip(
        Fk_blank_true + rng.normal(0, sigma_Fk(Fk_blank_true), n_rep).mean(), 0, 0.999)

    return Fk_blank, Fk_spiked, matrix_k_surf_eff


def compute_recovery(Fk_spiked, Fk_blank, conc_nM, k_surf_eff_calibration, T=T_INCUBATION, rho=RHO):
    """
    Recovery(%) = (F_k,spiked - F_k,blank) / F_k,calibration * 100
    where F_k,calibration is the NOMINAL buffer-only calibration curve's
    predicted F_k for the same spiked concentration.
    """
    E_M = nM_to_M(conc_nM)
    Fk_calibration = np.atleast_1d(Fk_forward(E_M, T, k_surf_eff_calibration, rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        recovery_pct = np.where(Fk_calibration > 0,
                                 (Fk_spiked - Fk_blank) / Fk_calibration * 100,
                                 np.nan)
    return recovery_pct


def fit_correction_factor(conc_nM, recovery_pct, Fk_calibration, min_Fk_calibration=0.10):
    """
    Mean multiplicative correction factor m such that m * mean(recovery) ~= 100%.
    Points where the nominal calibration F_k is small (near LOD) are excluded:
    the recovery ratio there is dominated by noise/offset artefacts rather
    than the true matrix attenuation (Layer 13's own F_min = 20% detection
    floor is the relevant scale here).
    """
    valid = ~np.isnan(recovery_pct) & (conc_nM > 0) & (Fk_calibration >= min_Fk_calibration)
    mean_recovery = np.nanmean(recovery_pct[valid])
    m = 100.0 / mean_recovery
    return m, mean_recovery


def apply_correction_to_concentration(c_nominal_nM, m):
    """[E]_calibrated = [E]_nominal * m"""
    return c_nominal_nM * m


def main():
    conc_nM = np.array([1, 3, 10, 30, 100])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    all_recovery_by_pH = {}
    for pH in PH_LEVELS:
        Fk_blank, Fk_spiked, matrix_k = simulate_spike_recovery(conc_nM, pH=pH, seed=int(pH * 10))
        recovery = compute_recovery(Fk_spiked, Fk_blank, conc_nM, KCAT_KM_MMP9_SURFACE_DEMO)
        all_recovery_by_pH[pH] = recovery
        ax1.plot(conc_nM, recovery, "o-", label=f"pH {pH}")

    ax1.axhspan(80, 120, alpha=0.1, color="green", label="Acceptance band (80-120%)")
    ax1.axhline(100, color="k", lw=1, ls="--")
    ax1.set_xscale("log")
    ax1.set_xlabel("Spiked [MMP-9] (nM)")
    ax1.set_ylabel("Recovery (%)")
    ax1.set_title("Spike-and-recovery vs concentration and pH")
    ax1.legend(fontsize=8)

    m, mean_recovery = fit_correction_factor(
        conc_nM, all_recovery_by_pH[7.4],
        Fk_forward(nM_to_M(conc_nM), T_INCUBATION, KCAT_KM_MMP9_SURFACE_DEMO, RHO))
    print(f"Mean recovery at pH 7.4: {mean_recovery:.1f}%")
    print(f"Fitted matrix correction factor m = {m:.3f}")
    print("Within 80-120% acceptance band without correction: "
          f"{'YES' if 80 <= mean_recovery <= 120 else 'NO — correction required'}")
    print(f"(Matrix attenuates k_surf_eff by {(1-MATRIX_KCAT_ATTENUATION)*100:.0f}%, "
          f"consistent with the ~{100-mean_recovery:.0f}% recovery shortfall observed above)")

    corrected = apply_correction_to_concentration(conc_nM, m)
    ax2.plot(conc_nM, conc_nM, "k--", label="Nominal (no correction)")
    ax2.plot(conc_nM, corrected, "bo-", label=f"Matrix-corrected (m={m:.2f})")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Nominal spiked [MMP-9] (nM)")
    ax2.set_ylabel("Calibrated [MMP-9] (nM)")
    ax2.set_title("Matrix correction applied to nominal concentration")
    ax2.legend(fontsize=8)

    fig.suptitle("Layer 15.4: Matrix effects and spike-and-recovery (Layers 9/13 model)", fontsize=11)
    plt.tight_layout()
    out = FIGDIR / "fig_matrix_correction.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    return m, all_recovery_by_pH


if __name__ == "__main__":
    main()
