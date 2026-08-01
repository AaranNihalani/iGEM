# Layer 15: Calibration curve, LOD, and clinical interpretation

## Purpose

The forward model (Layers 0-14) produces a predicted F_k for a known [E]_wound. The calibration layer inverts this: given a measured F_k, what is [E]_wound? The calibration curve must be built in a wound-fluid-matched matrix to account for matrix effects, and the LOD / LLOQ must be determined under realistic conditions.

This layer also contains the two-channel deconvolution (solving for [MMP-9] and [HNE] simultaneously from F_MMP9 and F_HNE), which connects directly to the substrate design work: the condition number kappa(S) of the cross-sensitivity matrix S is the iGEM-traceable metric linking substrate selectivity to clinical readout accuracy.

Finally, this layer outputs the clinical interpretation: from [MMP-9] and [HNE], what is the wound state probability P(wound_state | [MMP-9], [HNE])?

## Sub-components to model

### 15.1 Calibration experiment design

To build the calibration curve, spike known concentrations of active MMP-9 and active HNE into wound-fluid-matched matrix:
- Matrix: PBS + 4 mg/mL BSA + 50 µM fibrinogen + 1 mM Ca2+ + 0.1 mg/mL IgG (simulates wound fluid matrix effects without the proteases)
- pH: tested at pH 6.5, 7.0, 7.4, 8.0 (4 levels)
- MMP-9 concentrations: 0.1, 0.3, 1, 3, 10, 30, 100 nM (7 levels)
- HNE concentrations: 0.1, 0.3, 1, 3, 10, 30, 100 nM (7 levels)
- Each concentration in triplicate cartridges
- Also: both enzymes present simultaneously at various ratios (for cross-sensitivity matrix)

All calibration cartridges use the same SAM preparation protocol as production cartridges.

### 15.2 Calibration model (single analyte)

In the first-order regime (Da < 0.1, Gamma << K_m,surf):
    F_k ≈ 1 - exp(-k_obs * T)
    k_obs = (k_cat / K_m,surf) * [E]_chamber = (k_cat / K_m,surf) * [E]_wound * dilution_factor

For small F_k (early time or low [E]):
    F_k ≈ k_obs * T = (k_cat / K_m,surf * dilution_factor * T) * [E]_wound = slope * [E]_wound

Linear range: F_k < 0.5 (approximately). Above F_k = 0.5, the exponential saturates and the linear approximation fails.

Full nonlinear calibration model:
    F_k([E]) = 1 - exp(-slope * [E])
    slope = (k_cat / K_m,surf) * D_f * T

Fit slope from calibration data. Report uncertainty in slope (sigma_slope) from the fit.

### 15.3 LOD and LLOQ

LOD: the minimum [E]_wound that gives F_k statistically distinguishable from blank (F_k = 0):
    LOD = 3 * sigma_blank / slope_calibration

where sigma_blank is the standard deviation of F_k measurements on cartridges loaded with blank matrix (no enzyme).

LLOQ (lower limit of quantification): minimum [E] with acceptable quantitative accuracy (< 20% relative error):
    LLOQ = 10 * sigma_blank / slope_calibration

Expected (from Layers 9, 13, 14):
- slope ≈ (10^4 M^-1 s^-1) * 0.8 * 1800 s ≈ 1.44 × 10^7 M^-1 (kcat/Km = 10^4, dilution 0.8, T = 30 min)
- sigma_blank ≈ 0.002 (0.2%, from Layer 13)
- LOD ≈ 3 * 0.002 / (1.44 × 10^7) = 4.2 × 10^-10 M = 0.42 nM

Clinical MMP-9 in wound fluid: typically 1-100 nM → LOD well within range. But note: dilution factor D_f and elution efficiency eta_elut (Layer 1) reduce effective [E]_chamber. If D_f * eta_elut = 0.5: effective LOD = 0.84 nM (still adequate).

Upper limit of quantification (ULOQ): when F_k > 0.95, the exponential calibration curve is very steep (uncertain inverse):
    ULOQ = -ln(0.05) / slope ≈ 3 / slope ≈ 200 nM

Dynamic range: LLOQ to ULOQ = 0.7 nM to 200 nM ≈ 2.5 decades.

### 15.4 Matrix effects on calibration

Calibration in PBS buffer alone will systematically overestimate the slope vs calibration in wound-fluid-matched matrix. Sources of matrix effects:
- Endogenous inhibitors in wound fluid (TIMP-1, A1AT) → reduce k_obs → lower slope
- Competitor cleavage by interfering proteases → increase F_k → inflate apparent [MMP-9] or [HNE]
- Non-specific fouling → reduce A_MB → reduce R → systematic negative bias on F_k
- Ca2+, albumin, ionic strength → all affect SAM, k_ET, enzyme activity

Standard addition method: spike known [E] into real wound fluid samples to test recovery. Recovery = [(F_k,spiked - F_k,blank) / F_k,calibration] * 100%. Target: 80-120% recovery.

If recovery deviates from 100% systematically: apply a matrix correction factor m:
    [E]_calibrated = [E]_nominal * m

Determine m from a panel of wound fluid samples.

### 15.5 Two-channel deconvolution (MMP-9 and HNE)

The device has two active channels: MMP-9 probe and HNE probe. In practice, BOTH MMP-9 and HNE are present in wound fluid simultaneously, and each protease may partially cleave both probes (cross-sensitivity).

Define: F_obs = [F_MMP9-channel, F_HNE-channel]^T, c = [c_MMP9, c_HNE]^T (concentrations in nM).

Linear sensitivity model (valid when F_k << 1):
    F_obs = S * c
    S = [[S11, S12], [S21, S22]]

where:
    S11 = dF_MMP9-channel / dc_MMP9 (slope of MMP-9 probe vs MMP-9 concentration)
    S12 = dF_MMP9-channel / dc_HNE (cross-sensitivity: how much HNE cleaves MMP-9 probe)
    S21 = dF_HNE-channel / dc_MMP9 (cross-sensitivity: how much MMP-9 cleaves HNE probe)
    S22 = dF_HNE-channel / dc_HNE (slope of HNE probe vs HNE concentration)

From substrate predictor (Panels A and B): IPRTLTAA is selective for MMP-9 (not cleaved by HNE); AAPVSGIG is selective for HNE (not cleaved by MMP-9). If this selectivity is perfect: S12 ≈ 0 and S21 ≈ 0 → S is diagonal → deconvolution is trivial:
    c_MMP9 = F_MMP9 / S11
    c_HNE = F_HNE / S22

In practice, S12 and S21 are small but non-zero. Deconvolution:
    c_est = S^-1 * F_obs

Condition number of S:
    kappa(S) = ||S|| * ||S^-1||

A large kappa means noise in F_obs is amplified into large uncertainty in c_est.

Connection to substrate design: the substrate selection criterion kappa(S) < 30 (from Panel A/B reports) was specifically designed to ensure that the deconvolution noise amplification is bounded. This connects the bioinformatics dry-lab work (substrate selection) to the electrochemical device output (clinical [E] estimates). This is the key iGEM model-experiment integration point for the device.

Uncertainty in deconvolution:
    sigma_c = S^-1 * sigma_F (vector; applies element-wise)

For kappa(S) = 5 (well-conditioned) and sigma_F = 0.002:
    sigma_c_MMP9 ≈ kappa(S) * sigma_F / S11 = 5 * 0.002 / S11

If S11 = 1.44 × 10^7 M^-1: sigma_c_MMP9 ≈ 5 * 0.002 / (1.44 × 10^7) ≈ 0.7 nM uncertainty on MMP-9 concentration (after deconvolution). This is approximately equal to the LOD — deconvolution does not significantly degrade LOD for a well-conditioned S.

### 15.6 pH correction to calibration

E°_MB shifts with pH (Layer 10.3), which shifts the SWV peak position and potentially the peak area (if the fitting window clips the shifted peak). This causes a systematic bias in A_MB and hence in R and F_k.

Correction: include pH as a covariate in the calibration model:
    F_k = 1 - exp(-slope * [E] * pH_correction(pH))

where pH_correction(pH) accounts for the change in kcat (enzyme activity is pH-dependent) and any SWV peak fitting error with pH.

If a pH meter is included in the device (or pH is estimated from E_peak,MB shift): apply the correction in real-time. Otherwise: report F_k as calibrated at pH 7.4 with an uncertainty band covering the pH 6.5-8.2 range.

### 15.7 Bayesian wound state classifier

Given [c_MMP9, c_HNE] from the deconvolution, classify the wound state:
    P(wound_state | c_MMP9, c_HNE) = P(c_MMP9, c_HNE | wound_state) * P(wound_state) / P(c_MMP9, c_HNE)

Wound states:
- Healing: low MMP-9, low HNE
- Inflammatory: high HNE, moderate MMP-9
- Chronic non-healing: elevated MMP-9, elevated HNE, high ratio MMP-9/HNE

Likelihood P(c_MMP9, c_HNE | wound_state): Gaussian model fit from the wound biology ODE (Layer 0) predictions of [MMP-9] and [HNE] in each wound state, plus literature data.

This Bayesian classifier is the clinical output of the entire device model chain: from swab of wound → F_k → [MMP-9, HNE] → P(wound_state). This is the iGEM demonstration of end-to-end integration.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| slope_calibration | ~1.4×10^7 M^-1 | LOD, LLOQ |
| LOD | ~0.4-1 nM (pre-analytical losses included) | Clinical validation |
| LLOQ | ~1.5-3 nM | Clinical validation |
| Dynamic range | ~0.7-200 nM (~2.5 decades) | Clinical validation |
| S matrix (kappa(S)) | < 30 (substrate design requirement) | Links to substrate predictor |
| sigma_c (deconvolved) | ~0.7 nM | Clinical readout uncertainty |
| P(wound_state) | Posterior probability | Clinical output |

## Key literature to find

- LOD and LLOQ definitions: ICH Q2(R1) (analytical validation guideline)
- Electrochemical biosensor LOD in wound fluid: Yeung et al. (Biosens Bioelectron); Yager et al. (Nature 2006)
- Cross-sensitivity and condition number: signal processing / analytical chemistry references; also Layer 13 of this report
- Wound state classification from protease levels: Trengove et al. (Wound Repair Regen 1999); Lazaro et al.
- Bayesian classification for biomarker panels: general Bayesian statistics references

## Files to create

- `calibration_curve.py`: fit slope from calibration data; generate F_k vs [E] curves; LOD/LLOQ
- `deconvolution.py`: build S matrix from single-analyte calibrations; solve S^-1 * F_obs; condition number
- `matrix_correction.py`: spike-and-recovery model; matrix correction factor
- `wound_classifier.py`: Bayesian classifier P(wound_state | c_MMP9, c_HNE)
