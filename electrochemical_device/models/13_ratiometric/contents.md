# Layer 13: Ratiometric signal calculation and error propagation

## Purpose

This layer converts the raw SWV peak areas (A_MB, A_Fc) from Layer 12 into the final dimensionless signals R, rho, F_k, and kappa_k, and propagates uncertainty through each step. The ratiometric design is the core innovation that makes the device robust to electrode variability and fouling; this layer models exactly how well that robustness works and what residual errors remain.

The key output is sigma_F (uncertainty in the fractional cleavage signal), which determines the LOD and dynamic range of the entire device.

## Signal definitions and derivations

### 13.1 Ratiometric signal R_i(t)

For electrode i at time t, using peak areas from Layer 12:
    R_i(t) = A_MB,i(t) / A_Fc,i(t)

Justification: A_MB ∝ n^2 * F^2 * A_true * Gamma_intact * k_ET^0,MB * ΔE_SW / (4RT) (quasi-reversible peak, from Layer 11.4)
              A_Fc ∝ n_Fc^2 * F^2 * A_true * Gamma_Fc * k_ET^0,Fc * ΔE_SW / (4RT) (quasi-reversible)

The ratio R = A_MB / A_Fc cancels:
- True electrode area A_true (both MB and Fc are on the same electrode)
- Instrument gain (same TIA amplifies both)
- Temperature (both peaks scale equally with T through n^2*F^2/(4RT))
- ΔE_SW (both peaks scale equally with ΔE_SW)

R does NOT cancel:
- k_ET^0,MB vs k_ET^0,Fc (different ET rates → scale differently with kappa_SW)
- Gamma_Fc changes (if Fc is lost from surface)
- Ca2+ artefact (affects MB tether but not Fc tether differently)
- Probe cleavage (reduces A_MB but not A_Fc)

R before cleavage (baseline):
    R_i(0) = (n_MB^2 * Gamma_0,MB * k_ET^0,MB,eff) / (n_Fc^2 * Gamma_Fc * k_ET^0,Fc,eff) × (some factor)

This baseline is unique to each electrode (Gamma_0,MB and Gamma_Fc vary between electrodes due to SAM deposition variability, Layer 7.2). But since it is measured for each electrode individually and used to normalise, this variation cancels in rho.

Uncertainty in R:
    sigma_R / R = sqrt((sigma_MB / A_MB)^2 + (sigma_Fc / A_Fc)^2)

where sigma_MB and sigma_Fc are from the fit covariance matrix (Layer 12). For SNR = 1500 per peak (Layer 11.4): sigma_A/A ≈ 0.07% → sigma_R/R ≈ 0.1%.

### 13.2 Baseline measurement R_i(0)

The baseline R_i(0) is measured at T=0, immediately after chamber fill. But 'T=0' is ambiguous: enzyme begins cleaving as soon as it contacts the probe. If t_fill = 30 s and kobs * t = 0.005 * 30 = 0.15: about 0.15% of probes are cleaved before the first scan is taken. This systematic offset in R_i(0) introduces a small negative bias in F_k. Correction: extrapolate R_i(t) back to t = t_fill using the kinetic model. Or: define T=0 as the time of the first scan, and accept this small bias as a systematic error much smaller than other uncertainties.

If a pre-load baseline is possible (scan the electrode in buffer BEFORE adding sample): R_i(0) is truly uncontaminated. But this requires a two-step protocol (scan, then load sample). For a disposable cartridge, this is a design choice that must be specified.

### 13.3 Normalised signal rho_i

    rho_i = R_i(T) / R_i(0)

Uncertainty:
    sigma_rho / rho = sqrt((sigma_R(T) / R(T))^2 + (sigma_R(0) / R(0))^2) ≈ sqrt(2) * sigma_R / R

(assuming equal noise on R(0) and R(T) — valid if both measured with the same SWV scan)

rho captures the fractional change in R from baseline to endpoint. If no cleavage occurs and no fouling: rho = 1. If cleavage occurs (A_MB decreases, A_Fc stable): rho < 1. If fouling occurs (both A_MB and A_Fc decrease proportionally): rho ≈ 1 (fouling partially cancels). If only A_Fc is lost (Fc SAM degrades): rho > 1 (spurious signal).

### 13.4 Active/control double ratio F_k

For channel k (e.g., HNE channel):
- Active electrode(s): probe = AAPVSGIG (cleavable by HNE)
- Control electrode(s): probe = scrambled sequence (not cleaved)

    F_k = 1 - rho_active,k / rho_control,k

Uncertainty:
    sigma_F = |rho_active / rho_control| * sqrt((sigma_rho_a / rho_a)^2 + (sigma_rho_c / rho_c)^2)
            = |F_k - 1| * sqrt(2) * sigma_rho / rho   (if both have equal noise)

For sigma_rho/rho ≈ 0.14% (from above): sigma_F ≈ 1.4 * |1 - F_k| * 0.14% ≈ 0.20% (at F_k = 0, sigma_F ≈ 0.20%).

F_LOD = 3 * sigma_F = 0.60% → LOD corresponds to a fractional cleavage of 0.60%.

What F_k measures: F_k = 1 - rho_active/rho_control ≈ 1 - (Gamma_intact/Gamma_0) × (correction factors)

If ALL corrections factor perfectly: F_k = Gamma_cleaved / Gamma_0 = fractional cleavage = true thermochemical output.

### 13.5 What the double ratio cancels (and what it doesn't)

CANCELS (both active and control affected equally):
- Non-specific fouling: albumin adsorbs to both electrodes → A_MB drops proportionally → rho_active / rho_control ≈ 1 → F_k unchanged
- Temperature drift: both k_ET^0 change by same factor → R changes by same factor → rho stays → F_k unchanged
- Ca2+ artefact: same PEG architecture on both probes → same delta_d_Ca → same k_ET^0 increase → R changes by same factor → cancels (PROVIDED Gamma_p is the same on both electrodes)
- Reference electrode drift: E_ref shifts → both MB and Fc peaks shift by same amount → A_MB and A_Fc change by same factor → R unchanged → rho unchanged → F_k unchanged (assuming ratio of peak areas is independent of absolute peak positions, which is approximately true)
- Batch-to-batch Gamma variation: if Gamma_0,MB and Gamma_Fc are the same on active and control electrodes (they should be if SAM deposition is uniform) → R(0) normalises out

DOES NOT CANCEL:
- Probe-sequence-specific fouling: if albumin preferentially adsorbs to/near the AAPVSGIG sequence (unlikely but possible) → A_MB drops on active but not control → false positive signal
- Different Gamma_p on active vs control: if SAM deposition gives different Gamma_0 on active vs control → Ca2+ artefact, steric effects, etc. differ → incomplete cancellation
- Position-dependent effects: if electrode 1 (near inlet) experiences different matrix vs electrode 8 (far end) → R(0)_active ≠ R(0)_control in a position-correlated way

Mitigation of non-cancelling effects: alternate active and control electrodes along the chamber (ACTIVE-CONTROL-ACTIVE-CONTROL... pattern), so each adjacent pair is at similar position → position-dependent effects partially cancel within each pair.

### 13.6 Apparent rate kappa_k

    kappa_k = -ln(1 - F_k) / T

Validity: assumes first-order cleavage kinetics with constant [E]. Derivation: Gamma(T) = Gamma_0 * exp(-k_obs * T) → F = 1 - exp(-k_obs * T) → k_obs = -ln(1-F)/T = kappa_k.

Uncertainty:
    sigma_kappa = sigma_F / ((1 - F_k) * T)

Diverges as F_k → 1 (large uncertainty in kappa when almost all probes are cleaved). Saturation ceiling at F_k = 0.9: sigma_kappa ≈ 10 * sigma_F / T.

Dynamic range of kappa_k useful measurement:
    Lower bound: kappa_min = 3 * sigma_F / T (limited by LOD)
    Upper bound: kappa_max = -ln(0.1) / T = 2.303 / T (limited by saturation at F = 0.9)

For T = 1800 s (30 min):
    kappa_min = 3 * 0.002 / 1800 ≈ 3 × 10^-6 s^-1 (for sigma_F = 0.002, i.e., 0.2%)
    kappa_max = 2.303 / 1800 ≈ 1.28 × 10^-3 s^-1
    Dynamic range: kappa_max/kappa_min ≈ 427 (about 2.6 decades)

In terms of [E]:
    k_obs = (kcat/Km_surf) * [E] = kappa_k → [E] = kappa_k * Km_surf / kcat
    Dynamic range in [E]: 2.6 decades ≈ 427-fold concentration range

This is a reasonable dynamic range for a clinical assay.

### 13.7 Multiple active electrodes per channel

If 3 active + 1 control electrodes are used for each channel:
- Average F_k from 3 active electrodes vs 1 control: sigma_F_avg = sigma_F / sqrt(3) ≈ 0.12%
- F_LOD,avg = 0.36% (better by sqrt(3) = 1.73×)

Trade-off: more active electrodes per channel → better LOD → fewer channels (can't use electrodes for a second analyte).

Design choice for 6-8 WEs, two analytes (MMP-9, HNE):
- Option A: [MMP9_a, HNE_a, control, control, MMP9_a, HNE_a, control, control] → 4 active (2 per analyte) + 4 control
- Option B: [MMP9_a, MMP9_a, MMP9_a, control, HNE_a, HNE_a, HNE_a, control] → 6 active (3 per analyte) + 2 control

Report the expected sigma_F for each layout; recommend the best trade-off.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| R_i(t), sigma_R | ~0.5-2 ratio, 0.1% uncertainty | Internal (rho) |
| rho_i, sigma_rho | ~0.5-1.0, 0.14% uncertainty | F_k calculation |
| F_k, sigma_F | 0-0.9, ~0.2% uncertainty at F=0 | Layer 14, 15 |
| kappa_k, sigma_kappa | 10^-6 to 10^-3 s^-1 | Layer 15 (calibration) |
| F_LOD | ~0.6% | Layer 15 |
| kappa dynamic range | ~2.6 decades | Layer 15 |

## Key literature to find

- Ratiometric electrochemical biosensors: White & Plaxco (Anal Chem 2010) — original ratiometric design paper
- Active/control double ratio design: Lubin & Plaxco (Acc Chem Res 2010)
- Error propagation for ratio signals: Taylor (Introduction to Error Analysis, textbook)
- Fouling effects on ratiometric sensors: Kang et al. (JACS 2012); Swensen et al. (JACS 2009)

## Files to create

- `ratiometric.py`: compute R, rho, F_k, kappa_k from peak areas; full error propagation
- `cancellation_analysis.py`: model which effects cancel and which don't; compute residual error from each non-cancelling source
- `electrode_layout.py`: compute sigma_F for different electrode count configurations
