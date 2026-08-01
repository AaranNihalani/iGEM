# Layer 12: SWV peak extraction and signal processing

## Purpose

The SWV forward model (Layer 11) produces a theoretical voltammogram ΔI(E). The instrument measures a noisy, baseline-corrupted version of this trace. This layer takes the raw measured (E_n, ΔI_n) array and extracts the MB and Fc peak areas with quantified uncertainty. These areas feed directly into the ratiometric calculation (Layer 13).

This layer is a signal processing pipeline, not a physical model per se. But its implementation matters: poor peak extraction (wrong baseline, clipped peaks, misidentified peaks) will corrupt all downstream calculations regardless of how accurate the physical model is.

## Sub-components to model / implement

### 12.1 Raw data format

Input: array of (E_n, ΔI_n) with n ≈ 650 points for a -0.65 to +0.65 V scan at 2 mV step.

Expected features:
- MB reduction peak: centred near E°'_MB(pH), typically -0.22 to -0.31 V vs on-chip RE (pH 6.5-8.0)
- Fc reduction peak: centred near E°_Fc ≈ +0.45 V vs on-chip RE (pH-independent)
- Sloping background: from residual capacitive contribution ΔI_cap,n (Layer 11.3); approximately linear across the scan
- Noise floor: ±4 pA Johnson noise from TIA (Layer 14)
- Possible artefact: large step/spike at E = E_switch point if two-segment scan used

### 12.2 Smoothing

Savitzky-Golay filter: convolves data with a polynomial kernel to smooth high-frequency noise while preserving peak shape.
- Window: 11 points (22 mV at 2 mV step)
- Polynomial order: 3
- Effect: reduces per-point noise by factor ~sqrt(11/2) ≈ 2.3 without significant peak broadening (peak FWHM ≈ 65 mV >> window ≈ 22 mV)

Implementation: scipy.signal.savgol_filter

### 12.3 Background subtraction

The sloping capacitive baseline must be removed before peak fitting.

Method: fit a polynomial (3rd or 4th order) through user-defined or auto-selected non-peak baseline regions:
- MB baseline: E < -0.55 V and E > -0.05 V (MB peak window: -0.55 to -0.05 V)
- Fc baseline: E < +0.20 V and E > +0.65 V scan limit (Fc peak window: +0.20 to +0.65 V)

Note: baseline regions must not overlap with peak tails. Peak FWHM ≈ 65 mV → peak tails extend ±3σ ≈ ±75 mV. MB peak window should be ±100 mV around E°_MB: [-0.45, -0.05] V (350 mV window — may need adjustment if pH shifts E°_MB significantly).

Auto-baseline: fit a smoothing spline to all points outside the peak windows, then interpolate across the windows. More robust to varying baselines than a fixed polynomial.

Quality check: if the fitted baseline has a residual that deviates by more than 5× the noise floor, flag as baseline anomaly (possible fouling or bubble at electrode).

### 12.4 Peak fitting

After baseline subtraction, fit a Gaussian to the MB region and a separate Gaussian to the Fc region:
    ΔI(E) = A_peak * exp(-(E - E_peak)^2 / (2 * sigma_E^2))

Fitted parameters (3 per peak):
- A_peak: peak amplitude (nA)
- E_peak: peak position (V) — expected near E°_MB (pH-corrected)
- sigma_E: peak width (V) — expected ≈ FWHM/(2.355) ≈ 0.027 V for reversible surface wave (n=2)

Peak area:
    A_area = A_peak * sigma_E * sqrt(2 * pi)   (integrated area, nA·V)

Why peak area, not peak height:
- k_ET^0 varies with Ca2+, temperature, and between electrodes → SWV peak height changes even at constant Gamma (through kappa_SW)
- Peak AREA is more robust: for a surface-confined peak, the area = n * F * Gamma_intact * A_true × (some factor), and this factor varies less with k_ET^0 than the height does
- Peak area also integrates out per-point noise more effectively than peak height (more robust to noise spikes)

Nonlinear least squares: scipy.optimize.curve_fit with initial guesses from peak position and height. Report covariance matrix → uncertainties sigma_A_peak and sigma_sigma_E → propagate to uncertainty in A_area.

### 12.5 Two-Gaussian simultaneous fit

If the Fc and MB peaks are both needed from the same scan, fit two Gaussians simultaneously:
    ΔI(E) = A_MB * exp(-(E - E_MB)^2/(2*sigma_MB^2)) + A_Fc * exp(-(E - E_Fc)^2/(2*sigma_Fc^2)) + background

This is important near the boundaries of peak windows if the peaks overlap marginally (unlikely given 0.70 V separation, but peaks can be wider in quasi-reversible regime).

### 12.6 Quality diagnostics

Run these checks on every scan and flag if triggered:

| Diagnostic | Threshold | Interpretation if triggered |
|---|---|---|
| Peak width sigma_E | > 2 × expected (0.054 V) | SAM heterogeneous or signal from dissolved MB |
| Peak width sigma_E | < 0.5 × expected (0.014 V) | Noise spike misidentified as peak |
| Peak position shift |E_MB - E°'_MB(pH)| | > 25 mV vs expected | RE drift, pH error, or artefact |
| Fit residuals chi^2/dof | > 5 | Non-Gaussian peak shape; use two-Gaussian model |
| A_Fc change between scans | > 10% | Fc SAM damaged or Gamma_Fc changed |
| Negative A_peak | < -1σ | No detectable peak; signal below LOD |

### 12.7 Frequency diagnostic (implementation)

From Layer 11.8: scan the same electrode at multiple f_SWV values (5, 15, 30, 60, 120 Hz). Extract A_peak at each frequency. Plot A_peak vs f_SWV:
- If A_peak ∝ f^1 (at low f where reversible): surface-confined confirmation
- If A_peak ∝ f^0.5: diffusion-controlled (wrong — dissolved MB is dominating)

This diagnostic should be run once per batch of cartridges, not on every assay cartridge (too slow).

### 12.8 Handling edge cases

- Peak below LOD: if A_peak < 3 × sigma_baseline per point × sqrt(N_peak_pts): report A_area = 0 with uncertainty = A_LOD. Do not attempt to extract ratio R.
- Peak clipped (too large for ADC range): redesign TIA gain or reduce Gamma_p. Flag if I_peak > 0.9 × I_range_TIA.
- Double MB peak: if chi^2 from single-Gaussian is high and residuals show two humps, try two-Gaussian fit. Two MB peaks can arise from: (a) two populations of probes with different d_eff; (b) cleaved stub + intact probe contributing at different E; (c) physisorbed vs chemisorbed probe. Report both peak areas separately.

## Key outputs

| Output | Uncertainty source | Feeds |
|---|---|---|
| A_area,MB (nA·V) | sigma from fit covariance | Layer 13 (R = I_MB/I_Fc) |
| A_area,Fc (nA·V) | sigma from fit covariance | Layer 13 |
| E_peak,MB (V) | sigma from fit; RE drift | Layer 10 (pH correction check) |
| sigma_E,MB (V) | fit | Quality diagnostic |
| Per-scan quality flags | Boolean | Data pipeline quality control |

## Key literature to find

- Savitzky-Golay filter: Savitzky & Golay (Anal Chem 1964); Press et al. (Numerical Recipes)
- SWV peak fitting for DNA biosensors: Plaxco group protocols; Baker et al. (2006)
- Peak area vs peak height in SWV: Mirceski et al. (Square-Wave Voltammetry); Komorsky-Lovric
- Baseline subtraction for electrochemical data: Morales-Narvaez & Merkoci (review on biosensor signal processing)

## Files to create

- `peak_extraction.py`: Savitzky-Golay smoothing, polynomial baseline, Gaussian fit, area calculation
- `quality_checks.py`: automated diagnostic flags on each scan
- `frequency_diagnostic.py`: multi-frequency scan to confirm surface-confinement regime
