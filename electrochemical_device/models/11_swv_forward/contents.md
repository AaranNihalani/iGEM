# Layer 11: SWV forward model

## Purpose

Square-wave voltammetry (SWV) is the electrochemical measurement technique used to read out the MB and Fc signals. This layer takes as input the surface MB coverage Gamma_intact(t) (from Layer 9) and the effective ET rate k_ET^0,eff (from Layer 10), and produces as output the predicted SWV net current ΔI(E) — the complete voltammogram that the instrument records.

This is one of the two most critical modelling layers. The entire sensing chain culminates in this SWV trace; the ratiometric signal, noise floor, and calibration all depend on the predicted peak heights and positions that come from this model.

## SWV waveform

SWV superimposes a staircase ramp (step ΔE_step) with a symmetric square wave (amplitude ΔE_SW, frequency f_SWV):
- At each potential step E_n: voltage is stepped by ΔE_step (forward direction)
- Within each step: square wave adds +ΔE_SW (forward half-cycle) then -ΔE_SW (backward half-cycle)
- Current is sampled at the end of each half-cycle: I_forward,n and I_backward,n
- Net current: ΔI_n = I_forward,n - I_backward,n

Typical starting parameters:
- ΔE_step = 2 mV
- ΔE_SW = 25 mV
- f_SWV = 60 Hz (but should be matched to k_ET^0,eff — see below)
- Scan from E_initial = -0.65 V to E_final = +0.65 V vs on-chip RE
- Total steps: (1300 mV) / (2 mV) = 650 points per scan
- Scan time: 650 / 60 Hz ≈ 10.8 s per electrode

## Sub-components to model

### 11.1 Dimensionless kinetic parameter and regime selection

The Osteryoung-O'Dea dimensionless kinetic parameter for surface-confined SWV:
    kappa_SW = k_ET^0 / (f_SWV * n * F / (R * T))
    kappa_SW = k_ET^0 / (f_SWV * n * 38.9)   [at T = 298 K, n = 2 for MB]

Regimes:
- kappa_SW >> 1 (> 10): reversible. ΔI_peak is independent of k_ET^0. Peak height ∝ n^2 * Gamma * ΔE_SW * f_SWV. Peak FWHM = 5.56 RT/(nF) ≈ 65 mV (n=2).
- kappa_SW ≈ 1: quasi-reversible. ΔI_peak depends on both Gamma and k_ET^0. Optimal sensitivity regime.
- kappa_SW << 1 (< 0.1): totally irreversible. Peak height drops significantly; peak broadens and shifts cathodically. Very insensitive regime — avoid.

Target: design for kappa_SW ≈ 0.5-2 (quasi-reversible) to get maximum peak definition. For k_ET^0,eff = 50 s^-1:
    f_SWV = k_ET^0 / (kappa_SW * n * 38.9) = 50 / (1 * 2 * 38.9) ≈ 0.64 Hz

But this is too slow — practical minimum is ~5 Hz. Resolution: use a shorter tether (PEG4 not PEG6) to increase k_ET^0, or use a higher f_SWV and accept being in the quasi-reversible / reversible crossover.

For d_eff = 2 nm (a reasonable estimate from Layer 8): k_ET^0 ≈ 400 s^-1 (from Layer 10.1). Then:
    f_SWV = 400 / (1 * 2 * 38.9) ≈ 5.1 Hz for kappa_SW = 1

Or at f_SWV = 60 Hz: kappa_SW = 400 / (60 * 2 * 38.9) = 400 / 4668 = 0.086 — totally irreversible. This is wrong!

Resolution: the measured k_ET^0 for MB on Plaxco-group surface-tethered probes is typically 1-100 s^-1 (NOT 400 s^-1), because the effective tether length seen by MB is larger than the purely geometric 2 nm due to steric crowding and chain conformation. Literature values: k_ET^0 = 1-50 s^-1 for MB on 10-30 nm DNA hairpin tethers. For a 5 nm PEG-peptide tether: k_ET^0 ≈ 10-100 s^-1. At 60 Hz: kappa_SW = 50 / (60 * 2 * 38.9) ≈ 0.011 — still irreversible.

This is a critical issue. Need to measure k_ET^0 empirically (from peak position shift as a function of f_SWV) and then optimise f_SWV. The correct f_SWV must be determined from the actual probe design, not assumed from literature estimates.

The SWV forward model must be run for a range of (k_ET^0, f_SWV) to find the optimal operating point.

### 11.2 Numerical SWV forward model (Osteryoung-O'Dea)

For surface-confined redox at each SWV step n with potential E_n:

State: Gamma_ox (surface concentration of oxidised MB, i.e., MB^+)
       Gamma_red = Gamma_intact - Gamma_ox

Rate equations (Butler-Volmer):
    dGamma_ox/dt = k_b(eta_n) * Gamma_red - k_f(eta_n) * Gamma_ox

where eta_n = E_n (+- ΔE_SW during half-cycles) - E°'_MB(pH) and k_f, k_b from Layer 10.2.

For each half-cycle (duration Δt = 1/(2*f_SWV)):
    Solve ODE analytically: Gamma_ox(t) = Gamma_ox(0) + Delta_Gamma_ox * (1 - exp(-k_total * t))
    where k_total = k_f + k_b, Delta_Gamma_ox = (k_b * Gamma_intact - k_total * Gamma_ox(0)) / k_total

Current at end of half-cycle:
    I = n * F * A_true * dGamma_ox/dt|_{t=Δt} = n * F * A_true * Delta_Gamma_ox * k_total * exp(-k_total * Δt) / Δt

Net SWV current:
    ΔI_n = I_forward,n - I_backward,n

### 11.3 Capacitive contribution to SWV net current

In SWV, the capacitive current I_cap is ideally cancelled by the net current (forward - backward). But C_dl(E) is not perfectly symmetric:
    I_cap,forward(E_n + ΔE_SW) ≠ I_cap,backward(E_n - ΔE_SW) if C_dl varies with E

Residual capacitive contribution:
    ΔI_cap,n = C_dl(E_n + ΔE_SW) * A_true * ΔE_SW / Δt - C_dl(E_n - ΔE_SW) * A_true * ΔE_SW / Δt
             = A_true * ΔE_SW / Δt * [C_dl(E_n + ΔE_SW) - C_dl(E_n - ΔE_SW)]
             ≈ A_true * ΔE_SW / Δt * dC_dl/dE * 2*ΔE_SW

For typical dC_dl/dE ≈ 0.1 µF/cm^2/V (small), A = 0.013 cm^2, ΔE_SW = 25 mV, Δt = 8.3 ms:
    ΔI_cap,n ≈ 0.013 * 0.025 / 0.0083 * 10^-7 * 0.05 ≈ 1.9 × 10^-12 A ≈ 2 pA

Residual capacitive background ≈ 2 pA per step. Compared to MB faradaic peak ≈ 10-50 nA: negligible (0.004-0.02%). This confirms that SWV is excellent at suppressing capacitive background for surface-confined reactions.

### 11.4 Approximate analytical peak current

For the quasi-reversible case, the peak net SWV current (at E = E°'_MB) is approximately:
    ΔI_peak,MB ≈ n^2 * F^2 * A_true * Gamma_intact * k_ET^0 * ΔE_SW / (4 * R * T)

For n=2, A_true = 0.013 cm^2, Gamma_intact = 5 pmol/cm^2 = 5×10^-12 mol/cm^2, k_ET^0 = 50 s^-1, ΔE_SW = 25 mV, T = 298 K:
    ΔI_peak,MB = (2)^2 * 96485^2 * 0.013 * 5×10^-12 * 50 * 0.025 / (4 * 8.314 * 298)
               = 4 * 9.31×10^9 * 0.013 * 5×10^-12 * 50 * 0.025 / (9905)
               = 4 * 9.31×10^9 * 3.22×10^-15 / 9905
               ≈ 12 × 10^-9 A = 12 nA

Approximately 12 nA peak at Gamma = Gamma_0. After 50% cleavage (F_k = 0.5): ΔI_peak,MB ≈ 6 nA. Signal change from 100% to 50% cleavage: ~6 nA. Compare with noise floor from Layer 14 (≈ 4 pA): SNR = 6 nA / 4 pA = 1500. Excellent signal-to-noise.

### 11.5 Fc reference peak

Fc: n = 1, k_ET^0,Fc ≈ k_ET^0,C6 = k_ET^max * exp(-beta * d_Fc) where d_Fc = C6 length ≈ 0.9 nm → k_ET^0,Fc ~ 10^4-10^5 s^-1 (very fast, reversible at any practical f_SWV).

At f_SWV = 60 Hz for Fc: kappa_SW,Fc = 10^4 / (60 * 1 * 38.9) ≈ 4.3 → quasi-reversible to reversible.

Approximate Fc peak current:
    ΔI_peak,Fc ≈ n_Fc^2 * F^2 * A_true * Gamma_Fc * ΔE_SW / (4 * R * T) * (quasi-reversible factor)
    
For n_Fc=1, Gamma_Fc = 2 pmol/cm^2, ΔE_SW = 25 mV:
    ΔI_peak,Fc ≈ 1 * (96485)^2 * 0.013 * 2×10^-12 * 0.025 / (4 * 8.314 * 298) ≈ 0.6 nA (approximately)

Fc peak is smaller than MB peak (lower n and lower Gamma). But it is stable (Fc is not cleaved) and its stability is the key: Fc peak height = reference. Any change in I_MB/I_Fc must be due to MB changes (cleavage), not Fc changes.

Verify: can both MB and Fc peaks be resolved in a single SWV scan? ΔE° = 0.70 V >> (FWHM_MB/2 + FWHM_Fc/2) ≈ (65 + 131)/2 = 98 mV → yes, well-resolved.

### 11.6 Parameter sweep for optimisation

Run the numerical forward model over the parameter space:
- f_SWV: 10, 20, 30, 60, 100, 200 Hz
- ΔE_SW: 10, 25, 50 mV
- ΔE_step: 1, 2, 5 mV
- k_ET^0: 1, 5, 10, 50, 100, 500 s^-1 (range from literature/measurement)

Compute for each combination:
- ΔI_peak,MB (signal)
- kappa_SW (regime)
- ΔI_cap,net (background)
- SNR = ΔI_peak,MB / sigma_noise (from Layer 14)

Select parameters maximising SNR subject to kappa_SW > 0.1 (not totally irreversible).

### 11.7 Scan direction

Cathodic scan (E starts positive, scans negative): MB is first reduced at -0.25 V, Fc peak would be missed (it's at +0.45 V, reached on the initial positive potential before scanning negative).

Recommended: anodic scan starting from -0.65 V, scanning to +0.65 V. This way:
1. MB reduction peak appears near -0.25 V (going toward more positive E → MB in cathodic half-cycles is favoured going anodic? No — this is confusing).

Actually: SWV staircase direction and peak interpretation:
- For a cathodic reduction: the peak appears at E°_MB when scanning cathodically (E decreasing). In anodic scan, the same cathodic reduction peak appears as a dip, not a peak.
- Standard practice: scan cathodically from +0.65 to -0.65 V to see the MB reduction peak, then Fc oxidation peak would be seen in an anodic scan.

Alternative: run two separate SWV scans (cathodic for MB, anodic for Fc). Or run a single bidirectional scan (scan down then up). Or use a single cathodic scan from +0.65 to -0.65 V and use the cathodic SWV to detect both MB reduction (at -0.25 V) and Fc as a negative peak (if Fc is pre-reduced?).

This needs to be resolved by consulting the specific SWV implementation. For the DStat potentiostat: bidirectional scans are supported. Recommend: single cathodic scan from +0.70 to -0.65 V to capture Fc (reduction of Fc+ formed during incubation) and MB reduction in one pass. The Fc on the reference electrode should be in the Fc+ state initially (oxidised form) if the gold is at open circuit potential (~0 V vs Ag/AgCl, more positive than E°_Fc = +0.45 V — wait, 0 V > +0.45 V? No: in terms of reducing power, lower potential is more cathodic. 0 V vs Ag/AgCl at open circuit is between E°_MB = -0.25 V and E°_Fc = +0.45 V → Fc is in reduced form (Fc) at OCP, MB is in oxidised form (MB+) at OCP.

This needs careful thought. Include in the modelling as a required analysis.

### 11.8 Frequency diagnostic for surface confinement

To confirm that MB signal is surface-confined (not from diffusing MB fragments in solution):
- Surface-confined: ΔI_peak ∝ f_SWV (at fixed kappa_SW >> 1: proportional to f)
  Actually for surface confined: ΔI_peak ∝ n^2 * F^2 * A * Gamma * ΔE_SW (independent of f when reversible, proportional to f in quasi-reversible)
- Diffusion-controlled: ΔI_peak ∝ f^0.5

Plot log(ΔI_peak) vs log(f_SWV): slope 1 = surface, slope 0.5 = diffusion. This diagnostic must be included in the data analysis protocol and reported as a validation of the measurement regime.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| ΔI(E) — predicted SWV trace | Numerical array | Layer 12 (peak extraction) |
| ΔI_peak,MB (at Gamma_0) | ~10-50 nA (design-dependent) | Layer 14 (noise budget), Layer 13 |
| ΔI_peak,Fc | ~0.5-5 nA (stable reference) | Layer 13 |
| Optimal f_SWV | 5-60 Hz | Instrument design spec |
| Optimal ΔE_SW, ΔE_step | 25 mV, 2 mV (starting point) | Instrument design spec |
| Residual capacitive background | ~2 pA | Layer 14 |

## Key literature to find

- Osteryoung SWV theory for surface-confined reactions: Osteryoung & O'Dea (Electroanalytical Chemistry 1986)
- Analytical SWV expressions for quasi-reversible ET: Mirceski, Komorsky-Lovric & Lovric (Square-Wave Voltammetry, Springer 2007) — THE reference for SWV theory
- Experimental k_ET^0 for MB tethers: Plaxco group (various papers); Baker et al. (JACS 2006)
- SWV for aptamer-based biosensors: Lubin & Plaxco (Acc Chem Res 2010)
- DStat potentiostat SWV implementation: Dryden et al. (PLOS ONE 2015)

## Files to create

- `swv_forward.py`: full numerical SWV simulation (BV + half-cycle integration); returns ΔI(E) array
- `swv_analytical.py`: approximate analytical peak current formula; quick estimation
- `parameter_sweep.py`: sweep (f_SWV, ΔE_SW, k_ET^0) and compute SNR; return optimal parameters
- `frequency_diagnostic.py`: plot ΔI_peak vs f_SWV to confirm surface confinement regime
