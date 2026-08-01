# Layer 14: Electronics and potentiostat noise budget

## Purpose

This layer computes the fundamental noise floor of the electrochemical measurement. It identifies every source of current noise in the signal chain from the electrode to the digitised output, sums them in quadrature, and determines: (a) the minimum detectable SWV peak area (sigma_area), (b) the resulting sigma_R and sigma_F, and (c) whether the LOD is electronics-limited or chemistry-limited (Da, Gamma, kcat/Km).

The noise budget is also an engineering design tool: it tells us which components must be improved to reach a lower LOD. If the noise is Johnson-limited, a larger R_f reduces noise; if it is ADC-limited, a higher-resolution ADC helps; if it is fundamentally shot-noise-limited, only reducing the DC background current (by lowering C_dl) helps.

## DStat potentiostat architecture

DStat (Dryden et al., PLOS ONE 2015) is a USB-connected open-source potentiostat:
- Microcontroller: PIC or similar
- DAC: 16-bit, ±2.5 V range → step resolution 76 µV (for 5 V range) → adequate for SWV steps of 2 mV
- ADC: 24-bit sigma-delta → LSB at 1 µA range = 60 fA → not the limiting noise
- TIA: OPA2333 or similar low-noise FET op-amp
- Selectable R_f: typically 100 kΩ, 1 MΩ, 10 MΩ (user-selectable)
- USB bandwidth: limited but adequate for SWV (data rate << USB bandwidth)

## Noise sources (all in A/√Hz referred to input)

### 14.1 Johnson (thermal) noise in R_f

    i_J = sqrt(4 * k_B * T / R_f)

For R_f = 100 kΩ, T = 298 K:
    i_J = sqrt(4 * 1.38e-23 * 298 / 1e5) = sqrt(1.64e-25) = 4.06 × 10^-13 A/√Hz = 0.41 pA/√Hz

For R_f = 1 MΩ: i_J = 0.13 pA/√Hz (lower noise but lower bandwidth)
For R_f = 10 MΩ: i_J = 0.041 pA/√Hz

Choosing R_f: must balance noise (prefer high R_f) against bandwidth. Bandwidth of TIA: BW = 1/(2π*R_f*C_f) where C_f is the feedback capacitor (typically 1-10 pF for stability). For R_f = 100 kΩ, C_f = 10 pF: BW = 1/(2π * 10^5 * 10^-11) = 159 kHz >> f_SWV = 60 Hz → fine. For R_f = 1 MΩ: BW = 15.9 kHz >> 60 Hz → still fine.

Use R_f = 1 MΩ to reduce Johnson noise by 3.1× vs 100 kΩ, while retaining adequate bandwidth.

### 14.2 Op-amp input current noise

For FET-input op-amps (OPA2333, AD8628): input current noise i_n,amp ≈ 5-20 fA/√Hz. Negligible compared to Johnson noise.

### 14.3 Shot noise from DC background

During SWV, the electrode capacitive charging current I_cap ≈ 60-80 nA (Layer 11) is the dominant DC component:
    i_shot = sqrt(2 * e * I_cap * BW)

For I_cap = 70 nA, BW = 100 Hz:
    i_shot = sqrt(2 * 1.6e-19 * 70e-9 * 100) = sqrt(2.24e-24) = 1.49 × 10^-12 A = 1.49 pA (over full BW)

Shot noise is a few pA total, similar to or less than Johnson noise. Both are small.

### 14.4 ADC quantisation noise

For 24-bit ADC with input range ±V_ref = ±2.5 V and R_f = 100 kΩ → current range ±25 µA:
    LSB = 50 µA / 2^24 = 3.0 × 10^-9 A = 3 nA

This is large! But sigma-delta ADCs have effective bits (ENOB) of typically 16-22 bits at low frequencies. At 100 Hz measurement rate, ENOB ≈ 20 bits → effective LSB ≈ 50 µA / 2^20 ≈ 48 nA.

At R_f = 1 MΩ → current range ±2.5 µA → effective LSB ≈ 2.5 µA / 2^20 = 2.4 pA at 20 ENOB. This matches the Johnson noise level.

Lesson: scale R_f to match the expected signal range to the ADC dynamic range.

### 14.5 1/f (flicker) noise

Op-amps have 1/f noise below their corner frequency f_c:
    i_flicker = sqrt(i_n_white^2 * f_c / f)   for f < f_c

For OPA2333: f_c ≈ 0.1 Hz. At SWV frequencies (5-100 Hz >> f_c): 1/f noise is negligible.

For the SWV net current (forward - backward): 1/f noise on the forward half-cycle partially cancels in the backward (correlated noise), further reducing 1/f contribution to ΔI.

### 14.6 Mains (50/60 Hz) interference

If f_SWV = 60 Hz (US mains) or f_SWV = 50 Hz (EU mains), mains pickup at the same frequency couples directly into the SWV waveform. This is a serious systematic noise source for unshielded measurements.

Mitigation options:
1. Faraday cage around the measurement cell
2. Choose f_SWV ≠ mains frequency: use f_SWV = 45 Hz (midway between 40 and 50) or 80 Hz
3. Synchronise data acquisition to mains: if the ADC samples at exact mains multiples, mains noise averages out
4. Power line filter: common-mode choke on the USB cable

Quantitative: mains pickup amplitude depends on shielding. Without shielding: 1-100 nA possible (depends on loop area). With Faraday cage: < 0.1 nA. At SWV amplitudes of 10-50 nA for the MB peak, unshielded mains pickup is catastrophic; shielding is non-negotiable.

### 14.7 Multiplexer settling time and charge injection

When the multiplexer switches from WE i to WE j:
1. Charge injection: switching charge (channel charge of FET switch) is injected into the input → transient current spike
2. Settling time: the WE j's double-layer capacitance must charge to the new potential

Settling time τ_settle = (R_mux + R_s) * C_dl * A_WE ≈ (100 + 7000) * 2e-9 = 14 µs

Wait for at least 10τ_settle = 140 µs after switching before taking the first SWV step. At 60 Hz SWV (period = 16.7 ms): 140 µs << 16.7 ms → settling is fast enough.

Charge injection from multiplexer (ADG1606 or similar): Q_inject ~ 10-100 pC. This appears as a current spike with duration ~1 µs: I_spike ~ 10 pC / 1 µs = 10 mA (huge, but very brief). The TIA responds to this as a brief saturation event. Recovery time of TIA after injection: ~ 1-10 µs. Must ensure first SWV step is at least 100 µs after switch.

### 14.8 Total noise over SWV peak

For the SWV peak, current is sampled at N_peak points across the peak (width ~2× FWHM = 65 mV → 65 points at 1 mV resolution, or 33 points at 2 mV):

Per-point total noise (at f_SWV = 60 Hz, BW_per_point ~ f_SWV/2 = 30 Hz):
    i_total = sqrt(i_J^2 + i_shot^2 + i_ADC^2) ≈ sqrt((0.41)^2 + 0.15^2 + ~0) pA/√Hz × √30 Hz ≈ 2.3 pA

Over N_peak = 33 points at 2 mV step (integrating peak area):
    sigma_area = i_total * sqrt(N_peak) * ΔE_step ≈ 2.3e-12 * sqrt(33) * 2e-3 = 2.65e-14 A·V

For MB peak area A_area ≈ 12 nA × 0.027 V × sqrt(2π) ≈ 8.1e-10 A·V (from Gaussian: area = A*sigma*sqrt(2pi)):
    SNR = A_area / sigma_area = 8.1e-10 / 2.65e-14 ≈ 30,600

Excellent SNR. sigma_A / A_area ≈ 0.003% → sigma_R/R ≈ 0.004% → sigma_F ≈ 0.006% → F_LOD ≈ 0.02%.

This is better than the 0.6% LOD estimated in Layer 13 using simpler noise estimates. The difference arises because the Gaussian fitting extracts more information from the peak than a simple peak-height measurement. Updating: the LOD is likely electronics-limited only at much lower protease concentrations than the clinical range. The dominant LOD limitation is probably from pre-analytical variability (Layer 1, sigma_V_abs ≈ 50%) rather than electronics.

### 14.9 Noise summary table

| Noise source | i_n (pA/√Hz) | Notes |
|---|---|---|
| Johnson in R_f = 1 MΩ | 0.13 | Dominant |
| Op-amp current noise | 0.005 | Negligible |
| Shot noise (I_cap = 70 nA) | ~0.025 at 30 Hz | Small |
| ADC (at 1 MΩ, 20 ENOB) | ~0.0024 | Negligible |
| Mains (unshielded) | 1-100 | Catastrophic without shielding |
| Mains (with Faraday cage) | < 0.1 | Acceptable |
| Multiplexer charge injection | spike, clears in 100 µs | Ignore after settling |

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| i_total per point | ~0.3 pA/√Hz (shielded, R_f=1MΩ) | Layer 13 (sigma_F via sigma_area) |
| sigma_area (Gaussian fit) | ~2-5 fA·V | Layer 13 |
| SNR for MB peak at Gamma_0 | ~30,000 | Confirms LOD is not electronics-limited |
| Mains shielding requirement | Faraday cage mandatory | Hardware design spec |
| R_f recommendation | 1 MΩ (balance noise vs range) | Electronics design spec |
| Multiplexer settling requirement | Wait > 100 µs after switch | Firmware timing spec |

## Key literature to find

- DStat potentiostat: Dryden et al. (PLOS ONE 2015) — must read; describes exact architecture
- TIA noise analysis: Horowitz & Hill (Art of Electronics, Chapter 8)
- Shot noise in electrochemistry: Bard & Faulkner (Appendix A)
- Mains rejection in electrochemistry: Analytical Chemistry noise reduction reviews
- Multiplexer charge injection: ADG1606 datasheet (Analog Devices)

## Files to create

- `noise_budget.py`: compute i_J, i_shot, i_ADC for given R_f, I_cap, BW; sum in quadrature
- `sigma_area.py`: propagate per-point noise through Gaussian peak integration to get sigma_area
- `snr_vs_gamma.py`: SNR as function of Gamma_intact; determine Gamma threshold for LOD
- `mains_spectrum.py`: model mains interference amplitude for different shielding scenarios
