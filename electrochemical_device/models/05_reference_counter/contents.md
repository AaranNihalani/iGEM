# Layer 5: Reference and counter electrodes

## Purpose

The reference electrode (RE) and counter electrode (CE) are essential components of a three-electrode electrochemical cell. The RE defines the potential scale against which all working electrode (WE) potentials are measured; the CE completes the circuit and carries the faradaic current. Problems with either directly degrade measurement accuracy: RE drift shifts SWV peak positions (corrupting E°_MB measurement and ratiometric calculations); an undersized CE causes current limitation that distorts SWV waveforms.

This layer models RE stability and drift, CE sizing, and the electrochemical environment at each electrode during the SWV scan.

## Sub-components to model

### 5.1 On-chip Ag/AgCl reference electrode

The RE is most practically implemented as an on-chip Ag/AgCl quasi-reference formed by electrochemical or chemical chloridation of a silver film.

Nernst equation for Ag/AgCl:
    E_ref = E°_AgCl/Ag - (RT/F) * ln(a_Cl-)
    E°_AgCl/Ag = +0.222 V vs NHE (25°C)

In wound fluid with [Cl-] = 110-120 mM (a_Cl- ≈ 0.085-0.093 at physiological activity coefficient γ ≈ 0.75):
    E_ref ≈ +0.222 - 0.025 * ln(0.090) ≈ +0.222 + 0.060 ≈ +0.282 V vs NHE
    E_ref ≈ +0.282 - 0.197 = +0.085 V vs saturated Ag/AgCl (3M KCl)

This is an offset from the 'standard' +0.197 V reference used in most literature tables. Peak positions reported in the literature (e.g., E°_MB = -0.25 V vs Ag/AgCl sat.) must be corrected for this offset when using an on-chip quasi-reference in wound fluid.

### 5.2 Reference electrode drift

Three drift mechanisms:

(a) Cl- depletion at the Ag/AgCl surface: as the assay proceeds, Cl- diffuses away from the RE surface into the bulk (or vice versa if wound fluid has higher [Cl-] than the reference chloride loading). Rate: governed by diffusion; for a compact Ag/AgCl deposit, the available Cl- reservoir is finite.

    delta_E_ref = (RT/F) * delta_ln(a_Cl-) ≈ 0.025 * delta_[Cl-] / [Cl-]

For a 5% change in [Cl-] at the RE surface: delta_E_ref ≈ 1.3 mV — tolerable.

(b) Ag/AgCl dissolution: AgCl has Ksp = 1.8 × 10^-10. In wound fluid with [Cl-] = 115 mM: [Ag+] = Ksp/[Cl-] ≈ 1.6 × 10^-9 M. Ag+ release rate is negligible and does not deplete the RE significantly over a 30-min assay.

(c) Light sensitivity: Ag/AgCl is photosensitive and decomposes under UV/visible light. Cartridge must be opaque or measured in the dark. If illuminated: potential drift of 1-10 mV is possible.

Acceptable total RE drift over assay: < 5 mV. This is because a 5 mV shift in E_ref shifts the apparent E°_MB by 5 mV, which at most shifts peak position — the ratiometric calculation uses peak AREAS (not positions), so small peak position shifts have minimal effect on the ratio. But: if the peak drifts by >25 mV (one HWHM), the peak integration window may clip the peak, causing a systematic error in the area estimate.

Model: treat E_ref(t) as a random walk with variance proportional to t. For short assays (<30 min), drift is likely < 2 mV if the RE design is sound.

### 5.3 Counter electrode (CE) sizing

The CE must supply the total current to all WEs during SWV. At peak SWV current (all WEs scanning simultaneously):
    I_CE = sum_i (I_cap,i + I_faradaic,i)
    I_cap,i = 2 * C_dl * A_WE * Delta_E_SW * f_SWV ≈ 60 nA per WE (from Layer 6)
    I_faradaic,i ≈ 10-100 nA per WE (peak)
    Total per WE: ~60-160 nA
    For 8 WEs: I_CE_max ≈ 500-1300 nA

The CE reaction (complement to WE):
- During WE cathodic peak (MB reduction): CE is anodic → oxidises solution species. Dominant reaction: water oxidation (O2 evolution) at +1.23 V vs NHE (+1.0 V vs Ag/AgCl). 
- During WE anodic peak (MB oxidation): CE is cathodic → reduces O2 dissolved in solution, or reduces protons.

Key check: does CE potential exceed O2 evolution potential during the anodic SWV half-cycles?

The WE scans from -0.65 V to +0.65 V vs on-chip RE. During cathodic sweep of WE, the CE current is anodic. CE potential is set by the potentiostat to be whatever is needed to pass the required current. If CE is large (low overpotential), CE potential stays below the oxidation limit. If CE is small (high current density), CE potential rises and gas evolution begins.

Minimum CE area (for CE not limiting):
    A_CE > I_CE_max / (j_CE_max)
where j_CE_max is the maximum practical current density without bubble formation (~1 mA/cm^2 for smooth gold in physiological solution). For I_CE_max = 1.3 µA: A_CE > 1.3 × 10^-4 cm^2. The CE area must be >> WE area to avoid being the limiting factor. A CE that is 10× larger than total WE area is standard practice.

O2 bubble risk: if O2 bubbles form at the CE and migrate to the WE area, they block electrode access and cause sudden current drops. The shallow chamber height (0.25-0.5 mm) means bubbles are highly confined. CE should be positioned away from WEs and ideally at the outlet of the chamber so bubbles exit. Design implication: CE near vent.

### 5.4 Shared vs individual CE

Options:
(a) Single shared CE for all 8 WEs: simple, but if WEs are measured simultaneously (all at same potential simultaneously), CE current is the sum. If WEs are measured sequentially (multiplexed, one at a time), CE current is only from one WE at a time → much smaller → CE can be smaller.

(b) Individual CE per WE pair: adds fabrication complexity but eliminates CE cross-talk between channels. Not necessary for this application.

For sequential SWV measurement (recommended): I_CE ≈ 100-200 nA max → CE sizing is not a practical constraint.

### 5.5 Solution resistance and iR compensation

Between RE and WE: solution resistance R_s = rho_sol * d_RE-WE / A_cross.

For wound fluid (sigma ≈ 15 mS/cm = 0.15 S/m, d_RE-WE ≈ 5 mm, A_cross ≈ 1 cm^2 = 10^-4 m^2):
    R_s = (1/0.15) * 5×10^-3 / 10^-4 = 6.67 * 0.05 = 0.333 Ohm/m * 0.05 m = ... 

Actually: R_s = rho_sol * L / A = (1/sigma) * L / A = (1/0.15) * 5×10^-3 / (5×10^-3 * 1×10^-3)
rho = 1/sigma = 6.67 Ohm·m. Wait, units: sigma = 0.015 S/cm = 0.15 S/m. rho = 1/0.15 = 6.67 Ohm·m.
For a flat cell: R_s = rho * L / A where L = path length (RE to WE) = ~5 mm = 0.005 m, A = cross-sectional area of current path = 5 mm × 0.375 mm = 1.875 × 10^-6 m^2.
R_s = 6.67 * 0.005 / (1.875 × 10^-6) ≈ 17,787 Ohm.

Wait, that seems very high. Let me reconsider. In a thin-layer cell, the resistance is very high because A is small. But in practice, the electrode is planar and the RE is also planar and nearby. For a parallel plate geometry (WE and RE facing each other or side-by-side in a thin channel): if the RE is embedded in the floor/side of the same thin channel at distance d_RE-WE = 2 mm away, and current flows along the channel (cross-section 5 mm × 0.375 mm = 1.875 × 10^-6 m^2):
    R_s = rho * d / A = 6.67 * 2×10^-3 / 1.875×10^-6 ≈ 7111 Ohm

Very high! But check: for SWV peak current ~100 nA, iR drop = 100×10^-9 * 7111 = 0.71 mV — still negligible (< 1 mV) compared to SWV parameters (ΔE_step = 2 mV, FWHM = 65 mV). Not a problem.

However, for capacitive current 60 nA: iR = 0.43 mV — still fine.

This is a thin-layer cell regime where R_s is high but currents are very small, so the iR drop remains negligible. Record this analysis and the result (iR drop < 1 mV in all expected cases) as a justification for not requiring iR compensation in the instrument.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| E_ref offset from sat Ag/AgCl | +0.085 V correction needed | Layer 10, 11 (potential axis) |
| Max RE drift over assay | < 2-5 mV | Layer 12 (peak position tolerance) |
| CE area requirement | > 10 × total WE area | Hardware design spec |
| R_s | ~7 kOhm (but iR < 1 mV) | Layer 6, 14 |

## Key literature to find

- Ag/AgCl quasi-reference electrodes for microfluidic systems: Polk et al. (J Electroanal Chem)
- Ag/AgCl stability in biological fluids: Polk et al., or White & Plaxco review
- CE design in microfluidic electrochemistry: general electrochemistry textbooks (Bard & Faulkner Chapter 15)

## Files to create

- `reference_electrode.py`: E_ref(t) drift model; [Ag+] release rate; potential correction table
- `counter_electrode.py`: CE sizing model; gas evolution threshold check
