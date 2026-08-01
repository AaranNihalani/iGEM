# Layer 0: Wound biology — system input

## Purpose

This layer defines the input state of the system: the concentrations and activities of MMP-9 and HNE in wound fluid at the time of sample collection, along with the physicochemical properties of the wound environment. It is the upstream boundary condition for all downstream modelling layers. No device physics is modelled here; this layer establishes what the device is trying to measure.

The outputs of this layer feed directly into Layer 1 (sample collection) and Layer 6 (double layer: pH and ionic strength determine Debye length).

## Sub-components to model

### 0.1 Protease activation ODE system

Pro-MMP-9 is secreted by neutrophils and macrophages in a zymogen form and activated extracellularly by plasmin. TIMP-1 is the primary endogenous inhibitor. HNE is released from azurophilic granules of degranulating neutrophils and inhibited by alpha-1 antitrypsin (A1AT).

```
d[proMMP-9]/dt = k_syn - k_act * [plasmin] * [proMMP-9]
d[MMP-9]/dt   = k_act * [plasmin] * [proMMP-9] - k_TIMP * [TIMP-1] * [MMP-9]
d[HNE]/dt     = k_rel - k_A1AT * [A1AT] * [HNE]
```

Parameters to source from literature: k_act, k_TIMP, k_rel, k_A1AT, baseline inhibitor concentrations [TIMP-1]_0, [A1AT]_0.

Wound state determines initial conditions. Acute wound: high [plasmin], low [TIMP-1], high [HNE_rel]. Chronic wound: lower plasmin, elevated TIMP-1, sustained but lower HNE.

Output: [MMP-9]_active(t) and [HNE]_active(t) as functions of wound state and time since injury / debridement. These are the quantities the device is designed to quantify.

### 0.2 pH of wound fluid

Range: 6.5-8.2 (acute healing wound: ~7.4-8.2; chronic, infected wound: ~6.5-7.0).

Why it matters for the device:
- Enzyme kcat has pH-dependent activity profiles (bell-shaped, with optima near neutral for MMP-9 and slightly acidic for HNE).
- MB formal potential: E°_MB shifts -59 mV per pH unit (2e-/2H+ redox couple). Across pH 6.5-8.2, E°_MB shifts by ~100 mV. If not corrected, this shifts the SWV peak position and distorts ratiometric calculations.
- Debye screening length depends on ionic strength (pH affects buffering but not I directly at physiological conditions).
- Local pH gradient near electrode during SWV: MB reduction/oxidation consumes/produces protons. In poorly buffered wound fluid, local pH at the electrode surface may differ from bulk by ~0.2-0.5 pH units during the scan.

Action: must measure or estimate wound pH as part of calibration; include pH correction to E°_MB in Layer 10 and Layer 12.

### 0.3 Ionic strength and Debye length

Wound fluid ionic strength I ≈ 130-160 mM (NaCl equivalent). Debye length:

    lambda_D = sqrt(epsilon * epsilon_0 * R * T / (2 * F^2 * I))

At I = 150 mM, T = 310 K: lambda_D ≈ 0.77 nm.

This is much shorter than the probe tether length (~5 nm), meaning probe charges (e.g., Arg in IPRTLTAA) are fully exposed in the diffuse layer and subject to the full electrostatic potential of the electrode. This affects the effective local concentration of enzyme near charged probes and may distort kcat/Km measurements made at different electrode potentials.

### 0.4 Temperature

Operating range: 20-37°C (room temperature to body temperature).

Temperature affects:
- Diffusion coefficients D ∝ T/η (Stokes-Einstein)
- kcat via Arrhenius: kcat(T) = kcat(T_ref) * exp(-Ea/R * (1/T - 1/T_ref))
- k_ET via Marcus theory: k_ET^0 ∝ T^0.5 * exp(-lambda/(4kT))
- MB formal potential: E°_MB has a temperature coefficient (~-1 mV/°C)
- Solution viscosity η (affects D and capillary fill)

Must parameterise temperature as an explicit variable and assess sensitivity of F_k to a ±5°C uncertainty in temperature.

### 0.5 Oxidative species

Wound fluid contains H2O2 (from neutrophil oxidative burst, 10-100 µM), hypochlorite (HOCl, from myeloperoxidase). These can:
- Oxidise the Cys-Au thiol bond of the probe → irreversible SAM degradation during incubation
- Oxidise MB irreversibly (MB can be bleached to leucomethylene blue or further products)

Model: first-order Cys oxidation rate k_ox * [ROS]. Fraction of probe lost to oxidation during incubation time T: delta_Gamma_ox = Gamma_0 * (1 - exp(-k_ox * [H2O2] * T)). This adds a systematic negative interference on F_k that cannot be cancelled by the ratiometric design (both active and control probes are oxidised equally, so it cancels in F_k — actually this DOES cancel). Verify this.

### 0.6 Matrix fouling proteins

Wound fluid contains albumin (~10-40 mg/mL in chronic wounds), fibrinogen, fibrin degradation products, IgG, and cellular debris. Non-specific adsorption (NSA) of these proteins to gold:
- Competes with SAM diluent for exposed gold patches
- Increases capacitive background current
- Can non-specifically block enzyme access to probe

NSA is partially prevented by PEG diluent (MCH + PEG-SH). Must estimate: fraction of gold blocked by NSA at relevant protein concentrations and incubation times. Use literature adsorption isotherms for albumin on gold-PEG surfaces.

### 0.7 Interfering proteases

Wound fluid contains many proteases beyond MMP-9 and HNE:
- Cathepsin G (serine protease, degranulation product): cleaves after F, Y; similar to HNE
- Cathepsin B, D (cysteine/aspartyl protease): less likely to cleave AAPVSGIG or IPRTLTAA
- Plasmin: cleaves after K, R; IPRTLTAA has R at position 3 (P6-P1 = I, P, R, T, L, T → cleavage at R possible?)
- Urokinase: similar specificity to plasmin
- MMP-2, MMP-8: closely related to MMP-9; may cross-react with IPRTLTAA

Must use MEROPS cleavage data and the existing substrate predictor to score AAPVSGIG and IPRTLTAA against each of these interfering proteases. Any protease with predicted LR > 0.3 is a potential interferent and must be included as an additive cleavage term in Layer 9.

## Outputs to pass downstream

| Output | Receives | Layer |
|---|---|---|
| [MMP-9]_active | Wound biology ODE | Layer 1 |
| [HNE]_active | Wound biology ODE | Layer 1 |
| pH | Literature / measured | Layer 6, 10, 12 |
| I (ionic strength) | Literature | Layer 6 |
| T (temperature) | Measured or assumed | All layers |
| [Ca2+] | Literature wound fluid | Layer 7, 10 |
| [interferent protease concentrations] | Literature / ODE | Layer 9 |

## Key literature to find

- Wound fluid protease concentrations: Westerhof et al., Trengove et al., Eming et al.
- MMP-9 activation kinetics by plasmin: Murphy et al. (Matrix Biology)
- TIMP-1 / MMP-9 inhibition constant: Olson et al.
- A1AT / HNE inhibition constant: Beatty et al.
- Wound fluid pH measurements: Gethin et al. (J Wound Care)
- Wound fluid ionic composition: Lamke et al.
- Interferent protease levels: Yager et al.

## Files to create

- `wound_ode.py`: ODE system solver, returns time-course of [MMP-9]_active and [HNE]_active
- `parameters.json`: all Layer 0 parameters with literature values and citations
