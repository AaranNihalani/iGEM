# Layer 10: Marcus-theory electron transfer

## Purpose

The SWV signal comes from the electrochemical redox of methylene blue (MB) tethered to the electrode surface. The rate of this electron transfer (ET) determines the SWV peak current, peak position, and peak shape. Marcus theory provides the physical framework. This layer converts the probe conformational state (Layer 8, d_eff) and the electrochemical potential (from the SWV waveform, Layer 11) into the forward and backward ET rate constants that drive the SWV current.

This is one of the two most critical modelling layers (along with Layer 11), because everything observable in the device output is ultimately determined by k_ET.

## Sub-components to model

### 10.1 Standard ET rate (Marcus outer-sphere)

Marcus theory for heterogeneous electron transfer between MB and the gold electrode:

    k_ET^0(d) = k_ET^max * exp(-beta * d) * exp(-lambda / (4 * k_B * T))

where:
- k_ET^max: pre-exponential factor ~ A * exp(-lambda_inner / (4kT)), where A ~ 10^13 s^-1 and lambda_inner ~ 0.1 eV for MB (inner-sphere reorganisation energy, from bond length changes in MB upon reduction)
- beta: tunnelling decay constant through the tether material. For flexible organic tethers (PEG, peptide): beta ≈ 0.9-1.1 Å^-1 (per Angstrom). Literature: Smalley et al. (JACS 1995) found beta = 1.08 Å^-1 for alkyl chains; PEG is slightly lower due to through-bond vs through-space.
- d: MB-to-gold tunnelling distance (effective value <d_eff> from Layer 8)
- lambda: total (outer + inner) reorganisation energy for MB. Literature: lambda ~ 0.3-0.5 eV for MB on gold from time-resolved spectroelectrochemistry. Outer sphere component: lambda_out = (Delta_e)^2/(8pi*epsilon_0) * (1/r_MB - 1/(2d)) * (1/epsilon_opt - 1/epsilon_s), typically ~ 0.2-0.4 eV.

Full expression with literature values:
    k_ET^0 ≈ 10^13 * exp(-beta * d [in Å]) * exp(-0.4 eV / (4 * 0.026 eV)) s^-1
    k_ET^0 ≈ 10^13 * exp(-beta * d) * exp(-3.85)
    k_ET^0 ≈ 2 × 10^11 * exp(-beta * d) s^-1

For d = 3 nm = 30 Å, beta = 1.0 Å^-1:
    k_ET^0 ≈ 2 × 10^11 * exp(-30) ≈ 2 × 10^11 * 9.4 × 10^-14 ≈ 19 s^-1

This is in the SWV-accessible range: SWV peak condition is k_ET^0 ≈ pi * f_SWV (quasi-reversible). For f_SWV = 60 Hz: pi * f ≈ 188 s^-1. So f_SWV should be lower (20-30 Hz) to match k_ET^0 = 19 s^-1 for this tether length.

For d = 2 nm = 20 Å: k_ET^0 ≈ 2 × 10^11 * exp(-20) = 2 × 10^11 * 2 × 10^-9 ≈ 400 s^-1. At f_SWV = 100 Hz: pi * f ≈ 314 s^-1 — quasi-reversible. Good match.

Conclusion: d_eff ≈ 2 nm from conformational dynamics (shorter than the contour length L_c ≈ 5-6 nm due to chain flexibility) gives k_ET^0 in a convenient range for SWV at 30-100 Hz. Must compute d_eff from Layer 8 to confirm.

### 10.2 Butler-Volmer potential dependence

The standard Marcus rate k_ET^0 is the rate at zero overpotential (E = E°_MB). At finite overpotential eta = E - E°_MB:

Butler-Volmer (valid when |eta| << lambda/2F ≈ 0.2 V, i.e. not deep in Marcus inverted region):
    k_f(eta) = k_ET^0 * exp(-alpha * F * eta / (R * T))   (cathodic, reduces MB^+ → MB_red)
    k_b(eta) = k_ET^0 * exp((1 - alpha) * F * eta / (R * T))  (anodic, oxidises MB_red → MB^+)

where alpha ≈ 0.5 (symmetric transfer coefficient), F = 96485 C/mol, R = 8.314 J/mol/K.

At T = 310 K: F/RT = 37.7 V^-1. At alpha = 0.5: k_f/k_ET^0 = exp(-18.85 * eta).

For eta = -0.1 V (cathodic, 100 mV below E°_MB): k_f = k_ET^0 * exp(1.885) = 6.6 × k_ET^0.

Full Marcus expression (for large overpotentials, beyond Butler-Volmer):
    k_f(eta) = k_ET^0 * exp(-lambda/(4kT)) * exp(-(lambda + F*eta)^2 / (4*lambda*kT))

This properly captures the Marcus inverted region (k_f decreasing at very large negative eta). For the SWV range (-0.65 to +0.65 V) and lambda ≈ 0.4 eV: the inverted region starts at |eta| > lambda/2 ≈ 0.2 eV. Within the SWV scan, the rate is not in the inverted region for most of the peak. But at the edges of the scan (E = -0.65 V, eta = -0.4 V for E°_MB = -0.25 V): eta approaches lambda/2 → full Marcus expression is needed.

### 10.3 Formal potential E°_MB and pH correction

MB is a two-electron, two-proton (2e-/2H+) redox couple:
    MB+ + 2H+ + 2e- ⇌ leucomethylene blue (LMB, colourless)

Formal potential:
    E°'_MB(pH) = E°_MB - (n_H/n_e) * (RT/F) * 2.303 * pH

where n_H = 2, n_e = 2, so n_H/n_e = 1:
    E°'_MB(pH) = E°_MB - 0.059 * (pH - 7.0)   (in V, at 25°C)

Literature: E°_MB at pH 7 ≈ -0.011 V vs NHE ≈ -0.208 V vs Ag/AgCl (sat.) ≈ -0.208 + 0.060 V vs on-chip RE in wound fluid.

Important: the exact E° depends on which MB form is measured. MB has three redox states (MB^2+, MB^+, MB^0 / LMB). The relevant couple for SWV is MB^+ / LMB (the semi-oxidised to fully reduced step). Literature values range from -0.18 to -0.29 V vs Ag/AgCl depending on pH and electrode material. Use -0.25 V vs Ag/AgCl (sat.) as a starting value; correct for on-chip RE offset (Layer 5).

pH correction across wound pH range:
    At pH 6.5: E°'_MB = -0.25 + 0.059 * (7.0 - 6.5) = -0.25 + 0.030 = -0.220 V vs Ag/AgCl
    At pH 8.0: E°'_MB = -0.25 - 0.059 * (8.0 - 7.0) = -0.25 - 0.059 = -0.309 V vs Ag/AgCl

Peak shifts by 89 mV across the pH range. Must either: (a) measure pH and apply E° correction, or (b) use peak area (integrated over a wide window) rather than peak height at a fixed E, which is less sensitive to peak position.

### 10.4 Fc reference formal potential and pH independence

Ferrocene (Fc) / ferricinium (Fc+) couple:
    Fc ⇌ Fc+ + e-   (1-electron, 0-proton process)
    E°_Fc = +0.45 V vs Ag/AgCl (sat.) approximately; pH-independent

The pH independence of Fc is a key advantage: the Fc reference peak position is stable regardless of wound pH. The MB-to-Fc peak separation ΔE° = E°_Fc - E°'_MB(pH) is pH-dependent:
    ΔE°(pH 7.0) = 0.45 - (-0.25) = 0.70 V
    ΔE°(pH 6.5) = 0.45 - (-0.22) = 0.67 V
    ΔE°(pH 8.0) = 0.45 - (-0.31) = 0.76 V

The separation varies by ~90 mV across the pH range — peaks remain well-resolved (FWHM ~65 mV for MB, 131 mV for Fc) but the midpoint shifts. The Gaussian fitting in Layer 12 must use floating peak positions to accommodate this.

### 10.5 Ca2+ artefact correction to d_eff

From Layer 7.5: Ca2+ chelation by PEG contracts the tether by delta_d_Ca = delta_d_max * [Ca2+]/(Kd + [Ca2+]).

Modified k_ET in presence of Ca2+:
    k_ET^0,Ca = k_ET^0(d_0 - delta_d_Ca) = k_ET^0(d_0) * exp(beta * delta_d_Ca)

This shifts the SWV peak height (more intense MB peak at higher [Ca2+]) but not the peak position. Since this effect is identical on active and control electrodes (same PEG spacer), it cancels in F_k. However, if the MB peak height changes while the Fc peak is unchanged, the ratiometric R = I_MB/I_Fc changes. This is exactly the quantity used in rho = R(T)/R(0) — if Ca2+ concentration changes between t=0 and t=T, or between active and control, the ratio is perturbed.

Mitigation: standardise the assay buffer to contain a defined [Ca2+] (e.g., add CaCl2 to elution buffer to fix [Ca2+] = 1 mM). Then delta_d_Ca is a known constant that affects all electrodes identically.

### 10.6 Temperature correction to k_ET^0

Marcus: k_ET^0 ∝ T^0.5 * exp(-lambda/(4kT)).
    k_ET^0(T) = k_ET^0(T_ref) * (T/T_ref)^0.5 * exp(-lambda/(4k) * (1/T - 1/T_ref))

For T_ref = 298 K, T = 310 K (37°C), lambda = 0.4 eV:
    Ratio = (310/298)^0.5 * exp(-0.4 * 11605/(4) * (1/310 - 1/298))
           = 1.020 * exp(-1161 * (-1.30×10^-4))
           = 1.020 * exp(0.151)
           = 1.020 * 1.163 = 1.186

About 19% increase in k_ET^0 from 25°C to 37°C. This shifts the SWV peak shape and height slightly (through kappa_SW). Effect on ratiometric F_k: small, since both active and control electrodes see the same temperature change. But if temperature drifts between baseline measurement and endpoint measurement, there is a systematic error.

### 10.7 Electrostatic bending correction for IPRTLTAA

From Layer 8.7: Arg+ in IPRTLTAA at cathodic potentials (E < PZC ≈ -0.05 V vs Ag/AgCl) is attracted toward the negatively polarised electrode. Effective d change:
    delta_d_elec(E) = -q_Arg * phi(z_Arg) / k_BT * l_persistent

where phi(z_Arg) is the electrostatic potential at the Arg position in the diffuse double layer, z_Arg ≈ h_cut + additional extension from Arg. Since the Arg is at P2 (second from cleavage, i.e., below the cleavage site in the probe sequence), z_Arg from the gold surface:
    z_Arg ≈ L(Cys) + L(PEG4) + L(IP) = 0.5 + 1.44 + 2 × 0.35 = 2.64 nm from Au

At 2.64 nm from Au, with Debye length lambda_D = 0.77 nm, the diffuse layer potential:
    phi(2.64 nm) = phi_0 * exp(-2.64/0.77) = phi_0 * exp(-3.43) ≈ phi_0 * 0.032

For phi_0 = -50 mV (reasonable at E = -0.25 V vs PZC = -0.05 V, overpotential = -0.2 V):
    phi(z_Arg) ≈ -1.6 mV

The electrostatic force on Arg+ (charge +e): F = +e * grad(phi) ≈ +e * 1.6 mV / 0.77 nm = 0.33 pN. This is tiny compared to thermal forces (k_BT/r_g ≈ 4 pN for a 5 nm probe). The electrostatic bending is a second-order effect for IPRTLTAA and can be included as a perturbative correction but is not dominant.

### 10.8 Combining all corrections into effective k_ET

The effective k_ET^0 passed to the SWV model (Layer 11):
    k_ET^0,eff = <k_ET^0> * f_Ca * f_T * f_elec

where:
- <k_ET^0>: average over P(d) distribution from Layer 8 (dominant term)
- f_Ca = exp(beta * delta_d_Ca) from Section 10.5
- f_T = (T/T_ref)^0.5 * exp(...) from Section 10.6
- f_elec = exp(beta * delta_d_elec(E)) from Section 10.7 — potential-dependent, incorporated into BV equations

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| k_ET^0,eff | ~10-500 s^-1 (d-dependent) | Layer 11 (SWV forward model) |
| Optimal f_SWV | k_ET^0,eff / pi ≈ 3-160 Hz | Layer 11 (parameter optimisation) |
| E°'_MB(pH) | pH-corrected formal potential | Layer 11, 12 |
| Peak separation ΔE°(pH) | 0.67-0.76 V | Layer 11, 12 |
| Ca2+ effect (k_ET^0 ratio) | 1-3× | Layer 13 (error term) |

## Key literature to find

- Marcus theory for heterogeneous ET: Marcus (J Chem Phys, 1965); comprehensive review by Schmickler
- Beta decay constants: Smalley et al. (JACS 1995); Sikes et al. (Science 2001); Chidsey (Science 1991)
- MB reorganisation energy: Pheeney & Barton (PNAS 2012); Kelley et al. (Science 1997)
- MB formal potential pH dependence: Ozoemena & Stefan (Electroanalysis 2005); literature compilation
- Butler-Volmer for surface ET: Bard & Faulkner, Chapter 3 (textbook)
- Fc formal potential and pH independence: standard electrochemistry reference values

## Files to create

- `marcus_et.py`: k_ET^0(d, T, lambda, beta) computation; BV rate constants k_f(eta), k_b(eta)
- `formal_potential.py`: E°_MB(pH, T, RE_offset); E°_Fc; peak separation ΔE°(pH)
- `corrections.py`: Ca2+ artefact f_Ca; temperature factor f_T; electrostatic correction f_elec(E)
- `effective_ket.py`: combine <k_ET> from Layer 8 P(d) with all corrections; return k_ET^0,eff
