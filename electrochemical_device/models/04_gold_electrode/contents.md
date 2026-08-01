# Layer 4: Gold electrode surface

## Purpose

This layer characterises the physical properties of the gold working electrode (WE) surface that are prerequisites for all electrochemical modelling. It is primarily a characterisation layer rather than a dynamic simulation: it establishes the true electrode area, ohmic resistances, and surface state that all subsequent electrochemical layers depend on.

Gold electrode surface quality is one of the hardest parameters to control in a disposable cartridge. Batch-to-batch variation in roughness, contamination, and film thickness directly propagates into uncertainty in Gamma (surface coverage) estimates, which is likely the dominant source of inter-electrode variability. The ratiometric design partially corrects for this (by normalising each electrode to its own baseline), but understanding the variation is essential.

## Sub-components to model

### 4.1 Gold film deposition and roughness factor

Evaporated gold: thermally or e-beam evaporated Au on Ti/Cr adhesion layer, on glass or PET/PETG substrate. Film thickness: typically 50-200 nm.

Electrochemical roughness factor:
    R_f = Q_oxide_stripping / (Q_ref * A_geo)

where Q_ref = 390 µC/cm² (charge density for stripping a monolayer of Au oxide, standard value from literature), A_geo is the geometric (mask-defined) electrode area, Q_oxide_stripping is measured by cyclic voltammetry in 0.5 M H2SO4.

For smooth evaporated gold: R_f = 1.1-1.5. For annealed or polycrystalline gold: R_f = 1.5-3.0. For rough gold (electrodeposited nanostructures): R_f >> 3.

Why it matters: All surface coverage densities Gamma in mol/cm^2 are per cm^2_TRUE, not cm^2_geometric. For R_f = 1.3, a measured Q_MB gives Gamma_geo = Q_MB/(nF*A_geo) and Gamma_true = Gamma_geo / R_f = Gamma_geo / 1.3. If R_f is not measured and is incorrectly assumed to be 1.0, Gamma is overestimated by 30%.

Action: must measure R_f by cyclic voltammetry for each cartridge batch; include R_f as a calibration parameter in the model.

### 4.2 Gold film electrical resistance

The gold film between the bonding pad and the active electrode area contributes an ohmic resistance:
    R_Au = rho_Au * L_trace / (w * t)

where rho_Au = 2.44 × 10^-8 Ohm·m (bulk gold resistivity), L_trace = trace length, w = width, t = film thickness.

Example: L = 5 mm, w = 0.5 mm, t = 100 nm:
    R_Au = 2.44×10^-8 × 5×10^-3 / (5×10^-4 × 1×10^-7) = 2.44 Ohm

This adds to the uncompensated resistance R_u = R_Au + R_contact + R_s. For SWV with peak currents ~10-100 nA: iR = 100 nA × 2.44 Ohm = 0.24 µV peak shift — negligible. For higher currents (if electrode area is enlarged): recheck.

### 4.3 Contact resistance

Electrical connection from the PCB/cartridge to the potentiostat occurs via:
- Spring-pin pogo contacts: R_contact ~ 10-100 mOhm typically
- Anisotropic conductive film (ACF): R_contact ~ 1-100 mOhm per contact
- Wire bonds (if hybrid construction): R_contact ~ 1-10 mOhm

Total contact resistance: R_contact < 1 Ohm. Combined with R_Au and R_s: R_u ~ 10-100 Ohm. The dominant contribution is R_s (solution resistance, Layer 6), not the film or contact resistance.

### 4.4 Electrode area variability

Photolithography defines electrode dimensions to ±1 µm accuracy. For an electrode of 1 mm × 1 mm, the area uncertainty is ±0.2% from lithography alone. Additional contributions:
- Dicing alignment: ±0.1 mm → ±10% for a 1 mm electrode
- Undercutting during wet etching of gold: ±2-5 µm per side

Total expected inter-electrode area variation: ±5-10% on geometric area A_geo. This directly propagates to ±5-10% variation in Gamma_0 (before ratiometric correction) and ±5-10% variation in I_MB and I_Fc absolute values. The ratiometric R_i(0) = I_MB(0)/I_Fc(0) partially corrects for this (I_MB and I_Fc scale together with area), but not if Gamma_MB/Gamma_Fc varies between electrodes.

### 4.5 Surface cleaning state

Manufacturing residues that must be removed before SAM formation:
- Photoresist residues (from lithography)
- Adhesive residues (from lamination of cartridge layers)
- Atmospheric hydrocarbons (thiol self-assembly is sensitive to organic contamination)

Cleaning protocols:
1. O2 plasma (2 min, 50 W): removes organics, makes surface hydrophilic (theta ≈ 0-10°). Gold oxide forms but is removed by thiol in seconds.
2. UV/ozone (15 min): similar to O2 plasma, less aggressive.
3. Piranha (H2SO4/H2O2, NOT for polymer substrates): removes organics completely.
4. Electrochemical cleaning (CV in H2SO4 from -0.3 to +1.7 V, 20 cycles): removes adsorbed species, produces clean, annealed gold surface. Cannot be done in assembled cartridge (needs acidic electrolyte).

For assembled disposable cartridge: O2 plasma immediately before cartridge sealing is the most practical option. Model: post-plasma theta < 10°, R_f = 1.1-1.3 for evaporated gold.

Surface state decays over time during storage: theta recovers toward 30-60° over days to weeks in air (hydrocarbon readsorption). This affects SAM quality if the cartridge is assembled and stored. Shelf-life model: theta(t_storage) = theta_0 + (theta_inf - theta_0) * (1 - exp(-k_decay * t_storage)).

### 4.6 Gold grain structure and crystallographic orientation

Evaporated gold on glass has a polycrystalline (111)-textured surface. The (111) face is the most thermodynamically stable and most commonly used for SAMs. Electron-beam evaporation produces larger grains (10-50 nm) than thermal evaporation (~5-10 nm). Larger grains mean smoother surface and lower R_f.

Annealing (300°C, 20 min in vacuum or inert atmosphere) after evaporation increases grain size and reduces surface defect density → more uniform SAM formation, lower R_f closer to 1.

For a disposable cartridge: annealing step must be compatible with the substrate (glass: OK; PET: NOT OK at 300°C; PETG: NOT OK). This constrains substrate choice.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| R_f | 1.1-1.5 (evaporated) | All Gamma calculations (Layers 7-13) |
| A_true | A_geo × R_f | Layers 7, 9, 11, 13 |
| R_Au, R_contact | < 5 Ohm total | Layer 6 (R_u), Layer 14 (noise) |
| Batch variability sigma_A/A | ±5-10% | Layer 13 (error propagation) |
| theta_post_clean | < 10° | Layer 2 (capillary fill), Layer 7 (SAM) |

## Key literature to find

- Au roughness factor by CV: Trasatti and Petrii (Pure and Applied Chemistry, 1991) — standard reference for Q_ref = 390 µC/cm²
- Evaporated gold surface characterisation: Hegner et al. (Surface Science, 1993)
- Contact angle recovery on gold: Ulman (Chemical Reviews, 1996)
- SAM quality on different gold substrates: Bain and Whitesides (Science, 1988)

## Files to create

- `electrode_characterisation.py`: R_f calculation from CV data; area variability model
- `cleaning_protocol.py`: theta(t_storage) model for shelf-life assessment
