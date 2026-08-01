# Layer 1: Sample collection and pre-analytical

## Purpose

This layer models the process of getting wound fluid from the wound surface into the cartridge chamber. It quantifies every step that dilutes, degrades, or otherwise alters the analyte concentrations between wound and device. The output is [E]_chamber — the effective enzyme concentration seen by the electrode — which is the true input to Layer 3.

This layer is critical for interpreting clinical results: a device that measures F_k accurately but whose concentration inference is corrupted by unknown dilution or degradation factors is clinically useless.

## Sub-components to model

### 1.1 Swab absorption volume

A swab is pressed against the wound surface and absorbs wound exudate. The absorbed volume V_abs depends on:
- Wound exudate level (low / moderate / heavy)
- Swab material and geometry (cotton, foam, polyester)
- Contact time and pressure

Literature range: V_abs ≈ 20-200 µL per swab pass on a moderate exudate wound. This is highly variable and constitutes the largest pre-analytical uncertainty.

Model: treat V_abs as a random variable with a log-normal distribution parameterised from literature swab studies. Propagate this uncertainty through to [E]_chamber.

### 1.2 Elution efficiency

After absorption, the swab is placed in elution buffer (e.g., 200 µL PBS, pH 7.4) and centrifuged or manually expressed. The fraction of absorbed analyte released is the elution efficiency eta_elut.

For proteins of molecular weight M_W:
- Small proteins (< 10 kDa): eta_elut ~ 0.7-0.9
- Medium proteins (30-100 kDa, relevant for HNE at 30 kDa and MMP-9 at 92 kDa): eta_elut ~ 0.4-0.7
- Large proteins / complexes: eta_elut ~ 0.2-0.5

[E]_eluate = [E]_wound * eta_elut

Elution efficiency is size-dependent and may also depend on swab material (hydrophobic materials retain hydrophobic proteins more strongly). Must source from literature or measure directly.

### 1.3 Elution buffer effect on matrix

Elution buffer (PBS, pH 7.4, 150 mM NaCl) mixes with wound fluid retained in the swab. Net effect:
- pH of eluate: buffered to ~7.4 even if wound pH ≠ 7.4 (only relevant if the device measures in undiluted wound fluid; if eluted into PBS buffer, pH is controlled)
- Ionic strength: approximately constant at 150 mM (PBS dominated)
- [Ca2+]: wound fluid Ca2+ ~1-2 mM, diluted ~2× into PBS (PBS has 0.9 mM Ca2+) → net [Ca2+] in eluate ≈ 1-2 mM

This simplifies the Edinburgh Ca2+ artefact model (Layer 7): [Ca2+] in the cartridge is approximately known if PBS is the elution buffer.

### 1.4 Analyte stability during transit

Between sample collection and cartridge loading, the eluate is held at ambient temperature for time t_transit (typically 0-15 min, but could be up to 2 hours in a field setting).

First-order degradation:
    [E](t_transit) = [E]_eluate(0) * exp(-k_deg * t_transit)

k_deg arises from:
- Protease self-digestion (auto-proteolysis): MMP-9 can cleave itself; rate ≈ 0.001-0.01 min^-1 at room temperature
- Inhibitor redistribution: endogenous TIMP-1 or A1AT in the eluate continues to inhibit MMP-9 / HNE during transit
- Surface adsorption to swab or tube: additional loss mechanism

For a 15-min transit: exp(-0.005 * 15) ≈ 0.93, so ~7% loss. Small but worth quantifying, especially for longer transit times in field use.

### 1.5 Particulate content and inlet filter

Wound fluid contains fibrin clots, cellular debris, and bacterial aggregates. These can:
- Block the cartridge inlet (requires a filter)
- Increase solution viscosity (affects capillary fill, Layer 2)
- Contribute to non-specific electrode fouling (Layer 7)

Model: filter at cartridge inlet with pore size d_pore. Species larger than d_pore are excluded. Fibrin clots: ~1-10 µm; bacteria: ~1-5 µm; cells: ~5-20 µm. Appropriate d_pore: 0.45-2 µm.

Pressure drop across filter: ΔP_filter = 8 η L Q / (π r_pore^4 N_pores) (Hagen-Poiseuille for a parallel pore membrane). This subtracts from the capillary driving pressure available for fill (Layer 2). Must check that ΔP_filter does not stall fill.

### 1.6 Net dilution factor

After elution and any pre-analytical steps:
    [E]_chamber = [E]_wound * eta_elut * exp(-k_deg * t_transit) * D_f

where D_f = V_eluate / V_chamber is the dilution factor (eluate volume loaded into chamber / total chamber volume).

For V_eluate = 30 µL (loaded from 200 µL eluate), V_chamber = 37.5 µL: D_f = 0.8 (some eluate remains outside). This concentrates or dilutes depending on the ratio. If D_f < 1, analyte is not further diluted at this step (assuming the chamber is completely filled with eluate); if D_f > 1, the eluate is diluted with any pre-loaded buffer.

### 1.7 Uncertainty propagation

Total uncertainty on [E]_chamber from this layer:
    sigma_[E]/[E] = sqrt((sigma_V_abs/V_abs)^2 + (sigma_eta/eta)^2 + (sigma_kdeg/kdeg)^2)

The dominant term is likely sigma_V_abs/V_abs (~50%), making swab absorption volume the largest pre-analytical uncertainty. This will dominate the Sobol sensitivity analysis in Layer 16.

## Coupling to other layers

| Feeds | Layer | Parameter |
|---|---|---|
| [E]_chamber | Layer 3 | Initial bulk enzyme concentration |
| pH_eluate | Layer 6, 10 | pH correction to E°_MB |
| [Ca2+]_eluate | Layer 7 | Edinburgh Ca2+ artefact |
| Viscosity (η) | Layer 2 | Lucas-Washburn fill |

## Key literature to find

- Swab absorption studies: Vowden et al. (J Wound Care), Gardner et al.
- Elution efficiency for proteins: Ono et al., or general wound fluid sampling papers
- MMP-9 stability in wound fluid eluates: test internally if not in literature
- Filter pressure drop: membrane filtration literature

## Files to create

- `preanalytical_model.py`: compute [E]_chamber from wound concentrations and sampling parameters
- `uncertainty_propagation.py`: Monte Carlo over V_abs, eta_elut, t_transit distributions → sigma_[E]
