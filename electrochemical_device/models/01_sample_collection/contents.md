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

### 1.3 Inhibitor co-elution

TIMP-1 (28 kDa), A1AT (52 kDa), and SLPI (11 kDa) are co-present in wound fluid alongside the target enzymes. They co-elute with similar or higher efficiency than MMP-9 and HNE and continue to inhibit those enzymes inside the cartridge during the 30-min incubation.

Co-eluted inhibitor concentrations:
    [I]_chamber = [I]_wound * eta_elut,I * D_f

where eta_elut,I is the size-dependent elution efficiency for each inhibitor (SLPI at 11 kDa: ~0.8; TIMP-1 at 28 kDa: ~0.6; A1AT at 52 kDa: ~0.5).

This produces an effective free enzyme concentration in the chamber that is lower than [E]_chamber calculated from 1.6 alone. The inhibitor binding kinetics are modelled in Layer 3 (in-device inhibition), but the initial inhibitor concentrations entering the chamber are determined here.

Critically, the inhibitor/enzyme ratio in the eluate differs from the wound fluid ratio if eta_elut,I ≠ eta_elut,E. For example, SLPI (small, high elution efficiency) may be proportionally enriched relative to HNE (medium elution efficiency) in the eluate → greater in-device HNE inhibition than wound biology ODE predicts. This ratio shift must be modelled.

### 1.4 Elution buffer effect on matrix

Elution buffer (PBS, pH 7.4, 150 mM NaCl) mixes with wound fluid retained in the swab. Net effect:
- pH of eluate: buffered to ~7.4 even if wound pH ≠ 7.4 (only relevant if the device measures in undiluted wound fluid; if eluted into PBS buffer, pH is controlled)
- Ionic strength: approximately constant at 150 mM (PBS dominated)
- [Ca2+]: wound fluid Ca2+ ~1-2 mM, diluted ~2× into PBS (PBS has 0.9 mM Ca2+) → net [Ca2+] in eluate ≈ 1-2 mM

This simplifies the Edinburgh Ca2+ artefact model (Layer 7): [Ca2+] in the cartridge is approximately known if PBS is the elution buffer.

However, this assumes complete pH buffering by the elution buffer. If the wound fluid volume is a significant fraction of the total (V_wound > 0.2 × V_buffer), the resulting pH must be calculated explicitly:

    pH_chamber = -log10( (V_wound × 10^(-pH_wound) + V_buffer × 10^(-pH_buffer)) / (V_wound + V_buffer) )

For a severely acidic wound (pH 6.5) with V_wound = 50 µL absorbed and V_buffer = 200 µL PBS:
    pH_chamber ≈ -log10((50 × 3.2×10^-7 + 200 × 4×10^-8)/250) ≈ 7.0

A 0.4 pH unit shift from buffer pH affects E°_MB by 24 mV — comparable to the SWV peak width — and shifts enzyme kcat meaningfully. Must propagate pH_chamber (not assumed 7.4) to Layer 6, 10, and 12.

Additionally: wound fluid redox components (glutathione ~1 mM, ascorbate ~0.1 mM, H2O2 ~10-100 µM) co-elute with the sample. These are tracked here as concentrations entering the chamber:
    [GSH]_chamber, [ascorbate]_chamber, [H2O2]_chamber

These are consumed downstream in Layer 7 (SAM oxidation) and Layer 11 (MB redox interference).

### 1.5 Analyte stability during transit

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
