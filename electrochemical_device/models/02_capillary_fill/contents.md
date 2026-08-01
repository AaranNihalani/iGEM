# Layer 2: Capillary fill

## Purpose

This layer models the physical process of sample entering and filling the cartridge chamber by capillary action. The chamber must fill completely and uniformly without trapping air bubbles before the incubation and measurement steps. Fill time, fill completeness, and uniformity of analyte exposure to each electrode are the key outputs.

Capillary fill is the first physical process inside the device after sample loading. If it fails or is incomplete, all downstream modelling is invalid.

## Chamber geometry

Nominal dimensions (from device document):
- Length: 20 mm
- Width: 5 mm
- Height h: 0.25-0.50 mm (design variable)
- Cross-sectional area: 100 mm^2
- Volume V = 100 mm^2 × h: 25-50 µL (h-dependent)
- Number of working electrodes: 6-8 (along the length)
- Electrode spacing: ~2-3 mm (to be fixed from layout)

## Sub-components to model

### 2.1 Capillary driving pressure

For a rectangular channel (h << w, where h = height, w = width):
    ΔP_cap = gamma * cos(theta) * (1/h + 1/w)
where gamma = surface tension of sample (~72 mN/m water; ~50-65 mN/m for wound fluid with proteins/surfactants), theta = advancing contact angle.

The chamber has three different surface materials:
- Floor: gold electrode regions + PCB/polymer substrate between electrodes
- Lid: PDMS, PETG, or PMMA
- Side walls: same as lid material

Contact angles must be specified for each material. If theta > 90° on any surface, that surface opposes fill. A large hydrophobic patch (e.g., uncleaned polymer) can stall fill even if other surfaces are hydrophilic.

Typical target: theta < 30° on all surfaces → ΔP_cap ~ 100-500 Pa for h = 0.5-0.25 mm.

### 2.2 Lucas-Washburn fill dynamics

Fill front position L(t) for a rectangular channel:
    L(t) = sqrt(h^2 * ΔP_cap * t / (3 * eta))
    
where eta = dynamic viscosity of sample (~1.2 mPa·s for wound fluid at 37°C; ~1.0 mPa·s at 25°C).

Fill time to reach the far electrode (L = L_chamber = 20 mm):
    t_fill = 3 * eta * L_chamber^2 / (h^2 * ΔP_cap)

Compute t_fill for h in {0.25, 0.30, 0.375, 0.50} mm and confirm t_fill < 30 s (acceptable for a disposable cartridge).

Viscosity effect: wound fluid with elevated fibrinogen (up to 1 mg/mL) has higher viscosity than PBS. Must use wound-fluid-appropriate viscosity, not pure water.

### 2.3 Air venting

Without a dedicated air vent, filling compresses the air column ahead of the liquid front. Trapped air pressure:
    P_air(L) = P_atm * L_chamber / (L_chamber - L)

Fill stalls when ΔP_cap = P_air(L) - P_atm:
    L_stall = L_chamber * ΔP_cap / (ΔP_cap + P_atm)

For ΔP_cap = 200 Pa, P_atm = 101325 Pa: L_stall ≈ 0.04 mm — fill barely starts! A vent is absolutely required.

Vent membrane requirements: must allow air to escape at the volumetric flow rate Q = A_cross * dL/dt without significant back-pressure. Gas permeability of ePTFE vent membranes: P_g ~ 10^-8 - 10^-7 m^3/s/m^2/Pa. Must verify ΔP_vent << ΔP_cap.

### 2.4 Wetting failure criteria

If theta > 90° at any local region (e.g., a hydrophobic contamination spot), ΔP_cap locally reverses and fill stalls. Model as: fill proceeds only if theta_local < 90° everywhere along fill path. Plasma treatment or oxygen-plasma cleaning of all surfaces to achieve theta < 30° is essential.

Wettability may degrade during storage (contact angle recovery, especially on polymer surfaces). Must model shelf-life effect on theta.

### 2.5 Inlet dead volume and electrode exposure timing

Dead volume: volume consumed before sample reaches electrode 1, located at distance x_1 from inlet. Electrode at position x_i is wetted at time:
    t_wet,i = 3 * eta * x_i^2 / (h^2 * ΔP_cap)

For a 30-min incubation: if t_wet,8 - t_wet,1 << 30 min, all electrodes see effectively equal incubation time. For x_1 = 2 mm and x_8 = 18 mm: t_wet,8 - t_wet,1 ~ (18^2 - 2^2)/(2^2) * t_wet,1 ~ 80 × t_wet,1. But t_wet,1 is very short (seconds), so the absolute differential is seconds — negligible vs 30-min incubation.

### 2.6 Fill uniformity across electrode width

The electrodes are 5 mm wide (matching chamber width). Lateral uniformity: if contact angles are uniform across the width, fill is uniform. Lateral non-uniformity (e.g., gold electrode vs polymer region at same cross-section) can cause the fill front to advance faster over the more hydrophilic region, trapping air pockets at the hydrophobic region. This is a manufacturing defect risk and must be flagged.

### 2.7 Filter pressure drop (coupling to Layer 1)

If a particulate filter is at the inlet (Layer 1, Section 1.5), its pressure drop ΔP_filter reduces the effective driving pressure:
    ΔP_effective = ΔP_cap - ΔP_filter - ΔP_vent

Must verify ΔP_effective > 0 (fill still proceeds) and re-compute t_fill with reduced driving pressure.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| t_fill(h) | Function of h (seconds) | Design spec |
| Uniform fill? | Yes if theta uniform | Manufacturing requirement |
| t_wet,i differences | Seconds | Layer 3 (incubation start time) |

## Key literature to find

- Lucas-Washburn equation: original (1921) or review (Washburn, Phys Rev)
- Capillary fill in microfluidic cartridges: Lim et al. (Lab Chip), Zimmermann et al.
- Contact angle of wound fluid on PDMS / gold: search for 'wettability wound fluid' or use pure water as approximation
- Vent membrane specifications: Millipore ePTFE vent membrane product sheets

## Files to create

- `capillary_fill.py`: computes L(t), t_fill, L_stall for given h, theta, gamma, eta
- `fill_uniformity.py`: 2D model of fill front across chamber width for non-uniform contact angle
