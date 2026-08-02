# Layer 9: Enzyme kinetics on the surface

## Purpose

This layer models the rate at which the protease cleaves probe molecules from the electrode surface. It is the chemical heart of the assay: the observable quantity (F_k, fraction cleaved) is a direct output of this kinetics model. The layer couples the bulk enzyme concentration (from Layer 3 PDE at z=0) with the surface chemistry (Gamma_p from Layer 7, K_m,surf from Layer 8) to compute Gamma_intact(t), the time-course of intact probe density.

The wet lab work (pES-01 to pES-08) will directly measure kcat/Km for each substrate-enzyme pair in solution. This layer converts those solution measurements into predictions of surface cleavage rates, accounting for steric correction, enzyme depletion, and competing interfering proteases.

## Sub-components to model

### 9.1 Surface Michaelis-Menten ODE

On a planar surface, the substrate is not diffusing in 3D — it is fixed at the surface with surface concentration Gamma (mol/cm^2). The enzyme diffuses in 3D and must bind to the surface-tethered substrate. This leads to a modified Michaelis-Menten equation:

    dGamma/dt = -k_cat * [E]_surface * Gamma / (K_m,surf + Gamma)

where [E]_surface = [E](z=0, t) from the Layer 3 diffusion PDE.

Note the units: Gamma in mol/cm^2; [E] in mol/L = mol/cm^3 × 10^3; k_cat in s^-1; K_m,surf in mol/cm^2 (surface concentration units, NOT mol/L). The Km for a surface substrate has different units from the solution Km.

Relationship between K_m,surf and K_m,solution: the surface K_m is related to the solution K_m by:
    K_m,surf = K_m,sol * Gamma_ref / c_ref

where Gamma_ref and c_ref are a reference surface and volume concentration pair used to make the units consistent. In practice, the most useful form is:

    v_surface = k_cat * [E] * Gamma / (K_m,eff + Gamma)

where K_m,eff (with units mol/cm^2) is fit from calibration data. The relationship to solution K_m requires:
    K_m,eff ≈ K_m,sol [mol/L] × delta [cm] × 10^-3 [L/cm^3]

where delta is the effective thickness of the surface layer (approximately the probe extent, h_probe ≈ 5 nm = 5×10^-7 cm).

So K_m,eff ≈ K_m,sol (in M) × 5×10^-7 (cm) × 10^-3 (L/cm^3) = K_m,sol × 5×10^-10 mol/cm^2.

For K_m,sol = 100 µM = 10^-4 mol/L:
    K_m,eff = 10^-4 × 5×10^-10 = 5×10^-14 mol/cm^2 = 0.05 fmol/cm^2

This is much smaller than typical Gamma_0 ≈ 5 pmol/cm^2 = 5×10^-12 mol/cm^2. So:
    Gamma >> K_m,eff → zeroth-order regime: dGamma/dt = -k_cat * [E]

In the zeroth-order regime, the cleavage rate is independent of Gamma! The signal would not tell us how much enzyme is present. We want the first-order (unsaturated) regime where dGamma/dt ≈ -(k_cat/K_m,eff) * [E] * Gamma.

Resolution: the steric correction to K_m,surf (Layer 8) increases K_m,surf by a factor f_steric^-1 ≈ 2-10×. This pushes K_m,eff up. Additionally, the effective solution K_m includes the accessibility effect — enzyme must diffuse through the SAM — so K_m,surf is further elevated. Literature reports K_m,surf / K_m,sol ratios of 3-100× for surface-tethered substrates.

Must empirically calibrate K_m,surf from the calibration experiment (Layer 15) rather than predicting it purely from K_m,sol.

### 9.2 First-order limit and k_obs

When Gamma << K_m,surf (first-order regime):
    dGamma/dt ≈ -(k_cat/K_m,surf) * [E]_surface * Gamma

If [E]_surface is approximately constant (Da << 0.1), this integrates to:
    Gamma(t) = Gamma_0 * exp(-k_obs * t)

where k_obs = (k_cat/K_m,surf) * [E]_surface = (k_cat/K_m,surf) * [E]_chamber * C_depletion(t)

The depletion correction factor C_depletion(t) comes from the Layer 3 PDE solution: it represents how much [E] at the surface has been depleted below bulk. For Da < 0.1: C_depletion ≈ 1 (surface sees bulk [E]).

Fractional cleavage:
    F(T) = 1 - Gamma(T)/Gamma_0 = 1 - exp(-k_obs * T)

This is the formula used to extract k_obs from the measured F_k.

### 9.3 Steric correction from Layer 8

From Layer 8, steric factor f_steric = 1 - Gamma_p/Gamma_steric_max. The K_m,surf correction:
    K_m,surf = K_m,sol * delta / f_steric

(larger K_m → further from saturation → closer to first-order → better for linear response to [E]).

At Gamma_p = 3 pmol/cm^2 with Gamma_steric_max = 5.1 pmol/cm^2: f_steric = 0.41.
    K_m,surf = K_m,sol * delta / 0.41 = 2.44 * K_m,sol * delta

This correction must be included. It makes K_m,surf larger, which is good (keeps us in first-order regime).

### 9.4 Competing cleavage from interfering proteases

Multiple proteases in wound fluid (j = MMP-9, HNE, cathepsin G, plasmin, etc.) can all cleave the same probe with their own kcat,j and K_m,j. Total cleavage rate:
    dGamma/dt = -sum_j [kcat,j * [E_j]_surface * Gamma / (K_m,surf,j + Gamma)]

In the first-order limit:
    dGamma/dt ≈ -sum_j [(kcat,j/K_m,surf,j) * [E_j]_surface] * Gamma
    dGamma/dt ≈ -k_obs,total * Gamma

where k_obs,total = sum_j [(kcat,j/K_m,surf,j) * [E_j]_surface].

The fractional contribution of interfering protease j to F_k:
    delta_F_k^interferent,j / F_k = (kcat,j/K_m,surf,j * [E_j]) / k_obs,total

Must compute this for each interferent. Requires: (a) cross-reactivity scores from the substrate predictor for AAPVSGIG and IPRTLTAA vs each interferent, (b) interferent concentrations from Layer 0.

### 9.5 Cathepsin and plasmin cross-reactivity (MEROPS-based check)

The substrate predictor panel covers HNE, MMP-2, MMP-8, MMP-9, MMP-14, and FAP. However, wound fluid also contains cathepsin G, cathepsin B, and plasmin, for which no trained ML model is available. These must be assessed using MEROPS cleavage data and sequence specificity rules.

**Cathepsin G** (serine protease, neutrophil degranulation product, MW 28 kDa):
- Cleaves after aromatic residues (F, Y) and also after A, V (similar to HNE but slightly broader).
- AAPVSGIG: the A-P-V motif — cathepsin G has been reported to cleave Val-Ser bonds in some substrates. Must check MEROPS S1 family specificity matrix for Ala, Pro, Val at P4-P1.
- Expected risk: moderate. Cathepsin G shares the chymotrypsin fold with HNE; P1 = Val in AAPV↓SGIG is acceptable to cathepsin G.
- Literature: Olsen et al. cleavage data for cathepsin G suggests Val at P1 is cleaved ~20-50% efficiency vs Phe. This is non-negligible.

**Cathepsin B** (cysteine protease):
- Primarily an endopeptidase (cleaves at K, R) and exopeptidase (carboxydipeptidase).
- AAPVSGIG: no K or R → cathepsin B endopeptidase activity unlikely. Low risk.
- IPRTLTAA: R at position 3 (P6 in P10-P1 notation if cleavage at I-P-R↓T) — cathepsin B could cleave at Arg. Moderate risk.

**Plasmin** (serine protease, fibrinolytic):
- Highly specific for Arg and Lys at P1.
- IPRTLTAA: R at position 3. If plasmin cleaves I-P-R↓T-L-T-A-A, this would release the T-L-T-A-A-MB fragment and leave only I-P-R on the electrode (short stub → MB is GONE → false positive for MMP-9 channel). This is a significant risk.
- AAPVSGIG: no R or K → plasmin will not cleave. Low risk.

**Action items for this model:**
1. Query MEROPS (merops.sanger.ac.uk) for cathepsin G and plasmin cleavage data on sequences containing AAPV and IPRT motifs.
2. Estimate [cathepsin G]_wound and [plasmin]_wound from literature (Yager et al., Trengove et al.).
3. Compute delta_F_k for each interferent using the same multi-protease competition formula in 9.4.
4. If delta_F_k > 0.05 (5% false positive from interferent at expected wound concentrations): flag as a selectivity risk requiring probe sequence modification.

This check must be completed before the substrate sequences are finalised. Plasmin/IPRTLTAA cross-reactivity is the highest-priority concern.

### 9.6 Zeroth-order regime (saturation)

When Gamma >> K_m,surf: dGamma/dt = -kcat * [E]_surface. The rate is constant (zeroth order in Gamma) and proportional to [E]. In this regime:
    Gamma(t) = Gamma_0 - kcat * [E] * t
    F(T) = kcat * [E] * T / Gamma_0

F is now linear in [E] and linear in T but with a different slope than the first-order case. The assay still works but is in a different quantitative regime. K_m,surf must be measured to know which regime is operating.

### 9.6 Coupled PDE-ODE system (summary)

The full coupled system to solve (from Layer 3 and Layer 9):

PDE: ∂[E]/∂t = D_E * ∂²[E]/∂z²   (Layer 3)
ODE: dGamma/dt = -k_cat * [E](z=0,t) * Gamma / (K_m,surf + Gamma)   (this layer)
BC:  D_E * ∂[E]/∂z|_{z=0} = k_cat * [E](0,t) * Gamma / (K_m,surf + Gamma)

Solve simultaneously (Crank-Nicolson for PDE, RK4 for ODE) to get [E](z,t) and Gamma(t).

Output: Gamma(T) = Gamma_0 - integral_0^T [dGamma/dt] dt → F(T) = 1 - Gamma(T)/Gamma_0.

### 9.7 Temperature dependence of kcat

Arrhenius for enzyme kcat:
    kcat(T) = kcat(T_ref) * exp(-Ea/R * (1/T - 1/T_ref))

Ea for serine proteases (HNE): ~50-60 kJ/mol. For metalloproteinases (MMP-9): ~40-50 kJ/mol.

Effect of a 10°C temperature change (e.g., 25 to 35°C):
    kcat(308)/kcat(298) = exp(-50000/8.314 * (1/308 - 1/298)) = exp(50000 * 3.3×10^-5) ≈ exp(1.65) ≈ 5.2

A factor of ~5 increase in kcat! This means F_k at 35°C is ~5× larger than at 25°C for the same incubation time and [E]. Temperature is a major confound if not controlled or measured.

Ratiometric design: F_k = 1 - rho_active/rho_control does NOT cancel the temperature effect (both active and control probes are not being cleaved by the same amount — the control probe is not cleaved, so temperature affects only the active probe). Temperature must be measured or the device must be thermostated.

### 9.8 Time to reach steady-state (pre-steady-state kinetics)

At t=0 (just after fill), the enzyme must diffuse to the surface (t_diff ≈ h^2/(2D) ≈ 1400 s, Layer 3.2) before cleavage begins efficiently. For early incubation times (T << t_diff), the pre-steady-state cleavage kinetics will show a lag phase:
    F(T) is sub-linear in T for T < t_mix

For T >> t_mix (≈ 23 min), the enzyme has reached a quasi-steady distribution and F(T) follows the exponential kinetics. For a 30-min incubation: only the last ~7 min are in steady-state. This means the simple exponential model F(T) = 1 - exp(-k_obs*T) is an approximation. The full PDE-ODE simulation is needed for accurate prediction.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| Gamma(t) — surface probe density | Time-course | Layer 10, 11, 13 |
| F(T) = 1 - Gamma(T)/Gamma_0 | Fractional cleavage | Layer 13 |
| k_obs | s^-1 | Layer 13 (kappa_k) |
| K_m,surf/K_m,sol ratio | 3-100× (steric) | Layer 15 (calibration) |
| Interferent delta_F | Fractional contribution | Layer 15 (LOD, specificity) |

## Key literature to find

- Surface Michaelis-Menten: Peterson et al. (JACS 2002); Perez et al. (JACS 2012) — key papers on peptide substrate cleavage on surfaces
- Solution-to-surface Km correction: Squires et al. (Nature Biotech 2008) — fundamental paper on transport effects in biosensors
- kcat for HNE on AAPVSGIG: MEROPS database; Birktoft & Blow; or from pES wet lab
- kcat for MMP-9 on IPRTLTAA: MEROPS database or Bickett et al.
- Enzyme kinetics at steric saturation: Lund et al. (Biophys J)

## Files to create

- `surface_kinetics.py`: coupled PDE-ODE system solver; returns Gamma(t), F(T), k_obs
- `km_surface_correction.py`: K_m,surf from K_m,sol, steric factor, effective thickness delta
- `competing_cleavage.py`: multi-protease competition model; delta_F per interferent
- `temperature_correction.py`: Arrhenius correction for kcat; expected F_k vs T_amb
