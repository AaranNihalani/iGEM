# Electrochemical Device Modelling

Eton iGEM 2026 — Dry Lab  
End-to-end computational model of the multiplexed electrochemical cartridge for wound protease detection (MMP-9 / HNE).

---

## Scope

There is no physical prototype. Every physical process from wound fluid biology through to the clinical readout number must be modelled mathematically. The chain is fully coupled: outputs of each layer feed as inputs to the next. The model must produce falsifiable predictions (predicted signal F_k vs known protease concentration) that can be validated against wet lab calibration data.

---

## Folder structure

```
electrochemical_device/
├── reports/                          LaTeX modelling reports
├── models/
│   ├── 01_wound_fluid/               ODE system for active protease concentrations
│   ├── 02_sample_collection/         Swab absorption, elution efficiency, dilution
│   ├── 03_fluidics/                  Capillary fill, diffusion PDE, Damköhler analysis
│   ├── 04_sam_chemistry/             Thiol adsorption kinetics, SAM composition, Ca2+ artefact
│   ├── 05_surface_kinetics/          Michaelis-Menten on monolayer, steric factor, depletion
│   ├── 06_electron_transfer/         Marcus theory, tether conformational model, Butler-Volmer
│   ├── 07_swv_forward/               SWV current model, peak simulation, parameter sweep
│   ├── 08_ratiometric/               R, rho, F_k, kappa_k with full error propagation
│   ├── 09_noise_budget/              TIA, ADC, shot noise, mains interference, LOD floor (Layer 14)
│   ├── 10_calibration/               Calibration curve, LOD, dynamic range (Layer 15)
│   ├── 11_interpretation/            Deconvolution matrix, Bayesian wound classifier (Layer 15-16)
│   ├── 12_peak_extraction/           Savitzky-Golay smoothing, baseline, Gaussian peak fit (Layer 12)
│   ├── 13_ratiometric/               R, rho, F_k, kappa_k with full error propagation (Layer 13)
│   ├── 14_noise_budget/              Full electronics noise analysis (Layer 14)
│   ├── 15_calibration/               LOD, LLOQ, dynamic range, matrix spike-recovery (Layer 15)
│   └── 16_forward_chain/             Full end-to-end forward simulation chaining all layers (Layer 16)
├── data/
│   └── literature_parameters/        CSV/JSON tables of all numeric parameters with citations
└── figures/                          All generated figures
```

---

## Negative control task (immediate)

Scramble AAPVSGIG (HNE substrate) to produce a non-cleavable control electrode sequence.

**Charge balance:** All 8 residues (A, A, P, V, S, G, I, G) are neutral at pH 7.4. The probe termini are acetylated (N-term) and amidated (C-term), so the net charge of any permutation is exactly 0. "Same charge balance" is automatically satisfied by any scramble that preserves amino acid composition.

**Protocol:**
1. Enumerate permutations of {A, A, P, V, S, G, I, G} (unique: 8!/(2!×2!) = 10,080).
2. Filter: remove sequences containing the AAPV motif at positions 4-7 (P4-P1 for HNE) or AAPV anywhere that might regenerate HNE recognition.
3. Score all remaining candidates with the existing HNE LR model + PSSM scorer (from `substrate_predictor/`).
4. Also score with MMP-9 model to confirm cross-channel non-reactivity.
5. Select the candidate with the lowest combined HNE + MMP-9 score. Report sequence, LR probability, and PSSM score.

**Expected outcome:** Selected scramble should have LR < 0.2 (vs AAPVSGIG LR = 0.981) and PSSM < 0.

---

## Full modelling plan — all 16 layers

### Layer 0: Wound biology (system input)

Sub-components requiring explicit modelling:

- ODE system: pro-MMP-9 activation by plasmin, TIMP-1 inhibition, A1AT-HNE inhibition (existing Pillar 2 ODEs feed directly into this device model)
- Active vs zymogen fractions: [MMP-9_active] = f(plasmin, TIMP-1, time); [HNE_active] = f(A1AT, time)
- pH of wound fluid: range 6.5-8.2 (chronic vs acute wound). Affects enzyme activity (kcat pH-profile), MB formal potential E° (shifts -59 mV/pH unit for the 2e-/2H+ MB couple), Debye screening length
- Ionic strength I: physiological ~ 150 mM → Debye length lambda_D = sqrt(epsilon*epsilon_0*RT / 2F^2*I) ~ 0.8 nm. Controls double-layer screening and electrostatic interaction with peptide charges
- Temperature: operating range 20-37°C. Affects D (Stokes-Einstein), kcat (Arrhenius), k_ET (Marcus), MB E° (Nernst)
- Oxidative species: H2O2, hypochlorite in wound fluid can oxidise Cys-Au thiol bond → SAM degradation rate model
- Fouling matrix proteins: albumin (~40 mg/mL serum equivalent, diluted in wound fluid), fibrinogen → non-specific adsorption to gold (competes with and potentially displaces SAM)
- Interfering proteases: cathepsins B/D/G, plasmin, urokinase — must cross-check their cleavage specificity against both AAPVSGIG and IPRTLTAA (using MEROPS data) to quantify off-target signal contribution

### Layer 1: Sample collection and pre-analytical

- Swab absorption volume: literature range 20-200 µL absorbed per swab pass; model as function of wound exudate level
- Elution efficiency E_elut: fraction of absorbed analyte released into elution buffer (typically 40-80% for proteins; size-dependent)
- Elution buffer composition: pH, ionic strength → net effect on sample pH and ionic strength after mixing with wound fluid
- Analyte stability: kcat for protease self-digestion and substrate auto-degradation during the swab-to-device transit time (model as exponential decay with first-order rate k_deg)
- Dilution factor D_f = V_eluate / V_chamber → final [E]_chamber = [E]_wound × E_elut × D_f
- Particulate content: fibrin clots filter at inlet (need mesh pore size and pressure-drop model across filter); cell debris contribution to background current

### Layer 2: Cartridge inlet and capillary fill

- Chamber geometry: 20 × 5 mm × h (h = 0.25-0.50 mm); V = 100 mm² × h ≈ 25-50 µL
- Capillary pressure: ΔP_cap = γ·cos(θ)·(1/h + 1/w) where γ = surface tension of sample (~ 72 mN/m for water, reduced by surfactants), θ = advancing contact angle on each surface (PDMS lid, gold floor, PETG walls — each different)
- Lucas-Washburn fill dynamics: dL/dt = h²·ΔP_cap / (3ηL) → fill time t_fill(h)
- Air venting: without a vent, back-pressure from trapped air stalls fill at L < L_chamber. Model: P_air(L) = P_atm · L_chamber / (L_chamber - L); fill stops when P_cap = P_air → maximum fill fraction. Vent membrane permeability and back-pressure must be included
- Wetting failure criterion: if θ > 90° on any surface the capillary pressure term reverses → fill requires applied pressure
- Inlet dead volume: volume consumed before the leading edge reaches electrode 1 (positions electrodes determine minimum sample volume)
- Concentration profile along chamber length at t = t_fill: convective plug flow gives uniform [E] if Pe >> 1 during fill; diffusion during fill is negligible (t_fill << t_diff)
- Non-uniform electrode exposure: if 6-8 WEs span the chamber length (L = 20 mm), electrode 1 (near inlet) is exposed from t ≈ 0 while electrode 8 (far end) is exposed from t = t_fill. For a 30-min incubation this offset is small, but must quantify

### Layer 3: Mass transport inside chamber (incubation phase)

- Post-fill: device is static (no convection). Transport is diffusion-dominated
- Diffusion coefficients (Stokes-Einstein: D = k_BT / (6πη r)):
  - MMP-9 (92 kDa, r ~ 3.5 nm): D ≈ 5×10^-11 m²/s
  - HNE (30 kDa, r ~ 2.0 nm): D ≈ 9×10^-11 m²/s
  - Released MB peptide fragment (~1 kDa, r ~ 0.6 nm): D ≈ 3×10^-10 m²/s
- Vertical mixing time (across channel height h): t_mix ≈ h²/(2D). For h = 0.375 mm, MMP-9: t_mix ≈ 1400 s (>> 30 min). Channel is NOT well-mixed vertically during incubation — must solve 1D diffusion PDE in z-direction
- Governing PDE: ∂[E]/∂t = D_E · ∂²[E]/∂z², z ∈ [0, h]; IC: [E](z,0) = [E]_0; BC at z=h (lid): no-flux; BC at z=0 (electrode): D_E · ∂[E]/∂z = kcat·[E]·Γ / (Km_surf + Γ)
- Depletion layer: [E] near electrode is depleted below bulk value; the Robin boundary condition couples this PDE to the surface ODE in Layer 5
- Damköhler number: Da = (kcat·Γ_0 / Km_surf) / (D_E/h² × [E]_0) — must have Da < 1 for reaction-limited regime (signal proportional to kcat/Km, not to diffusion coefficient)
- MB fragment transport: after cleavage, released fragment (r ~ 0.6 nm) diffuses away; must clear the diffusion layer before SWV to avoid MB contributing a bulk diffusion peak that distorts the surface-confined peak. Clearance time t_clear ~ δ²/D_frag where δ = Nernst diffusion layer thickness (10-100 µm for quiescent solution)
- Electrode cross-talk: MB fragment from electrode i can diffuse laterally (D_frag, L_elec-to-elec ~ 2-5 mm) and re-adsorb or contribute to electrode j. Time for fragment to travel 2 mm: t ~ (2mm)²/(2D_frag) ~ 7000 s — negligible cross-talk for typical incubation times unless electrodes are <1 mm apart

### Layer 4: Gold electrode surface

- Electrochemical roughness factor Rf: Rf = Q_Au-oxide / (Q_ref × A_geo), where Q_Au-oxide is charge under gold oxide stripping peak. Rf ≈ 1.1-1.5 for smooth evaporated gold; Rf > 2 for rough or polycrystalline. All surface densities Γ are in mol/cm²_true
- Gold film resistance: ρ_Au (2.4×10^-8 Ω·m) × L_trace / (w × t_film) — for L=5 mm, w=0.5 mm, t=100 nm: R_Au ~ 2.4 Ω. Small but non-negligible for low-current measurements
- Contact pad resistance: wire bond / anisotropic conductive adhesive (ACA) — typical R_contact ~ 0.01-1 Ω per bond; must add to uncompensated resistance R_u
- Surface cleaning state: manufacturing residues (photoresist, adhesive) can passivate gold and prevent thiol SAM formation. Plasma (O2, 2 min) or electrochemical cleaning (cyclic sweep in H2SO4) protocol — model as setting initial contact angle θ_clean < 30° and Rf close to theoretical
- Electrode geometric area variability: ±5% from photolithography and dicing; contributes directly to inter-electrode Γ variation (can be partially corrected by ratiometric R_i(0))

### Layer 5: Reference and counter electrodes

- Reference electrode (on-chip Ag/AgCl):
  - E_ref = E°_AgCl - (RT/F)·ln(a_Cl) = 0.197 - 0.0257·ln(a_Cl) vs NHE
  - [Cl^-] in wound fluid: ~ 110-120 mM → E_ref ≈ +0.20 V vs NHE ≈ -0.005 V vs standard Ag/AgCl
  - Drift: as assay proceeds, Cl^- at the Ag/AgCl surface is consumed or diluted → δE_ref
  - Acceptable drift: < 5 mV over 30 min (otherwise SWV peaks shift → errors in ratiometric calculation)
  - Ag/AgCl dissolution: AgCl solubility product Ksp = 1.8×10^-10; model [Ag+] release over time
- Counter electrode area: must supply current to all WEs simultaneously during SWV. Peak SWV current per WE: ~10-100 nA. For 8 WEs: I_CE_max ~ 800 nA. CE must be large enough that it is not kinetically limiting (area > 10× combined WE area as rule of thumb)
- CE products: at +0.6 V vs Ag/AgCl (most positive SWV point), CE is cathodic → reduces O2 or H2O. At -0.6 V, CE is anodic → oxidises water (O2 evolution). Bubble generation at CE could disrupt solution. Check if CE potential exceeds oxygen evolution potential (E ~ +1.23 V vs RHE → +0.97 V vs Ag/AgCl) — unlikely in this scan range but must confirm
- Junction / iR drop: R_s = ρ_sol × d_WE-RE / A_cell. For wound fluid (σ ~ 15 mS/cm, d ~ 5 mm, A ~ 1 cm²): R_s ~ 33 Ω. SWV capacitive current 150 nA × 33 Ω = 5 µV offset — negligible. Faradaic peak current ~10 nA × 33 Ω = 0.3 µV — also negligible. iR compensation likely not required, but must verify at maximum expected currents

### Layer 6: Double layer and solution resistance

- Double-layer capacitance for SAM-modified gold: C_dl ≈ 1-5 µF/cm² (vs 20-40 µF/cm² for bare gold). SAM acts as thin dielectric; C_dl = ε_0·ε_SAM / d_SAM
- Capacitive SWV background: I_cap = C_dl × A × 2·ΔE_SW × f_SWV. For C_dl=2 µF/cm², A=1 mm², ΔE_SW=25 mV, f=60 Hz: I_cap = 60 nA. This is the noise floor the faradaic peak must exceed — dimensioning Γ is critical
- Gouy-Chapman-Stern (GCS) model: total capacitance C_total^-1 = C_Helmholtz^-1 + C_diffuse^-1. At I=150 mM, C_diffuse >> C_Helmholtz → total ≈ Helmholtz (inner layer dominates). Debye length λ_D = 0.8 nm — much less than tether length (5 nm) so most of the peptide is in the diffuse layer, unshielded
- Electrostatic effect on probe: charged residues on peptide experience full electrostatic potential of the diffuse layer. For AAPVSGIG (all neutral) this is zero, but IPRTLTAA has R (pKa 12.5, fully +1 at pH 7.4) and T (neutral) — R contributes +1 charge → electrostatic repulsion from the electrode at cathodic potentials could distort kinetics
- pH gradient near electrode: during SWV, reduction/oxidation of MB consumes/produces H+. Local pH shift at electrode surface = f(MB surface concentration, buffer capacity of wound fluid). Wound fluid has low buffer capacity → local pH shift could shift E°_MB during scan → peak broadening

### Layer 7: SAM formation and composition

- Thiol adsorption kinetics: Langmuir model with competitive adsorption between probe-thiol (c_p, rate k_ad,p) and diluent-thiol (c_d, rate k_ad,d). Short diluents adsorb faster (lower steric penalty): k_ad,d >> k_ad,p
- Equilibrium surface composition: Γ_p/Γ_d = (c_p/c_d) × (k_ad,p/k_ad,d) × (exp(-ΔG_p/RT) / exp(-ΔG_d/RT)). Surface is diluent-enriched vs solution. Must parametrise from literature mixed-SAM adsorption isotherms
- Phase separation: mixed thiols can form nanoscale (5-20 nm) phase-separated domains. Probe-rich domains create local steric crowding even at low overall Γ_p. This modifies effective enzyme accessibility
- Equilibrium surface coverage Γ_p: target range 1-10 pmol/cm² for good enzyme access. Literature: >15 pmol/cm² gives poor cleavage efficiency (steric blockage); <0.5 pmol/cm² gives poor signal-to-noise
- Fc electrode SAM: C6-Fc-SH thiol with diluent. Fc-Fc through-space electron hopping becomes significant at Γ_Fc > ~ 5 pmol/cm² → distorts SWV peak to asymmetric shape. Must keep Γ_Fc < 3 pmol/cm²; confirm with SWV diagnostic (symmetric peak = isolated molecules)
- Edinburgh Ca2+ artefact — full mechanistic model:
  - PEG spacer (PEG4 or PEG6) chelates Ca2+ and other divalent cations (Mg2+, Zn2+)
  - Chelation constant K_d for PEG-Ca2+: ~ 1-10 mM (literature for PEG crown ether analogy)
  - At [Ca2+] ~ 1 mM (wound fluid range): significant fraction of PEG is complexed
  - Complexed PEG is more compact (less hydrated) → MB-to-Au effective distance d_eff decreases
  - Δd = f([Ca2+], K_d, PEG contour length) — parametrise as: Δd = Δd_max × [Ca2+]/(K_d + [Ca2+])
  - k_ET^Ca = k_ET^0 × exp(-β × (d_0 - Δd)) → peak height increases
  - This is a confound: ratiometric F_k = 1 - ρ_active/ρ_control. If active and control electrodes both have PEG spacers (and negative control probe is scrambled AAPVSGIG with same PEG architecture), the Ca2+ effect is identical on both → cancels in F_k. Must verify this assumption holds (same Γ_p on both electrodes → same Ca2+ artefact magnitude)
- SAM stability during assay: thiol-gold bond has finite lifetime. Reductive desorption at negative SWV potentials: E_des ~ -0.9 V vs Ag/AgCl for alkanethiols. SWV scan limit is ~ -0.6 V → safe margin exists but must verify. Thermal desorption (half-life ~ months at 25°C) negligible over 30-min assay

### Layer 8: Probe architecture and conformational dynamics

- Probe (HNE channel): Ac-Cys-[PEG4 or PEG6]-AAPV↓SGIG-[PEG2]-Lys(MB)-NH2
  - PEG4: 4 × 0.36 nm = 1.44 nm; PEG6: 6 × 0.36 = 2.16 nm
  - AAPVSGIG: 8 residues; extended (beta-strand): 8 × 0.35 nm = 2.8 nm; random coil Rg ~ N^0.6 × l_0 where l_0 = 0.38 nm → Rg ~ 1.3 nm; helix: 8 × 0.15 nm = 1.2 nm
  - PEG2: 2 × 0.36 = 0.72 nm
  - Total contour L_c ≈ 5.0-6.0 nm (depending on PEG4 vs PEG6 and peptide conformation)
- End-to-end distance distribution: Kratky-Porod worm-like chain for PEG (l_p ~ 0.38 nm) and peptide (l_p ~ 0.4-1.0 nm). Mean end-to-end distance <r> = L_c × (1 - exp(-2L_c/l_p))^0.5 × ... → numerical integration. Distribution P(r) is broad → distribution of d_eff values, hence distribution of k_ET values
- k_ET is exponentially sensitive to d: <k_ET> = integral_0^inf k_ET^max × exp(-β d) × P(d) dd — this average is dominated by the short-d tail of P(d). The measured peak height reflects this average, not k_ET at <d>
- SAM-density effect on conformation: at high Γ_p, steric repulsion between probe chains forces them into extended (upright) conformations → d_eff ≈ L_c (maximum). At low Γ_p, probes collapse onto electrode → d_eff << L_c → faster k_ET. Must model conformation as function of Γ_p
- Steric accessibility of cleavage site: enzyme footprint (MMP-9 ~ 5 nm diameter, HNE ~ 3 nm diameter) must fit in the SAM and reach the cleavage site at distance h_cut from the surface. h_cut (HNE) = L(PEG4/6) + L(AAPV) ≈ 1.44 + 1.5 = 2.9 nm from Au surface. Steric exclusion criterion: Γ_p < Γ_steric_max = 1/(π × r_enzyme²) ≈ 1/(π × (2.5nm)²) ≈ 5 pmol/cm²
- Post-cleavage state: cleaved at AAPV↓SGIG. Remaining stub on surface: Ac-Cys-PEG4/6-AAPV. This stub has no MB → contributes zero faradaic signal. Released fragment: SGIG-PEG2-Lys(MB)-NH2 → diffuses away (D_frag ~ 3×10^-10 m²/s)
- Negative control probe: scrambled sequence (e.g. VGASGIPA or whichever is selected) — identical PEG architecture, identical Cys anchor, identical MB reporter, different 8-mer. Same d_eff distribution, same k_ET, same Ca2+ artefact, no cleavage → Γ_control stays at Γ_0 throughout assay

### Layer 9: Enzyme kinetics on the surface

- Surface Michaelis-Menten: dΓ/dt = -kcat × [E]_surface × Γ / (Km_surf + Γ)
- Km_surf vs Km_solution: steric factor = Γ_steric_max / (Γ_steric_max + Γ_p). Km_surf = Km_solution / steric_factor → larger Km_surf at higher Γ_p. Literature correction factors: 2-20× depending on packing
- [E]_surface: from Layer 3 PDE at z=0 boundary — not necessarily equal to bulk [E]_0 if Da > 0.1
- Coupled system:
  - dΓ/dt = -kcat × [E](0,t) × Γ / (Km_surf + Γ)  ... (surface ODE)
  - ∂[E]/∂t = D_E × ∂²[E]/∂z² ... (bulk PDE, z ∈ [0,h])
  - D_E × ∂[E]/∂z|_{z=0} = kcat × [E](0,t) × Γ / (Km_surf + Γ)  ... (flux BC)
- Solution: numerical (finite difference for PDE, coupled Euler for ODE). Time step must satisfy CFL: Δt < (Δz)²/(2D_E)
- First-order limit: when Γ << Km_surf, dΓ/dt ≈ -(kcat/Km_surf) × [E] × Γ → exponential decay: Γ(t) = Γ_0 × exp(-kobs × t) where kobs = (kcat/Km_surf) × [E]. This is the regime where the assay is informative about kcat/Km
- Competition from interfering proteases: dΓ/dt = -sum_j [kcat_j × [E_j] × Γ / (Km_j + Γ)] for all proteases j that can cleave the substrate. Off-target contribution to F_k: δF_k = integral_0^T [kcat_interf × [E_interf]] dt / Km_interf

### Layer 10: Marcus-theory electron transfer

- Standard rate constant: k_ET^0(d) = A × exp(-β × d) × exp(-λ / (4RT))
  - β ≈ 1.0 Å^-1 for flexible organic tethers (PEG, peptide); β ≈ 1.1 Å^-1 for all-alkyl
  - λ = reorganisation energy of MB: λ ~ 0.3-0.5 eV (inner sphere ~0.1 eV, outer sphere ~0.2-0.4 eV)
  - A ~ k_BT/h × exp(-ΔG^‡_0/RT) ~ 10^12-10^13 s^-1
  - For d = 3 nm: k_ET^0 ~ 10^13 × exp(-30) ≈ 10 s^-1 (reasonable for surface-confined SWV)
- Potential-dependent rates (Butler-Volmer combined with Marcus):
  - k_f(η) = k_ET^0 × exp(-α × F × η / (RT)) (forward, cathodic branch reduces MB)
  - k_b(η) = k_ET^0 × exp((1-α) × F × η / (RT)) (backward, anodic branch oxidises MB)
  - α ≈ 0.5; η = E_applied - E°_MB
- Formal potential E°_MB:
  - E°_MB ≈ -0.25 V vs Ag/AgCl at pH 7.0
  - pH correction: dE°/dpH = -59 mV/pH unit (for 2e^-/2H^+ process)
  - At wound pH 6.5: E°_MB ≈ -0.22 V; at pH 8.0: E°_MB ≈ -0.34 V → 120 mV shift across operating range
- Formal potential E°_Fc: +0.45 V vs Ag/AgCl; pH-independent (outer-sphere ET, no proton transfer)
- Ca2+ correction: d_eff = d_0 - Δd([Ca2+]) from Layer 7 → k_ET^Ca = k_ET^0 × exp(β × Δd)
- Temperature correction: k_ET^0(T) = k_ET^0(T_ref) × exp(-(λ/4)(1/T - 1/T_ref) / k_B) via Marcus; also A ∝ T^0.5 prefactor
- Average k_ET for heterogeneous probe population: <k_ET> = integral k_ET(d) × P(d) dd where P(d) from Layer 8 conformational model
- After cleavage: Γ_intact(t) = Γ_0 - Γ_cleaved(t). Only Γ_intact contributes to MB peak. I_MB ∝ Γ_intact × <k_ET>

### Layer 11: SWV forward model

- SWV waveform: staircase ramp ΔE_step (typ. 2 mV) superimposed with symmetric square wave ΔE_SW (typ. 25 mV) at frequency f_SWV (typ. 60 Hz). Net current ΔI_n = I_forward,n - I_backward,n at each step
- Dimensionless kinetic parameter: κ_SW = k_ET^0 × (RT/(nF × f_SWV))^0.5. For quasi-reversible surface: κ_SW ~ 1 gives maximum net peak
- Fully numerical forward model (Osteryoung-O'Dea): for each potential step n:
  - Solve Butler-Volmer with time step = 1/(2f_SWV) for forward and backward half-cycles
  - Accumulate surface coverage of oxidised MB (Γ_ox) and reduced MB (Γ_red) = Γ_intact - Γ_ox
  - ΔI_n = n × F × A × (dΓ_ox/dt|_forward - dΓ_ox/dt|_backward)
- Peak current (approximate analytical result for quasi-reversible): I_peak ≈ n²F²AΓ_intact × k_ET^0 × ΔE_SW / (4RT)
- Peak position: at E = E°_MB when system is quasi-reversible; shifts cathodic for slow k_ET (κ_SW << 1); use peak shift to extract k_ET^0 from data
- Capacitive contribution: ΔI_cap = 2 × C_dl × A × ΔE_SW × f_SWV — constant background added to each step; approximately cancels in net current (forward - backward) but not perfectly for asymmetric C_dl(E)
- MB and Fc peak resolution: ΔE° = E°_Fc - E°_MB = +0.45 - (-0.25) = 0.70 V. Peak FWHM (reversible surface wave) = 5.56 × RT/(nF) ≈ 65 mV (n=2 for MB) and ≈ 131 mV (n=1 for Fc). Sum of half-widths < 0.7V → peaks are resolved. Verify numerically for quasi-reversible case (peaks broaden and overlap more)
- Scan window: E_min to E_max must contain both peaks with >200 mV baseline either side. Recommended: -0.65 V to +0.65 V vs Ag/AgCl. Check SAM reductive desorption does not occur at -0.65 V (typical threshold -0.9 to -1.1 V for alkanethiols)
- Parameter optimisation: sweep (f_SWV, ΔE_SW, ΔE_step) to maximise ΔI_MB / I_cap ratio. ΔI_MB ∝ ΔE_SW × f_SWV × (quasi-reversible factor); I_cap ∝ ΔE_SW × f_SWV → ratio is approximately constant with frequency, but changes with κ_SW
- SWV for Fc reference: at same scan, Fc peak appears at +0.45 V. If k_ET^0_Fc >> k_ET^0_MB (expected: C6 tether d ~ 0.9 nm → k_ET^0_Fc ~ 10^4 s^-1), then f_SWV = 60 Hz may give κ_SW >> 1 for Fc (reversible limit) and κ_SW ~ 1 for MB (quasi-reversible). Both peaks accessible in single scan — good design
- Frequency diagnostic: if measured ΔI_peak ∝ f^0.5 → diffusion-controlled (bad: means non-tethered MB in solution is dominating). If ΔI_peak ∝ f^1 → surface-confined (correct regime). Must verify

### Layer 12: SWV peak extraction and signal processing

- Raw data: array of (E_n, ΔI_n) with ~ (ΔE_total/ΔE_step) = (1300 mV / 2 mV) = 650 points per scan
- Smoothing: Savitzky-Golay filter (window ~ 11 points, polynomial order 3) to reduce high-frequency noise while preserving peak shape
- Baseline subtraction: cubic spline through user-defined or auto-selected non-peak regions (e.g., -0.65 to -0.45 V and -0.05 to +0.25 V for the MB channel). Subtracts sloping capacitive background
- Peak fitting: Gaussian model ΔI(E) = A × exp(-(E-E_peak)²/(2σ²)) fitted separately to MB region (-0.45 to -0.05 V) and Fc region (+0.25 to +0.65 V). Fit parameters: A (amplitude), E_peak (peak position), σ (width). Area = A × σ × sqrt(2π)
- Use peak area (not height) as the signal: area integrates over the full peak regardless of peak position shift with k_ET or pH → more robust to temperature and Ca2+ artefacts
- Two-Gaussian fit: simultaneously fit both peaks to check for overlap and cross-talk between channels
- Peak position shift monitoring: δE_peak from scan to scan is a diagnostic for SAM degradation, k_ET changes, or reference electrode drift
- Quality control per scan: if peak width σ > 2× expected (65 mV) → SAM heterogeneous or measurement error → flag scan

### Layer 13: Ratiometric signal calculation with error propagation

- Per electrode i, SWV peak areas: A_MB,i(t) and A_Fc,i(t)
- Ratiometric signal: R_i(t) = A_MB,i(t) / A_Fc,i(t)
  - Uncertainty: σ_R/R = sqrt[(σ_MB/A_MB)² + (σ_Fc/A_Fc)²] where σ from peak fit covariance
- Baseline measurement R_i(0): taken from T=0 scan (immediately after chamber fill, before significant cleavage). For incubation time T = 30 min and kobs × T ~ 0.1-0.5: R_i(0) captures ~5-15% cleavage → must correct or use T=0 extrapolation
- Normalised signal: ρ_i = R_i(T) / R_i(0). σ_ρ/ρ = sqrt(2) × σ_R/R (if both measurements have equal variance, and R_i(0) is from a single early scan with the same noise)
- Active/control double ratio:
  - ρ_active,k = normalised signal from WE k with active probe (AAPVSGIG for HNE, IPRTLTAA for MMP-9)
  - ρ_control,k = normalised signal from WE k (or neighbouring paired electrode) with scrambled probe
  - F_k = 1 - ρ_active,k / ρ_control,k
  - σ_F = |ρ_active/ρ_control| × sqrt[(σ_ρa/ρ_a)² + (σ_ρc/ρ_c)²]
- Fouling cancellation: assumes fouling affects A_MB equally on active and control probes. Valid if (a) both electrodes have same diluent SAM composition and (b) fouling is a bulk matrix effect (adsorption of albumin etc.) rather than specific to the probe structure. If fouling is probe-sequence dependent (e.g., albumin binds to AAPVSGIG but not scrambled) → imperfect cancellation → must characterise
- Apparent rate: κ_k = -ln(1 - F_k) / T. Valid derivation: assumes dΓ/dt = -k_obs × Γ (first-order surface kinetics with constant [E]) → Γ(T) = Γ_0 × exp(-k_obs × T) → F_k = 1 - exp(-k_obs × T) → κ_k = k_obs. Breaks down if Da > 0.1 (enzyme-depleted surface → kobs is time-varying) or if saturation (Γ > Km_surf, zeroth order). Must verify first-order assumption for expected [E] and T
- Dynamic range: F_k is useful in range [F_min, F_max] = [3 × σ_F, 0.90]. Outside this range: below LOD or approaching saturation (κ_k diverges). Translates to κ range: [κ_min, κ_max] = [-ln(0.97)/T, -ln(0.10)/T] = [0.1/T, 2.3/T]

### Layer 14: Electronics and potentiostat noise budget

- TIA: transimpedance amplifier, gain R_f. Select R_f such that I_max × R_f < V_rail:
  - I_max per electrode: sum of capacitive + faradaic peak: ~ 200 nA. For R_f = 100 kΩ: V_out_max = 20 mV — safely in range
- Current noise sources:
  - Johnson noise in R_f: i_J = sqrt(4 k_B T BW / R_f). For R_f=100 kΩ, BW=100 Hz: i_J = 0.4 pA/√Hz → i_J_total ≈ 4 pA over BW
  - Op-amp input current noise: i_n_amp ~ 1-10 fA/√Hz for low-noise FET input → negligible
  - Shot noise from DC bias current: i_shot = sqrt(2 e I_DC BW). I_DC ≈ C_dl × A × dV/dt ≈ 60 nA during scan. i_shot = sqrt(2 × 1.6e-19 × 60e-9 × 100) = 44 fA — negligible vs Johnson
- ADC: 24-bit (DStat architecture). For input range ±5 mV (with R_f = 100 kΩ, I_range = ±50 nA): LSB = 100 nA / 2^24 = 6 fA. ADC quantisation noise sigma_ADC = LSB/sqrt(12) ≈ 2 fA — negligible
- Total current noise: i_total ≈ i_J ≈ 4 pA per measurement point. For peak area (integrating over ~50 points per peak): sigma_area ~ 4 pA × sqrt(50) × 2 mV / sigma ≈ need numerical integration
- Required SNR: to resolve F_LOD = 5%, need sigma_F < 1.7%. With ρ_active = ρ_control = 0.95 (5% change): sigma_F/F ~ sqrt(2) × sigma_R/R ~ sqrt(2) × 1.7% = 2.4% → need sigma_R/R < 1.7%. With A_MB ~ 1 pA·V and sigma_A_MB ~ 4 pA × 50pts × 2mV = 400 fA·V: SNR_peak = 1 pA·V / 0.4 fA·V ~ 2500 → sigma_A/A = 0.04% — more than sufficient
- Multiplexer: switching between 6-8 WEs. After switch: settling time tau = R_s × C_dl × A ~ 33 Ω × 2e-8 F = 660 ns → settling complete in < 1 ms before first SWV step. No issue
- EMI/mains pickup: 50 Hz (EU) in same frequency range as SWV (60 Hz). Must use shielded enclosure or synchronise f_SWV to avoid beating with mains. Alternative: run f_SWV at 50 Hz in non-EU or choose f_SWV >> 100 Hz (but check κ_SW)
- DAC output noise (SWV waveform): 16-bit DAC (DStat). Step resolution = 5V / 2^16 = 76 µV. For ΔE_step = 2 mV: DAC introduces <4% voltage quantisation per step — acceptable but should use DAC with output filter

### Layer 15: Calibration curve and limit of detection

- Calibration experiment: spike known concentrations of active MMP-9 and active HNE into wound-fluid-matched matrix (saline + 4 mg/mL BSA + relevant inhibitor concentrations). Measure F_k for each channel at each concentration
- Calibration model: F_k = 1 - exp(-k_obs × T) where k_obs = (kcat/Km_surf) × [E]. In first-order regime: linear F_k ~ (kcat/Km_surf) × [E] × T for F_k << 1
- LOD = 3 × sigma_blank / slope_calibration where slope = dF_k/d[E] at [E]=0
- LLOQ = 10 × sigma_blank / slope_calibration (lower limit of quantification)
- Dynamic range: [LLOQ, [E]_saturation] where saturation is when Da > 1 or Γ << Km_surf fails
- Expected LOD: from noise analysis sigma_F ~ 1-2%. Slope = (kcat/Km) × T ~ (10^4 M^-1 s^-1) × (1800 s) = 1.8×10^7 M^-1. LOD ~ 0.01 / (1.8×10^7) ≈ 0.5 nM active protease. Compare with reported wound fluid MMP-9: 1-100 nM → LOD well within clinically relevant range
- Matrix interference on calibration: perform spike-and-recovery experiment across wound pH range (6.5, 7.0, 7.4, 8.0) and across [Ca2+] range (0.1-2 mM) to map calibration curve shift
- Two-channel deconvolution:
  - F = S × c where F = [F_MMP9_channel, F_HNE_channel]^T, c = [c_MMP9, c_HNE]^T, S = sensitivity matrix
  - S_11 = dF_MMP9_channel/dc_MMP9 (from MMP-9 calibration); S_12 = dF_MMP9_channel/dc_HNE (cross-sensitivity); etc.
  - c_estimated = S^-1 × F_observed. Condition number kappa(S) = |S| × |S^-1| determines how noise in F amplifies into error in c
  - Connection to substrate design: kappa(S) < 30 was the criterion from Panel A/B. The selectivity of the substrate sequences (IPRTLTAA vs AAPVSGIG) determines off-diagonal elements of S. Good substrate selectivity → S close to diagonal → kappa(S) small → accurate deconvolution

### Layer 16: Full chain integration and falsifiable predictions

- Forward simulation: given [MMP-9_active]_wound, [HNE_active]_wound, T_incubation, T_assay, wound pH, [Ca2+]:
  1. Layer 0 ODE → [E]_wound
  2. Layer 1-2 → [E]_chamber = [E]_wound × D_f (dilution)
  3. Layer 3 PDE → [E](z=0, t) (surface enzyme concentration over time)
  4. Layer 7 → Γ_p, Γ_d (SAM composition)
  5. Layer 8 → d_eff distribution, steric factor
  6. Layer 9 coupled ODE-PDE → Γ_intact(t)
  7. Layer 10 → <k_ET> from d_eff and Ca2+ correction
  8. Layer 11 → SWV forward model → A_MB(t), A_Fc(t) (peak areas)
  9. Layer 13 → R(t), ρ, F_k for each electrode
  10. Layer 14-15 → add noise, compute σ_F, check against LOD
- Falsifiable output: predicted F_k(T) curve as function of [E]_input → compare to calibration experiment
- Sensitivity analysis (Sobol indices): which layer parameters contribute most variance to σ_F? Likely candidates: Γ_p (from SAM), kcat/Km_surf (from wet lab), d_eff (from conformational model)
- Wet lab integration points:
  - kcat/Km from pES-01 to pES-08 → feeds Layer 9 (replaces literature estimate)
  - Substrate ranking (Spearman correlation) → validates Layer 9 relative ordering of Γ(T) across probes
  - Cross-reactivity matrix → validates S matrix condition number (kappa < 30 criterion)

---

## Literature parameters to gather (data/literature_parameters/)

| Parameter | Value | Source needed |
|---|---|---|
| D_MMP9 (92 kDa) | ~ 5×10^-11 m²/s | Stokes-Einstein / experimental |
| D_HNE (30 kDa) | ~ 9×10^-11 m²/s | Stokes-Einstein / experimental |
| kcat/Km MMP-9 on IPRTLTAA | — | pES-01 to pES-08 / MEROPS |
| kcat/Km HNE on AAPVSGIG | — | pES-01 to pES-08 / MEROPS |
| beta (tunnelling decay) | ~ 1.0 Å^-1 | Smalley et al. 2003, Sikes et al. 2001 |
| lambda (MB reorganisation energy) | ~ 0.3-0.5 eV | Pheeney & Barton 2012, Kelley et al. |
| E°_MB vs Ag/AgCl | ~ -0.25 V (pH 7) | Kelley et al. 1997 |
| E°_Fc vs Ag/AgCl | ~ +0.45 V | Nowicka et al. |
| C_dl (SAM-modified gold) | ~ 1-5 µF/cm² | Campuzano et al., Rant et al. |
| Gamma_p optimal | 1-10 pmol/cm² | Lubin & Plaxco 2010 |
| Wound fluid [Ca2+] | 1-2 mM | Literature wound fluid composition |
| Wound fluid pH range | 6.5-8.2 | Gethin et al. |
| l_p PEG | ~ 0.38 nm | Oesterhelt et al. 1999 |
| Rf evaporated gold | 1.1-1.5 | Roughness literature |

---

## Immediate next steps (in order)

1. Generate scrambled negative control: run `substrate_predictor/predict.py` over all unique permutations of {A,A,P,V,S,G,I,G}, score with HNE and MMP-9 models, select lowest-scoring candidate
2. Write LaTeX report `reports/report_device_modelling.tex` covering all 16 layers with equations
3. Populate `data/literature_parameters/` with CSV of all numeric parameters + DOI sources
4. Implement Layer 3 (fluidics PDE) and Layer 9 (surface kinetics ODE-PDE) as Python in `models/03_fluidics/` and `models/05_surface_kinetics/`
5. Implement Layer 11 (SWV forward model) in `models/07_swv_forward/` — this is the single most critical modelling component
6. Chain layers 0-16 into a single forward simulation in `models/forward_chain.py`
7. Run sensitivity analysis (SALib Sobol) to identify top uncertainty drivers
