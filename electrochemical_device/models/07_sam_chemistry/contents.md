# Layer 7: SAM formation and composition

## Purpose

The self-assembled monolayer (SAM) is the functional interface between the gold electrode and the biological sample. It defines: (a) how many probe molecules are present per unit area (Gamma), (b) the antifouling protection against non-specific protein adsorption, and (c) the physical environment experienced by each probe. Everything downstream — enzyme kinetics, electron transfer, SWV signal — depends directly on Gamma and SAM composition.

The SAM is formed before the device is used (during manufacturing or immediately before use). The device has no feedback on SAM quality. This makes characterisation and tight manufacturing control essential, and robust predictive modelling of SAM composition from deposition conditions is the goal of this layer.

## Sub-components to model

### 7.1 Thiol adsorption kinetics

Gold-thiol bond formation (for the Cys terminus of the probe):
    Au + R-SH → Au-SR + H+

Kinetics follow a two-step process:
1. Fast Langmuir adsorption (seconds to minutes): direct chemisorption from solution, governed by rate k_ad = k_ad^0 × c_thiol
2. Slow reorganisation (hours): surface diffusion and packing of adsorbed molecules, independent of solution concentration

For a mixed SAM of probe-thiol (species p, bulky) and diluent-thiol (species d, small, e.g. 6-mercaptohexanol, MCH):
    dGamma_p/dt = k_ad,p * c_p * (Gamma_max - Gamma_p - Gamma_d) - k_des,p * Gamma_p
    dGamma_d/dt = k_ad,d * c_d * (Gamma_max - Gamma_p - Gamma_d) - k_des,d * Gamma_d

where Gamma_max is the maximum possible surface coverage (packing limit). The diluent MCH is smaller → k_ad,d >> k_ad,p (faster adsorption due to lower steric cost). This means the surface is transiently enriched in diluent before the probe has time to adsorb.

In practice, for disposable cartridges: SAM is pre-formed by incubating the gold electrode in a mixed thiol solution (e.g., 1 µM probe + 100 µM MCH in ethanol) for 2-16 hours, then rinsing. The equilibrium surface composition is what matters for the device.

### 7.2 Equilibrium surface composition (competitive Langmuir)

At equilibrium, competitive Langmuir adsorption gives:
    Gamma_p / Gamma_max = (K_a,p * c_p) / (1 + K_a,p * c_p + K_a,d * c_d)
    Gamma_d / Gamma_max = (K_a,d * c_d) / (1 + K_a,p * c_p + K_a,d * c_d)

The surface mole fraction of probe:
    chi_p = Gamma_p / (Gamma_p + Gamma_d) = (K_a,p * c_p) / (K_a,p * c_p + K_a,d * c_d)

K_a,p/K_a,d is the selectivity ratio. Literature values for bulky thiol vs MCH: K_a,p/K_a,d ≈ 0.01-0.1 (probe is less strongly adsorbed due to steric cost of the large peptide headgroup).

Target: Gamma_p ≈ 2-5 pmol/cm^2 (from Layer 8 steric accessibility requirement). Must solve for the solution concentration ratio c_p/c_d that gives this surface Gamma_p.

Example: if K_a,p/K_a,d = 0.05 and we want chi_p = 0.01 (1% of surface is probe):
    c_p/c_d = chi_p * (K_a,d/K_a,p) / (1 - chi_p) = 0.01 * 20 / 0.99 ≈ 0.20
    So c_p = 0.20 * c_d. For c_d = 100 µM: c_p = 20 µM.

Note: Gamma_p = chi_p * (K_a,p * c_p * Gamma_max) / (1 + K_a,p * c_p + K_a,d * c_d). This depends on the absolute adsorption constants. Must fit from literature calibration curves.

### 7.3 Phase separation and nanodomain formation

Mixed thiols do not always form a homogeneous mixed monolayer. Thiols with mismatched chain lengths or head groups tend to phase-separate into nanoscale domains (5-50 nm diameter), visible by AFM, STM, or NSOM.

For probe (large, peptide headgroup) + MCH (small, hydroxyl headgroup): large size mismatch → strong tendency to phase-separate. This creates local regions of pure probe (high Gamma_p locally, poor enzyme access) and pure MCH (zero probe coverage, no signal contribution).

Consequence: effective local Gamma_p in probe-rich domains > mean Gamma_p. Enzyme access is governed by local (not mean) Gamma_p. Must use a domain model:
    Gamma_p,local = chi_p,local * Gamma_max (in probe-rich domains)
    enzyme access efficiency = f(Gamma_p,local, r_enzyme)

If Gamma_p,local > Gamma_steric_max: no enzyme access in those domains → effective cleavage only in mixed domains or at domain boundaries.

Mitigation: use a diluent with a longer chain or a PEG diluent (more compatible with the PEG spacer of the probe) to reduce phase separation. PEG-SH diluents are better matched to PEG-spacered probes.

### 7.4 Fc reference electrode SAM

The Fc control electrode uses C6-Fc-SH as the probe thiol, co-deposited with MCH diluent. Fc must be present at low surface coverage to behave as isolated, non-interacting redox centres.

Fc-Fc through-space electron hopping (electron 'hopping' between adjacent Fc centres): significant when Fc-Fc separation < ~2 nm, corresponding to Gamma_Fc > ~5 pmol/cm^2 (for a close-packed Fc monolayer: 1/(r_Fc^2) ≈ 1/(1 nm)^2 ≈ 10^-9 mol/cm^2 = 1000 pmol/cm^2 at close packing; hopping becomes important at Gamma > ~0.5-5 pmol/cm^2 depending on geometry).

Keep Gamma_Fc < 3 pmol/cm^2. Use a solution ratio c_Fc / c_MCH consistent with this target Gamma_Fc (same competitive Langmuir model, but with different K_a,Fc).

Diagnostic: Fc SWV peak should be symmetric (a symmetric surface wave at E°_Fc = +0.45 V). Asymmetric peak or broadened peak indicates Fc-Fc coupling → Gamma_Fc is too high → reduce c_Fc in deposition.

### 7.5 Edinburgh Ca2+ artefact — mechanistic model

Discovered at Edinburgh University: 1 mM Ca2+ in the sample increases the SWV MB peak height even without cleavage. Mechanism:

PEG spacer in the probe (PEG4 or PEG6) can coordinate divalent cations (Ca2+, Mg2+, Zn2+). Ethylene oxide oxygens of PEG form a pseudo-crown-ether complex with divalent cations. When Ca2+ is complexed by PEG, the PEG chain contracts (less solvated, less extended) → the MB reporter moves closer to the gold surface → shorter effective tunnelling distance d_eff → faster k_ET → higher SWV peak.

Quantitative model:
    Theta_Ca = [Ca2+] / (K_d + [Ca2+])  (fraction of PEG chelating Ca2+)

K_d for PEG-Ca2+ ≈ 0.5-5 mM (literature for analogous polyether complexes).

    delta_d = Theta_Ca * delta_d_max

where delta_d_max is the maximum contraction of the PEG spacer upon Ca2+ complexation (estimated from the persistence length reduction in chelated PEG: ~0.1-0.5 nm for PEG4-6).

    k_ET^Ca = k_ET^0 * exp(beta * delta_d)

For delta_d = 0.3 nm, beta = 1.0 Å^-1: k_ET^Ca / k_ET^0 = exp(10 * 0.3) = exp(3) ≈ 20×. This is a large effect! At 1 mM Ca2+ with K_d = 2 mM: Theta_Ca = 0.33; delta_d_Ca = 0.1 nm → k_ET^Ca = exp(1) ≈ 2.7× increase.

Crucially: the Ca2+ artefact affects BOTH the active probe electrode and the scrambled control electrode equally (same PEG spacer architecture). In the ratiometric F_k = 1 - rho_active/rho_control, this Ca2+ effect multiplies both R_active and R_control by the same factor, cancelling in the ratio.

BUT: this cancellation only holds if:
(a) Gamma_p is the same on active and control electrodes (so the Ca2+ effect is the same magnitude)
(b) [Ca2+] is the same throughout the chamber (uniform mixing)

If the SAM deposition gives different Gamma_p on active vs control electrodes, the Ca2+ artefact does not perfectly cancel. Model: residual Ca2+ error in F_k as a function of delta_Gamma_p (inter-electrode Gamma variation).

### 7.6 SAM and probe hydration on first sample contact

The device is manufactured and stored dry (desiccated, typically in a foil pouch). When sample first contacts the SAM and probe, a hydration step occurs:

1. Water ingress into the SAM: within ~1-5 s, water molecules penetrate the outer SAM surface and solvate the PEG spacer and peptide headgroup.
2. PEG hydration: PEG adopts a helical H-bonded structure in water, taking ~0.1-1 s to reach hydrated equilibrium (fast).
3. Peptide conformational equilibration: the 8-residue peptide transitions from a dry (possibly collapsed or adsorbed) state to its solution ensemble. Time ~1-10 s (depends on sequence — disordered peptides are fast).
4. MB solvation: methylene blue hydrates essentially instantaneously.

If the T=0 SWV scan is taken within 30 s of fill completion, the probe may still be partially in the dry-contact (non-equilibrium) conformation. This would give an anomalously high k_ET^0 (shorter d_eff while dry) and an anomalously high R(0).

Model: delay T=0 scan by at least 60 s after fill completion. Alternatively, take T=0 scan and a T=2 min scan; if R(T=0) and R(T=2 min) agree within 5%, the probe is fully hydrated at T=0.

This constraint should be included in the firmware timing specification for the device.

### 7.7 Wound redox agents and MB chemical oxidation/reduction

Wound fluid contains endogenous redox-active species that co-elute into the device and can alter MB oxidation state at the electrode surface without an applied electrochemical potential. This is a critical gap in the current model.

**Reducing agents:**
- Glutathione (GSH): ~0.1-1 mM in wound fluid (intracellular, released by lysed cells). In the chamber at [GSH]_chamber ≈ 0.05-0.5 mM after dilution.
- Ascorbate (vitamin C): ~0.05-0.2 mM.

GSH can chemically reduce oxidised MB (MB²⁺, the form detected by SWV reduction peak) to leuco-MB or MB•+ radical:
    MB²⁺ + 2 GSH → MB + GSSG + 2H⁺
Rate constant k_red ≈ 10-1000 M^-1 s^-1 (literature for MB reduction by thiols; strongly pH-dependent).

At [GSH] = 0.5 mM: pseudo-first-order rate = 500 × k_red s^-1. For k_red = 100 M^-1 s^-1: rate = 0.05 s^-1 → half-life 14 s. MB at the electrode surface could be significantly reduced to electroinactive leuco-MB within minutes of sample contact.

This reduces the MB SWV peak area independently of enzyme cleavage, causing a false-positive F_k signal.

**Oxidising agents:**
- H2O2: 10-100 µM in inflamed wound fluid.
- Hypochlorite (HOCl): 1-100 µM from myeloperoxidase.

H2O2 and HOCl oxidise MB to MB-N-oxide (thionine N-oxide), which has a shifted formal potential and reduced electroactivity. This also reduces peak area but in a different direction.

**Model:**
    d[MB_intact]/dt = -k_red * [GSH] * [MB_intact] - k_ox * [H2O2] * [MB_intact]

where [MB_intact] is the surface concentration of electroactive MB.

The ratiometric design DOES NOT cancel this effect if it preferentially affects the active probe electrode (e.g., if the scrambled control probe has a different local chemical environment that slows redox access). Must verify: is the MB on the intact probe (d_eff ≈ 5 nm from surface) equally accessible to solution GSH as MB on the cleaved stub (not present) or the scrambled control?

Mitigation: design elution buffer to include an MB-protecting agent (e.g., buffer with ascorbate scavenging pre-oxidised, or nitrogen purge to remove O2 and reduce oxidative stress). Or include a GSH-correction term in the calibration model.

### 7.8 SAM stability during assay

Au-S bond is thermally stable (k_des at 25°C is extremely slow, half-life >> years). During the 30-min assay, spontaneous SAM desorption is negligible.

However:
- Oxidative desorption: H2O2 in wound fluid (Layer 0.5) can oxidise the Au-S bond to Au + RSOH (sulphenic acid) or Au + RSO2H (sulphinic acid). Oxidation rate: k_ox,thiol * [H2O2]. At [H2O2] = 50 µM and k_ox,thiol ≈ 1 M^-1 s^-1: rate = 5×10^-5 s^-1 → 0.3% loss in 60 s. Negligible.
- Electrochemical desorption: at E < E_des ≈ -0.9 V. Our scan limit is -0.65 V → safe (Layer 6.5).
- Displacement by competing adsorbates: wound fluid thiols (cysteine, glutathione at ~µM concentrations) can exchange with probe thiols. Cys concentration in wound fluid: ~5-20 µM. Exchange rate at this concentration and typical k_exchange: probably < 1% exchange per 30 min. Must verify.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| Gamma_p (probe surface coverage) | 2-5 pmol/cm^2 | Layers 8, 9, 11, 13 |
| Gamma_d (diluent coverage) | 50-200 pmol/cm^2 | Layer 8 (conformation) |
| chi_p (surface probe fraction) | 0.005-0.05 | Layer 9 (steric factor) |
| delta_d_Ca(Ca2+) | 0.05-0.3 nm | Layer 10 (k_ET correction) |
| Ca2+ cancellation validity | Yes if Gamma_p uniform | Layer 13 (F_k error) |
| SAM stability | > 98% over 30 min | Assay validity |

## Key literature to find

- Competitive thiol SAM adsorption: Kakiuchi et al. (Langmuir), Steel et al.
- Phase separation in mixed SAMs: Stranick et al. (Science, 1994)
- PEG-Ca2+ complexation: Saenger (Structure, 1980) for crown-ether analogy; specific PEG-Ca2+ Kd from Branca et al.
- Edinburgh Ca2+ artefact: original observation in Plaxco group (cite Edinburgh iGEM 2025 or original Plaxco lab papers)
- Fc SAM hopping: Chidsey & Murray (Science, 1986)
- Thiol exchange in biological matrices: Perez-Luna et al.

## Files to create

- `sam_adsorption.py`: competitive Langmuir ODE system; equilibrium Gamma_p as function of c_p, c_d, K_a ratios
- `calcium_artefact.py`: theta_Ca model; delta_d_Ca(Ca2+); residual error in F_k from inter-electrode Gamma variation
- `sam_stability.py`: estimate % SAM lost to oxidation, exchange, and electrochemical desorption over assay time
