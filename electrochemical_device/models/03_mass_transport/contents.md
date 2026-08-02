# Layer 3: Mass transport inside the chamber (incubation phase)

## Purpose

After the chamber fills, the device sits static during incubation. There is no flow. All enzyme transport to the electrode surface is by diffusion. This layer models the concentration profile of enzyme in the chamber as a function of time and position, including the depletion layer that builds up at the electrode surface as cleavage proceeds.

This layer is arguably the most physically critical: it determines whether the assay operates in the reaction-limited regime (signal is proportional to kcat/Km, which is what we want) or the transport-limited regime (signal reflects diffusion speed, not enzyme activity). The Damköhler number analysis here is a key defensible result.

## Sub-components to model

### 3.1 Diffusion coefficients

Stokes-Einstein equation:
    D = k_B * T / (6 * pi * eta * r_h)

where r_h is the hydrodynamic radius. For globular proteins: r_h ≈ 0.066 * M_W^0.333 nm (M_W in Daltons).

Target values:
- MMP-9 (92 kDa): r_h ≈ 2.94 nm → D_MMP9 ≈ 5.0 × 10^-11 m²/s at 25°C
- HNE (30 kDa): r_h ≈ 2.06 nm → D_HNE ≈ 9.2 × 10^-11 m²/s at 25°C
- Released MB peptide fragment (SGIG-PEG2-Lys(MB), ~1.0 kDa): r_h ≈ 0.66 nm → D_frag ≈ 3.2 × 10^-10 m²/s

Note: MMP-9 exists as a monomer (92 kDa) and dimer (184 kDa) in wound fluid. The dimer has D ≈ D_monomer / 2^(1/3) ≈ 0.79 × D_monomer. If both forms are present, treat as two separate diffusing species. Pro-MMP-9 (92 kDa) is similar in size to active MMP-9.

Temperature correction: D(T) = D(T_ref) × (T/T_ref) × (eta_ref/eta(T)) (Stokes-Einstein).

### 3.2 Vertical mixing time (critical insight)

The channel height h ∈ [0.25, 0.50] mm is small, but the diffusion coefficients for large proteins are very small. Vertical mixing timescale:
    t_mix = h^2 / (2 * D)

For h = 0.375 mm (midpoint), MMP-9:
    t_mix = (3.75 × 10^-4)^2 / (2 × 5 × 10^-11) ≈ 1406 s ≈ 23 min

This is comparable to the incubation time (30 min). The chamber is NOT well-mixed during incubation for MMP-9. This is a fundamental physical fact that has major consequences:
- The enzyme concentration at the electrode surface is not [E]_0 throughout the incubation
- A depletion layer builds at the surface as enzyme is consumed
- The PDE must be solved to get the correct [E](z=0, t)

For HNE (D ≈ 1.8× larger): t_mix ≈ 13 min — still comparable to incubation time.

### 3.3 Governing equations (1D PDE in z-direction)

Let z be the vertical coordinate, z=0 at the electrode surface, z=h at the PDMS lid.

Diffusion PDE:
    ∂[E]/∂t = D_E * ∂²[E]/∂z²,   z ∈ (0, h),  t > 0

Initial condition (uniform after fill):
    [E](z, 0) = [E]_chamber  (from Layer 1)

Boundary conditions:
    ∂[E]/∂z|_{z=h} = 0  (no-flux at lid)
    D_E * ∂[E]/∂z|_{z=0} = J_surface(t)  (flux consumed by surface reaction)

The surface flux J_surface(t) comes from Layer 9 (surface kinetics):
    J_surface(t) = k_cat * [E](z=0, t) * Gamma(t) / (K_m,surf + Gamma(t))

This makes the system a coupled PDE (in z,t) + ODE (Gamma(t)) linked at z=0.

### 3.4 Numerical solution strategy

Discretise in z using finite differences (N_z = 50 grid points, dz = h/50).

Explicit method (FTCS scheme): stable for dt < (dz)^2 / (2 D_E). For dz = 7.5 µm and D_MMP9 = 5×10^-11 m²/s: dt_max = (7.5×10^-6)^2 / (2 × 5×10^-11) = 0.56 s. Total incubation time 1800 s → N_t = 1800/0.56 ≈ 3200 time steps. Computationally trivial.

Alternative: Crank-Nicolson (unconditionally stable, second-order in time) — allows larger dt at the cost of solving a tridiagonal system at each step. Preferred for accuracy.

Coupling: at each time step t_n, use [E](z=0, t_n) from the PDE to advance Gamma(t_n) via the Layer 9 ODE, then use J_surface(t_n) as the z=0 boundary condition for the next PDE step.

### 3.5 Damköhler number analysis

The Damköhler number Da characterises the ratio of the characteristic reaction rate to the characteristic diffusion rate:
    Da = (k_cat * Gamma_0 / K_m,surf) / (D_E * [E]_0 / h^2)

Interpretation:
- Da << 1: reaction is slow compared to diffusion → electrode surface sees [E] ≈ [E]_0 throughout → signal reports kcat/Km (desired)
- Da >> 1: diffusion is the bottleneck → signal reports D_E, not kcat/Km (wrong regime)
- Da ≈ 1: mixed regime; both matter

Target: Da < 0.1 for the design to be interpretable.

Key parameters: Gamma_0 ~ 5 pmol/cm^2 = 5×10^-11 mol/cm^2; K_m,surf ~ 10 × K_m,sol (steric correction); k_cat, K_m from pES experiments; [E]_0 from Layer 1.

Compute Da for expected parameter range and report the regime. If Da > 1, the probe design (higher Gamma or different assay time) must be changed, or the signal interpretation must switch to a transport-rate model.

### 3.6 MB fragment clearance

After cleavage of AAPV↓SGIG, the released fragment (SGIG-PEG2-Lys(MB)) has D_frag ≈ 3×10^-10 m²/s. It diffuses away from the electrode into the bulk. The fragment carries MB, so if it remains near the electrode during SWV, it will give a diffusion-controlled (not surface-confined) MB peak that distorts the measurement.

Time for the fragment to diffuse beyond the Nernst diffusion layer (delta ≈ 50-100 µm for quiescent solution):
    t_clear = delta^2 / D_frag = (50×10^-6)^2 / (3×10^-10) ≈ 8 s

This is fast. However, fragment accumulates in the bulk over the 30-min incubation. If F_k is large (many probes cleaved), bulk MB concentration could be significant. Check: total MB released = Gamma_0 * A_electrode * F_k. For Gamma_0 = 5 pmol/cm^2, A = 1 mm^2, F_k = 0.5: total MB released = 5×10^-11 × 0.01 × 0.5 = 2.5×10^-13 mol. Volume of chamber ~ 37.5 µL = 3.75×10^-8 L. Bulk [MB] ~ 2.5×10^-13 / 3.75×10^-8 ≈ 6.7 nM. This is very small compared to typical SWV detection limits in solution (~100 nM). Fragment contribution to signal: negligible.

### 3.7 Electrode cross-talk

Released MB fragment from electrode i can diffuse laterally to electrode j. Lateral diffusion time over separation distance d_elec:
    t_cross = d_elec^2 / (2 * D_frag)

For d_elec = 2 mm: t_cross = (2×10^-3)^2 / (2 × 3×10^-10) ≈ 6700 s ≈ 112 min >> incubation time.

Cross-talk is negligible for electrode separations > 1 mm and incubation times < 60 min. But MB remains in the bulk indefinitely (does not degrade). Verify: during SWV, the lateral fragment concentration at electrode j due to cleavage at i is estimated from the 2D diffusion Green's function at t = T_incubation. For T = 30 min: negligible as shown above.

### 3.8 Lateral transport during fill (Pe number during fill)

During the fill phase (capillary flow, velocity u ≈ L/t_fill), the Péclet number for MMP-9:
    Pe = u * L / D_MMP9 = (L^2 / t_fill) / D_MMP9

For L = 20 mm, t_fill = 60 s: u = 0.33 mm/s. Pe = 0.33×10^-3 × 20×10^-3 / 5×10^-11 ≈ 1.3×10^5. Highly convective during fill → enzyme is uniformly distributed along the channel length by the time fill completes. This justifies the uniform IC [E](z,0) = [E]_chamber.

### 3.9 In-device inhibitor binding kinetics

This sub-component is the most important addition to this layer identified in lifecycle analysis. After the sample fills the chamber, co-eluted TIMP-1 and A1AT/SLPI continue to bind and inactivate free MMP-9 and HNE in the bulk solution. This depletes [E_free] over time independently of surface cleavage, and is NOT captured by the pure diffusion PDE in 3.3.

The governing system becomes a reaction-diffusion PDE for each enzyme/inhibitor pair. For MMP-9 and TIMP-1:

    ∂[MMP9_free]/∂t = D_MMP9 * ∂²[MMP9_free]/∂z² - k_on,TIMP * [MMP9_free] * [TIMP1_free]
    ∂[TIMP1_free]/∂t = D_TIMP1 * ∂²[TIMP1_free]/∂z² - k_on,TIMP * [MMP9_free] * [TIMP1_free]
    ∂[complex]/∂t   = D_complex * ∂²[complex]/∂z² + k_on,TIMP * [MMP9_free] * [TIMP1_free]

Initial conditions: [MMP9_free](z,0) = [E]_chamber from Layer 1; [TIMP1_free](z,0) = [TIMP-1]_chamber from Layer 1; [complex](z,0) = 0.

Note: TIMP-1 is a tight-binding inhibitor of MMP-9 (K_i ~ 0.01 nM). The association rate k_on,TIMP ≈ 2×10^5 M^-1 s^-1 (literature). At [TIMP-1] = 10 nM, the pseudo-first-order inhibition rate constant is k_obs = k_on × [TIMP-1] = 2×10^5 × 10^-8 = 2×10^-3 s^-1 → half-life ~350 s (~6 min). Over 30 min incubation, substantial fraction of MMP-9 is inhibited by residual TIMP-1.

For HNE and A1AT (K_i ~ 1 nM, k_on,A1AT ≈ 2×10^7 M^-1 s^-1): at [A1AT] = 1 µM, k_obs = 20 s^-1 → half-life 35 ms — essentially instant. HNE is essentially fully inhibited by A1AT within seconds if [A1AT] >> [HNE]. The clinically relevant signal is the free HNE that escapes A1AT inhibition (i.e., [HNE] - [A1AT]).

For SLPI (K_i ~ 0.5 nM for HNE, k_on ≈ 3×10^7 M^-1 s^-1): similar timescale to A1AT.

Consequence for signal:
    [E_free](z,0,t) used in Layer 9 Robin BC must be the FREE enzyme, not total enzyme.
    F_k measured by the device reflects free enzyme activity, which is what has clinical meaning.

Implementation: extend the 1D PDE solver in `diffusion_pde.py` to track [E_free], [I_free], [EI_complex] as three coupled species. This is the critical correction to the naive single-species diffusion model.

## Key outputs to pass downstream

| Output | Layer |
|---|---|
| [E](z=0, t) — surface enzyme concentration over time | Layer 9 (surface kinetics) |
| Da — Damköhler regime classification | Layer 16 (defensible prediction) |
| t_clear — MB fragment clearance time | Layer 11 (SWV timing) |
| Confirmation of uniform IC after fill | Layer 9 |

## Key literature to find

- Stokes-Einstein radii for MMP-9 and HNE: compare with DLS or SAXS literature values
- Damköhler analysis for surface-enzyme sensors: Squires et al. (Nature Biotechnology, 2008) — essential reference for microfluidic biosensor transport limits
- MB peptide fragment diffusion: estimate from Stokes-Einstein or literature for similar-MW peptides
- Finite-difference PDE for coupled surface reactions: Bard & Faulkner, Electrochemical Methods (textbook)

## Files to create

- `diffusion_pde.py`: 1D Crank-Nicolson solver for the coupled PDE-ODE system; returns [E](z, t) and Gamma(t)
- `damkohler.py`: compute Da for parameter ranges; generate Da heatmap over (Gamma_0, [E]_0) space
- `fragment_clearance.py`: estimate bulk MB fragment concentration over time
