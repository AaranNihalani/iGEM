# Layer 8: Probe architecture and conformational dynamics

## Purpose

The probe is a flexible polymer chain tethered at one end (Cys-Au) with the MB reporter at the other end (Lys(MB)). The MB-to-gold tunnelling distance d is not fixed; it fluctuates with the conformational dynamics of the probe tether. Because k_ET ∝ exp(-beta * d), the electron transfer rate is exponentially sensitive to these fluctuations — the mean <k_ET> is NOT simply k_ET evaluated at the mean distance <d>.

This layer computes the distribution of MB-to-Au distances P(d) using polymer physics models, and propagates this into the effective average ET rate <k_ET>. It also establishes the steric gating: how surface coverage Gamma_p determines enzyme accessibility to the cleavage site.

This layer directly controls k_ET (Layer 10) and the steric correction to K_m (Layer 9). It is physically fundamental to the device.

## Probe structure

### HNE channel probe
    Ac-Cys-[PEG4 or PEG6]-AAPV↓SGIG-[PEG2]-Lys(MB)-NH2
    Cleavage at AAPV↓SGIG: between Val (P1) and Ser (P1')

### MMP-9 channel probe
    Ac-Cys-[PEG4 or PEG6]-IPRT↓LTAA-[PEG2]-Lys(MB)-NH2
    Cleavage site to be determined from MEROPS data

### Scrambled control probe
    Ac-Cys-[PEG4 or PEG6]-[VGASGIAP or other scramble]-[PEG2]-Lys(MB)-NH2
    No cleavage; MB remains tethered throughout assay

## Sub-components to model

### 8.1 Contour length inventory

PEG monomer: -CH2-CH2-O-, monomer length b_PEG = 0.36 nm (extended C-C-O bond length)
Amino acid: peptide bond + sidechain; extended strand: 0.35 nm/aa; typical for random coil segment: varies

Full probe:
    Cys residue (anchored): l_Cys ≈ 0.5 nm (Cys sidechain + backbone segment to anchor)
    PEG4: 4 × 0.36 = 1.44 nm; PEG6: 6 × 0.36 = 2.16 nm
    Octapeptide (AAPVSGIG or IPRTLTAA): 8 × 0.35 = 2.80 nm (extended beta-strand)
    PEG2: 2 × 0.36 = 0.72 nm
    Lys(MB) linker: ~0.5 nm (Lys backbone + epsilon-amine to MB)
    Total contour L_c ≈ 5.5-6.2 nm (PEG4) or 6.2-7.0 nm (PEG6)

Note: L_c is the fully extended length. The chain is flexible and thermally fluctuates; the mean end-to-end distance is << L_c.

### 8.2 Worm-like chain (WLC) model for tether

The probe is a concatenation of two types of polymer segments: PEG and peptide. Both are treated as worm-like chains with different persistence lengths.

Worm-like chain mean-squared end-to-end distance:
    <r^2> = 2 * l_p * L_c * [1 - (l_p/L_c) * (1 - exp(-L_c/l_p))]

Persistence lengths:
    PEG: l_p,PEG = 0.38 nm (experimental from AFM force spectroscopy, Oesterhelt 1999)
    Unstructured peptide: l_p,pep = 0.4-1.0 nm (sequence-dependent; for polyglycine l_p ≈ 0.4 nm; for beta-sheet-forming sequences l_p up to 10 nm)
    AAPVSGIG: mixed sequence (Pro has high cis content, adds stiffness); estimate l_p ≈ 0.5-0.8 nm
    IPRTLTAA: Arg and Thr may form hydrogen bonds → slightly stiffer; l_p ≈ 0.6-1.0 nm

For a heterogeneous chain (PEG + peptide segments): use an effective persistence length for the concatenated chain, weighted by each segment's l_p and L_c.

### 8.3 Distance distribution P(d) and average k_ET

The end-to-end distance distribution for a WLC is well-approximated by a Gaussian at long L_c and by the Marko-Siggia interpolation formula for intermediate L_c:

For short chains (L_c / l_p < 10, which is the case here): the distribution is NOT Gaussian. Use the radial distribution function for a WLC:
    P(r, L_c) = 4 * pi * r^2 * C(r, L_c, l_p)

where C(r, L_c, l_p) comes from the Kratky-Porod transfer matrix or the Marko-Siggia formula.

The MB is at the END of the tether, at distance r from the gold surface (the 'z-projection' of r). For a tethered probe on a flat surface, the relevant distance for tunnelling is the z-component of r (vertical distance to the gold plane). But in the Marcus model, we use the full r (3D distance from Au atom to MB) as the tunnelling distance d.

Average ET rate:
    <k_ET> = k_ET^max * integral_0^L_c exp(-beta * r) * P(r) * dr

This integral is dominated by the small-r tail of P(r). Even if <r> is large, the small-r excursions (where the chain is momentarily compressed) contribute most to <k_ET>.

Implementation: Monte Carlo sampling of chain configurations (Freely Jointed Chain or WLC), compute r for each configuration, compute k_ET(r), average over all configurations. Alternatively, compute the integral numerically using the WLC P(r).

Compare: <k_ET> from distribution vs k_ET evaluated at <r>. The ratio <exp(-beta*r)> / exp(-beta*<r>) quantifies how much the heterogeneous distribution increases the effective ET rate relative to the naive estimate.

### 8.4 SAM density effect on probe conformation

At high probe surface coverage Gamma_p:
- Steric repulsion between adjacent probe chains forces them to extend away from the surface (brush-like regime)
- At Gamma_p > Gamma_crossover ≈ 1/(4 * pi * l_p^2): chains are in the brush regime with effective extension d_eff ≈ N * b * (Gamma_p * pi * b^2)^(1/3) (Daoud-Cotton scaling)
- The chain is more extended → d_eff is larger → k_ET is slower

At low Gamma_p:
- Chains are in the mushroom regime: d_eff ≈ r_g (radius of gyration)
- Chains can lie flat or collapsed → d_eff << L_c → faster k_ET but also enzyme access is better

Model: compute d_eff(Gamma_p) using the brush-to-mushroom transition:
    d_eff = r_g * (1 + Gamma_p/Gamma_crossover)^(1/3)   (interpolation; exact scaling varies)

This makes d_eff an implicit function of Gamma_p from Layer 7.

### 8.5 Steric accessibility of cleavage site

For the enzyme to cleave the probe, it must approach to within binding distance of the cleavage site (AAPV↓SGIG for HNE; IPRTↃ for MMP-9).

Cleavage site position from the gold surface:
    h_cut = L(Cys anchor) + L(PEG4/6) ≈ 0.5 + 1.44 = 1.94 nm (for PEG4)
    h_cut = 0.5 + 2.16 = 2.66 nm (for PEG6)

For the enzyme to reach the cleavage site, it needs a clear approach within its active site diameter:
    r_enzyme ≈ 2.5 nm (MMP-9, ~5 nm footprint); 1.5 nm (HNE, ~3 nm footprint)

Steric exclusion criterion: the area per probe molecule must be larger than the enzyme footprint:
    A_probe = 1/Gamma_p > pi * r_enzyme^2

    Gamma_p < Gamma_steric_max = 1 / (pi * r_enzyme^2)

For MMP-9 (r_enzyme = 2.5 nm): Gamma_steric_max = 1/(pi * (2.5e-7)^2) cm^-2 ≈ 5.1 pmol/cm^2
For HNE (r_enzyme = 1.5 nm): Gamma_steric_max = 1/(pi * (1.5e-7)^2) ≈ 14.1 pmol/cm^2

So for the MMP-9 probe: Gamma_p must be < 5.1 pmol/cm^2 for enzyme access. This is a hard upper limit on useful Gamma_p.

Steric correction factor for K_m (passed to Layer 9):
    f_steric = 1 - Gamma_p / Gamma_steric_max

At Gamma_p = 3 pmol/cm^2 (MMP-9 probe): f_steric = 1 - 3/5.1 ≈ 0.41 → K_m,surf ≈ K_m,sol / 0.41 ≈ 2.4 × K_m,sol.

### 8.6 Post-cleavage state

After cleavage by HNE at AAPV↓SGIG:
- Remaining on surface: Ac-Cys-PEG4/6-AAPV (stub)
  - Length: L_Cys + L_PEG4 + 4 × 0.35 = 0.5 + 1.44 + 1.4 = 3.34 nm (extended)
  - This stub carries no MB → no contribution to SWV peak
  - The stub still occupies surface area and contributes to steric crowding of remaining intact probes
- Released: SGIG-PEG2-Lys(MB)-NH2 (diffuses away, Layer 3.6)
  - MB in this fragment: contributes to bulk MB diffusion signal (negligible as shown in Layer 3.6)

The stub persistence: does the cleaved stub change the surface properties? After cleavage, surface has:
- Gamma_stub = Gamma_0 - Gamma_intact (growing over time)
- Gamma_intact = Gamma_0 * (1 - F(t))

The stubs are shorter than intact probes → they do not sterically block enzyme from remaining intact probes (if anything they may increase local free volume). This is a second-order effect and probably negligible for F < 0.9.

### 8.7 IPRTLTAA-specific considerations

IPRTLTAA contains:
- Ile (I, P4): hydrophobic, beta-sheet former
- Pro (P, P3): cis/trans isomerism; adds kink to backbone; high stiffness
- Arg (R, P2): positively charged at pH 7.4 (pKa 12.5) → electrostatic interaction with electrode (see Layer 6.3)
- Thr (T, P1): hydroxyl, neutral
- Leu (L, P1'): hydrophobic
- Thr (T, P2'): hydroxyl
- Ala (A, P3'): small
- Ala (A, P4'): small

The Pro at P3 introduces a kink and prevents alpha-helix formation, keeping the peptide in an extended conformation near the cleavage site. This is good for enzyme access.

The Arg charge: at cathodic potentials (during MB reduction at -0.25 V), the electrode is negatively polarised → Arg is attracted toward the electrode → probe bends toward gold → d_eff decreases → k_ET increases at this potential. This is a potential-dependent k_ET contribution specific to IPRTLTAA that does not affect AAPVSGIG (all neutral residues). Must include this as a correction term in Layer 10.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| P(d) — MB-to-Au distance distribution | WLC model | Layer 10 (<k_ET>) |
| <k_ET> — average ET rate | Integral of k_ET(r) P(r) | Layer 10, 11 |
| h_cut — cleavage site height | 1.9-2.7 nm from Au | Layer 9 (enzyme access geometry) |
| Gamma_steric_max | 5.1 (MMP-9) / 14.1 (HNE) pmol/cm^2 | Layer 9 (K_m,surf correction) |
| f_steric(Gamma_p) | 0.3-0.8 | Layer 9 |
| d_eff(Gamma_p) | brush vs mushroom regime | Layer 10 |

## Key literature to find

- WLC model and P(r) for short chains: Marko & Siggia (Macromolecules, 1995)
- PEG persistence length: Oesterhelt et al. (New J Phys, 1999)
- Peptide persistence lengths: review by Flory or Creighton
- Probe conformation effect on k_ET: Rant et al. (Nano Letters, 2004) — surface-tethered DNA ET as precedent
- Steric exclusion in crowded SAMs: Steel et al. (Biophys J, 2000); Perez et al. (JACS, 2012)
- Electrostatic bending of probe by electrode charge: Netz & Joanny (Macromolecules, 1999)

## Files to create

- `probe_conformation.py`: WLC model; Monte Carlo simulation of chain conformations; P(r) distribution
- `average_ket.py`: numerical integral of k_ET(r) * P(r) over r; ratio <k_ET>/<k_ET at <r>>
- `steric_gate.py`: Gamma_steric_max computation; f_steric(Gamma_p); d_eff(Gamma_p) brush model
