# Layer 8: Probe architecture and conformational dynamics

## Purpose

The probe is a flexible polymer chain tethered at one end (Cys-Au) with the MB reporter at the other end Lys(MB). The MB-to-gold tunnelling distance $d$ is not fixed; it fluctuates with the conformational dynamics of the probe tether. Because $k_{ET} \propto e^{-\beta d}$, the electron transfer rate is exponentially sensitive to these fluctuations — the mean $\langle k_{ET} \rangle$ is **not** simply $k_{ET}$ evaluated at the mean distance $\langle d \rangle$.

This layer computes the distribution of MB-to-Au distances $P(d)$ using polymer physics models, and propagates this into the effective average ET rate $\langle k_{ET} \rangle$. It also establishes the steric gating: how surface coverage $\Gamma_p$ determines enzyme accessibility to the cleavage site.

This layer directly controls $k_{ET}$ (Layer 10) and the steric correction to $K_m$ (Layer 9). It is physically fundamental to the device.

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

PEG monomer: -CH2-CH2-O-, monomer length $b_{\text{PEG}} = 0.36 \text{ nm}$ (extended C-C-O bond length)
Amino acid: peptide bond + sidechain; extended strand: $0.35 \text{ nm/aa}$; typical for random coil segment: varies

Full probe:
    Cys residue (anchored): $l_{\text{Cys}} \approx 0.5 \text{ nm}$ (Cys sidechain + backbone segment to anchor)  
    PEG4: $4 \times 0.36 = 1.44 \text{ nm}$; PEG6: $6 \times 0.36 = 2.16 \text{ nm}$  
    Octapeptide (AAPVSGIG or IPRTLTAA): $8 \times 0.35 = 2.80 \text{ nm}$ (extended beta-strand)  
    PEG2: $2 \times 0.36 = 0.72 \text{ nm}$  
    Lys(MB) linker: $\sim 0.5 \text{nm}$ (Lys backbone + epsilon-amine to MB)  
    Total contour $L_c ≈ 5.5 \text{ to } 6.2 \text{ nm}$ (PEG4) or $6.2 \text{ to } 7.0 \text{ nm}$ (PEG6)

Note: $L_c$ is the fully extended length. The chain is flexible and thermally fluctuates; the mean end-to-end distance is $\ll L_c$.

### 8.2 Worm-like chain (WLC) model for tether

The probe is a concatenation of two types of polymer segments: PEG and peptide. Both are treated as worm-like chains with different persistence lengths.

Worm-like chain mean-squared end-to-end distance:

$$\langle r^2 \rangle = 2 \\, l_p \\, L_c \left( 1 - \frac{l_p}{L_c}\left( 1 - e^{-L_c/l_p} \right) \right)$$

Persistence lengths:
    PEG: $l_{p, \\, \text{PEG}} = 0.38 \text{ nm}$ (experimental from AFM force spectroscopy, Oesterhelt 1999)  
    Unstructured peptide: $l_{p, \\, \text{PEP}} = 0.4 \text{ to } 1.0 \text{ nm}$ (sequence-dependent; for polyglycine $l_p \approx 0.4 \text{ nm}$; for beta-sheet-forming sequences $l_p$ up to 10 nm)  
    AAPVSGIG: mixed sequence (Pro has high cis content, adds stiffness); estimate $l_p \approx 0.5 \text{ to } 0.8 \text{ nm}$  
    IPRTLTAA: Arg and Thr may form hydrogen bonds → slightly stiffer; $l_p \approx 0.6 \text{ to } 1.0 \text{ nm}$

For a heterogeneous chain (PEG + peptide segments): use an effective persistence length for the concatenated chain, weighted by each segment's $l_p$ and $L_c$.

### 8.3 Distance distribution $P(d)$ and average $k_{ET}$

The end-to-end distance distribution for a WLC is well-approximated by a Gaussian at long $L_c$ and by the Marko-Siggia interpolation formula for intermediate $L_c$:

For short chains ($L_c / l_p < 10$, which is the case here): the distribution is **not** Gaussian. Use the radial distribution function for a WLC:

$$P(r, L_c) = 4 \pi r^2 C(r, L_c, l_p)$$

where $C(r, L_c, l_p)$ comes from the Kratky-Porod transfer matrix or the Marko-Siggia formula.

The MB is at the **end** of the tether, at distance $r$ from the gold surface (the 'z-projection' of $r$). For a tethered probe on a flat surface, the relevant distance for tunnelling is the z-component of $r$ (vertical distance to the gold plane). But in the Marcus model, we use the full $r$ (3D distance from Au atom to MB) as the tunnelling distance $d$.

Average ET rate:

$$\langle k_{ET} \rangle = k_{ET}^{\max}  \int_0^{L_c} e^{-\beta r} P(r) \\:  \mathrm{d}r$$

This integral is dominated by the small-r tail of $P(r)$. Even if <r> is large, the small-r excursions (where the chain is momentarily compressed) contribute most to $\langle k_{ET} \rangle$.

Implementation: Monte Carlo sampling of chain configurations (Freely Jointed Chain or WLC), compute $r$ for each configuration, compute $k_{ET}(r)$, average over all configurations. Alternatively, compute the integral numerically using the WLC $P(r)$.

Compare: $\langle k_{ET} \rangle$ from distribution vs $k_{ET}$ evaluated at $\langle r \rangle$. The ratio $\langle e^{-\beta r} \rangle : e^{-\beta \langle r \rangle}$ quantifies how much the heterogeneous distribution increases the effective ET rate relative to the naive estimate.

### 8.4 SAM density effect on probe conformation

At high probe surface coverage $\Gamma_p$:
- Steric repulsion between adjacent probe chains forces them to extend away from the surface (brush-like regime)
- At $\Gamma_p > \Gamma_{\text{crossover}} \approx 1/(4 \pi {l_p}^2)$: chains are in the brush regime with effective extension $d_{\text{eff}} \approx N b (\Gamma_p \pi b^2)^{1/3}$ (Daoud-Cotton scaling)
- The chain is more extended → $d_\text{eff}$ is larger means $k_{ET}$ is slower

At low $\Gamma_p$:
- Chains are in the mushroom regime: $d_{\text{eff}} \approx r_g$ (radius of gyration)
- Chains can lie flat or collapsed, so $d_{\text{eff}} \ll L_c$, which means faster $k_{ET}$ but also enzyme access is better

Model: compute $d_{\text{eff}}(\Gamma_p)$ using the brush-to-mushroom transition:
    
$$d_{\text{eff}} = r_g \left( 1 + \frac{\Gamma_p}{\Gamma_{\text{crossover}}} \right)^{\frac{1}{3}}$$ 

(interpolation; exact scaling varies)

This makes $d_{\text{eff}}$ an implicit function of $\Gamma_p$ from Layer 7.

### 8.5 Steric accessibility of cleavage site

For the enzyme to cleave the probe, it must approach to within binding distance of the cleavage site (AAPV↓SGIG for HNE; IPRTↃ for MMP-9).

Cleavage site position from the gold surface:  
    $h_{\text{cut}} = L(\text{Cys anchor}) + L(\text{PEG4/6}) \approx 0.5 + 1.44 = 1.94 \text{ nm}$ (for PEG4)  
    $h_{\text{cut}} = 0.5 + 2.16 = 2.66 \text{ nm}$ (for PEG6)

For the enzyme to reach the cleavage site, it needs a clear approach within its active site diameter:
    $r_{\text{enzyme}} \approx 2.5 \text{ nm}$ (MMP-9, ~5 nm footprint); $1.5 \text{ nm}$ (HNE, ~3 nm footprint)

Steric exclusion criterion: the area per probe molecule must be larger than the enzyme footprint:
    
$$A_{\text{probe}} = \frac{1}{\Gamma_p} > \pi r_{\text{enzyme}}^2$$

$$\Gamma_p < \Gamma_{\text{steric}}^{\max} = \frac{1}{\pi r_{\text{enzyme}}^2}$$

For MMP-9 ($r_{\text{enzyme}} = 2.5 \text{ nm}$):

$$\Gamma_{\text{steric}}^{\max} = \frac{1}{\pi \left( 2.5 \times 10^{-7} \right)^2} \text{ cm}^{-2} \approx 5.1 \text{ pmol} / \text{cm}^2$$

For HNE ($r_{\text{enzyme}} = 1.5 \text{ nm}$): 

$$\Gamma_{\text{steric}}^{\max} = \frac{1}{\pi \left( 1.5 \times 10^{-7} \right)^2} \text{ cm}^{-2} \approx 14.1 \text{ pmol} / \text{cm}^2$$

So for the MMP-9 probe: $\Gamma_p$ must be less than $5.1 \text{ pmol} / \text{cm}^2$ for enzyme access. This is a hard upper limit on useful $\Gamma_p$.

Steric correction factor for $K_m$ (passed to Layer 9):

$$f_{\text{steric}} = 1 - \frac{\Gamma_p}{\Gamma_{\text{steric}}^{\max}}$$

At $\Gamma_p = 3 \text{ pmol} / \text{cm}^2$ (MMP-9 probe): $f_{\text{steric}} = 1 - 3/5.1 \approx 0.41$, so $K_{m, \, \text{surf}} \approx K_{m, \, \text{sol}} / 0.41 \approx 2.4 \times K_{m, \text{sol}}$.

### 8.6 Post-cleavage state

After cleavage by HNE at AAPV↓SGIG:
- Remaining on surface: Ac-Cys-PEG4/6-AAPV (stub)
  - Length: $L_{\text{Cys}} + L_{\text{PEG4}} + 4 \times 0.35 = 0.5 + 1.44 + 1.4 = 3.34 \text{ nm}$ (extended)
  - This stub carries no MB, so no contribution to SWV peak.
  - The stub still occupies surface area and contributes to steric crowding of remaining intact probes.
- Released: SGIG-PEG2-Lys(MB)-NH2 (diffuses away, Layer 3.6)
  - MB in this fragment: contributes to bulk MB diffusion signal (negligible as shown in Layer 3.6)

The stub persistence: does the cleaved stub change the surface properties? After cleavage, surface has:
- $\Gamma_{\text{stub}} = \Gamma_0 - \Gamma_{\text{intact}}$ (growing over time)
- $\Gamma_{\text{intact}} = \Gamma_0 (1 - F(t))$

The stubs are shorter than intact probes, so they do not sterically block enzyme from remaining intact probes (if anything they may increase local free volume). This is a second-order effect and probably negligible for $F < 0.9$.

### 8.7 IPRTLTAA-specific considerations

IPRTLTAA contains:
- Ile (I, P4): hydrophobic, beta-sheet former
- Pro (P, P3): cis/trans isomerism; adds kink to backbone; high stiffness
- Arg (R, P2): positively charged at pH 7.4 (pKa 12.5), electrostatic interaction with electrode (see Layer 6.3)
- Thr (T, P1): hydroxyl, neutral
- Leu (L, P1'): hydrophobic
- Thr (T, P2'): hydroxyl
- Ala (A, P3'): small
- Ala (A, P4'): small

The Pro at P3 introduces a kink and prevents alpha-helix formation, keeping the peptide in an extended conformation near the cleavage site. This is good for enzyme access.

The Arg charge: at cathodic potentials (during MB reduction at -0.25 V), the electrode is negatively polarised, so Arg is attracted toward the electrode. The probe bends toward gold and $d_{\text{eff}}$ decreases, so $k_{ET}$ increases at this potential. This is a potential-dependent $k_{ET}$ contribution specific to IPRTLTAA that does not affect AAPVSGIG (all neutral residues). Must include this as a correction term in Layer 10.

## Key outputs

| Output | Value | Feeds |
|--------|-------|-------|
| $P(d)$: MB-to-Au distance distribution | WLC model | Layer 10: $\langle k_{ET} \rangle$ |
| $\langle k_{ET} \rangle$: average ET rate | Integral of $k_{ET}(r) P(r)$ | Layer 10, 11 |
| $h_{\text{cut}}$: cleavage site height | $1.9 \text{ to } 2.7 \text{ nm}$ from Au | Layer 9: enzyme access geometry |
| $\Gamma_{\text{steric}}^{\max}$ | $5.1$ (MMP-9) or $14.1$ (HNE) $\text{ pmol}/\text{cm}^2$ | Layer 9: $K_{m, \\, \text{surf}}$ correction |
| $f_{\text{steric}}(\Gamma_p)$ | $0.3 \text{ to } 0.8$ | Layer 9 |
| $d_{\text{eff}}(\Gamma_p)$ | brush vs mushroom regime | Layer 10 |

## Key literature to find

- WLC model and $P(r)$ for short chains: Marko & Siggia (Macromolecules, 1995)
- PEG persistence length: Oesterhelt et al. (New J Phys, 1999)
- Peptide persistence lengths: review by Flory or Creighton
- Probe conformation effect on $k_{ET}$: Rant et al. (Nano Letters, 2004) — surface-tethered DNA ET as precedent
- Steric exclusion in crowded SAMs: Steel et al. (Biophys J, 2000); Perez et al. (JACS, 2012)
- Electrostatic bending of probe by electrode charge: Netz & Joanny (Macromolecules, 1999)

## Files to create

- `probe_conformation.py`: WLC model; Monte Carlo simulation of chain conformations; $P(r)$ distribution
- `average_ket.py`: numerical integral of $k_{ET}(r) P(r)$ over $r$; ratio $\langle k_{ET} \rangle : \langle k_{ET} \text{ at } \langle r \rangle \rangle$
- `steric_gate.py`: $\Gamma_{\text{steric}}^{\max}$ computation; $f_{\text{steric}}(\Gamma_p)$; $d_{\text{eff}}(\Gamma_p)$ brush model
