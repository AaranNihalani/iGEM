# Layer 6: Electrical double layer and solution resistance

## Purpose

When a gold electrode is immersed in wound fluid, an electrical double layer (EDL) forms at the interface: an inner Helmholtz layer of specifically adsorbed ions/molecules, and a diffuse Gouy-Chapman layer of electrostatically attracted counterions. The EDL is not just a background; it determines:
- The capacitive current background in SWV (which sets the noise floor for detecting the MB faradaic peak)
- The local electrostatic environment experienced by probe molecules (which affects enzyme accessibility and MB tunnelling trajectory)
- The effective local pH at the electrode surface during SWV (which can shift E°_MB)

This layer is a prerequisite for the SWV forward model (Layer 11) and the noise budget (Layer 14).

## Sub-components to model

### 6.1 Gouy-Chapman-Stern (GCS) model

The EDL is divided into three regions:
1. Inner Helmholtz plane (IHP): layer of specifically adsorbed ions and solvent dipoles; thickness d_1 ≈ 0.3-0.5 nm.
2. Outer Helmholtz plane (OHP): closest approach of hydrated counterions; thickness d_2 ≈ 0.4-0.8 nm.
3. Diffuse Gouy-Chapman layer: exponentially decaying potential from OHP to bulk; decay length = lambda_D.

Debye length (Layer 0, 6.3):
    lambda_D = sqrt(epsilon * epsilon_0 * R * T / (2 * F^2 * I))
    At I = 150 mM, T = 310 K, epsilon_r = 78.5: lambda_D = 0.77 nm

Total double layer capacitance (series combination):
    1/C_dl = 1/C_Helmholtz + 1/C_diffuse

C_Helmholtz = epsilon_0 * epsilon_IHP / d_Helmholtz ≈ 10-40 µF/cm^2 (inner layer)
C_diffuse = epsilon_0 * epsilon_r / lambda_D * cosh(z*F*phi_d/(2RT)) ≈ 200-500 µF/cm^2 at physiological I (very large)

At physiological ionic strength, C_diffuse >> C_Helmholtz → C_dl ≈ C_Helmholtz. The double layer is essentially determined by the inner (Helmholtz) layer.

With a SAM (Layer 7): the thiol monolayer acts as an additional dielectric layer in series:
    C_dl,SAM = epsilon_0 * epsilon_SAM / d_SAM

For an alkanethiol SAM (epsilon_SAM ≈ 2.2-2.5, d_SAM ≈ 1-2 nm):
    C_dl,SAM ≈ 1-2 µF/cm^2

This is much lower than C_Helmholtz on bare gold (~20 µF/cm^2), consistent with literature values of 1-5 µF/cm^2 for SAM-modified gold. The SAM dramatically reduces the capacitive background, which is one of the main advantages of using a thiol monolayer.

### 6.2 Capacitive SWV background

The capacitive current contribution to each SWV step:
    I_cap = C_dl * A_true * 2 * Delta_E_SW * f_SWV

For C_dl = 2 µF/cm^2, A_true = 0.01 cm^2 (1 mm^2 × R_f=1.3 → 0.013 cm^2), Delta_E_SW = 25 mV, f_SWV = 60 Hz:
    I_cap = 2×10^-6 * 0.013 * 2 * 25×10^-3 * 60 = 78 nA

This is the capacitive contribution. In SWV, the net current (forward - backward) cancels the symmetric capacitive contribution IF C_dl is identical in the forward and backward half-cycles. In practice, C_dl(E) is slightly potential-dependent (especially near the potential of zero charge, PZC), so a small residual capacitive background remains.

The SWV net current I_cap,net is NOT zero; typically I_cap,net ≈ 1-5% of I_cap per step due to C_dl(E) asymmetry. For I_cap = 78 nA: I_cap,net ≈ 1-4 nA. Compare with faradaic MB peak ~10-50 nA: the residual capacitive background is a significant fraction of the signal at low Gamma. This sets a signal-to-noise challenge and must be included in the SWV forward model.

### 6.3 Potential of zero charge (PZC) and electrode potential effects

The PZC of gold in physiological solution is approximately -0.05 to +0.05 V vs Ag/AgCl. When the electrode is polarised negative (during MB reduction at -0.25 V), it is below the PZC: the electrode surface is negatively charged, and cations (Na+, Ca2+) accumulate in the IHP.

Effect on probe: negatively charged probe residues (e.g., carboxylate groups) experience electrostatic repulsion from the negatively polarised surface → probe extends away from surface → d_eff increases at cathodic potentials → k_ET decreases. This is a potential-dependent k_ET effect superimposed on the intrinsic Marcus k_ET(d).

For AAPVSGIG (all neutral residues): this effect is small (only backbone amide dipoles interact).
For IPRTLTAA (Arg = +1): Arg is electrostatically attracted to the negatively polarised surface at cathodic potentials → probe bends toward electrode → d_eff decreases → k_ET increases. This is a potential-dependent positive bias on the IPRTLTAA channel MB peak height.

Must include: a correction term k_ET(E) = k_ET^0 × exp(-beta × delta_d(E)) where delta_d(E) is the potential-dependent change in effective tether length from electrostatic bending. Order of magnitude estimate using WLC model with electrostatic end-to-end force term (Pincus regime).

### 6.4 Local pH at electrode surface during SWV

MB redox reaction: MB + 2H+ + 2e- ⇌ leucomethylene blue (LMB)

During cathodic half-cycle: H+ is consumed at the electrode surface → local pH rises.
During anodic half-cycle: H+ is produced → local pH falls.

Net proton flux to electrode:
    J_H+ = -2 * I_MB / (F * A) (2 protons per MB reduced; cathodic current)

For I_MB = 50 nA peak, A = 0.013 cm^2, half-period = 1/(2*60) = 8.3 ms:
    delta_n_H+ = I * delta_t / (n * F) = 50e-9 * 8.3e-3 / (2 * 96485) ≈ 2.2 × 10^-15 mol per half-cycle

Volume of diffusion layer adjacent to electrode: V_diff ≈ A * delta ≈ 0.013 cm^2 * 50 µm = 0.065 µL
    delta_[H+] = 2.2e-15 / 6.5e-11 L ≈ 34 µM

At neutral pH, [H+] ≈ 100 nM. A perturbation of 34 µM would massively shift pH IF there were no buffer. But wound fluid contains bicarbonate buffer (~10-25 mM) and protein buffers. With 10 mM bicarbonate buffer:
    delta_pH ≈ delta_[H+] / (2.303 * [buffer] * 10) ≈ 34e-6 / (2.303 * 10e-3 * 10) ≈ 0.15 pH

A 0.15 pH shift in the diffusion layer shifts E°_MB by 0.15 × 59 mV = 8.9 mV per scan. This is a small but non-negligible systematic shift in the MB peak position, which will affect peak fitting in Layer 12 if uncorrected.

For wound fluid with lower buffer capacity (chronic infected wound, pH 6.5, low bicarbonate): this effect could be 3-5× larger.

### 6.5 Potential-dependent SAM stability

The Au-S bond can be reductively desorbed at sufficiently negative potentials:
    Au-SR + e- → Au(0) + RS-

Desorption potential (E_des) for alkanethiols on gold: approximately -0.8 to -1.1 V vs Ag/AgCl in phosphate buffer. Our SWV scan reaches -0.65 V at the cathodic limit — approximately 150-450 mV above E_des.

Safety margin: Adsorption-free energy of Au-S bond ~160 kJ/mol → very stable. At -0.65 V applied for 8.3 ms per step: negligible desorption rate. This is a safety check, not a quantitative concern, but must be explicitly stated as a design constraint.

### 6.6 Differential capacitance as quality metric

For a clean gold SAM electrode, C_dl should be flat and low (1-5 µF/cm^2) across the potential range. Elevated C_dl at a particular potential indicates:
- SAM defects (pinholes, domain boundaries)
- Non-specific protein adsorption displacing SAM diluent
- Partial SAM desorption

Monitor C_dl vs E as a diagnostic: if C_dl rises sharply at a particular E, it identifies a problem. In the SWV forward model (Layer 11), C_dl(E) is included as a potential-dependent parameter rather than a constant.

## Key outputs

| Output | Value | Feeds |
|---|---|---|
| C_dl (SAM-modified) | 1-5 µF/cm^2 | Layer 11 (I_cap), Layer 14 (noise) |
| lambda_D | 0.77 nm at 150 mM | Layer 7 (electrostatic probe model) |
| I_cap,net (SWV background) | 1-4 nA | Layer 11, 12 (baseline subtraction) |
| Local pH shift during SWV | 0.05-0.5 pH | Layer 10 (E°_MB correction) |
| E_des safety margin | > 150 mV | Design validation |

## Key literature to find

- GCS model for SAM-modified gold: Bard & Faulkner, Electrochemical Methods, Chapter 13
- C_dl measurements for thiol SAMs: Campuzano et al., Rant et al.
- PZC of gold in physiological solution: Lipkowski et al. (Electrochim Acta)
- Local pH effects at electrodes during MB SWV: Pheeney & Barton (2012 JACS)

## Files to create

- `double_layer.py`: GCS model implementation; C_dl(E, SAM coverage) calculation
- `local_ph.py`: estimate local pH perturbation from MB current; buffer capacity correction
