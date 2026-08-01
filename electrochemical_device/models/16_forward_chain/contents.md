# Layer 16: Full chain integration and falsifiable predictions

## Purpose

This layer chains all 16 preceding layers into a single end-to-end forward simulation. Given a set of input parameters (wound protease concentrations, environmental conditions, device design parameters), it produces a complete predicted output: the measured F_k values, the estimated protease concentrations, and the wound state probability.

The forward chain has two purposes:
1. Falsifiable prediction: predict F_k(T) for a known [E]_input and compare with calibration experiment → validates or refutes the model.
2. Sensitivity analysis: identify which parameters contribute the most uncertainty to the output → directs experimental effort and model improvement.

This is the iGEM gold-standard deliverable: the model produces a specific, quantitative, testable prediction. If the wet lab confirms it, we demonstrate model-experiment integration.

## Forward simulation algorithm

### 16.1 Input specification

Required inputs:
- [MMP-9]_wound, [HNE]_wound (nM): from wound biology ODE (Layer 0) or set directly for calibration experiment
- pH: wound fluid pH (6.5-8.2)
- [Ca2+]: wound fluid Ca2+ concentration (0.1-2 mM)
- T_amb: ambient temperature (20-37°C)
- T_incubation: incubation time (default 1800 s = 30 min)
- t_transit: sample transit time (0-15 min)

Device parameters (characterised per batch):
- Gamma_p (probe coverage, pmol/cm^2): from QCM or electrochemical characterisation
- Gamma_Fc (Fc coverage, pmol/cm^2): from Fc SWV calibration
- R_f (electrode roughness): from CV in H2SO4
- k_ET^0,MB, k_ET^0,Fc: from SWV frequency dependence measurement
- K_m,surf for each probe: from calibration experiment

Model parameters (from literature):
- D_MMP9, D_HNE: from Stokes-Einstein (Layer 3)
- beta, lambda: from Marcus theory literature (Layer 10)
- E°_MB(pH), E°_Fc: from literature (Layer 10)
- All other parameters tabulated in data/literature_parameters/

### 16.2 Forward simulation chain (layer order)

```
INPUT: [E]_wound, pH, [Ca2+], T_amb, T_incubation, t_transit

Layer 0:  wound_ode([E]_wound, pH, T_amb) → [MMP-9]_active, [HNE]_active at time of sampling

Layer 1:  preanalytical_model([MMP-9]_active, [HNE]_active, t_transit, V_abs, eta_elut, V_chamber)
          → [MMP-9]_chamber, [HNE]_chamber

Layer 2:  capillary_fill(h, theta, gamma, eta) → t_fill, fill_success (Boolean)

Layer 3:  [E](z, t=0) = [E]_chamber (uniform after fill)
          → start coupled PDE-ODE for layers 3 and 9

Layer 4:  electrode_characterisation(batch_params) → A_true = A_geo * R_f

Layer 5:  reference_electrode(Cl_conc, t_assay) → E_ref(t), delta_E_ref

Layer 6:  double_layer(Gamma_p, Gamma_d, I) → C_dl, lambda_D

Layer 7:  sam_chemistry(c_p, c_d, K_a_ratio) → Gamma_p, Gamma_d, Ca2+_effect(delta_d)

Layer 8:  probe_conformation(L_c, l_p_PEG, l_p_pep, Gamma_p) → P(d), steric_factor, h_cut

Layer 9:  [simultaneous with Layer 3]
          surface_kinetics(k_cat, K_m_surf, [E](z=0,t), Gamma_0) → Gamma_intact(t), F(T)

Layer 10: marcus_et(P(d), beta, lambda, T_amb, pH, [Ca2+], delta_d_Ca)
          → k_ET^0,eff,MB, k_ET^0,eff,Fc
          formal_potential(pH, E_ref_offset) → E_MB_prime, E_Fc

Layer 11: swv_forward(Gamma_intact(T), k_ET^0,eff,MB, E_MB_prime, f_SWV, ΔE_SW, ΔE_step, A_true, C_dl)
          → ΔI_MB(E), ΔI_Fc(E) [simulated voltammograms]

Layer 12: peak_extraction(ΔI_MB(E), ΔI_Fc(E)) → A_area,MB, A_area,Fc, sigma_MB, sigma_Fc

Layer 13: ratiometric(A_area,MB, A_area,Fc, A_area,MB_0, A_area,Fc_0, A_area,ctrl,MB, A_area,ctrl,Fc)
          → F_k, sigma_F, kappa_k, sigma_kappa

Layer 14: noise_budget(R_f, I_cap, f_SWV, BW) → sigma_A_area, sigma_F_noise

Layer 15: calibration(F_k, S_matrix) → c_MMP9_est, c_HNE_est, sigma_c
          wound_classifier(c_MMP9_est, c_HNE_est) → P(wound_state)

OUTPUT: F_k (predicted), c_MMP9_est, c_HNE_est, P(wound_state), sigma_F, LOD
```

### 16.3 Comparison with wet lab calibration data

The forward model predicts F_k for a known [E]_input. The calibration experiment (Layer 15.1) measures F_k for the same [E]_input. Comparison:

    residual(j) = F_k,predicted(j) - F_k,measured(j)

Metrics:
- RMSE: sqrt(mean(residual^2)) → overall accuracy
- Systematic bias: mean(residual) → suggests a missing or wrong parameter in the model
- Slope of predicted vs measured (should be 1.0): deviation suggests calibration factor error

If the model is accurate: residuals should be within 2σ_F of zero, with no systematic trend with [E].

This comparison is the iGEM-qualifying "model predicts experimental outcome, wet lab confirms" criterion.

### 16.4 Sobol global sensitivity analysis

Which parameters drive the most uncertainty in the output F_k?

Method (SALib package, Python):
1. Define input parameter distributions (uniform, normal, log-uniform) for all uncertain parameters
2. Generate Sobol sequence sample set (N = 1000-10000 parameter combinations)
3. Run forward chain for each sample
4. Compute Sobol first-order (S1) and total-order (ST) sensitivity indices

Candidate dominant parameters (hypothesis before running):
- Gamma_p (probe coverage): likely dominant — directly scales F_k
- k_cat / K_m_surf (surface kinetics): second most important
- D_f (dilution factor / swab elution): major pre-analytical uncertainty
- T_amb (temperature): large effect via kcat and k_ET^0
- K_m,surf / K_m,sol (steric correction): large uncertainty range

Parameters expected to be less important:
- beta (tunnelling decay): only affects SWV peak height, which cancels in ratiometric F_k
- lambda (reorganisation energy): similarly; cancels in ratio
- R_f (roughness): only affects absolute A_MB and A_Fc, cancels in R = A_MB/A_Fc

### 16.5 Parameter inference (inverse problem)

Given measured F_k from calibration experiment AND measured Gamma_p (from QCM), can we infer k_cat / K_m,surf in situ on the device?

Yes: if F_k is measured as a function of time (multiple scans at T = 0, 10, 20, 30 min), fit the exponential F_k(T) = 1 - exp(-k_obs * T) to extract k_obs = (k_cat / K_m,surf) * [E]_chamber. With [E]_chamber known (from the calibration spike), k_cat / K_m,surf is determined.

This k_cat / K_m,surf can be compared with the solution-phase value from pES wet lab experiments → validates the steric correction factor from Layer 8. This is a specific falsifiable prediction: if the surface kcat/Km is 3-10× less than solution kcat/Km (steric correction), the model predicts this, and the in-situ inference can confirm or refute the correction factor.

### 16.6 Wet-lab integration points

The forward chain has three explicit integration points with the wet lab (pES experiments):

| Integration point | Model parameter | Wet lab measurement | Layer |
|---|---|---|---|
| kcat/Km in solution | k_cat / K_m_sol | pES-01 to pES-08 (enzyme titration in fluorescence depletion assay) | 9 |
| Substrate ranking | Relative k_obs across substrates | Spearman rank correlation: model rank vs measured rank | 9, 15 |
| Cross-reactivity kappa(S) | kappa(S) < 30 | Cross-reactivity matrix: each substrate vs each protease | 15 |

The model PREDICTS the relative ranking of substrates by kobs (from kcat/Km differences) and PREDICTS kappa(S) from the substrate selectivity model. The wet lab measures these same quantities. Agreement between model and experiment validates the model; discrepancy identifies which layer's assumptions are wrong.

## Computational implementation

### 16.7 Code structure

```
models/16_forward_chain/
    forward_chain.py        # Main simulation runner; calls all layer modules
    sensitivity.py          # SALib Sobol analysis
    parameter_inference.py  # Inverse problem: k_obs from measured F_k(T)
    validate.py             # Compare predictions to calibration data; compute RMSE
    parameters_default.json # Default parameter values with literature sources
```

All layer modules (in their respective folders) must be importable as Python modules. The forward chain calls them in sequence, passing outputs as inputs to the next layer.

### 16.8 Computational cost estimate

Per simulation (single parameter set):
- Layer 3+9 (PDE-ODE): N_z=50 grid, N_t=3200 steps → ~3200 × 50 = 160,000 operations → milliseconds
- Layer 11 (SWV forward): 650 steps × 2 half-cycles → 1300 ODE solves → milliseconds
- All other layers: analytical calculations → microseconds

Total: < 1 second per simulation.

For Sobol analysis with N = 10,000 samples: ~10,000 seconds = 2.8 hours on a single CPU core → parallelisable to minutes on GPU or multi-core. Given GPU access: Sobol analysis is tractable.

## Key outputs

| Output | Value | Interpretation |
|---|---|---|
| F_k_predicted vs F_k_measured | Should agree within 2σ_F | Model validation |
| Sobol S1 ranking | Top 3 parameters | Guides where to invest experimental effort |
| k_cat/K_m,surf inferred | vs solution value | Validates steric correction |
| kappa(S) predicted | < 30 | Confirms substrate selectivity is sufficient for clinical use |

## Files to create

- `forward_chain.py`: top-level simulation runner; accepts parameter dict; returns output dict
- `sensitivity.py`: SALib Sobol wrapper; generates parameter samples, runs chain, computes indices
- `parameter_inference.py`: scipy.optimize.curve_fit to extract k_obs from F_k(T) time series
- `validate.py`: load calibration data (CSV), compare with predictions, report RMSE and bias
- `parameters_default.json`: all default parameter values with literature citations
- `run_demo.py`: example script demonstrating full chain from wound biology to clinical output
