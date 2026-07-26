# Substrate Design — Task 8 Plan
*Aaran Nihalani · Eton iGEM 2026*

---

## Project structure — two parallel strands

This repo spans two distinct diagnostic systems being built in parallel. Understanding both is necessary to know where Task 8 sits.

**Strand A — Chronic wound protease diagnostic (this task)**
A 3-plex electrochemical biosensor for wound-fluid MMP-2, MMP-9, and NE. Signal-off SWV with methylene blue redox label. Output: posterior P(healing) from a Bayesian kinetic deconvolution of three activity channels. This is the primary dry-lab development strand.

**Strand B — Cancer biosensor (MMP14 / DNA-hairpin / Cas12a)**
A point-of-care serum assay for CRC detection using MMP14 activity to trigger a toehold strand displacement cascade into Cas12a amplification. Structural feasibility is scored via ESM2 embeddings + AlphaFold3 geometry to estimate k_open. AlphaFold3 prediction complete (pTM = 0.74). RMSD vs 2B3P and SASA calculations in progress.

The two strands share the same electrochemical sensor physics and the same Bayesian framework; they are otherwise independent.

---

## Remaining large tasks — full roadmap

This plan covers **Task 8**. The table below shows all remaining major tasks across both strands so the development sequence is clear.

| Task | Description | Strand | Status |
|---|---|---|---|
| Post-Meeting 1–5 | Bayesian landscape model, Hill equation, Marcus theory, clinical threshold mapping, sample volume calculations | A | **Done** — implemented in Protease Landscape Dry Lab Update (05 Jul 2026) |
| **Task 8** | 3-plex device-ready substrate selection (this document) | A | **Current** |
| Task 6 / Post-Meeting 6 | Cross-reactivity specificity matrix (repo: `cross_reactivity_matrix/`) | A | **Done** |
| Task 7 / Post-Meeting 7 | ML substrate predictor — VHSE8 + LR/LGBM (repo: `substrate_predictor/`) | A | **Done** |
| **Task 9** | Bayesian multi-analyte kinetic deconvolution — digital twin forward model + MCMC inversion + identifiability analysis. Requires mentor scoping meeting before development starts. This is the Best Model nomination. | A | Pending |
| Structural feasibility scorer | ESM2 + AlphaFold3 geometric features trained to estimate k_open for the cancer biosensor hairpin. Feeds directly into the Strand B kinetics model. | B | In progress — AlphaFold3 done; RMSD/SASA in progress |
| Software tool | Web app + CLI wrapping the kinetics model and structural scorer. Targets Best Software Tool prize. Only starts once both models have working v1. | A+B | Pending |

---

## Background and scope

This task takes the existing ML substrate predictor (Task 7) and PSSM cross-reactivity matrix (Task 6) and produces three device-ready peptide sequences the wet lab can order — one per protease channel in the chronic-wound diagnostic 3-plex: MMP-2, MMP-9, and neutrophil elastase (NE).

The clinical motivation is clear. MMP-9 is the predominant protease in chronic wound fluid and its level tracks ulcer severity (Rayment et al. 2008). Wound-fluid MMP-9 predicts poor DFU healing with AUC 0.901 (Jindatanmanusan et al. 2018; sensitivity 81.8%, threshold 0.38 pg/µg protein). Total MMP activity is ~30× elevated in chronic vs. acute wounds (Trengove et al. 1999). The MMP-9/TIMP-1 ratio is an independent prognostic marker (Ladwig 2002, Muller 2008). We are not building a simple abundance sensor — we are building a healing-outcome predictor that uses the protease activity fingerprint as input to a Bayesian posterior.

---

## How this differs from MEROPS

MEROPS identifies what sequences a protease cleaves. This model identifies which sequences a target protease cleaves selectively over competing panel proteases, subject to electrochemical device constraints, with formal multi-analyte identifiability analysis. These are different optimisation objectives.

| Capability | MEROPS | This work |
|---|---|---|
| Supervised learning (positives vs negatives) | No | Yes — LR + LGBM on VHSE8 |
| Physicochemical generalisation | No — AA identity only | Yes — generalises beyond MEROPS entries |
| Multi-protease selectivity scoring | No | Yes — joint_score = p × log(S+1) |
| 3-plex identifiability analysis | No | Yes — condition number + profile-likelihood |
| Electrochemical device constraints | No | Yes — Cys, Trp, net charge, MW, scissile placement |
| Bayesian multi-analyte deconvolution | No | Yes — MCMC inversion of R(t) to activity fingerprint |

---

## Current model quality — honest assessment

**What we have:**
- NE classifier: AUROC 0.825, MCC 0.467 — strongest model; NE is mechanistically orthogonal to the MMPs (serine vs metalloprotease)
- MMP-9 LGBM: AUROC 0.787, MCC 0.464 — solid for 341 positives
- Selectivity scoring and VHSE8 encoding are genuine contributions over MEROPS PWMs

**Gaps to close before claiming "high quality":**
- No calibration curves — p_cleave = 0.70 does not necessarily mean 70% of similar sequences are cleaved
- No bootstrapped confidence intervals — CV means without SDs are insufficient for a serious report
- No composition baseline — we have not proved VHSE8 positional encoding outperforms a simple AA-frequency model
- kcat/KM proxy is a single-anchor calibration (MMP14 SGRSENIR → 5.5×10⁴ M⁻¹s⁻¹); must be reported as ±1 order of magnitude
- AAPV kcat/KM is strongly buffer-dependent (15,000–330,000 M⁻¹s⁻¹); always report assay buffer

---

## Substrate candidates

The five candidates below are literature-derived and serve as the **input** to the ML validation pipeline. The model validates, ranks, and filters them — it does not generate them from scratch. Literature-seeded candidates scored highly by our model are far more defensible to reviewers than raw PSSM top-1 outputs.

### Stage 1 — order as sfGFP fusions before MB-peptide synthesis

**Candidate 1 — MMP-9-biased**
```
Ac-Cys-GGGGS-Pro-Arg-Ser-Leu~Ser-Gly-Lys(MB)-NH₂
8-mer core (P4–P4'): P-R-S-L | S-G-L-K
```
Kridel 2001 phage-display motif: P2=Arg, P1'=Leu, P2'=Ser biases toward MMP-9 over MMP-2/-7/-13. Literature kcat/KM for analogous sequences ~10⁵ M⁻¹s⁻¹.

**Candidate 2 — MMP-2-leaning gelatinase**
```
Ac-Cys-GGGGS-Pro-Leu-Gly~Leu-Ala-Gly-Lys(MB)-NH₂
8-mer core (P4–P4'): P-L-G-L | A-G-x-x
```
ACPP PLGLAG backbone: strong P3-Pro/P1'-Leu deep-S1' fit; MMP-2 shows sharper P1'-Leu/P3-Pro consensus than MMP-9 in PICS/TAILS profiling (Eckhard 2016). Literature kcat/KM ≈ 2.8×10⁴ M⁻¹s⁻¹ for MMP-2 (Tsien/Whitney ACPP data). The ratiometric decay rate of Candidate 1 / Candidate 2 is the practical route to gelatinase A/B partial discrimination.

**Candidate 4 — NE baseline**
```
Ac-Cys-GGGGS-Ala-Ala-Pro-Val~[MB fragment]-Lys-NH₂
8-mer core (P4–P4'): A-A-P-V | x-x-x-x
```
Canonical AAPV core: cathepsin-G-resistant, robust, inexpensive. NE prefers Val ≈ Abu >> Ala at P1 (Kasperkiewicz 2014: Val 4,920; Abu 4,892; Ala 989 M⁻¹s⁻¹). kcat/KM is buffer-dependent (15,000–330,000 M⁻¹s⁻¹) — always report buffer.

### Stage 2 — contingency and stretch

**Candidate 3 — MMP-2 alternative (Nva P1')**
```
Ac-Cys-GGGGS-Pro-Leu-Ala-Nva~...-Lys(MB)-NH₂
```
Murphy 1994: kcat/KM = 3.97×10⁵ M⁻¹s⁻¹ for MMP-2. Norvaline fills the S1' pocket; blocks shallow-pocket proteases. Order only if PLGLAG (Candidate 2) underperforms in the sfGFP screen.

**Candidate 5 — NE high-performance (PK101, computational only)**
```
Ac-Nle(O-Bzl)-Met(O)₂-Oic-Abu-[Cys anchor + MB-Lys]
```
kcat/KM = 4.79×10⁷ M⁻¹s⁻¹ — >7,000× more efficient than AAPV (Kasperkiewicz 2014). Inhibits NE 150× more rapidly than PR3 (cite this figure only — the "~900× substrate selectivity" is an inference from secondary literature, not directly quoted in Tables 2–3 of the paper). Custom synthesis of 4 unnatural residues required; not ordering. Included computationally as the theoretical ceiling for the NE channel.

---

## What the ML pipeline does in this task

1. **Add MMP-2** (MEROPS M10.003) to the training pipeline and retrain — fills the gap in the 3-plex
2. **Score all 5 candidates** through the {MMP-2, MMP-9, NE} classifiers; report where they rank against the full 6,561-candidate PSSM enumeration pool
3. **Validate model agreement with literature**: if Candidates 1, 2, 4 do not score in the top 20% for their target protease, document the discrepancy (likely cause: MEROPS ECM-substrate bias vs. synthetic phage-display sequences)
4. **Apply the electrochemical filter** to the full candidate pool; report pass rates per rule
5. **Compute the 3×3 specificity matrix** for the three Stage 1 sequences and report the condition number

---

## Model quality: what we will build

### Calibration curves

For each protease: bin predicted p_cleave into 10 equal-width bins, plot observed fraction positive vs. predicted probability. A perfectly calibrated model lies on the diagonal. If Expected Calibration Error > 0.1, apply Platt scaling and re-report both calibrated and uncalibrated Brier scores.

### Bootstrapped 95% confidence intervals

Resample CV fold predictions 1,000× with replacement; report 2.5–97.5 percentile range on AUROC, MCC, and PR-AUC.

| Protease | AUROC (95% CI) | MCC (95% CI) | Notes |
|---|---|---|---|
| MMP-2 | TBD after training | TBD | New — M10.003 |
| MMP-9 | 0.787 (0.75–0.82) | 0.46 (0.40–0.52) | Existing |
| NE | 0.825 (0.80–0.85) | 0.47 (0.42–0.51) | Existing, best model |

### Composition baseline

Train a bag-of-AAs model (20-dim AA frequency vector + LR, no positional information). If VHSE8 does not exceed this by ≥0.03 AUROC for a given protease, the positional encoding is not contributing and we document why.

### Literature-handle validation

Score Candidates 1, 2, 4 and the bare literature motifs (PLGLAG, Pro-Arg-X-Leu-Ser, AAPV) through their respective classifiers. Report their percentile rank in the full candidate pool. This is the most direct external validation available before the August sfGFP assay.

### DeepCleave/PROSPER cross-validation

Submit candidates to DeepCleave (avg AUC 0.947, best-in-class for MMP-2) and PROSPER/ProsperousPlus (avg AUC 0.966). Use for narrative cross-validation only — not as a quality gate. Note: both tools perform worse on native ECM proteins than on synthetic peptide libraries (Ozols et al. 2021); treat rankings as hypotheses, not ground truth.

---

## Electrochemical constraints filter

New file: `substrate_design/src/echem_filter.py`. Applied after scoring so pass rates can be reported.

| Rule | Criterion | Justification |
|---|---|---|
| No Cys in 8-mer core | `'C' not in seq` | Cys is the Au–thiol anchor; must be in the flanking construct |
| No Trp | `'W' not in seq` | Electrochemically active at ~0.6 V vs Ag/AgCl; overlaps MB redox window |
| Net charge ∈ [−1, +2] at pH 7.4 | count(R+K) − count(D+E) | Prevents surface repulsion (highly negative) or NiNTA aggregation (highly positive) |
| No poly-Pro | `'PP' not in seq` | Rigid type II' β-turn occludes protease active site |
| No Met (warning, not hard filter) | configurable flag | H₂O₂ blocking steps oxidise Met to Met-sulfoxide |
| MW 800–1200 Da | residue mass sum | Outside this range: weak signal or steric interference with surface |
| Scissile bond at P1–P1' (pos 4–5) | PSSM information content peak | MB must be on the P'-side fragment; anchor on P-side → signal-off on cleavage |

González-Fernández et al. (Biosens. Bioelectron. 2016) establishes MB as superior to ferrocene for this sensor architecture. The ternary SAM (peptide + 6-mercapto-1-hexanol or hexanedithiol diluent) improves packing and signal quality.

---

## 3×3 specificity matrix and condition number

**Inputs:** Candidates 1, 2, 4 as column motifs; MMP-2, MMP-9, NE PSSMs as row scorers. Same normalisation as Task 6 (divide by |self-score|).

**Condition number decision rule:**

| κ(A) | Interpretation | Action |
|---|---|---|
| < 10 | Well-conditioned — channels separable | 3-plex as designed |
| 10–30 | Marginal | 3-plex with calibration note |
| > 30 | Poorly conditioned | Report combined gelatinase + NE 2-plex; explain why |

MMP-2 and MMP-9 are 72% sequence-identical with near-identical efficiency on generic PLG~L substrates. A condition number > 30 is the likely result on sequence-based matrices alone. This is not a failure — it is the correct scientific answer. The ratiometric kinetics and Bayesian deconvolution (Task 9) are the right way to recover partial gelatinase A/B resolution, not a better substrate choice. We frame this explicitly.

If any off-diagonal score ≥ 0.20 is flagged, swap to the next passing candidate from the ranked list and recompute before locking.

**kcat/KM proxy**: reported as predicted order-of-magnitude estimate, calibrated from the single anchor MMP14 SGRSENIR → 5.5×10⁴ M⁻¹s⁻¹. Always shown as ±1 OOM. Replaced by measured values from the August sfGFP assay.

**Redesign threshold**: if sfGFP-measured kcat/KM < 10³ M⁻¹s⁻¹ for any candidate, redesign before ordering MB-peptide synthesis. The sfGFP screen is cheap; MB synthesis is not.

---

## Full construct specification

For each locked Stage 1 sequence:

```
H₂N – Cys – Gly–Gly–Gly–Gly–Ser – [P4–P4' 8-mer] – Gly–Gly–Gly–Gly–Ser – Lys – [His₆] – COOH
       ↑                                                                          ↑
  Au–thiol anchor                                                    MB NHS-ester conjugation site
```

Report per sequence: full construct string, molecular weight, net charge at pH 7.4, predicted cleavage site, confidence tier of underlying ML model, kcat/KM proxy (±1 OOM), and recommendation (order / confirm with sfGFP first).

---

## Flagship model — Task 9 specification

The Bayesian multi-analyte kinetic deconvolution is the Best Model nomination. It unifies every prior workstream into one forward/inverse pair.

### Forward operator: biosensor digital twin

Four coupled physical layers:

1. **Reaction-diffusion transport** — PDE model of protease diffusion to the SAM. Damköhler number Da governs kinetic- vs diffusion-limited regime. Validated against response vs stirring rate.
2. **Surface Michaelis-Menten cleavage** — in the sub-K_M linear regime: `v_j = Σᵢ (kcat/KM)ᵢⱼ × [Eᵢ] × θⱼ`. Matrix seeded from PSSM/VHSE8, then refined by measured sfGFP kinetics.
3. **Marcus-theory electron transfer** — k_ET as a function of MB tunnelling distance (β ≈ 0.40 Å⁻¹ for flexible peptide tethers). Post-Meeting Task 3 Marcus model reused directly; current design (n = 23 spacer residues) gives k_ET ≈ 680 s⁻¹, well within the SWV-compatible window (1–10⁴ s⁻¹).
4. **SWV forward model** — produces R(t) = i_MB/i_Fc as a function of [Eᵢ] and time.

Buffer design constraint: MMP assays require Ca²⁺/Zn²⁺ at pH 7.5; NE works best at pH 8.0 with 0.5 M NaCl. The digital twin models buffer-dependent kcat/KM as a parameter, outputting predicted channel sensitivity under any compromise buffer. This is a genuine design contribution.

### Inverse problem: Bayesian deconvolution

Given measured {vⱼ} from three electrodes, infer {[Eᵢ]}.

**Identifiability analysis:**
- Structural: κ(A) from Task 8 is the linear-algebra proxy
- Practical: profile-likelihood — fix [Eⱼ] on a grid, optimise over others; a flat profile = non-identifiable, a bowl = identifiable

**MCMC (NumPyro — JAX-native, shares autodiff with forward model):**
- Prior: log-normal on [Eᵢ], μ from clinical wound-fluid ranges (Trengove 1999: chronic ~22.8 µg MMP Eq/ml, ~30× acute)
- Likelihood: Normal, σ from Task 2 noise characterisation (0.1% baseline)
- Output: posterior over {[MMP-2], [MMP-9], [NE]} → feeds the disease-posterior healing classifier

The digital twin generates synthetic R(t) curves to validate the inverse model before real device data arrive in August. The story is: simulation-validated now → wet-lab-confirmed in August.

---

## Wet-lab feedback loop

Every dry-lab model feeds a named experimental assay:

```
Substrate Predictor + Cross-Reactivity Matrix
        ↓ three locked sequences
Construct synthesis → sfGFP-fusion cleavage screen
        (team's own construct: FLAG-G4S-sfGFP-G4S-ybbR-G4S-[substrate]-G4S-S6-His6)
        ↓ measured kcat/KM
Updates kinetic/LOD model (Post-Meeting Task 2) + deconvolution matrix
        ↓
Digital twin ↔ Bayesian deconvolution (Task 9)
        ↓ validated on
Multiplexed electrode R(t) curves → disease posterior → healing call
```

This loop is what iGEM judges look for: not a one-directional prediction, but a model that generates hypotheses the wet lab tests and that wet-lab results feed back into.

---

## Additional computational opportunities — medium term

Ranked by feasibility vs. impact:

| Priority | Model | What it answers | Validates against |
|---|---|---|---|
| 1 | Ratiometric error propagation / noise floor | Analytic LOD; how Fc reference cancels drift | Replicate baseline SWV variance |
| 2 | MM parameter estimation with profile-likelihood | Fit kcat/KM from R(t); formal identifiability | Recombinant protease dilution series |
| 3 | Surface reaction-diffusion (Damköhler analysis) | Kinetic- vs transport-limited regime | Response vs stirring/flow rate |
| 4 | SWV frequency optimisation | Signal-on/off crossover; KDM drift correction | Frequency maps pre/post cleavage |
| 5 | Probe-density/packing optimisation | Steric access vs MB signal magnitude | Chronocoulometric surface coverage |
| 6 | Wound-healing classifier | Activity fingerprint → healing prediction | MMP-9 AUC 0.901 as clinical anchor |
| 7 | Multiplex orthogonality optimisation | Substrate set maximising det(A) | Mixed-protease spike-ins |

Priorities 1–2 connect directly to existing Task 2 work and are tractable in one week each.

---

## Benchmark LODs

These bracket the clinically relevant chronic-wound MMP-9 range and confirm target LODs are achievable:

| Sensor | LOD | Linear range |
|---|---|---|
| MB-peptide MMP-9, concentric electrode (Lee 2017) | 7 pM | 1 pM–1 nM |
| Microfabricated Au MB-peptide MMP-9 (González-Fernández) | 60 pM | to 50 nM |
| Signal-off FeMOF MMP-9 (Biosens. Bioelectron. 2023) | 0.34 pM | 5 pM–9 nM |
| Simultaneous MMP-1/2/9 voltammetric array | ~3.5 fM | 0.01–8 pM |

All of these report **concentration LOD** from **abundance sensors**. This project measures **activity rate** — the LOD must be expressed as activity (kcat/KM × [E]) not concentration alone.

---

## References — what to cite / what to avoid

| Status | Reference | Note |
|---|---|---|
| ✅ | Kridel et al. 2001 (J. Biol. Chem. 276:20572) | MMP-9 P2-Arg/P2'-Ser handle |
| ✅ | Kasperkiewicz et al. 2014 (PNAS 111:2518) | PK101; cite >7,000× efficiency gain and **150× probe selectivity over PR3 only** — "~900× substrate selectivity" is not directly in Tables 2–3 |
| ✅ | González-Fernández et al. (Biosens. Bioelectron. 2016) | MB vs Fc superiority |
| ✅ | Eckhard et al. 2016 (Matrix Biol. 49:37; PXD002265) | PICS profiling; data publicly available on ProteomeXchange |
| ✅ | Murphy et al. 1994 (J. Biol. Chem. 269:6632) | MMP-2 substrate kcat/KM = 3.97×10⁵ |
| ✅ | Team construct — sfGFP fusion (FLAG-G4S-sfGFP-G4S-ybbR-G4S-[substrate]-G4S-S6-His6) | Internal cleavage reporter; Sfp/PPTase CoA conjugation; not an external tool |
| ❌ | "Zhu et al. 2023 ACS Appl. Bio Mater." | Cannot be verified — do not cite. Use Ding et al. 2024 (Microchim. Acta 191:5) or ACS AMI 2021, 13, DOI 10.1021/acsami.1c04933 |
| ⚠️ | AAPV kcat/KM | Always report buffer. "~120,000 M⁻¹s⁻¹" is condition-specific; Wiesner 2005 gives 33,915 M⁻¹s⁻¹ |

---

## Development sequence

| Step | What | Estimate |
|---|---|---|
| 1 | Add MMP-2 (M10.003) to both pipeline utils; retrain | 2 hr |
| 2 | Calibration curves, bootstrapped CIs, composition baseline | 4 hr |
| 3 | Score Candidates 1–5 through 3-plex classifiers; literature-handle validation | 1 hr |
| 4 | Electrochemical filter (`src/echem_filter.py`) | 2 hr |
| 5 | 3×3 specificity matrix + condition number; candidate lock or swap | 1 hr |
| 6 | kcat/KM proxy table with ±1 OOM bounds | 0.5 hr |
| 7 | Full construct specification table | 0.5 hr |
| 8 | Report (`report_task8.tex`) | 4 hr |
| 9 | Digital twin forward model (Task 9) | 2–3 days |
| 10 | Bayesian MCMC inversion + identifiability (Task 9) | 2–3 days |
| **Task 8 total** | | **~15 hr** |
