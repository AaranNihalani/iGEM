# Substrate Design — Task 8
**Eton iGEM 2026 · Dry Lab**

Device-ready 3-plex substrate selection for the chronic-wound protease diagnostic panel (MMP-2 · MMP-9 · NE). This folder takes the ML predictor (Task 7) and cross-reactivity matrix (Task 6) and turns them into three locked peptide sequences the wet lab can order.

---

## What this folder produces

1. **Three locked 8-mer P4–P4' sequences** — one per protease channel, literature-seeded, ML-validated, electrochemically filtered
2. **3×3 specificity matrix + condition number** — honest identifiability analysis; determines whether MMP-2/MMP-9 are separable or must be reported as a combined gelatinase channel
3. **Full construct specifications** — Cys–GGGGS–[8-mer]–Lys(MB) ready to copy into a synthesis order form
4. **Task 8 report** (`report_task8.pdf`) — same LaTeX style as Tasks 6 and 7

---

## How it connects to the rest of the repo

```
substrate_predictor/    →  VHSE8 + LR/LGBM models (train here, use here)
cross_reactivity_matrix/ →  PSSM scorer (reused for the 3×3 matrix)
substrate_design/       →  THIS FOLDER: validation, filtering, locking sequences
```

The pipeline runs in one direction before wet-lab data arrive; once the August sfGFP cleavage assay returns measured kcat/KM values, those numbers feed back into the kinetic model (Task 2) and populate the deconvolution mixing matrix (Task 9).

---

## Planned folder structure

```
substrate_design/
├── plan.md                      # Full development plan (read this first)
├── README.md                    # This file
├── src/
│   ├── echem_filter.py          # Electrochemical constraints filter
│   ├── score_candidates.py      # 3-plex scoring restricted to {MMP-2, MMP-9, NE}
│   ├── condition_number.py      # 3×3 matrix + κ(A) + identifiability
│   └── calibration.py           # Calibration curves + bootstrapped CIs
├── data/
│   └── outputs/
│       ├── candidate_scores.csv
│       ├── specificity_matrix_3plex.csv
│       └── locked_sequences.csv
├── figures/
│   ├── fig_calibration_curves.png
│   ├── fig_3plex_heatmap.png
│   └── fig_roc_with_ci.png
├── report_task8.tex
└── report_task8.pdf
```

---

## Candidate sequences (Stage 1 order)

| Channel | Candidate | 8-mer core | kcat/KM literature anchor |
|---|---|---|---|
| MMP-9-biased | 1 | PRSL\|SGLK | ~10⁵ M⁻¹s⁻¹ (Kridel 2001 analogues) |
| Gelatinase (MMP-2-leaning) | 2 | PLGL\|AGxx | 2.8×10⁴ M⁻¹s⁻¹ (Tsien/Whitney ACPP) |
| NE baseline | 4 | AAPV\|xxxx | 15,000–330,000 M⁻¹s⁻¹ (buffer-dependent) |

Candidates 3 (Nva P1', MMP-2) and 5 (PK101, NE) are Stage 2 / computational-only respectively. See `plan.md` for full rationale.

---

## Key decision: condition number threshold

If κ(A) > 30 for the 3×3 MMP-2/MMP-9/NE matrix, the gelatinase channels cannot be separated by substrate selectivity alone. In that case the report pivots to a **2-plex: {gelatinase + NE}** and explains that ratiometric kinetics + Bayesian deconvolution (Task 9) is the correct route to partial A/B resolution. This is not a failure — it is the honest identifiability result.

---

## Dependencies

```
substrate_predictor/models/lr_MMP_2.pkl   # NEW — train first (Step 1)
substrate_predictor/models/lr_MMP_9.pkl   # existing
substrate_predictor/models/lr_NE.pkl      # existing
cross_reactivity_matrix/src/              # PSSM scorer reused directly
```

Run `python substrate_predictor/train_models.py` with MMP-2 added to `MEROPS_IDS` before running anything in this folder.
