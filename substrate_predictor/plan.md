# Task 7: ML Substrate Peptide Prediction Tool — Implementation Plan

## Overview

A production-quality CLI tool (`predict.py`) that, given a protease name, outputs a
ranked shortlist of candidate 8-9-mer substrate peptides scored for predicted cleavage
probability and selectivity over the rest of the panel. Primary Best Software Tool
candidate for iGEM 2026.

Validation anchor: the MMP14 known motif SGRSENIRTA (8-mer core: SGRSENIR) must rank
in the top 3 for MMP14 and have selectivity index >= 5 over MMP-9 and MMP-8.

---

## Repository Layout

```
task7_ml_predictor/
├── plan.md                         ← this file
├── predict.py                      ← CLI entry point
├── README.md                       ← install, usage, validation output
├── requirements.txt
├── data/
│   ├── raw/                        ← MEROPS downloads (same files as Task 6, symlinked or copied)
│   │   ├── pepunit.lib
│   │   └── merops_scan.lib
│   ├── processed/
│   │   ├── positives_<protease>.csv        ← confirmed cleavage P4-P4' windows (label=1)
│   │   ├── negatives_<protease>.csv        ← three-source negative set (label=0)
│   │   └── training_set_<protease>.csv     ← merged, balanced training set
│   └── outputs/
│       ├── cv_results.csv                  ← 5-fold CV metrics per protease x model
│       └── predictions_<protease>_<date>.csv
├── src/
│   ├── data_pipeline.py            ← MEROPS parsing, positive/negative set construction
│   ├── features.py                 ← VHSE8 physicochemical + ESM2 feature extraction
│   ├── train.py                    ← per-protease classifier training + CV
│   ├── evaluate.py                 ← ROC curves, MCC, calibration, figures
│   ├── selectivity.py              ← selectivity index and joint ranking score
│   └── utils.py                    ← shared helpers (AA alphabet, sequence validation)
├── models/
│   ├── lr_<protease>.pkl           ← trained LogisticRegression (physicochemical features)
│   └── lgbm_<protease>.pkl         ← trained LGBMClassifier (physicochemical + ESM2)
├── figures/
│   ├── fig_roc_curves.png          ← 6-panel ROC, one per protease
│   └── fig_feature_importance.png  ← LR coefficient heatmap
└── tests/
    ├── test_features.py
    └── test_pipeline.py
```

---

## Step-by-Step Implementation Plan

### Step 0: Environment Setup

**File:** `requirements.txt`

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
lightgbm>=4.0
imbalanced-learn>=0.11          # SMOTE
matplotlib>=3.7
seaborn>=0.12
biopython>=1.81
click>=8.1
joblib>=1.3
```

Optional (ESM2 path, graceful degradation if absent):
```
transformers>=4.35
torch>=2.0
```

**Graceful degradation**: `src/features.py` checks for `transformers` at import time.
If absent, ESM2 feature extraction is disabled and a logged warning is emitted.
All primary (physicochemical + LR) models run without it.

---

### Step 1: Data Pipeline

**File:** `src/data_pipeline.py`

#### 1a. Positive examples

Re-use the MEROPS parsing logic from Task 6 (or symlink `data/raw/pepunit.lib`).
For each protease, extract all confirmed P4-P4' 8-mer cleavage windows as positives.
Label = 1.

Minimum positive counts required (from brief):
| Protease | Min positives | Flag |
|---|---|---|
| MMP14 | 500 | - |
| MMP-8 | 300 | - |
| MMP-9 | 300 | - |
| NE | 200 | - |
| FAP | 50 | Low-data: confidence = low |
| Fecal serine | 100 | - |

If MEROPS falls short, `data_pipeline.py` logs a warning and augments with
literature-curated cleavage events (document source in README).

#### 1b. Negative examples: three sources

**Source 1: Hard negatives (substrate context, not cleaved)**
- For each substrate protein in the MEROPS positive set, extract all 8-mer sliding
  windows NOT overlapping any confirmed cleavage site.
- These share protein context with positives, making them harder for the model to
  separate — important for realistic performance estimation.
- Implementation: for each substrate UniProt ID in the positive set, fetch sequence
  from UniProt (via BioPython `ExPASy`). Slide an 8-mer window, flag all windows
  overlapping known cleavage sites as excluded. All others are hard negatives.

**Source 2: Composition-shuffled positives**
- For each positive 8-mer, generate 2 shuffled versions using `random.sample()` on
  the amino acid list. This preserves amino acid composition while destroying
  positional information.
- Controls for composition bias in the feature space.

**Source 3: Random Swiss-Prot 8-mers (easy negatives)**
- Sample random 8-mer windows from a held-out set of Swiss-Prot proteins not present
  in the MEROPS substrate list.
- Provides easy negatives to anchor the decision boundary.

**Mixing ratio**: combine all three sources to target 1:3 positive:negative ratio.
Priority: hard negatives > shuffled > random (if total negatives are limited).

#### 1c. Class imbalance handling

After mixing, if the effective positive:negative ratio is worse than 1:5, apply
SMOTE (`imbalanced-learn`) on the physicochemical feature space to oversample
positives. Log the pre/post-SMOTE class counts.

#### 1d. Output

Save merged, labelled datasets to:
- `data/processed/training_set_<protease>.csv`
  Columns: `sequence (8 chars), label (0/1), source (hard_neg/shuffled/random/positive)`

---

### Step 2: Feature Extraction

**File:** `src/features.py`

#### 2a. VHSE8 physicochemical features (primary)

VHSE8 descriptors encode each amino acid along 8 principal components derived from
240 physicochemical properties (Mei et al. 2005):
- VHSE1: hydrophobicity
- VHSE2: steric
- VHSE3-5: electronic (3 components)
- VHSE6: H-bond donor
- VHSE7: H-bond acceptor
- VHSE8: net charge at physiological pH

Each of 8 positions in the peptide gets 8 VHSE values = 64-dimensional feature vector.

Hard-code the full 20 AA x 8 descriptor VHSE8 table in `src/features.py`.
The values below are from Mei et al. 2005 (Table 2):

```python
VHSE8 = {
    'A': [ 0.15, -1.11, -1.35, -0.92,  0.02, -0.91,  0.36, -0.48],
    'R': [-1.47,  1.45,  1.24,  1.27,  1.05,  1.15,  2.16,  0.85],
    'N': [-0.99,  0.00, -0.37,  0.69, -0.55,  0.85,  0.73, -0.80],
    'D': [-1.15,  0.67, -0.41, -0.01, -2.68,  1.31,  0.03,  0.56],
    'C': [ 0.18, -1.67, -0.46, -0.21,  0.00,  1.20, -1.61, -0.19],
    'Q': [-0.96,  0.12,  0.18,  0.16,  0.09,  0.42, -0.20, -0.41],
    'E': [-1.18,  0.40,  0.10,  0.36, -2.16, -0.17,  0.91,  0.02],
    'G': [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
    'H': [-0.43, -0.25, -0.37,  0.19,  0.51,  1.28,  0.93,  0.65],
    'I': [ 1.27, -0.14,  0.30, -1.80,  0.30, -1.61, -0.16, -0.13],
    'L': [ 1.36,  0.07,  0.26, -0.80,  0.22, -1.37,  0.08, -0.62],
    'K': [-1.17,  0.70,  0.70,  0.80,  1.64,  0.67,  1.63,  0.13],
    'M': [ 1.01, -0.53,  0.43,  0.00,  0.23, -0.68,  0.11, -1.86],
    'F': [ 1.52,  0.61,  0.96, -0.16,  0.25, -1.36,  0.28, -0.60],
    'P': [ 0.22, -0.17, -0.50,  0.05, -0.01, -1.34, -0.19,  3.56],
    'S': [-0.67, -0.86, -1.07, -0.41,  0.39,  0.21, -0.43, -0.25],
    'T': [-0.34, -0.51, -0.55, -1.06,  0.01, -0.01, -0.44, -0.40],
    'W': [ 1.50,  2.06,  1.79,  0.75,  0.75, -0.13, -1.06, -0.60],
    'Y': [ 0.61,  1.60,  1.17,  0.47,  0.45,  0.45, -0.45, -0.41],
    'V': [ 0.76, -0.92,  0.17, -1.91,  0.22, -1.40, -0.24, -0.03],
}
```

The function `encode_vhse8(seq: str) -> np.ndarray` returns shape (64,).

#### 2b. ESM2 embeddings (optional, comparison model)

Model: `esm2_t6_8M_UR50D` (8M parameters, CPU-feasible).
- Load via HuggingFace: `AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")`
- Input: single 8-mer sequence string
- Extract last-layer mean-pooled representation over residue tokens (exclude [CLS] and [EOS])
- Output: 320-dimensional embedding vector

Availability check:
```python
def esm2_available() -> bool:
    try:
        import transformers
        import torch
        return True
    except ImportError:
        return False
```

If ESM2 is available: cache embeddings to `data/processed/esm2_cache_<protease>.pkl`
to avoid recomputation. The cache is keyed by sequence string.

If not available: log a warning and skip ESM2 in all downstream steps.

#### 2c. Combined features for LightGBM

For LightGBM: concatenate VHSE8 (64-dim) + ESM2 (320-dim) = 384-dim feature vector.
If ESM2 unavailable: LightGBM falls back to VHSE8-only (64-dim).

---

### Step 3: Classifier Training

**File:** `src/train.py`

#### 3a. Logistic Regression (primary model)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'))
])
```

Features: VHSE8 (64-dim).
`class_weight='balanced'` handles residual imbalance after SMOTE.

#### 3b. LightGBM (comparison model)

```python
import lightgbm as lgb

lgbm_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbose': -1
}
```

Features: VHSE8 + ESM2 (384-dim) if available, else VHSE8-only.

#### 3c. Stratified 5-fold cross-validation

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Metrics computed per fold:
- AUROC: `roc_auc`
- MCC: `matthews_corrcoef` (via `make_scorer`)
- Precision-Recall AUC: `average_precision`
- Brier score: `brier_score_loss` (calibration)

Report mean +/- std across 5 folds.

#### 3d. Final model fitting

After CV, refit each model on the full training set and serialize:
- `models/lr_<protease>.pkl` (joblib)
- `models/lgbm_<protease>.pkl` (joblib)

#### 3e. FAP handling

If FAP has < 50 positives after augmentation, set confidence = "low" and
skip LightGBM training (not enough data to prevent overfitting). LR only.
Log a warning: "FAP model trained on <N> positives; predictions are low confidence."

---

### Step 4: Evaluation Figures

**File:** `src/evaluate.py`

#### 4a. ROC curve figure

`figures/fig_roc_curves.png`:
- 6-panel subplot (2 rows x 3 cols), one per protease
- Both LR (solid line) and LightGBM (dashed line) on the same axes
- AUROC values in the legend
- Diagonal (random) reference line
- X-axis: False Positive Rate, Y-axis: True Positive Rate
- Figure size: (14, 9) inches, 300 dpi

#### 4b. Feature importance heatmap

`figures/fig_feature_importance.png`:
- Uses the LR model's `.coef_[0]` attribute (64 values)
- Reshape to (8, 8): rows = positions P4 through P4', columns = VHSE1-VHSE8
- seaborn heatmap with `annot=False`, `cmap='RdBu_r'`
- Row labels: P4, P3, P2, P1, P1', P2', P3', P4'
- Column labels: VHSE1 (hydrophobic), VHSE2 (steric), VHSE3-5 (electronic),
  VHSE6 (H-bond don.), VHSE7 (H-bond acc.), VHSE8 (charge)
- Title: "LR coefficient heatmap: position x VHSE descriptor"
- Figure size: (10, 8) inches, 300 dpi
- This is the biologically interesting figure: shows which position/property
  combinations drive prediction for each protease.

Produce one subplot per protease, or show the MMP14 figure as the primary example
with others in supplementary.

---

### Step 5: Selectivity Scoring

**File:** `src/selectivity.py`

#### 5a. Selectivity index

For a candidate peptide, run it through all 6 trained LR classifiers and collect
predicted probabilities `p_1 ... p_6` (using `predict_proba`):

```
S_i = p_i / max(p_j for j != i)
```

This is the ratio of the target protease's predicted cleavage probability to the
highest competing protease's probability.

#### 5b. Joint ranking score

```
score = p_i * log(S_i + 1)
```

The log term softens extreme selectivity values; +1 avoids log(0) when S_i = 0.

#### 5c. Output table

| Rank | Motif | P(cleave) | Selectivity | Joint score | Confidence |
|---|---|---|---|---|---|
| 1 | SGRSENIR | 0.94 | 12.3 | 2.64 | high |

Confidence tiers:
- high: trained on >= 200 positives
- medium: 50-199 positives
- low: < 50 positives

---

### Step 6: CLI Entry Point

**File:** `predict.py`

#### 6a. Usage

```
python predict.py --protease MMP14 --top-k 10
python predict.py --protease MMP-8 --scan SEQUENCE.fasta --top-k 5
```

#### 6b. Implementation using `click`

```python
import click

@click.command()
@click.option('--protease', required=True,
              type=click.Choice(['MMP14', 'MMP-8', 'MMP-9', 'NE', 'fecal', 'FAP'],
                                case_sensitive=False))
@click.option('--top-k', default=10, show_default=True)
@click.option('--scan', type=click.Path(exists=True), default=None,
              help='FASTA file to scan with sliding 8-mer window')
@click.option('--models-dir', default='models/', show_default=True)
def main(protease, top_k, scan, models_dir):
    ...
```

#### 6c. Default mode (no --scan)

Generate candidate peptides by:
1. For each of the 8 positions in the PSSM, take the top 3 amino acids by PSSM score.
2. Enumerate all combinations: 3^8 = 6,561 candidates.
3. Score all candidates through the selectivity pipeline.
4. Return top-k by joint ranking score.

This is computationally trivial (< 1 second for 6,561 x 6 classifiers).

#### 6d. Scan mode (--scan FASTA)

Parse the FASTA file with BioPython. Slide an 8-mer window across each sequence.
Score each window through the selectivity pipeline for the target protease.
Return top-k hits with their position (sequence ID, start position, end position).

Output columns: Rank, Position, Motif, P(cleave), Selectivity, Joint score, Confidence

#### 6e. Output format

Print to stdout as a formatted table using Python's `tabulate` library or
manual string formatting. Also save to
`data/outputs/predictions_<protease>_<YYYYMMDD>.csv`.

---

### Step 7: Validation Test Case

**File:** `README.md` (documented), verified by running the trained model.

Test: `python predict.py --protease MMP14 --top-k 10`

Requirements (from brief):
1. SGRSENIR (8-mer core of SGRSENIRTA) must appear in top 3 results.
2. Selectivity index for SGRSENIR over MMP-9 must be >= 5.
3. Selectivity index for SGRSENIR over MMP-8 must be >= 5.

The README will contain the actual stdout output from this command, proving the
model works on the known positive control.

---

### Step 8: Tests

**File:** `tests/test_features.py`

- Test `encode_vhse8('SGRSENIR')` returns shape (64,) with correct values for known AAs
- Test `encode_vhse8` raises ValueError for invalid/non-standard AA
- Test ESM2 availability check function
- Test feature vector concatenation shape

**File:** `tests/test_pipeline.py`

- Test MEROPS parser extracts correct window length (8) from a mock record
- Test negative set generation produces correct label column (all 0)
- Test SMOTE is applied when positive:negative ratio > 1:5
- Test training set CSV has required columns

Run with: `pytest tests/ -v`

---

### Step 9: LaTeX Snippet

**File:** `reports/task7_latex_snippet.tex`

Ready to paste into `protease_panel_characterisation.tex` as `\section{Task 7: ...}`.
Length: <= 2 pages in the document.

Content:
1. Methods description: data pipeline, feature engineering, model architecture
2. Validation: SGRSENIRTA test case with actual model output
3. Figures: ROC curves (`fig_roc_curves.png`) and feature importance heatmap
   (`fig_feature_importance.png`)
4. Cross-reference to Task 6: the motifs produced by Task 7 feed the Task 6
   re-run for the final specificity matrix

---

## Circular Dependency with Task 6

Task 6 needs the final panel motifs (one per protease) as column headers.
Task 7 produces those motifs. The resolution:

1. **Phase 1**: Run Task 6 with literature placeholder motifs to verify the PSSM/scoring
   pipeline is correct.
2. **Phase 2**: Run Task 7 to train classifiers and produce ranked motif lists.
3. **Phase 3**: Select the top-ranked motif from Task 7 for each protease, update the
   motif table in Task 6's `score_motifs.py`, and re-run Task 6 to produce the final
   specificity matrix.

Document this dependency chain in both READMEs.

---

## Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MEROPS positives below minimum threshold | Medium (FAP certain) | Log warning; augment with literature data; set confidence = low |
| ESM2 download fails or too slow | Medium | Graceful fallback to physicochemical-only; LightGBM uses VHSE8 only |
| LightGBM overfits on small datasets (FAP) | High | Skip LightGBM for < 50 positives; LR only with high regularization |
| SMOTE distorts feature space | Low | Compare CV metrics pre/post SMOTE; use only if improvement is significant |
| SGRSENIR fails top-3 validation | Low | Investigate: check PSSM coverage, check feature encoding, check class balance |
| UniProt sequence fetch fails (hard negatives) | Medium | Fallback to shuffled + random negatives only if UniProt unreachable |
| `tabulate` library not installed | Low | Fall back to pandas `.to_string()` for table output |

---

## Execution Order

```
Step 0: install requirements
Step 1: data_pipeline.py     ← builds training sets
Step 2: features.py          ← verify feature extraction (also used in steps 3, 5, 6)
Step 3: train.py             ← fits all models, saves to models/
Step 4: evaluate.py          ← generates ROC + feature importance figures
Step 5: selectivity.py       ← used live in predict.py, also tested standalone
Step 6: predict.py           ← CLI, depends on trained models in models/
Step 7: README validation    ← run predict.py and paste output
Step 8: tests/               ← run pytest
Step 9: LaTeX snippet        ← write after all figures are finalised
```

---

## Deliverables Checklist

- [ ] `data/processed/training_set_<protease>.csv` for all 6 proteases
- [ ] `data/outputs/cv_results.csv`
- [ ] `models/lr_<protease>.pkl` for all 6 proteases
- [ ] `models/lgbm_<protease>.pkl` for all 6 proteases (except FAP if low-data)
- [ ] `figures/fig_roc_curves.png` at 300 dpi
- [ ] `figures/fig_feature_importance.png` at 300 dpi
- [ ] Working CLI: `python predict.py --protease MMP14 --top-k 10`
- [ ] `README.md` with SGRSENIR validation output
- [ ] `reports/task7_latex_snippet.tex`
- [ ] All pytest tests passing

---

## Estimated Compute

| Step | Time (CPU, no GPU) |
|---|---|
| Data pipeline (including UniProt fetch) | ~5-10 min |
| VHSE8 feature extraction (all proteases) | < 1 min |
| ESM2 embedding (if available, CPU) | ~30 min for all training data |
| LR training (all proteases, 5-fold CV) | < 2 min |
| LightGBM training (all proteases, 5-fold CV) | ~5-15 min |
| Evaluation figures | ~2 min |
| Candidate generation + scoring | < 1 min |
| **Total (no ESM2)** | **~15-25 min** |
| **Total (with ESM2, CPU)** | **~50-60 min** |
