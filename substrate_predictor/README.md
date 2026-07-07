# Substrate Predictor

ML tool for ranking candidate 8-mer substrate peptides for each protease in the Eton iGEM 2026 diagnostic panel. Primary candidate for the iGEM Best Software Tool prize.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Train models

```bash
# Full pipeline: download data (via Task 6), build training sets, train, generate figures
python train_models.py

# Skip data prep if training sets already exist
python train_models.py --skip-data

# Include ESM2 embeddings (requires torch + transformers)
python train_models.py --esm2
```

### Predict

```bash
# Rank top 10 candidate motifs for MMP14
python predict.py --protease MMP14 --top-k 10

# Scan a FASTA sequence for MMP-8 cleavage sites
python predict.py --protease MMP-8 --scan SEQUENCE.fasta --top-k 5

# Save predictions to CSV
python predict.py --protease MMP14 --top-k 10 --save
```

### Run tests

```bash
pytest tests/ -v
```

## Validation: SGRSENIRTA (MMP14)

The known MMP14 sensor motif (8-mer core: SGRSENIR) must rank in the top 3, with
selectivity index >= 5 over MMP-9 and MMP-8.

```
[ACTUAL OUTPUT TO BE PASTED HERE AFTER TRAINING]
```

## How it works

**Data:** Positive training examples are confirmed P4-P4' cleavage windows from the MEROPS
substrate database. Negatives are shuffled positives (composition-matched) plus random
sequences sampled from Swiss-Prot background amino acid frequencies, combined at a 1:3
positive-to-negative ratio.

**Features:** Each 8-mer is encoded as a 64-dimensional VHSE8 physicochemical vector
(8 positions x 8 descriptors: hydrophobicity, steric bulk, 3 electronic components,
H-bond donor, H-bond acceptor, charge). This encoding is biologically interpretable
and maps directly to the protease subsite specificity framework.

**Models:** Per-protease logistic regression (L2, primary) trained on VHSE8 features.
A LightGBM comparison model is also trained where sufficient data exists. Both are
evaluated with stratified 5-fold cross-validation (AUROC, MCC, PR-AUC, Brier score).

**Selectivity:** For a candidate peptide, all 6 per-protease classifiers return predicted
cleavage probabilities. The selectivity index is `p_target / max(p_competitors)`. The
joint ranking score is `p_target * log(selectivity + 1)`.

**Candidate generation (default mode):** 6,561 candidates are enumerated from the top-3
PSSM-preferred amino acids at each of the 8 positions (3^8). All are scored and ranked.

## Confidence tiers

| Tier | Training positives |
|---|---|
| high | >= 200 |
| medium | 50-199 |
| low | < 50 |

FAP has only 2 MEROPS entries and is low-confidence throughout.
