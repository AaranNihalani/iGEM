# Task 6: Cross-Reactivity Specificity Matrix — Implementation Plan

## Overview

Produce a publication-quality 6x6 cross-reactivity heatmap and supporting CSV/LaTeX
deliverables confirming that the six panel substrate motifs are orthogonal. Each cell
answers: "how efficiently does protease i cleave motif j?" Diagonal = 1.0 (self-score),
off-diagonal ideally < 0.05 (orthogonal), flagged if >= 0.2.

---

## Directory Layout

```
task6_specificity_matrix/
├── plan.md                     ← this file
├── requirements.txt            ← Python dependencies
├── data/
│   ├── raw/                    ← MEROPS bulk downloads (do not modify after download)
│   │   ├── pepunit.lib         ← all cleavage-site sequences from MEROPS FTP
│   │   └── merops_scan.lib     ← per-substrate records (backup/cross-check)
│   ├── processed/
│   │   ├── cleavage_windows_<protease>.csv   ← extracted P4-P4' windows per protease
│   │   └── swissprot_aa_background.json      ← Swiss-Prot amino acid frequencies (background)
│   └── outputs/
│       ├── specificity_matrix_raw.csv        ← raw PSSM log-odds scores
│       └── specificity_matrix_normalised.csv ← self-score normalised (diagonal = 1.0)
├── src/
│   ├── parse_merops.py         ← Step 1: parse pepunit.lib, extract windows
│   ├── build_pssm.py           ← Step 2: build per-protease PSSMs
│   ├── score_motifs.py         ← Step 3 & 4: score panel motifs, normalise
│   ├── plot_heatmap.py         ← Step 6: heatmap figure
│   └── utils.py                ← shared helpers (AA alphabet, background freqs)
├── figures/
│   └── fig_specificity_heatmap.png
├── reports/
│   └── task6_specificity.md    ← written interpretation, flagged pairs, caveats
└── run_all.py                  ← single entry point: runs all steps end-to-end
```

---

## Step-by-Step Implementation Plan

### Step 0: Environment Setup

**File:** `requirements.txt`

Dependencies:
```
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
biopython>=1.81
requests>=2.31
```

No GPU required. All computation is CPU-bound matrix operations.

---

### Step 1: Parse MEROPS Bulk Data

**File:** `src/parse_merops.py`

#### 1a. Download strategy

MEROPS bulk data is available at:
- FTP: `ftp://ftp.ebi.ac.uk/pub/databases/merops/current_release/`
- Key file: `pepunit.lib` — one record per confirmed cleavage event, with substrate
  sequence and scissile-bond position annotated.
- Format: MEROPS-specific flat-file; each record has fields including MER ID (protease),
  substrate protein, cleavage-site sequence (typically given as P4-P4' = 8 residues),
  and species.

The script will attempt FTP download; if FTP is blocked, fall back to a curated
manual-download instruction and a synthetic stub for testing.

#### 1b. MEROPS identifiers to extract

| Protease | MEROPS ID | Notes |
|---|---|---|
| MMP14 | MER000220 | Family M10; best-characterized in panel |
| MMP-8 | MER000218 | Family M10 |
| MMP-9 | MER000219 | Family M10 |
| Neutrophil elastase (NE) | MER000053 | Family S1 |
| FAP | MER005153 | Family S9; sparse data (<20 entries expected) |
| Trypsin-like (fecal) | MER000001 | Family S1; used as fecal sub-row |
| Chymotrypsin-like (fecal) | MER000003 | Family S1; used as fecal sub-row |

Fecal serine proteases will be represented by the union of trypsin-like,
chymotrypsin-like, and elastase-like (MER000053 shared with NE) PSSMs.
If space is tight in the heatmap, they are collapsed to a single "Fecal serine" row
using the union set.

#### 1c. Parsing logic

For each record in `pepunit.lib`:
1. Identify the MER ID field.
2. Extract the substrate sequence string around the scissile bond.
3. Identify the cleavage position (P1|P1' boundary).
4. Slice out an exactly 8-residue window: P4, P3, P2, P1, P1', P2', P3', P4'.
5. Validate: window must be exactly 8 standard amino acids (no X, B, Z, U).
6. Append to per-protease list.

Output: `data/processed/cleavage_windows_<protease>.csv` with columns:
`mer_id, substrate_id, sequence (8 chars), source_species`

#### 1d. Data quality checks

- Log count of extracted windows per protease.
- Flag FAP if < 20 entries (as noted in brief).
- Flag any protease with < 50 entries (model will have low confidence).
- For each window, verify all 8 positions are standard 20 AAs.

---

### Step 2: Build Position-Specific Scoring Matrices (PSSMs)

**File:** `src/build_pssm.py`

#### 2a. Background amino acid frequencies

Use Swiss-Prot background frequencies (Uniprot 2023 release). These are the
standard reference for log-odds scoring. Hard-code in `utils.py`:

```python
# Swiss-Prot background frequencies (Robinson & Robinson 1991, updated)
SWISSPROT_BG = {
    'A': 0.0777, 'R': 0.0526, 'N': 0.0427, 'D': 0.0530, 'C': 0.0157,
    'Q': 0.0391, 'E': 0.0629, 'G': 0.0691, 'H': 0.0228, 'I': 0.0591,
    'L': 0.0960, 'K': 0.0573, 'M': 0.0237, 'F': 0.0405, 'P': 0.0492,
    'S': 0.0694, 'T': 0.0550, 'W': 0.0130, 'Y': 0.0322, 'V': 0.0667
}
```

#### 2b. PSSM construction

For protease p with N confirmed cleavage windows of length 8:

1. **Count matrix C[pos][aa]**: count occurrences of amino acid `aa` at position
   `pos` (0-7 = P4 through P4').
2. **Pseudocount**: add Laplace pseudocount of 0.5 to avoid zero entries.
   `C_pseudo[pos][aa] = C[pos][aa] + 0.5`
3. **Frequency matrix f[pos][aa]**: `f = C_pseudo / sum(C_pseudo[pos])`
4. **Log-odds matrix**: `PSSM[pos][aa] = log2(f[pos][aa] / BG[aa])`

Output shape: 8 positions x 20 amino acids = (8, 20) numpy array per protease.

The PSSM captures which amino acids are enriched or depleted at each position
of the cleavage window for that protease.

#### 2c. Fecal serine PSSM

Merge all windows from trypsin-like, chymotrypsin-like, and elastase-like (NE)
into a single pool, then build one PSSM from the merged set. Note in the report
that this is a heterogeneous mixture and confidence is lower.

#### 2d. Serialise PSSMs

Save as `data/processed/pssm_<protease>.npy` (numpy array) and
`data/processed/pssm_<protease>.csv` (human-readable, for verification).

---

### Step 3: Score Panel Motifs Against All PSSMs

**File:** `src/score_motifs.py`

#### 3a. Panel motifs

The six motifs to score (one per protease column in the matrix):

| Protease | Motif | Status |
|---|---|---|
| MMP14 | SGRSENIR | Fixed, validated (8-mer core of SGRSENIRTA) |
| MMP-8 | TBD | Output of Task 7; use placeholder for now |
| MMP-9 | TBD | Output of Task 7; use placeholder for now |
| NE | TBD | Output of Task 7; use placeholder for now |
| Fecal serine | TBD | Output of Task 7; use placeholder for now |
| FAP | TBD | Output of Task 7; use placeholder for now |

**Important:** Task 6 and Task 7 have a circular dependency here. The plan is:
- Run Task 6 once with literature-sourced placeholder motifs (selected from known
  substrates in the MEROPS literature for each protease, not wet-lab validated).
- After Task 7 produces its ranked shortlist, re-run Task 6 with the top-ranked
  motifs from Task 7 to produce the final matrix.

**Placeholder motifs for initial run** (from literature, not Task 7 output):
- MMP-8: `GPLGIAGQ` (collagen-derived, known MMP-8 substrate, from Turk et al. 2001)
- MMP-9: `VPLSLYSG` (gelatin-derived, known MMP-9 substrate)
- NE: `AAPVAALS` (elastin-derived, canonical NE substrate)
- Fecal serine: `KQLRAIRM` (trypsin-like substrate, Arg at P1)
- FAP: `GPRFPIGS` (FAP canonical dipeptidyl peptidase substrate, from Edosada 2006)

Document all placeholder sources in `reports/task6_specificity.md`.

#### 3b. Scoring function

For a given motif M (8 residues) scored under PSSM P:
```
score(M, P) = sum(P[pos][M[pos]] for pos in 0..7)
```
This is the standard PSSM log-odds sum score.

#### 3c. Build raw score matrix

Rows = 6 (or 8 if fecal is split) proteases.
Columns = 6 motifs.
Cell (i, j) = PSSM score of motif j under protease i's PSSM.

#### 3d. Normalise

For each row i (protease i):
```
normalised[i][j] = raw[i][j] / raw[i][i]
```
So the diagonal = 1.0 (self-score).

**Edge case**: if the raw self-score is negative (possible for log-odds scores),
normalisation flips signs. Handle by using the absolute value of the diagonal
for normalisation and clamping the result to [-0.5, 2.0] for display.
Document this in the report if it occurs.

---

### Step 4: Compute Selectivity Ratios

**File:** `src/score_motifs.py` (continuation)

For each off-diagonal cell (i, j where i != j):
```
selectivity_ratio[i][j] = normalised[i][j] / normalised[i][i]
                        = normalised[i][j]   (since diagonal = 1.0 after normalisation)
```

**Flag cells**: if selectivity_ratio >= 0.20, flag with a red annotation in the heatmap.

Save to `data/outputs/specificity_matrix_normalised.csv` with columns:
`protease, MMP14_motif, MMP8_motif, MMP9_motif, NE_motif, fecal_motif, FAP_motif`

Also save `data/outputs/specificity_matrix_raw.csv` with raw log-odds scores.

---

### Step 5: Produce the Heatmap Figure

**File:** `src/plot_heatmap.py`

#### 5a. Figure specification (from brief)

- Library: seaborn `heatmap` with `annot=True`, `fmt=".2f"`
- Colourmap: `RdYlGn_r` — red = high cross-reactivity, green = orthogonal
- Diagonal: outlined in black (use a secondary axes overlay to draw black rectangles
  on each diagonal cell)
- Row labels: protease names (MMP14, MMP-8, MMP-9, NE, Fecal serine, FAP)
- Column labels: the 8-mer motif sequences
- Figure size: (9, 7) inches
- DPI: 300, `bbox_inches='tight'`
- Output: `figures/fig_specificity_heatmap.png`

#### 5b. Annotation logic

Each cell is annotated with its normalised score to 2 decimal places.
Cells with score >= 0.20 (off-diagonal) are additionally annotated in red bold text
by overlaying a text annotation via `ax.text()` after the seaborn call.

#### 5c. Figure style

Apply consistent style with the existing document figures:
- Font: sans-serif (default matplotlib), 10pt labels
- Axis titles: "Protease (row)" on y-axis, "Substrate motif (column)" on x-axis
- Colour bar label: "Normalised cleavage score (self = 1.0)"
- Title: "Cross-reactivity specificity matrix: protease panel vs. substrate motifs"
- No em dashes in any text

---

### Step 6: Write the Interpretation Report

**File:** `reports/task6_specificity.md`

Structure:
1. **Summary**: one paragraph, what the matrix shows and the main conclusions
2. **Flagged pairs**: list any (protease, motif) pairs with selectivity ratio >= 0.20,
   with a brief mechanistic explanation (e.g., shared M10 family for MMP14/MMP-8/MMP-9)
3. **Orthogonal pairs**: confirm which pairs are < 0.05
4. **Caveats**:
   - FAP: sparse MEROPS data (< 20 entries); PSSM is low-confidence
   - Fecal serine: heterogeneous mixture of trypsin/chymotrypsin/elastase activities;
     PSSM represents population average, not a single enzyme
   - Placeholder motifs: MMP-8, MMP-9, NE, fecal, FAP motifs are literature-sourced
     placeholders; matrix should be re-run with Task 7 final motifs
   - PSSM scores are in vitro; in-sample matrix effects (protease inhibitors, pH) not modelled
5. **Confidence bands**: report 95% bootstrap CI on each PSSM score by resampling
   the training windows (500 bootstrap resamples)

---

### Step 7: LaTeX Snippet

**File:** `reports/task6_latex_snippet.tex`

Ready to paste into `protease_panel_characterisation.tex` as a new `\section{Task 6: ...}`.

Style rules (from existing document):
- `\section{Task 6: Cross-Reactivity Specificity Matrix}`
- `\subsection{...}` for sub-sections
- `\begin{figure}[H]` with `\includegraphics[width=0.96\textwidth]{...}`
- booktabs tables if needed
- No em dashes; use colons, semicolons, commas
- `\usepackage{float}` already in preamble

---

### Step 8: Integration Run Script

**File:** `run_all.py`

Single command to reproduce all outputs from scratch:
```
python run_all.py [--download] [--skip-download]
```

Flags:
- `--download`: attempt FTP download of MEROPS data (requires network)
- `--skip-download`: use existing files in `data/raw/`; error if absent

The script runs Steps 1-7 in order, with progress logging to stdout.

---

## Dependency Notes

- Steps 1-3 are sequential (each feeds the next).
- Step 4 depends on Step 3.
- Steps 5-7 all depend on Step 4 but are independent of each other (can run in parallel).
- Step 8 is the wrapper.

---

## Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| FTP access to MEROPS blocked | Medium | Manual download fallback; script checks for cached files first |
| FAP has < 20 MEROPS entries | High (expected) | Flag explicitly in report; use low-confidence PSSM; add note to heatmap |
| Negative self-score (diagonal) | Low | Clamp and document; use absolute value for normalisation |
| Fecal serine PSSM too broad to be specific | High | Collapse to trypsin/chymotrypsin sub-rows if separation is informative |
| MEROPS file format changed | Low | Add format version check; use BioPython's SeqIO as fallback parser |
| Task 7 motifs not yet available | Certain (first run) | Use validated literature placeholders; re-run after Task 7 |

---

## Deliverables Checklist

- [ ] `data/processed/cleavage_windows_<protease>.csv` for all 6 proteases
- [ ] `data/processed/pssm_<protease>.npy` for all 6 proteases
- [ ] `data/outputs/specificity_matrix_raw.csv`
- [ ] `data/outputs/specificity_matrix_normalised.csv`
- [ ] `figures/fig_specificity_heatmap.png` at 300 dpi
- [ ] `reports/task6_specificity.md`
- [ ] `reports/task6_latex_snippet.tex`

---

## Estimated Compute

| Step | Time (CPU) |
|---|---|
| MEROPS download | ~2 min (network) |
| Parsing pepunit.lib | ~30 s |
| PSSM construction | < 1 s |
| Scoring all motifs | < 1 s |
| Bootstrap CI | ~10 s (500 resamples x 6 proteases) |
| Heatmap rendering | ~5 s |
| **Total** | **~3-5 min** |

No GPU required.
