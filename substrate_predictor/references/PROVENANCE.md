# Archived reference data used by the substrate predictor

| File | What it is | How it is used |
|---|---|---|
| vhse8_table_peptides_0.5.0.json | VHSE1-VHSE8 descriptor values for the 20 amino acids, as distributed in the `peptides` Python package v0.5.0 (PyPI) | Feature encoding (src/features.py) |
| blosum62_biopython.json | BLOSUM62 substitution matrix from Biopython 1.87 | PSSM pseudocounts (src/pssm.py) |
| swissprot_composition_2026_03.json, expasy_relstat_2026_03.html | Amino-acid composition of UniProtKB/Swiss-Prot release 2026_03 (Expasy release statistics, section 6.1) | Background frequencies for PSSM log-odds and random negatives |

## Sources for training and validation data

| Data | Source | Where |
|---|---|---|
| Cleavage windows, all proteases | MEROPS `Substrate_search.txt` | `data/raw/` (not tracked) |
| MMP-2/-8/-9/-14 PICS windows | Eckhard et al. 2016 supplementary tables (mmc4/mmc5), mapped onto UniProt sequences | `data/sources/eckhard2016_*.csv` |
| LasB windows in elastin | Yang et al. 2015, Supplementary Table S1 (extracted from the PDF; all 75 rows verified against UniProt P04985) | `data/sources/yang2015_*` |
| Measured MMP-9 kinetics | Kridel et al. 2001, Table II | `data/validation/kridel2001_table_ii.csv` |
