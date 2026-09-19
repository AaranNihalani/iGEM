# Archived reference data used by the substrate predictor

| File | What it is | How it is used |
|---|---|---|
| vhse8_table_peptides_0.5.0.json | VHSE1-VHSE8 descriptor values for the 20 amino acids, as distributed in the `peptides` Python package v0.5.0 (PyPI) | Feature encoding (src/features.py) |
| blosum62_biopython.json | BLOSUM62 substitution matrix from Biopython 1.87 | PSSM pseudocounts (src/pssm.py) |
| swissprot_composition_2026_03.json, expasy_relstat_2026_03.html | Amino-acid composition of UniProtKB/Swiss-Prot release 2026_03 (Expasy release statistics, section 6.1) | Background frequencies for PSSM log-odds and random negatives |
