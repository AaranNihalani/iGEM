# Substrate predictor

Per-protease models that score 8-residue substrate windows (P4-P4') and rank candidate peptides by
how specifically they are recognised by a target protease within a panel.

```
python -m spred build-data              # rebuild data/positives/ from source files
python -m spred train                   # cluster-aware CV, nested tuning, fit + save models/*.json
python -m spred predict GPLGIAGQ AAPVSGIG --panel wound
python -m spred rank --target MMP9 --panel wound --pool 50000
python -m spred mutate IPRTLTAA --target MMP9      # every single substitution
python -m spred scan protein.fasta --target NE
make test study validate                # tests, methodology study, external validation
make web                                # build report/substrate_predictor_explorer.html (needs node)
```

## Layout

| Path | Contents |
|---|---|
| `spred/` | package: `tables` (verified VHSE8, BLOSUM62, Swiss-Prot background), `features`, `pssm`, `data`, `evaluation`, `model`, `panel`, `candidates`, `validation`, `train`, `cli` |
| `data/sources/` | derived source tables (Eckhard 2016 windows, Yang 2015 Table S1, tropoelastin sequence) |
| `data/positives/` | final positive windows per protease with provenance |
| `data/validation/` | Kridel et al. 2001 Table II (measured MMP-9 kinetics) |
| `data/raw/` | downloaded originals, not tracked (MEROPS `Substrate_search.txt`, Eckhard xlsx, Yang PDF) |
| `models/` | one JSON per protease (arrays only; shared with the browser tool) |
| `references/` | archived reference tables and `PROVENANCE.md` |
| `results/` | cross-validation, methodology study, external validation, legacy audit |
| `scripts/` | methodology study, validation, source extraction |
| `web/` | browser tool: scoring engine (`engine.js`), template, build script, parity and browser tests |
| `report/` | technical report (LaTeX), regenerable figures and tables, the built interactive tool |

## Method in one paragraph

Positives are confirmed cleavage windows (MEROPS, Eckhard 2016 PICS, Yang 2015). Each protease is trained
one-vs-rest: negatives are half other proteases' positives (equal share per competitor) and half
random Swiss-Prot 8-mers, at 3:1. Features are the 64 VHSE8 descriptors; the model is L2 logistic
regression (C chosen by inner cluster-aware CV) alongside a PSSM with BLOSUM62 pseudocounts. Scores are
converted to percentiles of a background null, combined by the harmonic mean, and selectivity is the
margin between the target and its strongest competitor. All performance figures come from
similarity-clustered cross-validation (sequences within Hamming distance 2 never straddle train and test).
Scores are relative to the training mixture, not absolute cleavage probabilities.

## Scope

Trainable proteases: MMP-2, MMP-8, MMP-9, MMP-14, neutrophil elastase, LasB, and the faecal serine-protease
group. FAP is registered but has only two confirmed windows in MEROPS and cannot be trained.
