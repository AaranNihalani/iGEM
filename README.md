# Eton iGEM 2026: Dry Lab

We are building a modular electrochemical biosensor that detects six disease-linked proteases from a single patient sample. A redox-labelled peptide substrate sits on a gold electrode; when the target protease cleaves it, the label leaves and the current drops. Square-wave voltammetry reads the signal. No antibodies, no PCR, no laboratory.

The six proteases span colorectal cancer, periodontitis, chronic wounds, COPD, and IBD — chosen because their activity in accessible biofluids (serum, saliva, wound fluid, stool) is clinically meaningful and underexploited as a diagnostic signal.

Dry lab drives two things: proving the sensor physics works on paper before anything goes near a bench, and giving wet lab a ranked shortlist of peptide substrates to synthesise rather than a guess.

---

## What lives here

| Folder | What it does |
|---|---|
| `cross_reactivity_matrix/` | PSSM-based specificity matrix: confirms the six substrate motifs do not cross-cleave within the panel |
| `substrate_predictor/` | ML tool that ranks candidate 8-mer substrates by predicted cleavage probability and selectivity; primary Best Software Tool candidate |

Earlier dry-lab deliverables (Bayesian landscape model, Hill equation dose-response, Marcus theory electron transfer model, clinical threshold map, sample volume calculations) are documented in `protease_panel_characterisation.pdf`.

---

## Team

Eton College iGEM 2026 — Dry Lab
Aaran Nihalani
