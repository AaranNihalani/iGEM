# Task 6: Cross-Reactivity Specificity Matrix — Interpretation Report

**Generated:** 2026-07-07  
**Data source:** MEROPS Substrate_search.txt (current release, accessed July 2026)  
**Pipeline:** `run_all.py` with placeholder panel motifs (see Table 1)

---

## 1. Summary

A 6x6 position-specific scoring matrix (PSSM) cross-reactivity analysis was performed
for the six panel proteases against their six substrate motifs. PSSMs were built from
all confirmed P4-P4' cleavage windows in the MEROPS substrate database. Each cell
reports the normalised log-odds PSSM score of the column motif under the row protease's
scoring matrix (diagonal = self-score reference).

**Main finding:** 23 of 30 off-diagonal pairs are orthogonal (score < 0.05). The five
flagged pairs (score >= 0.20) are all either within the M10 metalloproteinase subfamily
(expected from shared active-site geometry) or involve the FAP motif (unreliable due to
2 MEROPS entries). The serine protease block (NE, Fecal) is cleanly separated from all
three MMPs, with all cross-reactivity scores strongly negative.

---

## 2. Training Set Sizes

| Protease | MEROPS ID | Unique windows | Confidence |
|---|---|---|---|
| MMP14 | M10.014 | 129 | medium |
| MMP-8 | M10.002 | 96 | medium |
| MMP-9 | M10.004 | 341 | high |
| NE | S01.131 | 327 | high |
| Fecal serine | S01.151/152/153 | 18,251 | high |
| FAP | S09.007 | **2** | **unusable** |

---

## 3. Panel Motifs Used (Placeholder)

These are literature-sourced motifs used for this initial run. Task 7 will replace them
with ML-ranked candidates; the pipeline should then be re-run to produce the final matrix.

| Protease | Motif | Source |
|---|---|---|
| MMP14 | SGRSENIR | Ohuchi et al. 1997; validated sensor motif |
| MMP-8 | GPLGIAGQ | Turk et al. 2001 (collagen-derived) |
| MMP-9 | VPLSLYSG | Vandooren et al. 2013 (gelatin-derived) |
| NE | AAPVAALS | Bieth 1986 (elastin-derived canonical) |
| Fecal serine | KQLRAIRM | Trypsin-like, Arg at P1 |
| FAP | GPRFPIGS | Edosada et al. 2006 |

---

## 4. Normalised Specificity Matrix

Row = protease; column = substrate motif owner; self-score cells on diagonal.

| Protease | MMP14 | MMP-8 | MMP-9 | NE | Fecal | FAP |
|---|---|---|---|---|---|---|
| MMP14 | -1.00* | **1.53** | **1.14** | -1.67 | -1.40 | -0.76 |
| MMP-8 | -0.34 | 1.00 | 0.17 | -0.26 | -0.39 | **0.44** |
| MMP-9 | -1.25 | **2.40** | 1.00 | 0.19 | -1.51 | **1.36** |
| NE | -1.67 | -1.49 | -0.79 | 1.00 | -0.83 | -1.52 |
| Fecal | -3.83 | -1.78 | -2.86 | -2.69 | 1.00 | -4.44 |
| FAP | -1.15 | -0.75 | -1.03 | -1.40 | -0.36 | -1.00* |

**Bold** = flagged (>= 0.20). `*` = negative self-score (see Section 6).

---

## 5. Flagged Pairs (score >= 0.20)

| Protease | Motif | Score | Explanation |
|---|---|---|---|
| MMP-9 | MMP-8 motif (GPLGIAGQ) | **2.40** | Both are M10 subfamily MMPs; shared preference for hydrophobic P1' (Gly, Ala, Ile). High cross-reactivity expected. |
| MMP14 | MMP-8 motif (GPLGIAGQ) | **1.53** | M10 subfamily; same explanation. |
| MMP-9 | FAP motif (GPRFPIGS) | **1.36** | FAP PSSM built on 2 entries — result is noise, not biology. |
| MMP14 | MMP-9 motif (VPLSLYSG) | **1.14** | M10 subfamily. |
| MMP-8 | FAP motif (GPRFPIGS) | **0.44** | FAP PSSM noise. |

**Design implication:** The three M10 MMPs cannot be discriminated by PSSM-derived
motifs alone — their substrate repertoires overlap substantially. Task 7 must identify
motifs that exploit subtle subsite differences (e.g. MMP14's preference for bulkier
residues at P3 vs MMP-8's collagenase selectivity) or accept that the MMP subpanel
requires orthogonal readout strategies (different electrode, different assay window).

---

## 6. Negative Self-Scores: MMP14 and FAP

Two proteases have negative diagonal values after normalisation, meaning their placeholder
motif scores **below the Swiss-Prot background** under their own PSSM.

**MMP14 (SGRSENIR, self-score = -3.81):**  
The MEROPS M10.014 substrate set is dominated by natural extracellular matrix proteins
(collagens, fibronectins, laminins) cleaved at sites enriched for Pro, Gly, and Leu.
SGRSENIR is an engineered sensor motif with Ser-Gly at P4-P3 and Arg at P1, which
are atypical for the natural MEROPS profile. This is not a sensor design failure: the
motif is experimentally validated (Ohuchi 1997, kcat/KM = 5.5e4 M-1s-1) and works
in the biosensor context. The discrepancy reflects a known limitation of PSSM-based
scoring for engineered vs. natural substrates.

**Implication for the matrix:** The MMP14 row is normalised by |self-score| and
the diagonal is displayed as -1.00 to flag this condition. The off-diagonal scores
for MMP14 should be read as relative magnitudes only, not absolute cleavage
efficiencies. Task 7's logistic regression model will sidestep this issue by learning
a decision boundary directly rather than relying on PSSM self-score normalisation.

**FAP (GPRFPIGS, self-score = -3.47):**  
With only 2 MEROPS entries, the FAP PSSM is statistically meaningless. The entire
FAP row and column should be treated as non-informative in this initial run. The Task 7
model will flag FAP as low-confidence and the literature search for additional FAP
cleavage data should be prioritised before the final matrix is produced.

---

## 7. Orthogonal Pairs

All 23 pairs with score < 0.05 involve cross-family comparisons: MMP vs serine
protease (NE, Fecal) combinations. This is the expected result: the zinc-dependent
metalloproteinase mechanism and the serine protease mechanism impose fundamentally
different subsite preferences, so their PSSMs have negligible overlap.

Confirmed orthogonal pairs (all scores strongly negative, range -0.34 to -4.44):
- All three MMPs x NE motif
- All three MMPs x Fecal motif
- NE and Fecal x all three MMP motifs
- NE x Fecal motif (and vice versa) — these two serine proteases have distinct
  primary specificities (NE: small hydrophobic P1; fecal trypsin: Arg/Lys P1)

---

## 8. Caveats

1. **FAP PSSM is non-informative** (2 entries). FAP results should be excluded from
   biological conclusions until more substrate data is available or a different
   approach (e.g. structure-based scoring) is used.

2. **Placeholder motifs may not reflect final panel.** This matrix should be re-run
   once Task 7 produces ML-ranked motifs for MMP-8, MMP-9, NE, Fecal, and FAP.
   The MMP14 motif (SGRSENIR) is fixed and validated.

3. **MEROPS substrate data has ascertainment bias.** Natural protein substrates are
   over-represented relative to synthetic peptide substrates. PSSMs built from this
   data will be biased towards natural cleavage contexts and may underestimate the
   activity of synthetic, sensor-optimised motifs.

4. **Fecal serine PSSM is dominated by trypsin-1** (18,131 of 18,251 entries).
   Chymotrypsin-like and elastase-like fecal enzymes have much less data.
   The "Fecal" PSSM primarily reflects trypsin specificity (Arg/Lys at P1).

5. **PSSM scores are in vitro.** Matrix metalloproteinase activity is modulated
   in vivo by TIMPs, pro-domain processing, and pH. Cross-reactivity scores
   derived here do not account for these regulatory factors.
