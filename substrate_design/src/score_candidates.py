"""
Score the Task 8 substrate candidates through the MMP-2 / MMP-9 / NE panel of the substrate predictor and
rank each candidate against the PSSM top-k enumeration pool of its target protease.

Selectivity is the background-percentile specificity margin defined in substrate_predictor/spred/panel.py.

Alignment. Windows are P4-P4' with the scissile bond after the fourth residue. Candidates 1-3 are motif cores whose
bond lies after the THIRD residue (Pro-Arg-Ser|Leu, Pro-Leu-Gly|Leu: P3 = Pro, P1' = Leu, matching the Kridel 2001
consensus Pro-Arg-Ser/Thr|Hy-Ser/Thr), so read as an 8-mer P4-P4' they would be shifted by one residue. Each candidate
therefore carries `bond_after` and is scored as a gapped window (unknown P4 for candidates 1-3, dropped P5' residue).

Stage 1 candidates (order for wet-lab):
  Candidate 1 - MMP-9-biased:  PRS|LSGLK  (Kridel 2001 phage-display consensus, P3 Pro / P2 Arg / P1' Leu)
  Candidate 2 - MMP-2-leaning: PLG|LAGQS  (PLGLAG core)
  Candidate 4 - NE baseline:   AAPV|SGIG  (AAPV core)
Stage 2 (contingency, computational only):
  Candidate 3 - MMP-2 alternative: PLG|VALAA
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spred import candidates as spred_candidates
from spred.validation import gapped_window
from spred.panel import PRESETS, Panel

log = logging.getLogger(__name__)

TRIPLEX = list(PRESETS["mmp_triplex"])          # ['MMP2', 'MMP9', 'NE']
TARGET_ID = {"MMP-2": "MMP2", "MMP-9": "MMP9", "NE": "NE"}

CANDIDATES = {
    "Cand1_MMP9":  {"seq": "PRSLSGLK", "bond_after": 3, "target": "MMP-9", "stage": 1,
                    "source": "Kridel 2001 phage-display consensus (P3 Pro, P2 Arg, P1' Leu)"},
    "Cand2_MMP2":  {"seq": "PLGLAGQS", "bond_after": 3, "target": "MMP-2", "stage": 1, "source": "PLGLAG core"},
    "Cand4_NE":    {"seq": "AAPVSGIG", "bond_after": 4, "target": "NE", "stage": 1, "source": "AAPV core"},
    "Cand3_MMP2b": {"seq": "PLGVALAA", "bond_after": 3, "target": "MMP-2", "stage": 2, "source": "PLGLAG variant"},
}


def run(output_dir: Path, top_k: int = 3) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = Panel(TRIPLEX)
    rows = []
    for name, info in CANDIDATES.items():
        seq, target = info["seq"], TARGET_ID[info["target"]]
        win = gapped_window(seq, info["bond_after"])
        s = panel.score_gapped([win])
        margin = float(panel.margin(s, target)[0])
        miss = [i for i, c in enumerate(win) if c == "-"]
        pool = ["".join("A" if i in miss else c for i, c in enumerate(w))
                for w in spred_candidates.pssm_top_k(panel.models[target].log_odds, top_k)]
        pool = [("".join("-" if i in miss else c for i, c in enumerate(w))) for w in pool]
        pool = list(dict.fromkeys(pool))
        ps = panel.score_gapped(pool)
        pool_margin = panel.margin(ps, target)
        rows.append({"name": name, "sequence": seq, "window": win, "target": info["target"], "stage": info["stage"],
                     **{f"P_{m}": round(float(s[f"P_{m}"][0]), 4) for m in TRIPLEX},
                     **{f"u_{m}": round(float(s[f"u_{m}"][0]), 4) for m in TRIPLEX},
                     "margin": round(margin, 4),
                     "margin_percentile_in_pssm_pool": round(float(np.mean(pool_margin < margin) * 100), 1),
                     "pool_size": len(pool), "source": info["source"]})
        log.info(f"  {name} {win}: margin={margin:+.3f}  pool percentile={rows[-1]['margin_percentile_in_pssm_pool']}")
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "candidate_scores.csv", index=False)
    return df
