"""
Score the Task 8 substrate candidates through the MMP-2 / MMP-9 / NE panel of the substrate predictor and
rank each candidate against the PSSM top-k enumeration pool of its target protease.

Selectivity is the background-percentile specificity margin defined in substrate_predictor/spred/panel.py.

Stage 1 candidates (order for wet-lab):
  Candidate 1 - MMP-9-biased:  PRSLSGLK  (Kridel 2001 phage-display consensus, P3 Pro / P2 Arg / P1' Leu)
  Candidate 2 - MMP-2-leaning: PLGLAGQS  (PLGLAG core)
  Candidate 4 - NE baseline:   AAPVSGIG   (AAPV core)
Stage 2 (contingency, computational only):
  Candidate 3 - MMP-2 alternative: PLGVALAA
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spred import candidates as spred_candidates
from spred.panel import PRESETS, Panel

log = logging.getLogger(__name__)

TRIPLEX = list(PRESETS["mmp_triplex"])          # ['MMP2', 'MMP9', 'NE']
TARGET_ID = {"MMP-2": "MMP2", "MMP-9": "MMP9", "NE": "NE"}

CANDIDATES = {
    "Cand1_MMP9":  {"seq": "PRSLSGLK", "target": "MMP-9", "stage": 1,
                    "source": "Kridel 2001 phage-display consensus (P3 Pro, P2 Arg, P1' Leu)"},
    "Cand2_MMP2":  {"seq": "PLGLAGQS", "target": "MMP-2", "stage": 1, "source": "PLGLAG core"},
    "Cand4_NE":    {"seq": "AAPVSGIG", "target": "NE", "stage": 1, "source": "AAPV core"},
    "Cand3_MMP2b": {"seq": "PLGVALAA", "target": "MMP-2", "stage": 2, "source": "PLGLAG variant"},
}


def run(output_dir: Path, top_k: int = 3) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = Panel(TRIPLEX)
    rows = []
    for name, info in CANDIDATES.items():
        seq, target = info["seq"], TARGET_ID[info["target"]]
        s = panel.score([seq])
        margin = float(panel.margin(s, target)[0])
        pool = spred_candidates.pssm_top_k(panel.models[target].log_odds, top_k)
        ps = panel.score(pool)
        pool_margin = panel.margin(ps, target)
        rows.append({"name": name, "sequence": seq, "target": info["target"], "stage": info["stage"],
                     **{f"P_{m}": round(float(s[f"P_{m}"][0]), 4) for m in TRIPLEX},
                     **{f"u_{m}": round(float(s[f"u_{m}"][0]), 4) for m in TRIPLEX},
                     "margin": round(margin, 4),
                     "margin_percentile_in_pssm_pool": round(float(np.mean(pool_margin < margin) * 100), 1),
                     "pool_size": len(pool), "source": info["source"]})
        log.info(f"  {name} {seq}: margin={margin:+.3f}  pool percentile={rows[-1]['margin_percentile_in_pssm_pool']}")
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "candidate_scores.csv", index=False)
    return df
