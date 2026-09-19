"""Train and evaluate every trainable protease; write models/ and results/."""
import json
from pathlib import Path

import pandas as pd

from . import data, evaluation
from .constants import MIN_TRAINABLE_POSITIVES, RANDOM_SEED
from .model import fit_model

RESULTS = Path(__file__).resolve().parents[1] / "results"


def train_all(seed: int = RANDOM_SEED, n_boot: int = 500, verbose: bool = True):
    RESULTS.mkdir(exist_ok=True)
    pos = data.trainable_positives()
    rows, comp_rows, calib_rows = [], [], []
    for pid, seqs in pos.items():
        frame = data.training_frame(pid, pos, mode="panel", seed=seed)
        cv = evaluation.cross_validate(frame, seed=seed)
        y, p, s, g = cv["y"], cv["p"], cv["s"], cv["groups"]
        lr = evaluation.metrics(y, p)
        lo, hi = evaluation.cluster_bootstrap_auroc(y, p, g, n_boot, seed)
        plo, phi = evaluation.cluster_bootstrap_auroc(y, s, g, n_boot, seed)
        from sklearn.metrics import roc_auc_score
        ps = float(roc_auc_score(y, s))
        cons = float(roc_auc_score(y, cv["u"]))
        clo, chi = evaluation.cluster_bootstrap_auroc(y, cv["u"], g, n_boot, seed)
        cal, ece = evaluation.calibration_table(y, p)
        comp = evaluation.competitor_auroc(frame, p, pid)
        row = {"protease": pid, "n_pos": len(seqs), "n_neg": int((y == 0).sum()),
               "n_clusters": int(len(set(g))), **{f"lr_{k}": v for k, v in lr.items()},
               "lr_auroc_lo": lo, "lr_auroc_hi": hi, "pssm_auroc": ps, "pssm_auroc_lo": plo,
               "pssm_auroc_hi": phi, "consensus_auroc": cons, "consensus_auroc_lo": clo,
               "consensus_auroc_hi": chi, "lr_ece": ece, "chosen_C": str(cv["C"])}
        rows.append(row)
        comp_rows += [{"target": pid, "competitor": c, "auroc": a} for c, a in comp.items()]
        calib_rows += [{"protease": pid, "mean_p": a, "frac_pos": b, "n": n} for a, b, n in cal]
        meta = {"n_pos": len(seqs), "n_neg": int((y == 0).sum()), "data_fingerprint": data.fingerprint(seqs),
                "cv": {k: row[k] for k in row if k.startswith(("lr_", "pssm_", "consensus_"))}, "seed": seed}
        fit_model(pid, frame, seqs, meta, groups=g).save()
        if verbose:
            print(f"{pid:6s} n={len(seqs):5d}  LR {lr['auroc']:.3f} [{lo:.3f},{hi:.3f}]  PSSM {ps:.3f}  consensus {cons:.3f} [{clo:.3f},{chi:.3f}]  C={cv['C']}")
    pd.DataFrame(rows).to_csv(RESULTS / "cv_summary.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(RESULTS / "cv_competitor_auroc.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(RESULTS / "cv_calibration.csv", index=False)
    return pd.DataFrame(rows)
