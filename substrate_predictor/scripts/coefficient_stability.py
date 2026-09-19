"""Cluster-bootstrap stability of the standardised logistic-regression coefficients (position x VHSE descriptor)."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from spred import data, evaluation
from spred.features import encode
from spred.model import ProteaseModel, fit_logistic

OUT = Path(__file__).resolve().parents[1] / "results" / "coefficients"; OUT.mkdir(parents=True, exist_ok=True)
pos = data.trainable_positives()
for pid in pos:
    model = ProteaseModel.load(pid)
    C = model.meta["C"]
    fr = data.training_frame(pid, pos)
    y = fr.label.values.astype(int); X = encode(fr.sequence.tolist())
    g = evaluation.hamming_clusters(fr.sequence.tolist())
    order = np.argsort(g, kind="stable"); gs = g[order]
    u, st = np.unique(gs, return_index=True); bd = np.append(st, len(gs))
    mem = [order[bd[i]:bd[i + 1]] for i in range(len(u))]
    rng = np.random.default_rng(42)
    B = 60 if len(y) > 10000 else 200
    coefs = []
    for _ in range(B):
        idx = np.concatenate([mem[i] for i in rng.integers(0, len(mem), len(mem))])
        if 0 < y[idx].sum() < len(idx):
            coefs.append(fit_logistic(X[idx], y[idx], C)[2])
    coefs = np.array(coefs)
    point = model.coef
    lo, hi = np.percentile(coefs, 2.5, axis=0), np.percentile(coefs, 97.5, axis=0)
    sign_frac = (np.sign(coefs) == np.sign(point)).mean(axis=0)
    np.savez(OUT / f"{pid}.npz", coef=point, lo=lo, hi=hi, sign_frac=sign_frac, boot=coefs)
    print(pid, "C", C, "| coefficients with 95% CI excluding 0:", int(((lo > 0) | (hi < 0)).sum()), "/ 64", flush=True)
