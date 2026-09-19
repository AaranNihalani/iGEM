"""(1) How much of each position's specificity can the 8-dimensional VHSE representation express?
(2) Label-shuffle control on the final cross-validation pipeline."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from spred import data, evaluation
from spred.constants import POSITIONS
from spred.features import VHSE_MATRIX, encode
from spred.model import load_all, fit_logistic, sigmoid
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parents[1] / "results" / "methodology"
models = load_all()

# ---- (1) subspace coverage: project each position's PSSM log-odds (20-vector) onto span{VHSE columns, 1}
Phi = np.column_stack([VHSE_MATRIX, np.ones(20)])            # (20, 9)
Q, _ = np.linalg.qr(Phi)
rank = np.linalg.matrix_rank(Phi)
rows = []
for pid, m in models.items():
    for i, pos in enumerate(POSITIONS):
        v = m.log_odds[i] - m.log_odds[i].mean()
        resid = v - Q @ (Q.T @ v)
        rows.append({"protease": pid, "position": pos, "variance": float((v ** 2).sum()),
                     "r2_vhse_subspace": float(1 - (resid ** 2).sum() / max((v ** 2).sum(), 1e-12))})
df = pd.DataFrame(rows); df.to_csv(OUT / "vhse_subspace_coverage.csv", index=False)
agg = df.groupby("protease").apply(lambda d: 1 - ((1 - d.r2_vhse_subspace) * d.variance).sum() / d.variance.sum())
print("rank of [VHSE, 1] =", rank, "(of 20 residues)")
print("variance-weighted R2 of PSSM log-odds inside VHSE span:\n", agg.round(3).to_string())
pd.DataFrame({"protease": agg.index, "r2_weighted": agg.values}).to_csv(OUT / "vhse_subspace_coverage_summary.csv", index=False)

# ---- (2) label-shuffle control (permute labels, keep the cluster-aware fold structure)
pos = data.trainable_positives()
rows = []
for pid in pos:
    fr = data.training_frame(pid, pos)
    y = fr.label.values.astype(int); X = encode(fr.sequence.tolist())
    g = evaluation.hamming_clusters(fr.sequence.tolist())
    rng = np.random.default_rng(42); aucs = []
    for rep in range(5):
        yp = rng.permutation(y); p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, yp, g):
            mean, sc, coef, b = fit_logistic(X[tr], yp[tr], 0.1)
            p[te] = sigmoid(b + ((X[te] - mean) / sc) @ coef)
        aucs.append(roc_auc_score(yp, p))
    rows.append({"protease": pid, "auroc_label_shuffled_mean": float(np.mean(aucs)), "auroc_label_shuffled_sd": float(np.std(aucs))})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "label_shuffle_control.csv", index=False)
