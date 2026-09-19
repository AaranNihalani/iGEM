"""Random versus cluster-aware cross-validation for the final pipeline, plus cluster statistics."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from spred import data, evaluation
from spred.features import encode
from spred.model import fit_logistic, sigmoid

OUT = Path(__file__).resolve().parents[1] / "results" / "methodology"
pos = data.trainable_positives()
rows = []
for pid in pos:
    fr = data.training_frame(pid, pos)
    y = fr.label.values; X = encode(fr.sequence.tolist())
    g = evaluation.hamming_clusters(fr.sequence.tolist())
    sizes = pd.Series(g[y == 1]).value_counts()
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m, s, c, b = fit_logistic(X[tr], y[tr], 0.1)
        p[te] = sigmoid(b + ((X[te] - m) / s) @ c)
    grouped = evaluation.cross_validate(fr, tune=False)   # C fixed at 1 for a like-for-like comparison below
    p2 = np.zeros(len(y))
    for tr, te in [(np.where(grouped['fold'] != k)[0], np.where(grouped['fold'] == k)[0]) for k in range(5)]:
        m, s, c, b = fit_logistic(X[tr], y[tr], 0.1)
        p2[te] = sigmoid(b + ((X[te] - m) / s) @ c)
    rows.append({"protease": pid, "n_pos": int(y.sum()), "n_pos_clusters": int(len(sizes)), "largest_cluster": int(sizes.max()),
                 "frac_pos_in_multi_member_cluster": float((sizes[sizes > 1].sum()) / sizes.sum()),
                 "auroc_random_cv": roc_auc_score(y, p), "auroc_cluster_cv": roc_auc_score(y, p2)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "random_vs_cluster_cv.csv", index=False)
