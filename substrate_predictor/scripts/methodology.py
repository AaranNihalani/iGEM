"""Methodology study: baselines, representations, nested tuning, learning curves, negative-design ablation.

All comparisons use identical cluster-aware folds. Differences are tested with a paired cluster
bootstrap (same resampled clusters for both models).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from spred import data, evaluation, pssm as pssm_mod
from spred.constants import RANDOM_SEED, WINDOW
from spred.features import encode, index_matrix
from spred.model import fit_logistic, sigmoid
from spred.tables import AMINO_ACIDS

OUT = Path(__file__).resolve().parents[1] / "results" / "methodology"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 400


def onehot(seqs):
    idx = index_matrix(seqs)
    X = np.zeros((len(idx), WINDOW * 20))
    for p in range(WINDOW):
        X[np.arange(len(idx)), p * 20 + idx[:, p]] = 1
    return X


def composition(seqs):
    return np.array([[s.count(a) for a in AMINO_ACIDS] for s in seqs], dtype=float)


def lr_scores(Xtr, ytr, Xte, C=1.0):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def gbm_scores(Xtr, ytr, Xte):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                           class_weight="balanced", random_state=RANDOM_SEED, verbose=-1).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def paired_bootstrap(y, a, b, groups, n_boot=N_BOOT, seed=RANDOM_SEED):
    """Mean and 95% interval of AUROC(a) - AUROC(b), resampling clusters jointly."""
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable"); gs = groups[order]
    u, st = np.unique(gs, return_index=True); bd = np.append(st, len(gs))
    mem = [order[bd[i]:bd[i + 1]] for i in range(len(u))]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([mem[i] for i in rng.integers(0, len(mem), len(mem))])
        if 0 < y[idx].sum() < len(idx):
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def folds_for(frame, seed=RANDOM_SEED):
    seqs = frame.sequence.tolist(); y = frame.label.values
    g = evaluation.hamming_clusters(seqs)
    sp = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return g, list(sp.split(np.zeros(len(y)), y, g))


pos = data.trainable_positives()
comp_rows, tune_rows, curve_rows, abl_rows, diff_rows = [], [], [], [], []

for pid in pos:
    print(pid, flush=True)
    fr = data.training_frame(pid, pos, "panel")
    seqs = fr.sequence.tolist(); y = fr.label.values.astype(int)
    g, folds = folds_for(fr)
    Xv, Xo, Xc = encode(seqs), onehot(seqs), composition(seqs)

    oof = {k: np.zeros(len(y)) for k in ("prior", "composition-LR", "one-hot LR", "VHSE LR", "PSSM", "one-hot GBM", "VHSE GBM")}
    oof["prior"][:] = 0.5
    for tr, te in folds:
        oof["composition-LR"][te] = lr_scores(Xc[tr], y[tr], Xc[te])
        oof["one-hot LR"][te] = lr_scores(Xo[tr], y[tr], Xo[te])
        oof["VHSE LR"][te] = lr_scores(Xv[tr], y[tr], Xv[te])
        lo = pssm_mod.fit_log_odds([seqs[i] for i in tr if y[i] == 1])
        oof["PSSM"][te] = pssm_mod.score(lo, [seqs[i] for i in te])
        oof["one-hot GBM"][te] = gbm_scores(Xo[tr], y[tr], Xo[te])
        oof["VHSE GBM"][te] = gbm_scores(Xv[tr], y[tr], Xv[te])
    for k, s in oof.items():
        lo_, hi_ = evaluation.cluster_bootstrap_auroc(y, s, g, 200) if k != "prior" else (0.5, 0.5)
        comp_rows.append({"protease": pid, "model": k, "auroc": roc_auc_score(y, s), "lo": lo_, "hi": hi_})
    for k in oof:
        if k not in ("VHSE LR", "prior"):
            d, lo_, hi_ = paired_bootstrap(y, oof["VHSE LR"], oof[k], g)
            diff_rows.append({"protease": pid, "vs": k, "delta_auroc": d, "lo": lo_, "hi": hi_,
                              "significant": bool(lo_ > 0 or hi_ < 0)})

    # ---- nested selection of the regularisation strength C (inner 3-fold cluster CV on the training part)
    grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    nested = np.zeros(len(y)); chosen = []
    for tr, te in folds:
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_SEED).split(Xv[tr], y[tr], g[tr])
        inner = list(inner); best, bs = 1.0, -1
        for C in grid:
            sc_ = [roc_auc_score(y[tr][b], lr_scores(Xv[tr][a], y[tr][a], Xv[tr][b], C)) for a, b in inner]
            if np.mean(sc_) > bs: bs, best = np.mean(sc_), C
        chosen.append(best); nested[te] = lr_scores(Xv[tr], y[tr], Xv[te], best)
    tune_rows.append({"protease": pid, "auroc_fixed_C1": roc_auc_score(y, oof["VHSE LR"]), "auroc_nested": roc_auc_score(y, nested),
                      "chosen_C": str(chosen)})

    # ---- learning curves (fraction of each training fold used)
    for frac in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        sv, so = np.zeros(len(y)), np.zeros(len(y)); rng = np.random.default_rng(RANDOM_SEED)
        for tr, te in folds:
            sub = rng.choice(tr, max(int(len(tr) * frac), 30), replace=False)
            if y[sub].sum() < 3: sub = np.r_[sub, rng.choice(tr[y[tr] == 1], 5, replace=False)]
            sv[te] = lr_scores(Xv[sub], y[sub], Xv[te]); so[te] = lr_scores(Xo[sub], y[sub], Xo[te])
        curve_rows.append({"protease": pid, "fraction": frac, "n_train_pos": int(y[tr].sum() * frac),
                           "VHSE LR": roc_auc_score(y, sv), "one-hot LR": roc_auc_score(y, so)})

    # ---- negative-design ablation, evaluated on a common test set (panel frame test fold)
    sel = (fr.source.values != "background") & (y == 0)     # competitor positives (held-out selectivity test)
    for design in ("panel negatives", "background-only negatives", "shuffled+background negatives"):
        s_all = np.zeros(len(y))
        for tr, te in folds:
            trp = tr[y[tr] == 1]
            if design == "panel negatives":
                trn = tr[y[tr] == 0]
            elif design == "background-only negatives":
                trn = tr[(y[tr] == 0) & (fr.source.values[tr] == "background")]
            else:
                tp = [seqs[i] for i in trp]
                neg = data.make_negatives(pid, {pid: tp}, "baseline")
                Xn = encode(neg.sequence.tolist())
                s_all[te] = lr_scores(np.vstack([Xv[trp], Xn]), np.r_[np.ones(len(trp)), np.zeros(len(Xn))], Xv[te]); continue
            idx = np.r_[trp, trn]
            s_all[te] = lr_scores(Xv[idx], y[idx], Xv[te])
        m_bg = (y == 1) | (~sel)
        m_sel = (y == 1) | sel
        abl_rows.append({"protease": pid, "design": design, "auroc_vs_background": roc_auc_score(y[m_bg & ((y == 1) | (fr.source.values == "background"))], s_all[m_bg & ((y == 1) | (fr.source.values == "background"))]),
                         "auroc_vs_competitors": roc_auc_score(y[m_sel], s_all[m_sel]) if sel.sum() > 10 else np.nan})

pd.DataFrame(comp_rows).to_csv(OUT / "model_comparison.csv", index=False)
pd.DataFrame(diff_rows).to_csv(OUT / "paired_differences_vs_vhse_lr.csv", index=False)
pd.DataFrame(tune_rows).to_csv(OUT / "nested_regularisation.csv", index=False)
pd.DataFrame(curve_rows).to_csv(OUT / "learning_curves.csv", index=False)
pd.DataFrame(abl_rows).to_csv(OUT / "negative_design_ablation.csv", index=False)
print("done")
