"""
Audit of the substrate-predictor logic. Every suspected defect is tested with code.
Writes results/audit/*.csv and a printed summary. Nothing in models/ is modified.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

OUT = BASE / 'results' / 'audit'; OUT.mkdir(parents=True, exist_ok=True)
SETS = ['MMP14', 'MMP_2', 'MMP_8', 'MMP_9', 'NE', 'LasB']   # Fecal (20k) omitted from the audit for speed

def auroc(y, p): return float(roc_auc_score(y, p))

# ---------------------------------------------------------------- A. effect of the VHSE table bug
# ---------------------------------------------------------------- B. random vs similarity-grouped CV
rows = []
for name in SETS:
    df = load_set(name); y = df.label.values.astype(int)
    seqs = df.sequence.tolist()
    Xn, Xo = encode(seqs, VHSE_NEW), encode(seqs, VHSE_OLD)
    g = assign_groups(df)
    r = {'set': name, 'n_pos': int(y.sum()), 'n_pos_clusters': int(len(np.unique(g[y == 1])))}
    for tag, X in [('new', Xn), ('old', Xo)]:
        r[f'auroc_random_{tag}'] = auroc(y, oof_lr(X, y, rand_cv()))
        r[f'auroc_grouped_{tag}'] = auroc(y, oof_lr(X, y, GroupKFold(5), g))
    rows.append(r)
    print(r, flush=True)
A = pd.DataFrame(rows); A.to_csv(OUT / 'A_B_vhse_fix_and_grouped_cv.csv', index=False)

# ---------------------------------------------------------------- C. PSSM in-sample vs out-of-fold
rows = []
for name in SETS:
    df = load_set(name); y = df.label.values.astype(int); seqs = np.array(df.sequence.tolist())
    pos = seqs[y == 1].tolist()
    ps = build_pssm_from_positives(pos, name)
    ins = auroc(y, ps.score_batch(seqs.tolist()))
    oof = np.zeros(len(y))
    for tr, te in rand_cv().split(seqs, y):
        m = build_pssm_from_positives(seqs[tr][y[tr] == 1].tolist(), name)
        oof[te] = m.score_batch(seqs[te].tolist())
    g = assign_groups(df); oofg = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(seqs, y, g):
        m = build_pssm_from_positives(seqs[tr][y[tr] == 1].tolist(), name)
        oofg[te] = m.score_batch(seqs[te].tolist())
    rows.append({'set': name, 'pssm_in_sample': ins, 'pssm_oof_random': auroc(y, oof), 'pssm_oof_grouped': auroc(y, oofg)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / 'C_pssm_insample_vs_oof.csv', index=False)

# ---------------------------------------------------------------- D. controls: label shuffle, composition-only baseline
rows = []
rng = np.random.default_rng(SEED)
for name in SETS:
    df = load_set(name); y = df.label.values.astype(int); seqs = df.sequence.tolist()
    X = encode(seqs, VHSE_NEW)
    comp = np.array([[s.count(a) for a in STANDARD_AAS] for s in seqs], dtype=float)
    yperm = rng.permutation(y)
    is_sh = (df.source.values == 'shuffled')
    p_full = oof_lr(X, y, rand_cv()); p_comp = oof_lr(comp, y, rand_cv())
    # composition-only model can only separate composition-different negatives:
    mask_bg = ~is_sh
    r = {'set': name,
         'auroc_full': auroc(y, p_full),
         'auroc_label_shuffled': auroc(yperm, oof_lr(X, yperm, rand_cv())),
         'auroc_composition_only': auroc(y, p_comp),
         'auroc_full_vs_shuffled_neg_only': auroc(y[(y == 1) | is_sh], p_full[(y == 1) | is_sh]) if is_sh.any() else np.nan,
         'auroc_full_vs_random_bg_only': auroc(y[(y == 1) | mask_bg], p_full[(y == 1) | mask_bg]),
         'auroc_comp_vs_shuffled_neg_only': auroc(y[(y == 1) | is_sh], p_comp[(y == 1) | is_sh]) if is_sh.any() else np.nan}
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(OUT / 'D_controls.csv', index=False)

# ---------------------------------------------------------------- E. cross-model probability scale (selectivity index validity)
models = {}
for name in ['MMP14', 'MMP_2', 'MMP_8', 'MMP_9', 'NE', 'Fecal', 'FAP', 'LasB']:
    df = load_set(name); models[name] = make_lr().fit(encode(df.sequence.tolist(), VHSE_NEW), df.label.values.astype(int))
    print('trained', name, int(df.label.sum()), flush=True)
rs = np.random.default_rng(SEED)
aas = STANDARD_AAS; pr = np.array([SWISSPROT_BG[a] for a in aas])
bg = [''.join(aas[i] for i in row) for row in rs.choice(20, size=(20000, 8), p=pr / pr.sum())]
Xb = encode(bg, VHSE_NEW)
P = pd.DataFrame({k: m.predict_proba(Xb)[:, 1] for k, m in models.items()})
scale = pd.DataFrame({'mean_p_on_random_8mers': P.mean(), 'frac_p_gt_0.5': (P > 0.5).mean(), 'p95': P.quantile(0.95)})
scale['n_pos'] = [int(load_set(k).label.sum()) for k in scale.index]
scale.to_csv(OUT / 'E_probability_scale_by_model.csv'); print(scale.round(3))
# how often does each protease act as the max competitor?
tgt = 'MMP_9'; comp_cols = [c for c in P.columns if c != tgt]
maxc = P[comp_cols].idxmax(axis=1).value_counts(normalize=True)
maxc.to_csv(OUT / 'E_max_competitor_share_for_MMP9.csv'); print(maxc.round(3))
# FAP influence: selectivity when FAP included vs excluded, for the top-1% MMP_9 candidates
top = P[tgt] >= P[tgt].quantile(0.99)
S_incl = P.loc[top, tgt] / P.loc[top, comp_cols].max(axis=1)
S_excl = P.loc[top, tgt] / P.loc[top, [c for c in comp_cols if c != 'FAP']].max(axis=1)
pd.DataFrame({'median_S_with_FAP': [S_incl.median()], 'median_S_without_FAP': [S_excl.median()],
              'frac_top1pct_where_FAP_is_max_competitor': [float((P.loc[top, comp_cols].idxmax(axis=1) == 'FAP').mean())]}
             ).to_csv(OUT / 'E_fap_influence.csv', index=False)
print(pd.read_csv(OUT / 'E_fap_influence.csv').round(3))

# ---------------------------------------------------------------- F. reproduce the "grouped CV = 0.5" defect from retrain_panel_v2
sys.path.insert(0, str(BASE))
import retrain_panel_v2 as v2
df = pd.read_csv(BASE / 'data' / 'panel_v2' / 'training_set_v2_MMP_9.csv'); y = df.label.values.astype(int)
grp = np.array(['bg' if s == 'bg' else ('cross' if s == 'cross_neg' else 'pos') for s in df.source])
val, folds = v2.grouped_cv_auroc(encode(df.sequence.tolist(), VHSE_NEW), y, grp)
print('retrain_panel_v2.grouped_cv_auroc ->', val, folds)
json.dump({'grouped_cv_auroc_returned': val, 'fold_scores': folds, 'groups': sorted(set(grp))}, open(OUT / 'F_grouped_cv_placeholder.json', 'w'))
print('AUDIT DONE')
