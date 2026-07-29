"""
Retrain two panels with sound methodology:
  Panel A: MMP-9 / MMP-8 / HNE  (3-plex)
  Panel B: MMP-9 / HNE           (2-plex)

Methodology:
  - Protease-vs-protease negatives (task = "which protease", not "is this cleavable")
  - Swiss-Prot background fills remainder to 3:1 ratio
  - LR + PSSM dual scorer; both must agree for substrate acceptance
  - 1000-iteration bootstrap 95% CI on 5-fold CV held-out predictions
  - Promiscuous site audit: 8-mers in both MMP-8 and MMP-9 positive sets flagged
"""

import sys, logging, warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
log = logging.getLogger(__name__)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.features import build_feature_matrix
from src.pssm import build_pssm_from_positives
from src.utils import SWISSPROT_BG, STANDARD_AAS

RANDOM_SEED = 42
CV_FOLDS    = 5
N_BOOTSTRAP = 1000

PROC_DIR   = BASE / 'data' / 'processed'
MODELS_DIR = BASE / 'models'
OUT_A      = BASE / 'data' / 'panel_a'   # MMP-9 / MMP-8 / HNE
OUT_B      = BASE / 'data' / 'panel_b'   # MMP-9 / HNE
for d in [OUT_A, OUT_B]:
    d.mkdir(exist_ok=True)

PANEL_A = ['MMP_9', 'MMP_8', 'NE']
PANEL_B = ['MMP_9', 'NE']


def load_positives(name: str) -> list[str]:
    f = PROC_DIR / f'training_set_{name}.csv'
    df = pd.read_csv(f)
    return df[df['label'] == 1]['sequence'].str.upper().tolist()


def background_negatives(n: int, seed: int = RANDOM_SEED) -> list[str]:
    aas   = list(STANDARD_AAS)
    probs = np.array([SWISSPROT_BG[aa] for aa in aas], dtype=float)
    probs /= probs.sum()
    rng   = np.random.default_rng(seed)
    seqs  = []
    while len(seqs) < n:
        idxs = rng.choice(len(aas), size=8, p=probs)
        seqs.append(''.join(aas[i] for i in idxs))
    return seqs


def build_training_set(target: str, t_pos: list[str],
                        other_pos: dict[str, list[str]],
                        seed: int = RANDOM_SEED) -> pd.DataFrame:
    n_pos = len(t_pos)
    n_neg = n_pos * 3
    t_set = set(t_pos)

    # Cross-protease negatives: other panel proteases' positives
    cross = []
    for prot, pos in other_pos.items():
        if prot == target:
            continue
        cross.extend([s for s in pos if s not in t_set])
    cross = list(dict.fromkeys(cross))  # dedup, preserve order

    # Background fills remainder (at least half of total negatives)
    n_bg = max(n_neg // 2, n_neg - len(cross))
    bg   = background_negatives(n_bg, seed=seed)

    all_neg = cross + bg
    rng = np.random.default_rng(seed)
    if len(all_neg) > n_neg:
        idx = rng.choice(len(all_neg), n_neg, replace=False)
        all_neg = [all_neg[i] for i in idx]

    rows = (
        [{'sequence': s, 'label': 1, 'source': 'positive'} for s in t_pos] +
        [{'sequence': s, 'label': 0,
          'source': 'cross_neg' if s in set(cross) else 'bg'}
         for s in all_neg]
    )
    return pd.DataFrame(rows).drop_duplicates(subset='sequence').reset_index(drop=True)


def make_lr():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                                      random_state=RANDOM_SEED, solver='lbfgs')),
    ])


def bootstrap_auroc(X: np.ndarray, y: np.ndarray,
                     n_iter: int = N_BOOTSTRAP, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    y_true_all, y_prob_all = [], []
    fold_scores = []
    for tr, te in cv.split(X, y):
        m = make_lr()
        m.fit(X[tr], y[tr])
        probs = m.predict_proba(X[te])[:, 1]
        y_true_all.extend(y[te])
        y_prob_all.extend(probs)
        fold_scores.append(roc_auc_score(y[te], probs))

    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    point = roc_auc_score(y_true_all, y_prob_all)

    boots = []
    n = len(y_true_all)
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        yt, yp = y_true_all[idx], y_prob_all[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(roc_auc_score(yt, yp))

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), fold_scores


def pssm_auroc(pssm, positives: list[str], negatives: list[str]) -> float:
    seqs = positives + negatives
    y    = np.array([1]*len(positives) + [0]*len(negatives))
    scores = pssm.score_batch(seqs)
    valid  = np.isfinite(scores)
    if valid.sum() < 10:
        return 0.5
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(roc_auc_score(y[valid], scores[valid]))


def train_panel(panel: list[str], out_dir: Path, tag: str):
    all_pos = {name: load_positives(name) for name in panel}

    # Promiscuous audit between MMP members
    mmp_members = [n for n in panel if n.startswith('MMP')]
    if len(mmp_members) >= 2:
        sets = {n: set(all_pos[n]) for n in mmp_members}
        from itertools import combinations
        for a, b in combinations(mmp_members, 2):
            shared = sets[a] & sets[b]
            log.info(f"  Promiscuous {a}∩{b}: {len(shared)} 8-mers "
                     f"({100*len(shared)/min(len(sets[a]),len(sets[b])):.1f}% of smaller set)")
        pd.DataFrame({'sequence': sorted(
            set.intersection(*sets.values()) if len(sets) > 1 else set()
        )}).to_csv(out_dir / 'promiscuous_mmp.csv', index=False)

    results = {}
    for target in panel:
        log.info(f"\n{'='*55}\n{tag} — {target}\n{'='*55}")
        t_pos = all_pos[target]
        other = {n: all_pos[n] for n in panel if n != target}

        df = build_training_set(target, t_pos, other)
        df.to_csv(out_dir / f'training_{target}.csv', index=False)
        log.info(f"  {(df.label==1).sum()} pos / {(df.label==0).sum()} neg  "
                 f"({(df.source=='cross_neg').sum()} cross-neg, "
                 f"{(df.source=='bg').sum()} bg)")

        X = build_feature_matrix(df['sequence'].tolist(), use_esm2=False)
        y = df['label'].values.astype(int)

        auroc, ci_lo, ci_hi, folds = bootstrap_auroc(X, y)
        log.info(f"  LR  AUROC: {auroc:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        log.info(f"  Folds: {[round(f,3) for f in folds]}")

        neg_seqs = df[df['label']==0]['sequence'].tolist()
        pssm = build_pssm_from_positives(t_pos, target)
        pa   = pssm_auroc(pssm, t_pos, neg_seqs[:len(t_pos)*3])
        log.info(f"  PSSM AUROC: {pa:.3f}")

        lr = make_lr()
        lr.fit(X, y)

        suffix = tag.lower().replace('-', '_')
        joblib.dump(lr,   MODELS_DIR / f'lr_{target}_{suffix}.pkl')
        joblib.dump(pssm, MODELS_DIR / f'pssm_{target}_{suffix}.pkl')

        results[target] = {
            'n_pos': len(t_pos), 'auroc': auroc,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'pssm_auroc': pa, 'folds': folds,
        }

    log.info(f"\n{'='*65}")
    log.info(f"{'Protease':<10} {'N_pos':>6} {'AUROC':>7} "
             f"{'95% CI':>16} {'PSSM':>7}")
    log.info('-' * 52)
    for t, r in results.items():
        log.info(f"{t:<10} {r['n_pos']:>6} {r['auroc']:>7.3f} "
                 f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] {r['pssm_auroc']:>7.3f}")

    pd.DataFrame(results).T.to_csv(out_dir / 'model_summary.csv')
    return results


def score_candidates(panel: list[str], tag: str,
                      candidates: dict[str, str]) -> pd.DataFrame:
    """Score synthesis candidates against all panel models."""
    suffix = tag.lower().replace('-', '_')
    models = {n: joblib.load(MODELS_DIR / f'lr_{n}_{suffix}.pkl') for n in panel}
    pssms  = {n: joblib.load(MODELS_DIR / f'pssm_{n}_{suffix}.pkl') for n in panel}

    rows = []
    for seq, note in candidates.items():
        X = build_feature_matrix([seq], use_esm2=False)
        row = {'sequence': seq, 'note': note}
        for n in panel:
            row[f'lr_{n}']   = float(models[n].predict_proba(X)[0, 1])
            row[f'pssm_{n}'] = float(pssms[n].score(seq))
        rows.append(row)
    df = pd.DataFrame(rows)
    log.info(f"\n=== Candidate scores ({tag}) ===")
    for _, r in df.iterrows():
        lr_str   = '  '.join(f'{n}={r[f"lr_{n}"]:.3f}'   for n in panel)
        pssm_str = '  '.join(f'{n}={r[f"pssm_{n}"]:.2f}' for n in panel)
        log.info(f"  {r.sequence}  LR: {lr_str}")
        log.info(f"  {'':12}  PS: {pssm_str}  [{r.note}]")
    return df


if __name__ == '__main__':
    log.info('\n' + '#'*60)
    log.info('Panel A: MMP-9 / MMP-8 / HNE')
    log.info('#'*60)
    res_a = train_panel(PANEL_A, OUT_A, 'panel-a')

    log.info('\n' + '#'*60)
    log.info('Panel B: MMP-9 / HNE')
    log.info('#'*60)
    res_b = train_panel(PANEL_B, OUT_B, 'panel-b')

    # Score synthesis candidates for each panel
    candidates_a = {
        'IPRTLTAA': 'MMP-9 candidate (Leu P1\', Kridel 2001)',
        'GPLGIFGQ': 'MMP-8 candidate (PLGI scaffold, Eckhard 2016)',
        'AAPVSGIG': 'HNE candidate (AAPV scaffold, Senior 2002)',
    }
    candidates_b = {
        'IPRTLTAA': 'MMP-9 candidate',
        'AAPVSGIG': 'HNE candidate',
    }

    score_candidates(PANEL_A, 'panel-a', candidates_a)
    score_candidates(PANEL_B, 'panel-b', candidates_b)
