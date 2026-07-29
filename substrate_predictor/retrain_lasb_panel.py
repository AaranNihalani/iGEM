"""
Retrain the LasB / MMP-9 / HNE panel with corrected LasB training data.

LasB positives now include elastin-derived Ala-Ala cleavage sites (Galloway 1991)
and validated peptide substrates (Toth 1994), bringing n from 35 to 141.

The model with n=35 scored GPAAGPAA (elastin Ala-Ala) at LasB=0.003, showing it
had learned the wrong signal (hydrophobic P1' from non-elastin substrates).
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
OUT_DIR    = BASE / 'data' / 'panel_lasb'
OUT_DIR.mkdir(exist_ok=True)

PANEL = {
    'LasB': {'label': 'LasB (P. aeruginosa)'},
    'MMP_9': {'label': 'MMP-9'},
    'NE':    {'label': 'HNE'},
}


def load_positives(name: str) -> list[str]:
    # LasB gets the augmented v3 set; others use standard processed set
    if name == 'LasB':
        f = PROC_DIR / 'training_set_LasB_v3.csv'
    else:
        f = PROC_DIR / f'training_set_{name}.csv'
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")
    df = pd.read_csv(f)
    return df[df['label'] == 1]['sequence'].str.upper().tolist()


def background_negatives(n: int, seed: int = RANDOM_SEED) -> list[str]:
    aas   = list(STANDARD_AAS)
    probs = np.array([SWISSPROT_BG[aa] for aa in aas], dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    seqs = []
    while len(seqs) < n:
        idxs = rng.choice(len(aas), size=8, p=probs)
        seqs.append(''.join(aas[i] for i in idxs))
    return seqs


def build_training_set(target: str, target_pos: list[str],
                        other_pos: dict[str, list[str]],
                        seed: int = RANDOM_SEED) -> pd.DataFrame:
    n_pos = len(target_pos)
    n_neg_total = n_pos * 3
    target_set = set(target_pos)

    explicit_neg = []
    for prot, pos_list in other_pos.items():
        if prot == target:
            continue
        explicit_neg.extend([s for s in pos_list if s not in target_set])
    explicit_neg = list(dict.fromkeys(explicit_neg))

    # Background fills remaining ratio (at least half of negatives)
    n_bg = max(n_neg_total // 2, n_neg_total - len(explicit_neg))
    bg_neg = background_negatives(n_bg, seed=seed)

    all_neg = explicit_neg + bg_neg
    rng = np.random.default_rng(seed)
    if len(all_neg) > n_neg_total:
        idx = rng.choice(len(all_neg), n_neg_total, replace=False)
        all_neg = [all_neg[i] for i in idx]

    rows = (
        [{'sequence': s, 'label': 1, 'source': 'positive'} for s in target_pos] +
        [{'sequence': s, 'label': 0,
          'source': 'cross_neg' if s in set(explicit_neg) else 'bg'}
         for s in all_neg]
    )
    return pd.DataFrame(rows).drop_duplicates(subset='sequence').reset_index(drop=True)


def make_lr():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                                   random_state=RANDOM_SEED, solver='lbfgs')),
    ])


def bootstrap_auroc(X: np.ndarray, y: np.ndarray,
                     n_iter: int = N_BOOTSTRAP, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    y_true_all, y_prob_all = [], []
    for tr, te in cv.split(X, y):
        m = make_lr(); m.fit(X[tr], y[tr])
        y_true_all.extend(y[te])
        y_prob_all.extend(m.predict_proba(X[te])[:, 1])
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
    return point, float(lo), float(hi), [round(f,3) for f in
        [roc_auc_score(y[te], make_lr().fit(X[tr], y[tr]).predict_proba(X[te])[:,1])
         for tr, te in StratifiedKFold(CV_FOLDS, shuffle=True,
                                       random_state=seed).split(X, y)]]


def main():
    all_pos = {name: load_positives(name) for name in PANEL}
    for name, pos in all_pos.items():
        log.info(f"{name}: {len(pos)} positives")

    results = {}

    for target in PANEL:
        log.info(f"\n{'='*60}\nTraining {target}\n{'='*60}")
        t_pos = all_pos[target]
        other = {n: all_pos[n] for n in PANEL if n != target}

        df = build_training_set(target, t_pos, other)
        df.to_csv(OUT_DIR / f'training_set_{target}.csv', index=False)
        log.info(f"  {(df.label==1).sum()} pos / {(df.label==0).sum()} neg  "
                 f"({(df.source=='cross_neg').sum()} cross-neg)")

        X = build_feature_matrix(df['sequence'].tolist(), use_esm2=False)
        y = df['label'].values.astype(int)

        auroc, ci_lo, ci_hi, folds = bootstrap_auroc(X, y)
        log.info(f"  AUROC: {auroc:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        log.info(f"  Folds: {folds}")

        # PSSM
        neg_seqs = df[df['label']==0]['sequence'].tolist()
        pssm = build_pssm_from_positives(t_pos, target)
        from src.pssm import PSSM
        pos_scores = pssm.score_batch(t_pos)
        neg_scores = pssm.score_batch(neg_seqs[:len(t_pos)*3])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            all_s = np.concatenate([pos_scores, neg_scores])
            all_y = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
            valid = np.isfinite(all_s)
            pssm_auroc = roc_auc_score(all_y[valid], all_s[valid]) if valid.sum() > 10 else 0.5
        log.info(f"  PSSM AUROC: {pssm_auroc:.3f}")

        # Fit final model on full data
        lr = make_lr()
        lr.fit(X, y)
        joblib.dump(lr, MODELS_DIR / f'lr_{target}_v3.pkl')
        joblib.dump(pssm, MODELS_DIR / f'pssm_{target}_v3.pkl')

        results[target] = {
            'n_pos': len(t_pos), 'auroc': auroc,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'pssm_auroc': pssm_auroc, 'folds': folds,
        }

    log.info(f"\n{'='*65}")
    log.info(f"{'Protease':<10} {'N_pos':>6} {'AUROC':>7} {'95% CI':>16} {'PSSM':>7}")
    log.info('-'*50)
    for t, r in results.items():
        log.info(f"{t:<10} {r['n_pos']:>6} {r['auroc']:>7.3f} "
                 f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] {r['pssm_auroc']:>7.3f}")

    pd.DataFrame(results).T.to_csv(OUT_DIR / 'model_summary.csv')
    return results


if __name__ == '__main__':
    main()
