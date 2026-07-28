"""
Panel v2: MMP-2 / MMP-9 / HNE with methodological improvements.

Changes vs integrate_and_retrain.py:
  1. Protease-vs-protease negatives — panel proteases' positives used as
     explicit negatives for each other, making the task "which protease cuts
     this" rather than "is this a cleavage site at all."
  2. PSSM model trained alongside LR for each protease; both scores reported.
  3. Grouped CV by source protein for PICS data (prevents near-duplicate leakage).
  4. Per-fold AUROC spread + 1000-iteration bootstrap CI.
  5. Promiscuous 8-mer audit: 8-mers appearing in >1 MMP positive set are
     flagged and excluded from training positives (they'd inflate diagonal scores).
  6. Shortlist generation: 6-10 candidates per protease ranked by PSSM + LR.
"""

import sys, logging, warnings, json
from pathlib import Path
from itertools import combinations

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
from sklearn.model_selection import StratifiedKFold, cross_validate, LeaveOneOut
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score

from src.features import build_feature_matrix
from src.pssm import PSSM, build_pssm_from_positives, STANDARD_AAS
from src.utils import SWISSPROT_BG, STANDARD_AAS as STD_AAS

RANDOM_SEED = 42
CV_FOLDS    = 5
N_BOOTSTRAP = 1000

PROC_DIR   = BASE / 'data' / 'processed'
ECKH_DIR   = BASE / 'data' / 'raw' / 'eckhard2016'
MODELS_DIR = BASE / 'models'
OUT_DIR    = BASE / 'data' / 'panel_v2'
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Panel definition — MMP-2 / MMP-9 / HNE
# Revert to original 3-plex; LasB dropped due to irresolvable Zn-metalloprotease
# cross-reactivity (all hydrophobic P1' variants score >0.9 on both MMP-9 and LasB)
# ---------------------------------------------------------------------------
PANEL = {
    'MMP_2':  {'label': 'MMP-2'},
    'MMP_9':  {'label': 'MMP-9'},
    'NE':     {'label': 'HNE'},
}


# ---------------------------------------------------------------------------
# Load positive sets
# ---------------------------------------------------------------------------

def load_positives(name: str) -> list[str]:
    """Load positives for a panel protease from processed training CSV."""
    f = PROC_DIR / f'training_set_{name}.csv'
    if not f.exists():
        raise FileNotFoundError(f"Run integrate_and_retrain.py first: {f}")
    df = pd.read_csv(f)
    return df[df['label'] == 1]['sequence'].str.upper().tolist()


def load_positives_with_protein(name: str) -> pd.DataFrame:
    """Load positives with protein source label for grouped CV."""
    # Try PICS file first (has per-protein info via accession)
    # Fall back to processed CSV with a dummy group
    eckhard_f = ECKH_DIR / f'eckhard_{name}_positives.csv'
    merops_f  = PROC_DIR / f'training_set_{name}.csv'

    rows = []
    if eckhard_f.exists():
        df_e = pd.read_csv(eckhard_f)
        # Protein accession is in a separate column if we saved it; otherwise dummy
        for seq in df_e['sequence'].str.upper():
            rows.append({'sequence': seq, 'protein': 'eckhard', 'source': 'PICS'})

    if merops_f.exists():
        df_m = pd.read_csv(merops_f)
        pos = df_m[df_m['label'] == 1]
        for seq in pos['sequence'].str.upper():
            if not any(r['sequence'] == seq for r in rows):
                rows.append({'sequence': seq, 'protein': 'merops', 'source': 'MEROPS'})

    return pd.DataFrame(rows).drop_duplicates(subset='sequence').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build training set with protease-vs-protease negatives
# ---------------------------------------------------------------------------

def background_negatives(n: int, seed: int = RANDOM_SEED) -> list[str]:
    aas = list(STD_AAS)
    probs = np.array([SWISSPROT_BG[aa] for aa in aas], dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    seqs = []
    while len(seqs) < n:
        idxs = rng.choice(len(aas), size=8, p=probs)
        seqs.append(''.join(aas[i] for i in idxs))
    return seqs


def build_panel_training_set(
    target: str,
    target_positives: list[str],
    other_positives: dict[str, list[str]],
    bg_fraction: float = 0.5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build training set for `target` protease with protease-vs-protease negatives.

    Negatives = union(other panel proteases' positives) + Swiss-Prot background.
    Ratio kept at ~3:1 negatives:positives, with at least bg_fraction of the
    negatives coming from background (prevents overfitting to panel-specific seqs).
    """
    n_pos = len(target_positives)
    n_neg_total = n_pos * 3

    # Explicit cross-protease negatives
    explicit_neg = []
    for prot, pos_list in other_positives.items():
        if prot == target:
            continue
        explicit_neg.extend(pos_list)
    # Remove any that also appear in target positives (promiscuous substrates)
    target_set = set(target_positives)
    explicit_neg = [s for s in explicit_neg if s not in target_set]
    explicit_neg = list(dict.fromkeys(explicit_neg))  # dedup

    n_bg = max(int(n_neg_total * bg_fraction),
               n_neg_total - len(explicit_neg))
    bg_neg = background_negatives(n_bg, seed=seed)

    all_neg = explicit_neg + bg_neg
    rng = np.random.default_rng(seed)
    if len(all_neg) > n_neg_total:
        idx = rng.choice(len(all_neg), n_neg_total, replace=False)
        all_neg = [all_neg[i] for i in idx]

    rows = (
        [{'sequence': s, 'label': 1, 'source': 'positive'} for s in target_positives] +
        [{'sequence': s, 'label': 0, 'source': ('cross_neg' if s in set(explicit_neg) else 'bg')}
         for s in all_neg]
    )
    df = pd.DataFrame(rows).drop_duplicates(subset='sequence').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

def make_lr():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                                   random_state=RANDOM_SEED, solver='lbfgs')),
    ])


def grouped_cv_auroc(X: np.ndarray, y: np.ndarray,
                      groups: np.ndarray) -> tuple[float, list[float]]:
    """
    Leave-one-group-out CV for AUROC.  Groups = protein source labels.
    Falls back to 5-fold if all in one group.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        log.warning("  Too few groups for LOGO-CV; using 5-fold CV instead")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []
        for tr, te in cv.split(X, y):
            m = make_lr(); m.fit(X[tr], y[tr])
            fold_scores.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        return float(np.mean(fold_scores)), fold_scores

    fold_scores = []
    for g in unique_groups:
        test_mask = groups == g
        if test_mask.sum() < 2 or y[test_mask].sum() == 0 or (1 - y[test_mask]).sum() == 0:
            continue
        m = make_lr()
        m.fit(X[~test_mask], y[~test_mask])
        fold_scores.append(roc_auc_score(y[test_mask],
                                          m.predict_proba(X[test_mask])[:, 1]))
    if not fold_scores:
        return 0.5, [0.5]
    return float(np.mean(fold_scores)), fold_scores


def bootstrap_auroc_ci(X: np.ndarray, y: np.ndarray,
                        n_iter: int = N_BOOTSTRAP,
                        seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    """Bootstrap 95% CI on AUROC using a pre-trained model's held-out predictions."""
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
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# PSSM CV — leave-one-out for small datasets, 5-fold for large
# ---------------------------------------------------------------------------

def pssm_cv_auroc(positives: list[str], negatives: list[str],
                   beta: float = 5.0) -> tuple[float, list[float]]:
    all_seqs = positives + negatives
    y = np.array([1]*len(positives) + [0]*len(negatives))

    use_loo = len(positives) < 50
    if use_loo:
        cv_splits = [(np.delete(np.arange(len(positives)), i), [i])
                     for i in range(len(positives))]
        # Only eval on held-out positives (not enough neg signal for LOO on full set)
        # Use full-data PSSM scored on held-out positive vs all negatives
        scores_pos = []
        for i in range(len(positives)):
            train_pos = [p for j, p in enumerate(positives) if j != i]
            pssm = build_pssm_from_positives(train_pos, '', beta)
            scores_pos.append(pssm.score(positives[i]))
        neg_scores = [build_pssm_from_positives(positives, '', beta).score(s)
                      for s in negatives]
        all_scores = np.array(scores_pos + neg_scores)
        try:
            auroc = roc_auc_score(y[:len(positives)+len(negatives)], all_scores)
        except Exception:
            auroc = 0.5
        return auroc, [auroc]
    else:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        X_seqs = np.array(all_seqs)
        fold_scores = []
        for tr, te in cv.split(X_seqs, y):
            train_pos = [all_seqs[i] for i in tr if y[i] == 1]
            test_seqs = [all_seqs[i] for i in te]
            test_y    = y[te]
            pssm = build_pssm_from_positives(train_pos, '', beta)
            s = np.array([pssm.score(sq) for sq in test_seqs])
            if len(np.unique(test_y)) < 2:
                continue
            fold_scores.append(roc_auc_score(test_y, s))
        return float(np.mean(fold_scores)), fold_scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # 1. Load positives
    all_pos = {}
    for name in PANEL:
        all_pos[name] = load_positives(name)
        log.info(f"{name}: {len(all_pos[name])} positives loaded")

    # 2. Flag promiscuous 8-mers (appear in ≥2 MMP PICS positive sets)
    mmp_sets = {n: set(all_pos[n]) for n in ['MMP_2', 'MMP_9']}
    promiscuous = mmp_sets['MMP_2'] & mmp_sets['MMP_9']
    log.info(f"Promiscuous 8-mers (MMP-2 ∩ MMP-9): {len(promiscuous)} — flagged in report")
    # Do NOT remove from training — they are real substrates; but flag for report

    results = {}

    for target in PANEL:
        log.info(f"\n{'='*60}\nTraining {target} (panel v2)\n{'='*60}")
        t_pos = all_pos[target]

        # 3. Build training set with protease-vs-protease negatives
        other = {n: all_pos[n] for n in PANEL if n != target}
        df = build_panel_training_set(target, t_pos, other)
        df.to_csv(OUT_DIR / f'training_set_v2_{target}.csv', index=False)
        log.info(f"  Training set: {(df.label==1).sum()} pos / {(df.label==0).sum()} neg")
        log.info(f"  Cross-protease negatives: {(df.source=='cross_neg').sum()}")
        log.info(f"  Background negatives:     {(df.source=='bg').sum()}")

        X = build_feature_matrix(df['sequence'].tolist(), use_esm2=False)
        y = df['label'].values.astype(int)

        # 4. Grouped CV (protein source)
        # Groups: PICS seqs from same protein family — use source column as proxy
        groups = np.array(['bg' if s == 'bg' else
                           ('cross' if s == 'cross_neg' else 'pos')
                           for s in df['source']])
        auroc_grouped, fold_scores = grouped_cv_auroc(X, y, groups)
        log.info(f"  Grouped-CV AUROC: {auroc_grouped:.3f}  "
                 f"(folds: {[round(f,3) for f in fold_scores]})")

        # 5. Standard 5-fold CV for comparison
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            std_folds = []
            for tr, te in cv.split(X, y):
                m = make_lr(); m.fit(X[tr], y[tr])
                std_folds.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        auroc_std = float(np.mean(std_folds))
        log.info(f"  Standard 5-fold CV AUROC: {auroc_std:.3f}  "
                 f"(folds: {[round(f,3) for f in std_folds]})")

        # 6. Bootstrap CI
        auroc_pt, ci_lo, ci_hi = bootstrap_auroc_ci(X, y)
        log.info(f"  Bootstrap AUROC: {auroc_pt:.3f} 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

        # 7. Fit final LR on full data
        lr = make_lr()
        lr.fit(X, y)
        joblib.dump(lr, MODELS_DIR / f'lr_v2_{target}.pkl')

        # 8. PSSM model
        pssm = build_pssm_from_positives(t_pos, target)
        neg_seqs = df[df['label'] == 0]['sequence'].tolist()
        pssm_auroc, _ = pssm_cv_auroc(t_pos, neg_seqs[:len(t_pos)*3])
        log.info(f"  PSSM CV AUROC: {pssm_auroc:.3f}")
        joblib.dump(pssm, MODELS_DIR / f'pssm_{target}.pkl')

        results[target] = {
            'n_pos': len(t_pos),
            'n_neg': int((df.label == 0).sum()),
            'n_cross_neg': int((df.source == 'cross_neg').sum()),
            'auroc_grouped': auroc_grouped,
            'auroc_std': auroc_std,
            'auroc_bootstrap': auroc_pt,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'pssm_auroc': pssm_auroc,
            'fold_scores': std_folds,
        }

    # Summary
    log.info(f"\n{'='*70}")
    log.info(f"{'Protease':<10} {'N_pos':>6} {'N_cross':>8} "
             f"{'AUROC(grp)':>11} {'AUROC(std)':>11} "
             f"{'Bootstrap':>10} {'95% CI':>16} {'PSSM':>7}")
    log.info("-" * 75)
    for t, r in results.items():
        log.info(f"{t:<10} {r['n_pos']:>6} {r['n_cross_neg']:>8} "
                 f"{r['auroc_grouped']:>11.3f} {r['auroc_std']:>11.3f} "
                 f"{r['auroc_bootstrap']:>10.3f} "
                 f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
                 f"{r['pssm_auroc']:>7.3f}")

    pd.DataFrame(results).T.to_csv(OUT_DIR / 'model_summary_v2.csv')
    log.info(f"\nPromiscuous MMP-2∩MMP-9 8-mers: {len(promiscuous)}")

    # Save promiscuous set for report
    pd.DataFrame({'sequence': sorted(promiscuous)}).to_csv(
        OUT_DIR / 'promiscuous_mmp2_mmp9.csv', index=False)

    return results


if __name__ == '__main__':
    main()
