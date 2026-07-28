"""
Integrate Eckhard 2016 PICS data with existing MEROPS positives,
rebuild training sets, and retrain all MMP models + LasB.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
log = logging.getLogger(__name__)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score

from src.data_pipeline import build_training_set
from src.features import build_feature_matrix
from src.utils import confidence_tier

RANDOM_SEED = 42
CV_FOLDS = 5

ECKHARD_DIR = BASE / 'data' / 'raw' / 'eckhard2016'
PROCESSED_DIR = BASE / 'data' / 'processed'
MODELS_DIR = BASE / 'models'
PROCESSED_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MMP_MAP = {
    'MMP_9':  'eckhard_MMP_9_positives.csv',
    'MMP_8':  'eckhard_MMP_8_positives.csv',
    'MMP_2':  'eckhard_MMP_2_positives.csv',
    'MMP14':  'eckhard_MMP14_positives.csv',
}

LASB_CSV = BASE / 'data' / 'raw' / 'lasb_curated_positives.csv'


def make_lr():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight='balanced',
                                   random_state=RANDOM_SEED, solver='lbfgs')),
    ])


def cv_scores(model, X, y):
    scorers = {
        'roc_auc': 'roc_auc',
        'mcc': make_scorer(matthews_corrcoef),
        'pr_auc': make_scorer(average_precision_score, response_method='predict_proba'),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
    return (float(np.mean(res['test_roc_auc'])),
            float(np.mean(res['test_mcc'])),
            float(np.mean(res['test_pr_auc'])))


def train_one(name, positives):
    n = len(positives)
    tier = confidence_tier(n)
    log.info(f"{name}: {n} positives [{tier.upper()}]")

    df = build_training_set(positives, target_ratio=3.0, seed=RANDOM_SEED)
    df.to_csv(PROCESSED_DIR / f'training_set_{name}.csv', index=False)

    X = build_feature_matrix(df['sequence'].tolist(), use_esm2=False)
    y = df['label'].values.astype(int)

    lr = make_lr()
    auroc, mcc, prauc = cv_scores(lr, X, y)
    log.info(f"  CV  AUROC={auroc:.3f}  MCC={mcc:.3f}  PR-AUC={prauc:.3f}")

    lr.fit(X, y)
    joblib.dump(lr, MODELS_DIR / f'lr_{name}.pkl')
    return auroc, mcc, prauc, tier


results = {}

# --- MMP models ---
for key, fname in MMP_MAP.items():
    existing_csv = PROCESSED_DIR / f'training_set_{key}.csv'
    merops_pos = []
    if existing_csv.exists():
        df_ex = pd.read_csv(existing_csv)
        merops_pos = df_ex[df_ex['label'] == 1]['sequence'].tolist()

    eckhard_path = ECKHARD_DIR / fname
    eckhard_pos = pd.read_csv(eckhard_path)['sequence'].tolist() if eckhard_path.exists() else []

    combined = list(dict.fromkeys(merops_pos + eckhard_pos))
    results[key] = train_one(key, combined)

# --- LasB ---
# Use MEROPS-only positives.  Yang 2015 elastin sites are VPGVG/Gly-Val-Ala-Pro-rich and
# overlap compositionally with Swiss-Prot background "negatives", collapsing AUROC from
# 0.816 (MEROPS only) to 0.582 (mixed set).  Keep the elastin rows in the CSV for
# reference, but exclude them from training until better negative sampling is in place.
lasb_df = pd.read_csv(LASB_CSV)
lasb_pos = lasb_df[lasb_df['source'] == 'MEROPS M04.005']['sequence'].tolist()
log.info(f"LasB: {len(lasb_pos)} MEROPS-only positives "
         f"({len(lasb_df) - len(lasb_pos)} Yang2015 elastin rows excluded from training)")
results['LasB'] = train_one('LasB', lasb_pos)

# --- NE (HNE) — retrain unchanged to refresh model file ---
ne_csv = PROCESSED_DIR / 'training_set_NE.csv'
if ne_csv.exists():
    df_ne = pd.read_csv(ne_csv)
    ne_pos = df_ne[df_ne['label'] == 1]['sequence'].tolist()
    results['NE'] = train_one('NE', ne_pos)

log.info("\n=== SUMMARY ===")
log.info(f"{'Protease':<10} {'N_pos':>6}  {'AUROC':>6}  {'MCC':>6}  {'PR-AUC':>7}  Confidence")
log.info("-" * 55)
for name, (auroc, mcc, prauc, tier) in results.items():
    df_tmp = pd.read_csv(PROCESSED_DIR / f'training_set_{name}.csv')
    n = int(df_tmp['label'].sum())
    log.info(f"{name:<10} {n:>6}  {auroc:>6.3f}  {mcc:>6.3f}  {prauc:>7.3f}  {tier.upper()}")
