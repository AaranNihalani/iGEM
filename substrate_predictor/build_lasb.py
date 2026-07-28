"""
Build and train a LasB (M04.005 pseudolysin) classifier.

Positives: 35 literature-curated cleavage sites from MEROPS M04.005 substrates
           (see data/raw/lasb_curated_positives.csv)
Confidence: LOW (n=35, < 50 threshold) — use for candidate ranking only;
            validate hits with measured kcat/KM before ordering synthesis.

Usage:
    python build_lasb.py
"""

import sys
import logging
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score

from src.data_pipeline import build_training_set
from src.features import build_feature_matrix
from src.utils import confidence_tier

RANDOM_SEED = 42
CV_FOLDS = 5


def main():
    raw_csv = BASE / 'data' / 'raw' / 'lasb_curated_positives.csv'
    processed_dir = BASE / 'data' / 'processed'
    models_dir = BASE / 'models'
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load curated positives
    curated = pd.read_csv(raw_csv)
    positives = curated['sequence'].tolist()
    log.info(f"Loaded {len(positives)} LasB positives from curated CSV")

    tier = confidence_tier(len(positives))
    log.info(f"Confidence tier: {tier.upper()} ({len(positives)} positives)")

    # Build training set using same pipeline as other proteases
    df = build_training_set(positives, target_ratio=3.0, seed=RANDOM_SEED)
    log.info(f"Training set: {df['label'].sum()} pos / {(df['label']==0).sum()} neg")

    out_csv = processed_dir / 'training_set_LasB.csv'
    df.to_csv(out_csv, index=False)
    log.info(f"Saved training set to {out_csv}")

    # Build features
    seqs = df['sequence'].tolist()
    y = df['label'].values.astype(int)
    X = build_feature_matrix(seqs, use_esm2=False)

    # Cross-validate
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight='balanced',
                                   random_state=RANDOM_SEED, solver='lbfgs')),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scorers = {
        'roc_auc': 'roc_auc',
        'mcc': make_scorer(matthews_corrcoef),
        'pr_auc': make_scorer(average_precision_score, response_method='predict_proba'),
    }
    results = cross_validate(lr, X, y, cv=cv, scoring=scorers, n_jobs=-1)

    auroc = float(np.mean(results['test_roc_auc']))
    mcc   = float(np.mean(results['test_mcc']))
    prauc = float(np.mean(results['test_pr_auc']))
    log.info(f"5-fold CV  AUROC={auroc:.3f}  MCC={mcc:.3f}  PR-AUC={prauc:.3f}")

    # Fit final model on full training set
    lr.fit(X, y)
    out_model = models_dir / 'lr_LasB.pkl'
    joblib.dump(lr, out_model)
    log.info(f"Model saved to {out_model}")

    log.info(f"\n{'='*50}")
    log.info(f"LasB model summary")
    log.info(f"  Positives (curated):  {len(positives)}")
    log.info(f"  Confidence tier:      {tier.upper()}")
    log.info(f"  AUROC (5-fold CV):    {auroc:.3f}")
    log.info(f"  MCC  (5-fold CV):     {mcc:.3f}")
    log.info(f"  Model path:           {out_model}")
    log.info(f"{'='*50}")
    log.info("IMPORTANT: LOW CONFIDENCE — validate ranked candidates with")
    log.info("measured kcat/KM before ordering synthesis.")


if __name__ == '__main__':
    main()
