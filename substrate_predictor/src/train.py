"""
Per-protease classifier training with stratified 5-fold cross-validation.

Models:
  lr    -- LogisticRegression(L2) on VHSE8 features (primary, interpretable)
  lgbm  -- LGBMClassifier on VHSE8 [+ ESM2] features (comparison)

FAP is skipped for LGBM (< 20 positives).
All models are saved to models/ as joblib .pkl files.
CV results are saved to data/outputs/cv_results.csv.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer, matthews_corrcoef, average_precision_score, brier_score_loss,
)

from src.utils import PROTEASE_ORDER, confidence_tier
from src.features import build_feature_matrix, esm2_available
from src.data_pipeline import apply_smote_if_needed

log = logging.getLogger(__name__)

LGBM_MIN_POSITIVES = 20
CV_FOLDS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# LightGBM import (optional)
# ---------------------------------------------------------------------------

def _lgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Build LR pipeline
# ---------------------------------------------------------------------------

def _make_lr() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=1.0, max_iter=2000,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            solver='lbfgs',
        )),
    ])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _cv_scores(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Run stratified 5-fold CV and return mean metrics."""
    scorers = {
        'roc_auc':    'roc_auc',
        'mcc':        make_scorer(matthews_corrcoef),
        'pr_auc':     make_scorer(average_precision_score,
                                  response_method='predict_proba'),
        'brier':      make_scorer(brier_score_loss,
                                  response_method='predict_proba',
                                  greater_is_better=False),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = cross_validate(model, X, y, cv=cv, scoring=scorers,
                                 return_train_score=False, n_jobs=-1)
    return {
        'auroc':  float(np.mean(results['test_roc_auc'])),
        'mcc':    float(np.mean(results['test_mcc'])),
        'pr_auc': float(np.mean(results['test_pr_auc'])),
        'brier':  float(np.mean(-results['test_brier'])),  # stored as negative
    }


# ---------------------------------------------------------------------------
# Train one protease
# ---------------------------------------------------------------------------

def train_protease(protease: str,
                   df: pd.DataFrame,
                   models_dir: Path,
                   use_esm2: bool = False,
                   esm2_cache_dir: Path | None = None) -> dict:
    """
    Train LR and (optionally) LGBM models for one protease.
    Returns a dict of CV metrics for inclusion in cv_results.csv.
    """
    sequences = df['sequence'].tolist()
    y = df['label'].values.astype(int)
    n_pos = y.sum()
    tier = confidence_tier(n_pos)

    log.info(f"  {protease}: {n_pos} positives, {len(y)-n_pos} negatives "
             f"[confidence: {tier}]")

    cache_path = (esm2_cache_dir / f"esm2_cache_{protease.replace('-','_')}.pkl"
                  if esm2_cache_dir else None)
    X_phys = build_feature_matrix(sequences, use_esm2=False)
    X_comb = (build_feature_matrix(sequences, use_esm2=True, cache_path=cache_path)
              if use_esm2 and esm2_available() else X_phys)

    # Apply SMOTE on physicochemical features
    X_phys_sm, y_sm = apply_smote_if_needed(X_phys, y)

    rows = []

    # --- Logistic Regression (primary) ---
    lr = _make_lr()
    cv_lr = _cv_scores(lr, X_phys_sm, y_sm)
    log.info(f"    LR  AUROC={cv_lr['auroc']:.3f}  MCC={cv_lr['mcc']:.3f}")
    lr.fit(X_phys_sm, y_sm)
    joblib.dump(lr, models_dir / f"lr_{protease.replace('-','_')}.pkl")
    rows.append({'protease': protease, 'model': 'LR', **cv_lr, 'confidence': tier})

    # --- LightGBM (comparison) ---
    if n_pos < LGBM_MIN_POSITIVES:
        log.warning(f"    {protease}: skipping LGBM (< {LGBM_MIN_POSITIVES} positives)")
    elif not _lgbm_available():
        log.warning("    lightgbm not installed; skipping LGBM.")
    else:
        # SMOTE on combined features if available
        X_comb_sm, y_comb_sm = apply_smote_if_needed(X_comb, y)
        lgbm = _make_lgbm()
        cv_lgbm = _cv_scores(lgbm, X_comb_sm, y_comb_sm)
        log.info(f"    LGBM AUROC={cv_lgbm['auroc']:.3f}  MCC={cv_lgbm['mcc']:.3f}")
        lgbm.fit(X_comb_sm, y_comb_sm)
        joblib.dump(lgbm, models_dir / f"lgbm_{protease.replace('-','_')}.pkl")
        rows.append({'protease': protease, 'model': 'LGBM', **cv_lgbm,
                     'confidence': tier})

    return rows


# ---------------------------------------------------------------------------
# Train all proteases
# ---------------------------------------------------------------------------

def train_all(processed_dir: Path,
              models_dir: Path,
              outputs_dir: Path,
              use_esm2: bool = False) -> pd.DataFrame:
    """
    Load training CSVs, train all per-protease models, save artefacts.
    Returns the CV results DataFrame.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    esm2_cache = processed_dir if use_esm2 else None

    all_rows = []
    for protease in PROTEASE_ORDER:
        csv_path = processed_dir / f"training_set_{protease.replace('-','_')}.csv"
        if not csv_path.exists():
            log.warning(f"  {protease}: training set not found at {csv_path}; skipping.")
            continue

        df = pd.read_csv(csv_path)
        if df['label'].sum() == 0:
            log.warning(f"  {protease}: no positive examples; skipping.")
            continue

        log.info(f"Training {protease} ...")
        rows = train_protease(protease, df, models_dir,
                              use_esm2=use_esm2, esm2_cache_dir=esm2_cache)
        all_rows.extend(rows)

    cv_df = pd.DataFrame(all_rows,
                         columns=['protease', 'model', 'auroc', 'mcc',
                                  'pr_auc', 'brier', 'confidence'])
    out_path = outputs_dir / 'cv_results.csv'
    cv_df.to_csv(out_path, index=False)
    log.info(f"CV results saved to {out_path}")
    return cv_df
