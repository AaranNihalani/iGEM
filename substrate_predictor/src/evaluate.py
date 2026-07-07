"""
Evaluation figures for the trained classifiers.

Produces:
  figures/fig_roc_curves.png       -- 6-panel ROC, one per protease
  figures/fig_feature_importance.png -- LR coefficient heatmap (positions x VHSE descriptors)
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

from src.utils import PROTEASE_ORDER, STANDARD_AAS, confidence_tier
from src.features import build_feature_matrix, VHSE8_DIM, esm2_available
from src.data_pipeline import apply_smote_if_needed

log = logging.getLogger(__name__)

RANDOM_SEED = 42
VHSE_LABELS = ['Hydrophob.', 'Steric', 'Electr. 1', 'Electr. 2',
               'Electr. 3', 'H-bond don.', 'H-bond acc.', 'Charge']
POSITION_LABELS = ["P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'"]


def _apply_figure_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(processed_dir: Path,
                    models_dir: Path,
                    figures_dir: Path) -> None:
    """
    Generate a 2x3 ROC figure by running held-out CV predictions for each protease.
    Both LR and LGBM curves are overlaid on the same axes if LGBM is available.
    """
    _apply_figure_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for idx, protease in enumerate(PROTEASE_ORDER):
        ax = axes[idx]
        csv_path = processed_dir / f"training_set_{protease.replace('-','_')}.csv"
        if not csv_path.exists():
            ax.set_title(f"{protease}\n(no data)")
            ax.set_visible(False)
            continue

        df = pd.read_csv(csv_path)
        sequences = df['sequence'].tolist()
        y = df['label'].values.astype(int)
        X = build_feature_matrix(sequences, use_esm2=False)
        X_sm, y_sm = apply_smote_if_needed(X, y)

        # LR curve
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        lr = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=2000,
                                       class_weight='balanced',
                                       random_state=RANDOM_SEED)),
        ])
        _plot_cv_roc(ax, lr, X_sm, y_sm, cv, label='LR (VHSE8)',
                     color='steelblue', linestyle='-')

        # LGBM curve if available
        lgbm_path = models_dir / f"lgbm_{protease.replace('-','_')}.pkl"
        if lgbm_path.exists():
            try:
                from lightgbm import LGBMClassifier
                lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                      num_leaves=31, class_weight='balanced',
                                      random_state=RANDOM_SEED, verbose=-1)
                _plot_cv_roc(ax, lgbm, X_sm, y_sm, cv, label='LGBM',
                             color='darkorange', linestyle='--')
            except Exception:
                pass

        ax.plot([0, 1], [0, 1], 'k:', alpha=0.4, linewidth=0.8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        tier = confidence_tier(int(y.sum()))
        ax.set_title(f"{protease}  [{tier} confidence]", fontweight='bold')
        ax.legend(loc='lower right', fontsize=7, framealpha=0.7)

    fig.suptitle('ROC curves: per-protease substrate classifiers (5-fold CV)',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = figures_dir / 'fig_roc_curves.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"ROC figure saved to {out}")


def _plot_cv_roc(ax, model, X, y, cv, label, color, linestyle):
    """Plot mean ROC curve across CV folds on ax."""
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 200)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for train_idx, val_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[val_idx])[:, 1]
            fpr, tpr, _ = roc_curve(y[val_idx], proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))
    ax.plot(mean_fpr, mean_tpr, color=color, linestyle=linestyle, linewidth=1.8,
            label=f'{label} (AUC={mean_auc:.2f}$\\pm${std_auc:.2f})')
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    alpha=0.10, color=color)


# ---------------------------------------------------------------------------
# Feature importance heatmap
# ---------------------------------------------------------------------------

def plot_feature_importance(models_dir: Path,
                             figures_dir: Path,
                             protease: str = 'MMP14') -> None:
    """
    Heatmap of LR coefficients for one protease, reshaped to (8 positions x 8 VHSE descriptors).
    Shows which position/property combinations drive cleavage prediction.
    """
    _apply_figure_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"lr_{protease.replace('-','_')}.pkl"
    if not model_path.exists():
        log.warning(f"LR model for {protease} not found at {model_path}")
        return

    model = joblib.load(model_path)
    coefs = model.named_steps['clf'].coef_[0]  # shape (64,)

    if len(coefs) != VHSE8_DIM:
        log.warning(f"Unexpected coefficient length {len(coefs)}; expected {VHSE8_DIM}")
        return

    coef_matrix = coefs.reshape(8, 8)  # (positions, VHSE descriptors)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        coef_matrix,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7},
        xticklabels=VHSE_LABELS,
        yticklabels=POSITION_LABELS,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': 'LR coefficient', 'shrink': 0.8},
    )
    ax.set_title(
        f'LR coefficient heatmap: {protease}  (position x VHSE8 descriptor)',
        fontweight='bold', pad=10,
    )
    ax.set_xlabel('VHSE8 descriptor')
    ax.set_ylabel('Cleavage-site position')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    out = figures_dir / 'fig_feature_importance.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Feature importance figure saved to {out}")
