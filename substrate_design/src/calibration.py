"""
Model quality validation for the 3-plex classifiers (MMP-2, MMP-9, NE):
  1. Calibration curves — reliability diagrams with Expected Calibration Error
  2. Bootstrapped 95% CIs on AUROC, MCC, PR-AUC (1,000 bootstrap iterations)
  3. Composition baseline — compare VHSE8 LR vs bag-of-AAs LR

Outputs:
  figures/fig_calibration_curves.png
  figures/fig_roc_bootstrap.png
  data/outputs/calibration_results.csv
  data/outputs/bootstrap_ci.csv
  data/outputs/composition_baseline.csv
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, average_precision_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

BASE = Path(__file__).parent.parent.parent

log = logging.getLogger(__name__)

from src.features import encode_vhse8_batch  # local, avoids sys.path conflict

TRIPLEX = ['MMP-2', 'MMP-9', 'NE']
N_BINS = 10
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
CV_FOLDS = 5


def _get_vhse8(seqs: list[str]) -> np.ndarray:
    return encode_vhse8_batch(seqs)


def _bag_of_aas(seqs: list[str]) -> np.ndarray:
    """20-dim AA frequency vector per sequence — composition baseline."""
    STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
    aa_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}
    X = np.zeros((len(seqs), 20))
    for i, seq in enumerate(seqs):
        for aa in seq.upper():
            if aa in aa_idx:
                X[i, aa_idx[aa]] += 1.0
        total = X[i].sum()
        if total > 0:
            X[i] /= total
    return X


def _make_lr() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight='balanced',
                                   random_state=RANDOM_SEED, solver='lbfgs')),
    ])


def load_data(processed_dir: Path, protease: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    csv_path = processed_dir / f"training_set_{protease.replace('-', '_')}.csv"
    df = pd.read_csv(csv_path)
    seqs = df['sequence'].tolist()
    y = df['label'].values.astype(int)
    X = _get_vhse8(seqs)
    return X, y, seqs


# ---------------------------------------------------------------------------
# 1. Calibration curves
# ---------------------------------------------------------------------------

def calibration_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error (ECE) — weighted mean calibration deviation."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        mean_pred = y_prob[mask].mean()
        ece += mask.sum() / n * abs(frac_pos - mean_pred)
    return ece


def plot_calibration(processed_dir: Path, models_dir: Path,
                     figures_dir: Path) -> pd.DataFrame:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    rows = []

    for ax, protease in zip(axes, TRIPLEX):
        X, y, seqs = load_data(processed_dir, protease)
        model = joblib.load(models_dir / f"lr_{protease.replace('-', '_')}.pkl")

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

        frac_pos, mean_pred = calibration_curve(y, y_prob, n_bins=N_BINS,
                                                 strategy='uniform')
        ece = calibration_ece(y, y_prob)
        brier = brier_score_loss(y, y_prob)

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        ax.plot(mean_pred, frac_pos, 'o-', color='steelblue', lw=2,
                markersize=5, label=f'LR (ECE={ece:.3f})')
        ax.fill_between(mean_pred, frac_pos - 0.05, frac_pos + 0.05,
                        alpha=0.12, color='steelblue')
        ax.set_xlabel('Mean predicted probability', fontsize=10)
        ax.set_title(protease, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        rows.append({'protease': protease, 'ece': round(ece, 4), 'brier': round(brier, 4)})
        log.info(f"  {protease}: ECE={ece:.4f}, Brier={brier:.4f}")

    axes[0].set_ylabel('Fraction positive', fontsize=10)
    fig.suptitle('Calibration curves — 3-plex LR classifiers (5-fold CV)',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = figures_dir / 'fig_calibration_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Calibration figure saved to {out}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Bootstrapped 95% CIs
# ---------------------------------------------------------------------------

def bootstrap_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       n_iter: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> dict:
    """Bootstrap AUROC, MCC, PR-AUC with 95% CI."""
    rng = np.random.default_rng(seed)
    aurocs, mccs, prauc = [], [], []

    for _ in range(n_iter):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            aurocs.append(roc_auc_score(yt, yp))
            mccs.append(matthews_corrcoef(yt, (yp >= 0.5).astype(int)))
            prauc.append(average_precision_score(yt, yp))
        except Exception:
            continue

    def ci(vals):
        if not vals:
            return (float('nan'), float('nan'))
        return (round(float(np.percentile(vals, 2.5)), 4),
                round(float(np.percentile(vals, 97.5)), 4))

    return {
        'auroc_lo': ci(aurocs)[0], 'auroc_hi': ci(aurocs)[1],
        'mcc_lo':   ci(mccs)[0],   'mcc_hi':   ci(mccs)[1],
        'prauc_lo': ci(prauc)[0],  'prauc_hi': ci(prauc)[1],
        'n_iter':   len(aurocs),
    }


def compute_bootstrap_cis(processed_dir: Path, models_dir: Path) -> pd.DataFrame:
    rows = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for protease in TRIPLEX:
        X, y, _ = load_data(processed_dir, protease)
        model = joblib.load(models_dir / f"lr_{protease.replace('-', '_')}.pkl")
        y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

        point_auroc = roc_auc_score(y, y_prob)
        point_mcc = matthews_corrcoef(y, (y_prob >= 0.5).astype(int))
        point_prauc = average_precision_score(y, y_prob)

        ci = bootstrap_metrics(y, y_prob)

        log.info(f"  {protease}: AUROC={point_auroc:.3f} [{ci['auroc_lo']}, {ci['auroc_hi']}]  "
                 f"MCC={point_mcc:.3f} [{ci['mcc_lo']}, {ci['mcc_hi']}]")

        rows.append({'protease': protease,
                     'auroc': round(point_auroc, 4),
                     'auroc_95ci': f"[{ci['auroc_lo']}, {ci['auroc_hi']}]",
                     'mcc': round(point_mcc, 4),
                     'mcc_95ci': f"[{ci['mcc_lo']}, {ci['mcc_hi']}]",
                     'prauc': round(point_prauc, 4),
                     'prauc_95ci': f"[{ci['prauc_lo']}, {ci['prauc_hi']}]",
                     'n_bootstrap': ci['n_iter']})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Composition baseline
# ---------------------------------------------------------------------------

def composition_baseline(processed_dir: Path) -> pd.DataFrame:
    """
    Compare VHSE8 LR vs bag-of-AAs LR (no positional information).
    A difference < 0.03 AUROC means positional encoding is not contributing.
    """
    rows = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for protease in TRIPLEX:
        X, y, seqs = load_data(processed_dir, protease)
        X_bag = _bag_of_aas(seqs)

        lr_vhse8 = _make_lr()
        lr_bag = _make_lr()

        yp_vhse8 = cross_val_predict(lr_vhse8, X, y, cv=cv, method='predict_proba')[:, 1]
        yp_bag = cross_val_predict(lr_bag, X_bag, y, cv=cv, method='predict_proba')[:, 1]

        auroc_vhse8 = roc_auc_score(y, yp_vhse8)
        auroc_bag = roc_auc_score(y, yp_bag)
        delta = auroc_vhse8 - auroc_bag

        status = 'positional encoding contributing' if delta >= 0.03 else 'marginal benefit'
        log.info(f"  {protease}: VHSE8={auroc_vhse8:.3f}, BagAA={auroc_bag:.3f}, "
                 f"Δ={delta:+.3f} ({status})")

        rows.append({'protease': protease,
                     'auroc_vhse8': round(auroc_vhse8, 4),
                     'auroc_bag_of_aas': round(auroc_bag, 4),
                     'delta': round(delta, 4),
                     'status': status})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(processed_dir: Path, models_dir: Path,
        output_dir: Path, figures_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("1. Calibration curves ...")
    cal_df = plot_calibration(processed_dir, models_dir, figures_dir)
    cal_df.to_csv(output_dir / 'calibration_results.csv', index=False)

    log.info("2. Bootstrapped 95% CIs ...")
    ci_df = compute_bootstrap_cis(processed_dir, models_dir)
    ci_df.to_csv(output_dir / 'bootstrap_ci.csv', index=False)

    log.info("3. Composition baseline ...")
    base_df = composition_baseline(processed_dir)
    base_df.to_csv(output_dir / 'composition_baseline.csv', index=False)

    return {'calibration': cal_df, 'bootstrap_ci': ci_df, 'baseline': base_df}
