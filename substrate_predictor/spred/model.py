"""Per-protease model: logistic regression on VHSE8 + PSSM, with background null distributions.

The model is stored as plain JSON (arrays only) so that the Python package and the browser tool
share one definition:

    z      = (x - mean) / scale
    logit  = intercept + coef . z                  P(cleave) = sigmoid(logit)
    S_pssm = sum_i log2( p_i[x_i] / bg[x_i] )

Null distributions (quantile grids of each score over random Swiss-Prot 8-mers) convert raw scores
into background percentiles, which are comparable across proteases.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import pssm as pssm_mod
from .constants import POSITIONS, RANDOM_SEED, WINDOW
from .data import background_sequences
from .features import encode, index_matrix

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
GRID = np.linspace(0.0, 1.0, 1001)
NULL_SIZE = 100_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def fit_logistic(X, y, C: float = 1.0):
    """Standardise, then L2 logistic regression (balanced class weights)."""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="lbfgs",
                             random_state=RANDOM_SEED).fit(sc.transform(X), y)
    return sc.mean_, sc.scale_, clf.coef_[0], float(clf.intercept_[0])


def select_C(X, y, groups, seed: int = RANDOM_SEED, n_splits: int = 3) -> float:
    """Regularisation strength chosen by inner cluster-aware CV (mean AUROC over C_GRID)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    inner = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups))
    best, best_score = 1.0, -1.0
    for C in C_GRID:
        sc = []
        for a, b in inner:
            mean, scale, coef, i0 = fit_logistic(X[a], y[a], C)
            sc.append(roc_auc_score(y[b], ((X[b] - mean) / scale) @ coef + i0))
        if np.mean(sc) > best_score:
            best, best_score = C, float(np.mean(sc))
    return best


def _monotone(q):
    q = np.maximum.accumulate(np.asarray(q, dtype=float))
    return q + np.arange(len(q)) * 1e-12


@dataclass
class ProteaseModel:
    protease: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    log_odds: np.ndarray
    null_lr: np.ndarray = field(default=None)
    null_pssm: np.ndarray = field(default=None)
    meta: dict = field(default_factory=dict)

    # ---- scoring
    def logit(self, seqs) -> np.ndarray:
        z = (encode(seqs) - self.mean) / self.scale
        return self.intercept + z @ self.coef

    def prob(self, seqs) -> np.ndarray:
        return sigmoid(self.logit(seqs))

    def pssm(self, seqs) -> np.ndarray:
        return pssm_mod.score(self.log_odds, seqs)

    def lr_percentile(self, seqs) -> np.ndarray:
        return np.interp(self.prob(seqs), self.null_lr, GRID)

    def pssm_percentile(self, seqs) -> np.ndarray:
        return np.interp(self.pssm(seqs), self.null_pssm, GRID)

    def consensus(self, seqs) -> np.ndarray:
        """Harmonic mean of the LR and PSSM background percentiles (penalises disagreement)."""
        a, b = self.lr_percentile(seqs), self.pssm_percentile(seqs)
        return 2 * a * b / (a + b + 1e-12)

    def contributions(self, seq: str) -> np.ndarray:
        """Per-position contribution to the logit (sum + intercept = logit)."""
        z = (encode([seq])[0] - self.mean) / self.scale
        return (self.coef * z).reshape(WINDOW, -1).sum(axis=1)

    # ---- persistence
    def to_dict(self) -> dict:
        r = lambda a, n=6: np.round(np.asarray(a, dtype=float), n).tolist()
        return {"protease": self.protease, "positions": list(POSITIONS),
                "feature_mean": r(self.mean, 8), "feature_scale": r(self.scale, 8), "coef": r(self.coef, 8),
                "intercept": round(self.intercept, 8), "pssm_log_odds": r(self.log_odds, 6),
                "null_lr_quantiles": r(self.null_lr, 8), "null_pssm_quantiles": r(self.null_pssm, 6),
                "meta": self.meta}

    def save(self, directory: Path = MODELS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.protease}.json"
        p.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        return p

    @classmethod
    def from_dict(cls, d: dict) -> "ProteaseModel":
        return cls(d["protease"], np.array(d["feature_mean"]), np.array(d["feature_scale"]), np.array(d["coef"]),
                   d["intercept"], np.array(d["pssm_log_odds"]), np.array(d["null_lr_quantiles"]),
                   np.array(d["null_pssm_quantiles"]), d.get("meta", {}))

    @classmethod
    def load(cls, protease: str, directory: Path = MODELS_DIR) -> "ProteaseModel":
        return cls.from_dict(json.loads((directory / f"{protease}.json").read_text()))


def load_all(directory: Path = MODELS_DIR) -> dict:
    return {p.stem: ProteaseModel.load(p.stem, directory) for p in sorted(directory.glob("*.json"))}


def build_null(model: ProteaseModel, n: int = NULL_SIZE, seed: int = RANDOM_SEED + 1) -> None:
    seqs = background_sequences(n, np.random.default_rng(seed))
    model.null_lr = _monotone(np.quantile(model.prob(seqs), GRID))
    model.null_pssm = _monotone(np.quantile(model.pssm(seqs), GRID))


def fit_model(protease: str, frame, positives, meta=None, groups=None) -> ProteaseModel:
    """Fit on a training frame (sequence, label, ...) and the target's positives; C by inner CV."""
    from .evaluation import hamming_clusters
    X = encode(frame["sequence"].tolist())
    y = frame["label"].values.astype(int)
    if groups is None:
        groups = hamming_clusters(frame["sequence"].tolist())
    C = select_C(X, y, groups)
    mean, scale, coef, b = fit_logistic(X, y, C)
    meta = dict(meta or {}, C=C)
    m = ProteaseModel(protease, mean, scale, coef, b, pssm_mod.fit_log_odds(positives), meta=meta)
    build_null(m)
    return m
