"""Scoring a peptide against a panel of proteases.

Selectivity is defined on background percentiles, which put every protease on the same scale:

    u_i(x)  = consensus percentile of x under protease i (LR and PSSM background percentiles,
              combined by the harmonic mean)
    M_t(x)  = u_t(x) - max_{j != t} u_j(x)         specificity margin, in [-1, 1]

The ratio of raw probabilities p_t / max p_j is not used: independently trained classifiers have
different base rates (mean P on random 8-mers ranges 0.11 to 0.33), so that ratio is scale-dependent.
"""
import numpy as np
import pandas as pd

from .constants import PROTEASES
from .model import ProteaseModel, load_all

PRESETS = {
    "wound": ("MMP2", "MMP8", "MMP9", "NE", "LASB"),
    "mmp_triplex": ("MMP2", "MMP9", "NE"),
    "mmp9_ne": ("MMP9", "NE"),
    "all": tuple(PROTEASES),
}


class Panel:
    def __init__(self, members, models: dict | None = None):
        models = models or load_all()
        self.members = [m for m in members if m in models]
        if len(self.members) < 2:
            raise ValueError("a panel needs at least two trained proteases")
        self.models = {m: models[m] for m in self.members}

    def score(self, seqs) -> pd.DataFrame:
        seqs = list(seqs)
        cols = {"sequence": seqs}
        for m, mod in self.models.items():
            cols[f"P_{m}"] = mod.prob(seqs)
            cols[f"pct_lr_{m}"] = mod.lr_percentile(seqs)
            cols[f"pct_pssm_{m}"] = mod.pssm_percentile(seqs)
            cols[f"u_{m}"] = mod.consensus(seqs)
        return pd.DataFrame(cols)

    def margin(self, scores: pd.DataFrame, target: str) -> np.ndarray:
        others = [f"u_{m}" for m in self.members if m != target]
        return scores[f"u_{target}"].values - scores[others].max(axis=1).values

    def rank(self, seqs, target: str, min_target: float = 0.9) -> pd.DataFrame:
        s = self.score(seqs)
        s["margin"] = self.margin(s, target)
        s["strongest_competitor"] = s[[f"u_{m}" for m in self.members if m != target]].idxmax(axis=1).str[2:]
        s = s[s[f"u_{target}"] >= min_target]
        return s.sort_values("margin", ascending=False).reset_index(drop=True)
