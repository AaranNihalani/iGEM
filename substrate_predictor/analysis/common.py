"""Shared helpers for the audit and the report analysis (no writes to models/)."""
import re, subprocess, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.features import VHSE8 as VHSE_NEW              # corrected table
from src.utils import STANDARD_AAS, SWISSPROT_BG, AA_INDEX
from src.pssm import build_pssm_from_positives

SEED = 42
PROC = BASE / 'data' / 'processed'


def _old_vhse():
    """Pre-fix table, recovered from git history, to quantify the effect of the bug."""
    txt = subprocess.check_output(
        ['git', 'show', 'HEAD:substrate_predictor/src/features.py'],
        cwd=BASE.parent, text=True)
    m = re.search(r"VHSE8: dict\[str, list\[float\]\] = (\{.*?\n\})", txt, re.S)
    return eval(m.group(1))

VHSE_OLD = _old_vhse()


def encode(seqs, table):
    M = np.array([table[a] for a in STANDARD_AAS], dtype=np.float32)
    idx = np.array([[AA_INDEX[a] for a in s] for s in seqs])
    return M[idx].reshape(len(seqs), -1)


def make_lr():
    return Pipeline([('scaler', StandardScaler()),
                     ('clf', LogisticRegression(C=1.0, max_iter=2000,
                              class_weight='balanced', random_state=SEED,
                              solver='lbfgs'))])


def load_set(name):
    df = pd.read_csv(PROC / f'training_set_{name}.csv')
    df['sequence'] = df['sequence'].str.upper()
    return df


def hamming_groups(pos, max_dist=2):
    """Connected components of positives with Hamming distance <= max_dist."""
    A = np.array([[AA_INDEX[a] for a in s] for s in pos], dtype=np.int8)
    n = len(A)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        d = (A[i + 1:] != A[i]).sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0]:
            ri, rj = find(i), find(i + 1 + j)
            if ri != rj: parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def assign_groups(df, max_dist=2):
    """Group id per row. Shuffled negatives inherit the group of the positive with the
    same composition; other negatives are singletons."""
    pos_idx = np.nonzero(df.label.values == 1)[0]
    pos = df.sequence.values[pos_idx].tolist()
    gpos = hamming_groups(pos, max_dist)
    comp2g = {}
    for s, g in zip(pos, gpos):
        comp2g.setdefault(''.join(sorted(s)), g)
    groups = np.empty(len(df), dtype=np.int64)
    nxt = int(gpos.max()) + 1 if len(gpos) else 0
    groups[pos_idx] = gpos
    for k in np.nonzero(df.label.values == 0)[0]:
        key = ''.join(sorted(df.sequence.values[k]))
        if key in comp2g:
            groups[k] = comp2g[key]
        else:
            groups[k] = nxt; nxt += 1
    return groups


def oof_lr(X, y, splitter, groups=None):
    p = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tr, te in (splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)):
            m = make_lr().fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
    return p


def rand_cv():
    return StratifiedKFold(5, shuffle=True, random_state=SEED)
