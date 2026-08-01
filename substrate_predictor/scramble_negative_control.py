"""
scramble_negative_control.py
=============================
Generate a scrambled negative control for the HNE probe substrate AAPVSGIG.

Strategy
--------
* Enumerate all unique permutations of {A, A, P, V, S, G, I, G} (same amino
  acid composition → identical net charge = 0 at pH 7.4).
* Remove sequences that contain the AAPV motif anywhere (HNE recognition core).
* Score every remaining candidate with the Panel B models:
    - LR Panel B (HNE):    lr_NE_panel_b.pkl
    - PSSM Panel B (HNE):  pssm_NE_panel_b.pkl
    - LR Panel B (MMP-9):  lr_MMP_9_panel_b.pkl
    - PSSM Panel B (MMP-9): pssm_MMP_9_panel_b.pkl
* Rank by LR_HNE ascending; report top 10 and the recommended winner.

Run from the substrate_predictor directory:
    python scramble_negative_control.py
"""

import sys
import itertools
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

BASE   = Path(__file__).resolve().parent
MODELS = BASE / 'models'
sys.path.insert(0, str(BASE))

from src.features import encode_vhse8, build_feature_matrix   # noqa: E402

SOURCE = 'AAPVSGIG'
assert len(SOURCE) == 8

all_perms  = set(itertools.permutations(SOURCE))
candidates = [''.join(p) for p in all_perms]
print(f"Total unique permutations of {SOURCE}: {len(candidates)}")

filtered = [s for s in candidates if 'AAPV' not in s and s != SOURCE]
print(f"After removing AAPV-motif + original: {len(filtered)}")


def load(name):
    p = MODELS / name
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return joblib.load(p)


lr_hne       = load('lr_NE_panel_b.pkl')
lr_mmp9      = load('lr_MMP_9_panel_b.pkl')
pssm_hne_obj = load('pssm_NE_panel_b.pkl')
pssm_mmp9_obj = load('pssm_MMP_9_panel_b.pkl')


def score_pssm(pssm_obj, seq):
    from src.pssm import PSSM as PSSMClass
    if isinstance(pssm_obj, PSSMClass):
        return float(pssm_obj.score(seq))
    if isinstance(pssm_obj, np.ndarray):
        from src.pssm import AA_IDX
        s = 0.0
        for pos, aa in enumerate(seq):
            if aa not in AA_IDX:
                return -999.0
            s += pssm_obj[pos, AA_IDX[aa]]
        return float(s)
    X = build_feature_matrix([seq], use_esm2=False)
    return float(pssm_obj.predict_proba(X)[0, 1])


print(f"\nScoring {len(filtered)} candidates … ", end='', flush=True)
X = build_feature_matrix(filtered, use_esm2=False)
lr_hne_probs   = lr_hne.predict_proba(X)[:, 1]
lr_mmp9_probs  = lr_mmp9.predict_proba(X)[:, 1]
pssm_hne_sc    = np.array([score_pssm(pssm_hne_obj,  s) for s in filtered])
pssm_mmp9_sc   = np.array([score_pssm(pssm_mmp9_obj, s) for s in filtered])
print("done.")

df = pd.DataFrame({
    'sequence':       filtered,
    'LR_HNE':         lr_hne_probs,
    'PSSM_HNE':       pssm_hne_sc,
    'LR_MMP9':        lr_mmp9_probs,
    'PSSM_MMP9':      pssm_mmp9_sc,
    'combined_score': lr_hne_probs + lr_mmp9_probs,
})
df.sort_values('LR_HNE', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

X_ref         = build_feature_matrix([SOURCE], use_esm2=False)
ref_lr_hne    = float(lr_hne.predict_proba(X_ref)[0, 1])
ref_lr_mmp9   = float(lr_mmp9.predict_proba(X_ref)[0, 1])
ref_pssm_hne  = score_pssm(pssm_hne_obj,  SOURCE)
ref_pssm_mmp9 = score_pssm(pssm_mmp9_obj, SOURCE)

print("\n" + "="*72)
print("REFERENCE  (original HNE substrate)")
print("="*72)
print(f"  Sequence : {SOURCE}")
print(f"  LR_HNE   : {ref_lr_hne:.4f}   (Panel B)")
print(f"  PSSM_HNE : {ref_pssm_hne:.4f}")
print(f"  LR_MMP9  : {ref_lr_mmp9:.4f}   (Panel B)")
print(f"  PSSM_MMP9: {ref_pssm_mmp9:.4f}")

print("\n" + "="*72)
print("TOP 10 LOWEST-SCORING SCRAMBLES  (by LR_HNE, ascending)")
print("="*72)
print(f"{'Rank':<5} {'Sequence':<12} {'LR_HNE':>8} {'PSSM_HNE':>10} "
      f"{'LR_MMP9':>9} {'PSSM_MMP9':>11}")
print("-"*58)
for i, row in df.head(10).iterrows():
    print(f"  {i+1:<3} {row['sequence']:<12} {row['LR_HNE']:>8.4f} "
          f"{row['PSSM_HNE']:>10.4f} {row['LR_MMP9']:>9.4f} "
          f"{row['PSSM_MMP9']:>11.4f}")

winner = df.iloc[0]
fold   = ref_lr_hne / winner['LR_HNE'] if winner['LR_HNE'] > 0 else float('inf')
print("\n" + "="*72)
print("RECOMMENDED NEGATIVE CONTROL")
print("="*72)
print(f"  Sequence  : {winner['sequence']}")
print(f"  LR_HNE    : {winner['LR_HNE']:.4f}   (vs {ref_lr_hne:.4f} for AAPVSGIG"
      f" → {fold:.1f}x reduction)")
print(f"  PSSM_HNE  : {winner['PSSM_HNE']:.4f}")
print(f"  LR_MMP9   : {winner['LR_MMP9']:.4f}   (vs {ref_lr_mmp9:.4f} for AAPVSGIG)")
print(f"  PSSM_MMP9 : {winner['PSSM_MMP9']:.4f}")
print(f"\n  Charge balance: all residues (A×2 P×1 V×1 S×1 G×2 I×1) are neutral")
print(f"  at pH 7.4. Net charge = 0, identical to AAPVSGIG.")

OUT = BASE / 'results' / 'negative_control_scores.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"\n  Full results saved to: {OUT}")
print("="*72)
