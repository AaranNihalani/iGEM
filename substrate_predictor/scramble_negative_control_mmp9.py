"""
scramble_negative_control_mmp9.py
==================================
Generate a scrambled negative control for the MMP-9 probe substrate A11
(Ac-SGKIPRT|LTA-NH2, ML window IPRTLTAA, P4'=A pad).

Strategy (mirrors scramble_negative_control.py, adapted for MMP-9/A11)
-----------------------------------------------------------------------
* Enumerate all unique permutations of {I, P, R, T, L, T, A, A} (same amino
  acid composition -> matched mass/hydrophobicity/net charge at pH 7.4).
* Remove sequences that recreate the T-L (P1-P1') scissile junction anywhere
  (the critical MMP-9 cleavage determinant per Kridel 2001: small P1 / bulky
  hydrophobic P1'), and the original sequence itself.
* Score every remaining candidate with the Panel B / v3 models:
    - LR Panel B (MMP-9):   lr_MMP_9_panel_b.pkl
    - PSSM Panel B (MMP-9): pssm_MMP_9_panel_b.pkl
    - LR Panel B (HNE):     lr_NE_panel_b.pkl      (cross-check, 3-plex panel)
    - LR v3 (LasB):         lr_LasB_v3.pkl         (cross-check, 3-plex panel)
* Rank by LR_MMP9 ascending; report top 10 and the recommended winner.

Run from the substrate_predictor directory:
    python scramble_negative_control_mmp9.py
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

SOURCE = 'IPRTLTAA'
assert len(SOURCE) == 8

all_perms  = set(itertools.permutations(SOURCE))
candidates = [''.join(p) for p in all_perms]
print(f"Total unique permutations of {SOURCE}: {len(candidates)}")

filtered = [s for s in candidates if 'TL' not in s and s != SOURCE]
print(f"After removing T-L scissile-junction + original: {len(filtered)}")


def load(name):
    p = MODELS / name
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return joblib.load(p)


lr_mmp9       = load('lr_MMP_9_panel_b.pkl')
pssm_mmp9_obj = load('pssm_MMP_9_panel_b.pkl')
lr_hne        = load('lr_NE_panel_b.pkl')
lr_lasb       = load('lr_LasB_v3.pkl')


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
lr_mmp9_probs  = lr_mmp9.predict_proba(X)[:, 1]
pssm_mmp9_sc   = np.array([score_pssm(pssm_mmp9_obj, s) for s in filtered])
lr_hne_probs   = lr_hne.predict_proba(X)[:, 1]
lr_lasb_probs  = lr_lasb.predict_proba(X)[:, 1]
print("done.")

df = pd.DataFrame({
    'sequence':       filtered,
    'LR_MMP9':        lr_mmp9_probs,
    'PSSM_MMP9':      pssm_mmp9_sc,
    'LR_HNE':         lr_hne_probs,
    'LR_LasB':        lr_lasb_probs,
    'combined_score': lr_mmp9_probs + lr_hne_probs + lr_lasb_probs,
})
df.sort_values('LR_MMP9', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

X_ref          = build_feature_matrix([SOURCE], use_esm2=False)
ref_lr_mmp9    = float(lr_mmp9.predict_proba(X_ref)[0, 1])
ref_pssm_mmp9  = score_pssm(pssm_mmp9_obj, SOURCE)
ref_lr_hne     = float(lr_hne.predict_proba(X_ref)[0, 1])
ref_lr_lasb    = float(lr_lasb.predict_proba(X_ref)[0, 1])

print("\n" + "="*72)
print("REFERENCE  (original MMP-9 substrate A11 / IPRTLTAA)")
print("="*72)
print(f"  Sequence : {SOURCE}")
print(f"  LR_MMP9  : {ref_lr_mmp9:.4f}   (Panel B)")
print(f"  PSSM_MMP9: {ref_pssm_mmp9:.4f}")
print(f"  LR_HNE   : {ref_lr_hne:.4f}")
print(f"  LR_LasB  : {ref_lr_lasb:.4f}")

print("\n" + "="*72)
print("TOP 10 LOWEST-SCORING SCRAMBLES  (by LR_MMP9, ascending)")
print("="*72)
print(f"{'Rank':<5} {'Sequence':<12} {'LR_MMP9':>9} {'PSSM_MMP9':>11} "
      f"{'LR_HNE':>8} {'LR_LasB':>9}")
print("-"*58)
for i, row in df.head(10).iterrows():
    print(f"  {i+1:<3} {row['sequence']:<12} {row['LR_MMP9']:>9.4f} "
          f"{row['PSSM_MMP9']:>11.4f} {row['LR_HNE']:>8.4f} "
          f"{row['LR_LasB']:>9.4f}")

winner = df.iloc[0]
fold   = ref_lr_mmp9 / winner['LR_MMP9'] if winner['LR_MMP9'] > 0 else float('inf')
print("\n" + "="*72)
print("RECOMMENDED NEGATIVE CONTROL")
print("="*72)
print(f"  Sequence  : {winner['sequence']}")
print(f"  LR_MMP9   : {winner['LR_MMP9']:.4f}   (vs {ref_lr_mmp9:.4f} for IPRTLTAA"
      f" → {fold:.1f}x reduction)")
print(f"  PSSM_MMP9 : {winner['PSSM_MMP9']:.4f}")
print(f"  LR_HNE    : {winner['LR_HNE']:.4f}   (vs {ref_lr_hne:.4f} for IPRTLTAA)")
print(f"  LR_LasB   : {winner['LR_LasB']:.4f}   (vs {ref_lr_lasb:.4f} for IPRTLTAA)")
print(f"\n  Composition preserved: I×1 P×1 R×1 T×2 L×1 A×2 — identical mass and")
print(f"  net charge (~+1, from Arg) to IPRTLTAA at pH 7.4.")

OUT = BASE / 'results' / 'negative_control_scores_mmp9.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"\n  Full results saved to: {OUT}")
print("="*72)
