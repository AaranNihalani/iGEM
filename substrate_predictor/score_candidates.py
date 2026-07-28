"""
Score user candidate substrates against MMP-9, NE (HNE), and LasB classifiers.

Substrate windows (P4-P4' = 8 residues, cleavage always between pos 4 and 5):

MMP-9  Ac-SGKIPRT|LTA-NH2  → window IPRTLTAA  (P4'=A pad; user T|L bond)
HNE    Ac-AAPV|SGIG-NH2    → window AAPVSGIG  (exact)
LasB   Ac-AAGLAAGG-NH2     → window AAAGLAAG  (P4=A pad; G|L bond)

Cross-reactivity is scored for all 3 sequences against all 3 protease models.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
log = logging.getLogger(__name__)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import numpy as np
import joblib
from src.features import build_feature_matrix

CANDIDATES = {
    'MMP-9_substrate':  'IPRTLTAA',   # Ac-SGKIPRT|LTA-NH2, pad P4'=A
    'HNE_substrate':    'AAPVSGIG',   # Ac-AAPV|SGIG-NH2, exact
    'LasB_substrate':   'AAAGLAAG',   # Ac-AAGLAAGG-NH2, pad P4=A (G|L bond)
}

MODELS = {
    'MMP-9': 'models/lr_MMP_9.pkl',
    'HNE':   'models/lr_NE.pkl',
    'LasB':  'models/lr_LasB.pkl',
}

CONFIDENCE = {
    'MMP-9': 'HIGH',
    'HNE':   'HIGH',
    'LasB':  'MEDIUM',
}


def load_model(path: Path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        log.warning(f"Model not found: {path}")
        return None


def score(model, seq: str) -> float:
    X = build_feature_matrix([seq], use_esm2=False)
    return float(model.predict_proba(X)[0, 1])


def main():
    models = {}
    for name, rel_path in MODELS.items():
        m = load_model(BASE / rel_path)
        if m is not None:
            models[name] = m
        else:
            log.error(f"Could not load {name} model — skipping.")

    if not models:
        log.error("No models loaded. Run the training pipeline first.")
        return

    seqs = list(CANDIDATES.values())
    labels = list(CANDIDATES.keys())
    X = build_feature_matrix(seqs, use_esm2=False)

    header = f"\n{'Substrate':<20} {'8-mer':<12} " + "  ".join(f"{p:<12}" for p in models)
    print(header)
    print("-" * (20 + 12 + 14 * len(models)))

    results = {}
    for label, seq in CANDIDATES.items():
        row = {}
        for prot, model in models.items():
            prob = score(model, seq)
            row[prot] = prob
        results[label] = row

        cols = "  ".join(
            f"{row[p]:.3f}{' !!WARN' if row[p] > 0.5 else '':<7}"
            for p in models
        )
        print(f"{label:<20} {seq:<12} {cols}")

    print()
    print("Model confidence levels:")
    for prot, conf in CONFIDENCE.items():
        if prot in models:
            print(f"  {prot:<8} {conf}")

    print()
    print("CROSS-REACTIVITY ASSESSMENT")
    print("=" * 60)
    for label, seq in CANDIDATES.items():
        target = label.split('_')[0]  # MMP-9, HNE, LasB
        row = results[label]
        target_score = row.get(target, None)
        cross = {p: s for p, s in row.items() if p != target}
        print(f"\n{label} ({seq}):")
        if target_score is not None:
            flag = "  <- intended target" + (" [HIGH SCORE ✓]" if target_score > 0.5 else " [LOW SCORE ✗]")
            print(f"  {target:<8}: {target_score:.3f}{flag}")
        for p, s in cross.items():
            flag = "  <- CROSS-REACTIVE RISK" if s > 0.4 else ""
            print(f"  {p:<8}: {s:.3f}{flag}")

    print()
    print("ORDERING RECOMMENDATION")
    print("=" * 60)
    for label, seq in CANDIDATES.items():
        target = label.split('_')[0]
        row = results[label]
        target_s = row.get(target, 0)
        cross_scores = [(p, s) for p, s in row.items() if p != target]
        max_cross = max(cross_scores, key=lambda x: x[1])
        selectivity = target_s - max_cross[1]
        if selectivity >= 0.2:
            verdict = "ORDER — adequate selectivity"
        elif selectivity >= 0.05:
            verdict = "BORDERLINE — consider alternative"
        else:
            verdict = "CAUTION — poor selectivity in silico"
        print(f"\n{label}:")
        print(f"  Target score:         {target_s:.3f}")
        print(f"  Max cross-reactivity: {max_cross[1]:.3f} ({max_cross[0]})")
        print(f"  Selectivity margin:   {selectivity:+.3f}")
        print(f"  Verdict:              {verdict}")


if __name__ == '__main__':
    main()
