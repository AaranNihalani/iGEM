"""
Build per-protease training sets from MEROPS Substrate_search.txt.

Positives: confirmed P4-P4' cleavage windows (label=1).
Negatives: three sources combined to a 1:3 pos:neg ratio:
  1. Shuffled positives (2 shuffles per positive) -- controls composition bias
  2. Random 8-mers sampled from Swiss-Prot background AA frequencies -- easy negatives

Hard negatives from the same substrate proteins (requiring UniProt API) are
not implemented in the default pipeline to avoid network dependency; the two
sources above are sufficient for a working classifier.

SMOTE is applied if the effective positive rate < 1:5 after combining sources.
"""

import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import (
    MEROPS_IDS, PROTEASE_ORDER, WINDOW_LEN, THREE_TO_ONE,
    SWISSPROT_BG, STANDARD_AAS, is_valid_window, confidence_tier,
)

log = logging.getLogger(__name__)

# Cap Fecal positives to avoid training imbalance vs. other proteases
FECAL_CAP = 5000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Locate Substrate_search.txt
# ---------------------------------------------------------------------------

def find_substrate_file(base: Path) -> Path:
    """
    Look for Substrate_search.txt in order:
      1. base/data/raw/Substrate_search.txt
      2. ../cross_reactivity_matrix/data/raw/Substrate_search.txt
    """
    local = base / 'data' / 'raw' / 'Substrate_search.txt'
    if local.exists():
        return local
    sibling = base.parent / 'cross_reactivity_matrix' / 'data' / 'raw' / 'Substrate_search.txt'
    if sibling.exists():
        log.info(f"Using Substrate_search.txt from Task 6: {sibling}")
        return sibling
    raise FileNotFoundError(
        "Substrate_search.txt not found. Run Task 6 first, or place the file at "
        f"{local}"
    )


# ---------------------------------------------------------------------------
# Extract positives
# ---------------------------------------------------------------------------

def extract_positives(substrate_path: Path) -> dict[str, list[str]]:
    """Return dict protease -> list of unique 8-mer positive windows."""
    mer_to_protease: dict[str, str] = {}
    for protease, ids in MEROPS_IDS.items():
        for mid in ids:
            mer_to_protease[mid] = protease

    buckets: dict[str, list[str]] = {p: [] for p in PROTEASE_ORDER}

    with open(substrate_path, 'r', errors='replace') as fh:
        for line in fh:
            cols = [c.strip().strip("'") for c in line.rstrip('\n').split('\t')]
            if len(cols) < 12:
                continue
            mer_id = cols[1]
            if mer_id not in mer_to_protease:
                continue
            residues = cols[4:12]
            one_letter = [THREE_TO_ONE.get(r) for r in residues]
            if None in one_letter:
                continue
            window = ''.join(one_letter)
            if is_valid_window(window):
                buckets[mer_to_protease[mer_id]].append(window)

    # Deduplicate
    return {p: list(dict.fromkeys(seqs)) for p, seqs in buckets.items()}


# ---------------------------------------------------------------------------
# Generate negatives
# ---------------------------------------------------------------------------

def shuffled_negatives(positives: list[str], n_per_positive: int = 2,
                       seed: int = RANDOM_SEED) -> list[str]:
    """Generate shuffled versions of each positive (preserves AA composition)."""
    rng = random.Random(seed)
    negatives = []
    for seq in positives:
        for _ in range(n_per_positive):
            chars = list(seq)
            rng.shuffle(chars)
            neg = ''.join(chars)
            if neg != seq:
                negatives.append(neg)
    return negatives


def random_background_negatives(n: int, seed: int = RANDOM_SEED) -> list[str]:
    """
    Sample n random 8-mers from Swiss-Prot background AA frequencies.
    These are easy negatives that anchor the negative class decision boundary.
    """
    rng = np.random.default_rng(seed)
    aas = STANDARD_AAS
    probs = np.array([SWISSPROT_BG[aa] for aa in aas])
    probs /= probs.sum()
    seqs = []
    while len(seqs) < n:
        indices = rng.choice(len(aas), size=WINDOW_LEN, p=probs)
        seq = ''.join(aas[i] for i in indices)
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Build training set
# ---------------------------------------------------------------------------

def build_training_set(positives: list[str],
                       target_ratio: float = 3.0,
                       seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Combine positives and negatives into a balanced DataFrame.
    Columns: sequence, label, source
    """
    n_pos = len(positives)
    n_neg_target = int(n_pos * target_ratio)

    # Source 1: shuffled positives (up to 2 per positive)
    shuffled = shuffled_negatives(positives, n_per_positive=2, seed=seed)
    shuffled = list(dict.fromkeys(shuffled))  # deduplicate

    # Source 2: random background (fill remainder)
    n_random = max(0, n_neg_target - len(shuffled))
    random_bg = random_background_negatives(n_random, seed=seed + 1)

    all_negatives = shuffled + random_bg

    # Trim to target ratio
    rng = random.Random(seed)
    if len(all_negatives) > n_neg_target:
        all_negatives = rng.sample(all_negatives, n_neg_target)

    pos_rows = [{'sequence': s, 'label': 1, 'source': 'positive'} for s in positives]
    neg_rows = []
    for seq in all_negatives[:len(shuffled)]:
        neg_rows.append({'sequence': seq, 'label': 0, 'source': 'shuffled'})
    for seq in all_negatives[len(shuffled):]:
        neg_rows.append({'sequence': seq, 'label': 0, 'source': 'random_bg'})

    df = pd.DataFrame(pos_rows + neg_rows)
    df = df.drop_duplicates(subset='sequence').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# SMOTE application
# ---------------------------------------------------------------------------

def apply_smote_if_needed(X: np.ndarray, y: np.ndarray,
                          threshold_ratio: float = 5.0,
                          seed: int = RANDOM_SEED):
    """Apply SMOTE if positive:negative ratio exceeds threshold_ratio."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_neg == 0 or (n_neg / max(n_pos, 1)) <= threshold_ratio:
        return X, y

    log.info(f"  Applying SMOTE (pos={n_pos}, neg={n_neg}, ratio={n_neg/n_pos:.1f}:1)")
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=seed, k_neighbors=min(5, n_pos - 1))
        X_res, y_res = sm.fit_resample(X, y)
        log.info(f"  After SMOTE: pos={y_res.sum()}, neg={len(y_res)-y_res.sum()}")
        return X_res, y_res
    except ImportError:
        log.warning("imbalanced-learn not installed; skipping SMOTE.")
        return X, y


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_all_training_sets(base: Path,
                             processed_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Build and save training set CSVs for all proteases.
    Returns dict protease -> DataFrame.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    substrate_path = find_substrate_file(base)

    log.info(f"Extracting positives from {substrate_path.name} ...")
    all_positives = extract_positives(substrate_path)

    datasets = {}
    for protease in PROTEASE_ORDER:
        positives = all_positives[protease]
        n = len(positives)

        if protease == 'Fecal' and n > FECAL_CAP:
            rng = random.Random(RANDOM_SEED)
            positives = rng.sample(positives, FECAL_CAP)
            log.info(f"  Fecal: capped positives from {n} to {FECAL_CAP}")
            n = FECAL_CAP

        tier = confidence_tier(n)
        flag = f' [{tier.upper()} CONFIDENCE]' if tier != 'high' else ''
        log.info(f"  {protease:12s}: {n:5d} positives{flag}")

        if n == 0:
            log.warning(f"  {protease}: 0 positives -- skipping training set.")
            continue

        df = build_training_set(positives)
        out_path = processed_dir / f"training_set_{protease.replace('-', '_')}.csv"
        df.to_csv(out_path, index=False)
        datasets[protease] = df

    return datasets


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    base = Path(__file__).parent.parent
    build_all_training_sets(base, base / 'data' / 'processed')
