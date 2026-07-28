"""
Generate ranked substrate shortlists for panel v2 (MMP-2 / MMP-9 / HNE).

Strategy
--------
1. Literature-anchored seeds: start from known high-specificity sequences from
   Kridel 2001 (MMP-9), Wysocki 2014 / Senior 2002 (MMP-2), Tralau 2004 /
   Nakajima 1997 (HNE/NE) plus MEROPS canonical sites.
2. In silico pool: 200,000 random Swiss-Prot-composition 8-mers + substitution
   variants of each seed.
3. Scoring: v2 LR + PSSM for each protease. Within-model percentile ranking
   avoids cross-model probability calibration issues.
4. Selectivity filter:
     - Target percentile ≥ 0.80 (top 20 % of pool for that protease)
     - All off-target percentiles < 0.60
5. Final ranking by harmonic mean of LR prob and PSSM percentile (favours
   candidates both models agree on).

Output: data/panel_v2/shortlist.csv  and shortlist_summary.txt
"""

import sys, logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
log = logging.getLogger(__name__)

BASE      = Path(__file__).parent
MODELS_DIR = BASE / 'models'
OUT_DIR   = BASE / 'data' / 'panel_v2'
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE))
from src.features import build_feature_matrix
from src.pssm import build_pssm_from_positives, STANDARD_AAS, SWISSPROT_BG
from src.utils import SWISSPROT_BG as BG_DICT

RANDOM_SEED = 42
POOL_SIZE   = 50_000

# ---------------------------------------------------------------------------
# Literature-anchored seeds
# ---------------------------------------------------------------------------
SEEDS = {
    'MMP_9': [
        # MEROPS canonical: collagenolytic MMP-9 substrates
        'RPKPVELN', 'GPLGIAGQ', 'GPLGIFGQ', 'LPQGIAGQ',
        'IPVSLRSG', 'RPFSMIMG',
        # Kridel 2001 PICS-derived high-activity 8-mers
        'PLGLFASQ', 'AGLFASQV', 'GPLFASQM',
        # IPRTLTAA scaffold (original, Leu P1')
        'IPRTLTAA', 'IPRTMLAA', 'IPRTFTAA', 'IPRTITAA',
    ],
    'MMP_2': [
        # MEROPS canonical: gelatinase A substrates
        'GPLGIAGQ', 'GPQGIAGQ', 'GPLGIFGQ', 'AGLALPGQ',
        'PVQPVGIS', 'AIPVSLDM',
        # Known high-activity from Eckhard PICS
        'PLGLWARG', 'PLGLIAGQ', 'GLFIAGQG',
    ],
    'NE': [
        # MEROPS canonical: neutrophil elastase substrates
        'AAPVSGIG', 'AAPVSGIA', 'AAPLTMMR',
        # Senior 2002 (AAPV scaffold = classic NE substrate)
        'AAPVSGIG', 'VAPVSGIG', 'AAPLAGIG',
        # Tralau 2004 high-activity sites
        'AAPVAATG', 'AAPVAMTG', 'AAPVAEIG',
        # P1=Ala strict requirement variants
        'AAPVVTIG', 'GAPVSGIG', 'AAPVSIIG',
    ],
}

# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------

def random_pool(n: int, seed: int = RANDOM_SEED) -> list[str]:
    aas   = list(BG_DICT.keys())
    probs = np.array([BG_DICT[aa] for aa in aas], dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(aas), size=(n, 8), p=probs)
    return [''.join(aas[i] for i in row) for row in idxs]


def substitution_variants(seq: str, n_per_pos: int = 5) -> list[str]:
    """Single-AA substitution variants at all positions."""
    aas = STANDARD_AAS
    variants = []
    for pos in range(len(seq)):
        for aa in aas[:n_per_pos]:
            if aa != seq[pos]:
                v = seq[:pos] + aa + seq[pos+1:]
                variants.append(v)
    return variants


def build_candidate_pool() -> list[str]:
    pool = set()
    # Background pool
    pool.update(random_pool(POOL_SIZE))
    # Seed sequences + variants
    for seqs in SEEDS.values():
        for s in seqs:
            pool.add(s)
            pool.update(substitution_variants(s, n_per_pos=20))
    return list(pool)


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_lr(name: str):
    f = MODELS_DIR / f'lr_v2_{name}.pkl'
    if not f.exists():
        f = MODELS_DIR / f'lr_{name}.pkl'   # fall back to v1
        log.warning(f"Using v1 model for {name}")
    return joblib.load(f)


def load_pssm(name: str):
    f = MODELS_DIR / f'pssm_{name}.pkl'
    if f.exists():
        return joblib.load(f)
    return None


# ---------------------------------------------------------------------------
# Score pool
# ---------------------------------------------------------------------------

def score_all(pool: list[str], panel_names: list[str]) -> pd.DataFrame:
    lr_models   = {n: load_lr(n)   for n in panel_names}
    pssm_models = {n: load_pssm(n) for n in panel_names}

    X = build_feature_matrix(pool, use_esm2=False)

    df = pd.DataFrame({'sequence': pool})

    for name in panel_names:
        lr_probs = lr_models[name].predict_proba(X)[:, 1]
        df[f'lr_{name}'] = lr_probs

        pssm = pssm_models.get(name)
        if pssm is not None:
            pssm_raw = pssm.score_batch(pool)
            df[f'pssm_{name}'] = pssm_raw
        else:
            df[f'pssm_{name}'] = np.nan

    return df


def add_percentile_ranks(df: pd.DataFrame, panel_names: list[str]) -> pd.DataFrame:
    for name in panel_names:
        col = f'lr_{name}'
        vals = df[col].values
        df[f'pctile_lr_{name}'] = (vals[:, None] > vals[None, :]).mean(axis=1)

        pcol = f'pssm_{name}'
        if df[pcol].notna().any():
            pvals = df[pcol].values
            valid = ~np.isnan(pvals)
            ranks = np.full(len(pvals), np.nan)
            ranks[valid] = (pvals[valid, None] > pvals[valid][None, :]).mean(axis=1)
            df[f'pctile_pssm_{name}'] = ranks
    return df


# ---------------------------------------------------------------------------
# Shortlist extraction
# ---------------------------------------------------------------------------

SELECTIVITY_THRESHOLDS = {
    'MMP_9': {
        'target_pctile': 0.80,
        'off_target_pctile': 0.60,
        'off_targets': ['MMP_2', 'NE'],
    },
    'MMP_2': {
        'target_pctile': 0.80,
        'off_target_pctile': 0.60,
        'off_targets': ['MMP_9', 'NE'],
    },
    'NE': {
        'target_pctile': 0.80,
        'off_target_pctile': 0.60,
        'off_targets': ['MMP_2', 'MMP_9'],
    },
}


def extract_shortlist(df: pd.DataFrame, target: str,
                       n: int = 10) -> pd.DataFrame:
    cfg = SELECTIVITY_THRESHOLDS[target]
    mask = df[f'pctile_lr_{target}'] >= cfg['target_pctile']
    for ot in cfg['off_targets']:
        mask &= df[f'pctile_lr_{ot}'] < cfg['off_target_pctile']

    sel = df[mask].copy()
    if len(sel) == 0:
        log.warning(f"No candidates pass selectivity filter for {target}; relaxing")
        mask = df[f'pctile_lr_{target}'] >= 0.70
        for ot in cfg['off_targets']:
            mask &= df[f'pctile_lr_{ot}'] < 0.70
        sel = df[mask].copy()

    # Ranking score: harmonic mean of LR percentile and PSSM percentile
    pssm_col = f'pctile_pssm_{target}'
    lr_col   = f'pctile_lr_{target}'
    if sel[pssm_col].notna().any():
        sel['rank_score'] = 2 * sel[lr_col] * sel[pssm_col] / (sel[lr_col] + sel[pssm_col] + 1e-9)
    else:
        sel['rank_score'] = sel[lr_col]

    sel = sel.sort_values('rank_score', ascending=False).drop_duplicates('sequence')
    return sel.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    panel = ['MMP_2', 'MMP_9', 'NE']

    log.info("Building candidate pool...")
    pool = build_candidate_pool()
    log.info(f"Pool size: {len(pool):,}")

    log.info("Scoring all candidates against panel v2 models...")
    df = score_all(pool, panel)

    log.info("Computing within-model percentile ranks...")
    df = add_percentile_ranks(df, panel)

    all_shortlists = []
    summary_lines  = []

    for target in panel:
        label = {'MMP_2': 'MMP-2', 'MMP_9': 'MMP-9', 'NE': 'HNE/NE'}[target]
        short = extract_shortlist(df, target, n=10)
        short.insert(0, 'protease', label)
        all_shortlists.append(short)

        header = f"\n{'='*65}\n{label} shortlist\n{'='*65}"
        log.info(header)
        summary_lines.append(header)

        for _, row in short.iterrows():
            line = (f"  {row['sequence']}  "
                    f"LR={row[f'lr_{target}']:.3f} ({row[f'pctile_lr_{target}']:.0%})  "
                    f"PSSM-pctile={row.get(f'pctile_pssm_{target}', float('nan')):.0%}  "
                    f"rank={row['rank_score']:.3f}")
            # Show off-target scores too
            off_parts = []
            for ot in SELECTIVITY_THRESHOLDS[target]['off_targets']:
                ot_label = {'MMP_2': 'MMP-2', 'MMP_9': 'MMP-9', 'NE': 'NE'}[ot]
                off_parts.append(f"{ot_label}={row[f'lr_{ot}']:.3f}")
            line += f"  [{', '.join(off_parts)}]"
            log.info(line)
            summary_lines.append(line)

    combined = pd.concat(all_shortlists, ignore_index=True)
    combined.to_csv(OUT_DIR / 'shortlist.csv', index=False)
    (OUT_DIR / 'shortlist_summary.txt').write_text('\n'.join(summary_lines))
    log.info(f"\nSaved shortlist to {OUT_DIR / 'shortlist.csv'}")

    return combined


if __name__ == '__main__':
    main()
