"""Training data: positives from verifiable sources, negatives generated deterministically.

Positive P4-P4' windows come from
  * MEROPS Substrate_search (all proteases),
  * Eckhard et al. 2016 PICS cleavage sites (MMP-2/-8/-9/-14), pre-extracted to data/sources/,
  * Yang et al. 2015 pseudolysin cleavage sites in bovine elastin (LasB), extracted from the
    supplementary PDF and mapped onto UniProt P04985 (data/sources/).
"""
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import AA_INDEX, MIN_TRAINABLE_POSITIVES, PROTEASES, RANDOM_SEED, WINDOW
from .tables import AMINO_ACIDS, SWISSPROT_BACKGROUND

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCES = DATA / "sources"
POSITIVES_DIR = DATA / "positives"
FECAL_CAP = 5000

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def is_window(seq: str) -> bool:
    return len(seq) == WINDOW and all(a in AA_INDEX for a in seq)


def merops_file() -> Path:
    for p in (DATA / "raw" / "Substrate_search.txt",
              ROOT.parent / "cross_reactivity_matrix" / "data" / "raw" / "Substrate_search.txt"):
        if p.exists():
            return p
    raise FileNotFoundError("MEROPS Substrate_search.txt not found (data/raw/ or cross_reactivity_matrix/data/raw/)")


def merops_positives(path: Path | None = None) -> dict[str, list[str]]:
    """P4-P4' windows per registered protease from MEROPS (column 2 = peptidase id, 5-12 = P4..P4')."""
    lookup = {m: pid for pid, p in PROTEASES.items() for m in p.merops}
    out = {pid: [] for pid in PROTEASES}
    with open(path or merops_file(), errors="replace") as fh:
        for line in fh:
            c = [x.strip().strip("'") for x in line.rstrip("\n").split("\t")]
            if len(c) < 12 or c[1] not in lookup:
                continue
            aa = [THREE_TO_ONE.get(r) for r in c[4:12]]
            if None in aa:
                continue
            w = "".join(aa)
            if is_window(w):
                out[lookup[c[1]]].append(w)
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def eckhard_positives() -> dict[str, list[str]]:
    out = {}
    for pid in ("MMP2", "MMP8", "MMP9", "MMP14"):
        df = pd.read_csv(SOURCES / f"eckhard2016_{pid}.csv")
        out[pid] = [s for s in df["sequence"].str.upper() if is_window(s)]
    return out


def yang_positives() -> list[str]:
    """Both termini of each pseudolysin-released elastin peptide (each is a cleavage site)."""
    seq = "".join(l.strip() for l in open(SOURCES / "yang2015_P04985_bovine_tropoelastin.fasta")
                  if not l.startswith(">"))
    tab = pd.read_csv(SOURCES / "yang2015_table_s1_peptides.csv")
    out = []
    for r in tab.itertuples():
        # 1-indexed peptide residues start..end; flanks at start-1 and end+1
        if seq[r.start - 1:r.end] != r.peptide or seq[r.start - 2] != r.left_p1 or seq[r.end] != r.right_p1prime:
            raise ValueError(f"Yang table row {r.index} inconsistent with P04985")
        out.append(seq[r.start - 5:r.start + 3])      # P4..P4' around the N-terminal cleavage
        out.append(seq[r.end - 4:r.end + 4])          # P4..P4' around the C-terminal cleavage
    return [w for w in dict.fromkeys(out) if is_window(w)]


def build_positives(seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    merops = merops_positives()
    eck = eckhard_positives()
    out = {}
    for pid in PROTEASES:
        rows = [(s, "MEROPS") for s in merops[pid]]
        rows += [(s, "Eckhard2016") for s in eck.get(pid, [])]
        if pid == "LASB":
            rows += [(s, "Yang2015") for s in yang_positives()]
        df = pd.DataFrame(rows, columns=["sequence", "source"]).drop_duplicates("sequence", keep="first")
        if pid == "FECAL" and len(df) > FECAL_CAP:
            df = df.sample(FECAL_CAP, random_state=seed)
        out[pid] = df.sort_values("sequence").reset_index(drop=True)
    return out


def write_positives(pos: dict[str, pd.DataFrame]) -> None:
    POSITIVES_DIR.mkdir(parents=True, exist_ok=True)
    for pid, df in pos.items():
        df.to_csv(POSITIVES_DIR / f"{pid}.csv", index=False)


def load_positives(min_n: int = 0) -> dict[str, list[str]]:
    out = {}
    for pid in PROTEASES:
        f = POSITIVES_DIR / f"{pid}.csv"
        if f.exists():
            seqs = pd.read_csv(f)["sequence"].tolist()
            if len(seqs) >= min_n:
                out[pid] = seqs
    return out


def trainable_positives() -> dict[str, list[str]]:
    return load_positives(MIN_TRAINABLE_POSITIVES)


def fingerprint(seqs) -> str:
    return hashlib.sha256("\n".join(sorted(seqs)).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------ negatives
def background_sequences(n: int, rng: np.random.Generator, exclude=frozenset()) -> list[str]:
    p = np.array([SWISSPROT_BACKGROUND[a] for a in AMINO_ACIDS])
    out, seen = [], set(exclude)
    while len(out) < n:
        draw = rng.choice(20, size=(max(n - len(out), 64) * 2, WINDOW), p=p)
        for row in draw:
            s = "".join(AMINO_ACIDS[i] for i in row)
            if s not in seen:
                seen.add(s); out.append(s)
                if len(out) == n:
                    break
    return out


def make_negatives(target: str, positives: dict[str, list[str]], mode: str = "panel",
                   ratio: int = 3, cross_fraction: float = 0.5,
                   seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Negatives for one target protease. Columns: sequence, source (competitor id | 'background' | 'shuffled').

    mode='panel'   : half from other proteases' positives (equal share per competitor), half background.
    mode='baseline': composition-shuffled positives (2 per positive) + background (generic cleavability task).
    """
    rng = np.random.default_rng(seed)
    tset = set(positives[target])
    n_neg = ratio * len(tset)
    rows = []
    if mode == "panel":
        comps = [c for c in positives if c != target]
        quota = int(n_neg * cross_fraction) // max(len(comps), 1)
        seen = set(tset)
        for c in comps:
            pool = [s for s in positives[c] if s not in seen]
            take = rng.permutation(len(pool))[:quota]
            for i in take:
                seen.add(pool[i]); rows.append((pool[i], c))
        rows += [(s, "background") for s in background_sequences(n_neg - len(rows), rng, exclude=seen)]
    elif mode == "baseline":
        r = random.Random(seed)
        seen = set(tset)
        for s in sorted(tset):
            for _ in range(2):
                ch = list(s); r.shuffle(ch); t = "".join(ch)
                if t not in seen:
                    seen.add(t); rows.append((t, "shuffled"))
        rows += [(s, "background") for s in background_sequences(max(n_neg - len(rows), 0), rng, exclude=seen)]
    else:
        raise ValueError(mode)
    return pd.DataFrame(rows, columns=["sequence", "source"])


def training_frame(target: str, positives: dict[str, list[str]], mode: str = "panel",
                   seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Positives + negatives with columns sequence, label, source."""
    neg = make_negatives(target, positives, mode=mode, seed=seed)
    pos = pd.DataFrame({"sequence": positives[target], "source": target})
    pos["label"] = 1
    neg["label"] = 0
    return pd.concat([pos, neg], ignore_index=True)[["sequence", "label", "source"]]
