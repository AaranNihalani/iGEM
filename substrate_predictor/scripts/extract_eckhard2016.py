"""
Parse Eckhard 2016 MMP PICS data → per-MMP lists of P4-P4' 8-mer windows.

Strategy:
  1. Read each MMP-X sheet from mmc4.xlsx (tryptic) and mmc5.xlsx (GluC).
  2. Parse the 'peptide' column to extract P1 (prefix char) and prime-side seq.
  3. Collect unique UniProt accessions, batch-fetch sequences from UniProt REST.
  4. For each cleavage site, locate the prime-side peptide in the protein sequence
     to find P1' position, then extract P4-P4' (8 chars).
  5. Write one CSV per MMP to this directory.

Output: eckhard_{MMP}_positives.csv with column 'sequence' (8-mer, uppercase).

Requires network access (UniProt). Writes eckhard_*_positives.csv into data/raw/eckhard2016/;
copy them to data/sources/eckhard2016_<PROTEASE>.csv.
    python scripts/extract_eckhard2016.py
"""

import re
import time
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'eckhard2016'

TRYPTIC_FILE = HERE / '1-s2.0-S0945053X15001298-mmc4.xlsx'
GLUC_FILE    = HERE / '1-s2.0-S0945053X15001298-mmc5.xlsx'

PROTEASE_SHEETS = {
    'MMP9-T':  ('MMP-9',  TRYPTIC_FILE),
    'MMP9-G':  ('MMP-9',  GLUC_FILE),
    'MMP8-T':  ('MMP-8',  TRYPTIC_FILE),
    'MMP8-G':  ('MMP-8',  GLUC_FILE),
    'MMP2-T':  ('MMP-2',  TRYPTIC_FILE),
    'MMP2-G':  ('MMP-2',  GLUC_FILE),
    'MMP14-T': ('MMP14',  TRYPTIC_FILE),
    'MMP14-G': ('MMP14',  GLUC_FILE),
}

UNIPROT_REST = 'https://rest.uniprot.org/uniprotkb/{acc}.fasta'
RATE_LIMIT_S = 0.15


# ---------------------------------------------------------------------------
# Peptide column parsing
# ---------------------------------------------------------------------------

def strip_mods(seq: str) -> str:
    seq = re.sub(r'n\[[\d.]+\]', '', seq)
    seq = re.sub(r'([A-Z])\[[\d.]+\]', r'\1', seq)
    return seq


def parse_peptide_col(raw: str):
    if not isinstance(raw, str):
        return None, None
    raw = raw.strip()
    # Format: X.PEPTIDE.Y  but PEPTIDE can contain '.' inside mod brackets
    # Split on FIRST and LAST dot only
    first = raw.find('.')
    last = raw.rfind('.')
    if first == -1 or last == first:
        return None, None
    prefix = raw[:first]
    middle = raw[first + 1: last]
    if len(prefix) != 1 or not prefix.isalpha():
        return None, None
    p1 = prefix.upper()
    clean = strip_mods(middle).upper()
    if not clean or not all(c.isalpha() for c in clean):
        return None, None
    return p1, clean


# ---------------------------------------------------------------------------
# UniProt fetcher
# ---------------------------------------------------------------------------

_seq_cache: dict[str, str] = {}


def fetch_sequence(accession: str) -> str | None:
    if accession in _seq_cache:
        return _seq_cache[accession]
    try:
        r = requests.get(UNIPROT_REST.format(acc=accession), timeout=10)
        if r.status_code != 200:
            _seq_cache[accession] = None
            return None
        lines = r.text.splitlines()
        seq = ''.join(l for l in lines if not l.startswith('>'))
        _seq_cache[accession] = seq
        return seq
    except Exception:
        _seq_cache[accession] = None
        return None


def extract_accession(protein_col: str) -> str | None:
    if not isinstance(protein_col, str):
        return None
    first = protein_col.split(',')[0].strip()
    parts = first.split('|')
    if len(parts) >= 2:
        return parts[1]
    return first.strip() or None


# ---------------------------------------------------------------------------
# 8-mer extraction
# ---------------------------------------------------------------------------

def find_8mer(protein_seq: str, p1: str, prime_seq: str) -> str | None:
    probe = (p1 + prime_seq[:min(6, len(prime_seq))]).upper()
    pos = protein_seq.upper().find(probe)
    if pos == -1:
        return None
    p1_pos = pos
    if p1_pos < 3:
        return None
    end_idx = p1_pos + 5  # P1 + P1'..P4' = 5 chars
    if end_idx > len(protein_seq):
        return None
    mer = protein_seq[p1_pos - 3: p1_pos + 1] + protein_seq[p1_pos + 1: p1_pos + 5]
    if len(mer) != 8 or not all(c.isalpha() for c in mer):
        return None
    return mer.upper()


# ---------------------------------------------------------------------------
# Process one sheet
# ---------------------------------------------------------------------------

def process_sheet(sheet_name: str, filepath: Path) -> list[str]:
    log.info(f"Reading '{sheet_name}' from {filepath.name}")
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

    header_row = None
    for i, row in df.iterrows():
        vals = [str(v).strip().lower() for v in row]
        if 'peptide' in vals and 'protein' in vals:
            header_row = i
            break
    if header_row is None:
        log.warning(f"  No header found in {sheet_name}")
        return []

    df.columns = df.iloc[header_row].str.strip().str.lower()
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df = df.dropna(subset=['peptide', 'protein'])
    log.info(f"  {len(df)} rows")

    accessions = [extract_accession(str(p)) for p in df['protein']]
    unique_acc = [a for a in set(accessions) if a]

    log.info(f"  Fetching {len(unique_acc)} UniProt sequences...")
    for acc in tqdm(unique_acc, desc=f"  UniProt [{sheet_name}]", unit="seq"):
        if acc not in _seq_cache:
            fetch_sequence(acc)
            time.sleep(RATE_LIMIT_S)

    mers, skipped = [], 0
    for row_acc, pep_raw in tqdm(
        zip(accessions, df['peptide']),
        total=len(df),
        desc=f"  8-mers  [{sheet_name}]",
        unit="site",
    ):
        p1, prime = parse_peptide_col(str(pep_raw))
        if p1 is None or len(prime) < 4:
            skipped += 1
            continue
        seq = _seq_cache.get(row_acc) if row_acc else None
        if not seq:
            skipped += 1
            continue
        mer = find_8mer(seq, p1, prime)
        if mer:
            mers.append(mer)
        else:
            skipped += 1

    log.info(f"  → {len(mers)} 8-mers extracted, {skipped} skipped")
    return mers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    protease_mers: dict[str, list[str]] = defaultdict(list)

    for sheet, (protease, filepath) in PROTEASE_SHEETS.items():
        if not filepath.exists():
            log.warning(f"File not found: {filepath}")
            continue
        mers = process_sheet(sheet, filepath)
        protease_mers[protease].extend(mers)

    print()
    for protease, mers in protease_mers.items():
        unique = list(dict.fromkeys(mers))
        out = HERE / f'eckhard_{protease.replace("-", "_")}_positives.csv'
        pd.DataFrame({'sequence': unique}).to_csv(out, index=False)
        log.info(f"{protease}: {len(unique)} unique 8-mers → {out.name}")

    log.info("Done.")


if __name__ == '__main__':
    main()
