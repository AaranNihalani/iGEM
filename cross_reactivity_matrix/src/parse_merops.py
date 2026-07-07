"""
Parse MEROPS Substrate_search.txt and extract P4-P4' cleavage windows.

File format: tab-separated, one row per confirmed cleavage event.
Relevant columns (0-indexed):
  0  : Cleavage ID (e.g. 'CLE0166506')
  1  : MEROPS protease ID (e.g. 'M10.014')
  2  : Substrate name
  3  : Cleavage description
  4  : P4  (3-letter AA code, e.g. 'Arg')
  5  : P3
  6  : P2
  7  : P1
  8  : P1'
  9  : P2'
  10 : P3'
  11 : P4'
  12+: Reference, UniProt ID, position, species, protease name, ...

Values are single-quoted strings. Non-standard residues (modified AAs, '-', NULL)
are excluded.
"""

import logging
import sys
from ftplib import FTP
from pathlib import Path

import pandas as pd

from src.utils import MEROPS_IDS, PROTEASE_ORDER, WINDOW_LEN, is_valid_window

log = logging.getLogger(__name__)

MEROPS_FTP_HOST = 'ftp.ebi.ac.uk'
SUBSTRATE_FTP_PATH = '/pub/databases/merops/current_release/Substrate_search.txt'

# Three-letter to one-letter amino acid code conversion
THREE_TO_ONE = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}


def download_substrate_search(dest: Path) -> None:
    """Download Substrate_search.txt from the MEROPS FTP server."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Connecting to {MEROPS_FTP_HOST} ...")
    try:
        ftp = FTP(MEROPS_FTP_HOST, timeout=120)
        ftp.login()
        log.info(f"Downloading {SUBSTRATE_FTP_PATH} (~29 MB) -> {dest}")
        with open(dest, 'wb') as f:
            ftp.retrbinary(f'RETR {SUBSTRATE_FTP_PATH}', f.write)
        ftp.quit()
        log.info("Download complete.")
    except Exception as e:
        log.error(f"FTP download failed: {e}")
        log.error("Please download Substrate_search.txt manually from:")
        log.error(f"  ftp://{MEROPS_FTP_HOST}{SUBSTRATE_FTP_PATH}")
        log.error(f"and place it at: {dest}")
        sys.exit(1)


def _parse_row(cols: list[str]) -> str | None:
    """
    Extract and validate an 8-mer window from a parsed row.
    Columns 4-11 are P4..P4' as 3-letter codes.
    Returns a 1-letter 8-mer string, or None if any position is non-standard.
    """
    if len(cols) < 12:
        return None
    residues = cols[4:12]
    one_letter = []
    for r in residues:
        aa = THREE_TO_ONE.get(r)
        if aa is None:
            return None
        one_letter.append(aa)
    window = ''.join(one_letter)
    return window if is_valid_window(window) else None


def extract_windows(substrate_path: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Parse Substrate_search.txt and write one CSV per protease to output_dir.

    Returns a dict mapping protease name -> DataFrame with columns:
        merops_id, cleavage_id, substrate, sequence
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build reverse lookup: MEROPS ID -> protease name
    mer_to_protease: dict[str, str] = {}
    for protease, mer_ids in MEROPS_IDS.items():
        for mid in mer_ids:
            mer_to_protease[mid] = protease

    buckets: dict[str, list] = {p: [] for p in MEROPS_IDS}
    total = skipped = 0

    log.info(f"Parsing {substrate_path} ...")
    with open(substrate_path, 'r', errors='replace') as fh:
        for line in fh:
            cols = [c.strip().strip("'") for c in line.rstrip('\n').split('\t')]
            if len(cols) < 12:
                continue
            total += 1
            mer_id = cols[1]
            if mer_id not in mer_to_protease:
                continue

            window = _parse_row(cols)
            if window is None:
                skipped += 1
                continue

            protease = mer_to_protease[mer_id]
            buckets[protease].append({
                'merops_id':  mer_id,
                'cleavage_id': cols[0],
                'substrate':  cols[2],
                'sequence':   window,
            })

    log.info(f"Scanned {total} rows total.")

    results = {}
    for protease in PROTEASE_ORDER:
        rows = buckets[protease]
        df = pd.DataFrame(rows, columns=['merops_id', 'cleavage_id', 'substrate', 'sequence'])
        df = df.drop_duplicates(subset='sequence')
        out_path = output_dir / f"cleavage_windows_{protease.replace('-', '_')}.csv"
        df.to_csv(out_path, index=False)

        n = len(df)
        flag = ''
        if n < 20:
            flag = ' *** CRITICALLY LOW DATA'
        elif n < 50:
            flag = ' *** LOW DATA'
        log.info(f"  {protease:12s}: {n:5d} unique windows -> {out_path.name}{flag}")

        if protease == 'FAP' and n < 20:
            log.warning(
                "FAP has <20 unique windows: PSSM will be unreliable. "
                "The FAP row in the final matrix should be treated as indicative only."
            )

        results[protease] = df

    return results


if __name__ == '__main__':
    base = Path(__file__).parent.parent
    substrate = base / 'data' / 'raw' / 'Substrate_search.txt'
    if not substrate.exists():
        download_substrate_search(substrate)
    extract_windows(substrate, base / 'data' / 'processed')
