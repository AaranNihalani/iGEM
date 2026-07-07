"""
Task 6: Cross-Reactivity Specificity Matrix -- full pipeline.

Usage:
    python run_all.py                  # download pepunit.lib then run
    python run_all.py --skip-download  # use existing data/raw/pepunit.lib
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

BASE = Path(__file__).parent
RAW_DIR       = BASE / 'data' / 'raw'
PROCESSED_DIR = BASE / 'data' / 'processed'
OUTPUTS_DIR   = BASE / 'data' / 'outputs'
FIGURES_DIR   = BASE / 'figures'
SUBSTRATE     = RAW_DIR / 'Substrate_search.txt'


def main() -> None:
    parser = argparse.ArgumentParser(description='Task 6: Specificity Matrix Pipeline')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip FTP download; use existing Substrate_search.txt')
    args = parser.parse_args()

    # --- Step 1: Download ---
    if not args.skip_download:
        if SUBSTRATE.exists():
            log.info(f"Substrate_search.txt already present at {SUBSTRATE}; skipping download.")
        else:
            log.info("Step 1: Downloading Substrate_search.txt from MEROPS FTP (~29 MB) ...")
            from src.parse_merops import download_substrate_search
            download_substrate_search(SUBSTRATE)
    else:
        if not SUBSTRATE.exists():
            log.error(f"--skip-download set but {SUBSTRATE} does not exist.")
            sys.exit(1)
        log.info(f"Step 1: Using existing {SUBSTRATE}")

    # --- Step 2: Parse cleavage windows ---
    log.info("Step 2: Extracting P4-P4' cleavage windows from Substrate_search.txt ...")
    from src.parse_merops import extract_windows
    windows = extract_windows(SUBSTRATE, PROCESSED_DIR)

    missing = [p for p, df in windows.items() if len(df) == 0]
    if missing:
        log.warning(f"No windows extracted for: {missing}. Check MEROPS IDs or file format.")

    # --- Step 3: Build PSSMs ---
    log.info("Step 3: Building per-protease PSSMs ...")
    from src.build_pssm import build_all_pssms
    pssms = build_all_pssms(PROCESSED_DIR, PROCESSED_DIR)

    if not pssms:
        log.error("No PSSMs built. Cannot continue.")
        sys.exit(1)

    # --- Step 4 & 5: Score motifs, normalise, compute selectivity ratios ---
    log.info("Step 4: Scoring panel motifs and normalising ...")
    from src.score_motifs import run_scoring
    raw, normed = run_scoring(PROCESSED_DIR, OUTPUTS_DIR)

    # --- Step 6: Heatmap ---
    log.info("Step 5: Rendering specificity heatmap ...")
    from src.plot_heatmap import plot_heatmap
    plot_heatmap(normed, FIGURES_DIR / 'fig_specificity_heatmap.png')

    # --- Summary ---
    log.info("=" * 60)
    log.info("Task 6 complete. Outputs:")
    log.info(f"  {OUTPUTS_DIR / 'specificity_matrix_raw.csv'}")
    log.info(f"  {OUTPUTS_DIR / 'specificity_matrix_normalised.csv'}")
    log.info(f"  {FIGURES_DIR / 'fig_specificity_heatmap.png'}")
    log.info("=" * 60)
    log.info("Next: review reports/task6_specificity.md, then update report_task6.tex")
    log.info("      Run: bash generate_report.sh")


if __name__ == '__main__':
    main()
