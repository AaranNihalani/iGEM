"""
Task 8 substrate design pipeline.

Usage:
    python run_all.py               # full pipeline
    python run_all.py --skip-scoring      # skip candidate scoring
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
REPO = BASE.parent

PREDICTOR_DIR   = REPO / 'substrate_predictor'

OUTPUT_DIR      = BASE / 'data' / 'outputs'
FIGURES_DIR     = BASE / 'figures'


def main():
    parser = argparse.ArgumentParser(description='Task 8: 3-plex substrate design pipeline')
    parser.add_argument('--skip-scoring', action='store_true',
                        help='Skip candidate scoring step')
    args = parser.parse_args()

    # Insert in reverse priority order so BASE ends up at index 0 (highest priority).
    sys.path.insert(0, str(PREDICTOR_DIR))
    sys.path.insert(0, str(BASE))

    # ------------------------------------------------------------------
    # Step 1: Electrochemical filter — apply to Stage 1 candidates
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Step 1: Electrochemical filter")
    log.info("=" * 60)
    from src.echem_filter import filter_candidates, pass_rate_report
    from src.score_candidates import CANDIDATES

    all_seqs = [info['seq'] for info in CANDIDATES.values()]
    echem_df = filter_candidates(all_seqs)
    pass_rate_report(echem_df)

    log.info("\nPer-candidate filter results:")
    for _, row in echem_df.iterrows():
        status = 'PASS' if row['passed'] else 'FAIL'
        log.info(f"  [{status}] {row['sequence']} "
                 f"(charge={row['charge']:+d}, MW={row['mw_da']:.0f} Da, "
                 f"Met={'yes' if row['met_warning'] else 'no'})")

    echem_df.to_csv(OUTPUT_DIR / 'echem_filter_results.csv', index=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    echem_df.to_csv(OUTPUT_DIR / 'echem_filter_results.csv', index=False)
    log.info(f"Filter results saved to {OUTPUT_DIR / 'echem_filter_results.csv'}")

    # -----------------------------------------------------------------------
    # Step 2: Score candidates against 3-plex classifiers
    # -----------------------------------------------------------------------
    if not args.skip_scoring:
        log.info("\n" + "=" * 60)
        log.info("Step 2: Candidate scoring (3-plex classifiers + percentile ranks)")
        log.info("=" * 60)
        from src.score_candidates import run as run_scoring
        score_df = run_scoring(OUTPUT_DIR)
        log.info("\nCandidate scoring summary:")
        display_cols = ['name', 'sequence', 'target', 'stage', 'margin', 'margin_percentile_in_pssm_pool']
        log.info("\n" + score_df[display_cols].to_string(index=False))
    else:
        log.info("Skipping Step 2 (--skip-scoring).")
        score_df = None

    # -----------------------------------------------------------------------
    # Step 3: 3×3 specificity matrix + condition number
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("Step 3: 3×3 specificity matrix + condition number")
    log.info("=" * 60)
    from src.condition_number import run as run_condition
    cond = run_condition(OUTPUT_DIR)

    log.info(f"\nCondition number κ(A) = {cond['kappa']:.2f}")
    log.info(f"Verdict: {cond['verdict']}")
    log.info(f"Action:  {cond['action']}")

    # ------------------------------------------------------------------
    # Final summary
    log.info("\n" + "=" * 60)
    log.info("Task 8 pipeline complete. Outputs:")
    log.info(f"  {OUTPUT_DIR}/echem_filter_results.csv")
    log.info(f"  {OUTPUT_DIR}/candidate_scores.csv")
    log.info(f"  {OUTPUT_DIR}/specificity_matrix_3plex_normalised.csv")
    log.info("=" * 60)
    log.info(f"Next: compile substrate_design/report_task8.tex")


if __name__ == '__main__':
    main()
