"""
Task 8 substrate design pipeline.

Usage:
    python run_all.py               # full pipeline
    python run_all.py --skip-calibration  # skip slow calibration step
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
PREDICTOR_PROC  = PREDICTOR_DIR / 'data' / 'processed'
PREDICTOR_MDLS  = PREDICTOR_DIR / 'models'
PSSM_DIR        = REPO / 'cross_reactivity_matrix' / 'data' / 'processed'

OUTPUT_DIR      = BASE / 'data' / 'outputs'
FIGURES_DIR     = BASE / 'figures'


def main():
    parser = argparse.ArgumentParser(description='Task 8: 3-plex substrate design pipeline')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip calibration/CI/baseline step (saves ~5 min)')
    parser.add_argument('--skip-scoring', action='store_true',
                        help='Skip candidate scoring step')
    args = parser.parse_args()

    # Insert in reverse priority order so BASE ends up at index 0 (highest priority).
    # substrate_predictor also has a src/ package; BASE must come first so
    # "from src.echem_filter" resolves to substrate_design/src, not predictor/src.
    sys.path.insert(0, str(PREDICTOR_DIR))
    sys.path.insert(0, str(BASE.parent))
    sys.path.insert(0, str(BASE))

    # -----------------------------------------------------------------------
    # Step 0: Build and cache MMP-2 PSSM to the shared PSSM directory
    # (needed by score_candidates.py for percentile-rank enumeration)
    # -----------------------------------------------------------------------
    import numpy as np
    import pandas as pd
    mmp2_pssm_path = PSSM_DIR / 'pssm_MMP_2.npy'
    if not mmp2_pssm_path.exists():
        log.info("Building MMP-2 PSSM from training CSV ...")
        from src.condition_number import _build_pssm_from_csv
        pssm = _build_pssm_from_csv(PREDICTOR_PROC / 'training_set_MMP_2.csv')
        np.save(mmp2_pssm_path, pssm)
        log.info(f"  Saved MMP-2 PSSM to {mmp2_pssm_path}")

    # -----------------------------------------------------------------------
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
        score_df = run_scoring(PREDICTOR_MDLS, PSSM_DIR, OUTPUT_DIR)
        log.info("\nCandidate scoring summary:")
        display_cols = ['name', 'sequence', 'target', 'stage',
                        'p_target', 'selectivity', 'joint_score', 'percentile_in_pssm_pool']
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
    cond = run_condition(PREDICTOR_PROC, OUTPUT_DIR)

    log.info(f"\nCondition number κ(A) = {cond['kappa']:.2f}")
    log.info(f"Verdict: {cond['verdict']}")
    log.info(f"Action:  {cond['action']}")

    # -----------------------------------------------------------------------
    # Step 4: Calibration, bootstrapped CIs, composition baseline
    # -----------------------------------------------------------------------
    if not args.skip_calibration:
        log.info("\n" + "=" * 60)
        log.info("Step 4: Calibration curves / bootstrap CIs / composition baseline")
        log.info("=" * 60)
        from src.calibration import run as run_calibration
        cal_results = run_calibration(PREDICTOR_PROC, PREDICTOR_MDLS, OUTPUT_DIR, FIGURES_DIR)

        log.info("\nBootstrap 95% CIs:")
        log.info(cal_results['bootstrap_ci'].to_string(index=False))
        log.info("\nComposition baseline (VHSE8 vs bag-of-AAs):")
        log.info(cal_results['baseline'].to_string(index=False))
    else:
        log.info("Skipping Step 4 (--skip-calibration).")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("Task 8 pipeline complete. Outputs:")
    log.info(f"  {OUTPUT_DIR}/echem_filter_results.csv")
    log.info(f"  {OUTPUT_DIR}/candidate_scores.csv")
    log.info(f"  {OUTPUT_DIR}/specificity_matrix_3plex_normalised.csv")
    log.info(f"  {OUTPUT_DIR}/calibration_results.csv")
    log.info(f"  {OUTPUT_DIR}/bootstrap_ci.csv")
    log.info(f"  {OUTPUT_DIR}/composition_baseline.csv")
    log.info(f"  {FIGURES_DIR}/fig_calibration_curves.png")
    log.info("=" * 60)
    log.info(f"Next: compile substrate_design/report_task8.tex")


if __name__ == '__main__':
    main()
