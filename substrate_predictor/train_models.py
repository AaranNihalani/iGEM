"""
Training entry point for the substrate predictor.

Usage:
    python train_models.py                  # full pipeline: data + train + figures
    python train_models.py --skip-data      # use existing processed CSVs
    python train_models.py --esm2           # include ESM2 features (requires torch)
    python train_models.py --no-figures     # skip figure generation
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

BASE         = Path(__file__).parent
PROCESSED    = BASE / 'data' / 'processed'
OUTPUTS      = BASE / 'data' / 'outputs'
MODELS       = BASE / 'models'
FIGURES      = BASE / 'figures'


def main():
    parser = argparse.ArgumentParser(description='Train Task 7 substrate classifiers')
    parser.add_argument('--skip-data',   action='store_true',
                        help='Skip data pipeline; use existing processed CSVs')
    parser.add_argument('--esm2',        action='store_true',
                        help='Include ESM2 embeddings (requires transformers + torch)')
    parser.add_argument('--no-figures',  action='store_true',
                        help='Skip figure generation')
    args = parser.parse_args()

    # --- Step 1: Data pipeline ---
    if not args.skip_data:
        log.info("Step 1: Building training sets ...")
        from src.data_pipeline import build_all_training_sets
        datasets = build_all_training_sets(BASE, PROCESSED)
        if not datasets:
            log.error("No training sets produced. Check Substrate_search.txt is accessible.")
            sys.exit(1)
    else:
        log.info("Step 1: Skipping data pipeline (--skip-data).")

    # --- Step 2: Train classifiers ---
    log.info("Step 2: Training per-protease classifiers ...")
    from src.train import train_all
    cv_df = train_all(PROCESSED, MODELS, OUTPUTS, use_esm2=args.esm2)

    log.info("\nCross-validation summary:")
    log.info(cv_df.to_string(index=False))

    # --- Step 3: Figures ---
    if not args.no_figures:
        log.info("Step 3: Generating evaluation figures ...")
        from src.evaluate import plot_roc_curves, plot_feature_importance
        plot_roc_curves(PROCESSED, MODELS, FIGURES)
        plot_feature_importance(MODELS, FIGURES, protease='MMP14')

    # --- Summary ---
    log.info("=" * 60)
    log.info("Training complete. Outputs:")
    log.info(f"  Models:      {MODELS}/")
    log.info(f"  CV results:  {OUTPUTS / 'cv_results.csv'}")
    if not args.no_figures:
        log.info(f"  ROC figure:  {FIGURES / 'fig_roc_curves.png'}")
        log.info(f"  Feature imp: {FIGURES / 'fig_feature_importance.png'}")
    log.info("=" * 60)
    log.info("Next: python predict.py --protease MMP14 --top-k 10")


if __name__ == '__main__':
    main()
