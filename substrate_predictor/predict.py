"""
predict.py -- Substrate peptide prediction CLI for the Eton iGEM 2026 panel.

Usage:
    python predict.py --protease MMP14 --top-k 10
    python predict.py --protease MMP-8 --scan SEQUENCE.fasta --top-k 5
"""

import sys
import logging
from pathlib import Path

import click
import pandas as pd

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE      = Path(__file__).parent
MODELS    = BASE / 'models'
OUTPUTS   = BASE / 'data' / 'outputs'
PROCESSED = BASE / 'data' / 'processed'

TASK6_PSSM = BASE.parent / 'cross_reactivity_matrix' / 'data' / 'processed'
PSSM_DIRS  = [PROCESSED, TASK6_PSSM]

PROTEASE_CHOICES = ['MMP14', 'MMP-8', 'MMP-9', 'NE', 'Fecal', 'FAP']

from src.utils import confidence_tier, PROTEASE_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_models_or_exit():
    from src.selectivity import load_models
    models = load_models(MODELS)
    if not models:
        click.echo(
            "ERROR: No trained models found in models/. "
            "Run 'python train_models.py' first.", err=True
        )
        sys.exit(1)
    return models


def _n_positives(protease: str) -> int:
    """Return number of positives used in training (from training set CSV)."""
    csv = PROCESSED / f"training_set_{protease.replace('-','_')}.csv"
    if not csv.exists():
        return 0
    df = pd.read_csv(csv, usecols=['label'])
    return int(df['label'].sum())


def _format_table(results: list[dict], top_k: int,
                  protease: str, scan_mode: bool = False) -> str:
    """Format ranked results as an aligned text table."""
    n_pos = _n_positives(protease)
    tier  = confidence_tier(n_pos)

    header_extra = "  Position      " if scan_mode else ""
    lines = [
        f"\nProtease: {protease}  |  Confidence: {tier} "
        f"(trained on {n_pos} positives)",
        "",
        f"{'Rank':<5} {'Motif':<10} {'P(cleave)':<11} "
        f"{'Selectivity':<13} {'Joint score':<13} "
        f"{'Confidence':<12}{'Position' if scan_mode else ''}",
        "-" * (65 + (20 if scan_mode else 0)),
    ]
    for rank, r in enumerate(results[:top_k], 1):
        pos_str = f"  {r.get('position', '')}" if scan_mode else ""
        lines.append(
            f"{rank:<5} {r['sequence']:<10} {r['p_cleave']:<11.4f} "
            f"{r['selectivity']:<13.3f} {r['joint_score']:<13.4f} "
            f"{tier:<12}{pos_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--protease', required=True,
              type=click.Choice(PROTEASE_CHOICES, case_sensitive=False),
              help='Target protease to predict substrates for.')
@click.option('--top-k', default=10, show_default=True,
              help='Number of top candidates to return.')
@click.option('--scan', type=click.Path(exists=True), default=None,
              help='FASTA file to scan with a sliding 8-mer window.')
@click.option('--models-dir', type=click.Path(), default=None,
              help='Override path to models directory.')
@click.option('--save', is_flag=True, default=False,
              help='Save predictions to data/outputs/.')
def main(protease, top_k, scan, models_dir, save):
    """
    Rank candidate 8-mer substrate peptides for a panel protease by predicted
    cleavage probability and selectivity over the remaining five panel members.
    """
    global MODELS
    if models_dir:
        MODELS = Path(models_dir)

    models = _load_models_or_exit()

    from src.selectivity import score_candidates, generate_candidates_from_pssm

    if scan:
        results = _scan_fasta(scan, protease, models)
        output = _format_table(results, top_k, protease, scan_mode=True)
    else:
        # Default: enumerate candidates from PSSM top-3 per position (3^8 = 6,561)
        candidates = generate_candidates_from_pssm(protease, PSSM_DIRS, top_k_per_pos=3)
        results = score_candidates(candidates, protease, models, pssm_dirs=PSSM_DIRS)
        output = _format_table(results, top_k, protease)

    click.echo(output)

    if save:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        from datetime import date
        out_path = OUTPUTS / f"predictions_{protease.replace('-','_')}_{date.today()}.csv"
        rows = []
        for rank, r in enumerate(results[:top_k], 1):
            row = {'rank': rank, 'sequence': r['sequence'],
                   'p_cleave': r['p_cleave'], 'selectivity': r['selectivity'],
                   'joint_score': r['joint_score']}
            row.update({f"p_{p.replace('-','_')}": r['probs'].get(p, '')
                        for p in PROTEASE_ORDER})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        click.echo(f"\nSaved to {out_path}")


def _scan_fasta(fasta_path: str, protease: str,
                models: dict) -> list[dict]:
    """Slide an 8-mer window across each FASTA sequence and score all windows."""
    from Bio import SeqIO
    from src.utils import is_valid_window
    from src.selectivity import score_candidates

    candidates = []
    positions  = []

    for record in SeqIO.parse(fasta_path, 'fasta'):
        seq = str(record.seq).upper()
        for i in range(len(seq) - 7):
            window = seq[i:i + 8]
            if is_valid_window(window):
                candidates.append(window)
                positions.append(f"{record.id}:{i+1}-{i+8}")

    if not candidates:
        click.echo("No valid 8-mer windows found in FASTA.", err=True)
        sys.exit(1)

    results = score_candidates(candidates, protease, models, pssm_dirs=PSSM_DIRS)

    # Attach position to each result
    pos_map = {seq: pos for seq, pos in zip(candidates, positions)}
    for r in results:
        r['position'] = pos_map.get(r['sequence'], '')

    return results


if __name__ == '__main__':
    main()
