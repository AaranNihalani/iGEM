"""
Produce the publication-quality specificity matrix heatmap.

Spec:
  - seaborn heatmap, annot=True, fmt='.2f'
  - Colourmap: RdYlGn_r (red = cross-reactive, green = orthogonal)
  - Diagonal cells outlined in black
  - Off-diagonal cells >= 0.20 annotated in bold red
  - Row labels: protease names
  - Column labels: motif sequences (e.g. 'MMP14\nSGRSENIR')
  - Figure size: (9, 7), 300 dpi, bbox_inches='tight'
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from src.utils import PANEL_MOTIFS, PROTEASE_ORDER
from src.score_motifs import FLAG_THRESHOLD

log = logging.getLogger(__name__)


def _column_labels(motifs: dict[str, str]) -> list[str]:
    """Two-line column labels: protease name + motif sequence."""
    return [f"{p}\n{motifs.get(p, '?')}" for p in PROTEASE_ORDER]


def _apply_figure_style() -> None:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
    })


def plot_heatmap(normed: pd.DataFrame,
                 output_path: Path,
                 motifs: dict[str, str] | None = None) -> None:
    """
    Render and save the specificity heatmap from a normalised score DataFrame.
    """
    if motifs is None:
        motifs = PANEL_MOTIFS

    _apply_figure_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    # Clamp display range to [-0.5, 1.5] so outliers don't compress the colour scale
    display = normed.clip(-0.5, 1.5)

    sns.heatmap(
        display,
        ax=ax,
        annot=normed.round(2),   # annotate with true (unclamped) values
        fmt='.2f',
        cmap='RdYlGn_r',
        vmin=-0.5,
        vmax=1.5,
        linewidths=0.4,
        linecolor='white',
        cbar_kws={'label': 'Normalised cleavage score (self = 1.00)', 'shrink': 0.85},
        xticklabels=_column_labels(motifs),
        yticklabels=PROTEASE_ORDER,
        annot_kws={'size': 9},
    )

    # Bold red text on flagged off-diagonal cells
    n = len(PROTEASE_ORDER)
    for i, row_p in enumerate(PROTEASE_ORDER):
        for j, col_p in enumerate(PROTEASE_ORDER):
            if row_p == col_p:
                continue
            val = normed.loc[row_p, col_p]
            if not np.isnan(val) and val >= FLAG_THRESHOLD:
                ax.texts[i * n + j].set_color('darkred')
                ax.texts[i * n + j].set_fontweight('bold')

    # Black outline on each diagonal cell
    for k in range(n):
        ax.add_patch(mpatches.Rectangle(
            (k, k), 1, 1,
            fill=False, edgecolor='black', lw=2.0, zorder=5,
        ))

    ax.set_title(
        'Cross-reactivity specificity matrix: protease panel vs. substrate motifs',
        pad=12, fontsize=11, fontweight='bold',
    )
    ax.set_ylabel('Protease', labelpad=8)
    ax.set_xlabel('Substrate motif', labelpad=8)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    # Legend patch for flagged cells
    flag_patch = mpatches.Patch(
        facecolor='none', edgecolor='darkred',
        label=f'Score >= {FLAG_THRESHOLD} (flagged)',
    )
    ax.legend(
        handles=[flag_patch], loc='upper right',
        bbox_to_anchor=(1.0, -0.12), fontsize=8, frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Heatmap saved to {output_path}")


if __name__ == '__main__':
    base = Path(__file__).parent.parent
    normed = pd.read_csv(base / 'data' / 'outputs' / 'specificity_matrix_normalised.csv',
                         index_col=0)
    plot_heatmap(normed, base / 'figures' / 'fig_specificity_heatmap.png')
