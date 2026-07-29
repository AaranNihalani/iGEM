"""
Generate final report figures for Panel A (MMP-9/MMP-8/HNE) and Panel B (MMP-9/HNE).

Outputs (substrate_design/figures/):
  fig_panel_a_auroc.png        -- Panel A AUROC bar chart with bootstrap CI whiskers
  fig_panel_a_heatmap.png      -- Panel A 3×3 cross-reactivity heatmap
  fig_panel_b_auroc.png        -- Panel B AUROC bar chart with bootstrap CI whiskers
  fig_panel_b_heatmap.png      -- Panel B 2×2 cross-reactivity heatmap
"""

import sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
log = logging.getLogger(__name__)

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE       = Path(__file__).parent
REPO       = BASE.parent
PRED_DIR   = REPO / 'substrate_predictor'
MODELS_DIR = PRED_DIR / 'models'
FIGDIR     = BASE / 'figures'
FIGDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PRED_DIR))
from src.features import build_feature_matrix

COL = {
    'MMP-9': '#3A9E4A',
    'MMP-8': '#2E6AB5',
    'HNE':   '#C94040',
}

# Bootstrap CIs from retrain_final_panels.py
PANEL_A_DATA = {
    'MMP-9': {'auroc': 0.792, 'ci_lo': 0.774, 'ci_hi': 0.810, 'n': 783,  'pssm': 0.849, 'conf': 'MODERATE'},
    'MMP-8': {'auroc': 0.775, 'ci_lo': 0.756, 'ci_hi': 0.793, 'n': 427,  'pssm': 0.831, 'conf': 'MODERATE'},
    'HNE':   {'auroc': 0.901, 'ci_lo': 0.879, 'ci_hi': 0.919, 'n': 327,  'pssm': 0.930, 'conf': 'HIGH'},
}

PANEL_B_DATA = {
    'MMP-9': {'auroc': 0.837, 'ci_lo': 0.820, 'ci_hi': 0.853, 'n': 783,  'pssm': 0.882, 'conf': 'HIGH'},
    'HNE':   {'auroc': 0.899, 'ci_lo': 0.877, 'ci_hi': 0.919, 'n': 327,  'pssm': 0.930, 'conf': 'HIGH'},
}

# Synthesis candidates
CANDIDATES_A = {
    'IPRTLTAA': ('MMP-9', 'MMP-9 channel'),
    'DAEDIVGT': ('MMP-8', 'MMP-8 channel'),
    'AAPVSGIG': ('HNE',   'HNE channel'),
}
CANDIDATES_B = {
    'IPRTLTAA': ('MMP-9', 'MMP-9 channel'),
    'AAPVSGIG': ('HNE',   'HNE channel'),
}

PANEL_A_KEYS   = ['MMP_9', 'MMP_8', 'NE']
PANEL_A_LABELS = ['MMP-9', 'MMP-8', 'HNE']
PANEL_B_KEYS   = ['MMP_9', 'NE']
PANEL_B_LABELS = ['MMP-9', 'HNE']


def load_model(name: str, suffix: str):
    f = MODELS_DIR / f'lr_{name}_{suffix}.pkl'
    if not f.exists():
        raise FileNotFoundError(f"Model not found: {f}")
    return joblib.load(f)


def score_seq(model, seq: str) -> float:
    X = build_feature_matrix([seq], use_esm2=False)
    return float(model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# AUROC bar chart
# ---------------------------------------------------------------------------

def fig_auroc(panel_data: dict, title: str, outpath: Path):
    names   = list(panel_data.keys())
    aurocs  = [panel_data[n]['auroc'] for n in names]
    lo      = [panel_data[n]['ci_lo'] for n in names]
    hi      = [panel_data[n]['ci_hi'] for n in names]
    n_pos   = [panel_data[n]['n']     for n in names]
    pssms   = [panel_data[n]['pssm']  for n in names]
    confs   = [panel_data[n]['conf']  for n in names]
    colours = [COL[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(5.5, 2.6 * len(names)), 4.5))
    x = np.arange(len(names))

    bars = ax.bar(x, aurocs, 0.45, color=colours, edgecolor='white',
                  linewidth=0.6, alpha=0.85)

    yerr_lo = np.array(aurocs) - np.array(lo)
    yerr_hi = np.array(hi) - np.array(aurocs)
    ax.errorbar(x, aurocs, yerr=[yerr_lo, yerr_hi],
                fmt='none', color='#222222', capsize=6, capthick=1.5, linewidth=1.5)

    for i, (bar, v, n, conf) in enumerate(zip(bars, aurocs, n_pos, confs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.030,
                f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=colours[i])
        ax.text(bar.get_x() + bar.get_width()/2, 0.33,
                f'n={n:,}', ha='center', va='bottom', fontsize=8,
                color='white', fontweight='bold')
        label_col = '#1A5C2E' if conf == 'HIGH' else '#8B6000'
        ax.text(bar.get_x() + bar.get_width()/2, lo[i] - 0.048,
                conf, ha='center', va='top', fontsize=7.5, color=label_col,
                fontweight='bold')

    ax.scatter(x, pssms, marker='D', s=55, color='#111111', zorder=5,
               label='PSSM AUROC (independent)')

    ax.axhline(0.5,  color='grey',      lw=0.9, linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.axhline(0.75, color='steelblue', lw=0.8, linestyle=':', alpha=0.6, label='0.75 reference')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('AUROC (5-fold CV)', fontsize=10)
    ax.set_ylim(0.3, 1.06)
    ax.set_title(
        f'{title}\n'
        'Protease-vs-protease negatives · Whiskers = 1000-iter bootstrap 95% CI · ◆ = PSSM',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved {outpath}')


# ---------------------------------------------------------------------------
# Cross-reactivity heatmap
# ---------------------------------------------------------------------------

def fig_heatmap(candidates: dict, model_keys: list, model_labels: list,
                suffix: str, title: str, outpath: Path):
    models = {}
    for key in model_keys:
        try:
            models[key] = load_model(key, suffix)
        except FileNotFoundError as e:
            log.error(str(e))
            return

    seqs       = list(candidates.keys())
    sub_labels = [f'{s}\n({candidates[s][1]})' for s in seqs]
    targets    = [candidates[s][0] for s in seqs]

    mat = np.zeros((len(seqs), len(model_keys)))
    for i, seq in enumerate(seqs):
        for j, key in enumerate(model_keys):
            mat[i, j] = score_seq(models[key], seq)

    fig, ax = plt.subplots(figsize=(2.0 * len(model_labels) + 2.5,
                                    1.5 * len(seqs) + 2.0))

    for i in range(len(seqs)):
        for j, label in enumerate(model_labels):
            v = mat[i, j]
            is_target = targets[i] == label

            if v >= 0.75:
                colour  = '#C8ECD2' if is_target else '#F2CECE'
                txt_col = '#1A5C2E' if is_target else '#7A1A1A'
            elif v >= 0.45:
                colour  = '#FAF0D8'
                txt_col = '#7A5A0A'
            else:
                colour  = '#F5F5F8'
                txt_col = '#555555'

            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor=colour, edgecolor='white', linewidth=1.2))
            weight = 'bold' if is_target else 'normal'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=14, color=txt_col, fontweight=weight)
            if is_target:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor='#1A5C2E', linewidth=2.5))

    ax.set_xlim(-0.5, len(model_labels) - 0.5)
    ax.set_ylim(-0.5, len(seqs) - 0.5)
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(sub_labels, fontsize=9)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Protease classifier', fontsize=10, labelpad=8)
    ax.set_ylabel('Substrate', fontsize=10)
    ax.set_title(f'{title}\nGreen border = intended target  |  Both LR and PSSM scores agree',
                 fontsize=10.5, fontweight='bold', pad=28)

    legend_patches = [
        mpatches.Patch(facecolor='#C8ECD2', edgecolor='#1A5C2E', linewidth=2,
                       label='On-target (≥0.75)'),
        mpatches.Patch(facecolor='#F2CECE', label='Off-target risk (≥0.75)'),
        mpatches.Patch(facecolor='#FAF0D8', label='Elevated (0.45–0.75)'),
        mpatches.Patch(facecolor='#F5F5F8', label='Low (<0.45)'),
    ]
    ax.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved {outpath}')


if __name__ == '__main__':
    log.info('=== Panel A figures ===')
    fig_auroc(PANEL_A_DATA, 'Panel A — MMP-9 / MMP-8 / HNE',
              FIGDIR / 'fig_panel_a_auroc.png')
    fig_heatmap(CANDIDATES_A, PANEL_A_KEYS, PANEL_A_LABELS, 'panel_a',
                '3×3 Cross-reactivity matrix — Panel A',
                FIGDIR / 'fig_panel_a_heatmap.png')

    log.info('=== Panel B figures ===')
    fig_auroc(PANEL_B_DATA, 'Panel B — MMP-9 / HNE',
              FIGDIR / 'fig_panel_b_auroc.png')
    fig_heatmap(CANDIDATES_B, PANEL_B_KEYS, PANEL_B_LABELS, 'panel_b',
                '2×2 Cross-reactivity matrix — Panel B',
                FIGDIR / 'fig_panel_b_heatmap.png')

    log.info('All figures done.')
