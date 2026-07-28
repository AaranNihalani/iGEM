"""
Generate all Task 8 (revised) report figures.

Outputs (all to substrate_design/figures/):
  fig_model_panel.png          -- AUROC bar chart for all 6 classifiers
  fig_cross_reactivity.png     -- 3x3 heatmap: final substrates vs target proteases
  fig_mmp9_substitution.png    -- IPRTLTAA vs IPRTHTAA score comparison

Reads model files from substrate_predictor/models/ and data from
substrate_predictor/data/raw/ directly; no pre-generated CSVs required.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
log = logging.getLogger(__name__)

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

BASE      = Path(__file__).parent
REPO      = BASE.parent
PRED_DIR  = REPO / 'substrate_predictor'
MODELS_DIR = PRED_DIR / 'models'
FIGDIR    = BASE / 'figures'
FIGDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PRED_DIR))
from src.features import build_feature_matrix

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
COL = {
    'MMP-2':  '#4878CF',
    'MMP-8':  '#6FB0D0',
    'MMP-9':  '#3A9E4A',
    'MMP14':  '#A8D46F',
    'NE':     '#C94040',
    'LasB':   '#8B3DAF',
}
GREY = '#777777'
HIGH_GREEN  = '#2E7D4F'
MED_AMBER   = '#B3720C'


def load_model(name: str):
    pkl = MODELS_DIR / f'lr_{name}.pkl'
    if not pkl.exists():
        raise FileNotFoundError(f"Model not found: {pkl}")
    return joblib.load(pkl)


def score_seq(model, seq: str) -> float:
    X = build_feature_matrix([seq], use_esm2=False)
    return float(model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Figure 1 – Model panel AUROC
# ---------------------------------------------------------------------------

PANEL_DATA = {
    # name: (auroc, n_positives, confidence_tier)
    'MMP-2':  (0.866, 2819, 'HIGH'),
    'MMP-8':  (0.830,  698, 'HIGH'),
    'MMP-9':  (0.800,  783, 'HIGH'),
    'MMP14':  (0.813,  717, 'HIGH'),
    'NE':     (0.825,  327, 'HIGH'),
    'LasB':   (0.816,   35, 'LOW'),    # MEROPS-only; LOW tier (<50 positives)
}

def fig_model_panel():
    names  = list(PANEL_DATA.keys())
    aurocs = [PANEL_DATA[n][0] for n in names]
    n_pos  = [PANEL_DATA[n][1] for n in names]
    tiers  = [PANEL_DATA[n][2] for n in names]
    colours = [COL[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(names))
    bars = ax.bar(x, aurocs, 0.55, color=colours, edgecolor='white', linewidth=0.6, alpha=0.88)

    for i, (bar, v, n, tier) in enumerate(zip(bars, aurocs, n_pos, tiers)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                color=COL[names[i]])
        ax.text(bar.get_x() + bar.get_width() / 2, 0.31,
                f'n={n:,}', ha='center', va='bottom', fontsize=7.5, color='white',
                fontweight='bold')

    ax.axhline(0.5, color='grey', lw=0.9, linestyle='--', alpha=0.6, label='Random (0.5)')
    ax.axhline(0.75, color='steelblue', lw=0.8, linestyle=':', alpha=0.7, label='0.75 reference')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('AUROC (5-fold stratified CV)', fontsize=10)
    ax.set_ylim(0.3, 1.02)
    ax.set_title('Expanded 6-protease classifier panel\nVHSE8 + logistic regression, 5-fold CV',
                 fontsize=10.5, fontweight='bold')

    high_patch = mpatches.Patch(facecolor=HIGH_GREEN, label='HIGH confidence (≥200 positives)')
    ax.legend(handles=[high_patch,
                        mpatches.Patch(facecolor='grey', linestyle='--', label='Random (0.5)'),
                        mpatches.Patch(facecolor='steelblue', alpha=0.5, label='0.75 reference')],
              fontsize=8, loc='upper right')

    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    out = FIGDIR / 'fig_model_panel.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 2 – 3×3 cross-reactivity heatmap (from live model scores)
# ---------------------------------------------------------------------------

FINAL_SUBS = {
    'IPRTHTAA': ('MMP-9', 'MMP-9 channel'),
    'AAPVSGIG': ('NE',    'HNE channel'),
    'AAAGLAAG': ('LasB',  'LasB channel'),
}
TARGET_MODELS = ['MMP_9', 'NE', 'LasB']
TARGET_LABELS = ['MMP-9', 'HNE (NE)', 'LasB']


def fig_cross_reactivity():
    models = {}
    for key in TARGET_MODELS:
        try:
            models[key] = load_model(key)
        except FileNotFoundError as e:
            log.error(e)
            return

    seqs = list(FINAL_SUBS.keys())
    sub_labels = [f'{s}\n({FINAL_SUBS[s][1]})' for s in seqs]
    targets    = [FINAL_SUBS[s][0] for s in seqs]

    # Score matrix [substrate, protease]
    mat = np.zeros((len(seqs), len(TARGET_MODELS)))
    for i, seq in enumerate(seqs):
        for j, key in enumerate(TARGET_MODELS):
            mat[i, j] = score_seq(models[key], seq)

    # Custom diverging cmap: low=white, mid=amber, high varies by diagonal/off
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    # Draw cells manually so diagonal can use green and off-diagonal red
    for i in range(len(seqs)):
        for j, key in enumerate(TARGET_MODELS):
            v = mat[i, j]
            prot_label = TARGET_LABELS[j].split(' ')[0]   # 'MMP-9', 'HNE', 'LasB'
            is_target = targets[i] == prot_label or (targets[i] == 'NE' and j == 1)

            if v >= 0.75:
                if is_target:
                    colour = '#C8ECD2'    # green tint
                    txt_col = '#1A5C2E'
                else:
                    colour = '#F2CECE'    # red tint
                    txt_col = '#7A1A1A'
            elif v >= 0.45:
                colour = '#FAF0D8'        # amber tint
                txt_col = '#7A5A0A'
            else:
                colour = '#F5F5F8'        # near-white
                txt_col = '#555'

            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  facecolor=colour, edgecolor='white', linewidth=1.2)
            ax.add_patch(rect)

            weight = 'bold' if is_target else 'normal'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=13, color=txt_col, fontweight=weight)

            if is_target:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor='#1A5C2E', linewidth=2.2))

    ax.set_xlim(-0.5, len(TARGET_MODELS) - 0.5)
    ax.set_ylim(-0.5, len(seqs) - 0.5)
    ax.set_xticks(range(len(TARGET_LABELS)))
    ax.set_xticklabels(TARGET_LABELS, fontsize=10)
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(sub_labels, fontsize=9)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Protease classifier (scorer)', fontsize=10, labelpad=8)
    ax.set_ylabel('Substrate', fontsize=10)
    ax.set_title('3×3 Cross-reactivity matrix\n(outlined cell = intended target)',
                 fontsize=10.5, fontweight='bold', pad=28)

    legend_patches = [
        mpatches.Patch(facecolor='#C8ECD2', edgecolor='#1A5C2E', linewidth=2, label='On-target, high (≥0.75)'),
        mpatches.Patch(facecolor='#F2CECE', label='Off-target, high — cross-reactive risk'),
        mpatches.Patch(facecolor='#FAF0D8', label='Elevated (0.45–0.75)'),
        mpatches.Patch(facecolor='#F5F5F8', label='Low (<0.45)'),
    ]
    ax.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=True)

    fig.tight_layout()
    out = FIGDIR / 'fig_cross_reactivity.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 3 – MMP-9 P1' substitution: IPRTLTAA vs IPRTHTAA
# ---------------------------------------------------------------------------

SUB_DATA = {
    'IPRT|LTAA\n(original)': {
        'seq': 'IPRTLTAA',
        'MMP-9': None, 'HNE': None, 'LasB': None,
    },
    'IPRT|HTAA\n(recommended)': {
        'seq': 'IPRTHTAA',
        'MMP-9': None, 'HNE': None, 'LasB': None,
    },
}

def fig_mmp9_substitution():
    mmp9_m  = load_model('MMP_9')
    ne_m    = load_model('NE')
    lasb_m  = load_model('LasB')
    model_map = {'MMP-9': mmp9_m, 'HNE': ne_m, 'LasB': lasb_m}

    for label, d in SUB_DATA.items():
        d['MMP-9'] = score_seq(mmp9_m, d['seq'])
        d['HNE']   = score_seq(ne_m,   d['seq'])
        d['LasB']  = score_seq(lasb_m, d['seq'])

    variants = list(SUB_DATA.keys())
    proteases = ['MMP-9', 'HNE', 'LasB']
    prot_cols = {'MMP-9': COL['MMP-9'], 'HNE': COL['NE'], 'LasB': COL['LasB']}

    x = np.arange(len(proteases))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    orig_scores = [SUB_DATA[variants[0]][p] for p in proteases]
    recd_scores = [SUB_DATA[variants[1]][p] for p in proteases]

    b1 = ax.bar(x - width / 2, orig_scores, width,
                label=variants[0], color=[prot_cols[p] for p in proteases],
                edgecolor='black', linewidth=0.8, alpha=0.55, hatch='//')
    b2 = ax.bar(x + width / 2, recd_scores, width,
                label=variants[1], color=[prot_cols[p] for p in proteases],
                edgecolor='black', linewidth=1.2, alpha=0.9)

    for bar, v in zip(b1, orig_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9.5, color='#444')
    for bar, v in zip(b2, recd_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.axhline(0.5, color='grey', lw=0.8, linestyle='--', alpha=0.6)

    # Mark the intended target MMP-9 bars
    for bar in [b1[0], b2[0]]:
        ax.annotate('', xy=(bar.get_x() + bar.get_width() / 2, -0.04),
                    xytext=(bar.get_x() + bar.get_width() / 2, -0.01),
                    arrowprops=dict(arrowstyle='->', color=COL['MMP-9'], lw=1.5))
    ax.text(x[0], -0.07, 'target', ha='center', fontsize=8, color=COL['MMP-9'], style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(proteases, fontsize=11)
    ax.set_ylabel('Predicted cleavage probability', fontsize=10)
    ax.set_ylim(-0.1, 1.08)
    ax.set_title('MMP-9 substrate: P1\' Leu→His substitution\n'
                 'IPRTLTAA (original, hatched) vs IPRTHTAA (recommended, solid)',
                 fontsize=10.5, fontweight='bold')

    orig_patch = mpatches.Patch(facecolor='grey', alpha=0.55, hatch='//', edgecolor='black',
                                 label='IPRT|LTAA  original  (LasB > MMP-9)')
    recd_patch = mpatches.Patch(facecolor='grey', alpha=0.9, edgecolor='black',
                                 label='IPRT|HTAA  recommended  (margin +0.556)')
    ax.legend(handles=[orig_patch, recd_patch], fontsize=8.5, loc='upper right')

    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    fig.tight_layout()

    out = FIGDIR / 'fig_mmp9_substitution.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved {out}')


if __name__ == '__main__':
    log.info('Generating figure 1: model panel AUROC')
    fig_model_panel()
    log.info('Generating figure 2: cross-reactivity heatmap')
    fig_cross_reactivity()
    log.info('Generating figure 3: MMP-9 substitution')
    fig_mmp9_substitution()
    log.info('All figures done.')
