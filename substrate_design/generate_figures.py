"""
Generate Task 8 v2 report figures.

Outputs (all to substrate_design/figures/):
  fig_model_panel_v2.png      -- AUROC bar chart with bootstrap 95% CI whiskers
  fig_cross_reactivity_v2.png -- 3x3 heatmap: first-order candidates vs panel v2 models
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

BASE      = Path(__file__).parent
REPO      = BASE.parent
PRED_DIR  = REPO / 'substrate_predictor'
MODELS_DIR = PRED_DIR / 'models'
FIGDIR    = BASE / 'figures'
FIGDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PRED_DIR))
from src.features import build_feature_matrix

COL = {
    'MMP-2': '#4878CF',
    'MMP-9': '#3A9E4A',
    'HNE':   '#C94040',
}
GREY = '#777777'

# Bootstrap CIs from retrain_panel_v2.py output
PANEL_DATA = {
    'MMP-2': {'auroc': 0.871, 'ci_lo': 0.864, 'ci_hi': 0.879, 'n': 2819, 'pssm': 0.895},
    'MMP-9': {'auroc': 0.697, 'ci_lo': 0.675, 'ci_hi': 0.717, 'n':  783, 'pssm': 0.669},
    'HNE':   {'auroc': 0.919, 'ci_lo': 0.900, 'ci_hi': 0.935, 'n':  327, 'pssm': 0.903},
}

def load_v2_model(name: str):
    f = MODELS_DIR / f'lr_v2_{name}.pkl'
    if not f.exists():
        f = MODELS_DIR / f'lr_{name}.pkl'
        log.warning(f"Falling back to v1 model for {name}")
    return joblib.load(f)

def score_seq(model, seq: str) -> float:
    X = build_feature_matrix([seq], use_esm2=False)
    return float(model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Figure 1 – Panel v2 AUROC with bootstrap CI whiskers
# ---------------------------------------------------------------------------

def fig_model_panel_v2():
    names  = list(PANEL_DATA.keys())
    aurocs = [PANEL_DATA[n]['auroc'] for n in names]
    lo     = [PANEL_DATA[n]['ci_lo'] for n in names]
    hi     = [PANEL_DATA[n]['ci_hi'] for n in names]
    n_pos  = [PANEL_DATA[n]['n']     for n in names]
    pssms  = [PANEL_DATA[n]['pssm']  for n in names]
    colours = [COL[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))

    bars = ax.bar(x, aurocs, 0.48, color=colours, edgecolor='white',
                  linewidth=0.6, alpha=0.85)

    # Bootstrap CI whiskers
    yerr_lo = np.array(aurocs) - np.array(lo)
    yerr_hi = np.array(hi) - np.array(aurocs)
    ax.errorbar(x, aurocs, yerr=[yerr_lo, yerr_hi],
                fmt='none', color='#333333', capsize=5, capthick=1.4, linewidth=1.4)

    for i, (bar, v, n) in enumerate(zip(bars, aurocs, n_pos)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold',
                color=colours[i])
        ax.text(bar.get_x() + bar.get_width() / 2, 0.33,
                f'n={n:,}', ha='center', va='bottom', fontsize=8, color='white',
                fontweight='bold')

    # PSSM dots
    ax.scatter(x, pssms, marker='D', s=50, color='#111111', zorder=5,
               label='PSSM AUROC')

    ax.axhline(0.5,  color='grey',     lw=0.9, linestyle='--', alpha=0.5)
    ax.axhline(0.75, color='steelblue', lw=0.8, linestyle=':', alpha=0.6, label='0.75 reference')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('AUROC (5-fold CV)', fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.set_title('Panel v2 AUROC — protease-vs-protease negatives\n'
                 'Whiskers = 1000-iteration bootstrap 95% CI  |  ◆ = PSSM',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    out = FIGDIR / 'fig_model_panel_v2.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 2 – 3×3 cross-reactivity heatmap (v2 models)
# ---------------------------------------------------------------------------

FINAL_SUBS = {
    'GPLGIAGQ': ('MMP-2', 'MMP-2 channel'),
    'IPRTLTAA': ('MMP-9', 'MMP-9 channel'),
    'AAPVSGIG': ('HNE',   'HNE channel'),
}
TARGET_KEYS   = ['MMP_2', 'MMP_9', 'NE']
TARGET_LABELS = ['MMP-2', 'MMP-9', 'HNE']


def fig_cross_reactivity_v2():
    models = {}
    for key, label in zip(TARGET_KEYS, TARGET_LABELS):
        try:
            models[key] = load_v2_model(key)
        except FileNotFoundError as e:
            log.error(e)
            return

    seqs       = list(FINAL_SUBS.keys())
    sub_labels = [f'{s}\n({FINAL_SUBS[s][1]})' for s in seqs]
    targets    = [FINAL_SUBS[s][0] for s in seqs]

    mat = np.zeros((len(seqs), len(TARGET_KEYS)))
    for i, seq in enumerate(seqs):
        for j, key in enumerate(TARGET_KEYS):
            mat[i, j] = score_seq(models[key], seq)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for i in range(len(seqs)):
        for j, label in enumerate(TARGET_LABELS):
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

    ax.set_xlim(-0.5, len(TARGET_LABELS) - 0.5)
    ax.set_ylim(-0.5, len(seqs) - 0.5)
    ax.set_xticks(range(len(TARGET_LABELS)))
    ax.set_xticklabels(TARGET_LABELS, fontsize=11)
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(sub_labels, fontsize=9)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Protease classifier (v2)', fontsize=10, labelpad=8)
    ax.set_ylabel('Substrate', fontsize=10)
    ax.set_title('3×3 Cross-reactivity matrix — panel v2\n'
                 'Green border = intended target',
                 fontsize=10.5, fontweight='bold', pad=28)

    legend_patches = [
        mpatches.Patch(facecolor='#C8ECD2', edgecolor='#1A5C2E', linewidth=2,
                       label='On-target (≥0.75)'),
        mpatches.Patch(facecolor='#F2CECE',
                       label='Off-target risk (≥0.75)'),
        mpatches.Patch(facecolor='#FAF0D8',
                       label='Elevated (0.45–0.75)'),
        mpatches.Patch(facecolor='#F5F5F8',
                       label='Low (<0.45)'),
    ]
    ax.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8, frameon=True)

    fig.tight_layout()
    out = FIGDIR / 'fig_cross_reactivity_v2.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved {out}')


if __name__ == '__main__':
    log.info('Figure 1: panel v2 AUROC with bootstrap CI')
    fig_model_panel_v2()
    log.info('Figure 2: cross-reactivity heatmap v2')
    fig_cross_reactivity_v2()
    log.info('All figures done.')
