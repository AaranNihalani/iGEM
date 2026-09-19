"""
Task 8 figures for the MMP-9 / MMP-8 / NE and MMP-9 / NE panels.

Cross-validated performance is read from substrate_predictor/results (not typed in), and candidate
scores are computed live with the substrate predictor.
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
PRED = BASE.parent / "substrate_predictor"
FIGDIR = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PRED))

from spred.panel import Panel  # noqa: E402

log = logging.getLogger(__name__)
LABEL = {"MMP9": "MMP-9", "MMP8": "MMP-8", "NE": "HNE", "MMP2": "MMP-2"}
COL = {"MMP9": "#3A9E4A", "MMP8": "#2E6AB5", "NE": "#C94040", "MMP2": "#8E5BB5"}

PANELS = {
    "a": {"members": ["MMP9", "MMP8", "NE"], "title": "Panel A: MMP-9 / MMP-8 / HNE",
          "candidates": {"IPRTLTAA": "MMP9", "DAEDIVGT": "MMP8", "AAPVSGIG": "NE"}},
    "b": {"members": ["MMP9", "NE"], "title": "Panel B: MMP-9 / HNE",
          "candidates": {"IPRTLTAA": "MMP9", "AAPVSGIG": "NE"}},
}


def fig_auroc(members, title, out):
    cv = pd.read_csv(PRED / "results" / "cv_summary.csv").set_index("protease").loc[members]
    x = np.arange(len(members))
    fig, ax = plt.subplots(figsize=(max(5.5, 2.6 * len(members)), 4.5))
    au = cv["consensus_auroc"].values
    err = [au - cv["consensus_auroc_lo"].values, cv["consensus_auroc_hi"].values - au]
    ax.bar(x, au, 0.45, color=[COL[m] for m in members], alpha=0.85)
    ax.errorbar(x, au, yerr=err, fmt="none", color="#222", capsize=6, lw=1.5)
    ax.scatter(x - 0.12, cv["lr_auroc"], marker="o", s=40, color="#111", zorder=5, label="LR (cross-validated)")
    ax.scatter(x + 0.12, cv["pssm_auroc"], marker="D", s=45, color="#111", zorder=5, label="PSSM (cross-validated)")
    for i, m in enumerate(members):
        ax.text(i, 0.32, f"n={int(cv.loc[m, 'n_pos']):,}", ha="center", color="white", fontsize=8, fontweight="bold")
        ax.text(i, au[i] + 0.035, f"{au[i]:.3f}", ha="center", fontsize=11, fontweight="bold", color=COL[m])
    ax.axhline(0.5, color="grey", ls="--", lw=0.9, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in members], fontsize=12)
    ax.set_ylim(0.3, 1.06); ax.set_ylabel("AUROC (cluster-aware 5-fold CV)")
    ax.set_title(f"{title}\nBars: LR+PSSM consensus, whiskers: 95% cluster-bootstrap CI", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_heatmap(members, candidates, title, out):
    panel = Panel(members)
    seqs = list(candidates)
    sc = panel.score(seqs)
    fig, ax = plt.subplots(figsize=(2.0 * len(members) + 2.5, 1.5 * len(seqs) + 2.0))
    for i, s in enumerate(seqs):
        for j, m in enumerate(members):
            v = float(sc[f"u_{m}"][i]); tgt = candidates[s] == m
            face = "#C8ECD2" if (v >= 0.9 and tgt) else "#F2CECE" if v >= 0.9 else "#FAF0D8" if v >= 0.7 else "#F5F5F8"
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=face, edgecolor="white", lw=1.2))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=14, fontweight="bold" if tgt else "normal")
            if tgt:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="#1A5C2E", lw=2.5))
    ax.set_xlim(-.5, len(members) - .5); ax.set_ylim(-.5, len(seqs) - .5)
    ax.set_xticks(range(len(members))); ax.set_xticklabels([LABEL[m] for m in members])
    ax.set_yticks(range(len(seqs))); ax.set_yticklabels(seqs, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_title(f"{title}\nConsensus background percentile (LR and PSSM); green border = intended target",
                 fontsize=10.5, fontweight="bold", pad=28)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for k, cfg in PANELS.items():
        fig_auroc(cfg["members"], cfg["title"], FIGDIR / f"fig_panel_{k}_auroc.png")
        fig_heatmap(cfg["members"], cfg["candidates"], f"Cross-reactivity matrix, {cfg['title']}",
                    FIGDIR / f"fig_panel_{k}_heatmap.png")
    log.info("figures done")
