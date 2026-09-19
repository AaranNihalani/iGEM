"""Figures on methodology: model comparison, learning curves, negative design, nested tuning, ROC/PR, calibration, selectivity."""
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

from fig_common import *

MET = ROOT / "results" / "methodology"
MODEL_ORDER = ["prior", "composition-LR", "one-hot LR", "VHSE LR", "PSSM", "one-hot GBM", "VHSE GBM"]
MODEL_COL = {"prior": "#b8bec7", "composition-LR": "#9aa5b5", "one-hot LR": "#4c78a8", "VHSE LR": "#e45756", "PSSM": "#54a24b",
             "one-hot GBM": "#b279a2", "VHSE GBM": "#f2a541"}


def fig_model_comparison():
    c = pd.read_csv(MET / "model_comparison.csv")
    d = pd.read_csv(MET / "paired_differences_vs_vhse_lr.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    shown = [m for m in MODEL_ORDER if m != "prior"]
    for i, pid in enumerate(ORDER):
        for k, m in enumerate(shown):
            r = c[(c.protease == pid) & (c.model == m)].iloc[0]
            y = i + (k - (len(shown) - 1) / 2) * 0.11
            ax.plot([r.lo, r.hi], [y, y], color=MODEL_COL[m], lw=1.2, alpha=0.8)
            ax.scatter(r.auroc, y, s=16, color=MODEL_COL[m], zorder=3, label=m if i == 0 else None)
    ax.axvline(0.5, color=MUTE, lw=0.8, ls=":")
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[p] for p in ORDER]); ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("cluster-aware CV AUROC (95% cluster-bootstrap CI)")
    ax.set_title("Six models on identical folds"); ax.legend(fontsize=6.8, loc="lower left", ncol=1, borderaxespad=0.2)
    panel_letter(ax, "a", -0.2)
    ax = axes[1]
    vs = ["one-hot LR", "PSSM", "one-hot GBM", "VHSE GBM"]
    for i, pid in enumerate(ORDER):
        for k, m in enumerate(vs):
            r = d[(d.protease == pid) & (d.vs == m)].iloc[0]
            y = i + (k - (len(vs) - 1) / 2) * 0.13
            ax.plot([-r.hi, -r.lo], [y, y], color=MODEL_COL[m], lw=1.2, alpha=0.85)
            ax.scatter(-r.delta_auroc, y, s=16, color=MODEL_COL[m], zorder=3, marker="o" if r.significant else "o", facecolors=MODEL_COL[m] if r.significant else "white", edgecolors=MODEL_COL[m])
    ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(-0.06, 0.06)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlabel("AUROC(alternative) − AUROC(VHSE LR)\n(filled: 95% interval excludes 0)")
    ax.set_title("Paired differences vs VHSE LR"); panel_letter(ax, "b", -0.08)
    ax.text(0.045, -0.75, "composition-LR\n(−ca. 0.2) off scale", fontsize=6.5, color=MUTE, ha="center")
    fig.tight_layout(); save(fig, "F05_model_comparison")


def fig_learning_curves():
    l = pd.read_csv(MET / "learning_curves.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.0), sharey=False)
    for ax, pid in zip(axes.ravel(), ORDER):
        s = l[l.protease == pid]
        ax.plot(s.n_train_pos, s["VHSE LR"], "o-", color=MODEL_COL["VHSE LR"], ms=3.5, lw=1.5, label="VHSE LR")
        ax.plot(s.n_train_pos, s["one-hot LR"], "s--", color=MODEL_COL["one-hot LR"], ms=3.5, lw=1.5, label="one-hot LR")
        ax.set_xscale("log"); ax.set_title(SHORT[pid], color=COL[pid], fontsize=9); ax.tick_params(labelsize=7)
        if pid in ("MMP2", "NE"):
            ax.set_ylabel("AUROC")
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("training positives", fontsize=8)
    fig.suptitle("Learning curves: VHSE is more sample-efficient below ~200 positives", fontsize=9.5, fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, "F06_learning_curves")


def fig_negative_design():
    a = pd.read_csv(MET / "negative_design_ablation.csv")
    designs = ["shuffled+background negatives", "background-only negatives", "panel negatives"]
    dcol = {"shuffled+background negatives": "#b8bec7", "background-only negatives": "#7b8fb2", "panel negatives": "#2a9d6f"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True)
    for ax, col, title in ((axes[0], "auroc_vs_background", "Target vs random background"), (axes[1], "auroc_vs_competitors", "Target vs other proteases' substrates")):
        for k, dsg in enumerate(designs):
            v = [a[(a.protease == p) & (a.design == dsg)][col].iloc[0] for p in ORDER]
            ax.bar(np.arange(len(ORDER)) + (k - 1) * 0.27, v, 0.26, color=dcol[dsg], label=dsg.replace(" negatives", ""))
        ax.axhline(0.5, color=MUTE, lw=0.8, ls=":"); ax.set_ylim(0.5, 1.0)
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([SHORT[p] for p in ORDER], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("held-out AUROC"); axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(1.1, -0.28), ncol=3, title="training negatives", title_fontsize=7)
    panel_letter(axes[0], "a", -0.16); panel_letter(axes[1], "b", -0.06)
    fig.tight_layout(); save(fig, "F07_negative_design")


def fig_nested_and_leakage():
    n = pd.read_csv(MET / "nested_regularisation.csv").set_index("protease").loc[ORDER]
    r = pd.read_csv(MET / "random_vs_cluster_cv.csv").set_index("protease").loc[ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]; x = np.arange(len(ORDER))
    ax.bar(x - 0.18, n.auroc_fixed_C1, 0.34, color="#b8bec7", label="C = 1 (fixed)")
    ax.bar(x + 0.18, n.auroc_nested, 0.34, color="#2a9d6f", label="C by inner CV (nested)")
    ax.set_ylim(0.7, 1.0); ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in ORDER], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("cluster-aware CV AUROC"); ax.legend(fontsize=7, loc="upper left"); ax.set_title("Regularisation selected inside each fold"); panel_letter(ax, "a", -0.16)
    ax = axes[1]
    ax.bar(x - 0.18, r.auroc_random_cv, 0.34, color="#b8bec7", label="random 5-fold")
    ax.bar(x + 0.18, r.auroc_cluster_cv, 0.34, color="#3d5a80", label="cluster-aware 5-fold")
    ax.set_ylim(0.7, 1.0); ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in ORDER], rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("Near-duplicate leakage check"); panel_letter(ax, "b", -0.12)
    fig.tight_layout(); save(fig, "F08_tuning_and_leakage")


def fig_roc_pr():
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for pid in ORDER:
        z = oof(pid); y, u = z["y"], z["u"]
        f, t, _ = roc_curve(y, u); axes[0].plot(f, t, color=COL[pid], lw=1.5, label=f"{SHORT[pid]} {roc_auc_score(y, u):.2f}")
        p, r, _ = precision_recall_curve(y, u); axes[1].plot(r, p, color=COL[pid], lw=1.5)
        axes[2].hist(u[y == 1], bins=20, range=(0, 1), histtype="step", color=COL[pid], lw=1.3, density=True)
    axes[0].plot([0, 1], [0, 1], color=MUTE, lw=0.8, ls=":"); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].legend(fontsize=6.5, loc="lower right", title="consensus AUROC", title_fontsize=6.5); axes[0].set_title("ROC (out-of-fold)")
    axes[1].axhline(0.25, color=MUTE, lw=0.8, ls=":"); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("Precision-recall (prior 0.25)")
    axes[2].set_xlabel("held-out positives: consensus percentile"); axes[2].set_ylabel("density"); axes[2].set_title("Positives vs background null")
    for ax, l in zip(axes, "abc"):
        panel_letter(ax, l, -0.2)
    fig.tight_layout(); save(fig, "F09_roc_pr")


def fig_calibration():
    cal = pd.read_csv(ROOT / "results" / "cv_calibration.csv")
    s = cv_summary()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTE, lw=0.9, ls=":")
    for pid in ORDER:
        c = cal[cal.protease == pid]
        ax.plot(c.mean_p, c.frac_pos, "o-", color=COL[pid], ms=3.5, lw=1.3, label=f"{SHORT[pid]} (ECE {s.loc[pid, 'lr_ece']:.2f})")
    ax.axhline(0.25, color=MUTE, lw=0.6, ls="--"); ax.text(0.02, 0.265, "training prior 0.25", fontsize=6.5, color=MUTE)
    ax.set_xlabel("mean out-of-fold LR score"); ax.set_ylabel("observed fraction of positives"); ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("LR scores are not absolute probabilities"); panel_letter(ax, "a", -0.14)
    ax = axes[1]
    from spred import candidates
    pool = candidates.random_pool(20000, seed=1234)
    base = [float(MODELS[p].prob(pool).mean()) for p in ORDER]
    ax.bar(range(len(ORDER)), base, color=[COL[p] for p in ORDER], width=0.62)
    for i, v in enumerate(base):
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([SHORT[p] for p in ORDER], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean LR score on random 8-mers"); ax.set_ylim(0, 0.42)
    ax.set_title("Base rates differ between models"); panel_letter(ax, "b", -0.16)
    fig.tight_layout(); save(fig, "F10_calibration")


def fig_selectivity_matrix():
    c = pd.read_csv(ROOT / "results" / "cv_competitor_auroc.csv")
    M = np.full((len(ORDER), len(ORDER)), np.nan)
    for r in c.itertuples():
        M[ORDER.index(r.target), ORDER.index(r.competitor)] = r.auroc
    fig, ax = plt.subplots(figsize=(4.7, 3.9))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(ORDER))); ax.set_yticks(range(len(ORDER)))
    ax.set_xticklabels([SHORT[p] for p in ORDER], rotation=45, ha="right"); ax.set_yticklabels([SHORT[p] for p in ORDER]); ax.grid(False)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            ax.text(j, i, "" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlabel("competitor whose substrates are the negatives"); ax.set_ylabel("target model")
    ax.set_title("Can the model tell the target's substrates\nfrom a competitor's? (held-out AUROC)")
    fig.colorbar(im, ax=ax, fraction=0.045, label="AUROC")
    save(fig, "F11_selectivity_matrix")


if __name__ == "__main__":
    fig_model_comparison(); fig_learning_curves(); fig_negative_design(); fig_nested_and_leakage(); fig_roc_pr(); fig_calibration(); fig_selectivity_matrix()
