"""Figures on data provenance, sequence logos and protease similarity."""
from matplotlib import patches
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch

from fig_common import *

AA_COLOR = {**{a: "#1a1a1a" for a in "AVLIMFWC"}, **{a: "#2a9d5b" for a in "GSTYNQ"}, **{a: "#1f5fbf" for a in "KRH"},
            **{a: "#d1495b" for a in "DE"}, "P": "#e08a00"}
FP = FontProperties(family="DejaVu Sans", weight="bold")


def draw_letter(ax, ch, x, y, w, h):
    tp = TextPath((0, 0), ch, size=1, prop=FP)
    bb = tp.get_extents()
    tr = Affine2D().translate(-bb.x0, -bb.y0).scale(w / bb.width, h / max(bb.height, 1e-9)).translate(x, y)
    ax.add_patch(PathPatch(tr.transform_path(tp), fc=AA_COLOR.get(ch, "#444"), ec="none"))


def logo(ax, lo, ymax=None):
    p = pssm_mod.BACKGROUND * 2.0 ** lo                     # position-specific probabilities
    ic = (p * lo).sum(axis=1)                               # bits
    for pos in range(8):
        order = np.argsort(p[pos])
        y = 0.0
        for j in order:
            h = p[pos, j] * ic[pos]
            if h > 0.03:
                draw_letter(ax, AMINO_ACIDS[j], pos + 0.06, y, 0.88, h)
            y += h
    ymax = ymax or float(np.ceil(ic.max() * 1.15 * 4) / 4)
    ax.set_xlim(0, 8); ax.set_ylim(0, ymax); ax.set_yticks(np.arange(0, ymax + 1e-9, 0.25 if ymax <= 1.25 else 1.0 if ymax <= 5 else 2.0))
    ax.set_xticks(np.arange(8) + 0.5); ax.set_xticklabels(POSITIONS, fontsize=8)
    ax.grid(False); ax.axvline(4, color=MUTE, lw=0.8, ls=(0, (3, 3)))


def fig_logos():
    fig, axes = plt.subplots(len(ORDER), 1, figsize=(6.4, 1.32 * len(ORDER)))
    for ax, pid in zip(axes, ORDER):
        logo(ax, MODELS[pid].log_odds)
        ax.text(-0.13, 0.5, f"{SHORT[pid]}\n(n={MODELS[pid].meta['n_pos']:,})", transform=ax.transAxes, ha="right", va="center",
                fontsize=8.5, fontweight="bold", color=COL[pid])
        if pid != ORDER[-1]:
            ax.set_xticklabels([])
    axes[0].set_title("Substrate specificity logos (bits; P4-P4'; dashed line = scissile bond; each row has its own scale)", fontsize=9, loc="left")
    save(fig, "F03_sequence_logos")


def fig_data_composition():
    pos = {pid: pd.read_csv(ROOT / "data" / "positives" / f"{pid}.csv") for pid in ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    srcs = ["MEROPS", "Eckhard2016", "Yang2015"]
    cols = {"MEROPS": "#3d5a80", "Eckhard2016": "#98c1d9", "Yang2015": "#ee6c4d"}
    left = np.zeros(len(ORDER))
    for s in srcs:
        v = np.array([(pos[p].source == s).sum() for p in ORDER])
        ax.barh(range(len(ORDER)), v, left=left, color=cols[s], label=s.replace("2016", " 2016").replace("2015", " 2015"), height=0.62)
        left += v
    for i, p in enumerate(ORDER):
        ax.text(left[i] + left.max() * 0.015, i, f"{int(left[i]):,}", va="center", fontsize=8)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[p] for p in ORDER]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(60, 12000); ax.set_xlabel("confirmed P4-P4' windows (log scale)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.5); ax.set_title("Positive windows by source"); panel_letter(ax, "a", -0.22)
    ax = axes[1]
    lk = pd.read_csv(ROOT / "results" / "methodology" / "random_vs_cluster_cv.csv").set_index("protease").loc[ORDER]
    ax.barh(range(len(ORDER)), lk.frac_pos_in_multi_member_cluster * 100, color=[COL[p] for p in ORDER], height=0.62)
    for i, (v, m) in enumerate(zip(lk.frac_pos_in_multi_member_cluster * 100, lk.largest_cluster)):
        ax.text(v + 0.5, i, f"{v:.0f}%  (max cluster {m})", va="center", fontsize=7.5)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlim(0, 45)
    ax.set_xlabel("% of positives with a near-duplicate\n(Hamming distance ≤ 2)"); ax.set_title("Near-duplicate structure"); panel_letter(ax, "b", -0.08)
    fig.tight_layout(); save(fig, "F02_data_composition")


def fig_similarity():
    n = len(ORDER)
    J = np.zeros((n, n)); P = np.zeros((n, n))
    sets = {p: set(pd.read_csv(ROOT / "data" / "positives" / f"{p}.csv").sequence) for p in ORDER}
    lo = {p: MODELS[p].log_odds.ravel() for p in ORDER}
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            J[i, j] = len(sets[a] & sets[b]) / len(sets[a] | sets[b]) if i != j else np.nan
            P[i, j] = np.corrcoef(lo[a], lo[b])[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for ax, M, title, cmap, vmax, fmt in ((axes[0], J, "Shared positive windows (Jaccard)", "Blues", np.nanmax(J), "{:.3f}"),
                                          (axes[1], P, "PSSM similarity (Pearson r of log-odds)", "RdBu_r", 1, "{:.2f}")):
        im = ax.imshow(M, cmap=cmap, vmin=-vmax if cmap == "RdBu_r" else 0, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([SHORT[p] for p in ORDER], rotation=45, ha="right", fontsize=8); ax.set_yticklabels([SHORT[p] for p in ORDER], fontsize=8)
        ax.grid(False); ax.set_title(title, fontsize=9)
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.6, color="white" if abs(M[i, j]) > 0.6 * vmax else INK)
    panel_letter(axes[0], "a", -0.2, 1.1); panel_letter(axes[1], "b", -0.2, 1.1)
    fig.tight_layout(); save(fig, "F04_protease_similarity")


if __name__ == "__main__":
    fig_logos(); fig_data_composition(); fig_similarity()
