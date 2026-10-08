"""Figures on learned specificity, external validation, candidate ranking and structures."""
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from fig_common import *
from spred import candidates, validation as V
from spred.features import encode
from spred.panel import Panel

AA_ORDER = list("AVLIMFWYCGPSTNQHKRDE")   # hydrophobic / aromatic / small / polar / basic / acidic


def fig_coefficients():
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 9.2))
    norm = TwoSlopeNorm(0, -1, 1)
    for ax, pid in zip(axes.ravel(), ORDER):
        z = np.load(ROOT / "results" / "coefficients" / f"{pid}.npz")
        c = z["coef"].reshape(8, 8); lo = z["lo"].reshape(8, 8); hi = z["hi"].reshape(8, 8)
        scale = np.abs(c).max()
        im = ax.imshow((c / scale).T, cmap="RdBu_r", norm=norm, aspect="auto")
        stable = ((lo > 0) | (hi < 0)).T
        for i in range(8):
            for j in range(8):
                if stable[i, j]:
                    ax.text(j, i, "•", ha="center", va="center", fontsize=9, color="white" if abs(c[j, i] / scale) > 0.55 else INK)
        ax.set_xticks(range(8)); ax.set_xticklabels(POSITIONS, fontsize=7.5)
        ax.set_yticks(range(8)); ax.set_yticklabels([f"{i + 1}" for i in range(8)], fontsize=7.5); ax.grid(False)
        ax.set_title(f"{LABEL[pid] if pid != 'FECAL' else 'Digestive serine'}   (max |coef| = {scale:.2f})", color=COL[pid], fontsize=8.6, loc="left")
        for b in (1.5, 3.5):
            ax.axhline(b, color="white", lw=1.6)
        if pid in ("MMP2", "MMP9", "NE", "FECAL"):
            ax.set_ylabel("VHSE descriptor", fontsize=8)
    ax = axes.ravel()[-1]; ax.axis("off")
    cax = ax.inset_axes([0.12, 0.62, 0.76, 0.06])
    fig.colorbar(im, cax=cax, orientation="horizontal"); cax.set_title("standardised coefficient (fraction of row maximum)", fontsize=7.5)
    ax.text(0.12, 0.42, "•  95% cluster-bootstrap interval excludes 0\n\nDescriptor rows: 1-2 hydrophobic, 3-4 steric,\n5-8 electronic (white rules separate the groups)", fontsize=7.5, transform=ax.transAxes, va="top")
    fig.tight_layout(); save(fig, "F12_coefficients")


def fig_position_importance():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    pos = data.trainable_positives()
    ax = axes[0]
    for pid in ORDER:
        m = MODELS[pid]
        C = np.array([m.contributions(s) for s in pos[pid][:800]])
        imp = np.abs(C).mean(axis=0); imp = imp / imp.sum()
        ax.plot(range(8), imp, "o-", color=COL[pid], lw=1.5, ms=4, label=SHORT[pid])
    ax.set_xticks(range(8)); ax.set_xticklabels(POSITIONS); ax.axvline(3.5, color=MUTE, lw=0.8, ls=(0, (3, 3)))
    ax.set_ylabel("share of |contribution|"); ax.set_title("Which positions drive the score"); ax.legend(fontsize=6.8, ncol=2, loc="upper right")
    panel_letter(ax, "a", -0.14)
    ax = axes[1]
    grp = {"hydrophobic": (0, 1), "steric": (2, 3), "electronic": (4, 5, 6, 7)}
    bottom = np.zeros(len(ORDER)); gcol = {"hydrophobic": "#e8a33d", "steric": "#5b8bb5", "electronic": "#a0556d"}
    for g, idx in grp.items():
        v = []
        for pid in ORDER:
            m = MODELS[pid]; X = (encode(pos[pid][:800]) - m.mean) / m.scale
            contrib = (X * m.coef).reshape(len(X), 8, 8)
            tot = np.abs(contrib).sum(axis=(1, 2)).mean()
            v.append(np.abs(contrib[:, :, list(idx)]).sum(axis=(1, 2)).mean() / tot)
        ax.barh(range(len(ORDER)), v, left=bottom, color=gcol[g], label=g, height=0.62); bottom += np.array(v)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[p] for p in ORDER]); ax.invert_yaxis(); ax.set_xlabel("share of |contribution|")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3); ax.set_title("By descriptor family"); panel_letter(ax, "b", -0.2)
    fig.tight_layout(); save(fig, "F13_position_importance")


def fig_kridel_consensus():
    lo = MODELS["MMP9"].log_odds
    cols = [AA_ORDER.index(a) for a in AA_ORDER]
    M = np.array([[lo[p, list("ACDEFGHIKLMNPQRSTVWY").index(a)] for a in AA_ORDER] for p in range(8)])
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -2.5, 2.9), aspect="auto")
    ax.set_xticks(range(20)); ax.set_xticklabels(AA_ORDER); ax.set_yticks(range(8)); ax.set_yticklabels(POSITIONS); ax.grid(False)
    consensus = {2: "P", 3: "R", 5: "L", 4: "S", 6: "S"}          # not used for drawing (see below)
    marks = [(1, "P", "Pro"), (2, "R", "Arg"), (3, "S", "Ser/Thr"), (3, "T", None), (4, "L", "Leu/Ile"), (4, "I", None), (5, "S", "Ser/Thr"), (5, "T", None)]
    for pos, aa, _ in marks:
        ax.add_patch(plt.Rectangle((AA_ORDER.index(aa) - 0.5, pos - 0.5), 1, 1, fill=False, ec="#111", lw=1.8))
    for p in range(8):
        for a in range(20):
            if M[p, a] > 1.0 or M[p, a] < -1.5:
                ax.text(a, p, f"{M[p, a]:+.1f}", ha="center", va="center", fontsize=5.6, color="white" if abs(M[p, a]) > 1.6 else INK)
    ax.axhline(3.5, color="#111", lw=1.2, ls=(0, (3, 3)))
    ax.set_title("Learned MMP-9 PSSM (log2 odds vs background). Boxes: residues in Kridel's phage-display consensus, Pro-Arg-Ser/Thr↓Hy-Ser/Thr", fontsize=8.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="log2 odds")
    save(fig, "F14_mmp9_consensus")


def fig_kridel_validation():
    """External validation scatter. No in-plot stats box (redundant with Table 4 and it
    previously hid the C15 point); crowded labels (A7/A19/B1 in both panels, A11/A6 in
    lr_logit) get a small offset plus a thin leader line back to their true point, so a
    displaced label is never mistaken for sitting next to the wrong marker."""
    df = pd.read_csv(ROOT / "results" / "validation" / "kridel_scores.csv")
    # per-panel (dx, dy) label offsets in points; only peptides crowded enough to need
    # displacement are listed, everything else uses the small default.
    OFF_LR = {"A7": (-28, 10), "A19": (6, -13), "B1": (6, 12), "A11": (6, 10), "A6": (-26, -12)}
    OFF_PSSM = {"A7": (-28, 10), "A19": (6, -13), "B1": (6, 12)}
    LEADER = {"A7", "A19", "B1", "A11", "A6"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, col, title, offsets in (
        (axes[0], "lr_logit", "Logistic regression (partial logit)", OFF_LR),
        (axes[1], "pssm", "PSSM (partial score)", OFF_PSSM),
    ):
        m = df[df.status == "measured"]; n = df[df.status != "measured"]
        ax.scatter(m[col], np.log10(m.kcat_km), s=42, color=COL["MMP9"], zorder=3)
        for r in m.itertuples():
            x, y = getattr(r, col), np.log10(r.kcat_km)
            dx, dy = offsets.get(r.peptide, (4, 3))
            ax.annotate(r.peptide, (x, y), fontsize=6.6, xytext=(dx, dy), textcoords="offset points", color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6, shrinkA=1, shrinkB=1) if r.peptide in LEADER else None)
        ax.scatter(n[col], [np.log10(1000)] * len(n), s=42, facecolors="white", edgecolors=COL["MMP9"], zorder=3)
        for r in n.itertuples():
            ax.annotate(f"{r.peptide}\n(not detected)", (getattr(r, col), np.log10(1000)), fontsize=6.3, xytext=(4, -14), textcoords="offset points", color=MUTE)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("model score"); ax.set_ylabel(r"measured $\log_{10}$ $k_{cat}/K_m$ (MMP-9)")
    panel_letter(axes[0], "a", -0.16); panel_letter(axes[1], "b", -0.16)
    fig.tight_layout(); save(fig, "F15_kridel_validation")


def fig_score_distributions():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    for ax, pid in zip(axes, ("MMP9", "NE")):
        z = oof(pid); src = z["source"]; y = z["y"]; u = z["u"]
        groups = [("target", (y == 1))] + [(SHORT[c], (src == c) & (y == 0)) for c in ORDER if c != pid and ((src == c) & (y == 0)).sum() > 20] + [("random", src == "background")]
        parts = ax.violinplot([u[m] for _, m in groups], positions=range(len(groups)), showmedians=True, widths=0.85)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(COL[pid] if i == 0 else "#9FAEA7" if groups[i][0] != "random" else "#D9DED6"); b.set_alpha(0.85); b.set_edgecolor("none")
        for k in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[k].set_color(INK); parts[k].set_linewidth(0.8)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("out-of-fold consensus percentile"); ax.set_title(f"{LABEL[pid]} model", color=COL[pid])
    panel_letter(axes[0], "a", -0.14); panel_letter(axes[1], "b", -0.14)
    fig.tight_layout(); save(fig, "F16_score_distributions")


def _pool():
    rng_pool = candidates.random_pool(150_000, seed=11)
    extra = []
    for pid in ("MMP2", "MMP8", "MMP9", "NE", "LASB"):
        extra += candidates.pssm_top_k(MODELS[pid].log_odds, 3)
    return [x for x in dict.fromkeys(rng_pool + extra) if candidates.sensor_ok(x)]


def fig_candidate_landscape():
    panel = Panel(("MMP2", "MMP8", "MMP9", "NE", "LASB"), MODELS)
    pool = _pool()
    sc = panel.score(pool)
    rows = []
    for t in panel.members:
        s = sc.copy(); s["margin"] = panel.margin(s, t); s["target"] = t
        s = s[s[f"u_{t}"] >= 0.9].sort_values("margin", ascending=False).head(4)
        rows.append(s)
    top = pd.concat(rows, ignore_index=True)
    ref = panel.score_gapped(["IPRTLTA-", "-PRSLSGL", "-PLGLAGQ", "AAPVSGIG"])
    ref["target"] = ["MMP9", "MMP9", "MMP2", "NE"]; ref["margin"] = [panel.margin(ref.iloc[[i]], ref.target.iloc[i])[0] for i in range(len(ref))]
    top.to_csv(ROOT / "results" / "top_candidates_wound_panel.csv", index=False); ref.to_csv(ROOT / "results" / "reference_peptides_wound_panel.csv", index=False)
    allr = pd.concat([top, ref], ignore_index=True)
    labels = [f"{r.sequence}  → {SHORT[r.target]}" for r in allr.itertuples()]
    M = allr[[f"u_{m}" for m in panel.members]].values
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(panel.members))); ax.set_xticklabels([SHORT[m] for m in panel.members]); ax.xaxis.tick_top(); ax.grid(False)
    ax.set_yticks(range(len(allr))); ax.set_yticklabels(labels, fontsize=7.5)
    for i, r in enumerate(allr.itertuples()):
        for j, m in enumerate(panel.members):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 0.6 else INK)
            if m == r.target:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#ff5d5d", lw=2))
        ax.text(len(panel.members) - 0.35, i, f"margin {r.margin:+.2f}", va="center", fontsize=7, color=INK)
    ax.axhline(len(top) - 0.5, color="white", lw=4)
    for k in range(1, 5):
        ax.axhline(4 * k - 0.5, color="white", lw=1.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.16, label="consensus background percentile")
    ax.set_title("Top-ranked sensor-compatible candidates per target (150k random + PSSM top-3 pools) and four\nreference cores below the white bar (gapped windows). Red box = intended target", fontsize=8.5, pad=34)
    save(fig, "F17_candidate_landscape")


def fig_mutation_scan(seq="IPRTLTA-", target="MMP9"):
    panel = Panel(("MMP2", "MMP8", "MMP9", "NE", "LASB"), MODELS)
    model = MODELS[target]
    known = [i for i, c in enumerate(seq) if c != "-"]
    wm = float(panel.margin(panel.score_gapped([seq]), target)[0])
    def logit(w):
        f, miss = V.fill_gaps(w); p = float(V.masked_prob(model, [f], miss)[0]); return np.log(p / (1 - p))
    wl = logit(seq)
    wins, keys = [], []
    for i in known:
        for a in AMINO_ACIDS:
            if a != seq[i]:
                wins.append(seq[:i] + a + seq[i + 1:]); keys.append((i, a))
    mg = panel.margin(panel.score_gapped(wins), target)
    A = np.full((20, 8), np.nan); B = np.full((20, 8), np.nan)
    for (i, a), w, v in zip(keys, wins, mg):
        A[AA_ORDER.index(a), i] = logit(w) - wl; B[AA_ORDER.index(a), i] = v - wm
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.6))
    for ax, M, lim, lab, title in ((axes[0], A, 3.0, f"change in {SHORT[target]} LR log-odds", f"(a) Target score ({SHORT[target]} logit)"),
                                   (axes[1], B, 0.6, f"change in specificity margin", "(b) Specificity margin (wound panel)")):
        im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        ax.set_xticks(range(8)); ax.set_xticklabels([f"{POSITIONS[i]}\n{seq[i]}" for i in range(8)], fontsize=7.5); ax.set_yticks(range(20)); ax.set_yticklabels(AA_ORDER, fontsize=7.5); ax.grid(False)
        for i in known:
            ax.add_patch(plt.Rectangle((i - .5, AA_ORDER.index(seq[i]) - .5), 1, 1, fill=True, fc="#ddd", ec="#111", lw=1.1))
            ax.text(i, AA_ORDER.index(seq[i]), "wt", ha="center", va="center", fontsize=6)
        for i in range(8):
            if i not in known:
                ax.add_patch(plt.Rectangle((i - .5, -.5), 1, 20, fill=True, fc="#f3f3f3", ec="none", hatch="///", alpha=0.5))
        fig.colorbar(im, ax=ax, fraction=0.05, label=lab); ax.set_title(title, fontsize=9)
    fig.suptitle(f"Single-substitution scan of {seq.replace('-', '')} (P4' unknown; hatched)", fontsize=9.5, fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, "F18_mutation_scan")


if __name__ == "__main__":
    fig_coefficients(); fig_position_importance(); fig_kridel_consensus(); fig_kridel_validation(); fig_score_distributions()
    fig_candidate_landscape(); fig_mutation_scan()
