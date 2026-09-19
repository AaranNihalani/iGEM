"""F19: protease structures (C-alpha traces, orthographic, oriented toward the active site) and F01: pipeline schematic."""
import json

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from Bio.PDB import PDBParser

from fig_common import *

META = json.load(open(ROOT / "data" / "structures" / "structures.json"))
STRUCT_ORDER = ["MMP2", "MMP8", "MMP9", "NE", "LASB"]


def _rotation_to(v):
    v = v / np.linalg.norm(v); z = np.array([0, 0, 1.0]); ax = np.cross(v, z); s = np.linalg.norm(ax); c = v @ z
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def draw_structure(ax, pid):
    m = META[pid]
    s = PDBParser(QUIET=True).get_structure(pid, ROOT / "data" / "structures" / "processed" / f"{pid}.pdb")[0]
    ca, ca_res, lig, zn = [], [], [], None
    pocket = {int("".join(c for c in x if c.isdigit())) for x in m["pocket_residues"]}
    for chain in s:
        for res in chain:
            name = res.get_resname().strip()
            if res.id[0] == " " and "CA" in res and chain.id != "I":
                ca.append(res["CA"].coord); ca_res.append(res.id[1])
            elif name == "ZN":
                zn = res["ZN"].coord if zn is None or (m["catalytic_zinc_coord"] and np.allclose(res["ZN"].coord, m["catalytic_zinc_coord"], atol=0.2)) else zn
            elif name not in ("CA", "HOH"):
                lig += [a.coord for a in res if a.element != "H"]
    ca = np.array(ca); ca_res = np.array(ca_res); lig = np.array(lig) if lig else None
    focus = np.mean(lig, axis=0) if lig is not None else (np.array(m["catalytic_zinc_coord"]) if m["catalytic_zinc_coord"] else ca.mean(axis=0))
    R = _rotation_to(focus - ca.mean(axis=0))
    P = (ca - ca.mean(axis=0)) @ R.T
    pts = P[:, :2]; depth = P[:, 2]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    gap = np.linalg.norm(ca[1:] - ca[:-1], axis=1) < 4.2                     # do not join across chain breaks
    d = (depth[:-1] + depth[1:]) / 2; d = (d - d.min()) / (np.ptp(d) + 1e-9)
    colors = plt.cm.YlGnBu(0.25 + 0.7 * (np.arange(len(segs)) / len(segs)))
    colors[:, 3] = 0.35 + 0.65 * d
    lc = LineCollection(segs[gap], colors=colors[gap], linewidths=1.0 + 1.9 * d[gap]); ax.add_collection(lc)
    isp = np.array([r in pocket for r in ca_res])
    ax.scatter(pts[isp, 0], pts[isp, 1], s=14, color="#e8590c", zorder=4, alpha=0.95, lw=0)
    if lig is not None:
        L = (lig - ca.mean(axis=0)) @ R.T
        ax.scatter(L[:, 0], L[:, 1], s=9, color="#111", zorder=6)
    if zn is not None:
        Z = (zn - ca.mean(axis=0)) @ R.T; ax.scatter(Z[0], Z[1], s=90, color="#d4af37", ec="#7a5c00", lw=1, zorder=7)
    ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
    ax.set_xlim(pts[:, 0].min() - 3, pts[:, 0].max() + 3); ax.set_ylim(pts[:, 1].min() - 3, pts[:, 1].max() + 3)
    tag = LABEL[pid] if pid != "FECAL" else "Faecal"
    ax.set_title(f"{tag}   PDB {m['pdb']}", fontsize=9, color=COL[pid], loc="left")
    cat = ", ".join(m["catalytic_metal_ligands"])
    ax.text(0.0, -0.02, f"{m['resolution']} Å · {m['n_residues']} residues\ncatalytic site: {cat}", transform=ax.transAxes, fontsize=6.6, va="top", color=INK)


def fig_structures():
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.6))
    for ax, pid in zip(axes.ravel(), STRUCT_ORDER):
        draw_structure(ax, pid)
    ax = axes.ravel()[-1]; ax.axis("off")
    handles = [Line2D([0], [0], color=plt.cm.YlGnBu(0.6), lw=2, label="Cα trace (N to C terminus: light to dark)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#e8590c", ms=6, label="pocket residue (Cα)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#111", ms=5, label="bound inhibitor atoms"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#d4af37", markeredgecolor="#7a5c00", ms=9, label="catalytic zinc")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.2)
    ax.text(0, 0.42, "Each view looks down onto the active site.\nPocket: residues within 6 Å of the bound\ninhibitor (or 9 Å of the catalytic zinc when apo).\nThese are experimental structures; they are\nshown for context and are not model output.", fontsize=7, transform=ax.transAxes, va="top", color=MUTE)
    fig.tight_layout(); save(fig, "F19_structures")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.6, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 44)
    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2", fc=fc, ec="#1c2430", lw=0.9))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, sub, ha="center", va="top", fontsize=6.2, color="#2b3646", linespacing=1.25)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#1c2430"))
    box(1, 28, 20, 14, "Sources", "MEROPS\nEckhard 2016 (PICS)\nYang 2015 (elastin)", "#e8eef7")
    box(26, 28, 20, 14, "Positives", "P4-P4' windows\nper protease\n(7 trainable)", "#dde9f5")
    box(26, 6, 20, 16, "Negatives", "half: other proteases'\npositives (equal shares)\nhalf: Swiss-Prot 8-mers", "#f5e6d8")
    box(51, 17, 21, 20, "Model per protease", "VHSE8 (64 features)\nlogistic regression\nC by inner CV\n+ PSSM (BLOSUM62\npseudocounts)", "#dcefe3")
    box(77, 28, 21, 14, "Background nulls", "100,000 random 8-mers\n→ score percentiles\n(comparable across models)", "#efe6f5")
    box(77, 6, 21, 18, "Panel specificity", "consensus = harmonic\nmean of LR and PSSM\npercentiles\nmargin vs strongest\ncompetitor", "#f7ecd0")
    arrow(21.5, 35, 25.5, 35); arrow(36, 27.5, 36, 22.7); arrow(46.5, 35, 51, 30); arrow(46.5, 14, 51, 22); arrow(72.5, 30, 76.5, 34); arrow(72.5, 22, 76.5, 17); arrow(87.5, 27.5, 87.5, 24.5)
    ax.text(50, 1.2, "Evaluation: similarity-clustered 5-fold CV · nested regularisation · cluster-bootstrap CIs · external validation on measured kinetics (Kridel 2001)",
            ha="center", fontsize=6.6, color=INK, style="italic")
    save(fig, "F01_pipeline")


if __name__ == "__main__":
    fig_pipeline(); fig_structures()
