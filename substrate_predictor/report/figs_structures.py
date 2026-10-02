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
    tag = LABEL[pid] if pid != "FECAL" else "Digestive serine"
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
    """Six-stage pipeline schematic, source data to panel selectivity score.

    Design: a genuine editorial diagram rather than a default flowchart-tool render --
    numbered stages (matching the report's own part-numbering convention), restrained
    ink-tinted fills instead of saturated pastels, one accent colour reserved for the
    two terminal outputs, and hand-set arrow routing rather than auto-layout connectors.
    """
    ACCENT = COL["MMP9"]
    TINT = "#eef1f4"
    fig, ax = plt.subplots(figsize=(7.8, 3.5)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)

    def box(x, y, w, h, n, title, sub, fc=TINT, ec=INK, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                     fc=fc, ec=ec, lw=1.1 if accent else 0.8, zorder=2))
        r = 1.55
        ax.add_patch(plt.Circle((x + r + 0.6, y + h - r - 0.6), r, fc=ACCENT if accent else INK, ec="none", zorder=3))
        ax.text(x + r + 0.6, y + h - r - 0.6, str(n), ha="center", va="center", fontsize=7.2,
                fontweight="bold", color="white", zorder=4)
        ax.text(x + w / 2 + 1.1, y + h - 2.7, title, ha="center", va="top", fontsize=8.1,
                fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="top", fontsize=6.4, color=MUTE,
                linespacing=1.45, zorder=4)

    def arrow(x1, y1, x2, y2, accent=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                      lw=1.0, color=ACCENT if accent else MUTE,
                                      connectionstyle="arc3,rad=0.0", zorder=1))

    box(1, 29, 20, 15, 1, "Sources", "MEROPS\nEckhard 2016 (PICS)\nYang 2015 (elastin)")
    box(27, 29, 20, 15, 2, "Positives", "P4-P4′ windows per\nprotease (7 trainable)")
    box(27, 6, 20, 17, 3, "Negatives", "half: other proteases’\npositives, equal shares\nhalf: Swiss-Prot 8-mers")
    box(53, 17.5, 21, 21, 4, "Model per protease", "VHSE8 (64 features)\nlogistic regression,\nC by inner CV\n+ PSSM (BLOSUM62\npseudocounts)")
    box(79, 29, 20, 15, 5, "Background nulls", "100,000 random 8-mers\n→ score percentiles,\ncomparable across models", fc="#eaf5ef", accent=True)
    box(79, 6, 20, 19, 6, "Panel specificity", "consensus: harmonic mean\nof LR and PSSM percentiles\nmargin vs strongest\ncompetitor", fc="#eaf5ef", accent=True)

    arrow(21.3, 36.5, 26.7, 36.5)
    arrow(37, 29, 37, 23)
    arrow(47.3, 36.5, 52.7, 32)
    arrow(47.3, 14.5, 52.7, 22)
    arrow(74.3, 32, 78.7, 36.5, accent=True)
    arrow(74.3, 22, 78.7, 15.5, accent=True)
    arrow(89, 29, 89, 25.5, accent=True)

    ax.plot([1, 99], [1.9, 1.9], color=GRID, lw=1.0, zorder=1)
    ax.text(50, 0.2, "Evaluation: similarity-clustered 5-fold CV · nested regularisation · cluster-bootstrap CIs · "
                      "external validation on measured kinetics (Kridel 2001)",
            ha="center", fontsize=6.8, color=MUTE, style="italic")
    save(fig, "F01_pipeline")


if __name__ == "__main__":
    fig_pipeline(); fig_structures()
