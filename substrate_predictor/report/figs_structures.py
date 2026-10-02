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

    Two-pass, measured layout, the standard way to avoid overlap in a hand-built
    diagram: pass 1 renders each box's title and body text to a throwaway, invisible
    text artist and reads back its *actual* extent from the renderer, so box size is
    never guessed; pass 2 lays out columns by the measured width of the widest box in
    each column (not a fixed grid), so boxes can never collide with their neighbours
    either. Arrows are then drawn edge-to-edge between the real, final box geometries.
    """
    ACCENT = COL["MMP9"]
    TINT = "#eef1f4"
    ACCENT_TINT = "#eaf5ef"
    TITLE_FS, BODY_FS, PAD, LINE_GAP, TITLE_GAP = 8.6, 6.6, 4.2, 2.0, 2.0
    COL_GAP, ROW_GAP = 9.0, 7.5

    fig = plt.figure(figsize=(9.2, 4.0))
    ax = fig.add_axes((0, 0, 1, 1))  # axes fill the figure exactly: 1 data unit == 1 pt, no margin to account for
    ax.axis("off")
    fig.canvas.draw()  # realise a renderer so text extents are measurable
    renderer = fig.canvas.get_renderer()

    def text_size(s, fontsize, weight="normal"):
        """Rendered (width, height) of a string in *points*, independent of axes limits."""
        t = ax.text(0, 0, s, fontsize=fontsize, fontweight=weight, alpha=0)
        bbox = t.get_window_extent(renderer=renderer)
        t.remove()
        return bbox.width * 72 / fig.dpi, bbox.height * 72 / fig.dpi

    def measure(n, title, lines, accent=False):
        """Compute a box's size (in points) from its own content; placement comes later."""
        tw, th = text_size(title, TITLE_FS, "bold")
        line_sizes = [text_size(ln, BODY_FS) for ln in lines]
        body_w = max((w for w, _ in line_sizes), default=0)
        body_h = sum(h for _, h in line_sizes) + LINE_GAP * max(0, len(lines) - 1)
        badge_d = th + 1.0
        w = max(tw + badge_d + 5.0, body_w) + 2 * PAD
        h = PAD + th + TITLE_GAP + body_h + PAD
        return dict(n=n, title=title, lines=lines, line_sizes=line_sizes, w=w, h=h, badge_d=badge_d, accent=accent)

    def draw(node, x, y):
        """Draw a measured node with its top-left corner at (x, y); return its geometry."""
        w, h, badge_d = node["w"], node["h"], node["badge_d"]
        fc = ACCENT_TINT if node["accent"] else TINT
        ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                     fc=fc, ec=INK, lw=1.1 if node["accent"] else 0.8, zorder=2))
        r = badge_d / 2
        bcx, bcy = x + PAD + r, y - PAD - r
        ax.add_patch(plt.Circle((bcx, bcy), r, fc=ACCENT if node["accent"] else INK, ec="none", zorder=3))
        ax.text(bcx, bcy, str(node["n"]), ha="center", va="center", fontsize=7.0, fontweight="bold", color="white", zorder=4)
        ax.text(x + PAD + badge_d + (w - 2 * PAD - badge_d) / 2, y - PAD, node["title"],
                ha="center", va="top", fontsize=TITLE_FS, fontweight="bold", color=INK, zorder=4)
        _, th = text_size(node["title"], TITLE_FS, "bold")
        ty = y - PAD - th - TITLE_GAP
        for ln, (_, lh) in zip(node["lines"], node["line_sizes"]):
            ax.text(x + w / 2, ty, ln, ha="center", va="top", fontsize=BODY_FS, color=MUTE, zorder=4)
            ty -= lh + LINE_GAP
        return dict(x=x, y=y - h, w=w, h=h, cx=x + w / 2, cy=y - h / 2)

    def arrow(a, b, accent=False):
        acx, acy, bcx, bcy = a["cx"], a["cy"], b["cx"], b["cy"]
        if abs(bcx - acx) > abs(acy - bcy):
            p1 = (a["x"] + a["w"] if bcx > acx else a["x"], acy)
            p2 = (b["x"] if bcx > acx else b["x"] + b["w"], bcy)
        else:
            p1 = (acx, a["y"] if bcy < acy else a["y"] + a["h"])
            p2 = (bcx, b["y"] + b["h"] if bcy < acy else b["y"])
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, lw=1.0,
                                      color=ACCENT if accent else MUTE, shrinkA=2, shrinkB=2, zorder=1))

    # pass 1: measure every box from its own content
    m1 = measure(1, "Sources", ["MEROPS", "Eckhard 2016 (PICS)", "Yang 2015 (elastin)"])
    m2 = measure(2, "Positives", ["P4-P4′ windows per", "protease (7 trainable)"])
    m3 = measure(3, "Negatives", ["half: other proteases’", "positives, equal shares", "half: Swiss-Prot 8-mers"])
    m4 = measure(4, "Model per protease",
                 ["VHSE8 (64 features)", "logistic regression,", "C by inner CV", "+ PSSM (BLOSUM62", "pseudocounts)"])
    m5 = measure(5, "Background nulls", ["100,000 random 8-mers", "→ score percentiles,", "comparable across models"], accent=True)
    m6 = measure(6, "Panel specificity",
                 ["consensus: harmonic mean", "of LR and PSSM percentiles", "margin vs strongest", "competitor"], accent=True)

    # pass 2: lay out by column, each column's width set by its widest box
    columns = [[m1], [m2, m3], [m4], [m5, m6]]
    col_w = [max(b["w"] for b in col) for col in columns]
    stack_h = [sum(b["h"] for b in col) + ROW_GAP * (len(col) - 1) for col in columns]
    top = max(stack_h)
    x = 1.0
    geo = {}
    for col, w, sh in zip(columns, col_w, stack_h):
        y = top - (top - sh) / 2  # vertically centre this column's stack in the tallest column
        for b in col:
            cx = x + w / 2
            geo[b["n"]] = draw(b, cx - b["w"] / 2, y)
            y -= b["h"] + ROW_GAP
        x += w + COL_GAP

    arrow(geo[1], geo[2]); arrow(geo[2], geo[3])
    arrow(geo[2], geo[4]); arrow(geo[3], geo[4])
    arrow(geo[4], geo[5], accent=True); arrow(geo[4], geo[6], accent=True)
    arrow(geo[5], geo[6], accent=True)

    total_w = x - COL_GAP + 1.0
    foot_h = 13.0
    total_h = top + foot_h
    ax.set_xlim(0, total_w)
    ax.set_ylim(-foot_h, top)
    ax.plot([1, total_w - 1], [-foot_h * 0.45, -foot_h * 0.45], color=GRID, lw=1.0, zorder=1)
    ax.text(total_w / 2, -foot_h * 0.72, "Evaluation: similarity-clustered 5-fold CV · nested regularisation · "
                                          "cluster-bootstrap CIs · external validation on measured kinetics (Kridel 2001)",
            ha="center", fontsize=6.8, color=MUTE, style="italic")
    # 1 data unit == 1 pt: figure size in inches must equal the data extent in points / 72
    fig.set_size_inches(total_w / 72, total_h / 72)
    save(fig, "F01_pipeline")


if __name__ == "__main__":
    fig_pipeline(); fig_structures()
