"""Figures for Part I (device and measurement), Part II (surface) and the
protocol overview. Drawn schematics are to scale in nm where stated."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle, FancyArrowPatch

import swv_forward_model as m
import report_lib as L
from report_lib import C, panel, save, num

SEQ = "SGKIPRTLTA"


def _zigzag(x0, y0, y1, amp=0.07, n=12):
    ys = np.linspace(y0, y1, n * 2 + 1)
    xs = x0 + amp * np.array([0] + [(-1) ** i for i in range(n * 2)])[: len(ys)]
    return xs, ys


def _peptide_coil(x0, y0, n=10, seed=3, step=0.38, wall=None):
    """Illustrative coil: fixed step directions, upward on average."""
    angles = np.deg2rad([80, 35, 125, 70, 15, 110, 55, 140, 85, 50])[:n]
    if seed % 2 == 0:
        angles = np.pi - angles
    pts = [(x0, y0)]
    for a in angles:
        pts.append((pts[-1][0] + step * np.cos(a), pts[-1][1] + step * np.sin(a)))
    return np.array(pts)


def _draw_fc(ax, x, y, s=0.14):
    """Ferrocene: two rings sandwiching an iron atom."""
    ax.add_patch(Ellipse((x, y - 0.11), 0.30, 0.09, fc=C["orange"], ec="none", zorder=4))
    ax.add_patch(Circle((x, y), 0.055, fc="#7A4E1D", ec="none", zorder=5))
    ax.add_patch(Ellipse((x, y + 0.11), 0.30, 0.09, fc=C["orange"], ec="none", zorder=4))


def _draw_mb(ax, x, y, faded=False):
    a = 0.35 if faded else 1.0
    ax.add_patch(Circle((x, y), 0.20, fc=C["blue"], ec=C["navy"], lw=0.8, alpha=a, zorder=6))
    ax.text(x, y, "MB", color="white", fontsize=6.5, fontweight="bold",
            ha="center", va="center", zorder=7, alpha=a)


def _bead(ax, x, y, ch, col, faded=False):
    a = 0.4 if faded else 1.0
    ax.add_patch(Circle((x, y), 0.135, fc=col, ec="white", lw=0.5, alpha=a, zorder=5))
    ax.text(x, y, ch, color="white", fontsize=5.5, ha="center", va="center",
            zorder=6, alpha=a, fontweight="bold")


def _probe(ax, x0, cleaved, seed):
    """One MB-peptide-PEG6 probe rooted at x0. Heights in nm."""
    ax.plot(*_zigzag(x0, 0.0, m.D_PEG_NM, amp=0.09, n=6), color=C["grey"], lw=2.2, zorder=3)
    pts = _peptide_coil(x0, m.D_PEG_NM, seed=seed)
    cut = 7
    for i in range(len(pts) - 1):
        gone = cleaved and i >= cut
        ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], color=C["navy"], lw=1.3,
                alpha=0.3 if gone else 1, ls=":" if gone else "-", zorder=3)
    shift = np.array([0.9, 0.7]) if cleaved else np.array([0.0, 0.0])
    for i, ch in enumerate(SEQ):
        p = (pts[i] + pts[i + 1]) / 2
        gone = cleaved and i >= cut
        col = C["purple"] if i < cut else C["green"]
        if gone:
            p = p + shift
        _bead(ax, p[0], p[1], ch, col, faded=gone)
    mb = pts[-1] + (shift if cleaved else 0)
    _draw_mb(ax, mb[0], mb[1] + 0.05, faded=cleaved)
    if not cleaved:
        mid = (pts[cut - 1] + pts[cut]) / 2
        ax.plot([mid[0] - 0.18, mid[0] + 0.18], [mid[1] + 0.2, mid[1] - 0.2],
                color=C["red"], lw=1.6, zorder=8)
    return pts, mb


def _slab(ax, xmax):
    ax.add_patch(Rectangle((-0.3, -0.55), xmax + 0.6, 0.55, fc=C["gold"], ec="none", zorder=1))
    ax.text(xmax / 2, -0.28, "gold electrode", ha="center", va="center", fontsize=7.5, color="#5A4300")
    ax.add_patch(Rectangle((-0.3, 0.0), xmax + 0.6, 6.4, fc="#F3F8FC", ec="none", zorder=0))


def _fc_layer(ax, xs):
    for x in xs:
        ax.plot(*_zigzag(x, 0.0, 0.95, amp=0.05, n=6), color=C["orange"], lw=1.4, zorder=3)
        _draw_fc(ax, x, 1.13)


def fig01_assembly():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.9))
    xmax = 4.6
    fc_x = [0.45, 1.05, 1.65, 3.0, 3.6, 4.15]
    for k, (ax, cleaved) in enumerate(zip(axes, (False, True))):
        ax.set_xlim(-0.3, xmax + 0.3)
        ax.set_ylim(-0.6, 6.4)
        ax.set_aspect("equal")
        ax.axis("off")
        _slab(ax, xmax)
        _fc_layer(ax, fc_x)
        pts, mb = _probe(ax, 2.3, cleaved, seed=11)
        panel(ax, "AB"[k], dx=-0.02, dy=0.985)
        ax.plot([0.0, 0.0], [5.4, 6.4], color="k", lw=1.5)
        ax.text(0.08, 5.9, "1 nm", fontsize=7, va="center")
        if not cleaved:
            ax.text(2.3, 6.25, "MB-peptide-PEG6 probe (intact)", ha="center", fontsize=8,
                    fontweight="bold", color=C["navy"])
            ax.annotate("Fc-hexanethiol\n(diluent)", xy=(fc_x[1], 1.3),
                        xytext=(-0.15, 2.5), fontsize=7, color="#8A4B0F",
                        arrowprops=dict(arrowstyle="-", color="#8A4B0F", lw=0.8))
            ax.annotate("PEG6 spacer\n(tip 1.4 nm above gold)", xy=(2.3, 0.7), xytext=(3.05, 1.6),
                        fontsize=7, color=C["grey"],
                        arrowprops=dict(arrowstyle="-", color=C["grey"], lw=0.8))
            ax.annotate("cleavage site\n(between residues 7 and 8)", xy=tuple((pts[6] + pts[7]) / 2),
                        xytext=(3.35, 3.4), fontsize=7, color=C["red"],
                        arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.8))
            ax.annotate("methylene blue\n(redox label)", xy=(pts[-1][0] + 0.1, pts[-1][1] + 0.15),
                        xytext=(-0.15, 4.5), fontsize=7, color=C["navy"],
                        arrowprops=dict(arrowstyle="-", color=C["navy"], lw=0.8))
            ax.add_patch(Ellipse((3.4, 5.2), 1.9, 0.95, fc="#FBE7D0", ec=C["orange"], lw=1.0, zorder=0, alpha=0.9))
            ax.text(3.4, 5.2, "MMP-9 in the\nwound fluid", ha="center", va="center", fontsize=7.5,
                    color=C["orange"], fontweight="bold")
        else:
            ax.text(2.3, 6.25, "After MMP-9 has cut the peptide", ha="center", fontsize=8,
                    fontweight="bold", color=C["navy"])
            ax.text(3.3, 5.5, "MB-bearing fragment released\ninto solution (not\nelectroactive)",
                    fontsize=7, color=C["navy"], ha="center")
            ax.text(0.7, 3.2, "truncated probe\nstays on the gold", fontsize=7, color=C["purple"], ha="center")
    fig.tight_layout(w_pad=0.5)
    save(fig, "swvr_01_assembly")


def fig02_anatomy():
    E = m.E_SCAN
    tot, d, w, k0, per_d = L.dist_avg(E, f=10.0, return_parts=True)
    pk, epk, fwhm = L.peak_stats(E, tot)
    num("peakNA", pk * 1e9, "{:.1f}")
    num("peakEmV", epk * 1e3, "{:.0f}")
    num("fwhmMV", fwhm * 1e3, "{:.0f}")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    ax = axes[0]
    ax.plot(E * 1e3, tot * 1e9, color=C["blue"])
    ax.fill_between(E * 1e3, 0, tot * 1e9, color=C["sky"], alpha=0.45)
    ax.axhline(0, color="k", lw=0.6)
    ax.annotate("", xy=(epk * 1e3, pk * 1e9), xytext=(epk * 1e3, 0),
                arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1.1))
    ax.annotate("height: proportional\nto intact probe", xy=(epk * 1e3, pk * 1e9 * 0.8),
                xytext=(-190, pk * 1e9 * 0.95), fontsize=7, color=C["red"],
                arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.7))
    ax.annotate("", xy=((epk + fwhm / 2) * 1e3, pk * 1e9 / 2), xytext=((epk - fwhm / 2) * 1e3, pk * 1e9 / 2),
                arrowprops=dict(arrowstyle="<->", color=C["green"], lw=1.1))
    ax.annotate("width (FWHM)", xy=((epk + fwhm / 2) * 1e3, pk * 1e9 / 2),
                xytext=(-190, pk * 1e9 * 0.42), fontsize=7, color=C["green"],
                arrowprops=dict(arrowstyle="-", color=C["green"], lw=0.7))
    ax.axvline(epk * 1e3, color=C["grey"], ls=":", lw=0.9)
    ax.annotate("position", xy=(epk * 1e3, pk * 1e9 * 1.02), xytext=(-190, pk * 1e9 * 1.13),
                fontsize=7, color=C["grey"], arrowprops=dict(arrowstyle="-", color=C["grey"], lw=0.7))
    ax.set_xlabel("applied potential (mV vs Ag/AgCl)")
    ax.set_ylabel("net current $\\Delta I$ (nA)")
    ax.set_ylim(-0.08 * pk * 1e9, pk * 1e9 * 1.2)
    ax.set_title("What a voltammogram shows")
    panel(ax, "A")
    # B: forward and reverse currents for a single mid-range probe
    ax = axes[1]
    kap = 0.5
    f = 10.0
    Ee = np.linspace(-0.40, -0.10, 601)
    F, R, T = m.F_const, m.R_const, m.T_K
    k0v = np.array([kap * 2 * f])
    tau = 1 / (2 * f)
    alpha, ne, E0, Esw = 0.37, 2, -0.25, 0.025
    def half(g, Ep):
        xi = ne * F * (Ep - E0) / (R * T)
        kf, kb = k0v * np.exp(-alpha * xi), k0v * np.exp((1 - alpha) * xi)
        K = kf + kb
        dd = (g - kf / K) * np.exp(-K * tau)
        return kf / K + dd, ne * F * K * dd
    g = np.ones(1)
    for _ in range(5):
        g, _ = half(g, Ee[0] + Esw); g, _ = half(g, Ee[0] - Esw)
    If, Ir = [], []
    for Es in Ee:
        g, a = half(g, Es + Esw); g, b = half(g, Es - Esw)
        If.append(a[0]); Ir.append(b[0])
    If, Ir = np.array(If), np.array(Ir)
    s = ne * F * 2 * f
    ax.plot(Ee * 1e3, If / s, color=C["red"], label="forward pulse, $I_{\\rm fwd}$")
    ax.plot(Ee * 1e3, Ir / s, color=C["navy"], label="reverse pulse, $I_{\\rm rev}$")
    ax.plot(Ee * 1e3, (If - Ir) / s, color=C["blue"], lw=2.4, label="net, $\\Delta I=I_{\\rm fwd}-I_{\\rm rev}$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("applied potential (mV vs Ag/AgCl)")
    ax.set_ylabel("current, units of $n_eF\\Gamma A\\cdot2f$")
    ax.set_ylim(-0.3, 0.62)
    ax.legend(loc="lower right", fontsize=6.6)
    ax.set_title("Where the net current comes from")
    panel(ax, "B")
    # C: before / after cleavage
    ax = axes[2]
    for fk, col in [(0.0, C["navy"]), (0.25, C["blue"]), (0.5, C["green"]), (0.75, C["orange"])]:
        ax.plot(E * 1e3, tot * (1 - fk) * 1e9, color=col, label=f"$F_k$ = {fk:.2f}")
    ax.set_xlabel("applied potential (mV vs Ag/AgCl)")
    ax.set_ylabel("net current $\\Delta I$ (nA)")
    ax.legend(loc="center right", fontsize=7)
    ax.set_title("Cleavage shrinks the peak")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_02_anatomy")


def fig03_protocol():
    fig, ax = plt.subplots(figsize=(7.4, 1.9))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.1)
    boxes = [
        (0.1, "1. Scan before", "SWV scan of the fresh\nelectrode: peak height\n$\\Delta I_{\\rm MB}(0)$", C["navy"]),
        (2.6, "2. Incubate", "wound fluid sits on the\nelectrode for time $T$;\nMMP-9 cleaves probes", C["orange"]),
        (5.1, "3. Scan after", "same electrode, same\nsettings: peak height\n$\\Delta I_{\\rm MB}(T)$", C["navy"]),
        (7.6, "4. Read out", "$F_k=1-\\dfrac{\\Delta I_{\\rm MB}(T)}{\\Delta I_{\\rm MB}(0)}$\nthen $F_k\\rightarrow$[MMP-9]", C["green"]),
    ]
    for x, ttl, body, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.15), 2.3, 1.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=col, lw=1.4))
        ax.text(x + 1.15, 1.7, ttl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=col)
        ax.text(x + 1.15, 0.85, body, ha="center", va="center", fontsize=7.6)
    for x in (2.4, 4.9, 7.4):
        ax.annotate("", xy=(x + 0.2, 1.05), xytext=(x, 1.05),
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.2))
    save(fig, "swvr_03_protocol")


def fig04_spacing():
    NA = 6.02214076e23
    gam = np.logspace(-13, -10, 300)          # mol cm^-2
    per_nm2 = gam * NA / 1e14
    spacing = 1 / np.sqrt(per_nm2)            # mean centre-to-centre, square lattice
    full = 1e14 / 0.2165 / NA                 # sqrt3 x sqrt3 R30 alkanethiol monolayer, 0.2165 nm^2 per molecule
    g0 = m.GAMMA_0
    s0 = 1 / np.sqrt(g0 * NA / 1e14)
    num("gammaZero", g0 * 1e12, "{:.0f}")
    num("spacingNM", s0, "{:.1f}")
    num("nnRandom", 0.5 * s0, "{:.1f}")
    num("lateralRms", np.sqrt(2.0) * m.SIGMA_Z_NM, "{:.1f}")
    num("overlapSpacing", 2 * np.sqrt(2.0) * m.SIGMA_Z_NM, "{:.1f}")
    num("areaPerProbeNM", 1 / (g0 * NA / 1e14), "{:.0f}")
    num("fullMono", full * 1e10, "{:.1f}")
    num("fractionMono", g0 / full * 100, "{:.1f}")
    num("oneInN", full / g0, "{:.0f}")
    lat = np.sqrt(2.0) * m.SIGMA_Z_NM
    num("gammaOverlap", 1e14 / NA / (2 * lat) ** 2 * 1e12, "{:.0f}")
    num("nProbes", g0 * m.A_ELEC * NA, "{:.2e}")
    num("nProbesFmol", g0 * m.A_ELEC * 1e15, "{:.0f}")
    lateral = np.sqrt(2.0) * m.SIGMA_Z_NM      # rms lateral reach of the label from its anchor
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    ax.loglog(gam * 1e12, spacing, color=C["blue"])
    ax.axhspan(0, 2 * lateral, color=C["red"], alpha=0.10)
    ax.text(0.11, 2 * lateral * 0.62, "label clouds overlap", color=C["red"], fontsize=7)
    ax.axhline(m.D_MAX_NM, color=C["grey"], ls="--", lw=1)
    ax.text(0.11, m.D_MAX_NM * 1.06, "longest vertical reach of the tether (5.2 nm)", color=C["grey"], fontsize=7)
    ax.axvline(g0 * 1e12, color=C["navy"], ls=":")
    ax.plot([g0 * 1e12], [s0], "o", color=C["navy"])
    ax.text(g0 * 1e12 * 1.08, s0 * 1.25, f"working value\n{g0*1e12:.0f} pmol cm$^{{-2}}$: {s0:.1f} nm", fontsize=7, color=C["navy"])
    ax.set_xlabel("probe coverage $\\Gamma_{\\rm tot}$ (pmol cm$^{-2}$)")
    ax.set_ylabel("mean probe spacing (nm)")
    ax.set_title("Probes are far apart")
    panel(ax, "A")
    ax = axes[1]
    rng = np.random.default_rng(1)
    box = 30.0
    n = int(per_nm2[0] * 0 + g0 * NA / 1e14 * box * box)
    pts = rng.uniform(0, box, size=(n, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=C["blue"], zorder=3)
    for p in pts:
        ax.add_patch(Circle(p, lateral, fc=C["sky"], ec="none", alpha=0.35))
    ax.set_xlim(0, box); ax.set_ylim(0, box); ax.set_aspect("equal")
    ax.set_xlabel("nm"); ax.set_ylabel("nm")
    ax.set_title(f"Top view: {n} probes in {box:.0f}$\\times${box:.0f} nm", fontsize=9)
    panel(ax, "B", dx=-0.17)
    fig.tight_layout(w_pad=1.2)
    save(fig, "swvr_04_spacing")


def build():
    fig01_assembly()
    fig02_anatomy()
    fig03_protocol()
    fig04_spacing()
