"""
report_lib.py: shared numerics and plotting style for the SWV forward-model
report. Everything here is computed from swv_forward_model.py; nothing is
hard-coded from a previous run.

The vectorised simulator `swv_multi` evaluates many rate constants at once
(one row per k0); tests assert it equals swv_forward_model.swv_voltammogram
row by row.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import swv_forward_model as m

FIG_DIR = m.FIGURES
NUMBERS = {}          # name -> (value, format string); spliced into the report .tex

# ---------------------------------------------------------------- style
C = dict(navy="#1F3A5F", blue="#2E7DB5", sky="#8FBFE0", orange="#D9822B",
         green="#3C9D6A", red="#C0392B", grey="#6B7280", light="#E5E7EB",
         purple="#7B5EA7", gold="#D4A72C")

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8,
    "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def panel(ax, letter, dx=-0.13, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def num(name, value, fmt="{:.3g}"):
    """Register a number for the LaTeX macro file."""
    NUMBERS[name] = (value, fmt)


BLOCKS = {}           # name -> LaTeX text spliced between markers in the report


def _splice(text, name, body):
    start, end = f"%% BEGIN GENERATED {name}", f"%% END GENERATED {name}"
    i, j = text.index(start), text.index(end)
    return text[: i + len(start)] + "\n" + body.rstrip("\n") + "\n" + text[j:]


def write_report_blocks(tex_path):
    """Splice generated numbers and tables into the single report .tex file."""
    path = Path(tex_path)
    text = path.read_text()
    import re

    def tex(x):
        mt = re.fullmatch(r"([0-9.]+)e([+-]\d+)", x)
        if mt:
            return r"\ensuremath{%s\times10^{%d}}" % (mt.group(1), int(mt.group(2)))
        if re.fullmatch(r"-[0-9.]+", x):
            return r"\ensuremath{-%s}" % x[1:]
        return x
    macros = [r"\newcommand{\%s}{%s}" % (k, tex(f.format(v))) for k, (v, f) in sorted(NUMBERS.items())]
    text = _splice(text, "NUMBERS", "\n".join(macros))
    for name, body in BLOCKS.items():
        text = _splice(text, name, body)
    path.write_text(text)


# ---------------------------------------------------------------- simulator
def swv_multi(E, k0, alpha=0.37, ne=2, f=10.0, Esw=0.025, E0=-0.250,
              burn=5):
    """Net SWV current per unit coverage and area, one row per k0.

    Same recursion as swv_forward_model.swv_voltammogram (closed-form
    half-pulse update, current I = ne F K (Gred - Geq)), vectorised over k0.
    Returns array (len(k0), len(E)) in A cm^-2 per (mol cm^-2).
    """
    k0 = np.atleast_1d(np.asarray(k0, dtype=float))
    tau = 1.0 / (2.0 * f)
    F, R, T = m.F_const, m.R_const, m.T_K

    def half(g, Ep):
        xi = ne * F * (Ep - E0) / (R * T)
        kf, kb = k0 * np.exp(-alpha * xi), k0 * np.exp((1 - alpha) * xi)
        K = kf + kb
        d = (g - kf / K) * np.exp(-K * tau)
        return kf / K + d, ne * F * K * d

    g = np.ones_like(k0)
    for _ in range(burn):
        g, _ = half(g, E[0] + Esw)
        g, _ = half(g, E[0] - Esw)
    out = np.empty((k0.size, len(E)))
    for i, Es in enumerate(E):
        g, If = half(g, Es + Esw)
        g, Ir = half(g, Es - Esw)
        out[:, i] = If - Ir
    return out


def pd_norm(d, d_peg=None, sigma=None, d_max=None):
    d_peg = m.D_PEG_NM if d_peg is None else d_peg
    sigma = m.SIGMA_Z_NM if sigma is None else sigma
    d_max = m.D_MAX_NM if d_max is None else d_max
    p = np.where(d >= d_peg, np.sqrt(2 / np.pi) / sigma
                 * np.exp(-(d - d_peg) ** 2 / (2 * sigma ** 2)), 0.0)
    grid = np.linspace(d_peg, d_max, 4001)
    pg = np.sqrt(2 / np.pi) / sigma * np.exp(-(grid - d_peg) ** 2 / (2 * sigma ** 2))
    return p / np.trapz(pg, grid)


def dist_avg(E, f=10.0, fk=0.0, n=400, k0_ref=None, beta=None, d_peg=None,
             sigma=None, d_max=None, gamma=None, area=None, weights=None,
             dgrid=None, return_parts=False, **kw):
    """Distance-averaged net SWV current (A) at cleavage fraction fk.

    Same integral as swv_forward_model.dist_averaged_voltammogram, with the
    tether parameters exposed and a vectorised inner loop.
    """
    k0_ref = m.K0_REF_S1 if k0_ref is None else k0_ref
    beta = m.BETA_NM if beta is None else beta
    d_peg = m.D_PEG_NM if d_peg is None else d_peg
    sigma = m.SIGMA_Z_NM if sigma is None else sigma
    d_max = m.D_MAX_NM if d_max is None else d_max
    gamma = m.GAMMA_0 if gamma is None else gamma
    area = m.A_ELEC if area is None else area
    d = np.linspace(d_peg, d_max, n) if dgrid is None else dgrid
    w = pd_norm(d, d_peg, sigma, d_max) if weights is None else weights
    k0 = k0_ref * np.exp(-beta * (d - d_peg))
    per_d = swv_multi(E, k0, f=f, **kw)                      # (n, nE)
    dd = np.gradient(d)
    tot = (w[:, None] * per_d * dd[:, None]).sum(axis=0)
    tot = tot * (1.0 - fk) * gamma * area
    if return_parts:
        return tot, d, w, k0, per_d
    return tot


def peak_stats(E, y):
    """Peak height, potential and FWHM (V) of a positive peak."""
    i = int(np.argmax(y))
    half = y >= y[i] / 2
    idx = np.where(half)[0]
    # linear interpolation of the half-height crossings
    lo, hi = idx[0], idx[-1]
    def cross(a, b):
        return E[a] + (y[i] / 2 - y[a]) * (E[b] - E[a]) / (y[b] - y[a])
    e_lo = cross(lo - 1, lo) if lo > 0 else E[lo]
    e_hi = cross(hi, hi + 1) if hi < len(E) - 1 else E[hi]
    return y[i], E[i], e_hi - e_lo


def g_of_kappa(kappas, f=10.0, alpha=0.37, ne=2, Esw=0.025):
    """Dimensionless single-probe peak response g = DeltaI_pk / (ne F 2f)."""
    E = np.linspace(-0.40, -0.10, 601)
    k0 = np.asarray(kappas) * 2 * f
    y = swv_multi(E, k0, alpha=alpha, ne=ne, f=f, Esw=Esw)
    return y.max(axis=1) / (ne * m.F_const * 2 * f)


def keff_surface(eta):
    return eta * m.KEFF_SOL_M1S1


def fk_of(conc_nM, t_min, eta=0.20):
    return 1.0 - np.exp(-keff_surface(eta) * conc_nM * 1e-9 * t_min * 60.0)
