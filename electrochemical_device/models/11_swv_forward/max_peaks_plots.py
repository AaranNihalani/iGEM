"""
max_peaks_plots.py -- MB / Fc / ratio-vs-time plots for Max's two SWV time
courses in data/max_data/, plus a background check for the BSA control.

Data identity (cross-checked against data/max_data/.../mmp9_20nM_MB_Fc.zip's
own README and peaks workbook, and independently re-derived from the raw
.xlsx scans here -- both agree exactly):

  - "MMP-9 20 nM"  (folder name HAS a space)  = CLEAVABLE probe electrode.
  - "MMP-9 20nM"   (folder name has NO space) = NON-CLEAVABLE probe electrode
    (the stable reference surface; Fc is present but the probe is not cut).
  - Both are single-run time courses, scan index -> time = (index-1) x 5 min.
  - Both carry both an MB peak (window -260 to -100 mV) and an Fc peak
    (window +150 to +400 mV), so both get MB, Fc AND MB/Fc ratio plots.
  - Scan #1 in both folders is a known outlier (the sweep for that scan
    starts at -200 mV instead of the usual -250/-260 mV, which clips/shifts
    the MB peak by ~70-90 mV) -- shown as a hollow grey point, excluded from
    the smoothed trend, exactly as already flagged in the existing
    mmp9_20nM_MB_Fc_peaks.xlsx workbook.
  - Cleavable #2 is a genuinely empty .xlsx export (0 bytes); its data only
    exists in an unreadable .nox file labelled "8 mins" -- shown as missing,
    not interpolated.
  - "BSA control" does show a real, reproducible feature, but it sits near
    0-100 mV -- clear of both the MB window (-260 to -100 mV, where this
    electrode's trace has no local maximum, just a falling edge) and the Fc
    window (+150 to +400 mV, where the trace has already decayed to a low,
    noisy baseline). So whatever that feature is, it is not MB or Fc, and
    this folder is NOT given MB/Fc/ratio treatment; instead we plot the raw
    traces across scans so the real shape is visible directly.

Smoothing: robust LOWESS (statsmodels, it=3) through the non-outlier,
non-missing points only. LOWESS down-weights points that sit far from the
local trend over its iterations, so a single rough scan cannot swing the
smoothed line the way a straight connecting line would.
"""
from pathlib import Path

import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

DATA = Path(__file__).parents[2] / "data" / "max_data"
OUT = Path(__file__).parent / "max_peaks_figures"
OUT.mkdir(exist_ok=True)

MB_WINDOW = (-0.260, -0.100)   # V
FC_WINDOW = (0.150, 0.400)     # V

GREEN, ORANGE, PURPLE, INK, MUTE, GRID, GREY = (
    "#15876A", "#eb6834", "#4a3aa7", "#173B34", "#8A9892", "#E3E8E1", "#B9C2BC",
)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 2.0,
    "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": MUTE,
})


def load_scan(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    pot = np.array([ws.cell(r, 2).value for r in range(2, ws.max_row + 1)], float)
    net = np.array([ws.cell(r, 6).value for r in range(2, ws.max_row + 1)], float)
    return pot, net


def peak_in_window(pot, net, lo, hi):
    m = (pot >= lo) & (pot <= hi)
    i = np.argmax(net[m])
    return net[m][i] * 1e9, pot[m][i] * 1000.0  # nA, mV


def gather(folder):
    """Returns per-scan dicts: t, mb, fc, ok (bool: both peaks real & not the
    sweep-start outlier), outlier (bool: scan 1's known clipped-window scan)."""
    rows = []
    for i in range(1, 14):
        f = DATA / folder / f"#{i}.xlsx"
        t = (i - 1) * 5
        if not f.exists() or f.stat().st_size == 0:
            rows.append(dict(i=i, t=t, mb=None, fc=None, ok=False, outlier=False, missing=True))
            continue
        pot, net = load_scan(f)
        mb, _ = peak_in_window(pot, net, *MB_WINDOW)
        fc, _ = peak_in_window(pot, net, *FC_WINDOW)
        outlier = (i == 1)
        rows.append(dict(i=i, t=t, mb=mb, fc=fc, ok=not outlier, outlier=outlier, missing=False))
    return rows


def smooth(t, y, frac=0.5):
    """Robust LOWESS through (t, y); returns (t_fit, y_fit) sorted by t."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    fit = lowess(y, t, frac=frac, it=3, return_sorted=True)
    return fit[:, 0], fit[:, 1]


def panel(ax, rows, key, color, ylabel, marker):
    ok = [r for r in rows if r["ok"]]
    out = [r for r in rows if r["outlier"] and r[key] is not None]
    t_ok = [r["t"] for r in ok]
    y_ok = [r[key] for r in ok]

    if len(t_ok) >= 4:
        tf, yf = smooth(t_ok, y_ok)
        ax.plot(tf, yf, color=color, lw=2.6, zorder=2, label="Robust LOWESS trend")
    ax.plot(t_ok, y_ok, marker, color=color, ms=7, lw=0, zorder=3,
             markeredgecolor=INK, markeredgewidth=0.6, label="Scan (used)")
    if out:
        ax.plot([r["t"] for r in out], [r[key] for r in out], "o", ms=7, zorder=3,
                 markerfacecolor="none", markeredgecolor=GREY, markeredgewidth=1.6,
                 label="Scan 1 (excluded: clipped sweep window)")
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-2, 62)


def plot_condition(folder, title, fname):
    rows = gather(folder)
    for r in rows:
        r["ratio"] = (r["mb"] / r["fc"]) if (r["ok"] and r["mb"] is not None and r["fc"] is not None) else None

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panel(axes[0], rows, "mb", GREEN, "MB peak height (nA)", "o")
    panel(axes[1], rows, "fc", ORANGE, "Fc peak height (nA)", "s")
    panel(axes[2], rows, "ratio", PURPLE, "MB / Fc peak ratio", "D")
    axes[0].set_title("MB peak (ΔI, -260 to -100 mV)", loc="left", fontsize=10)
    axes[1].set_title("Fc peak (ΔI, +150 to +400 mV)", loc="left", fontsize=10)
    axes[2].set_title("MB / Fc ratio", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")

    missing = [r["t"] for r in rows if r["missing"]]
    note = f"  (scan at t={missing[0]} min missing: empty export)" if missing else ""
    fig.suptitle(f"{title} -- single run, square-wave voltammetry{note}", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=220)
    plt.close(fig)
    print("saved", fname)
    return rows


def plot_bsa_background():
    """BSA control shows a real, reproducible feature (peak near 0-100 mV),
    but it sits well clear of both the MB window (-260 to -100 mV: here the
    trace is simply falling from the sweep's negative limit, no local max)
    and the Fc window (+150 to +400 mV: the trace has already decayed to a
    low, noisy baseline there). So it is not MB or Fc signal, and is not
    given MB/Fc/ratio treatment -- the raw traces are plotted instead so
    this is visible directly, rather than asserted."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cmap = plt.cm.viridis
    files = []
    for i in range(1, 14):
        f = DATA / "BSA control" / f"#{i}.xlsx"
        if f.exists() and f.stat().st_size > 0:
            files.append((i, f))
    rest = [(i, f) for i, f in files if i != 1]
    for n, (i, f) in enumerate(rest):
        pot, net = load_scan(f)
        t = (i - 1) * 5
        ax.plot(pot * 1000, net * 1e9, color=cmap(n / max(len(rest) - 1, 1)), lw=1.4, alpha=0.9,
                 label=f"t={t} min")
    first = [(i, f) for i, f in files if i == 1]
    if first:
        pot, net = load_scan(first[0][1])
        ax.plot(pot * 1000, net * 1e9, color=GREY, lw=1.6, ls="--", alpha=0.9,
                 label="t=0 min (excluded: first-scan transient, same pattern as scan 1 elsewhere)")
    ax.axvspan(MB_WINDOW[0] * 1000, MB_WINDOW[1] * 1000, color=GREEN, alpha=0.08, label="MB window")
    ax.axvspan(FC_WINDOW[0] * 1000, FC_WINDOW[1] * 1000, color=ORANGE, alpha=0.08, label="Fc window")
    ax.set_xlabel("Potential (mV)")
    ax.set_ylabel(r"Net SWV current, $\Delta I$ (nA)")
    ax.set_title("BSA control -- the real feature near 0-100 mV is not MB or Fc\n(it falls outside both the MB and Fc windows used for the probe electrodes)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "bsa_control_background.png", dpi=220)
    plt.close(fig)
    print("saved bsa_control_background.png")


if __name__ == "__main__":
    cleavable = plot_condition("MMP-9 20 nM", "Cleavable probe (MMP-9 20 nM)", "cleavable_mb_fc_ratio.png")
    noncleavable = plot_condition("MMP-9 20nM", "Non-cleavable probe (MMP-9 20nM)", "noncleavable_mb_fc_ratio.png")
    plot_bsa_background()

    for label, rows in [("Cleavable", cleavable), ("Non-cleavable", noncleavable)]:
        print(f"\n{label}:")
        for r in rows:
            if r["missing"]:
                print(f"  t={r['t']:>3}min  MISSING")
            else:
                tag = " (excluded outlier)" if r["outlier"] else ""
                print(f"  t={r['t']:>3}min  MB={r['mb']:.2f}nA  Fc={r['fc']:.2f}nA  ratio={r['mb']/r['fc']:.3f}{tag}")
