"""Figures for Part VI (measuring with MB alone) and Part VII (from MMP-9
concentration to predicted signal), plus the sensitivity summary."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import swv_forward_model as m
import report_lib as L
from report_lib import C, panel, save, num

E = m.E_SCAN
FREQ = 10.0
KSOL = m.KEFF_SOL_M1S1
ETA = 0.20
Z = 3.0                 # detection criterion: F_k^rec >= Z * sigma_F
SIGMA_F_OLD = 0.034     # earlier ratio-of-ratios noise floor, no longer applicable


def keff(eta=ETA):
    return eta * KSOL


def lod_nM(sigma_F, t_min, eta=ETA, z=Z):
    fmin = np.minimum(z * sigma_F, 0.999)
    return -np.log(1 - fmin) / (keff(eta) * t_min * 60.0) * 1e9


# ------------------------------------------------------------------ Part VI
def fig16_potentials():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.set_xlim(-0.55, 0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.axvspan(-0.40, -0.10, ymin=0.05, ymax=0.45, color=C["sky"], alpha=0.6)
    ax.text(-0.25, 0.25, "MB scan window\n$-0.40$ to $-0.10$ V", ha="center", va="center", fontsize=7.5, color=C["navy"])
    ax.axvline(-0.25, ymin=0.05, ymax=0.62, color=C["navy"], lw=1.8)
    ax.text(-0.25, 0.66, "MB\n$E^{0\\prime}=-0.25$ V", ha="center", fontsize=7.5, color=C["navy"])
    ax.axvspan(0.30, 0.40, ymin=0.05, ymax=0.62, color=C["orange"], alpha=0.5)
    ax.text(0.35, 0.66, "Fc, observed\n+0.30 to +0.40 V", ha="center", fontsize=7.5, color="#8A4B0F")
    ax.axvline(0.21, ymin=0.05, ymax=0.45, color=C["grey"], lw=1.4, ls="--")
    ax.text(0.21, 0.5, "Fc literature\n+0.21 V", ha="center", fontsize=7, color=C["grey"])
    ax.annotate("", xy=(0.30, 0.86), xytext=(-0.10, 0.86), arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1.1))
    ax.text(0.10, 0.9, "0.31 V (literature) to 0.40 V (observed)\nbetween the MB window edge and Fc", ha="center", fontsize=7, color=C["red"])
    ax.set_xlabel("potential (V vs Ag/AgCl)")
    ax.set_title("Where each label reacts")
    panel(ax, "A", dx=-0.04)
    # B: leakage envelope
    ax = axes[1]
    dE = np.linspace(0, 0.6, 200)
    xi = 1 * m.F_const * dE / (m.R_const * m.T_K)
    env = 4 * np.exp(-xi) / (1 + np.exp(-xi)) ** 2       # Nernstian (n=1) wave shape, worst-case tail
    ax.semilogy(dE * 1e3, env, color=C["red"])
    gap = 0.31
    leak = float(4 * np.exp(-m.F_const * gap / (m.R_const * m.T_K)) / (1 + np.exp(-m.F_const * gap / (m.R_const * m.T_K))) ** 2)
    num("gapFc", gap * 1e3, "{:.0f}")
    num("leakEnv", leak, "{:.0e}")
    ratio_worst = 1e-5 / (m.dist_averaged_voltammogram(E, 0.0, n_dist=200).max())
    num("fcMbRatio", ratio_worst, "{:.0f}")
    num("leakMB", leak * ratio_worst * 100, "{:.2f}")
    ax.axvline(gap * 1e3, color=C["grey"], ls=":")
    ax.set_xlabel("distance from the Fc peak (mV)"); ax.set_ylabel("Fc wave / its peak (upper bound)")
    ax.set_title("Fc tail at the MB window")
    panel(ax, "B")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_16_potentials")


def fig17_drift():
    fs = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    for dl, col in ((0.0, C["navy"]), (0.02, C["blue"]), (0.05, C["green"]), (0.10, C["orange"]), (0.20, C["red"])):
        ax.plot(fs, 1 - (1 - fs) * (1 - dl), color=col, label=f"$\\delta$ = {dl:.0%}")
    ax.set_xlabel("true cleavage fraction $F_k$"); ax.set_ylabel("recovered $F_k^{\\rm rec}$")
    ax.legend(fontsize=6.5)
    ax.set_title("Non-enzymatic loss reads as cleavage")
    panel(ax, "A")
    ax = axes[1]
    dl = np.linspace(0, 0.2, 100)
    for fk, col in ((0.0, C["navy"]), (0.14, C["blue"]), (0.52, C["orange"])):
        ax.plot(dl * 100, dl * (1 - fk) * 100, color=col, label=f"$F_k$ = {fk:.2f}")
    ax.set_xlabel("$\\delta$: non-enzymatic loss (%)"); ax.set_ylabel("upward bias in $F_k^{\\rm rec}$ (points, %)")
    ax.legend(fontsize=7)
    ax.set_title("Bias $=\\delta\\,(1-F_k)$")
    panel(ax, "B")
    ax = axes[2]
    for T, col in ((10, C["navy"]), (30, C["blue"]), (60, C["green"])):
        c_app = -np.log(1 - dl) / (keff() * T * 60) * 1e9
        ax.plot(dl * 100, c_app, color=col, label=f"$T$ = {T} min")
    ax.axhline(6, color=C["red"], ls="--", lw=0.9)
    ax.text(0.3, 6.6, "healing / chronic boundary (6 nM)", fontsize=6.8, color=C["red"])
    ax.set_xlabel("$\\delta$ (%)"); ax.set_ylabel("apparent [MMP-9] in a blank (nM)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.85))
    num("blankAppFive", -np.log(1 - 0.05) / (keff() * 30 * 60) * 1e9, "{:.1f}")
    ax.set_title("A false reading from drift")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_17_drift")


def fig18_coverage():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    gam = [1e-12, 3e-12, 1e-11]
    fk = 0.30
    h0, hT, fr = [], [], []
    for g in gam:
        y0 = L.dist_avg(E, FREQ, gamma=g, n=300).max()
        yT = L.dist_avg(E, FREQ, fk=fk, gamma=g, n=300).max()
        h0.append(y0 * 1e9); hT.append(yT * 1e9); fr.append(1 - yT / y0)
    x = np.arange(3)
    ax.bar(x - 0.18, h0, 0.34, color=C["navy"], label="before")
    ax.bar(x + 0.18, hT, 0.34, color=C["sky"], label="after")
    for i, r in enumerate(fr):
        ax.text(i, h0[i] * 1.03, f"{r:.2f}", ha="center", fontsize=7)
    ax.text(0.02, 0.62, "numbers: recovered $F_k$", transform=ax.transAxes, fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(["1", "3", "10"])
    ax.set_xlabel("electrode coverage $\\Gamma_{\\rm tot}$ (pmol cm$^{-2}$)"); ax.set_ylabel("peak height (nA)")
    ax.set_ylim(0, max(h0) * 1.15)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Same $F_k^{\\rm rec}$, any coverage")
    num("fkRecCov", np.ptp(fr), "{:.0e}")
    panel(ax, "A")
    ax = axes[1]
    fs = np.linspace(0, 0.9, 100)
    for r0, col in ((0.005, C["navy"]), (0.01, C["blue"]), (0.02, C["orange"])):
        sig = np.sqrt(r0 ** 2 * (1 + (1 - fs) ** 2))
        ax.plot(fs, sig * 100, color=col, label=f"$\\sigma_r$ = {r0:.1%}")
    ax.set_xlabel("true $F_k$"); ax.set_ylabel("$\\sigma(F_k^{\\rm rec})$ (points, %)")
    ax.legend(fontsize=7)
    ax.set_title("Noise in the readout")
    panel(ax, "B")
    ax = axes[2]
    gam2 = np.logspace(-12, -10.3, 60)
    for r0, col in ((0.005, C["navy"]), (0.01, C["blue"]), (0.02, C["orange"])):
        sigF = np.sqrt(2) * r0 * (3e-12 / gam2)         # fixed absolute noise, relative to the 3 pmol value
        ax.semilogx(gam2 * 1e12, lod_nM(sigF, 30), color=col, label=f"$\\sigma_r$ = {r0:.1%} at 3 pmol")
    ax.axhline(6, color=C["red"], ls="--", lw=0.9)
    ax.set_xlabel("coverage (pmol cm$^{-2}$)"); ax.set_ylabel("detection limit at 30 min (nM)")
    ax.set_ylim(0, 40)
    ax.legend(fontsize=6.3)
    ax.set_title("Low coverage costs sensitivity")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_18_coverage")


# ------------------------------------------------------------------ Part VII
def fig19_kinetics():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    t = np.linspace(0, 120, 200)
    for c, col in ((3, C["sky"]), (6, C["blue"]), (15, C["green"]), (30, C["orange"]), (60, C["red"])):
        ax.plot(t, L.fk_of(c, t) * 100, color=col, label=f"{c} nM")
    ax.axvline(30, color=C["grey"], ls=":")
    ax.set_xlabel("incubation time $T$ (min)"); ax.set_ylabel("cleavage fraction $F_k$ (%)")
    ax.legend(fontsize=6.5)
    ax.set_title("Cleavage over time")
    panel(ax, "A")
    ax = axes[1]
    cc = np.linspace(0, 60, 200)
    for T, col in ((10, C["sky"]), (30, C["blue"]), (60, C["navy"])):
        ax.plot(cc, L.fk_of(cc, T) * 100, color=col, label=f"$T$ = {T} min")
    ax.axvline(6, color=C["green"], ls="--", lw=0.9); ax.axvline(30, color=C["red"], ls="--", lw=0.9)
    ax.text(6.6, 92, "healing", fontsize=7, color=C["green"]); ax.text(24, 92, "chronic", fontsize=7, color=C["red"])
    ax.set_xlabel("[MMP-9] (nM)"); ax.set_ylabel("$F_k$ (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Dose response")
    panel(ax, "B")
    ax = axes[2]
    cc = np.logspace(0, 2, 60)
    ax.loglog(cc, np.log(2) / (keff() * cc * 1e-9) / 60, color=C["blue"])
    ax.axhline(30, color=C["grey"], ls=":")
    ax.set_xlabel("[MMP-9] (nM)"); ax.set_ylabel("half-time of the probe (min)")
    ax.set_title("Time for half the probes to be cut")
    panel(ax, "C")
    num("fkHealThirty", L.fk_of(6, 30) * 100, "{:.0f}")
    num("fkChronThirty", L.fk_of(30, 30) * 100, "{:.0f}")
    num("thalfHeal", np.log(2) / (keff() * 6e-9) / 60, "{:.0f}")
    num("thalfChron", np.log(2) / (keff() * 30e-9) / 60, "{:.0f}")
    num("keffM", keff(), "{:,.0f}".replace(",", ""))
    num("kSol", KSOL, "{:.0f}")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_19_kinetics")


def fig20_voltammograms():
    base = L.dist_avg(E, FREQ, n=300)
    concs = [0, 3, 6, 15, 30, 60]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(concs)))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    peaks = []
    for c, col in zip(concs, cols):
        fk = L.fk_of(c, 30)
        ax.plot(E * 1e3, base * (1 - fk) * 1e9, color=col, label=f"{c} nM")
        peaks.append(base.max() * (1 - fk) * 1e9)
    ax.set_xlabel("potential (mV vs Ag/AgCl)"); ax.set_ylabel("net current (nA)")
    ax.legend(fontsize=6.5)
    ax.set_title("30 min incubation")
    panel(ax, "A")
    ax = axes[1]
    cc = np.linspace(0, 60, 100)
    for T, col in ((10, C["sky"]), (30, C["blue"]), (60, C["navy"])):
        ax.plot(cc, base.max() * (1 - L.fk_of(cc, T)) * 1e9, color=col, label=f"$T$ = {T} min")
    ax.set_xlabel("[MMP-9] (nM)"); ax.set_ylabel("peak height (nA)")
    ax.legend(fontsize=7)
    ax.set_title("Calibration curve")
    panel(ax, "B")
    ax = axes[2]
    Ts = np.linspace(1, 90, 100)
    diff = (L.fk_of(30, Ts) - L.fk_of(6, Ts)) * 100
    ax.plot(Ts, diff, color=C["blue"])
    imax = int(np.argmax(diff))
    ax.plot(Ts[imax], diff[imax], "o", color=C["red"])
    ax.text(Ts[imax] + 3, diff[imax] - 2, f"largest gap {diff[imax]:.0f} points\nat {Ts[imax]:.0f} min", fontsize=7, color=C["red"])
    num("gapMax", diff[imax], "{:.0f}")
    num("gapMaxT", Ts[imax], "{:.0f}")
    ax.set_xlabel("incubation time (min)"); ax.set_ylabel("$F_k$(chronic) $-$ $F_k$(healing) (points, %)")
    ax.set_title("Separating the two wound states")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_20_voltammograms")


def fig21_lod():
    sig = np.linspace(0.005, 0.08, 120)
    Ts = np.linspace(5, 90, 120)
    S, TT = np.meshgrid(sig, Ts)
    lod = lod_nM(S, TT)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    im = ax.contourf(sig * 100, Ts, lod, levels=np.logspace(np.log10(0.5), np.log10(200), 25), norm=LogNorm(vmin=0.5, vmax=200), cmap="viridis_r")
    cs = ax.contour(sig * 100, Ts, lod, levels=[6], colors="white", linewidths=1.4)
    ax.clabel(cs, fmt="6 nM", fontsize=7)
    ax.plot([SIGMA_F_OLD * 100], [30], "o", color=C["red"])
    ax.text(SIGMA_F_OLD * 100 + 0.5, 10, "reference case\n3.4%", fontsize=6.8, color="white")
    ax.set_xlabel("noise in the readout $\\sigma_F$ (points, %)"); ax.set_ylabel("incubation time $T$ (min)")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("detection limit (nM)", fontsize=8)
    cb.set_ticks([1, 3, 10, 30, 100]); cb.set_ticklabels(["1", "3", "10", "30", "100"])
    ax.set_title("Detection limit, $\\eta=0.20$")
    ax.grid(False)
    panel(ax, "A")
    ax = axes[1]
    for eta, col in ((0.05, C["red"]), (0.10, C["orange"]), (0.20, C["blue"]), (0.50, C["green"]), (1.0, C["navy"])):
        ax.plot(sig * 100, lod_nM(sig, 30, eta=eta), color=col, label=f"$\\eta$ = {eta:g}")
    ax.axhline(6, color=C["grey"], ls="--", lw=0.9)
    ax.set_ylim(0, 60)
    ax.set_xlabel("$\\sigma_F$ (points, %)"); ax.set_ylabel("detection limit at 30 min (nM)")
    ax.legend(fontsize=7)
    ax.set_title("Depends on surface efficiency")
    panel(ax, "B")
    fig.tight_layout(w_pad=1.2)
    save(fig, "swvr_21_lod")
    num("lodOld", lod_nM(SIGMA_F_OLD, 30), "{:.1f}")
    num("fMinOld", Z * SIGMA_F_OLD * 100, "{:.1f}")
    num("sigmaMaxAtSix", 6e-9 * 0 + (1 - np.exp(-keff() * 6e-9 * 1800)) / Z * 100, "{:.1f}")
    num("sigmaRelOld", SIGMA_F_OLD / np.sqrt(2) * 100, "{:.1f}")
    num("lodMBonly", lod_nM(SIGMA_F_OLD / np.sqrt(2), 30), "{:.1f}")
    num("fMinMBonly", Z * SIGMA_F_OLD / np.sqrt(2) * 100, "{:.1f}")


def fig22_eta():
    etas = np.logspace(-2, 0, 120)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    for c, col in ((6, C["green"]), (30, C["red"])):
        ax.semilogx(etas, L.fk_of(c, 30, etas) * 100, color=col, label=f"{c} nM")
    ax.axvline(ETA, color=C["grey"], ls=":")
    ax.axhline(Z * SIGMA_F_OLD * 100, color=C["navy"], ls="--", lw=0.9)
    ax.text(0.011, Z * SIGMA_F_OLD * 100 + 1.5, "3$\\sigma$ threshold if $\\sigma_F$=3.4%", fontsize=6.5, color=C["navy"])
    ax.set_xlabel("surface efficiency $\\eta$"); ax.set_ylabel("$F_k$ at 30 min (%)")
    ax.legend(fontsize=7, loc="center left")
    ax.set_title("What $\\eta$ does to $F_k$")
    panel(ax, "A")
    ax = axes[1]
    rng = np.random.default_rng(3)
    draw = 10 ** rng.uniform(-2, 0, 20000)
    fh = L.fk_of(6, 30, draw); fc = L.fk_of(30, 30, draw)
    ax.hist(fh * 100, bins=50, color=C["green"], alpha=0.6, label="6 nM")
    ax.hist(fc * 100, bins=50, color=C["red"], alpha=0.6, label="30 nM")
    ax.set_xlabel("$F_k$ at 30 min (%)"); ax.set_ylabel("draws")
    ax.legend(fontsize=7)
    ax.set_title("$\\eta$ uniform in log, 0.01 to 1")
    num("etaFrac", np.mean(fh >= Z * SIGMA_F_OLD) * 100, "{:.0f}")
    num("etaMinHeal", -np.log(1 - Z * SIGMA_F_OLD) / (KSOL * 6e-9 * 1800), "{:.2f}")
    num("sigmaRMaxAtSix", (1 - np.exp(-keff() * 6e-9 * 1800)) / Z * 100 / np.sqrt(2), "{:.1f}")
    panel(ax, "B")
    ax = axes[2]
    Tneed = -np.log(1 - Z * SIGMA_F_OLD) / (keff(etas) * 6e-9) / 60
    ax.loglog(etas, Tneed, color=C["blue"])
    ax.axhline(30, color=C["grey"], ls=":"); ax.axvline(ETA, color=C["grey"], ls=":")
    ax.set_xlabel("surface efficiency $\\eta$"); ax.set_ylabel("time to detect 6 nM (min)")
    ax.set_title("Incubation needed")
    num("tNeedSix", -np.log(1 - Z * SIGMA_F_OLD) / (keff() * 6e-9) / 60, "{:.0f}")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_22_eta")


def fig23_tornado():
    base = L.dist_avg(E, FREQ, n=300).max()
    k_meas, x_meas = 5.0e4, 0.8               # Dauphin-Ducharme: k0 at 8 Angstrom
    def kref(beta, dpeg):
        return k_meas * np.exp(-beta * (dpeg - x_meas))
    rows = []
    def add(name, lo, hi):
        rows.append((name, lo / base - 1, hi / base - 1))
    add("coverage $\\Gamma_{\\rm tot}$ ($\\div3$, $\\times3$)",
        L.dist_avg(E, FREQ, gamma=m.GAMMA_0 / 3, n=300).max(), L.dist_avg(E, FREQ, gamma=m.GAMMA_0 * 3, n=300).max())
    add("area $A$ ($\\pm$5%)", base * 0.95, base * 1.05)
    add("frequency (10 to 20 Hz)", base, L.dist_avg(E, 20.0, n=300).max())
    add("tip position $d_{\\rm PEG}$ (1.26 to 2.1 nm)",
        L.dist_avg(E, FREQ, d_peg=1.26, d_max=1.26 + 3.8, k0_ref=kref(10, 1.26), n=300).max(),
        L.dist_avg(E, FREQ, d_peg=2.1, d_max=2.1 + 3.8, k0_ref=kref(10, 2.1), n=300).max())
    add("decay constant $\\beta$ (5 to 13 nm$^{-1}$)",
        L.dist_avg(E, FREQ, beta=5, k0_ref=kref(5, m.D_PEG_NM), n=300).max(),
        L.dist_avg(E, FREQ, beta=13, k0_ref=kref(13, m.D_PEG_NM), n=300).max())
    add("rate anchor $k_{\\rm ref}$ ($\\times0.1$, $\\times10$)",
        L.dist_avg(E, FREQ, k0_ref=m.K0_REF_S1 * 0.1, n=300).max(), L.dist_avg(E, FREQ, k0_ref=m.K0_REF_S1 * 10, n=300).max())
    sig_wall = L.NUMBERS["sigWall"][0] * 1e-9
    add("tether model (no bead below tip)", sig_wall, base)
    add("transfer coefficient $\\alpha$ (0.3 to 0.5)",
        L.dist_avg(E, FREQ, alpha=0.3, n=300).max(), L.dist_avg(E, FREQ, alpha=0.5, n=300).max())
    rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    for i, (n, lo, hi) in enumerate(rows):
        ax.barh(i, hi * 100, color=C["blue"], height=0.6)
        ax.barh(i, lo * 100, color=C["orange"], height=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("change in predicted peak height (%)")
    ax.set_title("Absolute current: sensitive to placeholders")
    ax.grid(axis="y", visible=False)
    panel(ax, "A", dx=-0.02)
    tab = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "Parameter varied & low & high \\\\", "\\midrule"]
    for n, lo, hi in sorted(rows, key=lambda r: -max(abs(r[1]), abs(r[2]))):
        tab.append(f"{n.replace(chr(37), chr(92) + chr(37))} & ${lo*100:+.0f}\\%$ & ${hi*100:+.0f}\\%$ \\\\")
    tab += ["\\bottomrule", "\\end{tabular}"]
    L.BLOCKS["TORNADO"] = "\n".join(tab)
    ax = axes[1]
    labs = ["all of the\nabove", "cleavage\nrate depends\non $d$"]
    ax.bar([0], [0.0], color=C["blue"])
    # effect of distance-dependent cleavage on the recovered fraction (moderate case)
    lam = 2.0
    d = np.linspace(m.D_PEG_NM, m.D_MAX_NM, 300)
    w = L.pd_norm(d)
    dbar = np.trapezoid(w * d, d)
    q = np.exp(lam * (d - dbar))
    q /= np.trapezoid(w * q, d)
    kt = 0.6
    surv = np.exp(-kt * q)
    true_F = 1 - np.trapezoid(w * surv, d)
    y0 = L.dist_avg(E, FREQ, dgrid=d, weights=w, n=len(d)).max()
    yT = L.dist_avg(E, FREQ, dgrid=d, weights=w * surv, n=len(d)).max()
    rec = 1 - yT / y0
    ax.bar([1], [(rec - true_F) * 100], color=C["red"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("error in recovered $F_k$ (points, %)")
    ax.set_title("Recovered $F_k$")
    num("distBiasPts", abs(rec - true_F) * 100, "{:.0f}")
    num("distBiasTrue", true_F * 100, "{:.0f}")
    panel(ax, "B")
    fig.tight_layout(w_pad=1.2)
    save(fig, "swvr_23_tornado")


def fig24_distance_bias():
    d = np.linspace(m.D_PEG_NM, m.D_MAX_NM, 300)
    w = L.pd_norm(d)
    dbar = np.trapezoid(w * d, d)
    y0 = L.dist_avg(E, FREQ, dgrid=d, weights=w, n=len(d)).max()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax = axes[0]
    for lam, col in ((-2.0, C["red"]), (-1.0, C["orange"]), (0.0, C["navy"]), (1.0, C["green"]), (2.0, C["blue"])):
        q = np.exp(lam * (d - dbar)); q /= np.trapezoid(w * q, d)
        ax.plot(d, q, color=col, label=f"$\\lambda$ = {lam:g} nm$^{{-1}}$")
    ax.set_xlabel("$d$ (nm)"); ax.set_ylabel("relative cleavage rate $q(d)$")
    ax.set_yscale("log")
    ax.legend(fontsize=6.5)
    ax.set_title("Assumed distance dependence")
    panel(ax, "A")
    ax = axes[1]
    kts = np.linspace(0.01, 4, 60)
    for lam, col in ((-2.0, C["red"]), (-1.0, C["orange"]), (0.0, C["navy"]), (1.0, C["green"]), (2.0, C["blue"])):
        q = np.exp(lam * (d - dbar)); q /= np.trapezoid(w * q, d)
        tr, rc = [], []
        for kt in kts:
            s = np.exp(-kt * q)
            tr.append(1 - np.trapezoid(w * s, d))
            yT = L.dist_avg(E, FREQ, dgrid=d, weights=w * s, n=len(d)).max()
            rc.append(1 - yT / y0)
        ax.plot(np.array(tr) * 100, np.array(rc) * 100, color=col)
    ax.plot([0, 100], [0, 100], "--", color=C["grey"], lw=0.9)
    ax.set_xlabel("true fraction of probes cleaved (%)"); ax.set_ylabel("recovered $F_k^{\\rm rec}$ (%)")
    ax.set_title("Readout under biased cleavage")
    panel(ax, "B")
    fig.tight_layout(w_pad=1.2)
    save(fig, "swvr_24_distance_bias")


def fig25_chain_map():
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 2)
    steps = [
        ("[MMP-9], $T$", "wound fluid\nconcentration,\nincubation time", "VII", C["orange"]),
        ("$F_k$", "fraction of probes\ncut (first-order\nkinetics)", "VII", C["orange"]),
        ("$(1-F_k)\\,\\Gamma_{\\rm tot}$", "intact MB probes\nremaining", "II", C["green"]),
        ("$P(d)$, $k^0(d)$", "where the MB label\nsits, how fast it\nexchanges", "III", C["purple"]),
        ("$\\Delta I(E)$", "square-wave\nvoltammogram", "IV, V", C["blue"]),
        ("$F_k^{\\rm rec}$", "cleavage read back\nfrom the peak\nheight", "VI", C["navy"]),
    ]
    from matplotlib.patches import FancyBboxPatch
    w = 1.55
    for i, (a, b, part, col) in enumerate(steps):
        x = 0.05 + i * 1.65
        ax.add_patch(FancyBboxPatch((x, 0.2), w, 1.6, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.5))
        ax.text(x + w / 2, 1.5, a, ha="center", va="center", fontsize=8.5, color=col, fontweight="bold")
        ax.text(x + w / 2, 0.85, b, ha="center", va="center", fontsize=6.8)
        ax.text(x + w / 2, 0.32, f"Part {part}", ha="center", va="center", fontsize=6.5, color=C["grey"])
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + 0.1, 1.0), xytext=(x + w, 1.0), arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.1))
    save(fig, "swvr_25_chain_map")


def build():
    fig16_potentials()
    fig17_drift()
    fig18_coverage()
    fig19_kinetics()
    fig20_voltammograms()
    fig21_lod()
    fig22_eta()
    fig23_tornado()
    fig24_distance_bias()
    fig25_chain_map()
