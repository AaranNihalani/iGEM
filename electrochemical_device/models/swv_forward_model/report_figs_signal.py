"""Figures for Parts III to V: tether geometry, the single-probe model, the
distance-averaged voltammogram, and the case for square-wave voltammetry."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf

import swv_forward_model as m
import report_lib as L
from report_lib import C, panel, save, num

F, R, T = m.F_const, m.R_const, m.T_K
E = m.E_SCAN
FREQ = 10.0


# ------------------------------------------------------------------ helpers
def _kappa_grid():
    return np.logspace(-3, 1.6, 300)


def _two_term_swv(Ev, k0, alpha=0.37, ne=2, f=10.0, Esw=0.025, E0=-0.25):
    """The naive current I = ne F (kb Gred - kf Gox), for the stability figure."""
    tau = 1 / (2 * f)

    def half(g, Ep):
        xi = ne * F * (Ep - E0) / (R * T)
        kf, kb = k0 * np.exp(-alpha * xi), k0 * np.exp((1 - alpha) * xi)
        K = kf + kb
        ge = kf / K
        g1 = ge + (g - ge) * np.exp(-K * tau)
        return g1, ne * F * (kb * g1 - kf * (1 - g1))
    g = 1.0
    for _ in range(5):
        g, _ = half(g, Ev[0] + Esw); g, _ = half(g, Ev[0] - Esw)
    out = []
    for Es in Ev:
        g, a = half(g, Es + Esw); g, b = half(g, Es - Esw)
        out.append(a - b)
    return np.array(out)


# ------------------------------------------------------------------ Part III
def fig05_chain():
    rng = np.random.default_rng(0)
    b, N = m.B_KUHN_NM, int(round(m.N_KUHN))
    n = 400000
    u = rng.uniform(-1, 1, size=(n, N))
    zc = b * u.cumsum(axis=1)
    zend = zc[:, -1]
    wall_ok = (zc >= 0).all(axis=1)
    zw = zend[wall_ok]
    num("wallAccept", wall_ok.mean() * 100, "{:.0f}")
    num("nKuhn", N, "{:d}")
    num("sigmaZ", m.SIGMA_Z_NM, "{:.2f}")
    num("dPEG", m.D_PEG_NM, "{:.2f}")
    num("dMax", m.D_MAX_NM, "{:.1f}")
    num("Rsq", m.R2_MEAN_NM2, "{:.2f}")
    d_ana_mean = m.D_PEG_NM + m.SIGMA_Z_NM * np.sqrt(2 / np.pi)
    num("meanDana", d_ana_mean, "{:.2f}")
    num("meanDfold", (m.D_PEG_NM + np.abs(zend)).mean(), "{:.2f}")
    num("meanDwall", (m.D_PEG_NM + zw).mean(), "{:.2f}")
    # Kolmogorov distance between exact FJC z and the Gaussian
    zs = np.sort(zend)
    cdf_emp = np.arange(1, n + 1) / n
    cdf_gau = 0.5 * (1 + erf(zs / (np.sqrt(2) * m.SIGMA_Z_NM)))
    num("ksGauss", np.max(np.abs(cdf_emp - cdf_gau)), "{:.3f}")
    # signal under the three distributions
    def signal(dsamp):
        edges = np.linspace(m.D_PEG_NM, m.D_MAX_NM + 0.4, 90)
        h, _ = np.histogram(dsamp, bins=edges, density=True)
        c = 0.5 * (edges[1:] + edges[:-1])
        return L.dist_avg(E, FREQ, dgrid=c, weights=h, n=len(c), d_max=c[-1]).max()
    s_ana = L.dist_avg(E, FREQ, n=400).max()
    s_fold = signal(m.D_PEG_NM + np.abs(zend))
    s_wall = signal(m.D_PEG_NM + zw)
    num("sigAna", s_ana * 1e9, "{:.1f}")
    num("sigFold", s_fold * 1e9, "{:.1f}")
    num("sigWall", s_wall * 1e9, "{:.1f}")
    num("wallDrop", (1 - s_wall / s_ana) * 100, "{:.0f}")
    num("foldDiff", abs(s_fold / s_ana - 1) * 100, "{:.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    rs = np.random.default_rng(4)
    k = 0
    while k < 6:
        uu = rs.uniform(-1, 1, size=N)
        if (np.cumsum(b * uu) < 0).any():
            continue                       # keep only chains that stay above the tip
        phi = rs.uniform(0, 2 * np.pi, size=N)
        st = np.sqrt(1 - uu ** 2)
        xs = np.concatenate([[0], np.cumsum(b * st * np.cos(phi))])
        zz = np.concatenate([[0], np.cumsum(b * uu)])
        ax.plot(xs, zz, "-o", ms=3, lw=1.0, color=[C["blue"], C["purple"], C["green"], C["orange"], C["red"], C["navy"]][k], alpha=0.85)
        k += 1
    ax.axhline(0, color="k", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("sideways (nm)"); ax.set_ylabel("height above the PEG6 tip (nm)")
    ax.set_title("Six chains that stay above the tip")
    panel(ax, "A")
    ax = axes[1]
    zg = np.linspace(-4, 4, 400)
    ax.hist(zend, bins=80, density=True, color=C["sky"], label="exact chain")
    ax.plot(zg, np.exp(-zg ** 2 / (2 * m.SIGMA_Z_NM ** 2)) / (np.sqrt(2 * np.pi) * m.SIGMA_Z_NM),
            color=C["red"], label="Gaussian")
    ax.set_xlabel("end height $z$ (nm)"); ax.set_ylabel("probability density (nm$^{-1}$)")
    ax.set_title("Gaussian is a good fit")
    ax.legend(fontsize=7)
    panel(ax, "B")
    ax = axes[2]
    dg = np.linspace(m.D_PEG_NM, m.D_MAX_NM, 400)
    ax.plot(dg, L.pd_norm(dg), color=C["red"], label="half-Gaussian (used)")
    ax.hist(m.D_PEG_NM + np.abs(zend), bins=60, range=(m.D_PEG_NM, m.D_MAX_NM), density=True,
            histtype="step", color=C["blue"], lw=1.3, label="exact chain, folded")
    ax.hist(m.D_PEG_NM + zw, bins=60, range=(m.D_PEG_NM, m.D_MAX_NM), density=True,
            histtype="step", color=C["green"], lw=1.3, label="exact chain, no bead below tip")
    ax.set_xlabel("label distance from gold $d$ (nm)"); ax.set_ylabel("$P(d)$ (nm$^{-1}$)")
    ax.set_title("Three versions of $P(d)$")
    ax.legend(fontsize=6.6)
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_05_chain")


def fig06_rate_distance():
    d = np.linspace(m.D_PEG_NM, m.D_MAX_NM, 500)
    k0 = m.k0_of_d(d)
    num("kRef", m.K0_REF_S1, "{:.0f}")
    num("kZero", m.K0_AT_0_S1, "{:.1e}")
    num("betaNM", m.BETA_NM, "{:.0f}")
    num("kappaTenHz", m.K0_REF_S1 / 20, "{:.1f}")
    num("kappaTwentyHz", m.K0_REF_S1 / 40, "{:.1f}")
    num("kOrder", np.log(10) / m.BETA_NM * 10, "{:.2f}")   # angstrom per decade
    kap = _kappa_grid()
    g = L.g_of_kappa(kap, f=FREQ)
    kmax = kap[g.argmax()]
    num("kappaMax", kmax, "{:.2f}")
    num("gMax", g.max(), "{:.2f}")
    gk = L.g_of_kappa(np.array([3.0, 5.0, m.K0_REF_S1 / 20]), f=FREQ)
    num("gThreePct", gk[0] / g.max() * 100, "{:.1f}")
    num("gFivePct", gk[1] / g.max() * 100, "{:.2f}")
    num("gTipPct", gk[2] / g.max() * 100, "{:.4f}")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    ax.semilogy(d, k0, color=C["red"])
    for f, ls in ((10, "-"), (20, "--")):
        ax.axhline(2 * f * kmax, color=C["grey"], ls=ls, lw=0.9)
        ax.text(2.9, 2 * f * kmax * 1.3, f"$k^0$ = {2*f*kmax:.0f} (best at {f} Hz)", fontsize=6.3, color=C["grey"], ha="right")
    ax.set_xlim(1.3, 3.0); ax.set_ylim(1e-1, 300)
    ax.set_xlabel("$d$ (nm)"); ax.set_ylabel("$k^0(d)$ (s$^{-1}$)")
    ax.set_title("Rate falls 10-fold per 0.23 nm")
    panel(ax, "A")
    ax = axes[1]
    ax.semilogx(kap, g / g.max(), color=C["blue"])
    ax.axvline(kmax, color=C["green"], ls=":")
    ax.text(kmax * 1.2, 0.05, f"maximum,\n$\\kappa={kmax:.2f}$", fontsize=7, color=C["green"])
    ax.axvline(m.K0_REF_S1 / 20, color=C["red"], ls="--", lw=1)
    ax.text(m.K0_REF_S1 / 20 * 1.1, 0.55, "tip of PEG6\nat 10 Hz", fontsize=7, color=C["red"])
    ax.set_xlabel("$\\kappa=k^0/(2f)$"); ax.set_ylabel("single-probe peak signal (relative)")
    ax.set_title("A probe of the right speed")
    panel(ax, "B")
    ax = axes[2]
    for f, col in ((10, C["blue"]), (20, C["orange"])):
        gg = np.interp(np.log(m.k0_of_d(d) / (2 * f)), np.log(kap), g)
        w = L.pd_norm(d) * gg
        ax.plot(d, w / w.max(), color=col, label=f"{f} Hz")
    ax.plot(d, L.pd_norm(d) / L.pd_norm(d).max(), color=C["grey"], ls="--", lw=1, label="$P(d)$ alone")
    ax.set_xlabel("$d$ (nm)"); ax.set_ylabel("contribution to signal (relative)")
    ax.set_title("Which labels make the signal")
    ax.legend(fontsize=7)
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_06_rate_distance")


# ------------------------------------------------------------------ Part IV
def fig07_ode():
    alpha, ne, E0 = 0.37, 2, -0.25
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    ks = (1.0, 10.0, 100.0)
    cols = (C["navy"], C["blue"], C["orange"])
    tau = 1 / (2 * FREQ)
    Ep = E0 + 0.03
    xi = ne * F * (Ep - E0) / (R * T)
    for k0, col in zip(ks, cols):
        kf, kb = k0 * np.exp(-alpha * xi), k0 * np.exp((1 - alpha) * xi)
        K = kf + kb
        geq = kf / K
        sol = solve_ivp(lambda t, y: [kf * (1 - y[0]) - kb * y[0]], (0, tau), [0.5],
                        t_eval=np.linspace(0, tau, 120), rtol=1e-11, atol=1e-13)
        closed = geq + (0.5 - geq) * np.exp(-K * sol.t)
        ax.plot(sol.t * 1e3, closed, color=col, label=f"$k^0$ = {k0:g} s$^{{-1}}$")
        ax.plot(sol.t[::10] * 1e3, sol.y[0][::10], "o", ms=3.5, color=col, mfc="white")
    ax.set_xlabel("time in the pulse (ms)"); ax.set_ylabel("reduced fraction $\\Gamma_{\\rm red}/\\Gamma_{\\rm tot}$")
    ax.set_title("Closed form vs ODE solver")
    ax.legend(fontsize=6.8)
    panel(ax, "A")
    # B: error between closed form and solver
    ax = axes[1]
    ks2 = np.logspace(-1, 3, 30)
    err = []
    for k0 in ks2:
        kf, kb = k0 * np.exp(-alpha * xi), k0 * np.exp((1 - alpha) * xi)
        K = kf + kb; geq = kf / K
        sol = solve_ivp(lambda t, y: [kf * (1 - y[0]) - kb * y[0]], (0, tau), [0.2],
                        t_eval=[tau], rtol=1e-12, atol=1e-14, method="LSODA")
        err.append(abs(sol.y[0][0] - (geq + (0.2 - geq) * np.exp(-K * tau))))
    ax.loglog(ks2, np.maximum(err, 1e-17), "o-", ms=3, color=C["blue"])
    num("odeMaxErr", max(err), "{:.0e}")
    ax.set_xlabel("$k^0$ (s$^{-1}$)"); ax.set_ylabel("|closed form $-$ solver|")
    ax.set_title("Agreement to solver tolerance")
    panel(ax, "B")
    # C: stable vs naive current formula
    ax = axes[2]
    Ev = np.linspace(-0.40, -0.10, 200)
    kap = np.logspace(-2, 2, 40)
    naive, stable = [], []
    gs = L.g_of_kappa(kap, f=FREQ)
    for kp in kap:
        y = _two_term_swv(Ev, kp * 2 * FREQ, f=FREQ)
        naive.append(np.abs(y).max() / (ne * F * 2 * FREQ))
    ax.loglog(kap, gs, color=C["blue"], label="$I=n_eFK(\\Gamma_{\\rm red}-\\Gamma_{\\rm eq})$")
    ax.loglog(kap, np.maximum(naive, 1e-30), "--", color=C["red"], label="$I=n_eF(k_b\\Gamma_{\\rm red}-k_f\\Gamma_{\\rm ox})$")
    ax.set_ylim(1e-25, 3)
    ax.set_xlabel("$\\kappa=k^0/(2f)$"); ax.set_ylabel("peak signal (relative units)")
    ax.set_title("Same formula, stable form")
    ax.legend(fontsize=6.5, loc="lower left")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_07_ode")


def fig08_waveform():
    f = FREQ
    tau = 1 / (2 * f)
    n_steps = 5
    Estep = 0.012
    Esw = 0.025
    E0s = -0.30
    t, Et = [], []
    for k in range(n_steps):
        Ek = E0s + k * Estep
        for j, sgn in enumerate((+1, -1)):
            t0 = (2 * k + j) * tau
            t += [t0, t0 + tau]
            Et += [Ek + sgn * Esw] * 2
    t = np.array(t); Et = np.array(Et)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    ax.step(t * 1e3, Et * 1e3, where="post", color=C["navy"], lw=1.4)
    stair = [E0s + k * Estep for k in range(n_steps)]
    for k in range(n_steps):
        ax.plot([2 * k * tau * 1e3, (2 * k + 2) * tau * 1e3], [stair[k] * 1e3] * 2, color=C["grey"], ls=":", lw=1)
        for j in range(2):
            ts = ((2 * k + j + 1) * tau) * 1e3
            ax.plot(ts, (stair[k] + (Esw if j == 0 else -Esw)) * 1e3, "o", color=[C["red"], C["blue"]][j], ms=4, zorder=5)
    ax.annotate("", xy=(3 * tau * 1e3, (stair[1] + Esw) * 1e3), xytext=(3 * tau * 1e3, (stair[1] - Esw) * 1e3),
                arrowprops=dict(arrowstyle="<->", color=C["green"], lw=1))
    ax.text(3 * tau * 1e3 + 4, stair[1] * 1e3 - 4, "$2E_{\\rm sw}$", color=C["green"], fontsize=7.5)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("applied potential (mV)")
    ax.set_title("Staircase plus square wave")
    panel(ax, "A")
    ax = axes[1]
    alpha, ne, E0 = 0.37, 2, -0.25
    k0 = 0.5 * 2 * f
    g = 1.0
    ts, gs = [], []
    tcur = 0.0
    for k in range(n_steps):
        Ek = E0s + 0.045 + k * 0.016
        for sgn in (+1, -1):
            Ep = Ek + sgn * Esw
            xi = ne * F * (Ep - E0) / (R * T)
            kf, kb = k0 * np.exp(-alpha * xi), k0 * np.exp((1 - alpha) * xi)
            K = kf + kb; geq = kf / K
            tt = np.linspace(0, tau, 40)
            gg = geq + (g - geq) * np.exp(-K * tt)
            ts += list(tcur + tt); gs += list(gg)
            g = gg[-1]; tcur += tau
    ax.plot(np.array(ts) * 1e3, gs, color=C["blue"])
    ax.set_xlabel("time (ms)"); ax.set_ylabel("reduced fraction")
    ax.set_title("Coverage carried across pulses")
    panel(ax, "B")
    ax = axes[2]
    Ee = np.linspace(-0.40, -0.10, 400)
    for kap, col in ((0.05, C["navy"]), (0.5, C["blue"]), (5.0, C["orange"])):
        y = L.swv_multi(Ee, [kap * 2 * f], f=f)[0]
        ax.plot(Ee * 1e3, y / (ne * F * 2 * f), color=col, label=f"$\\kappa$ = {kap:g}")
    ax.set_xlabel("staircase potential (mV)"); ax.set_ylabel("net current (units of $n_eF\\Gamma A\\cdot2f$)")
    ax.set_title("Net current $=I_{\\rm fwd}-I_{\\rm rev}$")
    ax.legend(fontsize=7)
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_08_waveform")


def fig09_kappa_shapes():
    kap = np.logspace(-2, 1.4, 60)
    Ev = np.linspace(-0.40, -0.10, 601)
    y = L.swv_multi(Ev, kap * 2 * FREQ, f=FREQ)
    fw, ep, pk = [], [], []
    for row in y:
        h, e, w = L.peak_stats(Ev, row)
        fw.append(w * 1e3); ep.append((e + 0.25) * 1e3); pk.append(h)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    for kp, col in ((0.03, C["navy"]), (0.5, C["blue"]), (3.0, C["green"]), (8.0, C["orange"])):
        row = L.swv_multi(Ev, [kp * 2 * FREQ], f=FREQ)[0]
        ax.plot((Ev + 0.25) * 1e3, row / row.max(), color=col, label=f"$\\kappa$ = {kp:g}")
    ax.set_xlabel("$E-E^{0\\prime}$ (mV)"); ax.set_ylabel("net current / its peak")
    ax.legend(fontsize=7)
    ax.set_title("Peak shape")
    panel(ax, "A")
    ax = axes[1]
    ax.semilogx(kap, fw, color=C["green"])
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("FWHM (mV)")
    ax.set_title("Width")
    num("fwhmMin", min(fw), "{:.0f}"); num("fwhmMaxK", max(fw), "{:.0f}")
    panel(ax, "B")
    ax = axes[2]
    ax.semilogx(kap, ep, color=C["red"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("peak potential $-E^{0\\prime}$ (mV)")
    ax.set_title("Position")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_09_kappa_shapes")


def fig10_cleavage_response():
    fks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    base = L.dist_avg(E, FREQ, n=400)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(fks)))
    peaks, epk, fw = [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    for fk, col in zip(fks, cols):
        y = base * (1 - fk)
        ax.plot(E * 1e3, y * 1e9, color=col, label=f"$F_k$ = {fk:.2f}")
        h, e, w = L.peak_stats(E, y)
        peaks.append(h); epk.append(e * 1e3); fw.append(w * 1e3)
    ax.set_xlabel("potential (mV vs Ag/AgCl)"); ax.set_ylabel("net current (nA)")
    ax.legend(fontsize=6.5)
    ax.set_title("Voltammograms")
    panel(ax, "A")
    ax = axes[1]
    x = 1 - np.array(fks)
    ax.plot(x, np.array(peaks) * 1e9, "o", color=C["blue"], ms=5, label="model")
    ax.plot([0, 1], [0, peaks[0] * 1e9], "-", color=C["red"], lw=1, label="proportional line")
    ax.set_xlabel("intact probe fraction $1-F_k$"); ax.set_ylabel("peak height (nA)")
    ax.legend(fontsize=7)
    ax.set_title("Height vs intact probe")
    panel(ax, "B")
    ax = axes[2]
    ax.plot(np.array(fks) * 100, epk, "o-", color=C["blue"], ms=4)
    ax.set_xlabel("$F_k$ (%)"); ax.set_ylabel("peak potential (mV)", color=C["blue"])
    ax2 = ax.twinx()
    ax2.plot(np.array(fks) * 100, fw, "s--", color=C["red"], ms=4)
    ax2.set_ylabel("FWHM (mV)", color=C["red"])
    ax2.grid(False)
    ax.set_ylim(min(epk) - 5, max(epk) + 5); ax2.set_ylim(min(fw) - 5, max(fw) + 5)
    ax.set_title("Position and width unchanged")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_10_cleavage_response")
    num("peakLinDev", np.max(np.abs(np.array(peaks) / peaks[0] - x)), "{:.0e}")


def fig11_signal_band():
    freqs = [5, 10, 20, 50, 100]
    cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(freqs)))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    d_star = []
    for f, col in zip(freqs, cols):
        tot, d, w, k0, per_d = L.dist_avg(E, f, n=500, return_parts=True)
        i = int(np.argmax(tot))
        dens = w * per_d[:, i]
        ax.plot(d, dens / dens.max(), color=col, label=f"{f} Hz")
        d_star.append(d[np.argmax(dens)])
    ax.set_xlabel("$d$ (nm)"); ax.set_ylabel("contribution at the peak (relative)")
    ax.set_xlim(1.3, 3.0)
    ax.set_title("Signal comes from a narrow band")
    ax.legend(fontsize=6.5)
    panel(ax, "A")
    ax = axes[1]
    kmax = float(L.NUMBERS["kappaMax"][0])
    for f_, nm in ((10, "dStarTen"), (20, "dStarTwenty")):
        num(nm, m.D_PEG_NM + np.log(m.K0_REF_S1 / (2 * f_ * kmax)) / m.BETA_NM, "{:.2f}")
        num(nm + "Off", np.log(m.K0_REF_S1 / (2 * f_ * kmax)) / m.BETA_NM, "{:.2f}")
    fgrid = np.logspace(0.4, 2.3, 60)
    d_th = m.D_PEG_NM + np.log(m.K0_REF_S1 / (2 * fgrid * kmax)) / m.BETA_NM
    ax.semilogx(fgrid, d_th, color=C["grey"], label="$d_\\star=d_{\\rm PEG}+\\ln[k_{\\rm ref}/(2f\\kappa_{\\max})]/\\beta$")
    ax.semilogx(freqs, d_star, "o", color=C["red"], label="numerical band centre")
    ax.axhline(m.D_PEG_NM, color=C["light"], lw=1.2)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("$d_\\star$ (nm)")
    ax.set_title("Band centre vs frequency")
    ax.legend(fontsize=6.3, loc="upper right")
    panel(ax, "B")
    ax = axes[2]
    kap = _kappa_grid()
    g = L.g_of_kappa(kap)
    G = np.trapezoid(g, np.log(kap))
    num("Gint", G, "{:.2f}")
    fgrid2 = np.array([5, 10, 20, 40])
    exact = np.array([L.dist_avg(E, f, n=500).max() for f in fgrid2])
    P0 = np.array([L.pd_norm(np.array([m.D_PEG_NM + np.log(m.K0_REF_S1 / (2 * f * kmax)) / m.BETA_NM]))[0] for f in fgrid2])
    approx = 2 * fgrid2 * 2 * F * m.A_ELEC * m.GAMMA_0 * P0 / m.BETA_NM * G
    ax.loglog(fgrid2, exact * 1e9, "o", color=C["blue"], label="full model")
    ax.loglog(fgrid2, approx * 1e9, "-", color=C["red"], label="one-line formula")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("peak height (nA)")
    ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels(["5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_title("One-line formula")
    ax.legend(fontsize=7)
    num("approxErrTen", (approx[1] / exact[1] - 1) * 100, "{:.0f}")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_11_signal_band")


def fig12_frequency():
    fgrid = np.logspace(0, 2.6, 26)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    for s, col in ((0.1, C["navy"]), (1.0, C["blue"]), (10.0, C["orange"])):
        y = [L.dist_avg(E, f, n=500, k0_ref=m.K0_REF_S1 * s).max() * 1e9 for f in fgrid]
        ax.loglog(fgrid, y, color=col, label=f"$k_{{\\rm ref}}\\times${s:g}")
        if s == 1.0:
            fopt = fgrid[int(np.argmax(y))]
            base_y = y
    ax.axvspan(10, 20, color=C["gold"], alpha=0.25)
    ax.text(10.5, ax.get_ylim()[0] * 3, "10-20 Hz\n(operating)", fontsize=6.5)
    ax.set_xlabel("SWV frequency (Hz)"); ax.set_ylabel("peak height (nA)")
    ax.set_title("Predicted frequency response")
    ax.legend(fontsize=6.8, loc="upper left")
    num("fOpt", fopt, "{:.0f}")
    panel(ax, "A")
    ax = axes[1]
    scales = np.logspace(-2.5, 2.5, 21)
    y = np.array([L.dist_avg(E, FREQ, n=500, k0_ref=m.K0_REF_S1 * s).max() for s in scales])
    ax.semilogx(scales * m.K0_REF_S1, y / y[np.argmin(abs(scales - 1))], color=C["blue"])
    ax.axvline(m.K0_REF_S1, color=C["red"], ls="--", lw=1)
    ax.axhspan(0.75, 1.25, color=C["green"], alpha=0.12)
    ax.set_xlabel("$k_{\\rm ref}$ at the PEG6 tip (s$^{-1}$)"); ax.set_ylabel("peak height (relative to base)")
    ax.set_ylim(0, 1.5)
    ax.set_title("Insensitive to $k_{\\rm ref}$")
    sel = (scales >= 0.1) & (scales <= 100)
    rel = y / y[np.argmin(abs(scales - 1))]
    num("kRefSpanLo", rel[sel].min(), "{:.2f}")
    num("kRefSpanHi", rel[sel].max(), "{:.2f}")
    panel(ax, "B")
    ax = axes[2]
    ratio = []
    fpair = np.array([5, 10, 15, 20, 30, 40])
    y10 = L.dist_avg(E, 10.0, n=500).max()
    for f in fpair:
        ratio.append(L.dist_avg(E, f, n=500).max() / y10)
    ax.plot(fpair, ratio, "o-", color=C["blue"], ms=4, label="model")
    ax.plot(fpair, fpair / 10.0, "--", color=C["grey"], label="proportional to $f$")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("peak / peak at 10 Hz")
    ax.set_title("Testable prediction")
    ax.legend(fontsize=7)
    num("ratioTwenty", ratio[3], "{:.2f}")
    num("ratioForty", ratio[-1], "{:.2f}")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_12_frequency")


def fig13_quadrature():
    ns = np.array([10, 20, 40, 80, 160, 320, 640])
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6))
    for f, col in ((10, C["blue"]), (20, C["orange"]), (50, C["green"])):
        ref = L.dist_avg(E, f, n=2000).max()
        err = [abs(L.dist_avg(E, f, n=n).max() / ref - 1) * 100 for n in ns]
        ax[0].loglog(ns, np.maximum(err, 1e-6), "o-", ms=4, color=col, label=f"{f} Hz")
        if f == 10:
            num("quadErrEighty", err[3], "{:.5f}")
        if f == 20:
            num("quadErrEightyTwentyHz", err[3], "{:.3f}")
    ax[0].axvline(80, color=C["grey"], ls=":")
    ax[0].axhline(1, color=C["red"], lw=0.8, ls="--")
    ax[0].set_xlabel("quadrature points $n$"); ax[0].set_ylabel("error in peak height (%)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("Distance quadrature converges")
    panel(ax[0], "A")
    # B: rate of decay of neglected tail
    dmax = np.linspace(2.5, 6, 30)
    tail = 1 - erf((dmax - m.D_PEG_NM) / (np.sqrt(2) * m.SIGMA_Z_NM))
    ax[1].semilogy(dmax, tail, color=C["blue"])
    ax[1].axvline(m.D_MAX_NM, color=C["red"], ls="--", lw=1)
    num("tailMass", (1 - erf((m.D_MAX_NM - m.D_PEG_NM) / (np.sqrt(2) * m.SIGMA_Z_NM))), "{:.1e}")
    ax[1].set_xlabel("upper cut-off $d_{\\max}$ (nm)"); ax[1].set_ylabel("probability beyond cut-off")
    ax[1].set_title("Truncation at 5.2 nm is negligible")
    panel(ax[1], "B")
    fig.tight_layout(w_pad=1.2)
    save(fig, "swvr_13_quadrature")


# ------------------------------------------------------------------ Part V
def fig14_background():
    A = m.A_ELEC
    Cdl = 20e-6              # F cm^-2, typical gold in aqueous electrolyte (assumed)
    Ru = 1.0e3               # ohm, assumed
    Cap = Cdl * A
    rc = Ru * Cap
    tau = 1 / (2 * FREQ)
    num("CdlUF", Cdl * 1e6, "{:.0f}")
    num("Cdev", Cap * 1e6, "{:.2f}")
    num("RuOhm", Ru, "{:.0f}")
    num("RCms", rc * 1e3, "{:.2f}")
    num("halfPulseMs", tau * 1e3, "{:.0f}")
    num("chargeDecayExp", tau / rc, "{:.0f}")
    ratio = 2 ** 2 * F ** 2 * m.GAMMA_0 / (4 * R * T * Cdl)
    num("cvRatio", ratio, "{:.2f}")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    t = np.linspace(0, tau, 400)
    kf = 0.5 * 2 * FREQ
    ax.semilogy(t * 1e3, np.exp(-t / rc), color=C["red"], label="capacitive (charging)")
    ax.semilogy(t * 1e3, np.exp(-2 * kf * t), color=C["blue"], label="Faradaic, $\\kappa=0.5$")
    ax.set_ylim(1e-30, 3)
    ax.axvline(tau * 1e3, color=C["grey"], ls=":")
    ax.text(tau * 1e3 * 0.9, 1e-14, "sample\nhere", fontsize=7, color=C["grey"], ha="right")
    ax.set_xlabel("time after the potential jump (ms)"); ax.set_ylabel("current / its initial value")
    ax.set_title("Charging dies before sampling")
    ax.legend(fontsize=6.5, loc="lower left")
    panel(ax, "A")
    ax = axes[1]
    Ev = np.linspace(-0.55, 0.05, 400)
    xi = 2 * F * (Ev + 0.25) / (R * T)
    eta = np.exp(np.clip(xi, -60, 60))
    psi = eta / (1 + eta) ** 2 * 4 * ratio          # reversible CV wave in units of C_dl A v
    ax.plot(Ev * 1e3, 1 + psi, color=C["blue"], label="anodic sweep")
    ax.plot(Ev * 1e3, -(1 + psi), color=C["navy"], label="cathodic sweep")
    ax.axhline(1, color=C["red"], ls=":", lw=0.8); ax.axhline(-1, color=C["red"], ls=":", lw=0.8)
    ax.text(-540, 1.06, "charging plateau", color=C["red"], fontsize=7)
    ax.set_xlabel("potential (mV)"); ax.set_ylabel("current / ($C_{\\rm dl}Av$)")
    ax.set_title("CV: signal sits on a plateau")
    ax.legend(fontsize=6.5, loc="center left")
    panel(ax, "B")
    ax = axes[2]
    lab = ["CV: charging\nper Faradaic", "SWV: charging\nper Faradaic"]
    ax.bar([0], [1 / ratio], color=C["red"], width=0.5)
    ax.text(0, 1 / ratio * 1.05, f"{1/ratio:.1f}", ha="center", fontsize=8)
    expo = int(tau / rc * 0.4343)
    ax.bar([1], [10.0 ** (-expo)], color=C["blue"], width=0.5)
    ax.text(1, 10.0 ** (-expo) * 5, f"$\\approx10^{{-{expo}}}$", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(10.0 ** (-expo - 6), 100)
    ax.set_xticks([0, 1]); ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("charging / Faradaic current")
    ax.set_title("Background at the sampling instant")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.0)
    save(fig, "swvr_14_background")


def fig15_dd():
    kap = np.logspace(-2, 2.6, 120)
    Ev = np.linspace(-0.40, -0.10, 801)
    y = L.swv_multi(Ev, kap * 2 * FREQ, f=FREQ)
    fw = np.array([L.peak_stats(Ev, r)[2] * 1e3 if r.max() > 0 else np.nan for r in y])
    g = L.g_of_kappa(kap)
    k0_dd = 5.0e4
    kdd = {f: k0_dd / (2 * f) for f in (20, 100, 200, 500, 1000)}
    num("kappaDDhundred", kdd[100], "{:.0f}")
    num("kappaDDtwenty", kdd[20], "{:.0f}")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax = axes[0]
    ax.loglog(kap, np.maximum(g / g.max(), 1e-40), color=C["blue"])
    ax.axvspan(kdd[1000], kdd[20], color=C["red"], alpha=0.12)
    ax.text(kdd[1000] * 1.1, 1e-20, "Dauphin-Ducharme\nMB data,\n20-1000 Hz", fontsize=7, color=C["red"])
    ax.axvspan(m.K0_REF_S1 / 40, m.K0_REF_S1 / 20, color=C["green"], alpha=0.25)
    ax.text(m.K0_REF_S1 / 40 * 0.15, 1e-20, "this\ndevice", fontsize=7, color=C["green"])
    ax.set_ylim(1e-40, 3)
    ax.set_xlabel("$\\kappa=k^0/(2f)$"); ax.set_ylabel("single-probe peak (relative)")
    ax.set_title("Signal predicted by the model")
    panel(ax, "A")
    ax = axes[1]
    ax.semilogx(kap, fw, color=C["green"], label="model, single rate")
    ax.axhspan(70, 90, color=C["red"], alpha=0.18)
    ax.text(0.012, 82, "Fig. 4B, 100 Hz, 25 mV\n(read from figure, $\\pm10$ mV)", fontsize=7, color=C["red"], va="center")
    ax.set_ylim(40, 100); ax.set_xlim(0.01, 40)
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("FWHM of net peak (mV)")
    ax.set_title("Peak width")
    ax.legend(fontsize=7, loc="lower right")
    panel(ax, "B")
    fig.tight_layout(w_pad=1.2)
    save(fig, "swvr_15_dd")


def build():
    fig05_chain()
    fig06_rate_distance()
    fig07_ode()
    fig08_waveform()
    fig09_kappa_shapes()
    fig10_cleavage_response()
    fig11_signal_band()
    fig12_frequency()
    fig13_quadrature()
    fig14_background()
    fig15_dd()
