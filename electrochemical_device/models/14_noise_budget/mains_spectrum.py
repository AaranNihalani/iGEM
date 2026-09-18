"""
mains_spectrum.py  --  Layer 14, Sec. 14.6: mains interference
================================================================
Two questions:

 (1) HOW BIG is the mains pickup for a given shielding scenario?
        i_pk = 2 pi f_m C_c V_pk / 10^(SE/20)
     (displacement current through a stray coupling capacitance C_c from the
     mains wiring to the high-impedance WE loop, reduced by the shield's
     effectiveness SE).  C_c and SE are ASSUMED scenario values -- they must be
     measured (short the WE input, scan with the cell off) before being trusted.

 (2) WHAT DOES IT DO to the SWV measurement?
     The SWV net current is dI_n = I_fwd(t_2n) - I_rev(t_2n+1), one sample per half-cycle
     (spacing tau = 1/(2 f_SWV)).  A tone m(t) = A cos(2 pi f_m t + phi) therefore appears as

        dI_n = G * A * sin(2 pi nu n + phi'),
        G  = 2 |sin(pi f_m tau)| * |sinc(f_m T_int)|     (differencing x boxcar averaging)
        nu = f_m / f_SWV  mod 1     (cycles per staircase step; beat period = 1/nu steps)

     so mains is NOT white noise: it is a ripple in E-space of period dE_step/nu that
     the Savitzky-Golay filter, the cubic baseline and the Gaussian fit then convert
     into an area error.  The pipeline response is obtained by Monte Carlo over the
     (unknown) phase and small grid-frequency detuning.

     Design consequences printed at the end:
       * synchronous window: T_int = 1/f_m  nulls the tone exactly (needs tau >= 1/f_m,
         i.e. f_SWV <= f_m/2: 25 Hz on a 50 Hz grid, 30 Hz on a 60 Hz grid)
       * choosing f_SWV != f_m alone does NOT remove the tone; it only moves the
         ripple in E-space (contents.md option 2).

Run:  python mains_spectrum.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import noise_budget as nb
import sigma_area as sa

SQ2PI = np.sqrt(2 * np.pi)


# --------------------------------------------------------------------------
# (1) Amplitude scenarios
# --------------------------------------------------------------------------
def pickup_amplitude(C_c: float, SE_dB: float, f_m: float = 50.0, V_rms: float = 230.0) -> float:
    """Peak pickup current (A) at the TIA input."""
    return 2 * np.pi * f_m * C_c * np.sqrt(2) * V_rms / 10 ** (SE_dB / 20.0)


SCENARIOS = [
    # label, C_c (F), SE (dB)
    ("Unshielded, open bench (1 pF)", 1.0e-12, 0.0),
    ("Unshielded, short leads (0.1 pF)", 0.1e-12, 0.0),
    ("Grounded foil, partial (30 dB)", 0.1e-12, 30.0),
    ("Closed Faraday cage (60 dB)", 0.1e-12, 60.0),
]


# --------------------------------------------------------------------------
# (2) Sampling / aliasing model
# --------------------------------------------------------------------------
def transfer(f_m: float, f_swv: float, T_int: float | None = None) -> dict:
    """Gain of the net SWV current to a mains tone, alias frequency and E-space beat."""
    tau = 1.0 / (2 * f_swv)
    T_int = tau if T_int is None else T_int
    gain_diff = 2 * abs(np.sin(np.pi * f_m * tau))
    gain_box = abs(np.sinc(f_m * T_int))          # np.sinc(x) = sin(pi x)/(pi x)
    nu = (f_m / f_swv) % 1.0
    nu = min(nu, 1 - nu)                           # fold to [0, 0.5]
    return {"gain": gain_diff * gain_box, "gain_diff": gain_diff, "gain_box": gain_box, "nu": nu,
            "alias_Hz": nu * f_swv,
            "beat_mV": (np.inf if nu < 1e-9 else 2.0 / nu)}     # 2 mV step / nu


def synchronous_f_swv(f_m: float) -> float:
    """Largest f_SWV whose half-cycle contains exactly one mains period (T_int = tau = 1/f_m)."""
    return f_m / 2.0


def ripple_trace(E: np.ndarray, A_in: float, f_m: float, f_swv: float, phi: float,
                 dE: float = 2e-3, detune: float = 0.0, T_int: float | None = None) -> np.ndarray:
    tr = transfer(f_m + detune, f_swv, T_int)
    n = np.arange(E.size)
    nu = ((f_m + detune) / f_swv)
    return tr["gain"] * A_in * np.sin(2 * np.pi * nu * n + phi)


def area_error(A_in: float, f_m: float, f_swv: float, n_trials: int = 150, seed: int = 3,
               A_area: float = nb.A_MB_L13, detune_Hz: float = 0.2, T_int: float | None = None) -> dict:
    """RMS relative area error caused by a mains tone of peak amplitude A_in (A)."""
    rng = np.random.default_rng(seed)
    dE, s = 2e-3, nb.SIGMA_E
    E = np.arange(sa.E_START, sa.E_END + dE / 2, dE)
    A_pk = A_area / (s * SQ2PI)
    clean = sa.gaussian(E, A_pk, sa.E_MB, s)
    fit0, _ = sa.pipeline_once(E, clean, sa.E_MB, s)      # noiseless pipeline output (SG bias ~ -0.27 %)
    err = np.empty(n_trials)
    for i in range(n_trials):
        rip = ripple_trace(E, A_in, f_m, f_swv, rng.uniform(0, 2 * np.pi), dE,
                           rng.uniform(-detune_Hz, detune_Hz), T_int)
        fit, _ = sa.pipeline_once(E, clean + rip, sa.E_MB, s)
        err[i] = fit - fit0          # error caused by the mains ripple alone
    err = err[np.isfinite(err)]
    return {"rms_rel": float(np.sqrt(np.mean(err ** 2)) / A_area),
            "mean_rel": float(np.mean(err) / A_area), "n_ok": int(err.size)}


# --------------------------------------------------------------------------
def main() -> dict:
    res = {}
    E_uk, E_us = 50.0, 60.0

    # ---- (1) amplitudes
    print("=" * 72)
    print("Layer 14 / 14.6: mains pickup amplitude (230 V, 50 Hz grid)")
    print("=" * 72)
    amp = []
    sigma_dI = nb.budget(nb.DSTAT, nb.Cell())["sigma_dI_A"]
    A_peak = nb.A_MB_L13 / (nb.SIGMA_E * SQ2PI)
    for lab, Cc, SE in SCENARIOS:
        a = pickup_amplitude(Cc, SE, E_uk)
        amp.append({"label": lab, "C_c_pF": Cc * 1e12, "SE_dB": SE, "i_pk_A": a})
        print(f"{lab:<36} C_c = {Cc*1e12:5.2f} pF, SE = {SE:4.0f} dB -> i_pk = {a*1e9:9.4f} nA "
              f"({a/A_peak*100:8.3f}% of MB peak, {a/sigma_dI:8.1f}x electronic sigma_dI)")
    res["amplitudes"] = amp

    # ---- transfer table
    print("\nTransfer of a mains tone into the net SWV current (T_int = one half-cycle)")
    print(f"{'f_m':>5}{'f_SWV':>7}{'|G|':>8}{'alias (Hz)':>12}{'beat (mV)':>11}")
    rows = []
    for f_m in (50.0, 60.0):
        for f_swv in (25.0, 30.0, 45.0, 50.0, 60.0, 80.0, 100.0):
            t = transfer(f_m, f_swv)
            rows.append({"f_m": f_m, "f_swv": f_swv, **t})
            beat = "  (DC)" if np.isinf(t["beat_mV"]) else f"{t['beat_mV']:.1f}"
            print(f"{f_m:5.0f}{f_swv:7.0f}{t['gain']:8.3f}{t['alias_Hz']:12.1f}{beat:>11}")
    res["transfer"] = rows

    # ---- (2) pipeline response: area error per nA of pickup vs f_SWV
    f_grid = np.arange(15, 121, 1.0)
    curves = {}
    for f_m in (E_uk, E_us):
        per_nA = []
        for f_swv in f_grid:
            r = area_error(100e-12, f_m, f_swv)
            per_nA.append(r["rms_rel"] / 0.1)
        curves[f_m] = np.array(per_nA)
    res["f_grid"] = f_grid.tolist()
    res["rel_err_per_nA_UK"] = curves[E_uk].tolist()
    res["rel_err_per_nA_US"] = curves[E_us].tolist()

    # linearity check
    lin = [area_error(a, E_uk, 60.0)["rms_rel"] / (a / 1e-9) for a in (10e-12, 100e-12, 300e-12)]
    print(f"\nlinearity check (rel. area error per nA at f_m=50, f_SWV=60): 10 pA {lin[0]:.4f} | 100 pA {lin[1]:.4f} | 300 pA {lin[2]:.4f}")
    res["linearity_per_nA"] = lin

    # ---- scenario -> sigma_A/A
    print("\nPipeline area error from residual mains (UK 50 Hz), no other noise")
    sc = []
    for f_swv, T_int, tag in [(60.0, None, "60 Hz, boxcar = half-cycle"),
                              (45.0, None, "45 Hz, boxcar = half-cycle"),
                              (25.0, 1 / 50.0, "25 Hz, synchronous window (T_int = 20 ms)")]:
        for a in amp:
            r = area_error(a["i_pk_A"], E_uk, f_swv, T_int=T_int)
            rel = r["rms_rel"]
            valid = a["i_pk_A"] < 0.5 * A_peak       # tone comparable to the peak => fit meaningless
            sc.append({"f_swv": f_swv, "tag": tag, "label": a["label"], "rel_pct": 100 * rel, "valid": valid})
            txt = f"{100*rel:10.4f}%" if valid else "   > 100 % (peak unrecoverable)"
            print(f"  {tag:<44}{a['label']:<36} sigma_A/A = {txt}")
    res["scenario_area_error"] = sc
    elec = 0.0375
    res["electronic_floor_pct"] = elec

    # required attenuation to keep mains below electronics floor / 1 % / 3 %
    per_nA60 = area_error(100e-12, E_uk, 60.0)["rms_rel"] / 0.1
    unshielded = amp[0]["i_pk_A"]
    for tgt_pct, name in [(elec, "electronic floor"), (1.0, "1 %"), (3.0, "3 % (Layer 13 budget)")]:
        need = per_nA60 * unshielded / 1e-9 / (tgt_pct / 100)
        print(f"attenuation needed at f_SWV = 60 Hz for mains area error < {name}: {20*np.log10(max(need,1)):.1f} dB "
              f"(from 1 pF unshielded)")
        res[f"atten_dB_{name}"] = 20 * np.log10(max(need, 1))
    print(f"\nsynchronous f_SWV: {synchronous_f_swv(50):.0f} Hz (50 Hz grid), {synchronous_f_swv(60):.0f} Hz (60 Hz grid)")

    make_figure(res, amp, sigma_dI)
    (nb.results_dir() / "mains_spectrum.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def make_figure(res: dict, amp: list, sigma_dI: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    A_peak = nb.A_MB_L13 / (nb.SIGMA_E * SQ2PI)

    ax = axes[0]
    labs = ["Unshielded\n1 pF", "Unshielded\n0.1 pF", "Foil\n30 dB", "Cage\n60 dB"]
    vals = [a["i_pk_A"] * 1e9 for a in amp]
    ax.bar(labs, vals, color=["#c62828", "#ef6c00", "#f9a825", "#2e7d32"], alpha=0.85)
    ax.set_yscale("log")
    ax.axhline(A_peak * 1e9, color="k", ls="--", lw=1)
    ax.text(3.45, A_peak * 1e9 * 1.15, "MB peak (6 nA)", fontsize=7, ha="right")
    ax.axhline(0.01 * A_peak * 1e9, color="grey", ls="-.", lw=1)
    ax.text(3.45, 0.01 * A_peak * 1e9 * 1.15, "1% of peak", fontsize=7, ha="right")
    ax.axhline(sigma_dI * 1e9, color="#1565c0", ls=":", lw=1.5)
    ax.text(3.45, sigma_dI * 1e9 * 0.5, "electronic σ_dI (4.6 pA)", fontsize=7, ha="right", va="top", color="#1565c0")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.25, f"{v:.3g} nA", ha="center", fontsize=8)
    ax.set_ylabel("Peak mains pickup at TIA input (nA)")
    ax.set_ylim(1e-3, 1e3)
    ax.set_title("A — Pickup amplitude, 230 V / 50 Hz\ni = 2π f_m C_c V_pk / 10^(SE/20)")

    ax = axes[1]
    dE = 2e-3
    E = np.arange(sa.E_START, sa.E_END + dE / 2, dE)
    peak = sa.gaussian(E, A_peak, sa.E_MB, nb.SIGMA_E)
    A_in = 1e-9
    offs = 0.0
    for f_swv, T_int, col in [(60.0, None, "#c62828"), (45.0, None, "#ef6c00"), (25.0, 1 / 50.0, "#2e7d32")]:
        t = transfer(50.0, f_swv, T_int)
        rip = ripple_trace(E, A_in, 50.0, f_swv, 0.7, dE, 0.0, T_int)
        ax.plot(E, (peak + rip) * 1e9 + offs, color=col, lw=1.2,
                label=f"f_SWV = {f_swv:.0f} Hz: |G| = {t['gain']:.2f}, beat "
                      f"{'—' if np.isinf(t['beat_mV']) else '%.0f mV' % t['beat_mV']}")
        offs += 0.0
    ax.set_xlabel("Potential E (V vs on-chip RE)")
    ax.set_ylabel("Net SWV current (nA)")
    ax.set_title("B — 1 nA of 50 Hz pickup on the MB peak\n(same phase; the 25 Hz trace is on top of the clean peak)")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    f = np.array(res["f_grid"])
    ax.semilogy(f, np.array(res["rel_err_per_nA_UK"]) * 100, color="#1565c0", lw=1.8, label="50 Hz grid (UK)")
    ax.semilogy(f, np.array(res["rel_err_per_nA_US"]) * 100, color="#c62828", lw=1.4, ls="--", label="60 Hz grid (US)")
    ax.axvline(25, color="#2e7d32", ls=":", lw=1.2)
    ax.axvline(30, color="#2e7d32", ls=":", lw=1.2)
    ax.axhline(3.0, color="k", ls="-.", lw=0.8)
    ax.text(118, 3.4, "Layer 13 budget 3%", fontsize=7, ha="right")
    ax.set_xlabel("SWV frequency f_SWV (Hz)")
    ax.set_ylabel("σ_A/A per nA of pickup (%)")
    ax.set_title("C — Area error per nA of mains pickup\n(boxcar = one half-cycle; nulls at f_m/2 = 25, 30 Hz)")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Layer 14 / 14.6: mains interference amplitude, SWV aliasing, and area error", fontsize=10)
    fig.tight_layout()
    out = nb.figures_dir() / "fig_L14_mains.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
