"""
noise_budget.py  --  Layer 14: electronics and potentiostat noise budget
=========================================================================
Computes every input-referred current-noise source in the signal chain
(electrode -> multiplexer -> TIA -> ADC), sums them in quadrature, and
propagates the result to the per-point SWV noise sigma_dI that Layer 12
consumes ("sigma_noise from Layer 14").

Sources (contents.md Sec. 14.1-14.7, with corrections flagged in the report):
  14.1  Johnson noise of R_f                  i_J   = sqrt(4 kB T / R_f)
  14.2  Op-amp input current noise            i_n   (OPA2333: 100 fA/rtHz)
        + op-amp VOLTAGE noise acting on the electrode capacitance
          (NOT in the planning spec; e_n * 2*pi*f*C_cell)
  14.3  Shot noise of the DC background       i_sh  = sqrt(2 q I_cap B)
  14.4  ADC quantisation                      LSB/sqrt(12), LSB = FS/2^ENOB
  14.5  1/f (flicker)                         i_n * sqrt(fc * ln(B/fmin))
  14.6  Mains pickup                          -> see mains_spectrum.py
  14.7  Mux settling / charge injection       settle_time(), charge_injection()

Convention
----------
All sigma_* are RMS currents referred to the working-electrode input, for ONE
SWV current sample (the value at the end of one half-cycle).  The net SWV
current is dI = I_fwd - I_rev, so its per-point noise is sqrt(2) times larger.
The noise bandwidth of a sample integrated over one half-cycle
(tau = 1/(2 f_SWV)) is B = 1/(2 tau) = f_SWV.

Run:  python noise_budget.py          (prints tables, writes figure + JSON)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
kB = 1.380649e-23        # J/K
q = 1.602176634e-19      # C
F_CONST = 96485.33       # C/mol
R_GAS = 8.314462         # J/mol/K
T_K = 298.15             # K  (instrument temperature; Layer 11 uses 310 K for the MB couple)


def figures_dir() -> Path:
    """reports/figures if running inside the repo layout, else ./figures."""
    here = Path(__file__).resolve()
    repo = here.parents[2] / "reports" / "figures" if len(here.parents) > 2 else None
    if repo is not None and repo.parent.exists():
        repo.mkdir(parents=True, exist_ok=True)
        return repo
    d = Path(os.environ.get("L14_FIGDIR", here.parent / "figures"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    d = Path(os.environ.get("L14_RESDIR", Path(__file__).resolve().parent / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Hardware and cell descriptions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Frontend:
    name: str
    R_f: float          # feedback resistor, ohm
    v_half: float       # ADC input half-range, V (full span = 2*v_half)
    enob: float         # effective number of bits of the ADC at the output rate
    e_n: float          # op-amp voltage noise density, V/rtHz
    i_n: float          # op-amp current noise density, A/rtHz
    gbw: float          # op-amp gain-bandwidth product, Hz
    note: str = ""


# DStat reference architecture (Dryden & Wheeler, PLOS ONE 2015) as specified in
# contents.md.  OPA2333 numbers are from the TI datasheet (SBOS351):
# e_n = 55 nV/rtHz @ 1 kHz, i_n = 100 fA/rtHz @ 10 Hz, GBW = 350 kHz.
DSTAT = Frontend("DStat (24-bit sigma-delta, OPA2333)", R_f=1e6, v_half=2.5,
                 enob=20, e_n=55e-9, i_n=100e-15, gbw=350e3)

# ProteaSign schematic (Eton iGEM Stakeholder Doc.): AD5940 AFE, R_TIA 0.2-512 kOhm,
# 16-bit SAR ADC.  ADC span approximated from the 1.82 V internal reference; the
# AD5940 TIA amplifier noise is NOT verified here, so OPA2333 values are reused as
# a placeholder.  PGA gain (1-9x) is ignored (pessimistic).  FLAG: confirm vs datasheet.
AD5940 = Frontend("ProteaSign AD5940 (16-bit SAR, no PGA)", R_f=512e3, v_half=0.91,
                  enob=16, e_n=55e-9, i_n=100e-15, gbw=350e3,
                  note="amplifier noise and ADC span are placeholders")


@dataclass(frozen=True)
class Cell:
    """Electrode + solution parameters, all taken from upstream layers."""
    A_true: float = 0.013      # cm^2   (1 mm^2 geometric x Rf~1.3; Layers 4, 6)
    C_dl: float = 2.0e-6       # F/cm^2 (Layer 6 mid-range SAM value)
    R_s: float = 7.1e3         # ohm    (Layer 5: 5 mm x 0.375 mm channel)
    E_sw: float = 0.025        # V      (Layer 11)
    f_swv: float = 60.0        # Hz     (Layers 5, 11)
    dE_step: float = 2.0e-3    # V      (Layer 12)
    I_cap: float = 70e-9       # A      (Layers 5/6: 60-78 nA; contents.md 70 nA)

    @property
    def C(self) -> float:
        return self.C_dl * self.A_true

    @property
    def B_sample(self) -> float:
        """ENBW of a boxcar over one half-cycle."""
        return self.f_swv


# --------------------------------------------------------------------------
# Elementary noise formulas
# --------------------------------------------------------------------------
def johnson_density(R_f: float, T: float = T_K) -> float:
    """A/rtHz."""
    return np.sqrt(4.0 * kB * T / R_f)


def shot_density(I_dc: float) -> float:
    """A/rtHz (Schottky).  Upper bound when applied to capacitive charging current."""
    return np.sqrt(2.0 * q * I_dc)


def adc_lsb_current(fe: Frontend) -> float:
    """Input-referred current step of the ADC, A."""
    return 2.0 * fe.v_half / (fe.R_f * 2.0 ** fe.enob)


def en_cap_rms(e_n: float, C: float, B: float) -> float:
    """RMS current from op-amp voltage noise across the electrode capacitance.

    i(f) = e_n * 2*pi*f*C (valid for f << 1/(2 pi R_s C)); integrate 0..B:
    i_rms = e_n * 2 pi C * sqrt(B^3 / 3).
    """
    return e_n * 2.0 * np.pi * C * np.sqrt(B ** 3 / 3.0)


def flicker_rms(i_n: float, B: float, f_c: float = 0.1, f_min: float = 0.1) -> float:
    """1/f current noise below the corner f_c, integrated f_min..B (contents.md 14.5)."""
    return i_n * np.sqrt(f_c * np.log(B / f_min))


def quad(*xs) -> float:
    return float(np.sqrt(np.sum(np.square(xs))))


# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------
def budget(fe: Frontend = DSTAT, cell: Cell = Cell(), B: float | None = None,
           include_en_cap: bool = True) -> dict:
    """Return RMS noise per single current sample and per net-SWV point (A)."""
    B = cell.B_sample if B is None else B
    lsb = adc_lsb_current(fe)
    terms = {
        "Johnson (R_f)": johnson_density(fe.R_f) * np.sqrt(B),
        "Op-amp i_n": fe.i_n * np.sqrt(B),
        "Op-amp e_n x C_cell": en_cap_rms(fe.e_n, cell.C, B) if include_en_cap else 0.0,
        "Shot (I_cap)": shot_density(cell.I_cap) * np.sqrt(B),
        "ADC quantisation": lsb / np.sqrt(12.0),
        "1/f flicker": flicker_rms(fe.i_n, B),
    }
    sample = quad(*terms.values())
    return {
        "frontend": fe.name,
        "R_f": fe.R_f,
        "B_Hz": B,
        "terms_A": terms,
        "sigma_sample_A": sample,
        "sigma_dI_A": np.sqrt(2.0) * sample,      # I_fwd - I_rev, independent samples
        "adc_lsb_A": lsb,
        "dominant": max(terms, key=terms.get),
    }


def tia_bandwidth(fe: Frontend, cell: Cell) -> dict:
    """Stability-limited feedback capacitor and resulting TIA bandwidth.

    With source capacitance C_in the phase margin requires
    C_f >~ sqrt(C_in / (2 pi R_f GBW)); the signal bandwidth is then 1/(2 pi R_f C_f).
    (contents.md's C_f = 1-10 pF ignores C_in ~ 26 nF.)
    """
    C_f = np.sqrt(cell.C / (2 * np.pi * fe.R_f * fe.gbw))
    return {"C_f_F": C_f, "BW_Hz": 1.0 / (2 * np.pi * fe.R_f * C_f),
            "BW_naive_Hz": 1.0 / (2 * np.pi * fe.R_f * 10e-12)}


def r_f_max(fe: Frontend, I_max: float) -> float:
    """Largest R_f that keeps I_max*R_f inside the ADC range."""
    return fe.v_half / I_max


# --------------------------------------------------------------------------
# 14.7  Multiplexer
# --------------------------------------------------------------------------
def settle_time(cell: Cell = Cell(), R_mux: float = 2.5, dV: float = 0.3,
                I_target: float = 1e-12) -> dict:
    """Electronic settling after the mux connects WE i.

    tau = (R_mux + R_s) * C_cell.  A step dV (idle electrodes float, so dV can be
    a few 100 mV) gives an initial current dV/(R_mux+R_s); wait until it has decayed
    below I_target.
    """
    tau = (R_mux + cell.R_s) * cell.C
    I0 = dV / (R_mux + cell.R_s)
    n_tau = np.log(I0 / I_target)
    return {"tau_s": tau, "I0_A": I0, "n_tau": n_tau, "t_wait_s": n_tau * tau,
            "t_10tau_s": 10 * tau, "half_cycle_s": 1.0 / (2 * cell.f_swv)}


def charge_injection(cell: Cell = Cell(), Q_inj: float = 1.5e-12, R_mux: float = 2.5) -> dict:
    """Voltage step on the double layer from switch charge injection."""
    dV = Q_inj / cell.C
    tau = (R_mux + cell.R_s) * cell.C
    return {"Q_inj_C": Q_inj, "dV_V": dV, "dV_over_dEstep": dV / cell.dE_step,
            "t_to_1pA_s": tau * np.log((dV / (R_mux + cell.R_s)) / 1e-12) if dV > 0 else 0.0}


# --------------------------------------------------------------------------
# Signal model shared with the other scripts (Layers 11-13)
# --------------------------------------------------------------------------
SIGMA_E = 0.020          # V   Gaussian sigma of the MB peak (Layer 12 Eq. I.6)
A_MB_L13 = 0.3e-9        # A.V MB peak area used in Layer 13
A_FC_REF = 1.0e-9        # A.V Fc peak area (Layer 12 range 0.1-10 nA.V; mid value, FLAG)
SIGMA_E_FC = 0.040       # V   Fc peak sigma (Layer 12)
GAMMA0 = 2.0e-12         # mol/cm^2 (Layer 15 / Pheeney & Barton midpoint)


def area_reversible(Gamma: float, cell: Cell = Cell(), n_e: int = 2, f_kin: float = 1.0) -> float:
    """SWV peak area (A.V) in the reversible surface-confined limit.

    Integrating Layer 11 Eq. V.5 over the sech^2 peak (int = 4RT/(nF) * dI_p) gives
        A_area = 2 n F A_true Gamma E_sw f_SWV  * f_kin
    """
    return 2.0 * n_e * F_CONST * cell.A_true * Gamma * cell.E_sw * cell.f_swv * f_kin


def f_kin_from_layer13(cell: Cell = Cell()) -> float:
    """Kinetic factor that makes Layer 13's A_MB = 0.3 nA.V consistent with Gamma_0."""
    return A_MB_L13 / area_reversible(GAMMA0, cell)


# --------------------------------------------------------------------------
# Layer 13 chain (used to show what the ELECTRONICS floor implies)
# --------------------------------------------------------------------------
def sigma_F_chain(s_mb: float, s_fc: float, s_mb_ctrl: float | None = None,
                  s_mb0: float | None = None, s_mb0_ctrl: float | None = None,
                  F_k: float = 0.0, n0: int = 3, N_active: int = 1) -> dict:
    """Propagate relative area uncertainties through Layer 13 Eqs. I.4, II.4, III.5.

    s_mb   : sigma_A/A of the MB peak on the ACTIVE electrode at time T (depends on F_k)
    s_fc   : sigma_A/A of the Fc peak
    s_mb0  : same at T=0 (full coverage) -- defaults to s_mb evaluated by caller
    Control electrode is assumed to keep full coverage (rho_c = 1).
    """
    s_mb_ctrl = s_mb0 if s_mb_ctrl is None else s_mb_ctrl
    s_mb0 = s_mb if s_mb0 is None else s_mb0
    s_mb0_ctrl = s_mb0 if s_mb0_ctrl is None else s_mb0_ctrl
    sR_a_T = np.hypot(s_mb, s_fc)
    sR_a_0 = np.hypot(s_mb0, s_fc) / np.sqrt(n0)
    sR_c_T = np.hypot(s_mb_ctrl, s_fc)
    sR_c_0 = np.hypot(s_mb0_ctrl, s_fc) / np.sqrt(n0)
    s_rho_a = np.hypot(sR_a_T, sR_a_0)
    s_rho_c = np.hypot(sR_c_T, sR_c_0)
    sigma_F = (1.0 - F_k) * np.hypot(s_rho_a, s_rho_c) / np.sqrt(N_active)
    return {"s_rho_active": s_rho_a, "s_rho_ctrl": s_rho_c, "sigma_F": sigma_F}


# --------------------------------------------------------------------------
# Reporting and figure
# --------------------------------------------------------------------------
PA = 1e12


def print_report(fe: Frontend = DSTAT, cell: Cell = Cell()) -> dict:
    b = budget(fe, cell)
    bw = tia_bandwidth(fe, cell)
    st = settle_time(cell)
    print("=" * 72)
    print(f"Layer 14 noise budget -- {fe.name}")
    print("=" * 72)
    print(f"R_f = {fe.R_f:.3g} ohm | f_SWV = {cell.f_swv:g} Hz | B = {b['B_Hz']:g} Hz | "
          f"C_cell = {cell.C*1e9:.1f} nF | I_cap = {cell.I_cap*1e9:.0f} nA")
    print(f"{'source':<26}{'density (pA/rtHz)':>20}{'RMS per sample (pA)':>22}")
    dens = {
        "Johnson (R_f)": johnson_density(fe.R_f),
        "Op-amp i_n": fe.i_n,
        "Op-amp e_n x C_cell": fe.e_n * 2 * np.pi * cell.f_swv * cell.C,
        "Shot (I_cap)": shot_density(cell.I_cap),
        "ADC quantisation": np.nan,
        "1/f flicker": np.nan,
    }
    for k, v in b["terms_A"].items():
        d = dens[k] * PA
        print(f"{k:<26}{('%.3f' % d) if d == d else '   --':>20}{v*PA:>22.3f}")
    print(f"{'TOTAL per sample':<26}{'':>20}{b['sigma_sample_A']*PA:>22.3f}")
    print(f"{'TOTAL per net SWV point':<26}{'':>20}{b['sigma_dI_A']*PA:>22.3f}   (x sqrt2, dI = I_fwd - I_rev)")
    print(f"dominant term: {b['dominant']}")
    print(f"ADC LSB (input-referred): {b['adc_lsb_A']*PA:.2f} pA")
    print(f"TIA: C_f for stability = {bw['C_f_F']*1e12:.0f} pF -> BW = {bw['BW_Hz']:.0f} Hz "
          f"(spec's 10 pF would give {bw['BW_naive_Hz']/1e3:.1f} kHz but is unstable with C_cell)")
    print(f"Mux settling: tau = {st['tau_s']*1e6:.0f} us, 10 tau = {st['t_10tau_s']*1e3:.2f} ms, "
          f"decay below 1 pA from a 0.3 V step = {st['t_wait_s']*1e3:.2f} ms "
          f"(half-cycle = {st['half_cycle_s']*1e3:.2f} ms)")
    return b


def make_figure(cell: Cell = Cell(), out: Path | None = None) -> Path:
    fe = DSTAT
    Rf = np.logspace(4, 9, 200)
    curves = {k: [] for k in ["Johnson (R_f)", "Op-amp i_n", "Op-amp e_n x C_cell",
                              "Shot (I_cap)", "ADC quantisation", "total"]}
    for r in Rf:
        b = budget(replace(fe, R_f=r), cell)
        for k in curves:
            curves[k].append(b["sigma_sample_A"] if k == "total" else b["terms_A"][k])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    cols = {"Johnson (R_f)": "#1565c0", "Op-amp i_n": "#ef6c00", "Op-amp e_n x C_cell": "#c62828",
            "Shot (I_cap)": "#2e7d32", "ADC quantisation": "#6a1b9a", "total": "k"}
    for k, v in curves.items():
        ax.loglog(Rf, np.array(v) * PA, color=cols[k], lw=2.4 if k == "total" else 1.5,
                  ls="-" if k == "total" else "--", label=k if k != "total" else "Total (quadrature)")
    ax.axvspan(1e5, 1e7, color="grey", alpha=0.12, label="DStat selectable range used")
    ax.axvline(1e6, color="k", ls=":", lw=1)
    ax.axvline(r_f_max(fe, 0.27e-6), color="red", ls=":", lw=1.2, label="R_f,max (I_max = 0.27 µA)")
    ax.set_xlabel("Feedback resistor R_f (Ω)")
    ax.set_ylabel("RMS current noise per sample (pA)")
    ax.set_title("A — Noise vs R_f (B = 60 Hz, 20 ENOB)")
    ax.set_ylim(0.05, 300)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    b = budget(DSTAT, cell)
    names = list(b["terms_A"].keys()) + ["Total / sample", "Total / net point"]
    vals = [v * PA for v in b["terms_A"].values()] + [b["sigma_sample_A"] * PA, b["sigma_dI_A"] * PA]
    colors = [cols.get(n, "#9e9e9e") for n in list(b["terms_A"].keys())] + ["k", "#455a64"]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, alpha=0.85)
    for yi, v in zip(y, vals):
        ax.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("RMS current noise (pA)")
    ax.set_title("B — Budget at R_f = 1 MΩ, f_SWV = 60 Hz\n(dominant: %s)" % b["dominant"])

    ax = axes[2]
    analog = np.array([np.sqrt(sum(budget(replace(fe, R_f=r), cell)["terms_A"][k] ** 2 for k in
                                   ["Johnson (R_f)", "Op-amp i_n", "Op-amp e_n x C_cell", "Shot (I_cap)"]))
                       for r in Rf])
    ax.loglog(Rf, analog * PA, "k-", lw=2.2, label="Analogue noise (RMS/sample)")
    for bits, c in [(16, "#c62828"), (20, "#ef6c00"), (24, "#2e7d32")]:
        lsb = 2 * fe.v_half / (Rf * 2.0 ** bits)
        ax.loglog(Rf, lsb * PA, "--", color=c, label=f"1 LSB, {bits}-bit (±2.5 V)")
    ax.axvline(1e6, color="k", ls=":", lw=1)
    ax.set_xlabel("Feedback resistor R_f (Ω)")
    ax.set_ylabel("Input-referred current (pA)")
    ax.set_title("C — ADC step vs analogue noise\n(16-bit is step-limited at every R_f)")
    ax.set_ylim(1e-3, 1e4)
    ax.legend(fontsize=7)

    fig.suptitle("Layer 14: TIA / ADC / shot / Johnson noise budget for the SWV current sample", fontsize=10)
    fig.tight_layout()
    out = out or figures_dir() / "fig_L14_noise_budget.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)
    return out


def main() -> dict:
    cell = Cell()
    res = {}
    for fe in (DSTAT, AD5940):
        b = print_report(fe, cell)
        res[fe.name] = {
            "R_f": fe.R_f,
            "terms_pA": {k: v * PA for k, v in b["terms_A"].items()},
            "sigma_sample_pA": b["sigma_sample_A"] * PA,
            "sigma_dI_pA": b["sigma_dI_A"] * PA,
            "adc_lsb_pA": b["adc_lsb_A"] * PA,
            "dominant": b["dominant"],
        }
        print()
    # R_f trade-off table
    print("R_f trade-off (DStat, per net SWV point)")
    rows = []
    for rf in (1e5, 1e6, 1e7):
        b = budget(replace(DSTAT, R_f=rf), cell)
        bw = tia_bandwidth(replace(DSTAT, R_f=rf), cell)
        rows.append({"R_f": rf, "sigma_dI_pA": b["sigma_dI_A"] * PA, "dominant": b["dominant"],
                     "range_uA": DSTAT.v_half / rf * 1e6, "BW_Hz": bw["BW_Hz"],
                     "C_f_pF": bw["C_f_F"] * 1e12})
        print(f"  R_f = {rf:8.0e}: sigma_dI = {b['sigma_dI_A']*PA:5.2f} pA | range = ±{DSTAT.v_half/rf*1e6:6.2f} µA | "
              f"C_f = {bw['C_f_F']*1e12:5.0f} pF | BW = {bw['BW_Hz']:6.0f} Hz | dominant: {b['dominant']}")
    res["rf_tradeoff"] = rows
    # sensitivity to what the planning spec ignored
    b0 = budget(DSTAT, cell, include_en_cap=False)
    res["without_en_cap_sigma_dI_pA"] = b0["sigma_dI_A"] * PA
    print(f"\nWithout the e_n x C_cell term (planning-spec model): sigma_dI = {b0['sigma_dI_A']*PA:.2f} pA")
    res["mux"] = settle_time(cell)
    res["mux_inj_TMUX1104"] = charge_injection(cell, 1.5e-12)
    res["mux_inj_100pC"] = charge_injection(cell, 100e-12)
    res["tia"] = tia_bandwidth(DSTAT, cell)
    print("Charge injection: TMUX1104 1.5 pC -> dV = %.1f uV (%.1f%% of a 2 mV step); "
          "100 pC -> %.2f mV (%.0f%% of a step)" % (
              res["mux_inj_TMUX1104"]["dV_V"] * 1e6, res["mux_inj_TMUX1104"]["dV_over_dEstep"] * 100,
              res["mux_inj_100pC"]["dV_V"] * 1e3, res["mux_inj_100pC"]["dV_over_dEstep"] * 100))
    p = make_figure(cell)
    print("Saved", p)
    (results_dir() / "noise_budget.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    main()
