"""
local_ph.py  --  Layer 6.4
==============================
Local pH perturbation at the electrode surface during MB SWV cycling, and
the resulting systematic shift in the MB formal potential E_MB.

MB redox couple:  MB + 2H+ + 2e- <-> leucomethylene blue (LMB)

During the cathodic SWV half-cycle H+ is consumed at the surface (local pH
rises transiently); during the anodic half-cycle H+ is produced (local pH
falls). The proton flux is stoichiometrically tied to the Faradaic MB
current (n=2 protons per MB reduced).

Pipeline (Sec. 6.4 of Layer 6 contents.md):
  1. Proton flux -> moles of H+ turned over per half-cycle
  2. Diffusion-layer volume -> concentration perturbation delta[H+]
  3. Buffer capacity of wound fluid -> attenuated delta_pH
  4. Nernstian conversion -> shift in E_MB (59 mV/pH unit for a 2H+/2e- couple)

Feeds: Layer 10 (E_MB correction), Layer 12 (peak-fitting baseline / position).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
F_ = 96485.0          # C/mol
N_PROTONS = 2          # protons per MB <-> LMB turnover
NERNST_SLOPE_MV_PER_PH = 59.0   # mV/pH, 2H+/2e- couple at ~37 C (CONFIRMED, standard Nernstian slope)

# ---------------------------------------------------------------------------
# Worked-example device parameters (Sec. 6.4 of contents.md)  -- FLAG some
# ---------------------------------------------------------------------------
I_MB_PEAK_A   = 50e-9    # A, peak MB Faradaic current (contents.md worked example)
A_TRUE_CM2    = 0.013    # cm^2, true (roughness-corrected) electrode area
F_SWV_HZ      = 60.0     # Hz, SWV frequency
DELTA_LAYER_UM = 50.0    # um, effective diffusion-layer thickness adjacent to electrode
BUFFER_BICARB_MM = 10.0  # mM, wound-fluid bicarbonate buffer (mid-range, 10-25 mM)


def half_period_s(f_swv_hz=F_SWV_HZ):
    """SWV half-period (duration of one forward or backward pulse), s."""
    return 1.0 / (2.0 * f_swv_hz)


def delta_n_H_mol(I_MB_A, delta_t_s, n=N_PROTONS):
    """
    Moles of H+ turned over during one half-cycle (Sec. 6.4):
        delta_n_H+ = I * delta_t / (n * F)
    """
    return I_MB_A * delta_t_s / (n * F_)


def diffusion_layer_volume_L(A_true_cm2, delta_layer_um):
    """
    Volume of the diffusion layer adjacent to the electrode, litres:
        V_diff = A_true * delta
    """
    A_cm2 = A_true_cm2
    delta_cm = delta_layer_um * 1e-4      # um -> cm
    V_cm3 = A_cm2 * delta_cm              # cm^3 = mL
    return V_cm3 * 1e-3                   # mL -> L


def delta_H_concentration_M(delta_n_H_mol_, V_diff_L):
    """delta[H+] (M) = delta_n_H+ / V_diff."""
    return delta_n_H_mol_ / V_diff_L


def buffer_capacity_M_per_pH(buffer_conc_M, pH, pKa=6.1):
    """
    Standard Henderson-Hasselbalch buffer capacity (Van Slyke), mol/L per
    pH unit, for a single monoprotic buffer species of total concentration
    buffer_conc_M and pKa (bicarbonate pKa1 = 6.1 at 37 C -- CONFIRMED,
    standard value):
        beta = 2.303 * C * Ka * [H+] / (Ka + [H+])^2
    This replaces contents.md's informal "divide by 10" shortcut (Sec. 6.4)
    with the closed-form textbook result -- FLAG: contents.md's own worked
    numbers (delta_H=34 uM, buffer=10 mM -> delta_pH=0.15) do not reproduce
    from its stated formula either (see report Sec. IV / Limitations); this
    function is used in place of that formula throughout.
    """
    Ka = 10 ** (-pKa)
    H = 10 ** (-pH)
    return 2.303 * buffer_conc_M * Ka * H / (Ka + H) ** 2


def delta_pH_buffered(delta_H_M, buffer_conc_M, pH=7.4, pKa=6.1):
    """
    Buffer-capacity-attenuated pH perturbation (Sec. 6.4):
        delta_pH = delta[H+] / beta(buffer, pH)
    using the closed-form buffer capacity beta (see buffer_capacity_M_per_pH).
    """
    beta = buffer_capacity_M_per_pH(buffer_conc_M, pH, pKa)
    return delta_H_M / beta


def E_MB_shift_mV(delta_pH, slope_mV_per_pH=NERNST_SLOPE_MV_PER_PH):
    """Nernstian shift in E_MB from a local pH perturbation delta_pH."""
    return delta_pH * slope_mV_per_pH


def local_pH_shift_pipeline(I_MB_A=I_MB_PEAK_A, A_true_cm2=A_TRUE_CM2,
                             f_swv_hz=F_SWV_HZ, delta_layer_um=DELTA_LAYER_UM,
                             buffer_conc_M=BUFFER_BICARB_MM * 1e-3,
                             pH=7.4, pKa=6.1):
    """Run the full 6.4 pipeline and return a dict of intermediate results."""
    dt = half_period_s(f_swv_hz)
    dn = delta_n_H_mol(I_MB_A, dt)
    V = diffusion_layer_volume_L(A_true_cm2, delta_layer_um)
    dH = delta_H_concentration_M(dn, V)
    dpH = delta_pH_buffered(dH, buffer_conc_M, pH, pKa)
    dE = E_MB_shift_mV(dpH)
    return {
        "half_period_s": dt,
        "delta_n_H_mol": dn,
        "V_diff_L": V,
        "delta_H_M": dH,
        "delta_pH": dpH,
        "delta_E_MB_mV": dE,
    }


if __name__ == "__main__":
    print("=== Layer 6.4: Local pH perturbation (worked example) ===")
    r = local_pH_shift_pipeline()
    print(f"Half-period (f_SWV=60 Hz)             = {r['half_period_s']*1e3:.2f} ms")
    print(f"delta_n_H+ per half-cycle              = {r['delta_n_H_mol']:.2e} mol  (contents.md: ~2.2e-15 mol, OK)")
    print(f"V_diff (A_true=0.013 cm^2, d=50 um)    = {r['V_diff_L']*1e6:.3f} uL = {r['V_diff_L']:.2e} L")
    print(f"delta[H+] (dimensionally correct)      = {r['delta_H_M']*1e6:.4f} uM = {r['delta_H_M']*1e9:.1f} nM")
    print(f"delta_pH (dimensionally correct)       = {r['delta_pH']:.2e}")
    print(f"delta_E_MB (dimensionally correct)     = {r['delta_E_MB_mV']:.2e} mV")

    print("\n--- Cross-check against contents.md's literal worked example (Sec. 6.4) ---")
    # contents.md computes V_diff = 0.065 uL correctly, then substitutes
    # "6.5e-11 L" for it in the next line -- this is 0.065 nL, not 0.065 uL:
    # a 1000x (uL-vs-nL) unit slip.
    V_literal_L = 6.5e-11
    dH_literal = delta_H_concentration_M(r["delta_n_H_mol"], V_literal_L)
    print(f"delta[H+] using V=6.5e-11 L (as literally written) = {dH_literal*1e6:.1f} uM  (contents.md states: ~34 uM, matches)")
    # Separately, contents.md's own "/(2.303*[buffer]*10)" shortcut also does not
    # reproduce its own stated delta_pH=0.15 from delta_H=34 uM, buffer=10 mM:
    naive_denom = 2.303 * (BUFFER_BICARB_MM * 1e-3) * 10.0
    dpH_naive = dH_literal / naive_denom
    print(f"delta_pH via contents.md's own '/(2.303*C*10)' formula = {dpH_naive:.5f}  (contents.md states: ~0.15 -- does not reproduce)")
    print("=> Two separate arithmetic slips in contents.md Sec. 6.4: (1) a uL/L unit error")
    print("   (1000x) in the diffusion-layer volume, and (2) the stated '/(2.303*C*10)' formula")
    print("   does not itself evaluate to 0.15 from the source's own numbers. This script replaces")
    print("   both with dimensionally consistent volumes and the standard Henderson-Hasselbalch")
    print("   buffer-capacity formula (see report Sec. IV / Limitations).")

    print("\n=== Sensitivity: low-buffer chronic/infected wound (pH 6.5, low bicarbonate) ===")
    r_low = local_pH_shift_pipeline(buffer_conc_M=2e-3, pH=6.5)   # ~2 mM buffer, low-buffer wound
    print(f"delta_pH (2 mM buffer, pH 6.5)          = {r_low['delta_pH']:.2e}")
    print(f"delta_E_MB (2 mM buffer, pH 6.5)        = {r_low['delta_E_MB_mV']:.2e} mV")
    ratio = r_low['delta_pH'] / r['delta_pH']
    print(f"Ratio vs 10 mM/pH 7.4 case               = {ratio:.1f}x  (contents.md flags 3-5x larger for low-buffer wounds)")

    # ------------------------------------------------------------------
    def plot_local_ph():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Panel A: delta_pH vs buffer concentration
        ax = axes[0]
        buf_range_mM = np.linspace(1, 30, 200)
        dpH_vals = [local_pH_shift_pipeline(buffer_conc_M=b * 1e-3)["delta_pH"] for b in buf_range_mM]
        ax.plot(buf_range_mM, dpH_vals, "b-", lw=2)
        ax.axvline(10, color="g", ls="--", lw=1.2, label="Mid-range buffer (10 mM)")
        ax.axvline(2, color="r", ls=":", lw=1.2, label="Low-buffer chronic wound (~2 mM)")
        ax.set_xlabel("Bicarbonate buffer concentration (mM)")
        ax.set_ylabel("$\\Delta$pH per half-cycle")
        ax.set_title("A -- pH perturbation vs buffer capacity")
        ax.legend(fontsize=8)

        # Panel B: delta_pH and delta_E_MB vs I_MB peak current
        ax = axes[1]
        I_range_nA = np.linspace(5, 100, 200)
        dpH_I = [local_pH_shift_pipeline(I_MB_A=i * 1e-9)["delta_pH"] for i in I_range_nA]
        ax.plot(I_range_nA, dpH_I, "m-", lw=2)
        ax.axvline(50, color="g", ls="--", lw=1.2, label="Worked example (50 nA)")
        ax.set_xlabel("MB peak current $I_{MB}$ (nA)")
        ax.set_ylabel("$\\Delta$pH per half-cycle (10 mM buffer)")
        ax.set_title("B -- pH perturbation vs MB peak current")
        ax.legend(fontsize=8)

        # Panel C: E_MB shift (mV) vs buffer concentration, at two I_MB levels
        ax = axes[2]
        dE_10 = [local_pH_shift_pipeline(buffer_conc_M=b * 1e-3, I_MB_A=10e-9)["delta_E_MB_mV"]
                 for b in buf_range_mM]
        dE_50 = [local_pH_shift_pipeline(buffer_conc_M=b * 1e-3, I_MB_A=50e-9)["delta_E_MB_mV"]
                 for b in buf_range_mM]
        ax.plot(buf_range_mM, dE_10, "c-", lw=2, label="$I_{MB}$ = 10 nA")
        ax.plot(buf_range_mM, dE_50, "orange", lw=2, label="$I_{MB}$ = 50 nA")
        ax.set_xlabel("Bicarbonate buffer concentration (mM)")
        ax.set_ylabel("$\\Delta E_{MB}$ (mV)")
        ax.set_title("C -- Systematic $E^{\\circ}_{MB}$ shift vs buffer capacity")
        ax.legend(fontsize=8)

        fig.suptitle("Layer 6.4: Local pH perturbation and $E^{\\circ}_{MB}$ shift during SWV",
                     fontsize=12, y=1.03)
        plt.tight_layout()
        out = FIGURES / "fig_L6_local_ph.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"\nSaved {out}")
        plt.close(fig)

    plot_local_ph()
