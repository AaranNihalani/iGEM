"""
peak_extraction.py  —  Layer 12
==================================
Baseline subtraction and peak finding for SWV voltammograms.

Algorithm:
  1. Moving-average baseline (window = 20% of scan range)
  2. Subtract baseline from raw i_net
  3. Find peak by argmax of |i_net - baseline| in ±80 mV window around E°_MB
  4. Compute peak height AND peak area (integrated charge)

Peak area = integral of |i_net - baseline| around peak (∝ n*F*Gamma_MB)
Peak height = max |i_net - baseline| at E°_MB
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Synthetic SWV data generator ──────────────────────────────────────────────
def generate_swv(E_arr, Fk=0.3, noise_nA=0.2, k_ET=50.0,
                 Gamma_0=3e-12, E0_MB=-0.25, n=2, A=0.0177):
    """Generate synthetic SWV voltammogram with noise."""
    F_ = 96485; R_ = 8.314; T_ = 298.15; alpha = 0.5
    Gamma = Gamma_0 * (1 - Fk)
    tau   = 1/(2*15)   # 15 Hz
    kappa = k_ET * tau

    theta_E = n * F_ * (E_arr - E0_MB) / (R_ * T_)

    # Forward and backward pulses (+/- 25 mV)
    dE = 0.025
    def i_pulse(th):
        ef = np.exp( alpha    * th)
        eb = np.exp(-(1-alpha)* th)
        denom = ef + eb
        return n * F_ * A * Gamma * k_ET * (ef - eb) / denom * kappa / (1 + kappa)

    i_net = i_pulse(n*F_*(E_arr+dE-E0_MB)/(R_*T_)) - \
            i_pulse(n*F_*(E_arr-dE-E0_MB)/(R_*T_))

    # Add sloped background (capacitive + drift)
    background = 0.5e-9 * (E_arr - E_arr[0]) / (E_arr[-1] - E_arr[0])
    noise = np.random.normal(0, noise_nA * 1e-9, len(E_arr))

    return i_net + background + noise


def moving_average_baseline(i_arr, window_frac=0.15):
    """Estimate non-faradaic baseline by moving average far from peak."""
    w = max(int(len(i_arr) * window_frac), 3)
    # Pad with edge values
    padded = np.pad(i_arr, w//2, mode="edge")
    smoothed = np.convolve(padded, np.ones(w)/w, mode="valid")
    return smoothed[:len(i_arr)]


def extract_peak(E_arr, i_arr, E0_MB=-0.25, window_V=0.08):
    """
    Extract peak height and area within window_V of E0_MB.
    Returns: peak_E, peak_height_nA, peak_area_nC, baseline
    """
    baseline = moving_average_baseline(i_arr)
    i_corrected = np.abs(i_arr - baseline)

    # Search window around E°_MB
    mask = np.abs(E_arr - E0_MB) < window_V
    if not mask.any():
        return None, 0.0, 0.0, baseline

    i_window = i_corrected.copy()
    i_window[~mask] = 0

    idx_peak   = np.argmax(i_window)
    peak_E     = E_arr[idx_peak]
    peak_height = i_corrected[idx_peak] * 1e9   # nA

    # Area (charge) by trapezoidal integration within window
    peak_area   = trapezoid(i_window[mask], E_arr[mask]) * 1e9  # nA·V (proxy for nC)

    return peak_E, peak_height, peak_area, baseline


def plot_extraction(save=True):
    np.random.seed(42)
    E_scan = np.linspace(-0.5, 0.0, 500)
    Fk_vals = [0.0, 0.25, 0.50, 0.75]
    colors  = ["#1a237e", "#1565c0", "#0288d1", "#80d8ff"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    peaks_h = []
    peaks_a = []

    # Panel A: raw + baseline + corrected for F_k=0.25
    Fk_demo = 0.25
    i_raw   = generate_swv(E_scan, Fk=Fk_demo, noise_nA=0.3)
    bline   = moving_average_baseline(i_raw)
    peak_E, ph, pa, _ = extract_peak(E_scan, i_raw)

    ax = axes[0]
    ax.plot(E_scan*1e3, i_raw*1e9,   "b-",   lw=1.5, alpha=0.7, label="Raw SWV")
    ax.plot(E_scan*1e3, bline*1e9,   "r--",  lw=1.5, label="Baseline")
    ax.plot(E_scan*1e3, (i_raw-bline)*1e9, "k-", lw=2, label="Baseline-corrected")
    ax.axvline(peak_E*1e3, color="green", ls=":", lw=1.5,
               label=f"Peak at {peak_E*1e3:.0f} mV  ({ph:.2f} nA)")
    ax.set_xlabel("E (mV vs Ag/AgCl)")
    ax.set_ylabel("Current (nA)")
    ax.set_title(f"A  —  Baseline subtraction (F_k = {Fk_demo*100:.0f}%)")
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # Panel B: overlay of corrected voltammograms at all F_k
    ax2 = axes[1]
    for Fk, col in zip(Fk_vals, colors):
        i_raw = generate_swv(E_scan, Fk=Fk, noise_nA=0.2)
        bline = moving_average_baseline(i_raw)
        i_corr = (i_raw - bline) * 1e9
        ax2.plot(E_scan*1e3, i_corr, color=col, lw=2, label=f"F_k={Fk*100:.0f}%")
        _, ph, pa, _ = extract_peak(E_scan, i_raw)
        peaks_h.append(ph); peaks_a.append(pa)
    ax2.axvline(-250, color="gray", ls="--", lw=1, label="E°_MB")
    ax2.set_xlabel("E (mV vs Ag/AgCl)")
    ax2.set_ylabel("Corrected current (nA)")
    ax2.set_title("B  —  Baseline-corrected SWV at different F_k")
    ax2.legend(fontsize=8)
    ax2.invert_xaxis()

    # Panel C: peak height and area vs F_k
    ax3 = axes[2]
    ph_arr = np.array(peaks_h)
    pa_arr = np.array(peaks_a)
    ax3.plot(np.array(Fk_vals)*100, ph_arr/ph_arr[0]*100, "b-o", lw=2,
             label="Peak height (normalised)")
    ax3.plot(np.array(Fk_vals)*100, pa_arr/pa_arr[0]*100, "r-s", lw=2,
             label="Peak area (normalised)")
    ax3.set_xlabel("F_k (% cleavage)")
    ax3.set_ylabel("Signal (% of F_k=0)")
    ax3.set_title("C  —  Peak height vs area comparison")
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 110)

    fig.suptitle("Layer 12: Peak extraction — baseline subtraction and peak finding\n"
                 "Moving-average baseline | search window ±80 mV around E°_MB", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L12_peak_extraction.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_extraction()
