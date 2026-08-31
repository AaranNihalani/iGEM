"""
noise_budget.py  —  Layer 14
================================
Noise budget for the SWV measurement.

This module has two roles:
  1. Compute per-point electronic noise (TIA + ADC) and verify that
     it is NOT the limiting factor (SNR_peak >> 1 at the MB peak current).
  2. Propagate the correct sigma_Fk from the Layer 13 ratiometric chain
     to compute the true LOD and LOQ.

The key insight correcting an earlier version: going directly from
per-point current noise to sigma_Fk bypasses peak fitting and baseline
subtraction, which dominate the error budget. The correct chain is:

  sigma_noise (Layer 14, electronics)
      → sigma_AMB (Layer 12, Gaussian fit over N_window points)
      → sigma_R   (Layer 13, Eq. sigR)
      → sigma_rho (Layer 13, Eq. sigrho)
      → sigma_Fk  (Layer 13, Eq. sigFk)
      → LOD       (Layer 14, inversion through kinetic model)

The sigma_Fk from Layer 13 is 6.8 % per electrode pair (single scan);
with N_active averaged active electrodes it reduces to
6.8% / sqrt(N_active).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Constants ────────────────────────────────────────────────────────────────
kB  = 1.381e-23    # J/K
q   = 1.602e-19    # C
T_K = 298.15       # K

# ── TIA parameters ───────────────────────────────────────────────────────────
R_f   = 1e9       # Ω  (1 GΩ feedback resistor)  FLAG: verify with hardware
B     = 1.0       # Hz (effective noise bandwidth after digital filter)
#   SWV pulse width ~ 5 ms; after Savitzky-Golay (Layer 12, m=5 window)
#   effective bandwidth ~ 1 Hz. Varies with filter design.  FLAG: verify.

# ── Electronic noise sources ─────────────────────────────────────────────────
sigma_J    = np.sqrt(4 * kB * T_K * B / R_f) * 1e12          # pA RMS (Johnson)
K_f        = 5e-30                                             # A² (flicker, AD8627 typ)
f_min, f_max = 0.01, B
sigma_f    = np.sqrt(K_f * np.log(f_max / f_min)) * 1e12      # pA RMS (flicker)
I_dc       = 0.27e-9                                           # A  (MB peak current)
sigma_shot = np.sqrt(2 * q * I_dc * B) * 1e12                 # pA RMS (shot)
n_bits     = 16
range_A    = 20e-9                                             # A  FLAG: confirm ADC range
LSB_A      = range_A / (2**n_bits)
sigma_q    = LSB_A / np.sqrt(12) * 1e12                        # pA RMS (ADC quant)

sigma_total_pA = np.sqrt(sigma_J**2 + sigma_f**2 + sigma_shot**2 + sigma_q**2)
sigma_total_nA = sigma_total_pA * 1e-3

# ── Peak signal ───────────────────────────────────────────────────────────────
# From Layer 11: reversible SWV peak current ~ 0.27 nA at full probe coverage
A_peak_nA = 0.27      # nA  (MB peak amplitude at Gamma_0; decreases with cleavage)

# ── Electronic SNR at the MB peak ────────────────────────────────────────────
# Over N_window points in the peak window (120 mV / 2 mV/step = 60 pts),
# the effective per-point SNR is:
N_window   = 60
SNR_peak   = A_peak_nA / sigma_total_nA   # single-point SNR

# Gaussian fit uncertainty (Layer 12, Eq. 12.11):
# sigma_A/A ~ sigma_noise / A_peak / sqrt(N_window / 3)  (approximate)
# This is an order-of-magnitude estimate; the exact value depends on the
# covariance matrix from the NLS fit (Layer 12).
sigma_AMB_electronic_frac = (sigma_total_nA / A_peak_nA) / np.sqrt(N_window / 3.0)

# ── Ratiometric sigma_Fk from Layer 13 ───────────────────────────────────────
# Layer 13, Eq. sigR: sigma_R/R = sqrt((sigma_MB/MB)^2 + (sigma_Fc/Fc)^2)
# Dominated by peak fitting + baseline, not electronic noise.
# Layer 13 assumes sigma_AMB/AMB ~ 3 %; this is the target from hardware design.
sigma_AMB_frac = 0.03          # 3 % relative area uncertainty per scan (Layer 13 target)
sigma_AFc_frac = 0.03          # same for Fc
sigma_R_frac   = np.sqrt(sigma_AMB_frac**2 + sigma_AFc_frac**2)   # 4.24 %

# Baseline R(0) averaged over n0 = 3 scans:
n0 = 3
sigma_R0_frac  = sigma_R_frac / np.sqrt(n0)    # 2.45 %

# sigma_rho / rho = sqrt((sigma_R(T)/R(T))^2 + (sigma_R(0)/R(0))^2)
sigma_rho_frac = np.sqrt(sigma_R_frac**2 + sigma_R0_frac**2)   # 4.83 %

# sigma_Fk per electrode pair (Layer 13, Eq. sigFk), evaluated at Fk ~ 0:
sigma_Fk_per_pair = np.sqrt(2) * sigma_rho_frac                 # 6.83 %

# Averaged over N_active active electrodes:
N_active   = 4
sigma_Fk   = sigma_Fk_per_pair / np.sqrt(N_active)              # 3.41 %

# ── Kinetic parameters ────────────────────────────────────────────────────────
# kcat / K_m from Kridel 2001, Table II (confirmed), in papers/layer_09/:
keff_sol  = 67_500.0      # M-1 s-1  (solution upper bound; CONFIRMED)
# Surface correction factor (Layer 8 steric): unconfirmed, range 0.01 - 1.0
# Central estimate 0.20 (5x reduction for immobilised peptide)
surface_factors = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.00])
keff_surf_values = keff_sol * surface_factors  # M-1 s-1

# ── Clinical bounds from Layer 0 ──────────────────────────────────────────────
E_healing_nM  = 6.0    # nM  (free MMP-9, healing wound, Layer 0 ODE)
E_chronic_nM  = 30.0   # nM  (free MMP-9, chronic wound, Layer 0 ODE)
MW_MMP9       = 92_000 # g/mol

# ── LOD / LOQ calculation ─────────────────────────────────────────────────────
# In the linear regime (Fk << 1): Fk = keff_surf * [E] * T
# => d(Fk)/d[E] = keff_surf * T
# LOD (3-sigma criterion): [E]_LOD = 3 * sigma_Fk / (keff_surf * T)

def LOD_nM(T_s, keff):
    """LOD in nM given incubation time T_s (s) and keff_surf (M-1s-1)."""
    slope_per_M = keff * T_s
    return 3.0 * sigma_Fk / slope_per_M * 1e9

def LOQ_nM(T_s, keff):
    slope_per_M = keff * T_s
    return 10.0 * sigma_Fk / slope_per_M * 1e9


def plot_noise(save=True):
    T_arr = np.linspace(60, 7200, 500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel A: Electronic noise breakdown ───────────────────────────────────
    ax = axes[0]
    labels = ["Johnson\n(thermal)", "Flicker\n(1/f)", "Shot", "ADC\nquant.", "Total"]
    vals   = [sigma_J, sigma_f, sigma_shot, sigma_q, sigma_total_pA]
    cols   = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "k"]
    bars = ax.bar(labels[:-1], vals[:-1], color=cols[:-1], alpha=0.85)
    ax.axhline(sigma_total_pA, color="k", ls="--", lw=2,
               label=f"Total σ = {sigma_total_pA:.2f} pA")
    for bar, val in zip(bars, vals[:-1]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Noise σ (pA RMS)")
    ax.set_title("A — Electronic noise budget\n"
                 f"SNR at MB peak = {SNR_peak:.0f}  (not the limiting factor)")
    ax.legend(fontsize=9)

    # ── Panel B: LOD vs incubation time for surface correction range ──────────
    ax2 = axes[1]
    keff_central = keff_sol * 0.20   # central estimate (5x surface reduction)
    lod_central  = np.array([LOD_nM(T, keff_central) for T in T_arr])
    lod_best     = np.array([LOD_nM(T, keff_sol * 0.50) for T in T_arr])
    lod_worst    = np.array([LOD_nM(T, keff_sol * 0.01) for T in T_arr])

    ax2.fill_between(T_arr/60, lod_worst, lod_best, alpha=0.15, color="blue",
                     label="keff range (1%–50% of solution UB)")
    ax2.semilogy(T_arr/60, lod_central, "b-", lw=2,
                 label=f"Central (20%, keff={keff_central:.0f} M⁻¹s⁻¹)")
    ax2.axhline(E_healing_nM, color="green", ls="--", lw=1.5,
                label=f"Healing wound: {E_healing_nM:.0f} nM")
    ax2.axhline(E_chronic_nM, color="red", ls="--", lw=1.5,
                label=f"Chronic wound: {E_chronic_nM:.0f} nM")

    # Find T where LOD crosses healing bound for central estimate
    idx = np.searchsorted(-lod_central, -E_healing_nM)
    if idx < len(T_arr):
        ax2.axvline(T_arr[idx]/60, color="purple", ls=":", lw=1.5,
                    label=f"LOD < {E_healing_nM} nM at T > {T_arr[idx]/60:.0f} min")

    ax2.set_xlabel("Incubation time T (min)")
    ax2.set_ylabel("[MMP-9] LOD (nM)")
    ax2.set_title(f"B — LOD vs incubation time\n"
                  f"σ(Fk) = {sigma_Fk*100:.1f}% (Layer 13 ratiometric chain, N={N_active})")
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0.01, 1000)

    # ── Panel C: Error budget waterfall ───────────────────────────────────────
    ax3 = axes[2]
    stages = ["Electronic\nnoise", "Peak fit\n(σ_A/A)", "Ratio\n(σ_R/R)",
              "Normalise\n(σ_ρ/ρ)", "Double\nratio σ(Fk)\n(1 pair)",
              f"Avg {N_active}\nelectrodes"]
    percents = [
        sigma_AMB_electronic_frac * 100,
        sigma_AMB_frac * 100,
        sigma_R_frac * 100,
        sigma_rho_frac * 100,
        sigma_Fk_per_pair * 100,
        sigma_Fk * 100,
    ]
    cols_wf = ["#1565c0", "#c62828", "#f57f17", "#2e7d32", "#6a1b9a", "#00838f"]
    bars3 = ax3.bar(stages, percents, color=cols_wf, alpha=0.85)
    for bar, val in zip(bars3, percents):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"{val:.2f}%", ha="center", fontsize=8)
    ax3.set_ylabel("Uncertainty (% of signal)")
    ax3.set_title("C — Error propagation waterfall\n(Layer 14 → 12 → 13)")
    ax3.set_ylim(0, max(percents) * 1.3)

    fig.suptitle(
        f"Layer 14: Noise budget  |  σ_noise = {sigma_total_pA:.2f} pA  |  "
        f"SNR_peak = {SNR_peak:.0f}  |  σ(Fk) = {sigma_Fk*100:.1f}% (N={N_active} electrodes)\n"
        f"Limiting factor: peak fitting / baseline (NOT electronics)  |  "
        f"keff_surf (surface, unconfirmed): FLAG — measure at Layer 15 calibration",
        fontsize=9
    )
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B12_noise_budget.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nElectronic noise budget:")
    print(f"  Johnson (thermal): {sigma_J:.3f} pA")
    print(f"  Flicker (1/f):     {sigma_f:.3f} pA")
    print(f"  Shot:              {sigma_shot:.3f} pA")
    print(f"  ADC quantisation:  {sigma_q:.3f} pA")
    print(f"  Total:             {sigma_total_pA:.3f} pA")
    print(f"  SNR at MB peak:    {SNR_peak:.0f}  (electronics NOT the limiting factor)")
    print(f"  sigma_AMB from electronics alone: {sigma_AMB_electronic_frac*100:.3f}% (negligible)")
    print()
    print("Ratiometric error chain (Layer 13):")
    print(f"  sigma_AMB/AMB (target, Layer 12):  {sigma_AMB_frac*100:.1f}%")
    print(f"  sigma_R/R:                          {sigma_R_frac*100:.2f}%")
    print(f"  sigma_rho/rho:                      {sigma_rho_frac*100:.2f}%")
    print(f"  sigma_Fk per electrode pair:        {sigma_Fk_per_pair*100:.2f}%")
    print(f"  sigma_Fk (N={N_active} electrodes): {sigma_Fk*100:.2f}%")
    print()
    print("LOD/LOQ at central keff_surf estimate (20% of solution UB = "
          f"{keff_central:.0f} M-1s-1):")
    for T_min in [10, 20, 30, 60]:
        T_s = T_min * 60
        lod = LOD_nM(T_s, keff_central)
        loq = LOQ_nM(T_s, keff_central)
        flag = " *** below healing bound" if lod < E_healing_nM else ""
        print(f"  T={T_min:3d} min:  LOD = {lod:.2f} nM,  LOQ = {loq:.2f} nM{flag}")
    print()
    print(f"Clinical target: LOD < {E_healing_nM} nM (healing), LOQ < {E_chronic_nM} nM (chronic)")
    print("FLAG: keff_surf must be measured at Layer 15 calibration before LOD can be confirmed.")


if __name__ == "__main__":
    plot_noise()
