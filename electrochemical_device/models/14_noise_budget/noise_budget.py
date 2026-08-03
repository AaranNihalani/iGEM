"""
noise_budget.py  —  Layer 14
================================
Total noise budget for the SWV measurement.

Sources:
  1. Johnson (thermal) noise: sigma_J = sqrt(4*kB*T*R_f*B)
  2. Flicker (1/f) noise in TIA: sigma_f = K_f * sqrt(ln(f_max/f_min))
  3. Shot noise (negligible at nA currents): sigma_shot = sqrt(2*q*I*B)
  4. ADC quantisation noise: sigma_q = LSB / sqrt(12)
  5. Double-layer charging noise (baseline fluctuation)

Total: sigma_total = sqrt(sigma_J² + sigma_f² + sigma_q² + sigma_dl²)

LOD = 3 * sigma_Fk / (dF_k/d[E])   where dF_k/d[E] = kcat_Km * T at [E] → 0
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
T_  = 298.15       # K

# ── Transimpedance amplifier (TIA) parameters ────────────────────────────────
R_f   = 1e9       # Ohm  (1 GΩ feedback resistor)
# FLAG: confirm TIA R_f with hardware team
B     = 1.0       # Hz  (effective noise bandwidth for SWV pulse integration)
# SWV measures current at end of each pulse (typically 5 ms) → B ~ 1/(2*tau) ~ 100 Hz
# but after digital filtering, effective bandwidth drops to ~1 Hz
# FLAG: confirm B with signal processing team

# ── Johnson noise ────────────────────────────────────────────────────────────
sigma_J = np.sqrt(4 * kB * T_ / R_f * B) * 1e12   # pA RMS (input-referred current noise of TIA)

# ── Flicker noise ────────────────────────────────────────────────────────────
# K_f for typical TIA op-amp (e.g. AD8627): ~ 5e-30 A²/Hz at 1 Hz
# FLAG: get K_f from op-amp datasheet
K_f   = 5e-30     # A² (input current noise density × f at corner frequency) ← FLAG
f_min = 0.01      # Hz  (low-frequency cutoff)
f_max = B         # Hz
sigma_f = np.sqrt(K_f * np.log(f_max / f_min)) * 1e12   # pA RMS  (approximate)

# ── Shot noise ───────────────────────────────────────────────────────────────
I_dc   = 5e-9    # A  (typical peak current)
sigma_shot = np.sqrt(2 * q * I_dc * B) * 1e12   # pA RMS  (negligible)

# ── ADC quantisation ─────────────────────────────────────────────────────────
# 16-bit ADC, input range ±10 µA
# FLAG: confirm ADC specs with hardware team
n_bits      = 16
range_A     = 20e-9    # A  (full scale range) ← FLAG
LSB_A       = range_A / (2**n_bits)
sigma_q     = LSB_A / np.sqrt(12) * 1e12   # pA RMS

# ── Total noise ──────────────────────────────────────────────────────────────
sigma_total_pA = np.sqrt(sigma_J**2 + sigma_f**2 + sigma_shot**2 + sigma_q**2)
sigma_total_nA = sigma_total_pA * 1e-3

# ── LOD calculation ──────────────────────────────────────────────────────────
i_0_nA      = 5.0     # nA  (typical peak current at T=0)
kcat_Km_eff = 188e3 * 0.3   # M-1s-1
MW_MMP9     = 92e3

# sigma_Fk from noise propagation through ratiometric readout
# sigma_Fk ≈ sqrt(2) * sigma_i / i_0  (two independent measurements, active and control)
sigma_Fk = np.sqrt(2) * sigma_total_nA / i_0_nA

# dF_k/d[E] at [E] → 0 (linear regime): = kcat_Km_eff * T
def LOD_nM(T_s):
    slope = kcat_Km_eff * T_s   # 1/M = L/mol
    LOD_M = 3 * sigma_Fk / slope
    return LOD_M * 1e9

def LOQ_nM(T_s):
    slope = kcat_Km_eff * T_s
    LOQ_M = 10 * sigma_Fk / slope
    return LOQ_M * 1e9


def plot_noise(save=True):
    T_arr = np.linspace(60, 7200, 500)
    lod   = np.array([LOD_nM(T) for T in T_arr])
    loq   = np.array([LOQ_nM(T) for T in T_arr])
    E_lo_nM = 17e-6/MW_MMP9 * 1e9   # 0.185 nM

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: noise budget breakdown
    ax = axes[0]
    labels  = ["Johnson\n(thermal)", "Flicker\n(1/f)", "Shot", "ADC\nquant."]
    values  = [sigma_J, sigma_f, sigma_shot, sigma_q]
    bar_col = ["#1565c0","#c62828","#2e7d32","#6a1b9a"]
    bars = ax.bar(labels, values, color=bar_col, alpha=0.85)
    ax.axhline(sigma_total_pA, color="k", ls="--", lw=2,
               label=f"Total σ = {sigma_total_pA:.1f} pA")
    ax.set_ylabel("Noise σ (pA RMS)")
    ax.set_title("A  —  Noise budget breakdown")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.05,
                f"{val:.2f}", ha="center", fontsize=8)

    # Panel B: LOD and LOQ vs incubation time
    ax2 = axes[1]
    ax2.semilogy(T_arr/60, lod, "b-", lw=2, label="LOD (3σ)")
    ax2.semilogy(T_arr/60, loq, "r--", lw=2, label="LOQ (10σ)")
    ax2.axhline(E_lo_nM, color="green", ls=":", lw=1.5,
                label=f"Lower clinical bound {E_lo_nM:.2f} nM")
    idx_lod = np.searchsorted(-lod, -E_lo_nM)
    if idx_lod < len(T_arr):
        T_lod = T_arr[idx_lod]
        ax2.axvline(T_lod/60, color="orange", ls=":", lw=1.5,
                    label=f"LOD < {E_lo_nM:.2f} nM at T>{T_lod/60:.0f} min")
    else:
        ax2.text(0.5, 0.5, f"LOD > {E_lo_nM:.2f} nM\nthroughout range",
                 transform=ax2.transAxes, ha="center", va="center", color="orange")
    ax2.set_xlabel("Incubation time T (min)")
    ax2.set_ylabel("[MMP-9] LOD/LOQ (nM)")
    ax2.set_title("B  —  LOD and LOQ vs incubation time")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 120)

    # Panel C: SNR vs [E] at T=30 and 60 min
    ax3 = axes[2]
    E_range = np.linspace(0.05e-9, 3e-9, 300)
    for T_min, ls, col in [(30,"-","b"),(60,"--","r")]:
        Fk_arr = 1 - np.exp(-kcat_Km_eff * E_range * T_min * 60)
        SNR    = Fk_arr / sigma_Fk
        ax3.plot(E_range*MW_MMP9*1e9, SNR, color=col, ls=ls, lw=2, label=f"T={T_min} min")
    ax3.axhline(3, color="gray", ls=":", lw=1, label="SNR=3 (LOD)")
    ax3.axhline(10, color="gray", ls="--", lw=1, label="SNR=10 (LOQ)")
    ax3.axvspan(17, 200, alpha=0.07, color="green", label="Clinical range")
    ax3.set_xlabel("[MMP-9] (ng/mL)")
    ax3.set_ylabel("SNR = F_k / σ(F_k)")
    ax3.set_title("C  —  Signal-to-noise ratio vs [MMP-9]")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 250)

    fig.suptitle("Layer 14: Noise budget and LOD/LOQ\n"
                 f"σ_total = {sigma_total_pA:.1f} pA  |  σ_Fk = {sigma_Fk*100:.3f}%  |  "
                 f"R_f = {R_f/1e9:.0f} GΩ (FLAG: verify TIA specs)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B12_noise_budget.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    print(f"\nNoise budget:")
    print(f"  Johnson:       {sigma_J:.2f} pA")
    print(f"  Flicker:       {sigma_f:.2f} pA")
    print(f"  Shot:          {sigma_shot:.3f} pA")
    print(f"  ADC quant:     {sigma_q:.3f} pA")
    print(f"  Total:         {sigma_total_pA:.2f} pA")
    print(f"  sigma_Fk:      {sigma_Fk*100:.3f}%")
    for T_min in [30, 60]:
        print(f"  LOD at T={T_min}min: {LOD_nM(T_min*60):.4f} nM  "
              f"({LOD_nM(T_min*60)*MW_MMP9*1e-3:.3f} ng/mL)")


if __name__ == "__main__":
    plot_noise()
