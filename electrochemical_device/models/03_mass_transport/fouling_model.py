"""
fouling_model.py  —  Layer 3 (supplementary)
=============================================
Biological sample fouling: non-specific protein adsorption onto the SAM
grows a blocking layer that attenuates enzyme transport to the surface.

Model
-----
Fouling layer growth (Langmuir kinetics):
  dΘ_f/dt = k_f * [P]_bulk * (1 - Θ_f) - k_d * Θ_f

where Θ_f ∈ [0,1] is fractional fouling coverage, [P]_bulk is total
non-specific protein concentration in the chamber.

Fouling layer thickness: h_f(t) = h_f_max * Θ_f(t)

Effective diffusion distance to electrode surface:
  δ_eff(t) = h_channel + h_f(t)

Diffusion flux attenuation factor (first-order approximation):
  α_foul(t) = h_channel / δ_eff(t)   (enzyme flux ∝ 1/δ_eff)

Effect on F_k:
  F_k_fouled(T) = 1 - exp(-k_obs * α_foul_mean * T)

where α_foul_mean = (1/T) ∫₀ᵀ α_foul(t) dt
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Fouling parameters ───────────────────────────────────────────────────────
h_channel = 100e-6    # m  (diffusion distance in clean device)
h_f_max   = 10e-9     # m  (max fouling layer thickness, ~10 nm monolayer)
# Derived from protein monolayer height on SAM surfaces (e.g. albumin ~3–8 nm,
# fibrinogen ~4–6 nm) — CONFIDENT order of magnitude (Vroman effect literature)

# Adsorption/desorption rate constants:
# FLAG: k_f and k_d for wound-fluid proteins on MCH-SAM surface.
# Placeholder: k_f ~ 1e3 M-1s-1, k_d ~ 1e-4 s-1 (gives Kd = k_d/k_f ~ 0.1 µM)
k_f = 1e3    # M-1 s-1   ← FLAG
k_d = 1e-4   # s-1       ← FLAG

# Total non-specific protein in wound fluid chamber:
# Wound fluid total protein ~10–40 mg/mL; after elution/dilution ~5–20 mg/mL.
# Use albumin-equivalent: ~10 mg/mL = 10 g/L / 66.5 kDa = ~150 µM
P_bulk_M   = 150e-6   # mol/L  ← FLAG: highly variable; verify with wound-fluid proteomics


def fouling_ode(t, Theta):
    dTheta = k_f * P_bulk_M * (1 - Theta) - k_d * Theta
    return [dTheta]


def run_fouling(t_max_s=3600):
    sol = solve_ivp(fouling_ode, (0, t_max_s), [0.0],
                    t_eval=np.linspace(0, t_max_s, 500),
                    method="RK45", rtol=1e-8)
    return sol


def alpha_fouling(Theta_t):
    """Flux attenuation factor vs time."""
    h_f = h_f_max * Theta_t
    return h_channel / (h_channel + h_f)


def Fk_with_fouling(k_obs, T_s, sol):
    """
    F_k with fouling-attenuated enzyme flux.
    Uses mean alpha over [0, T].
    """
    idx = np.searchsorted(sol.t, T_s)
    t_sub    = sol.t[:idx+1]
    Theta_sub = sol.y[0, :idx+1]
    alpha_t  = alpha_fouling(Theta_sub)
    alpha_mean = np.trapz(alpha_t, t_sub) / (t_sub[-1] - t_sub[0] + 1e-12)
    return 1 - np.exp(-k_obs * alpha_mean * T_s)


def Fk_clean(k_obs, T_s):
    return 1 - np.exp(-k_obs * T_s)


def plot_fouling(save=True):
    sol = run_fouling(t_max_s=3600)
    t_min  = sol.t / 60
    Theta  = sol.y[0]
    alpha  = alpha_fouling(Theta)

    # Clinical [E] range for C15 substrate
    E_lo = 0.185e-9  # mol/L  (17 ng/mL)
    E_hi = 2.17e-9   # mol/L  (200 ng/mL)
    kcat_Km = 188e3  # M-1s-1 (C15)
    k_obs_lo = kcat_Km * E_lo
    k_obs_hi = kcat_Km * E_hi

    T_arr = np.linspace(60, 3600, 200)
    Fk_clean_lo  = np.array([Fk_clean(k_obs_lo, T) for T in T_arr])
    Fk_fouled_lo = np.array([Fk_with_fouling(k_obs_lo, T, sol) for T in T_arr])
    Fk_clean_hi  = np.array([Fk_clean(k_obs_hi, T) for T in T_arr])
    Fk_fouled_hi = np.array([Fk_with_fouling(k_obs_hi, T, sol) for T in T_arr])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: fouling coverage vs time
    axes[0].plot(t_min, Theta * 100, "k-", lw=2)
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Fouling coverage Θ_f (%)")
    axes[0].set_title("A  —  Fouling layer growth")
    axes[0].set_xlim(0, 60)

    # Panel B: flux attenuation factor
    axes[1].plot(t_min, alpha * 100, "m-", lw=2)
    axes[1].axhline(100, color="gray", ls="--", lw=1, label="No fouling")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Flux attenuation α_foul (%)")
    axes[1].set_title("B  —  Enzyme flux attenuation")
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylim(95, 101)

    # Panel C: F_k with and without fouling
    ax = axes[2]
    ax.plot(T_arr/60, Fk_clean_lo*100, "b-",  lw=2, label=f"Clean, 17 ng/mL")
    ax.plot(T_arr/60, Fk_fouled_lo*100,"b--", lw=2, label=f"Fouled, 17 ng/mL")
    ax.plot(T_arr/60, Fk_clean_hi*100, "r-",  lw=2, label=f"Clean, 200 ng/mL")
    ax.plot(T_arr/60, Fk_fouled_hi*100,"r--", lw=2, label=f"Fouled, 200 ng/mL")
    ax.axvline(30, color="gray", ls=":", lw=1.2, label="T=30 min")
    ax.axvline(60, color="gray", ls="-.", lw=1.2, label="T=60 min")
    ax.set_xlabel("Incubation time (min)")
    ax.set_ylabel("F_k (%)")
    ax.set_title("C  —  F_k with vs without fouling (C15)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 60)

    fig.suptitle("Layer 3: Biological sample fouling — flux attenuation model\n"
                 "(k_f, k_d, [P]_bulk flagged — see fouling_model.py)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B4_fouling.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    # Report attenuation at T=60 min
    T_report = 3600
    idx = np.argmin(np.abs(sol.t - T_report))
    print(f"At T=60 min: Θ_f = {Theta[idx]*100:.2f}%, α = {alpha[idx]*100:.3f}%")
    print(f"F_k suppression at 17 ng/mL:  "
          f"{(Fk_clean_lo[-1]-Fk_fouled_lo[-1])*100:.3f}% absolute")


if __name__ == "__main__":
    plot_fouling()
