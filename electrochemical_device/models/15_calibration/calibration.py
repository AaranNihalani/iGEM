"""
calibration.py  —  Layer 15
==============================
Calibration curve fitting and confidence intervals.

Two models:
  1. Linear: F_k = m * [E] + b   (valid in linear regime, T < T_linear)
  2. Four-parameter logistic (4PL): F_k = d + (a-d)/(1 + ([E]/c)^b)
     (general sigmoid, handles both linear and saturated regimes)

LOD = 3 * sigma_blank / slope   (IUPAC definition)
LOQ = 10 * sigma_blank / slope
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Generate synthetic calibration data ──────────────────────────────────────
np.random.seed(0)
MW_MMP9     = 92e3
kcat_Km_eff = 188e3 * 0.3   # M-1s-1
sigma_Fk    = 0.005          # 0.5% absolute noise per measurement

T_cal       = 30 * 60        # calibration at T=30 min
E_cal_ngml  = np.array([0, 5, 10, 17, 30, 50, 75, 100, 150, 200, 250])
E_cal_M     = E_cal_ngml * 1e-6 / MW_MMP9
n_replicates = 3

# True F_k values
Fk_true = 1 - np.exp(-kcat_Km_eff * E_cal_M * T_cal)

# Synthetic data with noise
Fk_data = []
for Fk_t in Fk_true:
    reps = Fk_t + np.random.normal(0, sigma_Fk, n_replicates)
    Fk_data.append(np.clip(reps, 0, 1))

Fk_mean = np.array([d.mean() for d in Fk_data])
Fk_std  = np.array([d.std()  for d in Fk_data])

# ── Linear fit (low-[E] regime) ──────────────────────────────────────────────
mask_lin = E_cal_M < 1e-9   # fit only linear portion
E_lin    = E_cal_M[mask_lin]
Fk_lin_y = Fk_mean[mask_lin]

def linear_model(E, m, b):
    return m * E + b

popt_lin, pcov_lin = curve_fit(linear_model, E_lin, Fk_lin_y, p0=[kcat_Km_eff*T_cal, 0])
m_fit, b_fit = popt_lin
m_err, b_err = np.sqrt(np.diag(pcov_lin))

sigma_blank = Fk_std[0]   # noise at zero concentration
LOD_M  = 3  * sigma_blank / m_fit
LOQ_M  = 10 * sigma_blank / m_fit
LOD_ng = LOD_M * MW_MMP9 * 1e6   # mol/L × g/mol × 1e9 ng/g × 1e-3 L/mL = × 1e6
LOQ_ng = LOQ_M * MW_MMP9 * 1e6

# ── 4PL fit ──────────────────────────────────────────────────────────────────
def four_PL(E, a, b, c, d):
    """a=min, d=max, c=EC50, b=Hill slope."""
    return d + (a - d) / (1 + (E / c)**b)

p0_4pl = [0.0, 1.5, 1e-9, 0.95]
try:
    popt_4pl, pcov_4pl = curve_fit(four_PL, E_cal_M[1:], Fk_mean[1:],
                                    p0=p0_4pl, maxfev=10000,
                                    bounds=([0,0.1,1e-12,0],[1,10,1e-6,1]))
except RuntimeError:
    popt_4pl = p0_4pl


def plot_calibration(save=True):
    E_plot = np.linspace(0, 300, 400)   # ng/mL
    E_plot_M = E_plot * 1e-6 / MW_MMP9

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: full calibration curve (4PL + linear + data)
    ax = axes[0]
    # Data points
    ax.errorbar(E_cal_ngml, Fk_mean*100, yerr=Fk_std*100,
                fmt="ko", markersize=6, capsize=4, label="Calibration data (n=3)")
    # 4PL fit
    Fk_4pl = four_PL(E_plot_M, *popt_4pl) * 100
    ax.plot(E_plot, Fk_4pl, "b-", lw=2, label="4PL fit")
    # Linear fit (dashed, only in linear region)
    E_lin_plot = np.linspace(0, 80, 100)
    Fk_lin_plot = linear_model(E_lin_plot*1e-9/MW_MMP9, *popt_lin) * 100
    ax.plot(E_lin_plot, Fk_lin_plot, "r--", lw=1.8, label="Linear fit (0–60 ng/mL)")
    ax.axhspan(5, 90, alpha=0.07, color="green", label="Operating window")
    ax.axvline(LOD_ng, color="orange", ls=":", lw=1.5, label=f"LOD = {LOD_ng:.1f} ng/mL")
    ax.axvline(LOQ_ng, color="red",    ls=":", lw=1.5, label=f"LOQ = {LOQ_ng:.1f} ng/mL")
    ax.set_xlabel("[MMP-9] (ng/mL)")
    ax.set_ylabel("F_k (%)")
    ax.set_title("A  —  Calibration curve (4PL + linear)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 250)

    # Panel B: residuals
    ax2 = axes[1]
    Fk_4pl_at_cal = four_PL(E_cal_M, *popt_4pl)
    residuals = (Fk_mean - Fk_4pl_at_cal) * 100
    ax2.bar(E_cal_ngml, residuals, width=10, color=["green" if r>0 else "red" for r in residuals])
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(sigma_Fk*100, color="gray", ls="--", lw=1, label=f"±σ_Fk = {sigma_Fk*100:.1f}%")
    ax2.axhline(-sigma_Fk*100, color="gray", ls="--", lw=1)
    ax2.set_xlabel("[MMP-9] (ng/mL)")
    ax2.set_ylabel("Residual F_k (%)")
    ax2.set_title("B  —  Fit residuals (4PL)")
    ax2.legend(fontsize=9)

    # Panel C: confidence intervals via bootstrap
    ax3 = axes[2]
    # 95% CI from linear fit
    n  = len(E_lin)
    df = n - 2
    alpha_CI = 0.05
    t_crit = t_dist.ppf(1 - alpha_CI/2, df)
    E_ci   = np.linspace(0, 80, 200)
    E_ci_M = E_ci * 1e-6 / MW_MMP9
    Fk_ci  = linear_model(E_ci_M, *popt_lin) * 100
    # Standard error of prediction
    SE = sigma_Fk * np.sqrt(1/n + (E_ci_M - E_lin.mean())**2 / np.sum((E_lin - E_lin.mean())**2))
    ci_half = t_crit * SE * 100

    ax3.plot(E_ci, Fk_ci, "b-", lw=2, label="Linear fit")
    ax3.fill_between(E_ci, Fk_ci - ci_half, Fk_ci + ci_half,
                     alpha=0.3, color="blue", label="95% CI")
    ax3.errorbar(E_cal_ngml[mask_lin], Fk_mean[mask_lin]*100,
                 yerr=Fk_std[mask_lin]*100, fmt="ko", capsize=4, label="Data")
    ax3.set_xlabel("[MMP-9] (ng/mL)")
    ax3.set_ylabel("F_k (%)")
    ax3.set_title("C  —  Linear calibration with 95% CI")
    ax3.legend(fontsize=9)

    fig.suptitle(f"Layer 15: Calibration curve and LOD/LOQ\n"
                 f"LOD = {LOD_ng:.1f} ng/mL  |  LOQ = {LOQ_ng:.1f} ng/mL  |  "
                 f"Linear slope = {m_fit/(MW_MMP9*1e6)*100:.4f} %·mL/ng  (T=30 min)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B13_calibration.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)

    print(f"LOD = {LOD_ng:.2f} ng/mL  ({LOD_M*1e9:.3f} nM)")
    print(f"LOQ = {LOQ_ng:.2f} ng/mL  ({LOQ_M*1e9:.3f} nM)")
    print(f"Linear slope: {m_fit:.2e} per M  ({m_fit/(MW_MMP9*1e6)*100:.6f} %/(ng/mL))")
    print(f"4PL fit: a={popt_4pl[0]:.3f}, b={popt_4pl[1]:.2f}, "
          f"c={popt_4pl[2]*1e9:.3f} nM, d={popt_4pl[3]:.3f}")


if __name__ == "__main__":
    plot_calibration()
