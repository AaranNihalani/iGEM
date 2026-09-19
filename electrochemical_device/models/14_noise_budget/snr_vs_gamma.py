"""
snr_vs_gamma.py  --  Layer 14: how much probe do we need for the ELECTRONICS to be irrelevant?
==============================================================================================
SNR of the MB peak area as a function of the intact-probe coverage Gamma_intact,
and the Gamma thresholds at which electronic noise would start to matter.

Signal model (Layer 11 Eq. V.5 integrated over the sech^2 peak):
    A_MB(Gamma) = 2 n_e F A_true Gamma E_sw f_SWV * f_kin          [A.V]
where f_kin (<= 1) is the Layer 11 kinetic factor.  It is not known until k0_ET is
measured (Layer 10/11 limitation 1), so it is INFERRED here by requiring
A_MB(Gamma_0 = 2 pmol/cm^2) = 0.3 nA.V, the value Layer 13 uses.

Noise model: sigma_area from sigma_area.py (independent of Gamma, see its docstring),
so SNR = A_MB / sigma_area is linear in Gamma.

Thresholds reported
  Gamma_det   SNR = 3      (electronic detectability; area analogue of the Layer 12
                            "signal below LOD" flag)
  Gamma_3%    sigma_A/A = 3 %   (electronics alone would use up Layer 13's whole budget)
  Gamma_1%    sigma_A/A = 1 %
  Gamma0_min  smallest starting coverage for which the ELECTRONIC part of sigma_Fk(F=0)
              is <= 1.7 % (the Layer 14 design requirement for F_LOD = 5 %)
  F_max,elec  cleavage fraction at which the remaining probe hits SNR = 3

Run:  python snr_vs_gamma.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import noise_budget as nb
import sigma_area as sa


# --------------------------------------------------------------------------
def build(n_trials: int = 3000) -> dict:
    cell = nb.Cell()
    base = sa.compute(cell=cell, n_trials=n_trials)
    f_kin = nb.f_kin_from_layer13(cell)

    # Noise on the area, three hardware/estimator cases (all linear in sigma_dI)
    sig_pipeline = base["sigma_area_mc_fit_AV"]          # full Layer 12 pipeline, DStat
    sig_fixed = base["sigma_area_fixed_AV"]              # fixed-shape optimum, DStat
    ad = nb.budget(nb.AD5940, cell)
    sig_ad5940 = sig_pipeline * ad["sigma_dI_A"] / (base["sigma_dI_pA"] * 1e-12)
    pipeline_factor = sig_pipeline / base["sigma_area_crb_AV"]
    sig_fc = base["Fc"]["sigma_area_crb_AV"] * pipeline_factor      # same penalty for Fc
    s_fc = sig_fc / nb.A_FC_REF

    def A_mb(G, fk=f_kin):
        return nb.area_reversible(G, cell, f_kin=fk)

    def snr(G, sigma, fk=f_kin):
        return A_mb(G, fk) / sigma

    def G_for_snr(target, sigma, fk=f_kin):
        # linear: G = target * sigma / A_mb(1)
        return target * sigma / A_mb(1.0, fk)

    def solve_G0(target, sigma):
        """Smallest G0 (mol/cm^2) with electronic sigma_F(F=0) <= target; solved in log10(G0)
        because brentq's default absolute tolerance (2e-12) is as large as the roots."""
        f = lambda lg: sigF_elec(10.0 ** lg, sigma=sigma) - target
        return 10.0 ** brentq(f, -17.0, -8.0, xtol=1e-10)

    def sigF_elec(G0, F=0.0, sigma=sig_pipeline, N_active=1, n0=3):
        s_mb0 = sigma / A_mb(G0)
        s_mbT = sigma / (A_mb(G0) * (1.0 - F))
        return nb.sigma_F_chain(s_mbT, s_fc, s_mb_ctrl=s_mb0, s_mb0=s_mb0, s_mb0_ctrl=s_mb0,
                                F_k=F, n0=n0, N_active=N_active)["sigma_F"]

    out = {"f_kin": f_kin, "sigma_area_pipeline_AV": sig_pipeline, "sigma_area_fixed_AV": sig_fixed,
           "sigma_area_ad5940_AV": sig_ad5940, "sigma_area_fc_AV": sig_fc, "s_fc_pct": 100 * s_fc,
           "A_MB_nominal_nAV": A_mb(nb.GAMMA0) * 1e9, "snr_nominal_pipeline": snr(nb.GAMMA0, sig_pipeline),
           "snr_nominal_fixed": snr(nb.GAMMA0, sig_fixed), "snr_nominal_ad5940": snr(nb.GAMMA0, sig_ad5940)}

    thr = {}
    for label, sig in [("DStat_pipeline", sig_pipeline), ("DStat_fixed_shape", sig_fixed),
                       ("AD5940_pipeline", sig_ad5940)]:
        d = {}
        d["Gamma_det_pmol"] = G_for_snr(3, sig) * 1e12
        d["Gamma_3pct_pmol"] = G_for_snr(1 / 0.03, sig) * 1e12
        d["Gamma_1pct_pmol"] = G_for_snr(100, sig) * 1e12
        d["F_max_elec"] = 1.0 - G_for_snr(3, sig) / nb.GAMMA0
        d["Gamma0_min_sigF17_pmol"] = solve_G0(0.017, sig) * 1e12
        d["margin_vs_Gamma0"] = nb.GAMMA0 * 1e12 / d["Gamma0_min_sigF17_pmol"]
        thr[label] = d
    out["thresholds"] = thr

    # electronics-only sigma_F versus F_k at the nominal coverage
    Fk = np.linspace(0.0, 0.995, 200)
    out["sigF_elec_1"] = np.array([sigF_elec(nb.GAMMA0, F, N_active=1) for F in Fk])
    out["sigF_elec_4"] = np.array([sigF_elec(nb.GAMMA0, F, N_active=4) for F in Fk])
    out["Fk"] = Fk
    out["sigF_elec_at_0_pct"] = 100 * sigF_elec(nb.GAMMA0, 0.0)
    out["sigF_elec_at_0_4el_pct"] = 100 * sigF_elec(nb.GAMMA0, 0.0, N_active=4)
    out["LOD_F_elec_3sigma_pct"] = 300 * sigF_elec(nb.GAMMA0, 0.0)
    out["gap_to_layer13"] = 6.8 / out["sigF_elec_at_0_pct"]
    # Layer 13 quoted values for comparison
    out["L13_sigF_1"] = 0.068 * (1 - Fk)
    out["L13_sigF_4"] = 0.034 * (1 - Fk)

    # Gamma0_min vs target sigma_F
    targets = np.logspace(-3.5, -1.3, 60)
    out["targets"] = targets
    out["G0min_pipeline"] = np.array([solve_G0(t, sig_pipeline) for t in targets]) * 1e12
    out["G0min_ad5940"] = np.array([solve_G0(t, sig_ad5940) for t in targets]) * 1e12
    out["_fn"] = {"A_mb": A_mb, "snr": snr}
    out["_sigmas"] = (sig_pipeline, sig_fixed, sig_ad5940)
    out["_A_mb"] = A_mb
    return out


def make_figure(res: dict) -> str:
    A_mb = res["_A_mb"]
    sp, sf, sad = res["_sigmas"]
    G = np.logspace(-14, -10.3, 300)            # mol/cm^2 : 0.01 - 5 pmol/cm^2
    Gp = G * 1e12

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

    ax = axes[0]
    ax.loglog(Gp, A_mb(G) / sf, color="#2e7d32", lw=1.6, ls="--", label="DStat, fixed-shape fit (optimum)")
    ax.loglog(Gp, A_mb(G) / sp, color="#1565c0", lw=2.4, label="DStat, full Layer 12 pipeline")
    ax.loglog(Gp, A_mb(G) / sad, color="#c62828", lw=1.6, label="AD5940 case (16-bit, no PGA)")
    ax.loglog(Gp, A_mb(G, 1.0) / sp, color="#9e9e9e", lw=1.2, ls=":", label="reversible limit f_kin = 1")
    for y, lab in [(3, "SNR = 3 (detect)"), (100 / 3, "σ_A/A = 3% (Layer 13 budget)"), (100, "σ_A/A = 1%")]:
        ax.axhline(y, color="k", ls="-.", lw=0.8)
        ax.text(1.06, y * 1.12, lab, fontsize=7)
    ax.axvspan(1, 5, color="green", alpha=0.10, label="Layer 13 valid Γ range")
    ax.axvline(nb.GAMMA0 * 1e12, color="k", ls=":", lw=1)
    ax.set_xlabel("Intact probe coverage Γ (pmol cm⁻²)")
    ax.set_ylabel("SNR of MB peak area (electronics only)")
    ax.set_title("A — SNR vs Γ  (linear: σ_area is Γ-independent)")
    ax.set_xlim(0.01, 5)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    Fk = res["Fk"]
    ax.semilogy(Fk * 100, res["L13_sigF_1"] * 100, color="#9e9e9e", ls="--", lw=1.6,
                label="Layer 13 model, 1 electrode (6.8%·(1−F))")
    ax.semilogy(Fk * 100, res["L13_sigF_4"] * 100, color="#9e9e9e", ls=":", lw=1.6,
                label="Layer 13 model, 4 electrodes")
    ax.semilogy(Fk * 100, res["sigF_elec_1"] * 100, color="#1565c0", lw=2.2, label="Electronics only, 1 pair")
    ax.semilogy(Fk * 100, res["sigF_elec_4"] * 100, color="#00838f", lw=1.6, label="Electronics only, 4 pairs")
    ax.axvline(20, color="orange", ls=":", lw=1.2)
    ax.axvline(95, color="red", ls=":", lw=1.2)
    ax.text(21, 0.013, "F_min = 20%", fontsize=7, color="darkorange")
    ax.text(72, 0.013, "F_max = 95%", fontsize=7, color="red")
    ax.set_xlabel("Fractional cleavage F_k (%)")
    ax.set_ylabel("σ(F_k) (%)")
    ax.set_ylim(1e-2, 20)
    ax.set_title("B — Electronic σ(F_k) at Γ₀ = 2 pmol cm⁻²\n"
                 f"({res['sigF_elec_at_0_pct']:.2f}% at F=0 vs 6.8%: {res['gap_to_layer13']:.0f}× below Layer 13)")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[2]
    ax.loglog(res["targets"] * 100, res["G0min_pipeline"], color="#1565c0", lw=2.2, label="DStat, full pipeline")
    ax.loglog(res["targets"] * 100, res["G0min_ad5940"], color="#c62828", lw=1.6, label="AD5940 case")
    ax.axhspan(1, 10, color="green", alpha=0.10, label="Γ_p design range (1–10 pmol cm⁻²)")
    ax.axvline(1.7, color="k", ls=":", lw=1)
    ax.text(1.8, 0.02, "σ_F = 1.7%\n(F_LOD = 5%)", fontsize=7)
    ax.axvline(6.8, color="grey", ls=":", lw=1)
    ax.text(7.2, 0.02, "Layer 13\nσ_F(0) = 6.8%", fontsize=7)
    ax.set_xlabel("Target electronic σ(F_k = 0) (%)")
    ax.set_ylabel("Minimum starting coverage Γ₀ (pmol cm⁻²)")
    ax.set_title("C — Γ threshold for the electronic LOD floor")
    ax.set_ylim(0.005, 20)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Layer 14: SNR vs intact-probe coverage and the coverage threshold at which electronics matters",
                 fontsize=10)
    fig.tight_layout()
    out = nb.figures_dir() / "fig_L14_snr_vs_gamma.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)
    return str(out)


def main() -> dict:
    res = build()
    print("=" * 72)
    print("Layer 14: SNR vs Gamma_intact")
    print("=" * 72)
    print(f"inferred kinetic factor f_kin (Layer 13 consistency): {res['f_kin']:.4f}")
    print(f"A_MB(Gamma_0 = 2 pmol/cm2) = {res['A_MB_nominal_nAV']:.3f} nA.V")
    print(f"sigma_area: pipeline {res['sigma_area_pipeline_AV']*1e15:.1f} fA.V | fixed-shape "
          f"{res['sigma_area_fixed_AV']*1e15:.1f} | AD5940 {res['sigma_area_ad5940_AV']*1e15:.1f}")
    print(f"SNR at Gamma_0: pipeline {res['snr_nominal_pipeline']:.0f} | fixed-shape {res['snr_nominal_fixed']:.0f} "
          f"| AD5940 {res['snr_nominal_ad5940']:.0f}")
    print(f"Fc relative area noise (electronics): {res['s_fc_pct']:.4f}%")
    print()
    hdr = f"{'case':<22}{'G_det(SNR3)':>13}{'G(3%)':>10}{'G(1%)':>10}{'F_max,el':>10}{'G0_min(1.7%)':>14}{'margin':>9}"
    print(hdr + "   [Gamma in pmol/cm2]")
    for k, d in res["thresholds"].items():
        print(f"{k:<22}{d['Gamma_det_pmol']:>13.4f}{d['Gamma_3pct_pmol']:>10.4f}{d['Gamma_1pct_pmol']:>10.4f}"
              f"{d['F_max_elec']:>10.4f}{d['Gamma0_min_sigF17_pmol']:>14.4f}{d['margin_vs_Gamma0']:>8.0f}x")
    print()
    print(f"Electronics-only sigma_F(F=0): 1 pair {res['sigF_elec_at_0_pct']:.3f}% | 4 pairs {res['sigF_elec_at_0_4el_pct']:.3f}%")
    print(f"Electronics-only F_LOD (3 sigma_F) = {res['LOD_F_elec_3sigma_pct']:.2f}%  vs Layer 13 F_min = 20%  "
          f"(gap {res['gap_to_layer13']:.0f}x)")
    for F in (0.2, 0.5, 0.9, 0.95, 0.99):
        i = np.argmin(abs(res["Fk"] - F))
        print(f"   F = {F:.2f}: sigma_F(elec) = {res['sigF_elec_1'][i]*100:.3f}%")
    print("Saved", make_figure(res))
    slim = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items() if not k.startswith("_")
            and k not in ("Fk", "sigF_elec_1", "sigF_elec_4", "L13_sigF_1", "L13_sigF_4", "targets",
                          "G0min_pipeline", "G0min_ad5940")}
    (nb.results_dir() / "snr_vs_gamma.json").write_text(json.dumps(slim, indent=2, default=float))
    return res


if __name__ == "__main__":
    main()
