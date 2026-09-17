"""
electrode_characterisation.py  —  Layer 4
============================================
Gold working-electrode (WE) surface characterisation.

Three linked calculations:
  (1) Roughness factor R_f from Au-oxide-stripping CV charge, and the
      true (electroactive) area A_true = A_geo * R_f that all Layer 7-13
      surface densities (Gamma, mol/cm^2) must be normalised against.
  (2) Ohmic resistance budget: gold-film trace resistance R_Au +
      contact resistance R_contact, compared against the solution
      resistance R_s that Layer 6 shows actually dominates R_u.
  (3) Electrode-area variability (lithography + dicing + undercut),
      propagated by Monte Carlo to Gamma_0, I_MB, I_Fc, and to the
      ratiometric baseline R_i(0) = I_MB(0)/I_Fc(0), to check how much
      of it the ratiometric design (Layer 13) actually cancels.

All numeric values not measured on a fabricated device are literature
placeholders and are flagged FLAG / UNCONFIRMED in-line, exactly as
Layer 4's contents.md source material specifies them.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

rng = np.random.default_rng(4)

# ─────────────────────────────────────────────────────────────────────────
# (1) Roughness factor and true electroactive area
# ─────────────────────────────────────────────────────────────────────────

# Standard charge density for stripping one monolayer of Au oxide.
# CONFIRMED literature constant: Trasatti & Petrii, Pure Appl. Chem. 63
# (1991) 711-734.
Q_REF_uC_cm2 = 390.0  # microC / cm^2

# Base-case electrode geometry: 1 mm x 1 mm square WE (contents.md Sec 4.4
# worked example). FLAG: confirm against the actual photomask design.
L_geo_cm = 0.1  # cm  (1 mm)
A_geo_cm2 = L_geo_cm ** 2  # cm^2 = 0.01 cm^2


def rf_from_Q_oxide(Q_oxide_stripping_uC, A_geo_cm2=A_geo_cm2):
    """
    Roughness factor from a measured Au-oxide-stripping CV charge.

        R_f = Q_oxide_stripping / (Q_ref * A_geo)

    Q_oxide_stripping_uC : measured stripping charge, microC (integrate the
        oxide-reduction peak of a CV run in 0.5 M H2SO4).
    """
    return Q_oxide_stripping_uC / (Q_REF_uC_cm2 * A_geo_cm2)


def A_true_cm2(A_geo_cm2, R_f):
    """True (electroactive) area = geometric area * roughness factor."""
    return A_geo_cm2 * R_f


def gamma_true_from_gamma_geo(Gamma_geo, R_f):
    """
    Convert a geometric-area surface density to a true-area surface
    density: Gamma_true = Gamma_geo / R_f.

    Same total moles of probe spread over more real area -> lower local
    (true-area) density. Using R_f = 1 when the real value is, say, 1.3
    overestimates Gamma_true by 30% (contents.md Sec 4.1).
    """
    return Gamma_geo / R_f


# Representative literature roughness-factor bands (contents.md Sec 4.1):
RF_BANDS = {
    "smooth evaporated Au":      (1.1, 1.5),
    "annealed / polycrystalline": (1.5, 3.0),
    "rough (electrodeposited)":  (3.0, 8.0),   # illustrative upper bound only
}

RF_NOMINAL = 1.3  # FLAG: representative smooth-evaporated-Au value; measure per batch


# ─────────────────────────────────────────────────────────────────────────
# (2) Ohmic resistance budget
# ─────────────────────────────────────────────────────────────────────────

RHO_AU = 2.44e-8  # Ohm*m, bulk gold resistivity (CONFIDENT, handbook value)


def R_Au(L_trace_m, w_m, t_m, rho_Au=RHO_AU):
    """Gold-film trace resistance: R = rho * L / (w * t)."""
    return rho_Au * L_trace_m / (w_m * t_m)


# Worked example from contents.md Sec 4.2: L=5 mm, w=0.5 mm, t=100 nm
R_Au_example = R_Au(L_trace_m=5e-3, w_m=0.5e-3, t_m=100e-9)

# Contact-resistance ranges by interconnect technology (contents.md Sec 4.3)
R_CONTACT_RANGES_OHM = {
    "spring-pin pogo": (0.010, 0.100),
    "ACF (anisotropic conductive film)": (0.001, 0.100),
    "wire bond (hybrid)": (0.001, 0.010),
}

R_S_TYPICAL_OHM = 33.0  # solution resistance, Layer 6 worked example — dominant term


def R_u_total(R_Au_val, R_contact_val, R_s_val=R_S_TYPICAL_OHM):
    """Total uncompensated resistance seen by the potentiostat."""
    return R_Au_val + R_contact_val + R_s_val


def iR_shift(I_peak_A, R_u_ohm):
    """Peak potential shift due to uncompensated resistance: V = I * R."""
    return I_peak_A * R_u_ohm


# ─────────────────────────────────────────────────────────────────────────
# (3) Electrode area variability -> Monte Carlo propagation
# ─────────────────────────────────────────────────────────────────────────

def sample_A_geo(n, L_nominal_cm=L_geo_cm, seed=None):
    """
    Monte Carlo sample of geometric electrode area for a square WE of
    nominal side L_nominal_cm, combining three independent error sources
    as FRACTIONAL AREA perturbations (contents.md Sec 4.4 gives each
    source's effect directly as a fractional-area figure, not a length,
    so sources are combined in quadrature on A/A_nominal rather than on
    L):

      - Photolithography edge placement (+/-1 um on a 1 mm side):
        +/-0.2% on area, as stated directly in contents.md.
      - Dicing / mask alignment (+/-0.1 mm offset on a 1 mm electrode):
        +/-10% on area, as stated directly in contents.md.
      - Wet-etch undercut (+/-2-5 um per side, mid-point 3.5 um, on all
        4 edges of a 1 mm square): ~1.4% on area.

    These three combine in quadrature (RSS) to ~10.1%, at the upper end
    of contents.md's stated total +/-5-10% inter-electrode range — dicing
    alignment dominates. Returns an array of areas in cm^2.
    """
    g = np.random.default_rng(seed) if seed is not None else rng
    A_nom_cm2 = L_nominal_cm ** 2

    litho_area_frac_sigma = 0.002     # +/-0.2% (contents.md, stated directly)
    dicing_area_frac_sigma = 0.10     # +/-10%  (contents.md, stated directly)
    undercut_area_frac_sigma = 0.014  # ~1.4%, from +/-3.5 um/side on 4 edges

    frac = (g.normal(0, litho_area_frac_sigma, n)
            + g.normal(0, dicing_area_frac_sigma, n)
            + g.normal(0, undercut_area_frac_sigma, n))
    A_cm2 = A_nom_cm2 * (1 + frac)
    return np.clip(A_cm2, 1e-6, None)


def propagate_to_ratiometric(n=200_000, gamma_mismatch_sigma=0.0, seed=7):
    """
    Propagate electrode-area variability to Gamma_0 (per-electrode, at
    fixed total probe moles), I_MB, I_Fc, and the ratiometric baseline
    R_i(0) = I_MB(0)/I_Fc(0).

    Because I_MB and I_Fc are both proportional to A_true (Layer 13,
    Eq. I.1-I.3), A_true cancels EXACTLY in R_i(0) if Gamma_MB/Gamma_Fc is
    identical on every electrode. gamma_mismatch_sigma introduces an
    extra, area-independent per-electrode relative mismatch between the
    MB-probe and Fc-reference coverages (contents.md Sec 4.4's caveat) to
    show how imperfect that cancellation becomes when it is nonzero.
    """
    g = np.random.default_rng(seed)
    A = sample_A_geo(n, seed=seed)
    A_rel = A / A.mean()

    # I_MB, I_Fc both scale with A_true (and hence with A_geo at fixed R_f)
    I_MB = A_rel * (1 + g.normal(0, gamma_mismatch_sigma, n))
    I_Fc = A_rel * (1 + g.normal(0, gamma_mismatch_sigma, n))
    R0 = I_MB / I_Fc

    return {
        "A_rel_sigma": A_rel.std(),
        "I_MB_sigma": I_MB.std(),
        "I_Fc_sigma": I_Fc.std(),
        "R0_sigma": R0.std(),
        "A_rel": A_rel,
        "R0": R0,
    }


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def make_figures(save=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # --- Panel A: R_f -> A_true and local Gamma_true --------------------
    rf_range = np.linspace(1.0, 4.0, 200)
    A_true_range = A_geo_cm2 * rf_range
    ax = axes[0]
    ax.plot(rf_range, A_true_range * 100, "b-", lw=2, label="A_true")  # cm^2 -> mm^2
    ax.axvline(RF_NOMINAL, color="r", ls="--", lw=1.5,
               label=f"Nominal R_f = {RF_NOMINAL} (FLAG)")
    for name, (lo, hi) in RF_BANDS.items():
        ax.axvspan(lo, hi, alpha=0.08)
    ax.axhline(A_geo_cm2 * 100, color="gray", ls=":", lw=1, label="A_geo")
    ax.set_xlabel("Roughness factor R_f")
    ax.set_ylabel("Area (mm$^2$)")
    ax.set_title("A — A_true vs R_f (1x1 mm WE)")
    ax.legend(fontsize=7, loc="upper left")

    # --- Panel B: Resistance budget --------------------------------------
    ax = axes[1]
    labels = ["R_Au\n(5mm trace)", "R_contact\n(pogo, max)", "R_u\n(Au+contact)", "R_s\n(solution, Layer 6)"]
    r_contact_max = R_CONTACT_RANGES_OHM["spring-pin pogo"][1]
    values = [R_Au_example, r_contact_max, R_Au_example + r_contact_max, R_S_TYPICAL_OHM]
    colors = ["#4C72B0", "#4C72B0", "#DD8452", "#55A868"]
    ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Resistance (Ohm)")
    ax.set_title("B — Ohmic resistance budget")
    for i, v in enumerate(values):
        ax.text(i, v * 1.3, f"{v:.2g}", ha="center", fontsize=8)

    # --- Panel C: Area-variability Monte Carlo, with / without Gamma mismatch
    ax = axes[2]
    clean = propagate_to_ratiometric(gamma_mismatch_sigma=0.0)
    mismatched = propagate_to_ratiometric(gamma_mismatch_sigma=0.05)
    bins = np.linspace(0.7, 1.3, 80)
    ax.hist(clean["A_rel"], bins=bins, alpha=0.5, density=True,
            label=f"A_geo/A_mean (sigma={clean['A_rel_sigma']*100:.1f}%)")
    ax.hist(clean["R0"], bins=bins, alpha=0.5, density=True,
            label=f"R_i(0), no Gamma mismatch (sigma={clean['R0_sigma']*100:.2f}%)")
    ax.hist(mismatched["R0"], bins=bins, alpha=0.5, density=True,
            label=f"R_i(0), 5% Gamma mismatch (sigma={mismatched['R0_sigma']*100:.1f}%)")
    ax.set_xlabel("Relative value")
    ax.set_ylabel("Density")
    ax.set_title("C — Area variability: raw vs ratiometric")
    ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle("Layer 4: Gold electrode characterisation "
                  "(R_f, R_u, and area-variability propagation)", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L4_electrode_characterisation.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved {out}")
    plt.close(fig)

    return clean, mismatched


if __name__ == "__main__":
    print(f"A_geo = {A_geo_cm2*1e2:.3f} mm^2 "
          f"(1x1 mm WE, FLAG: confirm against photomask)")
    print(f"Nominal R_f = {RF_NOMINAL} -> A_true = {A_true_cm2(A_geo_cm2, RF_NOMINAL)*1e2:.3f} mm^2")
    print(f"R_f back-calculated from Q_oxide = 4.5 uC on 0.01 cm^2: "
          f"{rf_from_Q_oxide(4.5, 0.01):.2f}")
    print(f"\nR_Au (5mm x 0.5mm x 100nm trace) = {R_Au_example:.3f} Ohm")
    for name, (lo, hi) in R_CONTACT_RANGES_OHM.items():
        print(f"R_contact [{name}] = {lo*1e3:.0f}-{hi*1e3:.0f} mOhm")
    R_u = R_u_total(R_Au_example, R_CONTACT_RANGES_OHM["spring-pin pogo"][1])
    print(f"R_u (Au + contact, pogo, worst case) = {R_u:.3f} Ohm "
          f"({R_u/R_S_TYPICAL_OHM*100:.1f}% of typical R_s = {R_S_TYPICAL_OHM} Ohm)")
    shift = iR_shift(100e-9, R_u)
    print(f"iR peak shift at I_peak=100 nA: {shift*1e6:.3f} uV — negligible vs ~mV peak widths")

    clean, mismatched = make_figures()
    print(f"\nMonte Carlo (n=200000), 1x1mm WE:")
    print(f"  sigma(A_geo)/A_geo        = {clean['A_rel_sigma']*100:.2f}%")
    print(f"  sigma(R_i(0)), Gamma-matched   = {clean['R0_sigma']*100:.3f}%  (near-total cancellation)")
    print(f"  sigma(R_i(0)), 5% Gamma mismatch = {mismatched['R0_sigma']*100:.2f}%  (residual, uncancelled)")
