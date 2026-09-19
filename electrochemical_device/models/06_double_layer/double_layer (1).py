"""
double_layer.py  --  Layer 6.1 / 6.2 / 6.6
=============================================
Gouy-Chapman-Stern (GCS) model for the electrochemical double layer at the
gold-SAM-wound-fluid interface, and its two downstream consequences:
  (a) the capacitive (non-Faradaic) background current in SWV (6.2), and
  (b) a differential-capacitance quality metric for SAM integrity (6.6).

Series capacitor model
-----------------------
    1/C_dl = 1/C_H + 1/C_diffuse

  C_H        Helmholtz (inner-layer) capacitance [uF/cm^2]
             bare Au: ~10-40 uF/cm^2 (specifically adsorbed ions + water dipoles)
             SAM-Au : C_H,SAM = eps0*eps_SAM/d_SAM  (thiol monolayer as a dielectric slab)
  C_diffuse  Gouy-Chapman diffuse-layer capacitance [uF/cm^2]
             C_diffuse = (eps0*eps_r/lambda_D) * cosh(z*F*phi_d / (2*R*T))

At physiological ionic strength C_diffuse >> C_H,SAM, so C_dl is set almost
entirely by the SAM dielectric layer (Sec. 6.1, Layer 6 contents.md).

Reference point (Layer 0, Sec. 6.3): I = 150 mM, T = 310 K, eps_r = 78.5
  ->  lambda_D = 0.77 nm

Feeds: Layer 7 (electrostatic probe/tether model, via lambda_D), Layer 11
(SWV forward model, I_cap(E) as a function of the potential-dependent C_dl),
Layer 14 (noise budget, capacitive background floor).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
eps0 = 8.854e-12      # vacuum permittivity, F/m
F_   = 96485.0        # Faraday constant, C/mol
R_   = 8.314          # gas constant, J/mol/K

# ---------------------------------------------------------------------------
# Physiological reference conditions (Layer 0, Sec. 6.3)  -- CONFIRMED inputs
# ---------------------------------------------------------------------------
T_PHYS   = 310.0       # K   (37 degrees C, wound / body temperature)
I_PHYS   = 0.150       # M   (physiological ionic strength)
EPS_R_W  = 78.5        # relative permittivity of water at 37 C (Layer 0, 6.3)
Z_ION    = 1           # counter-ion valence (Na+/Cl- dominated)


def debye_length_nm(I_M, T=T_PHYS, eps_r=EPS_R_W):
    """
    Debye screening length lambda_D [nm] (Layer 0, Eq. 6.3):
        lambda_D = sqrt( eps*eps0*R*T / (2*F^2*I) )
    I_M in mol/L (converted internally to mol/m^3).
    """
    eps = eps0 * eps_r
    I_SI = I_M * 1e3   # mol/m^3
    lam = np.sqrt(eps * R_ * T / (2 * F_**2 * I_SI))
    return lam * 1e9   # m -> nm


# ---------------------------------------------------------------------------
# 6.1  Helmholtz (inner-layer) capacitance
# ---------------------------------------------------------------------------
# Bare gold: specifically-adsorbed-ion + oriented-water-dipole layer.
# Literature range 10-40 uF/cm^2 (Bard & Faulkner, Ch. 13) -- FLAG: wide range,
# not measured on the fabricated electrode.
C_HELMHOLTZ_BARE_RANGE = (10.0, 40.0)   # uF/cm^2

# SAM-modified gold: the alkanethiol monolayer is treated as a parallel-plate
# dielectric slab in series with the (now much more distant) diffuse layer.
EPS_SAM_RANGE = (2.2, 2.5)     # relative permittivity of an alkyl chain (FLAG)
D_SAM_RANGE_NM = (1.0, 2.0)    # SAM thickness, nm (~C6-C11 alkanethiol)


def C_helmholtz_SAM(eps_sam, d_sam_nm):
    """SAM dielectric-slab capacitance C_H,SAM [uF/cm^2] (Sec. 6.1, Eq. within)."""
    d_m = d_sam_nm * 1e-9
    C = eps0 * eps_sam / d_m          # F/m^2
    return C * 100.0                  # F/m^2 -> uF/cm^2  (1 F/m^2 = 100 uF/cm^2)


# ---------------------------------------------------------------------------
# 6.1  Diffuse (Gouy-Chapman) layer capacitance
# ---------------------------------------------------------------------------
def C_diffuse(I_M, phi_d_V=0.0, T=T_PHYS, eps_r=EPS_R_W, z=Z_ION):
    """
    Gouy-Chapman diffuse-layer capacitance [uF/cm^2]:
        C_diffuse = (eps*eps0/lambda_D) * cosh( z*F*phi_d / (2*R*T) )
    phi_d_V: potential drop across the diffuse layer (OHP -> bulk), V.
    """
    eps = eps0 * eps_r
    lam_m = debye_length_nm(I_M, T, eps_r) * 1e-9
    C = (eps / lam_m) * np.cosh(z * F_ * phi_d_V / (2 * R_ * T))
    return C * 100.0   # F/m^2 -> uF/cm^2  (1 F/m^2 = 100 uF/cm^2)


# ---------------------------------------------------------------------------
# Series combination:  1/C_dl = 1/C_H + 1/C_diffuse
# ---------------------------------------------------------------------------
def C_dl_series(C_H_uF, C_diff_uF):
    return 1.0 / (1.0 / C_H_uF + 1.0 / C_diff_uF)


def C_dl_bare_gold(I_M, phi_d_V=0.0, C_H_uF=25.0):
    """Total double-layer capacitance on bare gold (no SAM)."""
    return C_dl_series(C_H_uF, C_diffuse(I_M, phi_d_V))


def C_dl_SAM(I_M, phi_d_V=0.0, eps_sam=2.35, d_sam_nm=1.5):
    """
    Total double-layer capacitance on SAM-modified gold (Sec. 6.1).
    At physiological I, C_diffuse >> C_H,SAM, so C_dl,SAM ~ C_H,SAM: the SAM
    dielectric slab -- not the diffuse layer -- sets the capacitance.
    """
    C_H = C_helmholtz_SAM(eps_sam, d_sam_nm)
    C_d = C_diffuse(I_M, phi_d_V)
    return C_dl_series(C_H, C_d), C_H, C_d


# ---------------------------------------------------------------------------
# 6.2  Capacitive SWV background current
# ---------------------------------------------------------------------------
def I_cap(C_dl_uF_cm2, A_true_cm2, delta_E_sw_V, f_swv_Hz):
    """
    Capacitive current contribution to one SWV forward/backward step (Sec. 6.2):
        I_cap = C_dl * A_true * 2 * Delta_E_SW * f_SWV
    Returns I_cap in nA. C_dl in uF/cm^2, A_true in cm^2, Delta_E_SW in V.
    """
    C_dl_F_cm2 = C_dl_uF_cm2 * 1e-6
    return C_dl_F_cm2 * A_true_cm2 * 2 * delta_E_sw_V * f_swv_Hz * 1e9  # A -> nA


def I_cap_net(I_cap_nA, asymmetry_frac=0.03):
    """
    Residual NET (forward-backward) capacitive background after SWV
    cancellation, arising from C_dl(E) asymmetry near PZC (Sec. 6.2).
    Typical asymmetry: 1-5% of I_cap (FLAG: not measured on this device).
    """
    return I_cap_nA * asymmetry_frac


# ---------------------------------------------------------------------------
# 6.3  Potential-dependent tether bending (WLC / Pincus electrostatic term)
# ---------------------------------------------------------------------------
def k_ET_potential_correction(k_ET0, beta_per_A, delta_d_A):
    """
    k_ET(E) = k_ET0 * exp(-beta * delta_d(E))  (Sec. 6.3).
    beta_per_A: tunnelling decay constant [1/Angstrom] (~1.0 /A, Layer 10).
    delta_d_A: potential-dependent change in effective tether length [Angstrom]
               (positive = probe pushed away from electrode -> k_ET decreases;
                negative = probe pulled toward electrode -> k_ET increases).
    """
    return k_ET0 * np.exp(-beta_per_A * delta_d_A)


def pincus_bend_estimate(charge_e, E_field_V_per_m, L_contour_nm, l_p_nm=0.38):
    """
    Order-of-magnitude Pincus-regime estimate of the electrostatic end-to-end
    force acting on a charged tether residue (e.g. Arg+ in IPRTLTAA) in the
    field of a polarised electrode, and the resulting change in effective
    tether extension delta_d (Sec. 6.3). This is explicitly an order-of-
    magnitude estimate, not a fitted model -- FLAG.
        F = q * E_field
        delta_d/L ~ (F * l_p / k_B*T)   for small perturbations (Pincus regime)
    Returns delta_d in nm (can be positive or negative depending on charge
    sign and field direction).
    """
    q = charge_e * 1.602e-19
    k_B = 1.381e-23
    T = T_PHYS
    F_force = q * E_field_V_per_m
    delta_d_nm = L_contour_nm * (F_force * l_p_nm * 1e-9 / (k_B * T))
    return delta_d_nm


# ---------------------------------------------------------------------------
# 6.5  Reductive SAM desorption safety margin
# ---------------------------------------------------------------------------
def desorption_safety_margin_mV(E_scan_limit_V, E_des_range_V=(-1.1, -0.8)):
    """
    Safety margin (mV) between the most cathodic SWV potential reached and
    the onset of reductive Au-S desorption (Sec. 6.5).
    Returns the margin to the LEAST negative (most conservative-for-risk,
    i.e. closest) literature desorption onset.
    """
    E_des_closest = max(E_des_range_V)   # least negative = closest to scan limit
    margin_V = E_scan_limit_V - E_des_closest   # positive: scan limit is less cathodic than onset
    return margin_V * 1e3   # V -> mV


# ---------------------------------------------------------------------------
# Worked example / self-check (values quoted in Layer 6 contents.md, Sec 6.1-6.2)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Layer 6.1: Debye length ===")
    lam = debye_length_nm(I_PHYS)
    print(f"lambda_D(I=150 mM, T=310K, eps_r=78.5) = {lam:.3f} nm  (contents.md target: 0.77 nm)")

    print("\n=== Layer 6.1: Double-layer capacitance ===")
    C_dl_sam, C_H_sam, C_diff = C_dl_SAM(I_PHYS, phi_d_V=0.0, eps_sam=2.35, d_sam_nm=1.5)
    print(f"C_H,SAM (eps=2.35, d=1.5 nm)         = {C_H_sam:.2f} uF/cm^2")
    print(f"C_diffuse(I=150 mM, phi_d=0)          = {C_diff:.1f} uF/cm^2")
    print(f"C_dl,SAM (series combination)         = {C_dl_sam:.2f} uF/cm^2  (target range: 1-5)")

    C_dl_bare = C_dl_bare_gold(I_PHYS, C_H_uF=25.0)
    print(f"C_dl, bare gold (C_H=25 uF/cm^2)      = {C_dl_bare:.1f} uF/cm^2  (target range: 10-40 dominated)")

    print("\n=== Layer 6.2: Capacitive SWV background ===")
    A_true = 0.01 * 1.3   # 1 mm^2 geometric x roughness factor R_f=1.3 -> 0.013 cm^2
    Icap = I_cap(C_dl_sam, A_true, delta_E_sw_V=0.025, f_swv_Hz=60.0)
    print(f"A_true                                = {A_true:.4f} cm^2")
    print(f"I_cap (model C_dl={C_dl_sam:.2f} uF/cm^2)   = {Icap:.1f} nA")
    Icap_check = I_cap(2.0, A_true, delta_E_sw_V=0.025, f_swv_Hz=60.0)
    print(f"I_cap (contents.md worked example, C_dl=2.0 uF/cm^2) = {Icap_check:.1f} nA  (target: 78 nA)")
    Icap_net_lo = I_cap_net(Icap_check, 0.01)
    Icap_net_hi = I_cap_net(Icap_check, 0.05)
    print(f"I_cap,net (1-5% asymmetry, on worked example) = {Icap_net_lo:.2f} - {Icap_net_hi:.2f} nA  (target: 1-4 nA)")

    print("\n=== Layer 6.5: SAM desorption safety margin ===")
    margin = desorption_safety_margin_mV(E_scan_limit_V=-0.65)
    print(f"Safety margin at E_scan=-0.65 V vs closest E_des=-0.8 V: {margin:.0f} mV  (target: >150 mV)")
    margin_far = desorption_safety_margin_mV(-0.65, E_des_range_V=(-1.1, -1.1))
    print(f"Safety margin vs far E_des=-1.1 V: {abs(margin_far):.0f} mV")

    # ------------------------------------------------------------------
    def plot_double_layer():
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))

        # Panel A: C_dl vs ionic strength, bare vs SAM
        ax = axes[0, 0]
        I_range = np.linspace(0.01, 0.3, 200)
        C_bare = [C_dl_bare_gold(I, C_H_uF=25.0) for I in I_range]
        C_sam = [C_dl_SAM(I, eps_sam=2.35, d_sam_nm=1.5)[0] for I in I_range]
        ax.plot(I_range * 1e3, C_bare, "b-", lw=2, label="Bare Au (C$_H$=25 $\\mu$F/cm$^2$)")
        ax.plot(I_range * 1e3, C_sam, "r-", lw=2, label="SAM-Au (C$_{H,SAM}$=1.4 $\\mu$F/cm$^2$)")
        ax.axvline(I_PHYS * 1e3, color="g", ls="--", lw=1.2, label=f"Physiological I={I_PHYS*1e3:.0f} mM")
        ax.set_xlabel("Ionic strength I (mM)")
        ax.set_ylabel("C$_{dl}$ ($\\mu$F/cm$^2$)")
        ax.set_title("A -- Double-layer capacitance vs ionic strength")
        ax.legend(fontsize=8)

        # Panel B: Debye length vs I
        ax = axes[0, 1]
        ld_vals = [debye_length_nm(I) for I in I_range]
        ax.plot(I_range * 1e3, ld_vals, "k-", lw=2)
        ax.axvline(I_PHYS * 1e3, color="g", ls="--", lw=1.2,
                   label=f"$\\lambda_D$({I_PHYS*1e3:.0f} mM) = {debye_length_nm(I_PHYS):.2f} nm")
        ax.set_xlabel("Ionic strength I (mM)")
        ax.set_ylabel("Debye length $\\lambda_D$ (nm)")
        ax.set_title("B -- Debye screening length")
        ax.legend(fontsize=8)

        # Panel C: C_diffuse vs potential phi_d (Frumkin-type asymmetry)
        ax = axes[1, 0]
        phi_range = np.linspace(-0.3, 0.3, 200)
        C_diff_phi = [C_diffuse(I_PHYS, phi) for phi in phi_range]
        ax.plot(phi_range * 1e3, C_diff_phi, "m-", lw=2)
        ax.set_xlabel("$\\phi_d$ (mV)")
        ax.set_ylabel("C$_{diffuse}$ ($\\mu$F/cm$^2$)")
        ax.set_title("C -- Diffuse-layer capacitance vs potential\n(source of C$_{dl}$(E) asymmetry, Sec. 6.2)")

        # Panel D: I_cap vs SWV frequency, for C_dl,SAM
        ax = axes[1, 1]
        f_range = np.linspace(5, 150, 200)
        Icap_range = [I_cap(C_dl_sam, A_true, 0.025, f) for f in f_range]
        Icap_net_range = [I_cap_net(v, 0.03) for v in Icap_range]
        ax.plot(f_range, Icap_range, "b-", lw=2, label="I$_{cap}$ (gross)")
        ax.plot(f_range, Icap_net_range, "r--", lw=2, label="I$_{cap,net}$ (3% asymmetry)")
        ax.axvline(60, color="g", ls=":", lw=1.2, label="f$_{SWV}$ = 60 Hz")
        ax.set_xlabel("SWV frequency f (Hz)")
        ax.set_ylabel("Capacitive current (nA)")
        ax.set_title("D -- Capacitive SWV background vs frequency")
        ax.legend(fontsize=8)

        fig.suptitle("Layer 6: Gouy-Chapman-Stern double layer and capacitive SWV background",
                     fontsize=12, y=1.00)
        plt.tight_layout()
        out = FIGURES / "fig_L6_double_layer.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"\nSaved {out}")
        plt.close(fig)

    plot_double_layer()
