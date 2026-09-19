"""
probe_conformation.py -- Layer 8, Sections I-II
================================================
Contour-length inventory and worm-like-chain (WLC) Monte Carlo model of the
Au(anchor) -> MB(reporter) tether on the fabricated MMP-9 probe.

Fabricated architecture (confirmed, Layer 7 Sec. I):
    Ac-Cys - PEG6 - SGKIPRTLTA - PEG2 - Lys(MB) - NH2
     anchor   spacer   substrate   linker   label

This supersedes the two-channel AAPVSGIG/IPRTLTAA/scrambled-probe framing in
the original planning notes (contents.md, Sec. 8, "Probe structure"): the
fabricated device carries a single MMP-9 probe (SGKIPRTLTA) with a
D-enantiomer stereospecific negative control (Layer 13, Sec. III), not a
scrambled-sequence control. The physical model below (WLC contour + Monte
Carlo P(r)) is unaffected by which peptide sequence is used; only the
sequence-specific contour length and persistence length differ.

The chain is modelled as five WLC segments in series, each with its own
contour length L_i and persistence length l_p,i:
    1. Cys anchor residue      (peptide-like)
    2. PEG6 spacer             (Oesterhelt 1999)
    3. SGKIPRTLTA (10 aa)      (peptide-like, extended-strand contour)
    4. PEG2 linker             (Oesterhelt 1999)
    5. Lys(MB) label linker    (peptide-like)

Segment junctions are treated as orientationally decorrelating (chemically
distinct backbones meet at the junction; no persistent correlation is
assumed to propagate across it). Under this assumption the total mean-square
end-to-end distance of the concatenated chain is the sum of each segment's
own WLC <r_i^2> (independent-segment / additive-variance approximation),
and an effective single persistence length l_p,eff for the whole chain is
back-calculated by variance-matching against a homogeneous WLC of length
L_c,total.

Monte Carlo generation: each segment is discretised into short steps of
length l0 << l_p using the standard bead-tangent algorithm: successive
tangent vectors differ by a random rotation whose polar angle theta is
drawn from the Boltzmann/von-Mises-Fisher density
    p(cos theta) ~ exp[(l_p/l0) cos theta],   phi ~ Uniform(0, 2 pi)
which reproduces the WLC tangent-tangent correlation <cos theta> = exp(-l0/l_p)
for l0 << l_p (Kratky-Porod). This is run independently per segment (own
l_p, own step count) and concatenated into a single 3-D path from the Au
anchor to the MB label; r = |R_MB - R_Au| is the tunnelling distance sampled
for Layer 10's Marcus expression k_ET(r) (average_ket.py).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

rng = np.random.default_rng(20260917)

# ─────────────────────────────────────────────────────────────────────────
# I. Contour length inventory (Sec. 8.1, updated to the fabricated sequence)
# ─────────────────────────────────────────────────────────────────────────
b_PEG        = 0.36e-9   # m, PEG monomer extended contour length (-CH2-CH2-O-)
b_aa         = 0.35e-9   # m, extended beta-strand contour length per residue
L_Cys        = 0.50e-9   # m, Cys anchor residue (backbone + sidechain to Au)
L_LysMB      = 0.50e-9   # m, Lys epsilon-amine -> MB linker

N_PEG6       = 6
N_PEG2       = 2
N_AA_PROBE   = 10        # SGKIPRTLTA (fabricated sequence, Layer 7)

L_PEG6       = N_PEG6 * b_PEG
L_PEG2       = N_PEG2 * b_PEG
L_peptide    = N_AA_PROBE * b_aa

# Persistence lengths (Sec. 8.2)
lp_PEG   = 0.38e-9   # m, Oesterhelt et al. 1999 (AFM force spectroscopy) -- NEW to Layer 8
lp_pep   = 0.60e-9   # m, central estimate for unstructured/mixed peptide, range 0.4-1.0 nm
                      # SGKIPRTLTA carries one Pro (P5) -> some local stiffening expected,
                      # but no beta-sheet-forming motif is confirmed; central literature
                      # estimate is used pending a sequence-specific measurement (Limitation).
lp_Cys   = lp_pep     # single-residue anchor treated as a peptide-type segment
lp_LysMB = lp_pep     # short peptide-type linker

SEGMENTS = [
    dict(name="Cys anchor",        L=L_Cys,     lp=lp_Cys),
    dict(name="PEG6 spacer",       L=L_PEG6,    lp=lp_PEG),
    dict(name="SGKIPRTLTA (10aa)", L=L_peptide, lp=lp_pep),
    dict(name="PEG2 linker",       L=L_PEG2,    lp=lp_PEG),
    dict(name="Lys(MB) linker",    L=L_LysMB,   lp=lp_LysMB),
]

L_c_total = sum(s["L"] for s in SEGMENTS)


def wlc_mean_r2(L, lp):
    """Kratky-Porod mean-squared end-to-end distance for a single WLC segment."""
    if L <= 0:
        return 0.0
    x = L / lp
    return 2 * lp * L * (1 - (lp / L) * (1 - np.exp(-x)))


def effective_persistence_length(segments):
    """
    Additive-variance approximation: <r^2>_total = sum_i <r_i^2>_i (segments
    orientationally decorrelated at junctions). l_p,eff is then the
    persistence length of a homogeneous WLC of length L_c_total that
    reproduces the same <r^2>_total (solved numerically).
    """
    r2_total = sum(wlc_mean_r2(s["L"], s["lp"]) for s in segments)
    L_tot = sum(s["L"] for s in segments)

    from scipy.optimize import brentq

    def f(lp_trial):
        return wlc_mean_r2(L_tot, lp_trial) - r2_total

    lp_eff = brentq(f, 1e-12, L_tot)
    return lp_eff, r2_total, L_tot


lp_eff, r2_total_analytic, L_tot_check = effective_persistence_length(SEGMENTS)
assert abs(L_tot_check - L_c_total) < 1e-15

# ─────────────────────────────────────────────────────────────────────────
# II. Monte Carlo worm-like-chain generation
# ─────────────────────────────────────────────────────────────────────────

def sample_wlc_segment(L, lp, l0, rng):
    """
    Generate one realisation of a discretised WLC segment of contour length L
    and persistence length lp, step size l0 (m). Returns an (n_steps, 3)
    array of bond vectors in the segment's own frame (first bond along +z).
    """
    n_steps = max(1, int(round(L / l0)))
    step = L / n_steps
    kappa = lp / step  # concentration parameter of the tangent distribution

    bonds = np.zeros((n_steps, 3))
    # first tangent: along local +z
    bonds[0] = np.array([0.0, 0.0, 1.0]) * step

    # vectorised draw of all (u, phi) pairs up front -- only the sequential
    # frame rotation below has a genuine step-to-step dependency
    if n_steps > 1:
        xi = rng.uniform(size=n_steps - 1)
        if kappa > 30:
            u_arr = np.clip(1 + np.log(xi) / kappa, -1.0, 1.0)
        else:
            u_arr = np.log(xi * (np.exp(kappa) - np.exp(-kappa)) + np.exp(-kappa)) / kappa
        phi_arr = rng.uniform(0, 2 * np.pi, size=n_steps - 1)
        sin_arr = np.sqrt(np.clip(1 - u_arr**2, 0.0, None))
        t_local_arr = np.column_stack([sin_arr * np.cos(phi_arr),
                                        sin_arr * np.sin(phi_arr),
                                        u_arr])

        for i in range(1, n_steps):
            # rotate t_local into the lab frame so its z-axis aligns with the
            # previous bond direction. Built directly from an orthonormal
            # frame with z-axis = prev (numerically stable for all prev,
            # including near-antiparallel steps where a naive Rodrigues
            # formula has a 1/(1+cos) singularity).
            prev = bonds[i - 1] / step
            R = _frame_from_zaxis(prev)
            bonds[i] = (R @ t_local_arr[i - 1]) * step

    return bonds


def _one_chain_path(rng, l0):
    """Generate one full-path realisation (all bead z-coordinates), lab frame
    z = distance from the gold plane (anchor fixed at z=0, initial bond
    pointing away from the surface, +z)."""
    pos = np.zeros(3)
    z_path = [0.0]
    orientation = np.eye(3)
    for seg in SEGMENTS:
        bonds = sample_wlc_segment(seg["L"], seg["lp"], l0, rng)
        bonds_lab = bonds @ orientation.T
        cum = pos + np.cumsum(bonds_lab, axis=0)
        z_path.extend(cum[:, 2].tolist())
        pos = cum[-1]
        last_dir = bonds_lab[-1] / np.linalg.norm(bonds_lab[-1])
        orientation = _frame_from_zaxis(last_dir)
    return pos, np.array(z_path)


def sample_full_chain(rng, l0=0.05e-9, enforce_surface=True, max_tries=200):
    """
    One MC realisation of the full 5-segment tether; returns end-to-end r (m).

    The Au-S anchor fixes the chain start at the gold surface (z=0). Since
    the electrode is an impenetrable half-space (z<0 is metal), any path
    that dips below z=0 is unphysical and is rejected and resampled
    (surface-avoiding / reflecting-wall WLC, cf. Rant et al. 2004 for
    surface-tethered ET probes). This excludes the subset of nominally
    short-r configurations that would require the chain to pass through
    the electrode, and is the physically appropriate boundary condition
    for the Layer 10 tunnelling-distance ensemble.
    """
    if not enforce_surface:
        pos, _ = _one_chain_path(rng, l0)
        return np.linalg.norm(pos), pos

    for _ in range(max_tries):
        pos, z_path = _one_chain_path(rng, l0)
        if np.all(z_path >= 0.0):
            return np.linalg.norm(pos), pos
    # extremely rare fallback: accept the last draw even if it dipped below
    # the surface, rather than silently bias the ensemble by discarding data
    return np.linalg.norm(pos), pos


def _frame_from_zaxis(z):
    """Build an orthonormal frame whose z-axis is the given unit vector."""
    z = z / np.linalg.norm(z)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def monte_carlo_P_r(n_samples=20000, l0=0.05e-9, seed=20260917):
    """Run the full MC ensemble; returns array of r samples (m)."""
    local_rng = np.random.default_rng(seed)
    r_samples = np.empty(n_samples)
    for i in range(n_samples):
        r_samples[i], _ = sample_full_chain(local_rng, l0=l0)
    return r_samples


def summarise(r_samples):
    r_nm = r_samples * 1e9
    return dict(
        mean_r_nm=r_nm.mean(),
        rms_r_nm=np.sqrt((r_nm**2).mean()),
        median_r_nm=np.median(r_nm),
        p10_nm=np.percentile(r_nm, 10),
        p90_nm=np.percentile(r_nm, 90),
    )


def plot_conformation(r_samples, save=True):
    r_nm = r_samples * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.bar(range(len(SEGMENTS)),
           [s["L"] * 1e9 for s in SEGMENTS],
           tick_label=[s["name"].split(" ")[0] for s in SEGMENTS],
           color="#3b6fa0")
    ax.set_ylabel("Segment contour length (nm)")
    ax.set_title(f"A — Contour inventory  (L$_c$ = {L_c_total*1e9:.2f} nm)")
    ax.tick_params(axis='x', rotation=30)

    ax2 = axes[1]
    ax2.hist(r_nm, bins=80, density=True, color="#3b6fa0", alpha=0.75,
              label="Monte Carlo $P(r)$\n(5-segment WLC)")
    stats = summarise(r_samples)
    ax2.axvline(stats["mean_r_nm"], color="k", ls="--", lw=1.5,
                label=f"$\\langle r \\rangle$ = {stats['mean_r_nm']:.2f} nm")
    ax2.axvline(L_c_total * 1e9, color="r", ls=":", lw=1.5,
                label=f"$L_c$ = {L_c_total*1e9:.2f} nm (full extension)")
    ax2.set_xlabel("MB-to-Au distance $r$ (nm)")
    ax2.set_ylabel("Probability density")
    ax2.set_title(f"B — $P(r)$,  $\\ell_{{p,eff}}$ = {lp_eff*1e9:.2f} nm")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, L_c_total * 1e9 * 1.05)

    fig.suptitle("Layer 8.1/8.2/8.3: Probe contour inventory and WLC end-to-end distribution "
                 "(Ac-Cys-PEG6-SGKIPRTLTA-PEG2-Lys(MB))", fontsize=10)
    plt.tight_layout()
    if save:
        out = FIGURES / "fig_L8_probe_conformation.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Segment contour-length inventory:")
    for s in SEGMENTS:
        print(f"  {s['name']:<20s}  L = {s['L']*1e9:5.2f} nm   l_p = {s['lp']*1e9:.2f} nm")
    print(f"  {'TOTAL':<20s}  L_c = {L_c_total*1e9:5.2f} nm")
    print()
    print(f"Effective persistence length (variance-matched): l_p,eff = {lp_eff*1e9:.3f} nm")
    print(f"Analytic <r^2>^0.5 (additive-variance WLC)       = {np.sqrt(r2_total_analytic)*1e9:.3f} nm")
    print(f"L_c,total / l_p,eff                              = {L_c_total/lp_eff:.2f}  "
          f"(short-chain regime, non-Gaussian P(r) expected)")

    N_MC = 12000
    r_samples = monte_carlo_P_r(n_samples=N_MC)
    stats = summarise(r_samples)
    print()
    print(f"Monte Carlo P(r) summary ({N_MC:,} chain realisations, surface-excluded):")
    for k, v in stats.items():
        print(f"  {k:<12s} = {v:.3f} nm")

    np.save(Path(__file__).parent / "r_samples_L8.npy", r_samples)
    plot_conformation(r_samples)
