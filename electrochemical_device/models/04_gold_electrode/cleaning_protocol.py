"""
cleaning_protocol.py  —  Layer 4
===================================
Gold surface cleaning-state model: which cleaning protocol is compatible
with an already-assembled disposable cartridge, and how contact angle
(a proxy for SAM-formation readiness) recovers toward its "dirty"
equilibrium during shelf storage after that cleaning step.

theta(t_storage) = theta_0 + (theta_inf - theta_0) * (1 - exp(-k_decay * t_storage))

theta_0    : contact angle immediately post-clean
theta_inf  : long-time (hydrocarbon-saturated) equilibrium contact angle
k_decay    : first-order recovery rate constant (1/day)

All k_decay values are ILLUSTRATIVE / UNCONFIRMED: contents.md (Sec 4.5)
specifies the functional form but not a rate constant, and Ulman (Chem.
Rev. 1996) is cited only as qualitative support for hydrocarbon
re-adsorption on clean gold/metal surfaces over hours-to-weeks, not for a
specific rate. A real shelf-life claim requires measuring theta(t) on
plasma-cleaned coupons stored under the intended packaging.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# Cleaning protocol comparison (contents.md Sec 4.5)
# ─────────────────────────────────────────────────────────────────────────

CLEANING_PROTOCOLS = {
    "O2 plasma (2 min, 50 W)": {
        "removes": "organics; makes surface hydrophilic",
        "theta_post_deg": (0, 10),
        "Rf_effect": "no significant change for evaporated Au (1.1-1.3)",
        "compatible_assembled_cartridge": True,
        "notes": "Most practical option: apply immediately before cartridge sealing.",
    },
    "UV/ozone (15 min)": {
        "removes": "organics (similar mechanism to O2 plasma, less aggressive)",
        "theta_post_deg": (5, 20),
        "Rf_effect": "no significant change",
        "compatible_assembled_cartridge": True,
        "notes": "Slower and milder than plasma; viable fallback if plasma tool unavailable.",
    },
    "Piranha (H2SO4/H2O2)": {
        "removes": "organics completely",
        "theta_post_deg": (0, 5),
        "Rf_effect": "no significant change",
        "compatible_assembled_cartridge": False,
        "notes": "NOT compatible with polymer (PET/PETG) substrates or an assembled wet cartridge.",
    },
    "Electrochemical (CV in H2SO4, 20 cycles)": {
        "removes": "adsorbed species; anneals/cleans gold surface",
        "theta_post_deg": (0, 10),
        "Rf_effect": "can reduce R_f toward 1 (surface annealing)",
        "compatible_assembled_cartridge": False,
        "notes": "Requires acidic electrolyte in contact with the WE — cannot be done post-assembly.",
    },
}

CHOSEN_PROTOCOL = "O2 plasma (2 min, 50 W)"  # for assembled disposable cartridge


# ─────────────────────────────────────────────────────────────────────────
# Contact-angle shelf-life model
# ─────────────────────────────────────────────────────────────────────────

def theta_storage(t_days, theta_0_deg=5.0, theta_inf_deg=45.0, k_decay_per_day=0.3):
    """
    theta(t) = theta_0 + (theta_inf - theta_0) * (1 - exp(-k_decay * t))

    t_days : storage time, days (array-like or scalar)
    theta_0_deg   : post-clean contact angle (deg)
    theta_inf_deg : long-time hydrocarbon-saturated equilibrium angle (deg)
    k_decay_per_day : recovery rate constant (1/day) — UNCONFIRMED, illustrative
    """
    t = np.asarray(t_days, dtype=float)
    return theta_0_deg + (theta_inf_deg - theta_0_deg) * (1 - np.exp(-k_decay_per_day * t))


def shelf_life_days(theta_threshold_deg, theta_0_deg=5.0, theta_inf_deg=45.0,
                     k_decay_per_day=0.3):
    """
    Time (days) for theta(t) to reach a given SAM-quality threshold angle.
    Inverts theta_storage(); returns np.inf if the threshold is never
    reached (theta_inf <= threshold).
    """
    if theta_inf_deg <= theta_threshold_deg:
        return np.inf
    frac = (theta_threshold_deg - theta_0_deg) / (theta_inf_deg - theta_0_deg)
    if frac <= 0:
        return 0.0
    if frac >= 1:
        return np.inf
    return -np.log(1 - frac) / k_decay_per_day


# Illustrative rate-constant scenarios spanning "fast" to "slow" hydrocarbon
# re-adsorption, bracketing the qualitative hours-to-weeks range in Ulman
# (1996). UNCONFIRMED — replace with a fitted k_decay once real theta(t)
# data exists on stored, plasma-cleaned coupons.
K_DECAY_SCENARIOS = {
    "fast re-contamination (tau ~ 1 day)":  1.0,
    "medium re-contamination (tau ~ 3 days)": 1 / 3,
    "slow re-contamination (tau ~ 10 days)":  1 / 10,
}

THETA_ACCEPTABLE_DEG = 30.0  # SAM-quality acceptability threshold (contents.md Sec 4.5 range)


def make_figures(save=True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # --- Panel A: theta(t) for each rate scenario -------------------------
    t_days = np.linspace(0, 21, 400)
    ax = axes[0]
    for label, k in K_DECAY_SCENARIOS.items():
        theta = theta_storage(t_days, k_decay_per_day=k)
        ax.plot(t_days, theta, lw=2, label=label)
        t_shelf = shelf_life_days(THETA_ACCEPTABLE_DEG, k_decay_per_day=k)
        if np.isfinite(t_shelf) and t_shelf <= t_days[-1]:
            ax.axvline(t_shelf, color="gray", ls=":", lw=1, alpha=0.6)

    ax.axhline(THETA_ACCEPTABLE_DEG, color="k", ls="--", lw=1.3,
               label=f"SAM-quality threshold ({THETA_ACCEPTABLE_DEG:.0f} deg)")
    ax.set_xlabel("Storage time (days)")
    ax.set_ylabel("Contact angle theta (deg)")
    ax.set_title("A — Post-plasma contact-angle recovery\n(UNCONFIRMED k_decay — illustrative)")
    ax.legend(fontsize=7)

    # --- Panel B: shelf life (days to threshold) vs k_decay ---------------
    ax = axes[1]
    k_range = np.linspace(0.05, 1.2, 200)
    shelf = [shelf_life_days(THETA_ACCEPTABLE_DEG, k_decay_per_day=k) for k in k_range]
    ax.plot(k_range, shelf, "b-", lw=2)
    for label, k in K_DECAY_SCENARIOS.items():
        ts = shelf_life_days(THETA_ACCEPTABLE_DEG, k_decay_per_day=k)
        ax.plot(k, ts, "o", ms=6)
        ax.annotate(label.split("(")[1].rstrip(")"), (k, ts),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel("k_decay (1/day)")
    ax.set_ylabel(f"Shelf life to theta={THETA_ACCEPTABLE_DEG:.0f} deg (days)")
    ax.set_title("B — Shelf-life sensitivity to (unconfirmed) k_decay")

    fig.suptitle("Layer 4: Cleaning protocol and post-clean shelf-life model", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_L4_cleaning_protocol.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Cleaning protocol comparison (contents.md Sec 4.5):")
    for name, spec in CLEANING_PROTOCOLS.items():
        compat = "OK for assembled cartridge" if spec["compatible_assembled_cartridge"] \
            else "NOT compatible post-assembly"
        lo, hi = spec["theta_post_deg"]
        print(f"  - {name}: theta_post = {lo}-{hi} deg | {compat}")
    print(f"\nChosen protocol for assembled disposable cartridge: {CHOSEN_PROTOCOL}")

    print(f"\nShelf life to reach theta = {THETA_ACCEPTABLE_DEG} deg, by scenario "
          f"(theta_0=5 deg, theta_inf=45 deg — UNCONFIRMED):")
    for label, k in K_DECAY_SCENARIOS.items():
        t_shelf = shelf_life_days(THETA_ACCEPTABLE_DEG, k_decay_per_day=k)
        print(f"  - {label}: {t_shelf:.1f} days")

    make_figures()
