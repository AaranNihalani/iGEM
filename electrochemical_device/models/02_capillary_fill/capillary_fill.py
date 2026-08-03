"""
capillary_fill.py  —  Layer 2
===============================
Lucas-Washburn model for capillary-driven fill of the microfluidic channel.

Governing equation (Lucas-Washburn):
  x(t) = sqrt( r * gamma * cos(theta) * t / (2 * eta_visc) )

where:
  x      = fill front position (m)
  r      = effective capillary radius (m)
  gamma  = liquid-air surface tension (N/m)
  theta  = contact angle (degrees)
  eta_visc = dynamic viscosity (Pa·s)

Outputs:
  x(t): fill front position vs time
  t_fill: time to fill channel length L_channel
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).parents[2] / "reports" / "figures"

# ── Channel geometry ─────────────────────────────────────────────────────────
L_channel = 10e-3        # m  (10 mm channel length — typical lateral-flow strip)
w_channel = 1e-3         # m  (1 mm channel width)
h_channel = 100e-6       # m  (100 µm channel height)
# Effective hydraulic radius for rectangular cross-section:
# r_h = 2*w*h / (w+h)
r_h = 2 * w_channel * h_channel / (w_channel + h_channel)

# ── Fluid properties (wound fluid ~= dilute aqueous) ─────────────────────────
gamma    = 0.072         # N/m  (water surface tension at 25°C — CONFIDENT)
eta_visc = 1.0e-3        # Pa·s (water viscosity at 25°C — CONFIDENT)
# Contact angle of wound fluid on PDMS/glass substrate:
# FLAG: depends on surface treatment (O2 plasma, coating). Use theta=30° for
# plasma-treated hydrophilic surface. Verify with fabrication team.
theta_deg = 30.0         # degrees  ← FLAG
theta_rad = np.radians(theta_deg)

# ── Lucas-Washburn solution ───────────────────────────────────────────────────
def fill_front(t):
    """Fill front position x(t) in metres."""
    return np.sqrt(r_h * gamma * np.cos(theta_rad) * t / (2 * eta_visc))

def fill_time(L):
    """Time to fill a channel of length L (m)."""
    return 2 * eta_visc * L**2 / (r_h * gamma * np.cos(theta_rad))

t_fill = fill_time(L_channel)

# ── Sensitivity to contact angle ─────────────────────────────────────────────
theta_range = np.linspace(10, 70, 60)
t_fill_range = [fill_time(L_channel) for th in theta_range
                if np.cos(np.radians(th)) > 0]
t_fill_theta = np.array([fill_time(L_channel) *
                         (np.cos(theta_rad) / np.cos(np.radians(th)))
                         for th in theta_range])


def plot_fill(save=True):
    t = np.linspace(0, t_fill * 1.2, 500)
    x_mm = fill_front(t) * 1e3   # convert to mm

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: fill front vs time
    ax = axes[0]
    ax.plot(t, x_mm, "b-", lw=2)
    ax.axhline(L_channel * 1e3, color="r", ls="--", lw=1.5, label=f"Channel end ({L_channel*1e3:.0f} mm)")
    ax.axvline(t_fill, color="gray", ls=":", lw=1.2)
    ax.annotate(f"t_fill = {t_fill:.1f} s", xy=(t_fill, L_channel*1e3*0.5),
                xytext=(t_fill*0.4, L_channel*1e3*0.5),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fill front position (mm)")
    ax.set_title("A  —  Lucas-Washburn fill front")
    ax.legend(fontsize=9)
    ax.set_xlim(0, t_fill * 1.2)

    # Panel B: t_fill sensitivity to contact angle
    ax2 = axes[1]
    ax2.plot(theta_range, t_fill_theta, "k-", lw=2)
    ax2.axvline(theta_deg, color="b", ls="--", lw=1.5, label=f"Design θ = {theta_deg:.0f}°")
    ax2.set_xlabel("Contact angle θ (°)")
    ax2.set_ylabel("Fill time (s)")
    ax2.set_title("B  —  Sensitivity to contact angle")
    ax2.legend(fontsize=9)

    fig.suptitle("Layer 2: Capillary fill dynamics (Lucas-Washburn)\n"
                 f"r_h = {r_h*1e6:.1f} µm, γ = {gamma:.3f} N/m", fontsize=10)
    plt.tight_layout()

    if save:
        out = FIGURES / "fig_B2_capillary_fill.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Channel geometry: L={L_channel*1e3:.0f} mm, w={w_channel*1e3:.0f} mm, "
          f"h={h_channel*1e6:.0f} µm")
    print(f"Hydraulic radius r_h = {r_h*1e6:.1f} µm")
    print(f"Contact angle θ = {theta_deg}° (FLAG: verify with fabrication)")
    print(f"Fill time = {t_fill:.1f} s ({t_fill/60:.2f} min)")
    plot_fill()
