"""Shared style, colours and loaders for the report figures."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "report" / "figures"
FIG.mkdir(exist_ok=True)

from spred import data, pssm as pssm_mod                      # noqa: E402
from spred.constants import POSITIONS, PROTEASES, VHSE_NAMES  # noqa: E402
from spred.model import load_all                              # noqa: E402
from spred.tables import AMINO_ACIDS                          # noqa: E402

ORDER = ["MMP2", "MMP8", "MMP9", "MMP14", "NE", "LASB", "FECAL"]
LABEL = {"MMP2": "MMP-2", "MMP8": "MMP-8", "MMP9": "MMP-9", "MMP14": "MMP-14", "NE": "Neutrophil elastase",
         "LASB": "LasB", "FECAL": "Digestive serine proteases"}
SHORT = {"MMP2": "MMP-2", "MMP8": "MMP-8", "MMP9": "MMP-9", "MMP14": "MMP-14", "NE": "NE", "LASB": "LasB", "FECAL": "Digestive serine"}
# Palette anchored on the site's own tokens (ink/green/gold/sky from
# styles/theme.css), extended to a validated 8-hue categorical set
# (see electrochemical_device/reports/PALETTE.md for the validation run).
# MMP-9 (our sensor target) takes the primary site green.
COL = {"MMP2": "#0C86AA", "MMP8": "#4a3aa7", "MMP9": "#15876A", "MMP14": "#C79A2E", "NE": "#e34948", "LASB": "#eb6834",
       "FECAL": "#8A9892"}
INK, MUTE, GRID = "#173B34", "#6b7f78", "#E3E8E1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight", "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=110)
    plt.close(fig)
    print("saved", name)


# Sequential (magnitude) and diverging (polarity) colormaps built from the same
# site-anchored tokens as the categorical palette, for heatmaps (replaces
# matplotlib's stock "Blues"/"RdBu_r"). One hue light->dark for sequential;
# two opposing hues + a neutral midpoint for diverging.
SEQ_GREEN = LinearSegmentedColormap.from_list("seq_green", ["#F5EADA", "#B2DFDB", "#15876A", "#173B34"])
DIV_GREEN_RED = LinearSegmentedColormap.from_list(
    "div_green_red", ["#e34948", "#f0a6a0", "#EDEAE1", "#8fcdbf", "#15876A"]
)


def panel_letter(ax, letter, x=-0.12, y=1.14):
    ax.text(x, y, letter.upper(), transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, va="bottom")


def oof(pid):
    z = np.load(ROOT / "results" / "oof" / f"{pid}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def cv_summary():
    return pd.read_csv(ROOT / "results" / "cv_summary.csv").set_index("protease")


MODELS = load_all()
