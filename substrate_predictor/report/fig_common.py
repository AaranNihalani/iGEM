"""Shared style, colours and loaders for the report figures."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
         "LASB": "LasB", "FECAL": "Faecal serine proteases"}
SHORT = {"MMP2": "MMP-2", "MMP8": "MMP-8", "MMP9": "MMP-9", "MMP14": "MMP-14", "NE": "NE", "LASB": "LasB", "FECAL": "Faecal"}
COL = {"MMP2": "#6A4C93", "MMP8": "#1982C4", "MMP9": "#2A9D6F", "MMP14": "#E28A00", "NE": "#C8455A", "LASB": "#8A6D3B",
       "FECAL": "#5F6B7A"}
INK, MUTE, GRID = "#1c2430", "#6b7686", "#e6e9ee"

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


def panel_letter(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK)


def oof(pid):
    z = np.load(ROOT / "results" / "oof" / f"{pid}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def cv_summary():
    return pd.read_csv(ROOT / "results" / "cv_summary.csv").set_index("protease")


MODELS = load_all()
