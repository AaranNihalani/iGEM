"""Shared constants: sequence window, protease registry."""
from dataclasses import dataclass

from .tables import AMINO_ACIDS

WINDOW = 8
POSITIONS = ("P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'")
VHSE_NAMES = tuple(f"VHSE{i}" for i in range(1, 9))
VHSE_GROUPS = {"hydrophobic": (0, 1), "steric": (2, 3), "electronic": (4, 5, 6, 7)}
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

RANDOM_SEED = 42
MIN_TRAINABLE_POSITIVES = 50   # models below this are excluded from panels


@dataclass(frozen=True)
class Protease:
    id: str
    name: str
    merops: tuple          # MEROPS peptidase identifiers (Substrate_search column 2)
    family: str


PROTEASES = {p.id: p for p in (
    Protease("MMP2", "MMP-2", ("M10.003",), "metallo (M10)"),
    Protease("MMP8", "MMP-8", ("M10.002",), "metallo (M10)"),
    Protease("MMP9", "MMP-9", ("M10.004",), "metallo (M10)"),
    Protease("MMP14", "MMP-14", ("M10.014",), "metallo (M10)"),
    Protease("NE", "Neutrophil elastase", ("S01.131",), "serine (S01)"),
    Protease("LASB", "LasB (pseudolysin)", ("M04.005",), "metallo (M04)"),
    Protease("FECAL", "Faecal serine proteases", ("S01.151", "S01.152", "S01.153"), "serine (S01)"),
)}

# Registered but not trainable: too few confirmed cleavage windows in MEROPS.
UNTRAINABLE = {"FAP": Protease("FAP", "Fibroblast activation protein", ("S09.007",), "serine (S09)")}
