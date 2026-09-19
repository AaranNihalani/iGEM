"""Reference outputs from the Python package for testing the JavaScript engine."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from spred import candidates
from spred.model import load_all
from spred.panel import Panel, PRESETS
from spred.tables import AMINO_ACIDS, VHSE8, SWISSPROT_BACKGROUND

models = load_all()
panel = Panel(PRESETS["all"], models)
rng = np.random.default_rng(7)
full = candidates.random_pool(150, seed=99) + ["IPRTLTAA", "AAPVSGIG", "GPLGIAGQ", "PRSLSGLK"]
gapped = ["-PRSLSGL", "-PLGLAGQ", "IPRTLTA-", "--SGLKAL", "AAPV----".replace("----", "SG--"), "-PRS-SGL", "GPRQITA-", "QPHYLTTA"]
fx = {"full": {}, "gapped": {}}
sc = panel.score(full)
for m in panel.members:
    fx["full"][m] = {"P": sc[f"P_{m}"].tolist(), "pct_lr": sc[f"pct_lr_{m}"].tolist(), "pct_pssm": sc[f"pct_pssm_{m}"].tolist(), "u": sc[f"u_{m}"].tolist()}
fx["full_windows"] = full
gs = panel.score_gapped(gapped)
fx["gapped_windows"] = gapped
for m in panel.members:
    fx["gapped"][m] = {"P": gs[f"P_{m}"].tolist(), "pct_lr": gs[f"pct_lr_{m}"].tolist(), "pct_pssm": gs[f"pct_pssm_{m}"].tolist(), "u": gs[f"u_{m}"].tolist()}
fx["contrib"] = {m: {w: models[m].contributions(w).tolist() for w in ["IPRTLTAA", "AAPVSGIG"]} for m in panel.members}
json.dump(fx, open(Path(__file__).parent / "fixture.json", "w"))
json.dump({"vhse": {a: list(VHSE8[a]) for a in AMINO_ACIDS}, "background": SWISSPROT_BACKGROUND}, open(Path(__file__).parent / "tables.json", "w"))
print("fixture written:", len(full), "full,", len(gapped), "gapped")
