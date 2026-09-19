"""Assemble the self-contained interactive tool: models, tables, structures, 3Dmol.js and the engine are inlined."""
import json, sys
from pathlib import Path
import pandas as pd

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))
from spred.tables import AMINO_ACIDS, VHSE8, SWISSPROT_BACKGROUND

models = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "models").glob("*.json"))}
structs = json.loads((ROOT / "data" / "structures" / "structures.json").read_text())
pdb = {k: (ROOT / "data" / "structures" / "processed" / f"{k}.pdb").read_text() for k in structs}
tables = {"vhse": {a: list(VHSE8[a]) for a in AMINO_ACIDS}, "background": SWISSPROT_BACKGROUND}

kr = pd.read_csv(ROOT / "data" / "validation" / "kridel2001_table_ii.csv")
examples = []
for r in kr.itertuples():
    inp = r.sequence[:r.scissile_after] + "|" + r.sequence[r.scissile_after:]
    note = "no activity" if r.mmp9_status != "measured" else f"{int(r.mmp9_kcat_km):,}"
    title = f"Kridel et al. 2001, Table II: MMP-9 kcat/Km = {'not detected' if r.mmp9_status != 'measured' else f'{int(r.mmp9_kcat_km):,} M-1 s-1'}"
    examples.append({"label": r.peptide, "input": inp, "note": note, "title": title})
examples += [
    {"label": "AAPV|SGIG", "input": "AAPV|SGIG", "note": "NE core", "title": "Neutrophil-elastase sensor core (AAPV motif)"},
    {"label": "PRS|LSGLK", "input": "PRS|LSGLK", "note": "MMP-9 core", "title": "MMP-9 sensor core; bond after the third residue"},
    {"label": "PLG|LAGQS", "input": "PLG|LAGQS", "note": "MMP-2 core", "title": "PLGLAG core; bond after the third residue"},
]

def script_safe(text: str) -> str:
    return text.replace("</script", "<\\/script")

tpl = (WEB / "template.html").read_text()
out = (tpl.replace("__MODELS__", script_safe(json.dumps(models, separators=(",", ":"))))
          .replace("__TABLES__", json.dumps(tables, separators=(",", ":")))
          .replace("__STRUCTS__", json.dumps(structs, separators=(",", ":")))
          .replace("__PDB__", script_safe(json.dumps(pdb, separators=(",", ":"))))
          .replace("__EXTRA__", json.dumps({"examples": examples}, separators=(",", ":")))
          .replace("__ENGINE__", (WEB / "engine.js").read_text())
          .replace("__3DMOL__", (WEB / "vendor" / "3Dmol-min.js").read_text()))
dest = ROOT / "report" / "substrate_predictor_explorer.html"
dest.write_text(out)
print(f"wrote {dest}  ({len(out) / 1e6:.2f} MB)")
