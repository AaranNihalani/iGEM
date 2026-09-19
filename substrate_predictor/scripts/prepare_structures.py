"""Prepare the five protease structures for the report and the browser tool.

For every structure: keep the protease chain (+ bound peptide-like ligand chain, catalytic metals, small-molecule
inhibitor), drop waters and other chains, and derive active-site facts from the coordinates:
  * catalytic zinc and the residues that coordinate it (< 2.7 A),
  * pocket residues (any heavy atom within 6 A of the bound ligand, or within 9 A of the catalytic zinc when no
    ligand is present).
Writes data/structures/processed/<PROTEASE>.pdb and data/structures/structures.json.
"""
import json
from pathlib import Path

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "structures"
OUT = SRC / "processed"
OUT.mkdir(exist_ok=True)

ENTRIES = {
    "MMP9": dict(pdb="1GKC", chains=["A"], ligand_resnames={"NFH"}, lig_chain=None, title="MMP-9 catalytic domain with a reverse hydroxamate inhibitor",
                 citation="Rowsell et al. (2002) J Mol Biol 319:173", resolution=2.3, catalytic_metal="ZN"),
    "MMP2": dict(pdb="1QIB", chains=["A"], ligand_resnames=set(), lig_chain=None, title="MMP-2 (gelatinase A) catalytic domain",
                 citation="RCSB 1QIB", resolution=2.8, catalytic_metal="ZN"),
    "MMP8": dict(pdb="1JAN", chains=["A", "I"], ligand_resnames=set(), lig_chain="I", title="MMP-8 catalytic domain with Pro-Leu-Gly-hydroxylamine",
                 citation="RCSB 1JAN", resolution=2.5, catalytic_metal="ZN"),
    "NE": dict(pdb="1HNE", chains=["E", "I"], ligand_resnames=set(), lig_chain="I", title="Human neutrophil elastase with MeOSuc-Ala-Ala-Pro-Ala-chloromethyl ketone",
               citation="RCSB 1HNE", resolution=1.84, catalytic_metal=None),
    "LASB": dict(pdb="1EZM", chains=["A"], ligand_resnames=set(), lig_chain=None, title="Pseudomonas aeruginosa elastase (pseudolysin, LasB)",
                 citation="RCSB 1EZM", resolution=1.5, catalytic_metal="ZN"),
}
KEEP_HET = {"ZN", "CA"}


def main():
    parser = PDBParser(QUIET=True)
    meta = {}
    for pid, e in ENTRIES.items():
        s = parser.get_structure(pid, SRC / f"{e['pdb']}.pdb")[0]
        atoms, ligand_atoms, prot_atoms = [], [], []
        zincs = []
        keep_lines = []
        for chain in s:
            if chain.id not in e["chains"]:
                continue
            for res in chain:
                het = res.id[0].strip()
                name = res.get_resname().strip()
                if het == "W":
                    continue
                is_lig = name in e["ligand_resnames"] or chain.id == e["lig_chain"]
                if het.startswith("H") and name not in KEEP_HET and not is_lig:
                    continue
                for a in res:
                    atoms.append(a)
                    if is_lig:
                        ligand_atoms.append(a)
                    elif het == "":
                        prot_atoms.append(a)
                    if name == "ZN":
                        zincs.append(a)
        ns = NeighborSearch(prot_atoms)
        # catalytic zinc: the zinc with the most protein N/O ligands within 2.7 A (MMPs / LasB)
        cat = None; coord = []
        if e["catalytic_metal"] and zincs:
            best = None
            for z in zincs:
                lig = {(a.get_parent().get_resname(), a.get_parent().id[1]) for a in ns.search(z.coord, 2.7) if a.element in ("N", "O")}
                his = sorted(n for r, n in lig if r == "HIS")
                has_hexxh = any(b - a_ == 4 for a_ in his for b in his)
                if has_hexxh:
                    best = (z, lig)
            if best is None:
                raise RuntimeError(f"{pid}: no zinc with an HExxH histidine pair")
            cat, coord = best
            coord = [(r, n) for r, n in sorted(coord, key=lambda x: x[1])]
        if pid == "NE":
            # catalytic triad (chymotrypsin numbering in 1HNE): His57, Asp102, Ser195
            coord = []
            for res in s["E"]:
                if res.id[1] in (57, 102, 195) and res.id[0] == " ":
                    coord.append((res.get_resname(), res.id[1], "triad"))
        # pocket residues
        if ligand_atoms:
            near = NeighborSearch(prot_atoms)
            pocket = {(a.get_parent().get_resname(), a.get_parent().id[1])
                      for l in ligand_atoms for a in near.search(l.coord, 6.0)}
        elif cat is not None:
            pocket = {(a.get_parent().get_resname(), a.get_parent().id[1]) for a in ns.search(cat.coord, 9.0)}
        else:
            pocket = set()
        with open(OUT / f"{pid}.pdb", "w") as fh:
            for chain in s:
                if chain.id not in e["chains"]:
                    continue
                for res in chain:
                    het = res.id[0].strip(); name = res.get_resname().strip()
                    if het == "W":
                        continue
                    is_lig = name in e["ligand_resnames"] or chain.id == e["lig_chain"]
                    if het.startswith("H") and name not in KEEP_HET and not is_lig:
                        continue
                    for a in res:
                        rec = "HETATM" if (het.startswith("H") or is_lig) else "ATOM  "
                        fh.write(f"{rec}{a.serial_number:5d} {a.fullname:4s} {name:>3s} {chain.id}{res.id[1]:4d}    "
                                 f"{a.coord[0]:8.3f}{a.coord[1]:8.3f}{a.coord[2]:8.3f}{a.occupancy:6.2f}{a.bfactor:6.2f}          {a.element:>2s}\n")
            fh.write("END\n")
        meta[pid] = {"pdb": e["pdb"], "title": e["title"], "citation": e["citation"], "resolution": e["resolution"],
                     "chains": e["chains"], "n_protein_atoms": len(prot_atoms),
                     "n_residues": len({(a.get_parent().get_parent().id, a.get_parent().id[1]) for a in prot_atoms}),
                     "catalytic_metal_ligands": sorted({f"{r}{n}" for r, n, *_ in
                                                        (sorted(coord) if pid == "NE" else
                                                         [(x[0], x[1]) for x in coord])}),
                     "catalytic_zinc_coord": None if cat is None else [round(float(c), 3) for c in cat.coord],
                     "pocket_residues": sorted({f"{r}{n}" for r, n in pocket}, key=lambda x: int("".join(ch for ch in x if ch.isdigit()))),
                     "ligand": sorted(e["ligand_resnames"]) or ["chain " + str(e["lig_chain"])] if e["lig_chain"] else sorted(e["ligand_resnames"])}
        print(pid, meta[pid]["pdb"], "residues", meta[pid]["n_residues"], "| catalytic ligands:", meta[pid]["catalytic_metal_ligands"],
              "| pocket:", len(pocket))
    (SRC / "structures.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
