"""Extract Table S1 of Yang et al. 2015 (pseudolysin-released elastin peptides) from the supplementary PDF.

Requires `pdftotext` (poppler). Writes data/sources/yang2015_table_s1_peptides.csv and checks every row
(peptide at the stated positions and both flanking residues) against UniProt P04985.
"""
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "yang2015" / "srep09936-s1.pdf"
SEQ = ROOT / "data" / "sources" / "yang2015_P04985_bovine_tropoelastin.fasta"
OUT = ROOT / "data" / "sources" / "yang2015_table_s1_peptides.csv"

lines = subprocess.check_output(["pdftotext", "-layout", str(PDF), "-"], text=True).splitlines()
head = re.compile(r"^\s*(\d+)\s+[\d.]+\s+[\d.]+\s+.*?\(([A-Z])\)([A-Z]+)(\(([A-Z])\))?")
rows, i = [], 0
while i < len(lines):
    m = head.match(lines[i])
    if m:
        idx, left, pep, right, k = int(m.group(1)), m.group(2), m.group(3), m.group(5), i
        if right is None:                       # peptide wrapped onto the next line
            k, cont = i + 1, lines[i + 1].strip()
            a, b = re.fullmatch(r"([A-Z])\)", cont), re.fullmatch(r"([A-Z]*)\(([A-Z])\)", cont)
            if a:
                right = a.group(1)
            elif b:
                pep, right = pep + b.group(1), b.group(2)
            else:
                raise SystemExit(f"unparsed continuation for row {idx}: {cont!r}")
        pos = re.search(r"(\d+)\s*-\s*(\d+)\s*$", lines[i])
        rows.append(dict(index=idx, left_p1=left, peptide=pep, right_p1prime=right,
                         start=int(pos.group(1)), end=int(pos.group(2))))
        i = k
    i += 1

df = pd.DataFrame(rows)
seq = "".join(l.strip() for l in open(SEQ) if not l.startswith(">"))
for r in df.itertuples():
    assert seq[r.start - 1:r.end] == r.peptide and seq[r.start - 2] == r.left_p1 and seq[r.end] == r.right_p1prime, r
df.to_csv(OUT, index=False)
print(f"{len(df)} rows written to {OUT}; all consistent with P04985")
