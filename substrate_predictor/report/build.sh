#!/usr/bin/env bash
# Build the report: regenerate tables, compile three passes (cross-references, table of contents), clean auxiliaries.
set -e
cd "$(dirname "$0")"
python make_tables.py > /dev/null
for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { grep -E "^!|^l\.[0-9]+" build.log | head -20; echo "pdflatex failed (pass $pass)"; exit 1; }
done
mv main.pdf substrate_predictor_report.pdf
grep -E "Overfull|Underfull|undefined|Undefined" build.log | sort | uniq -c | head -30 || true
rm -f main.aux main.out main.toc main.log
echo "built: $(pwd)/substrate_predictor_report.pdf"
