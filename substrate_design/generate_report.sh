#!/usr/bin/env bash
# Generate all figures then compile the Task 8 report PDF.
set -e
cd "$(dirname "$0")"
echo "Generating figures..."
python3 generate_figures.py
echo "Compiling LaTeX (pass 1)..."
pdflatex -interaction=nonstopmode report_task8.tex > /dev/null
echo "Compiling LaTeX (pass 2, cross-refs)..."
pdflatex -interaction=nonstopmode report_task8.tex > /dev/null
rm -f report_task8.aux report_task8.log report_task8.out
echo "Done. Output: report_task8.pdf"
