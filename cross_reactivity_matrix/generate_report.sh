#!/bin/bash
# Compile report_task6.tex to PDF. Run from this directory.
# Requires: pdflatex (TeX Live or MacTeX)

cd "$(dirname "$0")"

echo "Compiling report_task6.tex..."
pdflatex -interaction=nonstopmode report_task6.tex || true
pdflatex -interaction=nonstopmode report_task6.tex || true  # second pass for references

# Clean up auxiliary files
rm -f report_task6.aux report_task6.log report_task6.out report_task6.toc

if [ -f report_task6.pdf ]; then
    echo "Done: report_task6.pdf"
else
    echo "ERROR: PDF was not produced." && exit 1
fi
