#!/bin/bash
# Compile report_task7.tex to PDF. Run from this directory.
# Requires: pdflatex (TeX Live or MacTeX)

cd "$(dirname "$0")"

echo "Compiling report_task7.tex..."
pdflatex -interaction=nonstopmode report_task7.tex || true
pdflatex -interaction=nonstopmode report_task7.tex || true  # second pass for references

# Clean up auxiliary files
rm -f report_task7.aux report_task7.log report_task7.out report_task7.toc

if [ -f report_task7.pdf ]; then
    echo "Done: report_task7.pdf"
else
    echo "ERROR: PDF was not produced." && exit 1
fi
