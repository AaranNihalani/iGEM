#!/usr/bin/env bash
# Compile the electrochemical device modelling report PDF.
set -e
cd "$(dirname "$0")"
echo "Compiling LaTeX (pass 1)..."
pdflatex -interaction=nonstopmode report_device_modelling.tex > /dev/null
echo "Compiling LaTeX (pass 2, cross-refs)..."
pdflatex -interaction=nonstopmode report_device_modelling.tex > /dev/null
rm -f report_device_modelling.aux report_device_modelling.log report_device_modelling.out
echo "Done. Output: report_device_modelling.pdf"
