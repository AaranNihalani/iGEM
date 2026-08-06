#!/usr/bin/env bash
# Usage: ./generate_report.sh <name>
#   <name> is the .tex filename (without extension) inside the tex/ folder.
#   Example: ./generate_report.sh report_device_modelling
#            ./generate_report.sh flowchart
set -e

REPORTS_DIR="$(cd "$(dirname "$0")" && pwd)"
TEX_DIR="$REPORTS_DIR/tex"

if [[ -z "$1" ]]; then
  echo "Usage: $0 <tex-filename-without-extension>"
  echo "Available:"
  ls "$TEX_DIR"/*.tex | xargs -n1 basename | sed 's/\.tex$/   -->  ..\/tex\/&/'
  exit 1
fi

NAME="$1"
TEX_FILE="$TEX_DIR/${NAME}.tex"
PDF_OUT="$REPORTS_DIR/${NAME}.pdf"

if [[ ! -f "$TEX_FILE" ]]; then
  echo "Error: $TEX_FILE not found."
  exit 1
fi

echo "Compiling $NAME (pass 1)..."
pdflatex -interaction=nonstopmode -output-directory "$TEX_DIR" "$TEX_FILE" > /dev/null
echo "Compiling $NAME (pass 2, cross-refs)..."
pdflatex -interaction=nonstopmode -output-directory "$TEX_DIR" "$TEX_FILE" > /dev/null

mv "$TEX_DIR/${NAME}.pdf" "$PDF_OUT"
rm -f "$TEX_DIR/${NAME}.aux" "$TEX_DIR/${NAME}.log" \
      "$TEX_DIR/${NAME}.out" "$TEX_DIR/${NAME}.synctex.gz"

echo "Done. Output: $PDF_OUT"
