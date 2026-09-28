#!/usr/bin/env bash
# Live compile: rebuilds the PDF every time a .tex file in tex/ (or numbers/tables) is saved.
# Usage:  bash watch.sh [name]        default name: substrate_predictor_writeup
# Output: report/build/<name>.pdf     (open it once; the viewer refreshes on each rebuild)
NAME="${1:-substrate_predictor_writeup}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEX="$HERE/tex"; OUT="$HERE/build"
mkdir -p "$OUT"

build() {
  ( cd "$TEX" && pdflatex -interaction=nonstopmode -file-line-error -output-directory "$OUT" "$NAME.tex" >"$OUT/$NAME.log.txt" 2>&1
    pdflatex -interaction=nonstopmode -file-line-error -output-directory "$OUT" "$NAME.tex" >>"$OUT/$NAME.log.txt" 2>&1 )
  clear
  if [[ -f "$OUT/$NAME.pdf" ]] && ! grep -q "^!\|:[0-9]*: " "$OUT/$NAME.log.txt"; then
    echo "$(date +%H:%M:%S)  OK  ($(pdfinfo "$OUT/$NAME.pdf" 2>/dev/null | awk '/Pages/{print $2}') pages)"
  else
    echo "$(date +%H:%M:%S)  ERRORS:"; grep -m8 -A2 "^!\|:[0-9]*: " "$OUT/$NAME.log.txt"
  fi
  grep -c "Warning: Reference.*undefined" "$OUT/$NAME.log.txt" | sed 's/^/undefined references: /'
}

stamp() { find "$TEX" "$HERE/numbers.tex" "$HERE/tables" "$HERE/figures" -name '*.tex' -o -name '*.pdf' 2>/dev/null | xargs stat -f %m 2>/dev/null | sort | tail -1; }

build
[[ "$(uname)" == "Darwin" ]] && open "$OUT/$NAME.pdf"
last="$(stamp)"
echo "watching for changes (Ctrl-C to stop) ..."
while sleep 1; do
  now="$(stamp)"
  if [[ "$now" != "$last" ]]; then last="$now"; build; fi
done
