#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MSYS2_ARG_CONV_EXCL='*'
for mode in supplementary_methods supplementary_methods_anon; do
  if [ "$mode" = supplementary_methods_anon ]; then
    input='\def\ANON{}\input{supplementary_methods.tex}'
  else
    input='\input{supplementary_methods.tex}'
  fi
  pdflatex -halt-on-error -interaction=nonstopmode -jobname="$mode" "$input" > /dev/null
  bibtex "$mode" > /dev/null
  pdflatex -halt-on-error -interaction=nonstopmode -jobname="$mode" "$input" > /dev/null
  pdflatex -halt-on-error -interaction=nonstopmode -jobname="$mode" "$input" > /dev/null
done
