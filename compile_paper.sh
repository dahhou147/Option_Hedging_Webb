#!/bin/bash
# Script to compile the LaTeX paper

echo "Generating figures..."
#python generate_paper_figures.py

echo "Compiling LaTeX paper..."
pdflatex paper.tex
bibtex paper  # If using bibliography
pdflatex paper.tex
pdflatex paper.tex

echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.pdf 

echo "Done! Paper compiled as paper.pdf"

