# Retention Sensitivity: An Empirical Study of Compression Behavior in Tabular Models

## Overview
This repository contains the code, data, and results for the paper 
"Retention Sensitivity: An Empirical Study of Compression Behavior 
in Tabular Models".

## Repository Structure
- `data/` — dataset loader and preprocessing utilities
- `experiments/` — HPC experiment scripts
- `results/` — raw CSV results for all 5,625 configurations
- `analysis/` — Jupyter notebook for generating all tables and figures

## Requirements
Install dependencies with:
\```
pip install -r requirements.txt
\```

## Reproducing Results
1. Run experiments: `python experiments/run_experiments.py`
2. Open `analysis/analysis.ipynb` to reproduce all tables and figures
