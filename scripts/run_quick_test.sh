#!/usr/bin/env bash
set -e

python src/run_all_experiments.py --config config.yaml --quick --seeds 1
python src/run_ablation.py --config config.yaml || true
python src/run_robustness.py --config config.yaml || true
python src/run_complexity.py --config config.yaml
python src/plot_figures.py --config config.yaml
