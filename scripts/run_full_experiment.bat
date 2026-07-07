@echo off
python src\run_all_experiments.py --config config.yaml --seeds 1 2 3 4 5
python src\run_ablation.py --config config.yaml
python src\run_robustness.py --config config.yaml
python src\run_complexity.py --config config.yaml
python src\statistical_test.py --config config.yaml
python src\plot_figures.py --config config.yaml
