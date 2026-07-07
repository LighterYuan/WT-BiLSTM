@echo off
cd /d %~dp0\..
python src\run_major_revision_baselines.py --config config_major_revision.yaml --seeds 1 2 3 4 5 --quiet
python src\run_major_revision_sensitivity.py --config config_major_revision.yaml --seeds 1 2 3 --quiet
python src\run_major_revision_ablation.py --config config_major_revision.yaml --seeds 1 2 3 4 5 --quiet
python src\plot_major_revision_figures.py --config config_major_revision.yaml
