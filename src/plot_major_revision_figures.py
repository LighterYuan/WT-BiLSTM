from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import get_path, load_config


def _save_bar(df, x, y, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(df[x].astype(str), df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_line(df, x, y, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for dataset, g in df.groupby("Dataset"):
        ax.plot(g[x].astype(str), g[y], marker="o", label=dataset)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Major Revision experiment figures from CSV outputs.")
    parser.add_argument("--config", default="config_major_revision.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics_dir = get_path(cfg, "metrics_dir")
    figures_dir = get_path(cfg, "figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)

    sota_summary = metrics_dir / "sota_baselines_summary.csv"
    if sota_summary.exists():
        df = pd.read_csv(sota_summary)
        for dataset, g in df.groupby("Dataset"):
            _save_bar(g.sort_values("RMSE_mean"), "Model", "RMSE_mean", f"Additional baseline RMSE on {dataset}", "RMSE", figures_dir / f"mr_sota_rmse_{dataset}.png")

    sens = metrics_dir / "sensitivity_grouped_summary.csv"
    if sens.exists():
        df = pd.read_csv(sens)
        for exp, g in df.groupby("SensitivityExperiment"):
            _save_line(g.sort_values(["Dataset", "SensitivityValue"]), "SensitivityValue", "RMSE_mean", f"Sensitivity of {exp}", "RMSE", figures_dir / f"mr_sensitivity_{exp}_rmse.png")

    abl = metrics_dir / "denoising_ablation_summary.csv"
    if abl.exists():
        df = pd.read_csv(abl)
        for dataset, g in df.groupby("Dataset"):
            _save_bar(g.sort_values("RMSE_mean"), "Model", "RMSE_mean", f"Denoising ablation RMSE on {dataset}", "RMSE", figures_dir / f"mr_ablation_rmse_{dataset}.png")

    print(f"Saved Major Revision figures to: {figures_dir}")


if __name__ == "__main__":
    main()
