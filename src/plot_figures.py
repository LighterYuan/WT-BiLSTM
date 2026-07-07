from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import get_path, load_config


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_raw_sequences(cfg):
    raw_dir = get_path(cfg, "raw_data_dir")
    figures_dir = get_path(cfg, "figures_dir")
    for dataset, dcfg in cfg["datasets"].items():
        df = pd.read_csv(raw_dir / dcfg["file"])
        y = pd.to_numeric(df[dcfg["target_col"]], errors="coerce")
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(y)), y)
        plt.title(f"{dataset} raw traffic sequence")
        plt.xlabel("Time index")
        plt.ylabel("Traffic")
        savefig(figures_dir / f"{dataset}_raw_sequence.png")


def plot_wavelet_components(cfg):
    wavelet_dir = get_path(cfg, "wavelet_dir")
    figures_dir = get_path(cfg, "figures_dir")
    for path in wavelet_dir.glob("*_train_wavelet_components.csv"):
        df = pd.read_csv(path)
        dataset = path.name.replace("_train_wavelet_components.csv", "")
        for col in [c for c in df.columns if c in ["original", "denoised", "A", "D1", "D2", "D3"]]:
            plt.figure(figsize=(10, 3))
            plt.plot(df[col].values)
            plt.title(f"{dataset} wavelet component: {col}")
            plt.xlabel("Time index")
            plt.ylabel("Scaled traffic")
            savefig(figures_dir / f"{dataset}_wavelet_{col}.png")


def plot_overall_metrics(cfg):
    metrics_dir = get_path(cfg, "metrics_dir")
    figures_dir = get_path(cfg, "figures_dir")
    path = metrics_dir / "overall_metrics.csv"
    if not path.exists():
        print("[WARN] overall_metrics.csv not found, skip metric plots.")
        return
    df = pd.read_csv(path)
    for dataset in df["Dataset"].unique():
        sub = df[df["Dataset"] == dataset]
        for metric in ["RMSE", "MAE", "MAPE"]:
            plt.figure(figsize=(10, 4))
            plt.bar(sub["Model"], sub[metric])
            plt.xticks(rotation=30, ha="right")
            plt.title(f"{dataset} {metric} comparison")
            plt.xlabel("Model")
            plt.ylabel(metric)
            savefig(figures_dir / f"{dataset}_{metric}_comparison.png")


def plot_predictions(cfg):
    predictions_dir = get_path(cfg, "predictions_dir")
    figures_dir = get_path(cfg, "figures_dir")
    for dataset in cfg["datasets"]:
        files = sorted(predictions_dir.glob(f"{dataset}_WT-TBiLSTM_seed*_pred.csv"))
        if not files:
            files = sorted(predictions_dir.glob(f"{dataset}_*_seed*_pred.csv"))
        if not files:
            continue
        df = pd.read_csv(files[0])
        plt.figure(figsize=(10, 4))
        plt.plot(df["y_true"].values, label="Actual")
        plt.plot(df["y_pred"].values, label="Predicted")
        plt.title(f"{dataset} prediction curve: {files[0].stem}")
        plt.xlabel("Test sample index")
        plt.ylabel("Traffic")
        plt.legend()
        savefig(figures_dir / f"{dataset}_prediction_curve.png")


def plot_ablation(cfg):
    metrics_dir = get_path(cfg, "metrics_dir")
    figures_dir = get_path(cfg, "figures_dir")
    for filename in ["wavelet_ablation.csv", "topology_ablation.csv"]:
        path = metrics_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for metric in ["RMSE_Reduction_%", "MAE_Reduction_%", "MAPE_Reduction_%"]:
            if metric not in df.columns:
                continue
            plt.figure(figsize=(8, 4))
            labels = df["Dataset"] + ": " + df["Baseline"] + "→" + df["Improved"]
            plt.bar(labels, df[metric])
            plt.xticks(rotation=25, ha="right")
            plt.title(f"{filename.replace('.csv','')} {metric}")
            plt.xlabel("Comparison")
            plt.ylabel("Reduction (%)")
            savefig(figures_dir / f"{filename.replace('.csv','')}_{metric}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    plot_raw_sequences(cfg)
    plot_wavelet_components(cfg)
    plot_overall_metrics(cfg)
    plot_predictions(cfg)
    plot_ablation(cfg)
    print(f"Figures saved to {get_path(cfg, 'figures_dir')}")


if __name__ == "__main__":
    main()
