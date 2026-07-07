from __future__ import annotations

import argparse

import pandas as pd

from config import apply_quick_overrides, ensure_output_dirs, get_path, load_config
from evaluate import save_overall_tables
from train import train_one
from utils import print_banner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)
    ensure_output_dirs(cfg)

    datasets = args.datasets or list(cfg["datasets"].keys())
    if args.quick and args.models is None:
        models = cfg["quick"].get("models", cfg["experiments"]["models"])
    else:
        models = args.models or cfg["experiments"]["models"]
    seeds = args.seeds or cfg["experiments"].get("default_seeds", [1])

    rows = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                print_banner(f"Running dataset={dataset}, model={model}, seed={seed}")
                try:
                    row = train_one(cfg, dataset, model, seed, quick=args.quick, verbose=0 if args.quiet else 1)
                    rows.append(row)
                    print(pd.Series(row)[["Dataset", "Model", "Seed", "RMSE", "MAE", "MAPE"]].to_string())
                except Exception as exc:
                    print(f"[ERROR] {dataset} | {model} | seed={seed}: {exc}")
                    rows.append({
                        "Dataset": dataset,
                        "Model": model,
                        "Seed": seed,
                        "Error": str(exc),
                    })

    metrics_dir = get_path(cfg, "metrics_dir")
    pd.DataFrame(rows).to_csv(metrics_dir / "run_metrics_raw_with_errors.csv", index=False)
    ok_rows = [r for r in rows if "RMSE" in r]
    save_overall_tables(ok_rows, metrics_dir)
    print(f"\nSaved metrics to: {metrics_dir}")


if __name__ == "__main__":
    main()
