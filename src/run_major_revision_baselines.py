from __future__ import annotations

import argparse
import pandas as pd

from config import apply_quick_overrides, get_path, load_config
from major_revision_utils import save_run_outputs, train_major_revision_one
from utils import print_banner


def main():
    parser = argparse.ArgumentParser(description="Run additional Major Revision SOTA baselines without touching original outputs.")
    parser.add_argument("--config", default="config_major_revision.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)

    datasets = args.datasets or list(cfg["datasets"].keys())
    if args.quick and args.models is None:
        models = cfg.get("quick", {}).get("models", cfg["major_revision"].get("baseline_models", cfg["experiments"]["models"]))
    else:
        models = args.models or cfg["major_revision"].get("baseline_models", cfg["experiments"]["models"])
    seeds = args.seeds or cfg["experiments"].get("default_seeds", [1])

    rows = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                print_banner(f"Major Revision baseline: dataset={dataset}, model={model}, seed={seed}")
                try:
                    row = train_major_revision_one(
                        cfg,
                        dataset,
                        model,
                        seed,
                        quick=args.quick,
                        verbose=0 if args.quiet else 1,
                        use_wavelet=False,
                        experiment_tag="sota_baseline",
                    )
                    rows.append(row)
                    print(pd.Series(row)[["Dataset", "Model", "Seed", "RMSE", "MAE", "MAPE"]].to_string())
                except Exception as exc:
                    print(f"[ERROR] {dataset} | {model} | seed={seed}: {exc}")
                    rows.append({"Dataset": dataset, "Model": model, "Seed": seed, "Error": str(exc)})

    metrics_dir = get_path(cfg, "metrics_dir")
    save_run_outputs(rows, metrics_dir, prefix="sota_baselines")
    print(f"Saved Major Revision baseline metrics to: {metrics_dir}")


if __name__ == "__main__":
    main()
