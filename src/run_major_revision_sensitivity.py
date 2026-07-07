from __future__ import annotations

import argparse
import pandas as pd

from config import apply_quick_overrides, get_path, load_config
from major_revision_utils import WaveletParams, save_run_outputs, train_major_revision_one
from utils import print_banner


def _run_case(cfg, dataset, seed, time_steps, params, experiment_name, variable_name, variable_value, model_name, quick, quiet):
    return train_major_revision_one(
        cfg,
        dataset,
        model_name,
        seed,
        quick=quick,
        verbose=0 if quiet else 1,
        use_wavelet=True,
        wavelet_params=params,
        time_steps=time_steps,
        experiment_tag=f"sensitivity_{experiment_name}_{variable_value}",
        extra_metadata={
            "SensitivityExperiment": experiment_name,
            "SensitivityVariable": variable_name,
            "SensitivityValue": str(variable_value),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run Major Revision sensitivity experiments without touching original outputs.")
    parser.add_argument("--config", default="config_major_revision.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--experiments", nargs="*", default=None, choices=["window", "basis", "level", "mode", "strategy", "scale"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)

    mr = cfg.get("major_revision", {})
    datasets = args.datasets or list(cfg["datasets"].keys())
    seeds = args.seeds or mr.get("sensitivity_seeds", cfg["experiments"].get("default_seeds", [1]))
    model_name = args.model or mr.get("sensitivity_model", "WT-BiLSTM")
    experiments = args.experiments or ["window", "basis", "level", "mode", "strategy", "scale"]

    base_w = cfg.get("wavelet", {})
    base_params = WaveletParams(
        wavelet=base_w.get("wavelet", "db8"),
        level=int(base_w.get("level", 3)),
        threshold_mode=base_w.get("threshold_mode", "soft"),
        threshold_strategy=base_w.get("threshold_strategy", "universal"),
        threshold_scale=float(base_w.get("threshold_scale", 1.0)),
    )
    base_time_steps = int(cfg["preprocessing"].get("time_steps", 10))

    rows = []
    for dataset in datasets:
        for seed in seeds:
            if "window" in experiments:
                for value in mr.get("window_lengths", [5, 8, 10, 12, 15, 20]):
                    print_banner(f"Sensitivity window: dataset={dataset}, seed={seed}, time_steps={value}")
                    try:
                        rows.append(_run_case(cfg, dataset, seed, int(value), base_params, "window", "time_steps", value, model_name, args.quick, args.quiet))
                    except Exception as exc:
                        rows.append({"Dataset": dataset, "Seed": seed, "SensitivityExperiment": "window", "SensitivityValue": str(value), "Error": str(exc)})
                        print(f"[ERROR] {exc}")

            if "basis" in experiments:
                for value in mr.get("wavelet_bases", ["haar", "db2", "db4", "db6", "db8", "sym4", "coif3"]):
                    params = WaveletParams(wavelet=value, level=base_params.level, threshold_mode=base_params.threshold_mode, threshold_strategy=base_params.threshold_strategy, threshold_scale=base_params.threshold_scale)
                    print_banner(f"Sensitivity basis: dataset={dataset}, seed={seed}, wavelet={value}")
                    try:
                        rows.append(_run_case(cfg, dataset, seed, base_time_steps, params, "basis", "wavelet", value, model_name, args.quick, args.quiet))
                    except Exception as exc:
                        rows.append({"Dataset": dataset, "Seed": seed, "SensitivityExperiment": "basis", "SensitivityValue": str(value), "Error": str(exc)})
                        print(f"[ERROR] {exc}")

            if "level" in experiments:
                for value in mr.get("decomposition_levels", [1, 2, 3, 4, 5]):
                    params = WaveletParams(wavelet=base_params.wavelet, level=int(value), threshold_mode=base_params.threshold_mode, threshold_strategy=base_params.threshold_strategy, threshold_scale=base_params.threshold_scale)
                    print_banner(f"Sensitivity level: dataset={dataset}, seed={seed}, level={value}")
                    try:
                        rows.append(_run_case(cfg, dataset, seed, base_time_steps, params, "level", "level", value, model_name, args.quick, args.quiet))
                    except Exception as exc:
                        rows.append({"Dataset": dataset, "Seed": seed, "SensitivityExperiment": "level", "SensitivityValue": str(value), "Error": str(exc)})
                        print(f"[ERROR] {exc}")

            if "mode" in experiments:
                for value in mr.get("threshold_modes", ["soft", "hard", "garrote"]):
                    params = WaveletParams(wavelet=base_params.wavelet, level=base_params.level, threshold_mode=value, threshold_strategy=base_params.threshold_strategy, threshold_scale=base_params.threshold_scale)
                    print_banner(f"Sensitivity mode: dataset={dataset}, seed={seed}, threshold_mode={value}")
                    try:
                        rows.append(_run_case(cfg, dataset, seed, base_time_steps, params, "mode", "threshold_mode", value, model_name, args.quick, args.quiet))
                    except Exception as exc:
                        rows.append({"Dataset": dataset, "Seed": seed, "SensitivityExperiment": "mode", "SensitivityValue": str(value), "Error": str(exc)})
                        print(f"[ERROR] {exc}")

            if "strategy" in experiments:
                for value in mr.get("threshold_strategies", ["universal", "minimax", "sure", "bayes"]):
                    params = WaveletParams(wavelet=base_params.wavelet, level=base_params.level, threshold_mode=base_params.threshold_mode, threshold_strategy=value, threshold_scale=base_params.threshold_scale)
                    print_banner(f"Sensitivity strategy: dataset={dataset}, seed={seed}, threshold_strategy={value}")
                    try:
                        rows.append(_run_case(cfg, dataset, seed, base_time_steps, params, "strategy", "threshold_strategy", value, model_name, args.quick, args.quiet))
                    except Exception as exc:
                        rows.append({"Dataset": dataset, "Seed": seed, "SensitivityExperiment": "strategy", "SensitivityValue": str(value), "Error": str(exc)})
                        print(f"[ERROR] {exc}")

            if "scale" in experiments:
                for value in mr.get("threshold_scales", [0.5, 1.0, 1.5, 2.0]):
                    params = WaveletParams(wavelet=base_params.wavelet, level=base_params.level, threshold_mode=base_params.threshold_mode, threshold_strategy=base_params.threshold_strategy, threshold_scale=float(value))
                    print_banner(f"Sensitivity scale: dataset={dataset}, seed={seed}, threshold_scale={value}")
                    try:
                        rows.append(_run_case(cfg, dataset, seed, base_time_steps, params, "scale", "threshold_scale", value, model_name, args.quick, args.quiet))
                    except Exception as exc:
                        rows.append({"Dataset": dataset, "Seed": seed, "SensitivityExperiment": "scale", "SensitivityValue": str(value), "Error": str(exc)})
                        print(f"[ERROR] {exc}")

    metrics_dir = get_path(cfg, "metrics_dir")
    save_run_outputs(rows, metrics_dir, prefix="sensitivity")
    if rows:
        df = pd.DataFrame(rows)
        ok = df.dropna(subset=["RMSE", "MAE", "MAPE"], how="any") if all(c in df.columns for c in ["RMSE", "MAE", "MAPE"]) else pd.DataFrame()
        if not ok.empty:
            detail_summary = ok.groupby(["Dataset", "SensitivityExperiment", "SensitivityVariable", "SensitivityValue"], dropna=False).agg(
                Runs=("RMSE", "count"),
                RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
            ).reset_index()
            detail_summary.to_csv(metrics_dir / "sensitivity_grouped_summary.csv", index=False)
    print(f"Saved Major Revision sensitivity metrics to: {metrics_dir}")


if __name__ == "__main__":
    main()
