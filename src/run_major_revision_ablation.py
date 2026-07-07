from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import apply_quick_overrides, get_path, load_config
from major_revision_utils import WaveletParams, prepare_dataset_mr, save_run_outputs, train_major_revision_one
from evaluate import compute_metrics
from utils import print_banner, set_seed
from models import build_model
from data_preprocessing import inverse_transform_target


def moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def prepare_smoothed_dataset(cfg, dataset_name, method, time_steps, quick=False):
    # This function uses prepare_dataset_mr to get leakage-safe split/scaling, then replaces the scaled signal
    # with train/test separately smoothed alternatives, avoiding train-test mixing.
    base = prepare_dataset_mr(cfg, dataset_name, use_wavelet=False, time_steps=time_steps, quick=quick, save_npz=False, experiment_tag="smoothing_ablation")
    train_scaled = base["scaler"].transform(base["train_df"][["traffic"]].values).reshape(-1)
    test_scaled = base["scaler"].transform(base["test_df"][["traffic"]].values).reshape(-1)

    if method == "moving_average_3":
        train_signal, test_signal = moving_average(train_scaled, 3), moving_average(test_scaled, 3)
    elif method == "moving_average_5":
        train_signal, test_signal = moving_average(train_scaled, 5), moving_average(test_scaled, 5)
    elif method == "savgol_5_2":
        # Savitzky-Golay is applied separately to train and test segments to avoid future leakage across split.
        train_signal = savgol_filter(train_scaled, window_length=5, polyorder=2, mode="nearest") if len(train_scaled) >= 5 else train_scaled
        test_signal = savgol_filter(test_scaled, window_length=5, polyorder=2, mode="nearest") if len(test_scaled) >= 5 else test_scaled
    else:
        raise ValueError(method)

    from data_preprocessing import create_windows
    X_train, y_train = create_windows(train_signal, time_steps=time_steps)
    context = train_signal[-time_steps:]
    test_with_context = np.concatenate([context, test_signal], axis=0)
    X_test, y_test = create_windows(test_with_context, time_steps=time_steps)
    y_test_raw = base["test_df"]["traffic"].iloc[:len(y_test)].to_numpy(dtype=float)
    return {**base, "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "y_test_raw": y_test_raw}


def train_smoothed_bilstm(cfg, dataset_name, seed, method, quick=False, quiet=False):
    set_seed(seed)
    time_steps = int(cfg["preprocessing"].get("time_steps", 10))
    data = prepare_smoothed_dataset(cfg, dataset_name, method=method, time_steps=time_steps, quick=quick)
    model = build_model("BiLSTM", input_shape=data["X_train"].shape[1:], cfg=cfg)
    callbacks = []
    import tensorflow as tf
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(cfg["training"].get("patience", 12)), restore_best_weights=True))
    history = model.fit(
        data["X_train"], data["y_train"], epochs=int(cfg["training"].get("epochs", 80)), batch_size=int(cfg["training"].get("batch_size", 64)),
        validation_split=float(cfg["training"].get("validation_split", 0.1)), shuffle=False, callbacks=callbacks, verbose=0 if quiet else 1,
    )
    y_pred_scaled = model.predict(data["X_test"], batch_size=int(cfg["training"].get("batch_size", 64)), verbose=0).reshape(-1)
    y_pred = inverse_transform_target(data["scaler"], y_pred_scaled)
    y_true = data["y_test_raw"][:len(y_pred)]
    metrics = compute_metrics(y_true, y_pred)
    return {
        "ExperimentTag": "denoising_ablation",
        "Dataset": dataset_name,
        "Model": f"{method}-BiLSTM",
        "Seed": seed,
        **metrics,
        "Parameters": int(model.count_params()),
        "TimeSteps": time_steps,
        "UseWavelet": False,
        "AblationGroup": "alternative_smoothing",
        "AblationVariant": method,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Major Revision denoising ablations without touching original outputs.")
    parser.add_argument("--config", default="config_major_revision.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)
    datasets = args.datasets or list(cfg["datasets"].keys())
    seeds = args.seeds or cfg.get("experiments", {}).get("default_seeds", [1])

    rows = []
    for dataset in datasets:
        for seed in seeds:
            # Raw BiLSTM baseline under the same Major Revision output protocol.
            print_banner(f"Ablation raw BiLSTM: dataset={dataset}, seed={seed}")
            try:
                row = train_major_revision_one(cfg, dataset, "BiLSTM", seed, quick=args.quick, verbose=0 if args.quiet else 1, use_wavelet=False, experiment_tag="denoising_ablation", extra_metadata={"AblationGroup": "raw_vs_denoised", "AblationVariant": "raw"})
                rows.append(row)
            except Exception as exc:
                rows.append({"Dataset": dataset, "Seed": seed, "Model": "BiLSTM", "Error": str(exc)})
                print(f"[ERROR] {exc}")

            print_banner(f"Ablation WT-BiLSTM: dataset={dataset}, seed={seed}")
            try:
                params = WaveletParams(wavelet=cfg["wavelet"].get("wavelet", "db8"), level=int(cfg["wavelet"].get("level", 3)), threshold_mode=cfg["wavelet"].get("threshold_mode", "soft"), threshold_strategy=cfg["wavelet"].get("threshold_strategy", "universal"), threshold_scale=float(cfg["wavelet"].get("threshold_scale", 1.0)))
                row = train_major_revision_one(cfg, dataset, "WT-BiLSTM", seed, quick=args.quick, verbose=0 if args.quiet else 1, use_wavelet=True, wavelet_params=params, experiment_tag="denoising_ablation", extra_metadata={"AblationGroup": "raw_vs_denoised", "AblationVariant": "wavelet"})
                rows.append(row)
            except Exception as exc:
                rows.append({"Dataset": dataset, "Seed": seed, "Model": "WT-BiLSTM", "Error": str(exc)})
                print(f"[ERROR] {exc}")

            for method in ["moving_average_3", "moving_average_5", "savgol_5_2"]:
                print_banner(f"Ablation {method}-BiLSTM: dataset={dataset}, seed={seed}")
                try:
                    rows.append(train_smoothed_bilstm(cfg, dataset, seed, method, quick=args.quick, quiet=args.quiet))
                except Exception as exc:
                    rows.append({"Dataset": dataset, "Seed": seed, "Model": f"{method}-BiLSTM", "Error": str(exc)})
                    print(f"[ERROR] {exc}")

    metrics_dir = get_path(cfg, "metrics_dir")
    save_run_outputs(rows, metrics_dir, prefix="denoising_ablation")
    print(f"Saved Major Revision ablation metrics to: {metrics_dir}")


if __name__ == "__main__":
    main()
