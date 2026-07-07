from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from config import ensure_output_dirs, get_path, load_config, apply_quick_overrides
from data_preprocessing import inverse_transform_target, prepare_dataset
from evaluate import compute_metrics
from models import build_model
from topology_features import compute_topology_features
from utils import model_uses_topology, model_uses_wavelet, safe_name, set_seed


def _callbacks(cfg: Dict[str, Any]):
    training = cfg["training"]
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(training.get("patience", 12)),
            min_delta=float(training.get("min_delta", 1e-6)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=int(training.get("reduce_lr_patience", 6)),
            factor=0.5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]


def train_one(
    cfg: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    seed: int,
    quick: bool = False,
    verbose: int = 1,
) -> Dict[str, Any]:
    set_seed(seed)
    ensure_output_dirs(cfg)

    use_wavelet = model_uses_wavelet(model_name, cfg)
    data = prepare_dataset(cfg, dataset_name, use_wavelet=use_wavelet, quick=quick)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]

    if model_uses_topology(model_name):
        topo_train = compute_topology_features(X_train)
        topo_test = compute_topology_features(X_test)
        model = build_model(
            model_name,
            input_shape=X_train.shape[1:],
            topology_shape=topo_train.shape[1:],
            cfg=cfg,
        )
        fit_x = [X_train, topo_train]
        pred_x = [X_test, topo_test]
    else:
        model = build_model(model_name, input_shape=X_train.shape[1:], cfg=cfg)
        fit_x = X_train
        pred_x = X_test

    batch_size = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    validation_split = float(cfg["training"].get("validation_split", 0.1))

    start_train = time.perf_counter()
    history = model.fit(
        fit_x,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,
        callbacks=_callbacks(cfg),
        verbose=verbose,
    )
    train_seconds = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_pred_scaled = model.predict(pred_x, batch_size=batch_size, verbose=0).reshape(-1)
    inference_seconds = time.perf_counter() - start_pred

    y_pred = inverse_transform_target(data["scaler"], y_pred_scaled)
    y_true = data["y_test_raw"][:len(y_pred)]
    metrics = compute_metrics(y_true, y_pred)

    predictions_dir = get_path(cfg, "predictions_dir")
    model_dir = get_path(cfg, "model_dir")
    log_dir = get_path(cfg, "log_dir")
    for d in [predictions_dir, model_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stem = f"{safe_name(dataset_name)}_{safe_name(model_name)}_seed{seed}"
    pred_df = pd.DataFrame({
        "timestamp": data["test_timestamps"].astype(str).iloc[:len(y_pred)].values,
        "y_true": y_true,
        "y_pred": y_pred,
        "absolute_error": np.abs(y_true - y_pred),
        "percentage_error": np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8) * 100.0,
    })
    pred_path = predictions_dir / f"{stem}_pred.csv"
    pred_df.to_csv(pred_path, index=False)

    pd.DataFrame(history.history).to_csv(log_dir / f"{stem}_history.csv", index=False)

    if cfg["training"].get("save_model", True):
        model.save(model_dir / f"{stem}.keras", include_optimizer=False)

    row = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Seed": seed,
        **metrics,
        "TrainSeconds": train_seconds,
        "InferenceSeconds": inference_seconds,
        "InferenceSecondsPerSample": inference_seconds / max(1, len(y_pred)),
        "TrainWindows": int(len(X_train)),
        "TestWindows": int(len(X_test)),
        "Parameters": int(model.count_params()),
        "PredictionFile": str(pred_path.relative_to(Path(cfg["_project_root"]))),
    }

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)
    row = train_one(cfg, args.dataset, args.model, args.seed, quick=args.quick, verbose=0 if args.quiet else 1)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
