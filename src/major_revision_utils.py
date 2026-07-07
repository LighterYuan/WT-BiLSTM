from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import pywt
import tensorflow as tf

from config import ensure_output_dirs, get_path
from data_preprocessing import create_windows, inverse_transform_target, load_raw_dataset, make_scaler
from evaluate import compute_metrics, summarize_runs
from models import build_model as build_original_model
from models_major_revision import build_major_revision_model
from topology_features import compute_topology_features
from utils import model_uses_topology, safe_name, set_seed


ORIGINAL_MODEL_NAMES = {"GRU", "LSTM", "WT-LSTM", "BILSTM", "WT-BILSTM", "CNN-LSTM", "WT-TBILSTM"}
MR_MODEL_NAMES = {"TCN", "TRANSFORMER", "DLINEAR", "NLINEAR", "PATCHTST", "PATCHTST-STYLE", "PATCHTST_STYLE"}


@dataclass
class WaveletParams:
    wavelet: str = "db8"
    level: int = 3
    threshold_mode: str = "soft"
    threshold_strategy: str = "universal"
    threshold_scale: float = 1.0

    def tag(self) -> str:
        return f"w{self.wavelet}_l{self.level}_{self.threshold_strategy}_{self.threshold_mode}_s{self.threshold_scale:g}"


def clone_with_updates(cfg: Dict[str, Any], updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    new_cfg = copy.deepcopy(cfg)
    updates = updates or {}
    for section, values in updates.items():
        if isinstance(values, dict):
            new_cfg.setdefault(section, {})
            new_cfg[section].update(values)
        else:
            new_cfg[section] = values
    return new_cfg


def ensure_major_revision_dirs(cfg: Dict[str, Any]) -> None:
    ensure_output_dirs(cfg)
    for key in ["metrics_dir", "predictions_dir", "figures_dir", "tables_dir", "model_dir", "log_dir"]:
        get_path(cfg, key).mkdir(parents=True, exist_ok=True)


def _mad_sigma(detail_coeff: np.ndarray) -> float:
    detail_coeff = np.asarray(detail_coeff, dtype=float).reshape(-1)
    if detail_coeff.size == 0:
        return 0.0
    sigma = np.median(np.abs(detail_coeff - np.median(detail_coeff))) / 0.6745
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(detail_coeff)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.0
    return float(sigma)


def _rigrsure_threshold(coeff: np.ndarray, sigma: float) -> float:
    coeff = np.asarray(coeff, dtype=float).reshape(-1)
    n = coeff.size
    if n == 0 or sigma <= 0:
        return 0.0
    y = np.sort((coeff / sigma) ** 2)
    risks = (n - 2 * np.arange(1, n + 1) + np.cumsum(y) + np.arange(n, 0, -1) * y) / n
    idx = int(np.argmin(risks))
    return float(sigma * math.sqrt(max(y[idx], 0.0)))


def wavelet_threshold_value(coeff: np.ndarray, n: int, strategy: str = "universal") -> float:
    strategy = strategy.lower().strip()
    coeff = np.asarray(coeff, dtype=float).reshape(-1)
    sigma = _mad_sigma(coeff)
    if sigma <= 0:
        return 0.0
    if strategy == "universal":
        return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))
    if strategy == "minimax":
        if n <= 32:
            return 0.0
        return float(sigma * (0.3936 + 0.1829 * np.log2(n)))
    if strategy in {"sure", "rigrsure"}:
        return _rigrsure_threshold(coeff, sigma)
    if strategy in {"bayes", "bayesshrink", "bayes_shrink"}:
        var = float(np.var(coeff))
        sigma_x = math.sqrt(max(var - sigma**2, 1e-12))
        return float((sigma**2) / sigma_x)
    raise ValueError(f"Unsupported threshold_strategy: {strategy}")


def enhanced_wavelet_denoise(series: np.ndarray, params: WaveletParams) -> Dict[str, Any]:
    x = np.asarray(series, dtype=float).reshape(-1)
    if x.size < 8:
        return {"denoised": x.copy(), "used_level": 0, "thresholds": []}
    w = pywt.Wavelet(params.wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=w.dec_len)
    used_level = max(1, min(int(params.level), max_level))
    coeffs = pywt.wavedec(x, wavelet=params.wavelet, level=used_level, mode="symmetric")
    new_coeffs = [coeffs[0]]
    thresholds = []
    for c in coeffs[1:]:
        thr = wavelet_threshold_value(c, n=len(x), strategy=params.threshold_strategy) * float(params.threshold_scale)
        thresholds.append(thr)
        new_coeffs.append(pywt.threshold(c, value=thr, mode=params.threshold_mode))
    denoised = pywt.waverec(new_coeffs, wavelet=params.wavelet, mode="symmetric")[: len(x)]
    return {"denoised": denoised, "used_level": used_level, "thresholds": thresholds}


def prepare_dataset_mr(
    cfg: Dict[str, Any],
    dataset_name: str,
    *,
    use_wavelet: bool = False,
    wavelet_params: Optional[WaveletParams] = None,
    time_steps: Optional[int] = None,
    quick: bool = False,
    save_npz: bool = True,
    experiment_tag: str = "mr",
) -> Dict[str, Any]:
    df = load_raw_dataset(cfg, dataset_name, quick=quick)
    time_steps = int(time_steps if time_steps is not None else cfg["preprocessing"].get("time_steps", 10))
    train_ratio = float(cfg["datasets"][dataset_name].get("train_ratio", 0.8))
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    scaler = make_scaler(cfg["preprocessing"].get("scaler", "minmax"))
    train_scaled = scaler.fit_transform(train_df[["traffic"]].values).reshape(-1)
    test_scaled = scaler.transform(test_df[["traffic"]].values).reshape(-1)

    wavelet_meta = None
    if use_wavelet:
        wcfg = cfg.get("wavelet", {})
        params = wavelet_params or WaveletParams(
            wavelet=wcfg.get("wavelet", "db8"),
            level=int(wcfg.get("level", 3)),
            threshold_mode=wcfg.get("threshold_mode", "soft"),
            threshold_strategy=wcfg.get("threshold_strategy", "universal"),
            threshold_scale=float(wcfg.get("threshold_scale", 1.0)),
        )
        train_wt = enhanced_wavelet_denoise(train_scaled, params)
        test_wt = enhanced_wavelet_denoise(test_scaled, params)
        train_signal = train_wt["denoised"]
        test_signal = test_wt["denoised"]
        wavelet_meta = {
            "wavelet": params.wavelet,
            "level_requested": params.level,
            "threshold_mode": params.threshold_mode,
            "threshold_strategy": params.threshold_strategy,
            "threshold_scale": params.threshold_scale,
            "train_used_level": train_wt["used_level"],
            "test_used_level": test_wt["used_level"],
            "train_thresholds": train_wt["thresholds"],
            "test_thresholds": test_wt["thresholds"],
        }
    else:
        train_signal = train_scaled
        test_signal = test_scaled

    X_train, y_train = create_windows(train_signal, time_steps=time_steps)
    context = train_signal[-time_steps:]
    test_with_context = np.concatenate([context, test_signal], axis=0)
    X_test, y_test = create_windows(test_with_context, time_steps=time_steps)
    y_test_raw = test_df["traffic"].iloc[:len(y_test)].to_numpy(dtype=float)
    test_timestamps = test_df["timestamp"].iloc[:len(y_test)].reset_index(drop=True)

    if save_npz:
        processed_dir = get_path(cfg, "processed_dir")
        processed_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{experiment_tag}_{'WT' if use_wavelet else 'RAW'}_ts{time_steps}"
        np.savez_compressed(
            processed_dir / f"{safe_name(dataset_name)}_{suffix}_windows.npz",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            y_test_raw=y_test_raw,
            test_timestamps=test_timestamps.astype(str).to_numpy(),
        )

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_raw": y_test_raw,
        "test_timestamps": test_timestamps,
        "scaler": scaler,
        "wavelet_meta": wavelet_meta,
    }


def build_any_model(model_name: str, input_shape, cfg: Dict[str, Any], topology_shape=None) -> tf.keras.Model:
    name = model_name.upper().replace("_", "-")
    if name in ORIGINAL_MODEL_NAMES:
        return build_original_model(model_name, input_shape=input_shape, topology_shape=topology_shape, cfg=cfg)
    if name in MR_MODEL_NAMES:
        return build_major_revision_model(model_name, input_shape=input_shape, cfg=cfg)
    raise ValueError(f"Unsupported model name: {model_name}")


def model_requires_wavelet(model_name: str) -> bool:
    return model_name.upper().replace("_", "-") in {"WT-LSTM", "WT-BILSTM", "WT-TBILSTM"}


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


def train_major_revision_one(
    cfg: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    seed: int,
    *,
    quick: bool = False,
    verbose: int = 1,
    use_wavelet: Optional[bool] = None,
    wavelet_params: Optional[WaveletParams] = None,
    time_steps: Optional[int] = None,
    experiment_tag: str = "major_revision",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    ensure_major_revision_dirs(cfg)
    use_wavelet = model_requires_wavelet(model_name) if use_wavelet is None else bool(use_wavelet)
    data = prepare_dataset_mr(
        cfg,
        dataset_name,
        use_wavelet=use_wavelet,
        wavelet_params=wavelet_params,
        time_steps=time_steps,
        quick=quick,
        save_npz=True,
        experiment_tag=experiment_tag,
    )

    X_train, y_train, X_test = data["X_train"], data["y_train"], data["X_test"]
    if model_uses_topology(model_name):
        topo_train = compute_topology_features(X_train)
        topo_test = compute_topology_features(X_test)
        model = build_any_model(model_name, input_shape=X_train.shape[1:], topology_shape=topo_train.shape[1:], cfg=cfg)
        fit_x = [X_train, topo_train]
        pred_x = [X_test, topo_test]
    else:
        model = build_any_model(model_name, input_shape=X_train.shape[1:], cfg=cfg)
        fit_x = X_train
        pred_x = X_test

    batch_size = int(cfg["training"].get("batch_size", 64))
    epochs = int(cfg["training"].get("epochs", 80))
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
    log_dir = get_path(cfg, "log_dir")
    model_dir = get_path(cfg, "model_dir")
    stem = f"{safe_name(experiment_tag)}_{safe_name(dataset_name)}_{safe_name(model_name)}_seed{seed}"

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
        "ExperimentTag": experiment_tag,
        "Dataset": dataset_name,
        "Model": model_name,
        "Seed": int(seed),
        **metrics,
        "TrainSeconds": train_seconds,
        "InferenceSeconds": inference_seconds,
        "InferenceSecondsPerSample": inference_seconds / max(1, len(y_pred)),
        "TrainWindows": int(len(X_train)),
        "TestWindows": int(len(X_test)),
        "TimeSteps": int(time_steps if time_steps is not None else cfg["preprocessing"].get("time_steps", 10)),
        "UseWavelet": bool(use_wavelet),
        "Parameters": int(model.count_params()),
        "PredictionFile": str(pred_path.relative_to(Path(cfg["_project_root"]))),
    }
    if wavelet_params is not None:
        row.update({
            "Wavelet": wavelet_params.wavelet,
            "Level": wavelet_params.level,
            "ThresholdMode": wavelet_params.threshold_mode,
            "ThresholdStrategy": wavelet_params.threshold_strategy,
            "ThresholdScale": wavelet_params.threshold_scale,
        })
    if data.get("wavelet_meta"):
        row["WaveletMeta"] = str(data["wavelet_meta"])
    if extra_metadata:
        row.update(extra_metadata)
    return row


def save_run_outputs(rows: Iterable[Dict[str, Any]], metrics_dir: str | Path, prefix: str) -> None:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_csv(metrics_dir / f"{prefix}_raw.csv", index=False)
    ok = df.dropna(subset=["RMSE", "MAE", "MAPE"], how="any") if not df.empty else df
    if not ok.empty:
        summary = summarize_runs(ok)
        summary.to_csv(metrics_dir / f"{prefix}_summary.csv", index=False)
