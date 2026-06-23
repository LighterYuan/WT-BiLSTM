from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import get_path
from utils import save_json
from wavelet_denoising import save_wavelet_components, wavelet_denoise


def load_raw_dataset(cfg: Dict[str, Any], dataset_name: str, quick: bool = False) -> pd.DataFrame:
    dcfg = cfg["datasets"][dataset_name]
    path = get_path(cfg, "raw_data_dir") / dcfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    if dcfg["time_col"] not in df.columns or dcfg["target_col"] not in df.columns:
        raise ValueError(f"{dataset_name}: expected columns {dcfg['time_col']} and {dcfg['target_col']}, got {df.columns.tolist()}")

    df = df[[dcfg["time_col"], dcfg["target_col"]]].copy()
    df.columns = ["timestamp", "traffic"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
    df = df.dropna(subset=["traffic"]).reset_index(drop=True)
    if df["timestamp"].isna().all():
        df["timestamp"] = pd.RangeIndex(start=0, stop=len(df), step=1)
    else:
        df["timestamp"] = df["timestamp"].ffill().bfill()

    max_rows = cfg.get("_quick_max_rows") if quick else None
    if max_rows is not None and len(df) > int(max_rows):
        df = df.iloc[: int(max_rows)].copy().reset_index(drop=True)
    return df


def make_scaler(name: str):
    if name.lower() == "minmax":
        return MinMaxScaler()
    if name.lower() == "standard":
        return StandardScaler()
    raise ValueError(f"Unsupported scaler: {name}")


def create_windows(values: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype="float32").reshape(-1, 1)
    X, y = [], []
    for i in range(time_steps, len(arr)):
        X.append(arr[i - time_steps:i, :])
        y.append(arr[i, 0])
    if not X:
        raise ValueError(f"Not enough observations ({len(arr)}) for time_steps={time_steps}")
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def inverse_transform_target(scaler, y_scaled: np.ndarray) -> np.ndarray:
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return scaler.inverse_transform(y_scaled).reshape(-1)


def prepare_dataset(
    cfg: Dict[str, Any],
    dataset_name: str,
    use_wavelet: bool = False,
    quick: bool = False,
) -> Dict[str, Any]:
    """Load, split, scale, optionally denoise, and window one dataset."""
    df = load_raw_dataset(cfg, dataset_name, quick=quick)
    time_steps = int(cfg["preprocessing"]["time_steps"])
    train_ratio = float(cfg["datasets"][dataset_name].get("train_ratio", 0.8))
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    scaler = make_scaler(cfg["preprocessing"].get("scaler", "minmax"))
    train_scaled = scaler.fit_transform(train_df[["traffic"]].values).reshape(-1)
    test_scaled = scaler.transform(test_df[["traffic"]].values).reshape(-1)

    processed_dir = get_path(cfg, "processed_dir")
    wavelet_dir = get_path(cfg, "wavelet_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    wavelet_dir.mkdir(parents=True, exist_ok=True)

    train_signal = train_scaled
    test_signal = test_scaled
    wavelet_meta = None

    if use_wavelet:
        wcfg = cfg["wavelet"]
        train_wt = wavelet_denoise(
            train_scaled,
            wavelet=wcfg.get("wavelet", "db8"),
            level=int(wcfg.get("level", 3)),
            threshold_mode=wcfg.get("threshold_mode", "soft"),
        )
        test_wt = wavelet_denoise(
            test_scaled,
            wavelet=wcfg.get("wavelet", "db8"),
            level=int(wcfg.get("level", 3)),
            threshold_mode=wcfg.get("threshold_mode", "soft"),
        )
        train_signal = train_wt.denoised
        test_signal = test_wt.denoised
        save_wavelet_components(train_wt, str(wavelet_dir / f"{dataset_name}_train_wavelet_components.csv"))
        save_wavelet_components(test_wt, str(wavelet_dir / f"{dataset_name}_test_wavelet_components.csv"))
        pd.DataFrame({
            "timestamp": train_df["timestamp"],
            "traffic_scaled_denoised": train_signal,
        }).to_csv(wavelet_dir / f"{dataset_name}_train_denoised.csv", index=False)
        pd.DataFrame({
            "timestamp": test_df["timestamp"],
            "traffic_scaled_denoised": test_signal,
        }).to_csv(wavelet_dir / f"{dataset_name}_test_denoised.csv", index=False)
        wavelet_meta = {
            "train_used_level": train_wt.used_level,
            "test_used_level": test_wt.used_level,
            "train_threshold": train_wt.threshold,
            "test_threshold": test_wt.threshold,
        }

    X_train, y_train = create_windows(train_signal, time_steps=time_steps)

    context = train_signal[-time_steps:]
    test_with_context = np.concatenate([context, test_signal], axis=0)
    X_test, y_test = create_windows(test_with_context, time_steps=time_steps)

    test_timestamps = test_df["timestamp"].iloc[:len(y_test)].reset_index(drop=True)
    y_test_raw = test_df["traffic"].iloc[:len(y_test)].to_numpy(dtype=float)

    npz_name = f"{dataset_name}_{'WT_' if use_wavelet else ''}windows.npz"
    np.savez_compressed(
        processed_dir / npz_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_test_raw=y_test_raw,
        test_timestamps=test_timestamps.astype(str).to_numpy(),
    )

    train_df.to_csv(processed_dir / f"{dataset_name}_train.csv", index=False)
    test_df.to_csv(processed_dir / f"{dataset_name}_test.csv", index=False)

    report = {
        "dataset": dataset_name,
        "raw_observations": int(len(df)),
        "train_observations": int(len(train_df)),
        "test_observations": int(len(test_df)),
        "time_steps": int(time_steps),
        "train_windows": int(len(X_train)),
        "effective_test_windows": int(len(X_test)),
        "use_wavelet": bool(use_wavelet),
        "scaler": cfg["preprocessing"].get("scaler", "minmax"),
        "wavelet_meta": wavelet_meta,
    }
    suffix = "WT" if use_wavelet else "RAW"
    save_json(report, processed_dir / f"{dataset_name}_{suffix}_split_report.json")

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
        "report": report,
    }


if __name__ == "__main__":
    from config import load_config, ensure_output_dirs
    cfg = load_config("config.yaml")
    ensure_output_dirs(cfg)
    for ds in cfg["datasets"]:
        prepare_dataset(cfg, ds, use_wavelet=False)
        prepare_dataset(cfg, ds, use_wavelet=True)
        print(f"Prepared {ds}")
