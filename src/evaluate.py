from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["RMSE", "MAE", "MAPE"]
    rows = []
    for (dataset, model), g in df.groupby(["Dataset", "Model"], sort=False):
        row = {"Dataset": dataset, "Model": model, "Runs": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_overall_tables(metrics_rows: Iterable[Dict], metrics_dir: str | Path) -> None:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(metrics_rows))
    if df.empty:
        return
    df.to_csv(metrics_dir / "run_metrics.csv", index=False)
    summary = summarize_runs(df)
    summary.to_csv(metrics_dir / "repeated_runs.csv", index=False)

    overall = summary[[
        "Dataset", "Model", "RMSE_mean", "MAE_mean", "MAPE_mean"
    ]].rename(columns={
        "RMSE_mean": "RMSE",
        "MAE_mean": "MAE",
        "MAPE_mean": "MAPE",
    })
    overall.to_csv(metrics_dir / "overall_metrics.csv", index=False)
