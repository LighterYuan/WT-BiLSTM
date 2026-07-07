from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Please install it with: pip install pyyaml") from exc


METRICS: tuple[str, ...] = ("RMSE", "MAE", "MAPE")

# These four models correspond to the eight ablation prediction files expected by the paper plots:
#   2 datasets x 4 models = 8 files
DEFAULT_ABLATION_PREDICTION_MODELS: tuple[str, ...] = (
    "LSTM",
    "WT-LSTM",
    "WT-BiLSTM",
    "WT-TBiLSTM",
)

DEFAULT_WAVELET_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("LSTM", "WT-LSTM", "Wavelet denoising"),
    ("BiLSTM", "WT-BiLSTM", "Wavelet denoising"),
)

DEFAULT_TOPOLOGY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("WT-BiLSTM", "WT-TBiLSTM", "Topology-aware attention"),
)


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    metrics_dir: Path
    predictions_dir: Path


def _load_yaml_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config format: {path}")

    return cfg, path.parent


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _get_project_paths(cfg: dict, project_root: Path) -> ProjectPaths:
    paths = cfg.get("paths", {}) or {}
    if not isinstance(paths, dict):
        raise ValueError("config.yaml field 'paths' must be a mapping.")

    metrics_dir = _resolve_project_path(project_root, paths.get("metrics_dir", "results/metrics"))
    predictions_dir = _resolve_project_path(project_root, paths.get("predictions_dir", "results/predictions"))

    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_root=project_root,
        metrics_dir=metrics_dir,
        predictions_dir=predictions_dir,
    )


def _model_key(name: str) -> str:
    """Normalize model/dataset names so WT-LSTM, WT_LSTM and WT LSTM can be matched."""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _safe_name(name: str) -> str:
    """Convert names to stable file-name tokens used by this project."""
    return str(name).strip().replace("-", "_").replace(" ", "_")


def _load_overall_metrics(metrics_dir: Path) -> pd.DataFrame:
    path = metrics_dir / "overall_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"overall_metrics.csv not found at {path}. "
            "Please run: python src/run_all_experiments.py --config config.yaml first."
        )

    df = pd.read_csv(path)
    required = {"Dataset", "Model", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"overall_metrics.csv is missing required columns: {sorted(missing)}")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if df[list(METRICS)].isna().any().any():
        raise ValueError("overall_metrics.csv contains non-numeric or missing metric values.")

    df = df.copy()
    df["_model_key"] = df["Model"].map(_model_key)
    return df


def _find_model_row(dataset_df: pd.DataFrame, model_name: str) -> pd.Series | None:
    rows = dataset_df[dataset_df["_model_key"] == _model_key(model_name)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _pair_table(
    df: pd.DataFrame,
    pairs: Sequence[tuple[str, str, str]],
    out_path: Path,
) -> pd.DataFrame:
    rows: list[dict] = []

    for dataset in df["Dataset"].dropna().unique():
        dataset_df = df[df["Dataset"] == dataset]

        for baseline, improved, factor in pairs:
            base_row = _find_model_row(dataset_df, baseline)
            improved_row = _find_model_row(dataset_df, improved)

            if base_row is None or improved_row is None:
                print(
                    f"[Warning] Skip ablation pair on {dataset}: "
                    f"missing '{baseline}' or '{improved}' in overall_metrics.csv",
                    file=sys.stderr,
                )
                continue

            row: dict[str, object] = {
                "Dataset": dataset,
                "Ablation": factor,
                "Baseline": baseline,
                "Improved": improved,
            }

            for metric in METRICS:
                base_value = float(base_row[metric])
                improved_value = float(improved_row[metric])
                denominator = max(abs(base_value), 1e-12)
                reduction = (base_value - improved_value) / denominator * 100.0

                # Stable columns for downstream plotting/table generation.
                row[f"Baseline_{metric}"] = base_value
                row[f"Improved_{metric}"] = improved_value
                row[f"{metric}_Reduction_%"] = reduction

                # Backward-compatible columns matching the previous script style.
                row[f"{baseline}_{metric}"] = base_value
                row[f"{improved}_{metric}"] = improved_value

            rows.append(row)

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def _source_prediction_candidates(predictions_dir: Path, dataset: str, model: str) -> list[Path]:
    dataset_variants = [dataset, _safe_name(dataset)]
    model_variants = [model, _safe_name(model), model.replace("_", "-"), model.replace("-", "_")]

    candidates: list[Path] = []
    seen: set[Path] = set()

    for ds in dataset_variants:
        for md in model_variants:
            for suffix in ("pred", "prediction", "predictions"):
                candidate = predictions_dir / f"{ds}_{md}_{suffix}.csv"
                if candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)

    # Fallback: tolerate seed/version suffixes, e.g. UK_Ac_WT_TBiLSTM_seed1_pred.csv
    ds_key = _model_key(dataset)
    md_key = _model_key(model)
    for candidate in predictions_dir.glob("*.csv"):
        if candidate.name.endswith("_ablation_pred.csv"):
            continue
        stem_key = _model_key(candidate.stem)
        if stem_key.startswith(ds_key + md_key) and "pred" in stem_key:
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

    # Prefer exact/simple names over versioned names.
    return sorted(candidates, key=lambda p: (not p.exists(), len(p.name), p.name.lower()))


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized = {_model_key(col): col for col in df.columns}
    for name in candidates:
        hit = normalized.get(_model_key(name))
        if hit is not None:
            return hit
    return None


def _standardize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    y_true_col = _find_column(df, ("y_true", "true", "actual", "ground_truth", "target", "ytest", "y_test"))
    y_pred_col = _find_column(df, ("y_pred", "pred", "prediction", "predicted", "forecast", "yhat", "y_hat"))
    index_col = _find_column(df, ("sample_index", "index", "idx"))
    time_col = _find_column(df, ("timestamp", "time", "date", "datetime"))

    if y_true_col is None or y_pred_col is None:
        print(
            "[Warning] Could not identify y_true/y_pred columns in a prediction file. "
            "The file will be copied without standardization.",
            file=sys.stderr,
        )
        return df.copy()

    y_true = pd.to_numeric(df[y_true_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df[y_pred_col], errors="coerce").to_numpy(dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Prediction length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")

    out = pd.DataFrame()
    if time_col is not None:
        out["timestamp"] = df[time_col].values

    if index_col is not None:
        out["sample_index"] = df[index_col].values
    else:
        out["sample_index"] = np.arange(len(df))

    absolute_error = np.abs(y_true - y_pred)
    percentage_error = absolute_error / np.maximum(np.abs(y_true), 1e-12) * 100.0

    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["absolute_error"] = absolute_error
    out["percentage_error"] = percentage_error

    return out


def _create_ablation_prediction_files(
    df: pd.DataFrame,
    predictions_dir: Path,
    models: Sequence[str] = DEFAULT_ABLATION_PREDICTION_MODELS,
) -> tuple[list[Path], list[tuple[str, str]]]:
    saved: list[Path] = []
    missing: list[tuple[str, str]] = []

    datasets = list(df["Dataset"].dropna().unique())

    for dataset in datasets:
        for model in models:
            target_path = predictions_dir / f"{_safe_name(dataset)}_{_safe_name(model)}_ablation_pred.csv"
            source_path = _first_existing(_source_prediction_candidates(predictions_dir, str(dataset), model))

            if source_path is None:
                missing.append((str(dataset), model))
                print(
                    f"[Warning] Missing source prediction for {dataset} / {model}. "
                    f"Expected something like: {predictions_dir / f'{_safe_name(dataset)}_{_safe_name(model)}_pred.csv'}",
                    file=sys.stderr,
                )
                continue

            source_df = pd.read_csv(source_path)
            out_df = _standardize_prediction_df(source_df)
            out_df.to_csv(target_path, index=False)
            saved.append(target_path)
            print(f"[Saved] Ablation predictions: {target_path}")

    return saved, missing


def _read_prediction_models_from_config(cfg: dict) -> tuple[str, ...]:
    ablation_cfg = cfg.get("ablation", {}) or {}
    if isinstance(ablation_cfg, dict):
        models = ablation_cfg.get("prediction_models")
        if isinstance(models, list) and models:
            return tuple(str(x) for x in models)
    return DEFAULT_ABLATION_PREDICTION_MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation metric tables and ablation prediction CSV files.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--strict-predictions",
        action="store_true",
        help="Exit with an error if any expected ablation prediction file cannot be generated.",
    )
    args = parser.parse_args()

    cfg, project_root = _load_yaml_config(args.config)
    paths = _get_project_paths(cfg, project_root)

    df = _load_overall_metrics(paths.metrics_dir)

    wavelet = _pair_table(
        df,
        pairs=DEFAULT_WAVELET_PAIRS,
        out_path=paths.metrics_dir / "wavelet_ablation.csv",
    )
    topology = _pair_table(
        df,
        pairs=DEFAULT_TOPOLOGY_PAIRS,
        out_path=paths.metrics_dir / "topology_ablation.csv",
    )

    prediction_models = _read_prediction_models_from_config(cfg)
    saved, missing = _create_ablation_prediction_files(
        df,
        predictions_dir=paths.predictions_dir,
        models=prediction_models,
    )

    print("\nWavelet ablation:")
    print(wavelet.to_string(index=False) if not wavelet.empty else "[Empty] No valid wavelet ablation rows.")

    print("\nTopology ablation:")
    print(topology.to_string(index=False) if not topology.empty else "[Empty] No valid topology ablation rows.")

    print(f"\n[Done] Saved metric tables to: {paths.metrics_dir}")
    print(f"[Done] Saved {len(saved)} ablation prediction file(s) to: {paths.predictions_dir}")

    if missing:
        missing_text = ", ".join(f"{dataset}/{model}" for dataset, model in missing)
        message = (
            "Some ablation prediction files were not generated because their source prediction files are missing: "
            f"{missing_text}. Run run_all_experiments.py first and make sure it saves '*_pred.csv' files."
        )
        if args.strict_predictions:
            raise FileNotFoundError(message)
        print(f"[Warning] {message}", file=sys.stderr)


if __name__ == "__main__":
    main()
