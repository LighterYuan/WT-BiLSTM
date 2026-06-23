from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import get_path, load_config
from evaluate import compute_metrics


def pick_prediction_file(predictions_dir: Path, dataset: str, model: str, seed: int | None = None) -> Path | None:
    if seed is not None:
        candidates = sorted(predictions_dir.glob(f"{dataset}_{model}_seed{seed}_pred.csv"))
    else:
        candidates = sorted(predictions_dir.glob(f"{dataset}_{model}_seed*_pred.csv"))
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="WT-TBiLSTM")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictions_dir = get_path(cfg, "predictions_dir")
    metrics_dir = get_path(cfg, "metrics_dir")
    seg_size = int(cfg.get("robustness", {}).get("segment_size", 500))

    rows = []
    for dataset in cfg["datasets"]:
        path = pick_prediction_file(predictions_dir, dataset, args.model, args.seed)
        if path is None:
            print(f"[WARN] Prediction file not found for {dataset} {args.model}")
            continue
        df = pd.read_csv(path)
        full_out = predictions_dir / f"{dataset}_full_test_{args.model}.csv"
        df.to_csv(full_out, index=False)

        n = len(df)
        for start in range(0, n, seg_size):
            end = min(start + seg_size, n)
            seg = df.iloc[start:end]
            m = compute_metrics(seg["y_true"], seg["y_pred"])
            rows.append({
                "Dataset": dataset,
                "Model": args.model,
                "SegmentStart": start,
                "SegmentEnd": end - 1,
                "SegmentSize": len(seg),
                **m,
            })

    out = pd.DataFrame(rows)
    out.to_csv(metrics_dir / "segment_robustness.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
