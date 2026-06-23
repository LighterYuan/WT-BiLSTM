from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from config import get_path, load_config


def paired_test(a: np.ndarray, b: np.ndarray):
    """Return test name and p-value for paired scores."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return "insufficient_runs", np.nan
    diff = a - b
    if np.allclose(diff, 0):
        return "all_equal", 1.0
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return "wilcoxon_signed_rank", float(p)
    except Exception:
        stat, p = stats.ttest_rel(a, b)
        return "paired_t_test", float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics_dir = get_path(cfg, "metrics_dir")
    path = metrics_dir / "run_metrics.csv"
    if not path.exists():
        raise FileNotFoundError("run_metrics.csv not found. Run run_all_experiments.py with multiple seeds first.")

    df = pd.read_csv(path)
    ref_model = cfg["statistics"].get("reference_model", "WT-TBiLSTM")
    baselines = cfg["statistics"].get("tests", ["WT-BiLSTM", "CNN-LSTM", "LSTM"])
    alpha = float(cfg["statistics"].get("alpha", 0.05))

    rows = []
    for dataset in df["Dataset"].unique():
        ds = df[df["Dataset"] == dataset]
        ref = ds[ds["Model"] == ref_model].sort_values("Seed")
        for base in baselines:
            other = ds[ds["Model"] == base].sort_values("Seed")
            common = sorted(set(ref["Seed"]).intersection(other["Seed"]))
            if not common:
                continue
            ref2 = ref[ref["Seed"].isin(common)].sort_values("Seed")
            other2 = other[other["Seed"].isin(common)].sort_values("Seed")
            for metric in ["RMSE", "MAE", "MAPE"]:
                test_name, p = paired_test(other2[metric].values, ref2[metric].values)
                rows.append({
                    "Dataset": dataset,
                    "Metric": metric,
                    "ReferenceModel": ref_model,
                    "ComparedModel": base,
                    "CommonRuns": len(common),
                    "Test": test_name,
                    "p_value": p,
                    "SignificantAtAlpha": bool(p < alpha) if np.isfinite(p) else False,
                    "ComparedMean": other2[metric].mean(),
                    "ReferenceMean": ref2[metric].mean(),
                    "Improvement_%": (other2[metric].mean() - ref2[metric].mean()) / max(abs(other2[metric].mean()), 1e-12) * 100.0,
                })

    out = pd.DataFrame(rows)
    out.to_csv(metrics_dir / "statistical_tests.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
