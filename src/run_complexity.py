from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from config import get_path, load_config
from models import build_model
from topology_features import TOPOLOGY_FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics_dir = get_path(cfg, "metrics_dir")
    time_steps = int(cfg["preprocessing"]["time_steps"])
    input_shape = (time_steps, 1)
    topology_shape = (time_steps, len(TOPOLOGY_FEATURE_NAMES))

    rows = []
    for model_name in cfg["experiments"]["models"]:
        if model_name == "WT-TBiLSTM":
            model = build_model(model_name, input_shape, cfg, topology_shape=topology_shape)
            x = [
                np.random.rand(args.batch_size, *input_shape).astype("float32"),
                np.random.rand(args.batch_size, *topology_shape).astype("float32"),
            ]
        else:
            model = build_model(model_name, input_shape, cfg)
            x = np.random.rand(args.batch_size, *input_shape).astype("float32")

        model.predict(x, verbose=0)
        start = time.perf_counter()
        for _ in range(args.repeats):
            model.predict(x, verbose=0)
        elapsed = time.perf_counter() - start
        samples = args.batch_size * args.repeats

        rows.append({
            "Model": model_name,
            "Parameters": int(model.count_params()),
            "ApproxInferenceSecondsTotal": elapsed,
            "ApproxInferenceSecondsPerSample": elapsed / samples,
            "InputShape": str(input_shape),
            "TopologyShape": str(topology_shape) if model_name == "WT-TBiLSTM" else "",
        })

    out = pd.DataFrame(rows)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(metrics_dir / "complexity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
