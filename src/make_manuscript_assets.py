from __future__ import annotations

import subprocess
import sys


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    config = "config.yaml"
    run([sys.executable, "src/run_all_experiments.py", "--config", config, "--seeds", "1", "2", "3", "4", "5"])
    run([sys.executable, "src/run_ablation.py", "--config", config])
    run([sys.executable, "src/run_robustness.py", "--config", config])
    run([sys.executable, "src/run_complexity.py", "--config", config])
    run([sys.executable, "src/statistical_test.py", "--config", config])
    run([sys.executable, "src/plot_figures.py", "--config", config])


if __name__ == "__main__":
    main()
