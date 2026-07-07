from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | os.PathLike = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(project_root())
    return cfg


def resolve_path(cfg: Dict[str, Any], path: str | os.PathLike) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(cfg["_project_root"]) / p


def get_path(cfg: Dict[str, Any], key: str) -> Path:
    return resolve_path(cfg, cfg["paths"][key])


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    for key in cfg["paths"]:
        resolve_path(cfg, cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def parse_common_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke-test mode.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names to run.")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Random seeds.")
    return parser.parse_args()


def apply_quick_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    quick = cfg.get("quick", {})
    cfg = dict(cfg)
    cfg["training"] = dict(cfg["training"])
    cfg["training"]["epochs"] = quick.get("epochs", cfg["training"]["epochs"])
    cfg["training"]["batch_size"] = quick.get("batch_size", cfg["training"]["batch_size"])
    cfg["_quick_max_rows"] = quick.get("max_rows")
    return cfg
