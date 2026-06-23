from __future__ import annotations

import numpy as np


TOPOLOGY_FEATURE_NAMES = [
    "value",
    "first_difference",
    "second_difference",
    "local_maximum",
    "local_minimum",
    "persistence_strength",
    "relative_position",
]


def _window_topology_features(window: np.ndarray) -> np.ndarray:
    """Compute deterministic topology-inspired features for one time window."""
    x = np.asarray(window, dtype=float).reshape(-1)
    t = len(x)
    if t == 0:
        return np.zeros((0, len(TOPOLOGY_FEATURE_NAMES)), dtype=float)

    v_min, v_max = np.nanmin(x), np.nanmax(x)
    span = v_max - v_min
    value = (x - v_min) / span if span > 1e-12 else np.zeros_like(x)

    first = np.zeros_like(x)
    first[1:] = x[1:] - x[:-1]
    fd_scale = np.max(np.abs(first))
    first = first / fd_scale if fd_scale > 1e-12 else first

    second = np.zeros_like(x)
    if t > 2:
        second[1:-1] = x[2:] - 2 * x[1:-1] + x[:-2]
    sd_scale = np.max(np.abs(second))
    second = second / sd_scale if sd_scale > 1e-12 else second

    local_max = np.zeros_like(x)
    local_min = np.zeros_like(x)
    persistence = np.zeros_like(x)
    for i in range(1, t - 1):
        if x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            local_max[i] = 1.0
            persistence[i] = min(abs(x[i] - x[i - 1]), abs(x[i] - x[i + 1]))
        if x[i] <= x[i - 1] and x[i] <= x[i + 1]:
            local_min[i] = 1.0
            persistence[i] = min(abs(x[i] - x[i - 1]), abs(x[i] - x[i + 1]))
    p_scale = np.max(np.abs(persistence))
    persistence = persistence / p_scale if p_scale > 1e-12 else persistence

    pos = np.linspace(0.0, 1.0, t)
    return np.stack([value, first, second, local_max, local_min, persistence, pos], axis=-1).astype("float32")


def compute_topology_features(X: np.ndarray) -> np.ndarray:
    """Compute topology-inspired features for a batch of windows."""
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[..., None]
    windows = X[:, :, 0]
    features = np.stack([_window_topology_features(w) for w in windows], axis=0)
    return features.astype("float32")
