from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import pywt


@dataclass
class WaveletResult:
    denoised: np.ndarray
    components: Dict[str, np.ndarray]
    used_level: int
    threshold: float


def universal_threshold(detail_coeff: np.ndarray, n: int) -> float:
    if detail_coeff.size == 0:
        return 0.0
    sigma = np.median(np.abs(detail_coeff - np.median(detail_coeff))) / 0.6745
    if not np.isfinite(sigma) or sigma == 0:
        sigma = np.std(detail_coeff)
    if not np.isfinite(sigma):
        sigma = 0.0
    return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))


def wavelet_denoise(
    series: np.ndarray,
    wavelet: str = "db8",
    level: int = 3,
    threshold_mode: str = "soft",
) -> WaveletResult:
    """Denoise a 1D series with leakage-safe per-segment wavelet thresholding."""
    x = np.asarray(series, dtype=float).reshape(-1)
    if x.size < 8:
        return WaveletResult(denoised=x.copy(), components={"original": x.copy()}, used_level=0, threshold=0.0)

    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=w.dec_len)
    used_level = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=used_level, mode="symmetric")

    detail_for_sigma = coeffs[-1]
    thr = universal_threshold(detail_for_sigma, len(x))

    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, value=thr, mode=threshold_mode))

    denoised = pywt.waverec(new_coeffs, wavelet=wavelet, mode="symmetric")[: len(x)]

    components: Dict[str, np.ndarray] = {"original": x.copy(), "denoised": denoised.copy()}
    approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    components["A"] = pywt.waverec(approx_coeffs, wavelet=wavelet, mode="symmetric")[: len(x)]
    for i in range(1, len(coeffs)):
        single = [np.zeros_like(c) for c in coeffs]
        single[i] = coeffs[i]
        label = f"D{len(coeffs)-i}"
        components[label] = pywt.waverec(single, wavelet=wavelet, mode="symmetric")[: len(x)]

    return WaveletResult(denoised=denoised, components=components, used_level=used_level, threshold=thr)


def save_wavelet_components(result: WaveletResult, path: str) -> None:
    max_len = len(result.denoised)
    data = {}
    for k, v in result.components.items():
        arr = np.asarray(v).reshape(-1)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
        data[k] = arr[:max_len]
    pd.DataFrame(data).to_csv(path, index=False)
