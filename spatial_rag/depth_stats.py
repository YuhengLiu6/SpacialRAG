from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


def mask_depth_stats(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    trim_fraction: float = 0.10,
) -> Dict[str, Optional[float]]:
    depth = np.asarray(depth_map_m, dtype=np.float32)
    mask_arr = np.asarray(mask).astype(bool)
    valid = depth[np.logical_and(mask_arr, np.isfinite(depth))]
    valid = valid[valid > 0.0]
    if valid.size == 0:
        return {
            "median_m": None,
            "trimmed_median_m": None,
            "p10_m": None,
            "p90_m": None,
            "num_valid_px": 0,
        }
    sorted_vals = np.sort(valid.astype(np.float32))
    trim_count = int(math.floor(float(sorted_vals.size) * max(0.0, min(float(trim_fraction), 0.45))))
    if trim_count > 0 and sorted_vals.size > (2 * trim_count):
        trimmed = sorted_vals[trim_count:-trim_count]
    else:
        trimmed = sorted_vals
    return {
        "median_m": float(np.median(sorted_vals)),
        "trimmed_median_m": float(np.median(trimmed)),
        "p10_m": float(np.percentile(sorted_vals, 10.0)),
        "p90_m": float(np.percentile(sorted_vals, 90.0)),
        "num_valid_px": int(sorted_vals.size),
    }