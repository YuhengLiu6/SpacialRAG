from __future__ import annotations

from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={arr.shape!r}")
    return arr.astype(bool)


def _morph_kernel(radius: int) -> np.ndarray:
    value = max(int(radius), 0)
    size = (2 * value) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return mask_u8.astype(bool)
    eroded = cv2.erode(mask_u8, _morph_kernel(radius), iterations=1)
    return eroded.astype(bool)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return mask_u8.astype(bool)
    dilated = cv2.dilate(mask_u8, _morph_kernel(radius), iterations=1)
    return dilated.astype(bool)


def _zero_visible_occlusion_metrics(depth_margin_delta: float) -> Dict[str, Any]:
    return {
        "visible_occlusion_ratio": 0.0,
        "occluded_boundary_ratio": 0.0,
        "nearer_ring_overlap_ratio": 0.0,
        "object_depth_median": None,
        "boundary_pixel_count": 0,
        "occluded_boundary_pixel_count": 0,
        "ring_pixel_count": 0,
        "nearer_ring_pixel_count": 0,
        "depth_margin_delta": float(depth_margin_delta),
    }


def visible_occlusion_ratio_to_level(visible_occlusion_ratio: Any) -> str:
    try:
        ratio = float(visible_occlusion_ratio)
    except Exception:
        ratio = 0.0
    ratio = min(max(ratio, 0.0), 1.0)
    if ratio < 0.10:
        return "fully visible"
    if ratio < 0.30:
        return "slightly occluded"
    if ratio < 0.60:
        return "moderately occluded"
    return "heavily occluded"


def visible_occlusion_ratio_to_penalty(visible_occlusion_ratio: Any) -> float:
    try:
        ratio = float(visible_occlusion_ratio)
    except Exception:
        ratio = 0.0
    return float(min(max(ratio, 0.0), 1.0) * 0.5)


def compute_visible_occlusion_metrics(
    target_mask: np.ndarray,
    target_depth_map: np.ndarray,
    other_object_masks: Sequence[np.ndarray],
    depth_map: np.ndarray,
    *,
    boundary_width: int = 1,
    ring_radius: int = 5,
    depth_margin_delta: float = 0.08,
    boundary_neighbor_radius: int = 1,
) -> Dict[str, Any]:
    metrics = _zero_visible_occlusion_metrics(depth_margin_delta=float(depth_margin_delta))
    target_mask_bool = _as_bool_mask(target_mask)
    if not np.any(target_mask_bool):
        return metrics

    target_depth = np.asarray(target_depth_map, dtype=np.float32)
    full_depth = np.asarray(depth_map, dtype=np.float32)
    if target_depth.shape != target_mask_bool.shape or full_depth.shape != target_mask_bool.shape:
        raise ValueError(
            "target_mask, target_depth_map, and depth_map must share the same HxW shape"
        )

    valid_target_depth_mask = np.logical_and(target_mask_bool, np.isfinite(target_depth))
    valid_target_depth_mask = np.logical_and(valid_target_depth_mask, target_depth > 0.0)
    valid_target_depths = target_depth[valid_target_depth_mask]
    if valid_target_depths.size == 0:
        return metrics

    object_depth_median = float(np.median(valid_target_depths.astype(np.float32)))
    metrics["object_depth_median"] = object_depth_median

    boundary = np.logical_and(target_mask_bool, np.logical_not(_erode_mask(target_mask_bool, int(boundary_width))))
    boundary_pixel_count = int(np.count_nonzero(boundary))
    metrics["boundary_pixel_count"] = boundary_pixel_count
    if boundary_pixel_count == 0:
        return metrics

    ring = np.logical_and(_dilate_mask(boundary, int(ring_radius)), np.logical_not(target_mask_bool))
    ring_pixel_count = int(np.count_nonzero(ring))
    metrics["ring_pixel_count"] = ring_pixel_count

    valid_other_masks: List[np.ndarray] = []
    for other_mask in list(other_object_masks or []):
        try:
            other_mask_bool = _as_bool_mask(other_mask)
        except Exception:
            continue
        if other_mask_bool.shape != target_mask_bool.shape:
            continue
        other_mask_bool = np.logical_and(other_mask_bool, np.logical_not(target_mask_bool))
        if not np.any(other_mask_bool):
            continue
        valid_other_masks.append(other_mask_bool)

    if not valid_other_masks or ring_pixel_count == 0:
        return metrics

    other_union = np.zeros_like(target_mask_bool, dtype=bool)
    for other_mask_bool in valid_other_masks:
        other_union = np.logical_or(other_union, other_mask_bool)

    nearer_depth_mask = np.logical_and(np.isfinite(full_depth), full_depth > 0.0)
    nearer_depth_mask = np.logical_and(
        nearer_depth_mask,
        full_depth < (float(object_depth_median) - float(depth_margin_delta)),
    )
    nearer_ring_pixels = np.logical_and(ring, np.logical_and(other_union, nearer_depth_mask))
    nearer_ring_pixel_count = int(np.count_nonzero(nearer_ring_pixels))
    metrics["nearer_ring_pixel_count"] = nearer_ring_pixel_count

    boundary_hits = np.logical_and(boundary, _dilate_mask(nearer_ring_pixels, int(boundary_neighbor_radius)))
    occluded_boundary_pixel_count = int(np.count_nonzero(boundary_hits))
    metrics["occluded_boundary_pixel_count"] = occluded_boundary_pixel_count

    occluded_boundary_ratio = float(occluded_boundary_pixel_count / max(boundary_pixel_count, 1))
    nearer_ring_overlap_ratio = float(nearer_ring_pixel_count / max(ring_pixel_count, 1))
    visible_occlusion_ratio = float(
        min(
            max((0.7 * occluded_boundary_ratio) + (0.3 * nearer_ring_overlap_ratio), 0.0),
            1.0,
        )
    )

    metrics["occluded_boundary_ratio"] = occluded_boundary_ratio
    metrics["nearer_ring_overlap_ratio"] = nearer_ring_overlap_ratio
    metrics["visible_occlusion_ratio"] = visible_occlusion_ratio
    return metrics
