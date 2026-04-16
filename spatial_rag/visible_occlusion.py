from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _bbox_xyxy_ints(bbox_xyxy: Any) -> Optional[Tuple[int, int, int, int]]:
    values = np.asarray(bbox_xyxy).reshape(-1)
    if values.size < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in values[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_area(bbox_xyxy: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox_xyxy
    return int(max(x2 - x1, 0) * max(y2 - y1, 0))


def _bbox_intersection(
    left: Tuple[int, int, int, int],
    right: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _zero_visible_occlusion_metrics(
    *,
    target_depth_m: Optional[float],
    depth_margin_delta: float,
    target_overlap_threshold: float,
) -> Dict[str, Any]:
    return {
        "visible_occlusion_ratio": 0.0,
        "occluded_boundary_ratio": None,
        "nearer_ring_overlap_ratio": None,
        "object_depth_median": target_depth_m,
        "boundary_pixel_count": None,
        "occluded_boundary_pixel_count": None,
        "ring_pixel_count": None,
        "nearer_ring_pixel_count": None,
        "depth_margin_delta": float(depth_margin_delta),
        "occluding_overlap_pixel_count": 0,
        "foreground_occluder_count": 0,
        "occlusion_target_overlap_threshold": float(target_overlap_threshold),
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
    target_bbox_xyxy: Sequence[float],
    target_depth_m: Any,
    other_objects: Sequence[Mapping[str, Any]],
    *,
    target_overlap_threshold: float = 0.1,
    depth_margin_delta: float = 0.0,
) -> Dict[str, Any]:
    target_bbox = _bbox_xyxy_ints(target_bbox_xyxy)
    target_depth = _safe_float(target_depth_m)
    metrics = _zero_visible_occlusion_metrics(
        target_depth_m=target_depth,
        depth_margin_delta=float(depth_margin_delta),
        target_overlap_threshold=float(target_overlap_threshold),
    )
    if target_bbox is None or target_depth is None:
        return metrics

    target_area = _bbox_area(target_bbox)
    if target_area <= 0:
        return metrics

    target_width = target_bbox[2] - target_bbox[0]
    target_height = target_bbox[3] - target_bbox[1]
    overlap_union = np.zeros((target_height, target_width), dtype=bool)
    foreground_occluder_count = 0
    threshold = float(target_overlap_threshold)
    depth_margin = float(depth_margin_delta)

    for other in list(other_objects or []):
        if not isinstance(other, Mapping):
            continue
        other_bbox = _bbox_xyxy_ints(other.get("bbox_xyxy") or other.get("bbox"))
        if other_bbox is None:
            continue
        intersection_bbox = _bbox_intersection(target_bbox, other_bbox)
        if intersection_bbox is None:
            continue
        intersection_area = _bbox_area(intersection_bbox)
        if intersection_area <= 0:
            continue
        target_overlap_ratio = float(intersection_area / max(target_area, 1))
        if target_overlap_ratio < threshold:
            continue

        other_depth = _safe_float(other.get("object_depth_median"))
        if other_depth is None:
            other_depth = _safe_float(other.get("distance_from_camera_m"))
        if other_depth is None:
            continue
        if not (other_depth < (target_depth - depth_margin)):
            continue

        foreground_occluder_count += 1
        rel_x1 = intersection_bbox[0] - target_bbox[0]
        rel_y1 = intersection_bbox[1] - target_bbox[1]
        rel_x2 = intersection_bbox[2] - target_bbox[0]
        rel_y2 = intersection_bbox[3] - target_bbox[1]
        overlap_union[rel_y1:rel_y2, rel_x1:rel_x2] = True

    occluding_overlap_pixel_count = int(np.count_nonzero(overlap_union))
    metrics["occluding_overlap_pixel_count"] = occluding_overlap_pixel_count
    metrics["foreground_occluder_count"] = int(foreground_occluder_count)
    metrics["visible_occlusion_ratio"] = float(occluding_overlap_pixel_count / max(target_area, 1))
    return metrics
