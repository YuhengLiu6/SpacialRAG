import math

import numpy as np

from spatial_rag.visible_occlusion import (
    compute_visible_occlusion_metrics,
    visible_occlusion_ratio_to_level,
    visible_occlusion_ratio_to_penalty,
)


def _square_mask(size: int = 40, *, y1: int = 12, y2: int = 28, x1: int = 12, x2: int = 28) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_visible_occlusion_zero_when_no_other_objects_nearby():
    target_mask = _square_mask()
    depth = np.full(target_mask.shape, 2.0, dtype=np.float32)

    metrics = compute_visible_occlusion_metrics(
        target_mask=target_mask,
        target_depth_map=depth,
        other_object_masks=[],
        depth_map=depth,
    )

    assert metrics["visible_occlusion_ratio"] == 0.0
    assert metrics["occluded_boundary_ratio"] == 0.0
    assert metrics["nearer_ring_overlap_ratio"] == 0.0


def test_visible_occlusion_positive_for_nearer_object_touching_boundary():
    target_mask = _square_mask()
    occluder_mask = np.zeros_like(target_mask)
    occluder_mask[12:28, 28:34] = True
    depth = np.full(target_mask.shape, 2.0, dtype=np.float32)
    depth[occluder_mask] = 1.6

    metrics = compute_visible_occlusion_metrics(
        target_mask=target_mask,
        target_depth_map=depth,
        other_object_masks=[occluder_mask],
        depth_map=depth,
        ring_radius=3,
        depth_margin_delta=0.08,
        boundary_neighbor_radius=1,
    )

    assert metrics["visible_occlusion_ratio"] > 0.0
    assert metrics["occluded_boundary_ratio"] > 0.0
    assert metrics["nearer_ring_overlap_ratio"] > 0.0
    assert math.isclose(
        metrics["visible_occlusion_ratio"],
        (0.7 * metrics["occluded_boundary_ratio"]) + (0.3 * metrics["nearer_ring_overlap_ratio"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_visible_occlusion_ignores_farther_object_touching_boundary():
    target_mask = _square_mask()
    farther_mask = np.zeros_like(target_mask)
    farther_mask[12:28, 28:34] = True
    depth = np.full(target_mask.shape, 2.0, dtype=np.float32)
    depth[farther_mask] = 2.6

    metrics = compute_visible_occlusion_metrics(
        target_mask=target_mask,
        target_depth_map=depth,
        other_object_masks=[farther_mask],
        depth_map=depth,
        ring_radius=3,
    )

    assert metrics["visible_occlusion_ratio"] == 0.0
    assert metrics["nearer_ring_pixel_count"] == 0


def test_visible_occlusion_ignores_background_only_near_boundary():
    target_mask = _square_mask()
    depth = np.full(target_mask.shape, 2.0, dtype=np.float32)
    depth[:, 30:] = 1.0

    metrics = compute_visible_occlusion_metrics(
        target_mask=target_mask,
        target_depth_map=depth,
        other_object_masks=[],
        depth_map=depth,
        ring_radius=3,
    )

    assert metrics["visible_occlusion_ratio"] == 0.0


def test_visible_occlusion_safe_for_invalid_depth_and_tiny_masks():
    target_mask = np.zeros((10, 10), dtype=bool)
    target_mask[5, 5] = True
    depth = np.full(target_mask.shape, np.nan, dtype=np.float32)

    metrics = compute_visible_occlusion_metrics(
        target_mask=target_mask,
        target_depth_map=depth,
        other_object_masks=[],
        depth_map=depth,
    )

    assert metrics["visible_occlusion_ratio"] == 0.0
    assert metrics["object_depth_median"] is None
    assert metrics["boundary_pixel_count"] == 0


def test_visible_occlusion_does_not_count_border_truncation_without_nearer_object():
    target_mask = np.zeros((30, 30), dtype=bool)
    target_mask[:, :8] = True
    depth = np.full(target_mask.shape, 2.0, dtype=np.float32)

    metrics = compute_visible_occlusion_metrics(
        target_mask=target_mask,
        target_depth_map=depth,
        other_object_masks=[],
        depth_map=depth,
        ring_radius=4,
    )

    assert metrics["visible_occlusion_ratio"] == 0.0


def test_visible_occlusion_bucket_and_penalty_mapping_are_continuous_derivatives():
    assert visible_occlusion_ratio_to_level(0.05) == "fully visible"
    assert visible_occlusion_ratio_to_level(0.2) == "slightly occluded"
    assert visible_occlusion_ratio_to_level(0.4) == "moderately occluded"
    assert visible_occlusion_ratio_to_level(0.8) == "heavily occluded"
    assert math.isclose(visible_occlusion_ratio_to_penalty(0.6), 0.3)
