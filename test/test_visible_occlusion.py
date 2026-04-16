import math

from spatial_rag.config import OCCLUSION_TARGET_OVERLAP_THRESHOLD
from spatial_rag.visible_occlusion import (
    compute_visible_occlusion_metrics,
    visible_occlusion_ratio_to_level,
    visible_occlusion_ratio_to_penalty,
)


def test_visible_occlusion_zero_when_no_other_objects_overlap_enough():
    metrics = compute_visible_occlusion_metrics(
        target_bbox_xyxy=[10, 10, 30, 30],
        target_depth_m=2.0,
        other_objects=[
            {
                "bbox_xyxy": [28, 28, 40, 40],
                "distance_from_camera_m": 1.5,
            }
        ],
        target_overlap_threshold=OCCLUSION_TARGET_OVERLAP_THRESHOLD,
        depth_margin_delta=0.0,
    )

    assert metrics["visible_occlusion_ratio"] == 0.0
    assert metrics["occluding_overlap_pixel_count"] == 0
    assert metrics["foreground_occluder_count"] == 0


def test_visible_occlusion_positive_for_nearer_overlapping_object():
    metrics = compute_visible_occlusion_metrics(
        target_bbox_xyxy=[10, 10, 30, 30],
        target_depth_m=2.0,
        other_objects=[
            {
                "bbox_xyxy": [12, 12, 28, 28],
                "distance_from_camera_m": 1.5,
            }
        ],
        target_overlap_threshold=OCCLUSION_TARGET_OVERLAP_THRESHOLD,
        depth_margin_delta=0.0,
    )

    assert metrics["visible_occlusion_ratio"] > 0.0
    assert metrics["occluding_overlap_pixel_count"] == 16 * 16
    assert metrics["foreground_occluder_count"] == 1
    assert math.isclose(metrics["visible_occlusion_ratio"], (16 * 16) / (20 * 20))


def test_visible_occlusion_ignores_farther_overlapping_object():
    metrics = compute_visible_occlusion_metrics(
        target_bbox_xyxy=[10, 10, 30, 30],
        target_depth_m=2.0,
        other_objects=[
            {
                "bbox_xyxy": [12, 12, 28, 28],
                "distance_from_camera_m": 2.6,
            }
        ],
        target_overlap_threshold=OCCLUSION_TARGET_OVERLAP_THRESHOLD,
        depth_margin_delta=0.0,
    )

    assert metrics["visible_occlusion_ratio"] == 0.0
    assert metrics["foreground_occluder_count"] == 0


def test_visible_occlusion_respects_target_overlap_threshold_boundary():
    target_bbox = [10, 10, 30, 30]
    other_bbox = [29, 10, 31, 30]
    metrics = compute_visible_occlusion_metrics(
        target_bbox_xyxy=target_bbox,
        target_depth_m=2.0,
        other_objects=[
            {
                "bbox_xyxy": other_bbox,
                "distance_from_camera_m": 1.5,
            }
        ],
        target_overlap_threshold=OCCLUSION_TARGET_OVERLAP_THRESHOLD,
        depth_margin_delta=0.0,
    )

    assert metrics["visible_occlusion_ratio"] == 0.0
    assert metrics["occluding_overlap_pixel_count"] == 0


def test_visible_occlusion_unions_multiple_foreground_occluders_without_double_counting():
    metrics = compute_visible_occlusion_metrics(
        target_bbox_xyxy=[10, 10, 30, 30],
        target_depth_m=2.0,
        other_objects=[
            {
                "bbox_xyxy": [10, 10, 22, 30],
                "distance_from_camera_m": 1.7,
            },
            {
                "bbox_xyxy": [18, 10, 30, 30],
                "distance_from_camera_m": 1.6,
            },
        ],
        target_overlap_threshold=OCCLUSION_TARGET_OVERLAP_THRESHOLD,
        depth_margin_delta=0.0,
    )

    assert metrics["foreground_occluder_count"] == 2
    assert metrics["occluding_overlap_pixel_count"] == 20 * 20
    assert metrics["visible_occlusion_ratio"] == 1.0


def test_visible_occlusion_bucket_and_penalty_mapping_are_continuous_derivatives():
    assert visible_occlusion_ratio_to_level(0.05) == "fully visible"
    assert visible_occlusion_ratio_to_level(0.2) == "slightly occluded"
    assert visible_occlusion_ratio_to_level(0.4) == "moderately occluded"
    assert visible_occlusion_ratio_to_level(0.8) == "heavily occluded"
    assert math.isclose(visible_occlusion_ratio_to_penalty(0.6), 0.3)


def test_visible_occlusion_default_threshold_comes_from_config():
    metrics = compute_visible_occlusion_metrics(
        target_bbox_xyxy=[10, 10, 30, 30],
        target_depth_m=2.0,
        other_objects=[],
    )

    assert metrics["occlusion_target_overlap_threshold"] == OCCLUSION_TARGET_OVERLAP_THRESHOLD
