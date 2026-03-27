import numpy as np
import pytest

from spatial_rag.object_localization_query import (
    _aggregate_entry_scores,
    _best_object_match_per_entry,
    _bucket_orientation_deg,
    _laterality_from_bbox,
    _project_object_xy,
    _resolve_query_text,
    _select_query_text_for_mode,
    compose_object_query_overlay,
    crop_with_padding,
    filter_valid_detections,
    select_detection,
)


def test_filter_valid_detections_applies_conf_size_and_area_filters():
    detections = [
        {"label": "chair", "confidence": 0.90, "bbox": [10, 10, 90, 90]},
        {"label": "lamp", "confidence": 0.20, "bbox": [10, 10, 70, 70]},
        {"label": "cup", "confidence": 0.95, "bbox": [0, 0, 12, 12]},
        {"label": "wall", "confidence": 0.99, "bbox": [0, 0, 200, 200]},
    ]

    valid = filter_valid_detections(
        detections=detections,
        image_shape=(200, 200, 3),
        detector_conf=0.35,
        min_bbox_side_px=32,
        min_bbox_area_ratio=0.005,
        max_bbox_area_ratio=0.60,
    )

    assert len(valid) == 1
    assert valid[0]["label"] == "chair"
    assert valid[0]["bbox_xyxy"] == [10, 10, 90, 90]


def test_select_detection_is_reproducible_for_same_seed():
    valid = [
        {"label": "b", "confidence": 0.8, "bbox_xyxy": [20, 20, 60, 60], "area_ratio": 0.04},
        {"label": "a", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50], "area_ratio": 0.04},
        {"label": "c", "confidence": 0.7, "bbox_xyxy": [30, 30, 70, 70], "area_ratio": 0.04},
    ]

    first = select_detection(valid, selection_seed=123)
    second = select_detection(valid, selection_seed=123)

    assert first == second


def test_crop_with_padding_expands_and_clamps_bbox():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop, padded_bbox = crop_with_padding(image, [10, 10, 20, 20], padding_ratio=0.1)

    assert padded_bbox == [9, 9, 21, 21]
    assert crop.shape == (12, 12, 3)


def test_crop_with_padding_larger_ratio_improves_context_window():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop, padded_bbox = crop_with_padding(image, [10, 10, 20, 20], padding_ratio=0.25)

    assert padded_bbox == [8, 8, 22, 22]
    assert crop.shape == (14, 14, 3)


def test_aggregate_entry_scores_uses_max_hit_per_entry():
    scores = np.asarray([[0.90, 0.70, 0.80]], dtype=np.float32)
    indices = np.asarray([[0, 1, 2]], dtype=np.int64)
    object_meta = [
        {"entry_id": 11},
        {"entry_id": 12},
        {"entry_id": 11},
    ]

    out = _aggregate_entry_scores(
        search_scores=scores,
        search_indices=indices,
        object_meta=object_meta,
        entry_ids=[11, 12, 13],
    )

    assert out[11] == pytest.approx(0.9)
    assert out[12] == pytest.approx(0.7)
    assert out[13] == 0.0


def test_best_object_match_per_entry_returns_best_object_index():
    scores = np.asarray([[0.90, 0.70, 0.80]], dtype=np.float32)
    indices = np.asarray([[0, 1, 2]], dtype=np.int64)
    object_meta = [
        {"entry_id": 11, "label": "chair"},
        {"entry_id": 12, "label": "lamp"},
        {"entry_id": 11, "label": "table"},
    ]

    out = _best_object_match_per_entry(
        search_scores=scores,
        search_indices=indices,
        object_meta=object_meta,
        entry_ids=[11, 12, 13],
    )

    assert out[11]["object_index"] == 0
    assert out[11]["object_meta"]["label"] == "chair"
    assert out[12]["object_index"] == 1
    assert out[13] == {}


def test_query_bbox_laterality_uses_horizontal_thirds():
    image_width = 90
    assert _laterality_from_bbox([0, 0, 20, 20], image_width=image_width) == "left"
    assert _laterality_from_bbox([35, 0, 55, 20], image_width=image_width) == "center"
    assert _laterality_from_bbox([70, 0, 89, 20], image_width=image_width) == "right"


def test_project_object_xy_uses_bucket_orientation():
    left_orientation = _bucket_orientation_deg(frame_orientation_deg=0.0, laterality="left", angle_step_deg=30)
    obj_xy = _project_object_xy(origin_x=0.0, origin_y=0.0, orientation_deg=left_orientation, distance_m=2.0)

    assert left_orientation == pytest.approx(30.0)
    assert obj_xy is not None
    assert obj_xy[0] == pytest.approx(-1.0, abs=1e-3)
    assert obj_xy[1] == pytest.approx(-1.732, abs=1e-3)


def test_bucket_orientation_deg_rotates_right_bucket_clockwise():
    assert _bucket_orientation_deg(frame_orientation_deg=270.0, laterality="right", angle_step_deg=30) == pytest.approx(240.0)
    assert _bucket_orientation_deg(frame_orientation_deg=270.0, laterality="left", angle_step_deg=30) == pytest.approx(300.0)


def test_resolve_query_text_falls_back_to_yolo_label():
    query_text, query_text_long, source = _resolve_query_text(
        {"label": "unknown", "short_description": "unknown", "long_description": "", "source": "default-no-client"},
        yolo_label="chair",
    )

    assert query_text == "chair"
    assert query_text_long == "chair"
    assert source == "fallback_label"


def test_resolve_query_text_keeps_api_short_description_when_present():
    query_text, query_text_long, source = _resolve_query_text(
        {
            "label": "chair",
            "short_description": "ornate wooden chair",
            "long_description": "",
            "source": "api",
        },
        yolo_label="chair",
    )

    assert query_text == "ornate wooden chair"
    assert query_text_long == "ornate wooden chair"
    assert source == "api"


def test_resolve_query_text_falls_back_to_short_when_long_unknown():
    query_text, query_text_long, source = _resolve_query_text(
        {
            "label": "chair",
            "short_description": "edge-cropped chair",
            "long_description": "unknown",
            "source": "api",
        },
        yolo_label="chair",
    )

    assert query_text == "edge-cropped chair"
    assert query_text_long == "edge-cropped chair"
    assert source == "api"


def test_select_query_text_for_mode_uses_long_when_requested():
    assert (
        _select_query_text_for_mode(
            query_text_short="chair",
            query_text_long="dark wooden chair at image edge",
            object_text_mode="long",
        )
        == "dark wooden chair at image edge"
    )
    assert (
        _select_query_text_for_mode(
            query_text_short="chair",
            query_text_long="unknown",
            object_text_mode="long",
        )
        == "chair"
    )


def test_compose_object_query_overlay_contains_selected_bbox_region():
    map_canvas = np.zeros((700, 900, 3), dtype=np.uint8)
    query_rgb = np.full((320, 480, 3), 255, dtype=np.uint8)
    detections = [
        {"label": "chair", "confidence": 0.9, "bbox_xyxy": [60, 80, 200, 240], "area_ratio": 0.15},
        {"label": "lamp", "confidence": 0.7, "bbox_xyxy": [260, 60, 340, 180], "area_ratio": 0.08},
    ]
    selected = detections[0]
    crop_rgb = np.full((140, 160, 3), 180, dtype=np.uint8)
    top_k_records = [
        {"rank": 1, "distance_to_query_object_m_approx": 0.42, "object_score": 0.8123},
        {"rank": 2, "distance_to_query_object_m_approx": 0.77, "object_score": 0.7011},
        {"rank": 3, "distance_to_query_object_m_approx": 1.15, "object_score": 0.6502},
        {"rank": 4, "distance_to_query_object_m_approx": 1.63, "object_score": 0.6014},
        {"rank": 5, "distance_to_query_object_m_approx": 2.08, "object_score": 0.5809},
    ]

    overlay = compose_object_query_overlay(
        map_canvas_bgr=map_canvas,
        query_rgb=query_rgb,
        detections=detections,
        selected_detection=selected,
        crop_rgb=crop_rgb,
        query_text="ornate wooden chair",
        top1_score=0.8123,
        top_k_records=top_k_records,
    )

    assert overlay.shape == (1200, 2200, 3)
    right_top = overlay[:650, 1300:, :]
    red_mask = (right_top[:, :, 2] > 200) & (right_top[:, :, 1] < 120) & (right_top[:, :, 0] < 120)
    assert int(np.count_nonzero(red_mask)) > 100
