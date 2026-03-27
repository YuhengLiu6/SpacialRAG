import json
import sys
import types

import cv2
import numpy as np
import spatial_rag.object_geometry_pipeline as object_geometry_pipeline_module

from spatial_rag.object_geometry_pipeline import (
    NanoSAMMaskRefiner,
    ObjectGeometryPipeline,
    GeometryPipelineUnavailable,
    _validate_tensorrt_engine,
    mask_depth_stats,
    pixel_center_to_relative_angles_deg,
    planar_distance_from_forward_depth_m,
    project_global_xyz_from_geometry,
    relative_height_from_forward_depth_m,
)


class _FakeCaptioner:
    def select_object_types_with_meta(self, image_path: str, force_refresh: bool = False, camera_context=None):
        payload = {
            "view_type": "living room",
            "room_function": "resting",
            "style_hint": "traditional",
            "clutter_level": "low",
            "scene_attributes": ["beige wall"],
            "floor_pattern": "carpet",
            "lighting_ceiling": "mixed lighting",
            "wall_color": "beige",
            "additional_notes": "",
            "image_summary": "A living room scene.",
            "selected_object_types": ["chair"],
        }
        return {
            "payload": payload,
            "raw_json": json.dumps(payload, ensure_ascii=True),
            "raw_api_response": {"choices": [{"finish_reason": "stop"}]},
            "source": "api",
        }

    def describe_object_crop_with_meta(self, image_path: str, force_refresh: bool = False, yolo_label=None, yolo_confidence=None):
        return {
            "label": yolo_label or "chair",
            "short_description": "brown leather chair",
            "long_description": "brown leather chair near the center of the room",
            "attributes": ["brown", "leather"],
            "distance_from_camera_m": 2.1,
        }


class _FakeDetector:
    def __init__(self):
        self.class_names = []

    def set_class_names(self, class_names):
        self.class_names = list(class_names or [])

    def detect(self, image):
        return [{"label": "chair", "bbox": [700.0, 300.0, 1200.0, 1000.0], "confidence": 0.92}]


class _FakeSegmenter:
    def segment(self, image_rgb: np.ndarray, bbox_xyxy):
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[320:980, 760:1180] = True
        return mask


class _FakeDepthEstimator:
    def predict_depth(self, image_path: str, image_rgb: np.ndarray):
        depth = np.full(image_rgb.shape[:2], 2.0, dtype=np.float32)
        depth[350:360, 770:780] = 25.0
        return depth


def test_pixel_center_to_relative_angles_follow_negative_left_positive_right():
    left_angle, left_vertical = pixel_center_to_relative_angles_deg(
        0.0,
        540.0,
        width_px=1920,
        height_px=1080,
        horizontal_fov_deg=90.0,
    )
    right_angle, right_vertical = pixel_center_to_relative_angles_deg(
        1919.0,
        540.0,
        width_px=1920,
        height_px=1080,
        horizontal_fov_deg=90.0,
    )
    center_angle, _center_vertical = pixel_center_to_relative_angles_deg(
        959.5,
        540.0,
        width_px=1920,
        height_px=1080,
        horizontal_fov_deg=90.0,
    )

    assert left_angle < -40.0
    assert right_angle > 40.0
    assert abs(center_angle) < 0.1
    assert abs(left_vertical) < 1.0
    assert abs(right_vertical) < 1.0


def test_project_global_xyz_from_geometry_uses_negative_left_positive_right():
    projected_x, projected_y, projected_z = project_global_xyz_from_geometry(
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=270.0,
        distance_m=2.0,
        relative_bearing_deg=-30.0,
        relative_height_from_camera_m=-0.5,
    )

    assert round(projected_x, 3) == 1.732
    assert round(projected_z, 3) == -1.0
    assert round(projected_y, 3) == 1.1


def test_mask_depth_stats_uses_trimmed_median_inside_mask():
    depth = np.full((6, 6), 2.0, dtype=np.float32)
    depth[2, 2] = 40.0
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True

    stats = mask_depth_stats(depth, mask, trim_fraction=0.10)

    assert stats["num_valid_px"] == 16
    assert stats["median_m"] == 2.0
    assert stats["trimmed_median_m"] == 2.0
    assert stats["p90_m"] >= 2.0


def test_forward_depth_projection_helpers_use_pinhole_geometry():
    planar = planar_distance_from_forward_depth_m(2.0, 30.0)
    rel_height = relative_height_from_forward_depth_m(2.0, -15.0)

    assert round(planar, 3) == 2.309
    assert round(rel_height, 3) == -0.536


def test_nanosam_mask_refiner_accepts_engine_style_predictor_signature(tmp_path, monkeypatch):
    calls = {}

    class _FakePredictor:
        def __init__(self, image_encoder_engine: str, mask_decoder_engine: str):
            calls["image_encoder_engine"] = image_encoder_engine
            calls["mask_decoder_engine"] = mask_decoder_engine

    fake_nanosam = types.ModuleType("nanosam")
    fake_utils = types.ModuleType("nanosam.utils")
    fake_predictor = types.ModuleType("nanosam.utils.predictor")
    fake_predictor.Predictor = _FakePredictor
    class _FakeLogger:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeRuntime:
        def __init__(self, _logger):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def deserialize_cuda_engine(self, _engine_bytes):
            return object()

    fake_trt = types.SimpleNamespace(
        Logger=_FakeLogger,
        Runtime=_FakeRuntime,
    )
    monkeypatch.setitem(sys.modules, "nanosam", fake_nanosam)
    monkeypatch.setitem(sys.modules, "nanosam.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "nanosam.utils.predictor", fake_predictor)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    encoder_path = tmp_path / "encoder.engine"
    decoder_path = tmp_path / "decoder.engine"
    encoder_path.write_bytes(b"encoder")
    decoder_path.write_bytes(b"decoder")

    NanoSAMMaskRefiner(
        image_encoder=str(encoder_path),
        mask_decoder=str(decoder_path),
    )

    assert calls == {
        "image_encoder_engine": str(encoder_path),
        "mask_decoder_engine": str(decoder_path),
    }


def test_validate_tensorrt_engine_fails_fast_on_incompatible_plan(tmp_path, monkeypatch):
    class _FakeLogger:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeRuntime:
        def __init__(self, _logger):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def deserialize_cuda_engine(self, _engine_bytes):
            return None

    fake_trt = types.SimpleNamespace(
        Logger=_FakeLogger,
        Runtime=_FakeRuntime,
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    engine_path = tmp_path / "broken.engine"
    engine_path.write_bytes(b"broken-plan")

    try:
        _validate_tensorrt_engine(str(engine_path), "image encoder")
        assert False, "expected GeometryPipelineUnavailable"
    except GeometryPipelineUnavailable as exc:
        assert "could not be deserialized" in str(exc)


def test_object_geometry_pipeline_success_writes_expected_artifacts(tmp_path):
    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok

    pipeline = ObjectGeometryPipeline(
        captioner=_FakeCaptioner(),
        output_root=str(tmp_path),
        detector=_FakeDetector(),
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=True,
    )

    result = pipeline.run_for_view(
        entry_id=5,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    assert result.ok is True
    assert len(result.object_rows) == 1
    row = result.object_rows[0]
    assert row["geometry_source"] == "mask_depth"
    assert row["label"] == "chair"
    assert row["bbox_xywh_norm"][2] > 0.0
    assert row["distance_from_camera_m"] == 2.0
    assert row["projected_planar_distance_m"] >= 2.0
    assert row["relative_bearing_deg"] > 0.0
    assert abs(row["vertical_angle_deg"]) < 30.0
    assert row["depth_stat_median_m"] == 2.0
    assert row["depth_stat_p10_m"] == 2.0
    assert row["crop_path"]
    assert row["mask_path"]
    assert row["mask_overlay_path"]
    assert row["depth_map_path"]
    assert result.artifacts.detections_path
    assert result.artifacts.detection_overlay_path
    assert result.artifacts.depth_preview_path


def test_object_geometry_pipeline_returns_failure_when_selector_subset_empty(tmp_path):
    class _EmptySelectorCaptioner(_FakeCaptioner):
        def select_object_types_with_meta(self, image_path: str, force_refresh: bool = False, camera_context=None):
            payload = dict(super().select_object_types_with_meta(image_path, force_refresh, camera_context)["payload"])
            payload["selected_object_types"] = []
            return {
                "payload": payload,
                "raw_json": json.dumps(payload, ensure_ascii=True),
                "raw_api_response": None,
                "source": "api",
            }

    image_rgb = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    image_path = tmp_path / "view.jpg"
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok

    pipeline = ObjectGeometryPipeline(
        captioner=_EmptySelectorCaptioner(),
        output_root=str(tmp_path),
        detector=_FakeDetector(),
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=False,
    )

    result = pipeline.run_for_view(
        entry_id=1,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    assert result.ok is False
    assert result.failure_reason == "empty_selected_object_types"


def test_internal_detector_is_recreated_when_class_list_changes(tmp_path, monkeypatch):
    created_class_lists = []

    class _FactoryDetector:
        def __init__(self, detector_type=None, class_names=None):
            self.detector_type = detector_type
            self.class_names = list(class_names or [])
            created_class_lists.append(list(self.class_names))

    monkeypatch.setattr(object_geometry_pipeline_module, "Detector", _FactoryDetector)

    pipeline = ObjectGeometryPipeline(
        captioner=_FakeCaptioner(),
        output_root=str(tmp_path),
        detector=None,
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=False,
    )

    first = pipeline._ensure_detector(["chair"])
    second = pipeline._ensure_detector(["table"])
    third = pipeline._ensure_detector(["table"])

    assert created_class_lists == [["chair"], ["table"]]
    assert first is not second
    assert second is third
