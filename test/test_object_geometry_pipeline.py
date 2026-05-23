import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import spatial_rag.object_geometry_pipeline as object_geometry_pipeline_module

from spatial_rag.config import BBOX_CONF_THRESHOLD, OCCLUSION_TARGET_OVERLAP_THRESHOLD
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
from spatial_rag.occlusion_scoring import compute_reweighted_detection_score_from_penalty


class _FakeCaptioner:
    def __init__(self):
        self.batched_calls = []
        self.crop_calls = []

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

    def describe_detected_objects_with_meta(self, image_path: str, detections, force_refresh: bool = False):
        self.batched_calls.append(
            {
                "image_path": image_path,
                "detections": list(detections or []),
                "force_refresh": bool(force_refresh),
            }
        )
        objects = []
        for det in list(detections or []):
            objects.append(
                {
                    "object_local_id": det["object_local_id"],
                    "label": det["detector_label"],
                    "short_description": "brown leather chair",
                    "long_description": "brown leather chair near the center of the room",
                    "attributes": ["brown", "leather"],
                    "distance_from_camera_m": 2.1,
                }
            )
        return {
            "objects": objects,
            "raw_response": {"choices": [{"finish_reason": "stop"}]},
            "source": "api",
        }

    def describe_object_crop_with_meta(
        self,
        image_path: str,
        force_refresh: bool = False,
        yolo_label=None,
        yolo_confidence=None,
        include_occlusion: bool = False,
    ):
        self.crop_calls.append(
            {
                "image_path": image_path,
                "force_refresh": bool(force_refresh),
                "yolo_label": yolo_label,
                "yolo_confidence": yolo_confidence,
                "include_occlusion": bool(include_occlusion),
            }
        )
        payload = {
            "label": yolo_label or "chair",
            "short_description": "brown leather chair",
            "long_description": "brown leather chair near the center of the room",
            "attributes": ["brown", "leather"],
            "distance_from_camera_m": 2.1,
        }
        if include_occlusion:
            payload["occlusion_level"] = "moderately occluded"
        return payload


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


class _FakeDINOEmbedder:
    model_name = "fake-dinov3"
    normalize = True

    def encode_crop(self, image):
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)


class _FilteredConfidenceDetector:
    def __init__(self):
        self.class_names = []

    def set_class_names(self, class_names):
        self.class_names = list(class_names or [])

    def detect(self, image):
        return [
            {"label": "chair", "bbox": [700.0, 300.0, 1200.0, 1000.0], "confidence": 0.92},
            {"label": "chair", "bbox": [80.0, 90.0, 220.0, 320.0], "confidence": 0.15},
        ]


class _TwoObjectDetector:
    def __init__(self):
        self.class_names = []

    def set_class_names(self, class_names):
        self.class_names = list(class_names or [])

    def detect(self, image):
        return [
            {"label": "chair", "bbox": [10.0, 10.0, 30.0, 30.0], "confidence": 0.9},
            {"label": "chair", "bbox": [12.0, 12.0, 28.0, 28.0], "confidence": 0.85},
        ]


class _TwoObjectSegmenter:
    def segment(self, image_rgb: np.ndarray, bbox_xyxy):
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        x1 = float(bbox_xyxy[0])
        if x1 < 11.0:
            mask[10:30, 10:30] = True
            mask[12:28, 12:28] = False
        else:
            mask[12:28, 12:28] = True
        return mask


class _TwoObjectDepthEstimator:
    def predict_depth(self, image_path: str, image_rgb: np.ndarray):
        depth = np.full(image_rgb.shape[:2], 2.0, dtype=np.float32)
        depth[12:28, 12:28] = 1.5
        return depth


class _LowConfidenceOccluderDetector:
    def __init__(self):
        self.class_names = []

    def set_class_names(self, class_names):
        self.class_names = list(class_names or [])

    def detect(self, image):
        return [
            {"label": "chair", "bbox": [10.0, 10.0, 30.0, 30.0], "confidence": 0.9},
            {"label": "chair", "bbox": [12.0, 12.0, 28.0, 28.0], "confidence": 0.15},
        ]


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
    assert round(projected_y, 3) == 0.6
    assert round(projected_z, 3) == -0.5


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
    captioner = _FakeCaptioner()

    pipeline = ObjectGeometryPipeline(
        captioner=captioner,
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
    assert row["detector_label_raw"] == "chair"
    assert row["vlm_label"] == "chair"
    assert row["final_label"] == "chair"
    assert row["label_source"] == "vlm"
    assert row["label_conflict"] is False
    assert row["bbox_xywh_norm"][2] > 0.0
    assert row["distance_from_camera_m"] == 2.0
    assert row["projected_planar_distance_m"] >= 2.0
    assert row["relative_bearing_deg"] > 0.0
    assert abs(row["vertical_angle_deg"]) < 30.0
    assert row["depth_stat_median_m"] == 2.0
    assert row["depth_stat_p10_m"] == 2.0
    assert row["visible_occlusion_ratio"] == 0.0
    assert row["occluded_boundary_ratio"] is None
    assert row["nearer_ring_overlap_ratio"] is None
    assert row["object_depth_median"] == 2.0
    assert row["occluding_overlap_pixel_count"] == 0
    assert row["foreground_occluder_count"] == 0
    assert row["occlusion_target_overlap_threshold"] == OCCLUSION_TARGET_OVERLAP_THRESHOLD
    assert row["occlusion_level"] == "fully visible"
    assert row["occlusion_penalty_p_o"] == 0.0
    assert row["reweighted_detection_score_r"] == compute_reweighted_detection_score_from_penalty(0.92, 0.0)
    assert row["crop_path"]
    assert row["mask_path"]
    assert row["mask_overlay_path"]
    assert row["depth_map_path"]
    assert result.timings["object_description_call_count"] == 1
    assert row["occlusion_source"] == "visible_mask"
    assert len(captioner.batched_calls) == 1
    assert captioner.batched_calls[0]["detections"][0]["object_local_id"] == "det_000"
    assert result.artifacts.detections_path
    assert result.artifacts.detection_overlay_path
    assert result.artifacts.filtered_detections_path is None
    assert result.artifacts.filtered_detection_overlay_path is None
    assert result.artifacts.depth_preview_path


def test_object_geometry_pipeline_uses_default_description_when_batched_item_missing(tmp_path):
    class _MissingBatchCaptioner(_FakeCaptioner):
        def describe_detected_objects_with_meta(self, image_path: str, detections, force_refresh: bool = False):
            self.batched_calls.append({"image_path": image_path, "detections": list(detections or [])})
            return {
                "objects": [],
                "raw_response": {"choices": [{"finish_reason": "stop"}]},
                "source": "api",
            }

    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    captioner = _MissingBatchCaptioner()
    pipeline = ObjectGeometryPipeline(
        captioner=captioner,
        output_root=str(tmp_path),
        detector=_FakeDetector(),
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=False,
    )

    result = pipeline.run_for_view(
        entry_id=7,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    assert result.ok is True
    row = result.object_rows[0]
    assert row["label"] == "chair"
    assert row["vlm_label"] == "unknown"
    assert row["final_label"] == "chair"
    assert row["label_source"] == "detector"
    assert row["description"] == "chair"
    assert row["long_form_open_description"] == "chair"
    assert row["attributes"] == []
    assert row["visible_occlusion_ratio"] == 0.0
    assert row["occlusion_level"] == "fully visible"
    assert row["occlusion_penalty_p_o"] == 0.0
    assert result.timings["object_description_call_count"] == 1


def test_object_geometry_pipeline_defer_object_descriptions_skips_vlm_calls(tmp_path):
    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    captioner = _FakeCaptioner()
    pipeline = ObjectGeometryPipeline(
        captioner=captioner,
        output_root=str(tmp_path),
        detector=_FakeDetector(),
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=False,
    )

    result = pipeline.run_for_view(
        entry_id=9,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
        defer_object_descriptions=True,
    )

    assert result.ok is True
    assert result.description_requests[0]["object_local_id"] == "det_000"
    assert result.object_rows[0]["description"] == "chair"
    assert result.object_rows[0]["vlm_label"] == "unknown"
    assert result.object_rows[0]["final_label"] == "chair"
    assert result.object_rows[0]["visible_occlusion_ratio"] == 0.0
    assert result.object_rows[0]["occlusion_level"] == "fully visible"
    assert result.timings["object_description_call_count"] == 0
    assert captioner.batched_calls == []


def test_object_geometry_pipeline_writes_filtered_detection_artifacts_for_low_conf_boxes(tmp_path):
    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok

    pipeline = ObjectGeometryPipeline(
        captioner=_FakeCaptioner(),
        output_root=str(tmp_path),
        detector=_FilteredConfidenceDetector(),
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=True,
    )

    result = pipeline.run_for_view(
        entry_id=10,
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
    assert result.timings["detection_count_raw"] == 2
    assert result.timings["detection_count_class_matched"] == 2
    assert result.timings["detection_count_filtered_by_bbox_conf"] == 1
    assert result.timings["detection_count_kept"] == 1
    assert result.timings["detection_count_truncated_by_max_objects"] == 0
    assert result.artifacts.filtered_detections_path
    assert result.artifacts.filtered_detection_overlay_path

    filtered_path = Path(result.artifacts.filtered_detections_path)
    stored = json.loads(filtered_path.read_text(encoding="utf-8"))
    assert len(stored) == 1
    assert stored[0]["filter_reason"] == "bbox_conf_threshold"
    assert stored[0]["bbox_conf_threshold"] == BBOX_CONF_THRESHOLD
    assert stored[0]["confidence"] == 0.15


def test_object_geometry_pipeline_low_conf_overlap_does_not_occlude_kept_object(tmp_path):
    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((40, 40, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok

    pipeline = ObjectGeometryPipeline(
        captioner=_FakeCaptioner(),
        output_root=str(tmp_path),
        detector=_LowConfidenceOccluderDetector(),
        segmenter=_TwoObjectSegmenter(),
        depth_estimator=_TwoObjectDepthEstimator(),
        save_artifacts=False,
        horizontal_fov_deg=90.0,
        image_width_px=40,
        image_height_px=40,
    )

    result = pipeline.run_for_view(
        entry_id=11,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=0.0,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    assert result.ok is True
    assert len(result.object_rows) == 1
    row = result.object_rows[0]
    assert row["visible_occlusion_ratio"] == 0.0
    assert row["foreground_occluder_count"] == 0
    assert result.timings["detection_count_filtered_by_bbox_conf"] == 1


def test_object_geometry_pipeline_visible_occlusion_uses_nearer_overlapping_bbox_depth(tmp_path):
    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((40, 40, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok

    pipeline = ObjectGeometryPipeline(
        captioner=_FakeCaptioner(),
        output_root=str(tmp_path),
        detector=_TwoObjectDetector(),
        segmenter=_TwoObjectSegmenter(),
        depth_estimator=_TwoObjectDepthEstimator(),
        save_artifacts=False,
        horizontal_fov_deg=90.0,
        image_width_px=40,
        image_height_px=40,
    )

    result = pipeline.run_for_view(
        entry_id=10,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=0.0,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    assert result.ok is True
    rows_by_id = {row["object_local_id"]: row for row in result.object_rows}
    left_row = rows_by_id["det_000"]
    right_row = rows_by_id["det_001"]

    assert left_row["visible_occlusion_ratio"] > 0.0
    assert left_row["occluded_boundary_ratio"] is None
    assert left_row["nearer_ring_overlap_ratio"] is None
    assert left_row["occluding_overlap_pixel_count"] > 0
    assert left_row["foreground_occluder_count"] == 1
    assert left_row["occlusion_target_overlap_threshold"] == OCCLUSION_TARGET_OVERLAP_THRESHOLD
    assert left_row["occlusion_penalty_p_o"] == 0.5 * left_row["visible_occlusion_ratio"]
    assert left_row["reweighted_detection_score_r"] == compute_reweighted_detection_score_from_penalty(
        left_row["detector_confidence"],
        left_row["occlusion_penalty_p_o"],
    )

    assert right_row["visible_occlusion_ratio"] == 0.0
    assert right_row["occlusion_level"] == "fully visible"


def test_object_geometry_pipeline_uses_vlm_occlusion_source(tmp_path):
    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    captioner = _FakeCaptioner()
    pipeline = ObjectGeometryPipeline(
        captioner=captioner,
        output_root=str(tmp_path),
        detector=_FakeDetector(),
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=False,
        occlusion_source="vlm",
    )

    result = pipeline.run_for_view(
        entry_id=11,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    assert result.ok is True
    row = result.object_rows[0]
    assert row["occlusion_source"] == "vlm"
    assert row["occlusion_level"] == "moderately occluded"
    assert row["visible_occlusion_ratio"] is None
    assert row["occluded_boundary_ratio"] is None
    assert row["nearer_ring_overlap_ratio"] is None
    assert row["boundary_pixel_count"] is None
    assert row["ring_pixel_count"] is None
    assert row["occluding_overlap_pixel_count"] is None
    assert row["foreground_occluder_count"] is None
    assert result.timings["object_description_call_count"] == 1
    assert len(captioner.batched_calls) == 0
    assert captioner.crop_calls[0]["include_occlusion"] is True
    assert row["reweighted_detection_score_r"] == compute_reweighted_detection_score_from_penalty(
        row["detector_confidence"],
        row["occlusion_penalty_p_o"],
    )


def test_object_geometry_pipeline_uses_uncertain_when_vlm_occlusion_invalid(tmp_path):
    class _InvalidOcclusionCaptioner(_FakeCaptioner):
        def describe_object_crop_with_meta(
            self,
            image_path: str,
            force_refresh: bool = False,
            yolo_label=None,
            yolo_confidence=None,
            include_occlusion: bool = False,
        ):
            payload = super().describe_object_crop_with_meta(
                image_path,
                force_refresh=force_refresh,
                yolo_label=yolo_label,
                yolo_confidence=yolo_confidence,
                include_occlusion=include_occlusion,
            )
            if include_occlusion:
                payload["occlusion_level"] = "bad-value"
            return payload

    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok

    pipeline = ObjectGeometryPipeline(
        captioner=_InvalidOcclusionCaptioner(),
        output_root=str(tmp_path),
        detector=_FakeDetector(),
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=False,
        occlusion_source="vlm",
    )

    result = pipeline.run_for_view(
        entry_id=12,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    assert result.ok is True
    row = result.object_rows[0]
    assert row["occlusion_level"] == "uncertain"
    assert row["occlusion_penalty_p_o"] == 0.35
    assert row["visible_occlusion_ratio"] is None


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


def test_object_geometry_pipeline_falls_back_to_detector_for_out_of_prelist_vlm_label(tmp_path):
    class _OutOfPrelistRelabelCaptioner(_FakeCaptioner):
        def describe_detected_objects_with_meta(self, image_path: str, detections, force_refresh: bool = False):
            payload = super().describe_detected_objects_with_meta(image_path, detections, force_refresh)
            payload["objects"][0]["label"] = "sofa"
            payload["objects"][0]["short_description"] = "gray sofa"
            payload["objects"][0]["long_description"] = "gray sofa near the center of the room"
            return payload

    image_path = tmp_path / "view.jpg"
    image_rgb = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok

    pipeline = ObjectGeometryPipeline(
        captioner=_OutOfPrelistRelabelCaptioner(),
        output_root=str(tmp_path),
        detector=_FakeDetector(),
        segmenter=_FakeSegmenter(),
        depth_estimator=_FakeDepthEstimator(),
        save_artifacts=False,
    )

    result = pipeline.run_for_view(
        entry_id=13,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    row = result.object_rows[0]
    assert row["label"] == "chair"
    assert row["vlm_label"] == "unknown"
    assert row["final_label"] == "chair"
    assert row["label_source"] == "detector"
    assert row["label_conflict"] is False


def test_object_geometry_pipeline_stores_dinov3_status_and_embedding(tmp_path):
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
        dino_embedder=_FakeDINOEmbedder(),
        enable_dinov3_embedding=True,
        save_artifacts=False,
    )

    result = pipeline.run_for_view(
        entry_id=14,
        image_path=str(image_path),
        image_rgb=image_rgb,
        camera_x=0.0,
        camera_y=1.6,
        camera_z=0.0,
        camera_orientation_deg=0.0,
        max_objects=4,
    )

    row = result.object_rows[0]
    assert row["dinov3_status"] == "success"
    assert row["dinov3_model_name"] == "fake-dinov3"
    assert row["dinov3_embedding_dim"] == 3
    assert row["dinov3_input_type"] == "bbox_crop"
    assert row["dinov3_normalized"] is True
    assert result.timings["dinov3_total_sec"] >= 0.0


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
