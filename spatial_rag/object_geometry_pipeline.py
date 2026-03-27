from __future__ import annotations

import json
import inspect
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from spatial_rag.config import (
    DEPTH_PRO_MODEL_PATH,
    FOV,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NANOSAM_DECODER_PATH,
    NANOSAM_ENCODER_PATH,
)
from spatial_rag.detector import Detector
from spatial_rag.household_taxonomy import canonicalize_household_object_label, normalize_selector_subset


class GeometryPipelineUnavailable(RuntimeError):
    pass


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _vertical_fov_deg(horizontal_fov_deg: float, width_px: int, height_px: int) -> float:
    hfov_rad = math.radians(float(horizontal_fov_deg))
    return math.degrees(
        2.0 * math.atan(math.tan(hfov_rad / 2.0) * float(height_px) / max(float(width_px), 1.0))
    )


def pixel_center_to_relative_angles_deg(
    x_px: float,
    y_px: float,
    *,
    width_px: int,
    height_px: int,
    horizontal_fov_deg: float,
) -> Tuple[float, float]:
    width_f = max(float(width_px), 1.0)
    height_f = max(float(height_px), 1.0)
    cx = (width_f - 1.0) / 2.0
    cy = (height_f - 1.0) / 2.0
    fx = width_f / (2.0 * math.tan(math.radians(float(horizontal_fov_deg)) / 2.0))
    vfov_deg = _vertical_fov_deg(horizontal_fov_deg=float(horizontal_fov_deg), width_px=width_px, height_px=height_px)
    fy = height_f / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    horizontal_angle = math.degrees(math.atan((float(x_px) - cx) / max(fx, 1e-6)))
    vertical_angle = math.degrees(math.atan((cy - float(y_px)) / max(fy, 1e-6)))
    return float(horizontal_angle), float(vertical_angle)


def project_global_xyz_from_geometry(
    *,
    camera_x: float,
    camera_y: float,
    camera_z: float,
    camera_orientation_deg: float,
    distance_m: Optional[float],
    relative_bearing_deg: Optional[float],
    relative_height_from_camera_m: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if distance_m is None or relative_bearing_deg is None:
        return None, None, None
    dist = _safe_float(distance_m)
    bearing = _safe_float(relative_bearing_deg)
    if dist is None or bearing is None or dist < 0.0:
        return None, None, None
    global_bearing = (float(camera_orientation_deg) - float(bearing)) % 360.0
    yaw = math.radians(global_bearing)
    projected_x = float(camera_x - math.sin(yaw) * dist)
    projected_z = float(camera_z - math.cos(yaw) * dist)
    projected_y = None
    rel_h = _safe_float(relative_height_from_camera_m)
    if rel_h is not None:
        projected_y = float(camera_y + rel_h)
    return projected_x, projected_y, projected_z


def planar_distance_from_forward_depth_m(forward_depth_m: Optional[float], relative_bearing_deg: Optional[float]) -> Optional[float]:
    depth = _safe_float(forward_depth_m)
    bearing = _safe_float(relative_bearing_deg)
    if depth is None or bearing is None or depth <= 0.0:
        return None
    cos_h = math.cos(math.radians(float(bearing)))
    if abs(cos_h) < 1e-6:
        return None
    return float(depth / cos_h)


def relative_height_from_forward_depth_m(forward_depth_m: Optional[float], vertical_angle_deg: Optional[float]) -> Optional[float]:
    depth = _safe_float(forward_depth_m)
    angle = _safe_float(vertical_angle_deg)
    if depth is None or angle is None:
        return None
    return float(depth * math.tan(math.radians(float(angle))))


def mask_depth_stats(depth_map_m: np.ndarray, mask: np.ndarray, trim_fraction: float = 0.10) -> Dict[str, Optional[float]]:
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


def bbox_xywh_norm_from_xyxy(bbox_xyxy: Sequence[float], *, width_px: int, height_px: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    width_f = max(float(width_px), 1.0)
    height_f = max(float(height_px), 1.0)
    x = max(0.0, min(1.0, x1 / width_f))
    y = max(0.0, min(1.0, y1 / height_f))
    w = max(0.0, min(1.0, (x2 - x1) / width_f))
    h = max(0.0, min(1.0, (y2 - y1) / height_f))
    return [float(x), float(y), float(w), float(h)]


def relative_bins_from_geometry(
    *,
    centroid_x_px: float,
    centroid_y_px: float,
    width_px: int,
    height_px: int,
    distance_m: Optional[float],
) -> Dict[str, str]:
    x_norm = 0.0 if width_px <= 0 else float(centroid_x_px) / float(width_px)
    y_norm = 0.0 if height_px <= 0 else float(centroid_y_px) / float(height_px)
    laterality = "left" if x_norm < 0.34 else ("center" if x_norm < 0.67 else "right")
    verticality = "high" if y_norm < 0.34 else ("middle" if y_norm < 0.67 else "low")
    dist = _safe_float(distance_m)
    if dist is None:
        distance_bin = "middle"
    elif dist < 1.5:
        distance_bin = "near"
    elif dist < 4.0:
        distance_bin = "middle"
    else:
        distance_bin = "far"
    return {
        "laterality": laterality,
        "verticality": verticality,
        "distance_bin": distance_bin,
    }


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    mask_arr = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask_arr)
    if xs.size == 0 or ys.size == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def crop_image_from_bbox(image_rgb: np.ndarray, bbox_xyxy: Sequence[float]) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy[:4]]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return np.asarray(image_rgb[y1:y2, x1:x2]).copy()


def depth_preview_u8(depth_map_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map_m, dtype=np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if valid.size == 0:
        return cv2.applyColorMap(preview, cv2.COLORMAP_VIRIDIS)
    lo = float(np.percentile(valid, 5.0))
    hi = float(np.percentile(valid, 95.0))
    if hi <= lo:
        hi = lo + 1e-3
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    preview = np.asarray(normalized * 255.0, dtype=np.uint8)
    return cv2.applyColorMap(preview, cv2.COLORMAP_VIRIDIS)


def _validate_tensorrt_engine(engine_path: str, engine_kind: str) -> None:
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:
        raise GeometryPipelineUnavailable(
            "TensorRT is required for NanoSAM engine validation."
        ) from exc

    try:
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
    except Exception as exc:
        raise GeometryPipelineUnavailable(
            f"NanoSAM {engine_kind} engine failed to deserialize: {exc}"
        ) from exc

    if engine is None:
        raise GeometryPipelineUnavailable(
            f"NanoSAM {engine_kind} engine could not be deserialized by the current TensorRT runtime. "
            "This usually means the plan file was built with a different TensorRT/CUDA stack. "
            f"Rebuild this engine on the target machine: {engine_path}"
        )


class NanoSAMMaskRefiner:
    def __init__(
        self,
        image_encoder: Optional[str] = None,
        mask_decoder: Optional[str] = None,
    ):
        try:
            from nanosam.utils.predictor import Predictor  # type: ignore
        except Exception as exc:
            raise GeometryPipelineUnavailable(
                "NanoSAM predictor import failed. NanoSAM requires its Python package plus torch2trt/TensorRT support. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        encoder_path = str(image_encoder or NANOSAM_ENCODER_PATH or "").strip()
        decoder_path = str(mask_decoder or NANOSAM_DECODER_PATH or "").strip()
        if not encoder_path or not decoder_path:
            raise GeometryPipelineUnavailable(
                "NanoSAM requires NANOSAM_ENCODER_PATH and NANOSAM_DECODER_PATH engine paths."
            )
        if not Path(encoder_path).exists():
            raise GeometryPipelineUnavailable(f"NanoSAM image encoder not found: {encoder_path}")
        if not Path(decoder_path).exists():
            raise GeometryPipelineUnavailable(f"NanoSAM mask decoder not found: {decoder_path}")
        _validate_tensorrt_engine(encoder_path, "image encoder")
        _validate_tensorrt_engine(decoder_path, "mask decoder")

        try:
            try:
                predictor_signature = inspect.signature(Predictor)
            except Exception:
                predictor_signature = None
            predictor_kwargs: Dict[str, Any]
            if predictor_signature is not None and "image_encoder_engine" in predictor_signature.parameters:
                predictor_kwargs = {
                    "image_encoder_engine": encoder_path,
                    "mask_decoder_engine": decoder_path,
                }
            else:
                predictor_kwargs = {
                    "image_encoder": encoder_path,
                    "mask_decoder": decoder_path,
                }
            self.predictor = Predictor(**predictor_kwargs)
        except Exception as exc:
            raise GeometryPipelineUnavailable(f"NanoSAM Predictor init failed: {exc}") from exc

    def segment(self, image_rgb: np.ndarray, bbox_xyxy: Sequence[float]) -> np.ndarray:
        try:
            from PIL import Image
        except Exception as exc:
            raise GeometryPipelineUnavailable("Pillow is required for NanoSAM segmentation.") from exc

        try:
            import torch
        except Exception:
            torch = None  # type: ignore

        self.predictor.set_image(Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)))
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
        point_coords = np.asarray([[x1, y1], [x2, y2]], dtype=np.float32)
        point_labels = np.asarray([2, 3], dtype=np.int32)
        prediction = self.predictor.predict(point_coords, point_labels)
        if isinstance(prediction, tuple):
            mask = prediction[0]
        else:
            mask = prediction
        if torch is not None and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 4:
            mask_arr = mask_arr[0, 0]
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
        return np.asarray(mask_arr > 0, dtype=bool)


class DepthProAdapter:
    def __init__(self, model_path: Optional[str] = None):
        try:
            import depth_pro  # type: ignore
            self.depth_module = depth_pro
        except Exception:
            try:
                import depthpro  # type: ignore
                self.depth_module = depthpro
            except Exception as exc:
                raise GeometryPipelineUnavailable(
                    "Depth Pro is not installed. Install it before enabling OBJECT_GEOMETRY_PIPELINE_ENABLE."
                ) from exc

        create_model = getattr(self.depth_module, "create_model_and_transforms", None)
        if create_model is None:
            raise GeometryPipelineUnavailable("Depth Pro create_model_and_transforms() is unavailable.")

        self.model_path = str(model_path or DEPTH_PRO_MODEL_PATH or "").strip()
        if self.model_path and not Path(self.model_path).exists():
            raise GeometryPipelineUnavailable(f"Depth Pro model/checkpoint not found: {self.model_path}")

        device = None
        precision = None
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                precision = torch.float16
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
                precision = torch.float16
            else:
                device = torch.device("cpu")
                precision = torch.float32
        except Exception:
            torch = None  # type: ignore

        create_kwargs: Dict[str, Any] = {}
        try:
            signature = inspect.signature(create_model)
        except Exception:
            signature = None

        if signature is not None and "config" in signature.parameters:
            config_cls = None
            try:
                depth_impl = __import__("depth_pro.depth_pro", fromlist=["DepthProConfig", "DEFAULT_MONODEPTH_CONFIG_DICT"])
                config_cls = getattr(depth_impl, "DepthProConfig", None)
                default_config = getattr(depth_impl, "DEFAULT_MONODEPTH_CONFIG_DICT", None)
            except Exception:
                config_cls = None
                default_config = None
            if default_config is not None:
                checkpoint_uri = self.model_path or getattr(default_config, "checkpoint_uri", None)
                create_kwargs["config"] = config_cls(
                    patch_encoder_preset=default_config.patch_encoder_preset,
                    image_encoder_preset=default_config.image_encoder_preset,
                    decoder_features=default_config.decoder_features,
                    checkpoint_uri=checkpoint_uri,
                    fov_encoder_preset=default_config.fov_encoder_preset,
                    use_fov_head=default_config.use_fov_head,
                )
        elif self.model_path:
            for field_name in (
                "checkpoint_uri",
                "checkpoint_path",
                "checkpoint",
                "weights_path",
                "model_path",
            ):
                if signature is not None and field_name in signature.parameters:
                    create_kwargs[field_name] = self.model_path
                    break

        if signature is not None and "device" in signature.parameters and device is not None:
            create_kwargs["device"] = device
        if signature is not None and "precision" in signature.parameters and precision is not None:
            create_kwargs["precision"] = precision

        try:
            self.model, self.transform = create_model(**create_kwargs)
            self.model.eval()
        except Exception as exc:
            raise GeometryPipelineUnavailable(f"Depth Pro model init failed: {exc}") from exc

    def predict_depth(self, image_path: str, image_rgb: np.ndarray) -> np.ndarray:
        load_rgb = getattr(self.depth_module, "load_rgb", None)
        if load_rgb is None:
            raise GeometryPipelineUnavailable("Depth Pro load_rgb() is unavailable.")
        try:
            import torch
        except Exception:
            torch = None  # type: ignore

        image, _, f_px = load_rgb(image_path)
        image = self.transform(image)
        if torch is not None:
            with torch.no_grad():
                prediction = self.model.infer(image, f_px=f_px)
        else:
            prediction = self.model.infer(image, f_px=f_px)
        depth = prediction["depth"]
        if hasattr(depth, "detach"):
            depth = depth.detach()
        if hasattr(depth, "cpu"):
            depth = depth.cpu()
        return np.asarray(depth, dtype=np.float32)


@dataclass
class GeometryPipelineArtifacts:
    detections_path: Optional[str] = None
    detection_overlay_path: Optional[str] = None
    depth_map_path: Optional[str] = None
    depth_preview_path: Optional[str] = None


@dataclass
class GeometryPipelineResult:
    ok: bool
    failure_reason: Optional[str]
    selector_payload: Dict[str, Any]
    selector_raw_json: str
    selector_raw_api_response: Optional[Dict[str, Any]]
    selector_source: str
    object_rows: List[Dict[str, Any]]
    artifacts: GeometryPipelineArtifacts
    timings: Dict[str, Any]


def _draw_detection_overlay(image_rgb: np.ndarray, detections: Sequence[Mapping[str, Any]]) -> np.ndarray:
    canvas = cv2.cvtColor(np.asarray(image_rgb).copy(), cv2.COLOR_RGB2BGR)
    for det in detections:
        bbox = det.get("bbox_xyxy")
        if bbox is None:
            bbox = det.get("bbox")
        if bbox is None:
            continue
        bbox_values = np.asarray(bbox).reshape(-1)
        if bbox_values.size < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_values[:4]]
        label = str(det.get("label") or "unknown")
        score = det.get("confidence")
        text = label if score is None else f"{label} {float(score):.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 200, 40), 2)
        cv2.putText(canvas, text[:48], (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, text[:48], (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def _save_image_bgr(path: Path, image_bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save image to {path}")
    return str(path)


def _save_image_rgb(path: Path, image_rgb: np.ndarray) -> str:
    return _save_image_bgr(path, cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR))


def _save_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: Sequence[float], output_path: Path) -> str:
    canvas = cv2.cvtColor(np.asarray(image_rgb).copy(), cv2.COLOR_RGB2BGR)
    color = np.array([30, 200, 255], dtype=np.uint8)
    mask_bool = np.asarray(mask).astype(bool)
    canvas[mask_bool] = (
        0.35 * canvas[mask_bool].astype(np.float32) + 0.65 * color.astype(np.float32)
    ).astype(np.uint8)
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy[:4]]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 2)
    return _save_image_bgr(output_path, canvas)


def _selector_attribute_payload(selector_payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "view_type": str(selector_payload.get("view_type") or "unknown"),
        "room_function": str(selector_payload.get("room_function") or "unknown"),
        "style_hint": str(selector_payload.get("style_hint") or "unknown"),
        "clutter_level": str(selector_payload.get("clutter_level") or "unknown"),
        "floor_pattern": str(selector_payload.get("floor_pattern") or "unknown"),
        "lighting_ceiling": str(selector_payload.get("lighting_ceiling") or "unknown"),
        "wall_color": str(selector_payload.get("wall_color") or "unknown"),
        "scene_attributes": [str(v).strip() for v in list(selector_payload.get("scene_attributes") or []) if str(v).strip()],
        "additional_notes": str(selector_payload.get("additional_notes") or ""),
        "image_summary": str(selector_payload.get("image_summary") or ""),
    }


def _selector_source_object_types(selector_result: Mapping[str, Any]) -> List[str]:
    payload = selector_result.get("payload") if isinstance(selector_result, Mapping) else None
    if not isinstance(payload, Mapping):
        return []
    return normalize_selector_subset(payload.get("selected_object_types") or [])


def _detector_class_signature(class_names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(normalize_selector_subset(list(class_names or [])))


class ObjectGeometryPipeline:
    def __init__(
        self,
        *,
        captioner: Any,
        output_root: str,
        detector: Optional[Any] = None,
        segmenter: Optional[Any] = None,
        depth_estimator: Optional[Any] = None,
        horizontal_fov_deg: float = float(FOV),
        image_width_px: int = int(IMAGE_WIDTH),
        image_height_px: int = int(IMAGE_HEIGHT),
        save_artifacts: bool = True,
    ):
        self.captioner = captioner
        self.output_root = Path(output_root)
        self.detector = detector
        self._detector_is_external = detector is not None
        self._detector_class_signature = _detector_class_signature(
            list(getattr(detector, "class_names", []) or [])
        ) if detector is not None else None
        self.segmenter = segmenter
        self.depth_estimator = depth_estimator
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.image_width_px = int(image_width_px)
        self.image_height_px = int(image_height_px)
        self.save_artifacts = bool(save_artifacts)

    def _ensure_detector(self, selected_object_types: Sequence[str]) -> Any:
        selected_signature = _detector_class_signature(selected_object_types)
        if self.detector is None:
            self.detector = Detector(detector_type="YOLO_WORLD", class_names=list(selected_signature))
            self._detector_class_signature = selected_signature
            self._detector_is_external = False
            return self.detector
        if self._detector_is_external:
            if self._detector_class_signature != selected_signature and hasattr(self.detector, "set_class_names"):
                self.detector.set_class_names(list(selected_signature))
                self._detector_class_signature = selected_signature
            return self.detector
        if self._detector_class_signature != selected_signature:
            self.detector = Detector(detector_type="YOLO_WORLD", class_names=list(selected_signature))
            self._detector_class_signature = selected_signature
        return self.detector

    def _ensure_segmenter(self) -> Any:
        if self.segmenter is None:
            self.segmenter = NanoSAMMaskRefiner()
        return self.segmenter

    def _ensure_depth_estimator(self) -> Any:
        if self.depth_estimator is None:
            self.depth_estimator = DepthProAdapter()
        return self.depth_estimator

    def run_for_view(
        self,
        *,
        entry_id: int,
        image_path: str,
        image_rgb: np.ndarray,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        camera_orientation_deg: float,
        max_objects: int,
    ) -> GeometryPipelineResult:
        total_t0 = time.perf_counter()
        timings: Dict[str, Any] = {
            "selector_sec": 0.0,
            "dependency_setup_sec": 0.0,
            "detector_sec": 0.0,
            "depth_sec": 0.0,
            "mask_total_sec": 0.0,
            "angle_geometry_total_sec": 0.0,
            "crop_vlm_description_total_sec": 0.0,
            "mask_per_object_sec": [],
            "angle_geometry_per_object_sec": [],
            "crop_vlm_description_per_object_sec": [],
            "selected_object_type_count": 0,
            "detection_count_raw": 0,
            "detection_count_kept": 0,
            "object_count": 0,
        }

        def _finalize_timings() -> Dict[str, Any]:
            finalized = dict(timings)
            finalized["total_sec"] = float(time.perf_counter() - total_t0)
            per_crop = list(finalized.get("crop_vlm_description_per_object_sec") or [])
            finalized["crop_vlm_description_avg_sec"] = (
                float(sum(per_crop) / len(per_crop)) if per_crop else 0.0
            )
            return finalized

        def _fail(reason: str, *, selector_payload_out: Optional[Dict[str, Any]] = None, selector_result_out: Optional[Dict[str, Any]] = None, artifacts_out: Optional[GeometryPipelineArtifacts] = None) -> GeometryPipelineResult:
            result_payload = dict(selector_payload_out or {})
            result_selector = dict(selector_result_out or {})
            return GeometryPipelineResult(
                ok=False,
                failure_reason=reason,
                selector_payload=result_payload,
                selector_raw_json=str(result_selector.get("raw_json") or ""),
                selector_raw_api_response=result_selector.get("raw_api_response"),
                selector_source=str(result_selector.get("source") or ""),
                object_rows=[],
                artifacts=artifacts_out or GeometryPipelineArtifacts(),
                timings=_finalize_timings(),
            )

        camera_context = {
            "camera_x": float(camera_x),
            "camera_z": float(camera_z),
            "camera_orientation_deg": float(camera_orientation_deg),
        }
        selector_t0 = time.perf_counter()
        selector_result = self.captioner.select_object_types_with_meta(
            image_path=image_path,
            camera_context=camera_context,
        )
        timings["selector_sec"] = float(time.perf_counter() - selector_t0)
        selector_payload = dict(selector_result.get("payload") or {})
        selected_object_types = _selector_source_object_types(selector_result)
        timings["selected_object_type_count"] = int(len(selected_object_types))
        if not selected_object_types:
            return _fail(
                "empty_selected_object_types",
                selector_payload_out=selector_payload,
                selector_result_out=selector_result,
            )

        try:
            dependency_t0 = time.perf_counter()
            detector = self._ensure_detector(selected_object_types)
            segmenter = self._ensure_segmenter()
            depth_estimator = self._ensure_depth_estimator()
            timings["dependency_setup_sec"] = float(time.perf_counter() - dependency_t0)
        except Exception as exc:
            timings["dependency_setup_sec"] = float(time.perf_counter() - dependency_t0)
            return _fail(
                f"dependency_unavailable:{type(exc).__name__}:{exc}",
                selector_payload_out=selector_payload,
                selector_result_out=selector_result,
            )

        view_dir = self.output_root / "geometry" / f"view_{int(entry_id):05d}"
        objects_dir = view_dir / "objects"
        artifacts = GeometryPipelineArtifacts()

        try:
            detector_t0 = time.perf_counter()
            raw_detections = list(detector.detect(image_rgb))
            timings["detector_sec"] = float(time.perf_counter() - detector_t0)
        except Exception as exc:
            timings["detector_sec"] = float(time.perf_counter() - detector_t0)
            return _fail(
                f"detector_failed:{type(exc).__name__}",
                selector_payload_out=selector_payload,
                selector_result_out=selector_result,
                artifacts_out=artifacts,
            )
        timings["detection_count_raw"] = int(len(raw_detections))

        normalized_detections: List[Dict[str, Any]] = []
        for det_idx, det in enumerate(raw_detections):
            bbox = det.get("bbox_xyxy")
            if bbox is None:
                bbox = det.get("bbox")
            if bbox is None:
                continue
            bbox_values = np.asarray(bbox).reshape(-1)
            if bbox_values.size < 4:
                continue
            canonical_label = canonicalize_household_object_label(det.get("label"), default="")
            if not canonical_label or canonical_label not in selected_object_types:
                continue
            normalized_detections.append(
                {
                    "det_idx": int(det_idx),
                    "label": canonical_label,
                    "bbox_xyxy": [float(v) for v in bbox_values[:4]],
                    "confidence": _safe_float(det.get("confidence")),
                }
            )
        normalized_detections.sort(key=lambda item: (-(item.get("confidence") or 0.0), item["label"], item["det_idx"]))
        normalized_detections = normalized_detections[: max(1, int(max_objects))]
        timings["detection_count_kept"] = int(len(normalized_detections))
        if not normalized_detections:
            return _fail(
                "no_valid_detections",
                selector_payload_out=selector_payload,
                selector_result_out=selector_result,
                artifacts_out=artifacts,
            )

        if self.save_artifacts:
            detections_path = view_dir / "detections.json"
            detections_path.parent.mkdir(parents=True, exist_ok=True)
            detections_path.write_text(json.dumps(normalized_detections, ensure_ascii=True, indent=2), encoding="utf-8")
            artifacts.detections_path = str(detections_path)
            artifacts.detection_overlay_path = _save_image_bgr(
                view_dir / "detection_overlay.jpg",
                _draw_detection_overlay(image_rgb, normalized_detections),
            )

        try:
            depth_t0 = time.perf_counter()
            depth_map_m = np.asarray(depth_estimator.predict_depth(image_path, image_rgb), dtype=np.float32)
            timings["depth_sec"] = float(time.perf_counter() - depth_t0)
        except Exception as exc:
            timings["depth_sec"] = float(time.perf_counter() - depth_t0)
            return _fail(
                f"depth_failed:{type(exc).__name__}",
                selector_payload_out=selector_payload,
                selector_result_out=selector_result,
                artifacts_out=artifacts,
            )
        if depth_map_m.ndim != 2:
            return _fail(
                "depth_invalid_shape",
                selector_payload_out=selector_payload,
                selector_result_out=selector_result,
                artifacts_out=artifacts,
            )

        if self.save_artifacts:
            view_dir.mkdir(parents=True, exist_ok=True)
            depth_path = view_dir / "depth_map.npy"
            np.save(depth_path, depth_map_m.astype(np.float32))
            artifacts.depth_map_path = str(depth_path)
            artifacts.depth_preview_path = _save_image_bgr(view_dir / "depth_preview.jpg", depth_preview_u8(depth_map_m))

        object_rows: List[Dict[str, Any]] = []
        for local_index, det in enumerate(normalized_detections):
            bbox_xyxy = list(det["bbox_xyxy"])
            try:
                mask_t0 = time.perf_counter()
                mask = np.asarray(segmenter.segment(image_rgb, bbox_xyxy)).astype(bool)
                mask_sec = float(time.perf_counter() - mask_t0)
                timings["mask_total_sec"] = float(timings["mask_total_sec"] + mask_sec)
                timings["mask_per_object_sec"].append(mask_sec)
            except Exception as exc:
                return _fail(
                    f"mask_failed:{type(exc).__name__}",
                    selector_payload_out=selector_payload,
                    selector_result_out=selector_result,
                    artifacts_out=artifacts,
                )
            if mask.shape[:2] != image_rgb.shape[:2]:
                return _fail(
                    "mask_invalid_shape",
                    selector_payload_out=selector_payload,
                    selector_result_out=selector_result,
                    artifacts_out=artifacts,
                )
            angle_t0 = time.perf_counter()
            centroid = mask_centroid(mask)
            if centroid is None:
                return _fail(
                    "mask_empty",
                    selector_payload_out=selector_payload,
                    selector_result_out=selector_result,
                    artifacts_out=artifacts,
                )
            centroid_x_px, centroid_y_px = centroid
            stats = mask_depth_stats(depth_map_m, mask)
            forward_depth_m = stats["trimmed_median_m"]
            if forward_depth_m is None:
                return _fail(
                    "depth_no_valid_pixels",
                    selector_payload_out=selector_payload,
                    selector_result_out=selector_result,
                    artifacts_out=artifacts,
                )
            relative_bearing_deg, vertical_angle_deg = pixel_center_to_relative_angles_deg(
                centroid_x_px,
                centroid_y_px,
                width_px=image_rgb.shape[1],
                height_px=image_rgb.shape[0],
                horizontal_fov_deg=self.horizontal_fov_deg,
            )
            planar_distance_m = planar_distance_from_forward_depth_m(forward_depth_m, relative_bearing_deg)
            relative_height_from_camera_m = relative_height_from_forward_depth_m(forward_depth_m, vertical_angle_deg)
            if planar_distance_m is None or relative_height_from_camera_m is None:
                return _fail(
                    "geometry_projection_failed",
                    selector_payload_out=selector_payload,
                    selector_result_out=selector_result,
                    artifacts_out=artifacts,
                )
            estimated_global_x, estimated_global_y, estimated_global_z = project_global_xyz_from_geometry(
                camera_x=float(camera_x),
                camera_y=float(camera_y),
                camera_z=float(camera_z),
                camera_orientation_deg=float(camera_orientation_deg),
                distance_m=planar_distance_m,
                relative_bearing_deg=relative_bearing_deg,
                relative_height_from_camera_m=relative_height_from_camera_m,
            )
            bins = relative_bins_from_geometry(
                centroid_x_px=centroid_x_px,
                centroid_y_px=centroid_y_px,
                width_px=image_rgb.shape[1],
                height_px=image_rgb.shape[0],
                distance_m=forward_depth_m,
            )
            angle_sec = float(time.perf_counter() - angle_t0)
            timings["angle_geometry_total_sec"] = float(timings["angle_geometry_total_sec"] + angle_sec)
            timings["angle_geometry_per_object_sec"].append(angle_sec)
            crop_rgb = crop_image_from_bbox(image_rgb, bbox_xyxy)
            crop_rel = None
            mask_rel = None
            overlay_rel = None
            if self.save_artifacts:
                crop_rel = objects_dir / f"obj_{local_index:03d}_crop.jpg"
                mask_rel = objects_dir / f"obj_{local_index:03d}_mask.png"
                overlay_rel = objects_dir / f"obj_{local_index:03d}_mask_overlay.jpg"
                _save_image_rgb(crop_rel, crop_rgb)
                _save_image_bgr(mask_rel, np.asarray(mask.astype(np.uint8) * 255, dtype=np.uint8))
                _save_mask_overlay(image_rgb, mask, bbox_xyxy, overlay_rel)

            crop_t0 = time.perf_counter()
            crop_description = self.captioner.describe_object_crop_with_meta(
                str(crop_rel or image_path),
                yolo_label=det["label"],
                yolo_confidence=det.get("confidence"),
            )
            crop_sec = float(time.perf_counter() - crop_t0)
            timings["crop_vlm_description_total_sec"] = float(timings["crop_vlm_description_total_sec"] + crop_sec)
            timings["crop_vlm_description_per_object_sec"].append(crop_sec)
            detector_label = str(det["label"])
            crop_label = canonicalize_household_object_label(crop_description.get("label"), default=detector_label)
            final_label = detector_label if crop_label != detector_label else crop_label
            row = {
                "object_local_id": f"det_{local_index:03d}",
                "label": final_label,
                "detector_label": detector_label,
                "crop_vlm_label": crop_description.get("label"),
                "object_confidence": float(det.get("confidence") or 0.0),
                "bbox_xyxy": [float(v) for v in bbox_xyxy],
                "bbox_xywh_norm": bbox_xywh_norm_from_xyxy(
                    bbox_xyxy,
                    width_px=image_rgb.shape[1],
                    height_px=image_rgb.shape[0],
                ),
                "mask_area_px": int(np.count_nonzero(mask)),
                "mask_area_ratio": float(np.count_nonzero(mask) / max(mask.size, 1)),
                "mask_centroid_x_px": float(centroid_x_px),
                "mask_centroid_y_px": float(centroid_y_px),
                "mask_centroid_x_norm": float(centroid_x_px / max(float(image_rgb.shape[1]), 1.0)),
                "mask_centroid_y_norm": float(centroid_y_px / max(float(image_rgb.shape[0]), 1.0)),
                "distance_from_camera_m": float(forward_depth_m),
                "projected_planar_distance_m": float(planar_distance_m),
                "relative_bearing_deg": float(relative_bearing_deg),
                "vertical_angle_deg": float(vertical_angle_deg),
                "relative_height_from_camera_m": float(relative_height_from_camera_m),
                "estimated_global_x": estimated_global_x,
                "estimated_global_y": estimated_global_y,
                "estimated_global_z": estimated_global_z,
                "laterality": bins["laterality"],
                "distance_bin": bins["distance_bin"],
                "verticality": bins["verticality"],
                "description": str(crop_description.get("short_description") or detector_label).strip() or detector_label,
                "long_form_open_description": (
                    str(crop_description.get("long_description") or crop_description.get("short_description") or detector_label).strip()
                    or detector_label
                ),
                "attributes": [str(v).strip() for v in list(crop_description.get("attributes") or []) if str(v).strip()],
                "support_relation": "unknown",
                "any_text": "",
                "location_relative_to_other_objects": "",
                "surrounding_context": [],
                "scene_attributes": list(_selector_attribute_payload(selector_payload)["scene_attributes"]),
                "object_text_short": str(crop_description.get("short_description") or detector_label).strip() or detector_label,
                "object_text_long": (
                    str(crop_description.get("long_description") or crop_description.get("short_description") or detector_label).strip()
                    or detector_label
                ),
                "text_input_for_clip_short": str(crop_description.get("short_description") or detector_label).strip() or detector_label,
                "text_input_for_clip_long": (
                    str(crop_description.get("long_description") or crop_description.get("short_description") or detector_label).strip()
                    or detector_label
                ),
                "geometry_source": "mask_depth",
                "geometry_fallback_reason": None,
                "detector_confidence": float(det.get("confidence") or 0.0),
                "depth_stat_median_m": stats["median_m"],
                "depth_stat_p10_m": stats["p10_m"],
                "depth_stat_p90_m": stats["p90_m"],
                "vlm_distance_from_camera_m": crop_description.get("distance_from_camera_m"),
                "vlm_relative_bearing_deg": None,
                "crop_path": None if crop_rel is None else str(crop_rel),
                "mask_path": None if mask_rel is None else str(mask_rel),
                "mask_overlay_path": None if overlay_rel is None else str(overlay_rel),
                "depth_map_path": artifacts.depth_map_path,
                "timing_mask_sec": mask_sec,
                "timing_angle_geometry_sec": angle_sec,
                "timing_crop_vlm_description_sec": crop_sec,
            }
            object_rows.append(row)
        timings["object_count"] = int(len(object_rows))

        return GeometryPipelineResult(
            ok=True,
            failure_reason=None,
            selector_payload=selector_payload,
            selector_raw_json=str(selector_result.get("raw_json") or ""),
            selector_raw_api_response=selector_result.get("raw_api_response"),
            selector_source=str(selector_result.get("source") or ""),
            object_rows=object_rows,
            artifacts=artifacts,
            timings=_finalize_timings(),
        )
