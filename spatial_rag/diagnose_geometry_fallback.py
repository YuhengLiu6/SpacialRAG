from __future__ import annotations

import argparse
import importlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from spatial_rag.config import (
    FOV,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NANOSAM_DECODER_PATH,
    NANOSAM_ENCODER_PATH,
    OBJECT_CACHE_DIR,
    SPATIAL_DB_VLM_MODEL,
)
from spatial_rag.object_geometry_pipeline import ObjectGeometryPipeline
from spatial_rag.vlm_captioner import VLMCaptioner


def _safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_object_types(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _status_ok(*, details: Optional[Mapping[str, Any]] = None, duration_sec: Optional[float] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": True,
        "details": dict(details or {}),
    }
    if duration_sec is not None:
        payload["duration_sec"] = float(duration_sec)
    return payload


def _status_error(exc: BaseException, *, duration_sec: Optional[float] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    if duration_sec is not None:
        payload["duration_sec"] = float(duration_sec)
    return payload


def _build_selector_override(selected_object_types: Sequence[str]) -> Dict[str, Any]:
    payload = {
        "view_type": "diagnostic",
        "room_function": "",
        "style_hint": "",
        "clutter_level": "",
        "scene_attributes": [],
        "floor_pattern": "",
        "lighting_ceiling": "",
        "wall_color": "",
        "additional_notes": "",
        "image_summary": "Diagnostic selector override.",
        "selected_object_types": list(selected_object_types),
    }
    return {
        "payload": payload,
        "raw_json": json.dumps(payload, ensure_ascii=True),
        "raw_api_response": None,
        "source": "cli-override",
    }


def _detect_likely_failure_stage(result: Mapping[str, Any]) -> Optional[str]:
    selector_info = result.get("selector")
    if isinstance(selector_info, Mapping) and not selector_info.get("ok", True):
        return "selector"

    selector_types = []
    if isinstance(selector_info, Mapping):
        selector_types = list(selector_info.get("selected_object_types") or [])
    if not selector_types:
        return "selector_empty_selected_object_types"

    dependencies = result.get("dependencies")
    if isinstance(dependencies, Mapping):
        for stage_name in ("detector", "segmenter", "depth_estimator"):
            stage_info = dependencies.get(stage_name)
            if isinstance(stage_info, Mapping) and not stage_info.get("ok", True):
                return f"dependency_setup.{stage_name}"

    detector_dry_run = result.get("detector_dry_run")
    if isinstance(detector_dry_run, Mapping) and not detector_dry_run.get("ok", True):
        return "detector_run"

    full_result = result.get("full_geometry_run")
    if isinstance(full_result, Mapping) and not full_result.get("ok", True):
        reason = str(full_result.get("failure_reason") or "").strip()
        return f"geometry_pipeline:{reason}" if reason else "geometry_pipeline"
    return None


def summarize_db_fallbacks(db_root: Path) -> Dict[str, Any]:
    build_report_path = db_root / "build_report.json"
    per_image_path = db_root / "per_image_timings.jsonl"
    raw_api_path = db_root / "raw_api_responses.jsonl"

    build_report = json.loads(build_report_path.read_text(encoding="utf-8")) if build_report_path.exists() else {}
    per_image_rows = _load_jsonl(per_image_path) if per_image_path.exists() else []
    raw_api_rows = _load_jsonl(raw_api_path) if raw_api_path.exists() else []

    route_counts = Counter()
    parse_status_counts = Counter()
    for row in per_image_rows:
        route_counts[str(row.get("route"))] += 1
        parse_status_counts[str(row.get("parse_status"))] += 1

    fallback_reason_counts = Counter()
    dependency_selected_lengths = Counter()
    non_dependency_examples: List[Dict[str, Any]] = []
    for row in raw_api_rows:
        reason = str(row.get("geometry_fallback_reason") or "")
        if reason:
            fallback_reason_counts[reason] += 1
        if reason.startswith("dependency_unavailable:"):
            dependency_selected_lengths[len(list(row.get("selected_object_types") or []))] += 1
        elif reason:
            non_dependency_examples.append(
                {
                    "entry_id": row.get("entry_id"),
                    "frame_id": row.get("frame_id"),
                    "file_name": row.get("file_name"),
                    "geometry_fallback_reason": reason,
                    "selected_object_types": list(row.get("selected_object_types") or []),
                }
            )

    summary = {
        "mode": "db",
        "db_root": str(db_root),
        "build_report_excerpt": {
            "execution_mode": build_report.get("execution_mode"),
            "legacy_per_frame": build_report.get("legacy_per_frame"),
            "geometry_ok_count": build_report.get("geometry_ok_count"),
            "geometry_fallback_count": build_report.get("geometry_fallback_count"),
            "parse_ok_count": build_report.get("parse_ok_count"),
            "parse_fallback_count": build_report.get("parse_fallback_count"),
            "geometry_objects_before_r_threshold": build_report.get("geometry_objects_before_r_threshold"),
            "geometry_objects_after_r_threshold": build_report.get("geometry_objects_after_r_threshold"),
            "geometry_objects_filtered_by_r_threshold": build_report.get("geometry_objects_filtered_by_r_threshold"),
            "frames_all_geometry_objects_filtered": build_report.get("frames_all_geometry_objects_filtered"),
        },
        "per_image_route_counts": dict(route_counts),
        "per_image_parse_status_counts": dict(parse_status_counts),
        "geometry_fallback_reason_counts": dict(fallback_reason_counts),
        "dependency_unavailable_selected_object_type_count_distribution": dict(sorted(dependency_selected_lengths.items())),
        "non_dependency_fallback_examples": non_dependency_examples[:20],
    }
    return summary


def diagnose_image(
    *,
    image_path: Path,
    entry_id: int,
    camera_x: float,
    camera_y: float,
    camera_z: float,
    camera_orientation_deg: float,
    max_objects: int,
    selected_object_types_override: Sequence[str],
    output_root: Path,
    use_cache: bool,
    object_cache_dir: Path,
    save_artifacts: bool,
) -> Dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    captioner = VLMCaptioner(
        model_name=SPATIAL_DB_VLM_MODEL,
        use_cache=use_cache,
        cache_dir=str(output_root / "vlm_cache"),
        object_use_cache=use_cache,
        object_cache_dir=str(object_cache_dir),
    )
    pipeline = ObjectGeometryPipeline(
        captioner=captioner,
        output_root=str(output_root),
        horizontal_fov_deg=float(FOV),
        image_width_px=int(IMAGE_WIDTH),
        image_height_px=int(IMAGE_HEIGHT),
        save_artifacts=bool(save_artifacts),
    )

    result: Dict[str, Any] = {
        "mode": "image",
        "image_path": str(image_path),
        "entry_id": int(entry_id),
        "camera": {
            "x": float(camera_x),
            "y": float(camera_y),
            "z": float(camera_z),
            "orientation_deg": float(camera_orientation_deg),
        },
        "config": {
            "horizontal_fov_deg": float(FOV),
            "image_width_px": int(IMAGE_WIDTH),
            "image_height_px": int(IMAGE_HEIGHT),
            "nanosam_encoder_path": str(NANOSAM_ENCODER_PATH),
            "nanosam_decoder_path": str(NANOSAM_DECODER_PATH),
            "object_cache_dir": str(object_cache_dir),
        },
        "module_imports": {},
        "selector": {},
        "dependencies": {},
        "detector_dry_run": {},
        "full_geometry_run": {},
    }

    for module_name in ("nanosam", "torch2trt", "tensorrt", "depth_pro", "depthpro"):
        t0 = time.perf_counter()
        try:
            importlib.import_module(module_name)
            result["module_imports"][module_name] = _status_ok(duration_sec=time.perf_counter() - t0)
        except Exception as exc:
            result["module_imports"][module_name] = _status_error(exc, duration_sec=time.perf_counter() - t0)

    selector_result_override: Optional[Dict[str, Any]] = None
    if selected_object_types_override:
        selector_result_override = _build_selector_override(selected_object_types_override)
        result["selector"] = {
            "ok": True,
            "source": "cli-override",
            "selected_object_types": list(selected_object_types_override),
        }
    else:
        t0 = time.perf_counter()
        try:
            selector_result_override = dict(
                captioner.select_object_types_with_meta(
                    image_path=str(image_path),
                    camera_context={
                        "camera_x": float(camera_x),
                        "camera_y": float(camera_y),
                        "camera_orientation_deg": float(camera_orientation_deg),
                    },
                )
            )
            payload = dict(selector_result_override.get("payload") or {})
            result["selector"] = {
                "ok": True,
                "source": selector_result_override.get("source"),
                "selected_object_types": list(payload.get("selected_object_types") or []),
                "payload": payload,
                "duration_sec": float(time.perf_counter() - t0),
            }
        except Exception as exc:
            result["selector"] = _status_error(exc, duration_sec=time.perf_counter() - t0)
            selector_result_override = None

    selected_object_types = list(
        ((selector_result_override or {}).get("payload") or {}).get("selected_object_types") or []
    )

    detector_instance = None
    if selected_object_types:
        for stage_name, stage_callable in (
            ("detector", lambda: pipeline._ensure_detector(selected_object_types)),
            ("segmenter", pipeline._ensure_segmenter),
            ("depth_estimator", pipeline._ensure_depth_estimator),
        ):
            t0 = time.perf_counter()
            try:
                instance = stage_callable()
                details: Dict[str, Any] = {}
                if stage_name == "detector":
                    detector_instance = instance
                    details["class_names"] = list(getattr(instance, "class_names", []) or [])
                    details["model_path"] = str(getattr(instance, "model_path", "") or "")
                if stage_name == "depth_estimator":
                    details["model_path"] = str(getattr(instance, "model_path", "") or "")
                result["dependencies"][stage_name] = _status_ok(details=details, duration_sec=time.perf_counter() - t0)
            except Exception as exc:
                result["dependencies"][stage_name] = _status_error(exc, duration_sec=time.perf_counter() - t0)

        if detector_instance is not None:
            t0 = time.perf_counter()
            try:
                detections = list(detector_instance.detect(image_rgb))
                preview = [
                    {
                        "label": det.get("label"),
                        "confidence": det.get("confidence"),
                        "bbox": list(np.asarray(det.get("bbox") or det.get("bbox_xyxy") or []).reshape(-1)[:4]),
                    }
                    for det in detections[:10]
                ]
                result["detector_dry_run"] = _status_ok(
                    details={
                        "raw_detection_count": int(len(detections)),
                        "preview": preview,
                    },
                    duration_sec=time.perf_counter() - t0,
                )
            except Exception as exc:
                result["detector_dry_run"] = _status_error(exc, duration_sec=time.perf_counter() - t0)
    else:
        result["dependencies"] = {
            "skipped": True,
            "reason": "selector returned no selected_object_types; real pipeline would fail before dependency setup.",
        }
        result["detector_dry_run"] = {
            "skipped": True,
            "reason": "selector returned no selected_object_types; detector dry-run not attempted.",
        }

    t0 = time.perf_counter()
    try:
        geometry_result = pipeline.run_for_view(
            entry_id=int(entry_id),
            image_path=str(image_path),
            image_rgb=image_rgb,
            camera_x=float(camera_x),
            camera_y=float(camera_y),
            camera_z=float(camera_z),
            camera_orientation_deg=float(camera_orientation_deg),
            max_objects=int(max_objects),
            selector_result_override=selector_result_override,
            defer_object_descriptions=True,
        )
        result["full_geometry_run"] = {
            "ok": bool(geometry_result.ok),
            "failure_reason": geometry_result.failure_reason,
            "selector_source": geometry_result.selector_source,
            "selected_object_types": list(geometry_result.selector_payload.get("selected_object_types") or []),
            "object_row_count": int(len(list(geometry_result.object_rows or []))),
            "description_request_count": int(len(list(geometry_result.description_requests or []))),
            "timings": dict(geometry_result.timings or {}),
            "artifacts": {
                "detections_path": geometry_result.artifacts.detections_path,
                "detection_overlay_path": geometry_result.artifacts.detection_overlay_path,
                "depth_map_path": geometry_result.artifacts.depth_map_path,
                "depth_preview_path": geometry_result.artifacts.depth_preview_path,
            },
            "duration_sec": float(time.perf_counter() - t0),
        }
    except Exception as exc:
        result["full_geometry_run"] = _status_error(exc, duration_sec=time.perf_counter() - t0)

    result["diagnosis"] = {
        "likely_first_failure_stage": _detect_likely_failure_stage(result),
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose why geometry views fell back to VLM fallback.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    db_parser = subparsers.add_parser("db", help="Summarize recorded fallback reasons from an existing DB.")
    db_parser.add_argument("--db-root", required=True, help="Path to the built spatial DB root.")

    image_parser = subparsers.add_parser("image", help="Diagnose live geometry stages for a single image.")
    image_parser.add_argument("--image-path", required=True, help="Path to an image to test.")
    image_parser.add_argument("--entry-id", type=int, default=0, help="Synthetic entry id for this diagnostic run.")
    image_parser.add_argument("--camera-x", type=float, default=0.0)
    image_parser.add_argument("--camera-y", type=float, default=0.0)
    image_parser.add_argument("--camera-z", type=float, default=0.0)
    image_parser.add_argument("--camera-orientation-deg", type=float, default=0.0)
    image_parser.add_argument("--max-objects", type=int, default=24)
    image_parser.add_argument(
        "--selected-object-types",
        default="",
        help="Comma-separated selector override. Useful when you want to bypass the selector and test detector/dependencies directly.",
    )
    image_parser.add_argument(
        "--output-root",
        default="tmp_geometry_diagnostics",
        help="Directory used for any temporary diagnostic artifacts.",
    )
    image_parser.add_argument(
        "--object-cache-dir",
        default=OBJECT_CACHE_DIR,
        help="Object cache directory for selector/object cache resolution.",
    )
    image_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable VLM cache usage for this diagnostic run.",
    )
    image_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save geometry artifacts (detections/depth previews) into output-root.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "db":
        payload = summarize_db_fallbacks(Path(args.db_root).expanduser().resolve())
    else:
        payload = diagnose_image(
            image_path=Path(args.image_path).expanduser().resolve(),
            entry_id=int(args.entry_id),
            camera_x=float(args.camera_x),
            camera_y=float(args.camera_y),
            camera_z=float(args.camera_z),
            camera_orientation_deg=float(args.camera_orientation_deg),
            max_objects=int(args.max_objects),
            selected_object_types_override=_parse_object_types(args.selected_object_types),
            output_root=Path(args.output_root).expanduser().resolve(),
            use_cache=not bool(args.no_cache),
            object_cache_dir=Path(args.object_cache_dir).expanduser().resolve(),
            save_artifacts=bool(args.save_artifacts),
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_safe_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
