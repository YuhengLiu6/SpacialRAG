import argparse
import csv
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from spatial_rag.config import (
    BBOX_CONF_THRESHOLD,
    DEPTH_PRO_MODEL_PATH,
    DINOV3_BATCH_SIZE,
    DINOV3_MODEL_NAME,
    DINOV3_NORMALIZE,
    DISTANCE_PENALTY_DSQ0,
    ENABLE_DINOV3_EMBEDDING,
    ENABLE_DINOV3_SCORING,
    FOV,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NANOSAM_DECODER_PATH,
    NANOSAM_ENCODER_PATH,
    OBJECT_CACHE_DIR,
    OBJECT_GEOMETRY_PIPELINE_ENABLE,
    OBJECT_MAX_PER_FRAME,
    OBJECT_PARSE_RETRIES,
    OBJECT_PRELIST_TAXONOMY_PATH,
    OCCLUSION_SOURCE,
    OCCLUSION_REWEIGHT_B,
    OCCLUSION_REWEIGHT_EPS,
    OCCLUSION_REWEIGHT_W1,
    OCCLUSION_REWEIGHT_W2,
    OBJECT_SURROUNDING_MAX,
    OBJECT_USE_CACHE,
    OBJECT_VERTICAL_REL_EPS_M,
    SAVE_GEOMETRY_ARTIFACTS,
    SCAN_ANGLES,
    SCORE_WEIGHT_M1,
    SCORE_WEIGHT_M2,
    SCENE_PATH,
    SPATIAL_DB_DIR,
    SPATIAL_DB_VLM_MODEL,
    STORE_DINOV3_EMBEDDING,
    VISIBLE_OCC_BOUNDARY_NEIGHBOR_RADIUS,
    VISIBLE_OCC_BOUNDARY_WIDTH,
    VISIBLE_OCC_DEPTH_MARGIN_DELTA,
    VISIBLE_OCC_RING_RADIUS,
    OCCLUSION_TARGET_OVERLAP_THRESHOLD,
    VLM_ANGLE_SPLIT_ENABLE,
    VLM_ANGLE_STEP,
)
from spatial_rag.object_canonicalizer import UNKNOWN_TEXT_TOKEN, compose_frame_text, select_object_text, sorted_objects
from spatial_rag.household_taxonomy import canonicalize_household_object_label
from spatial_rag.object_geometry_pipeline import ObjectGeometryPipeline
from spatial_rag.occlusion_scoring import (
    OCCLUSION_SCORE_FORMULA_VERSION,
    compute_reweighted_detection_score,
    map_occlusion_level_to_penalty,
    normalize_occlusion_level,
)
from spatial_rag.object_parser import ParseResult, parse_scene_objects
from spatial_rag.vlm_captioner import VLMCaptioner


ObjectGroupItem = Tuple[Dict[str, Any], np.ndarray, np.ndarray, Optional[np.ndarray]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _builder_log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    tqdm.write(f"[SpatialDBBuilder][{ts}] {message}")


def _str_to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_scan_angles(value: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("scan angles cannot be empty")
    out: List[int] = []
    for p in parts:
        try:
            deg = int(float(p))
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"Invalid scan angle: {p}") from exc
        out.append(deg)
    return tuple(out)


def _rotation_to_orientation_deg(rotation) -> int:
    if hasattr(rotation, "w") and hasattr(rotation, "x") and hasattr(rotation, "y") and hasattr(rotation, "z"):
        w = float(rotation.w)
        x = float(rotation.x)
        y = float(rotation.y)
        z = float(rotation.z)
    elif isinstance(rotation, (list, tuple, np.ndarray)) and len(rotation) == 4:
        w = float(rotation[0])
        x = float(rotation[1])
        y = float(rotation[2])
        z = float(rotation[3])
    else:
        raise ValueError("Unsupported rotation format")

    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad = float(np.arctan2(siny_cosp, cosy_cosp))
    yaw_deg = (np.degrees(yaw_rad) + 360.0) % 360.0
    return int(round(yaw_deg)) % 360


def _normalize_scan_angles(scan_angles: Sequence[int]) -> Tuple[int, ...]:
    normalized = sorted({int(a) % 360 for a in scan_angles})
    if not normalized:
        raise ValueError("scan_angles cannot be empty")
    return tuple(normalized)


def _nearest_scan_angle(angle_deg: int, scan_angles: Sequence[int]) -> int:
    if not scan_angles:
        return int(angle_deg) % 360
    angle = int(angle_deg) % 360
    best = int(scan_angles[0]) % 360
    best_dist = 361
    for cand in scan_angles:
        c = int(cand) % 360
        dist = min((angle - c) % 360, (c - angle) % 360)
        if dist < best_dist:
            best_dist = dist
            best = c
    return best


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


_OBJECT_R_SCORES_PRE_THRESHOLD_COLUMNS: Tuple[str, ...] = (
    "entry_id",
    "frame_id",
    "file_name",
    "object_local_id",
    "object_route",
    "label",
    "bbox_xyxy",
    "bbox_xywh_norm",
    "object_confidence",
    "detector_confidence",
    "occlusion_source",
    "occlusion_level",
    "occlusion_penalty_p_o",
    "reweighted_detection_score_r",
    "r_threshold_used",
    "would_be_filtered_by_r_threshold",
)


_OBJECT_R_SCORES_COLUMNS: Tuple[str, ...] = (
    "object_global_id",
    "reweighted_detection_score_r",
)

_OBJECT_OCCLUSION_LEVELS_CSV_NAME = "object_occlusion_levels.csv"
_OBJECT_OCCLUSION_LEVELS_COLUMNS: Tuple[str, ...] = (
    "object_global_id",
    "occlusion_level",
)

_OBJECT_CROPS_BY_GLOBAL_ID_DIRNAME = "object_crops_by_global_id"
_OBJECT_CROPS_BY_GLOBAL_ID_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "object_global_id",
    "occlusion_level",
    "original_score",
    "original_score_token",
    "entry_id",
    "frame_id",
    "file_name",
    "label",
    "geometry_source",
    "crop_export_status",
    "crop_export_source",
    "source_path",
    "exported_path",
)


def _serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _serialize_csv_value(row.get(name)) for name in fieldnames})


def _resolve_existing_path(db_root: Path, raw_path: Optional[str]) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, db_root / path, db_root.parent / path])
        try:
            rel = path.relative_to(db_root.name)
            candidates.append(db_root / rel)
        except Exception:
            pass
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _bbox_xyxy_ints_from_row(row: Mapping[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    bbox = list(row.get("bbox_xyxy") or [])
    if len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _safe_filename_token(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        text = default
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or default


def _safe_score_token(value: Any, default: str = "unknown", precision: int = 3) -> str:
    try:
        score = float(value)
    except Exception:
        return default
    if not math.isfinite(score):
        return default
    text = f"{score:.{precision}f}".rstrip("0").rstrip(".")
    if not text:
        return default
    text = text.replace("-", "neg_").replace(".", "p")
    return _safe_filename_token(text, default=default)


def _object_crop_original_score(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("detector_confidence", "object_confidence"):
        value = row.get(key)
        try:
            score = float(value)
        except Exception:
            continue
        if math.isfinite(score):
            return score
    return None


def _object_crop_export_filename(
    object_global_id: int,
    label: Any,
    occlusion_level: Any,
    original_score: Any,
) -> str:
    safe_label = _safe_filename_token(label, default="unknown")
    normalized_occlusion_level = normalize_occlusion_level(occlusion_level, default="uncertain")
    safe_occlusion_level = _safe_filename_token(normalized_occlusion_level, default="uncertain")
    safe_original_score = _safe_score_token(original_score, default="unknown")
    return f"{int(object_global_id)}_{safe_label}_{safe_occlusion_level}_{safe_original_score}.jpg"


def export_object_crops_by_global_id(
    *,
    db_root: Path,
    object_rows: Sequence[Mapping[str, Any]],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    export_dir = Path(output_dir) if output_dir is not None else (db_root / _OBJECT_CROPS_BY_GLOBAL_ID_DIRNAME)
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    exported_count = 0
    copied_count = 0
    regenerated_count = 0
    skipped_count = 0

    for row in list(object_rows):
        try:
            object_global_id = int(row.get("object_global_id"))
        except Exception:
            skipped_count += 1
            continue

        normalized_occlusion_level = normalize_occlusion_level(row.get("occlusion_level"), default="uncertain")
        original_score = _object_crop_original_score(row)
        original_score_token = _safe_score_token(original_score, default="unknown")
        target_path = export_dir / _object_crop_export_filename(
            object_global_id,
            row.get("label"),
            normalized_occlusion_level,
            original_score,
        )
        crop_source_path = _resolve_existing_path(db_root, row.get("crop_path"))
        source_kind = ""
        export_status = "skipped"

        if crop_source_path is not None:
            source_image = cv2.imread(str(crop_source_path), cv2.IMREAD_COLOR)
            if source_image is not None and cv2.imwrite(str(target_path), source_image):
                exported_count += 1
                copied_count += 1
                export_status = "exported"
                source_kind = "existing_crop_path"
            else:
                crop_source_path = None

        if crop_source_path is None:
            bbox = _bbox_xyxy_ints_from_row(row)
            image_path = _resolve_existing_path(db_root, row.get("file_name"))
            if bbox is not None and image_path is not None:
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is not None:
                    h, w = image_bgr.shape[:2]
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(x1 + 1, min(w, x2))
                    y2 = max(y1 + 1, min(h, y2))
                    crop_bgr = image_bgr[y1:y2, x1:x2]
                    if crop_bgr.size > 0 and cv2.imwrite(str(target_path), crop_bgr):
                        exported_count += 1
                        regenerated_count += 1
                        export_status = "exported"
                        source_kind = "reconstructed_from_bbox"
            if not source_kind:
                skipped_count += 1
                if bbox is None:
                    export_status = "missing_crop_and_bbox"
                elif image_path is None:
                    export_status = "missing_source_image"
                else:
                    export_status = "crop_write_failed"
                source_kind = "unavailable"

        manifest_rows.append(
            {
                "object_global_id": object_global_id,
                "occlusion_level": normalized_occlusion_level,
                "original_score": original_score,
                "original_score_token": original_score_token,
                "entry_id": row.get("entry_id"),
                "frame_id": row.get("frame_id"),
                "file_name": row.get("file_name"),
                "label": row.get("label"),
                "geometry_source": row.get("geometry_source"),
                "crop_export_status": export_status,
                "crop_export_source": source_kind,
                "source_path": str(crop_source_path) if crop_source_path is not None else "",
                "exported_path": str(target_path) if export_status == "exported" else "",
            }
        )

    manifest_path = export_dir / "manifest.csv"
    _write_csv_rows(manifest_path, _OBJECT_CROPS_BY_GLOBAL_ID_MANIFEST_COLUMNS, manifest_rows)
    return {
        "enabled": True,
        "dir": str(export_dir),
        "manifest_path": str(manifest_path),
        "manifest_count": int(len(manifest_rows)),
        "exported_count": int(exported_count),
        "copied_count": int(copied_count),
        "regenerated_count": int(regenerated_count),
        "skipped_count": int(skipped_count),
    }


def export_object_occlusion_levels_csv(
    *,
    db_root: Path,
    object_rows: Sequence[Mapping[str, Any]],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    csv_path = Path(output_path) if output_path is not None else (db_root / _OBJECT_OCCLUSION_LEVELS_CSV_NAME)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    skipped_count = 0

    for row in list(object_rows):
        try:
            object_global_id = int(row.get("object_global_id"))
        except Exception:
            skipped_count += 1
            continue
        rows.append(
            {
                "object_global_id": object_global_id,
                "occlusion_level": normalize_occlusion_level(row.get("occlusion_level"), default="uncertain"),
            }
        )

    _write_csv_rows(csv_path, _OBJECT_OCCLUSION_LEVELS_COLUMNS, rows)
    return {
        "enabled": True,
        "csv_path": str(csv_path),
        "row_count": int(len(rows)),
        "skipped_count": int(skipped_count),
    }


def _frame_route_label(
    *,
    geometry_object_rows: Sequence[Mapping[str, Any]],
    geometry_all_objects_filtered_by_r_threshold: bool,
) -> str:
    if geometry_object_rows or geometry_all_objects_filtered_by_r_threshold:
        return "mask_depth"
    return "vlm_fallback"


def _compute_object_reweight_fields(
    *,
    detector_confidence: Optional[float],
    object_confidence: Optional[float],
    occlusion_level: Optional[str],
    occlusion_reweight_w1: float = float(OCCLUSION_REWEIGHT_W1),
    occlusion_reweight_w2: float = float(OCCLUSION_REWEIGHT_W2),
    occlusion_reweight_b: float = float(OCCLUSION_REWEIGHT_B),
    occlusion_reweight_eps: float = float(OCCLUSION_REWEIGHT_EPS),
) -> Tuple[str, float, float]:
    normalized_occlusion_level = normalize_occlusion_level(occlusion_level, default="uncertain")
    confidence = detector_confidence if detector_confidence is not None else object_confidence
    confidence_value = float(confidence or 0.0)
    occlusion_penalty = float(map_occlusion_level_to_penalty(normalized_occlusion_level))
    reweighted_detection_score = float(
        compute_reweighted_detection_score(
            confidence_value,
            normalized_occlusion_level,
            w1=occlusion_reweight_w1,
            w2=occlusion_reweight_w2,
            b=occlusion_reweight_b,
            eps=occlusion_reweight_eps,
        )
    )
    return normalized_occlusion_level, occlusion_penalty, reweighted_detection_score


def _filter_geometry_rows_by_r_threshold(
    geometry_object_rows: Sequence[Mapping[str, Any]],
    r_threshold: Optional[float],
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    rows = [dict(row) for row in list(geometry_object_rows or [])]
    before_count = int(len(rows))
    if r_threshold is None:
        return rows, before_count, before_count, 0
    threshold = float(r_threshold)
    filtered_rows: List[Dict[str, Any]] = []
    for row in rows:
        score = row.get("reweighted_detection_score_r")
        if score is None or float(score) >= threshold:
            filtered_rows.append(row)
    after_count = int(len(filtered_rows))
    return filtered_rows, before_count, after_count, int(before_count - after_count)


def _build_object_r_scores_pre_threshold_row(
    row: Mapping[str, Any],
    *,
    entry_id: int,
    frame_id: int,
    file_name: str,
    r_threshold: Optional[float],
) -> Dict[str, Any]:
    geometry_source = str(row.get("geometry_source") or "vlm_fallback")
    route = "vlm_fallback" if geometry_source == "vlm_fallback" else "geometry"
    score = row.get("reweighted_detection_score_r")
    threshold_value = None if r_threshold is None else float(r_threshold)
    would_be_filtered = bool(
        route == "geometry"
        and threshold_value is not None
        and score is not None
        and float(score) < threshold_value
    )
    return {
        "entry_id": int(entry_id),
        "frame_id": int(frame_id),
        "file_name": str(file_name),
        "object_local_id": str(row.get("object_local_id") or ""),
        "object_route": route,
        "label": str(row.get("label") or ""),
        "bbox_xyxy": list(row.get("bbox_xyxy") or []),
        "bbox_xywh_norm": list(row.get("bbox_xywh_norm") or []),
        "object_confidence": row.get("object_confidence"),
        "detector_confidence": row.get("detector_confidence"),
        "occlusion_level": row.get("occlusion_level"),
        "occlusion_penalty_p_o": row.get("occlusion_penalty_p_o"),
        "reweighted_detection_score_r": score,
        "r_threshold_used": threshold_value,
        "would_be_filtered_by_r_threshold": would_be_filtered,
    }


def _write_object_r_scores_pre_threshold_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    _write_csv_rows(path, _OBJECT_R_SCORES_PRE_THRESHOLD_COLUMNS, rows)
    return int(len(list(rows)))


def _load_object_r_scores_pre_threshold_rows(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    if not path.exists() or not path.is_file():
        return {}
    grouped_rows: Dict[int, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                entry_id = int(row.get("entry_id", ""))
            except Exception:
                continue
            grouped_rows.setdefault(entry_id, []).append(dict(row))
    return grouped_rows


def _store_object_r_scores_pre_threshold_rows(
    store: Dict[int, List[Dict[str, Any]]],
    *,
    entry_id: int,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    store[int(entry_id)] = [dict(row) for row in list(rows)]


def _flatten_object_r_scores_pre_threshold_rows(
    rows_by_entry_id: Mapping[int, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for entry_id in sorted(int(key) for key in rows_by_entry_id.keys()):
        flattened.extend(dict(row) for row in list(rows_by_entry_id.get(entry_id, [])))
    return flattened


def _rebuild_object_r_scores_pre_threshold_rows_from_records(
    object_records: Sequence[Mapping[str, Any]],
    *,
    r_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    rebuilt_rows: List[Dict[str, Any]] = []
    for row in list(object_records or []):
        try:
            entry_id = int(row.get("entry_id", -1))
            frame_id = int(row.get("frame_id", entry_id))
        except Exception:
            continue
        rebuilt_rows.append(
            _build_object_r_scores_pre_threshold_row(
                row,
                entry_id=entry_id,
                frame_id=frame_id,
                file_name=str(row.get("file_name") or ""),
                r_threshold=r_threshold,
            )
        )
    return rebuilt_rows


def _write_object_r_scores_csv(path: Path, object_metadata_records: Sequence[Mapping[str, Any]]) -> int:
    rows = [
        {
            "object_global_id": int(record.get("object_global_id") or 0),
            "reweighted_detection_score_r": record.get("reweighted_detection_score_r"),
        }
        for record in list(object_metadata_records or [])
    ]
    _write_csv_rows(path, _OBJECT_R_SCORES_COLUMNS, rows)
    return int(len(rows))


def _serialize_floor_plan_projection(projection: Any) -> Optional[Dict[str, float]]:
    if not isinstance(projection, dict):
        return None
    required = ("view_min_x", "view_max_x", "view_min_y", "view_max_y")
    serialized: Dict[str, float] = {}
    try:
        for key in required:
            serialized[key] = float(projection[key])
    except Exception:
        return None
    return serialized


def _write_floor_plan_projection(path: Path, projection: Any) -> Optional[str]:
    serialized = _serialize_floor_plan_projection(projection)
    if serialized is None:
        return None
    path.write_text(json.dumps(serialized, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)


def _save_faiss_index(embeddings: np.ndarray, output_path: Path) -> int:
    import faiss

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    if embeddings.shape[0] > 0:
        index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(output_path))
    return int(index.ntotal)


def _load_jsonl_if_exists(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_npy_if_exists(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def _response_has_length_finish_reason(raw_api_response: Any) -> bool:
    return VLMCaptioner._response_has_length_finish_reason(raw_api_response)


def _build_view_attribute(scene_objects: Any = None, raw_vlm_output: Any = None) -> Dict[str, Any]:
    attribute = {
        "view_type": "unknown",
        "room_function": "unknown",
        "style_hint": "unknown",
        "clutter_level": "unknown",
        "floor_pattern": "unknown",
        "lighting_ceiling": "unknown",
        "wall_color": "unknown",
        "scene_attributes": [],
        "additional_notes": "",
        "image_summary": "",
    }

    if scene_objects is None and raw_vlm_output not in (None, ""):
        parsed = parse_scene_objects(raw_vlm_output)
        scene_objects = parsed.scene_objects

    if scene_objects is None:
        return attribute

    attribute.update(
        {
            "view_type": str(getattr(scene_objects, "view_type", "unknown") or "unknown").strip() or "unknown",
            "room_function": (
                str(getattr(scene_objects, "room_function", "unknown") or "unknown").strip() or "unknown"
            ),
            "style_hint": str(getattr(scene_objects, "style_hint", "unknown") or "unknown").strip() or "unknown",
            "clutter_level": (
                str(getattr(scene_objects, "clutter_level", "unknown") or "unknown").strip() or "unknown"
            ),
            "floor_pattern": (
                str(getattr(scene_objects, "floor_pattern", "unknown") or "unknown").strip() or "unknown"
            ),
            "lighting_ceiling": (
                str(getattr(scene_objects, "lighting_ceiling", "unknown") or "unknown").strip() or "unknown"
            ),
            "wall_color": str(getattr(scene_objects, "wall_color", "unknown") or "unknown").strip() or "unknown",
            "scene_attributes": [
                str(v).strip() for v in list(getattr(scene_objects, "scene_attributes", []) or []) if str(v).strip()
            ],
            "additional_notes": str(getattr(scene_objects, "additional_notes", "") or "").strip(),
            "image_summary": str(getattr(scene_objects, "image_summary", "") or "").strip(),
        }
    )
    return attribute


def _ensure_metadata_record_attribute(row: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(row)
    existing_attribute = record.get("attribute")
    if isinstance(existing_attribute, dict):
        merged = _build_view_attribute()
        merged.update({str(k): v for k, v in existing_attribute.items()})
        merged["scene_attributes"] = [
            str(v).strip() for v in list(merged.get("scene_attributes", []) or []) if str(v).strip()
        ]
        record["attribute"] = merged
        return record

    record["attribute"] = _build_view_attribute(raw_vlm_output=record.get("raw_vlm_output"))
    return record


def _load_resume_state(output_root: Path, emb_dim: int) -> Dict[str, Any]:
    meta_rows = _load_jsonl_if_exists(output_root / "meta.jsonl")
    raw_api_rows = _load_jsonl_if_exists(output_root / "raw_api_responses.jsonl")
    object_rows = _load_jsonl_if_exists(output_root / "object_meta.jsonl")
    image_arr = _load_npy_if_exists(output_root / "image_emb.npy")
    text_arr_short = _load_npy_if_exists(output_root / "text_emb_short.npy")
    text_arr_long = _load_npy_if_exists(output_root / "text_emb_long.npy")
    object_arr_short = _load_npy_if_exists(output_root / "object_text_emb_short.npy")
    object_arr_long = _load_npy_if_exists(output_root / "object_text_emb_long.npy")
    object_arr_dinov3 = _load_npy_if_exists(output_root / "object_dinov3_emb.npy")

    if not meta_rows or image_arr is None or text_arr_short is None or text_arr_long is None:
        return {
            "metadata_records": [],
            "image_embs": [],
            "text_embs_short": [],
            "text_embs_long": [],
            "raw_api_records": [],
            "object_groups_by_entry_id": {},
            "file_name_to_entry_id": {},
        }

    entry_count = len(meta_rows)
    expected_shapes = [
        image_arr.ndim == 2 and image_arr.shape[0] == entry_count,
        text_arr_short.ndim == 2 and text_arr_short.shape[0] == entry_count,
        text_arr_long.ndim == 2 and text_arr_long.shape[0] == entry_count,
        image_arr.shape[1] == emb_dim,
        text_arr_short.shape[1] == emb_dim,
        text_arr_long.shape[1] == emb_dim,
    ]
    if not all(expected_shapes):
        return {
            "metadata_records": [],
            "image_embs": [],
            "text_embs_short": [],
            "text_embs_long": [],
            "raw_api_records": [],
            "object_groups_by_entry_id": {},
            "file_name_to_entry_id": {},
        }

    object_groups_by_entry_id: Dict[int, List[ObjectGroupItem]] = {}
    if object_rows and object_arr_short is not None and object_arr_long is not None:
        if (
            object_arr_short.ndim == 2
            and object_arr_long.ndim == 2
            and object_arr_short.shape[0] == len(object_rows)
            and object_arr_long.shape[0] == len(object_rows)
            and object_arr_short.shape[1] == emb_dim
            and object_arr_long.shape[1] == emb_dim
        ):
            for idx, row in enumerate(object_rows):
                entry_id = int(row.get("entry_id", -1))
                if entry_id < 0:
                    continue
                dinov3_embedding = None
                dino_row_index = row.get("dinov3_embedding_row_index")
                try:
                    dino_row_index_int = int(dino_row_index) if dino_row_index is not None else None
                except Exception:
                    dino_row_index_int = None
                if (
                    object_arr_dinov3 is not None
                    and dino_row_index_int is not None
                    and object_arr_dinov3.ndim == 2
                    and 0 <= int(dino_row_index_int) < int(object_arr_dinov3.shape[0])
                ):
                    dinov3_embedding = object_arr_dinov3[int(dino_row_index_int)].astype("float32")
                object_groups_by_entry_id.setdefault(entry_id, []).append(
                    (
                        dict(row),
                        object_arr_short[idx].astype("float32"),
                        object_arr_long[idx].astype("float32"),
                        dinov3_embedding,
                    )
                )

    raw_api_by_entry_id: Dict[int, Dict] = {}
    for row in raw_api_rows:
        entry_id = int(row.get("entry_id", -1))
        if entry_id < 0:
            continue
        raw_api_by_entry_id[entry_id] = dict(row)

    metadata_records = [_ensure_metadata_record_attribute(dict(row)) for row in meta_rows]
    raw_api_records: List[Dict] = []
    file_name_to_entry_id: Dict[str, int] = {}
    for entry_idx, row in enumerate(metadata_records):
        entry_id = int(row.get("id", entry_idx))
        if entry_id != entry_idx:
            return {
                "metadata_records": [],
                "image_embs": [],
                "text_embs_short": [],
                "text_embs_long": [],
                "raw_api_records": [],
                "object_groups_by_entry_id": {},
                "file_name_to_entry_id": {},
            }
        file_name = str(row.get("file_name") or "").strip()
        if file_name:
            file_name_to_entry_id[file_name] = entry_id
        raw_api_records.append(
            raw_api_by_entry_id.get(
                entry_id,
                {
                    "entry_id": int(entry_id),
                    "frame_id": int(row.get("frame_id", entry_id)),
                    "file_name": file_name,
                    "raw_api_source": "",
                    "raw_api_response": None,
                    "object_prompt_variant": str(row.get("object_prompt_variant") or ""),
                },
            )
        )

    return {
        "metadata_records": metadata_records,
        "image_embs": [image_arr[idx].astype("float32") for idx in range(entry_count)],
        "text_embs_short": [text_arr_short[idx].astype("float32") for idx in range(entry_count)],
        "text_embs_long": [text_arr_long[idx].astype("float32") for idx in range(entry_count)],
        "raw_api_records": raw_api_records,
        "object_groups_by_entry_id": object_groups_by_entry_id,
        "file_name_to_entry_id": file_name_to_entry_id,
    }


def _should_reuse_existing_entry(
    *,
    existing_meta: Optional[Dict],
    existing_raw_api: Optional[Dict],
    existing_image_emb: Optional[np.ndarray],
    existing_text_emb_short: Optional[np.ndarray],
    existing_text_emb_long: Optional[np.ndarray],
    existing_object_group: Optional[Sequence[ObjectGroupItem]],
    expected_file_name: str,
    require_geometry_fields: bool = False,
) -> bool:
    if not existing_meta or not isinstance(existing_meta, dict):
        return False
    if str(existing_meta.get("file_name") or "").strip() != str(expected_file_name or "").strip():
        return False
    if existing_image_emb is None or existing_text_emb_short is None or existing_text_emb_long is None:
        return False
    if existing_object_group is None or len(existing_object_group) == 0:
        return False
    if not existing_raw_api or not isinstance(existing_raw_api, dict):
        return False
    if _response_has_length_finish_reason(existing_raw_api.get("raw_api_response")):
        return False
    if require_geometry_fields:
        first_record = dict(existing_object_group[0][0]) if existing_object_group and existing_object_group[0] else {}
        geometry_source = str(first_record.get("geometry_source") or "").strip()
        if geometry_source not in {"mask_depth", "vlm_fallback"}:
            return False
    return True


def _parse_objects_with_retry(
    captioner: VLMCaptioner,
    image_path: str,
    image_id: str,
    max_objects: int,
    retries: int,
    prompt_variant: str = "standard",
    camera_context: Optional[Dict[str, float]] = None,
) -> ParseResult:
    retries = max(0, int(retries))
    last_result: Optional[ParseResult] = None
    for attempt in range(retries + 1):
        result = captioner.extract_objects_with_meta(
            image_path=image_path,
            max_objects=max_objects,
            force_refresh=attempt > 0,
            prompt_variant=prompt_variant,
            camera_context=camera_context,
        )
        raw_output = result.get("raw_json", "")
        parsed = parse_scene_objects(raw_output, image_context={"image_id": image_id})
        parsed.raw_api_response = result.get("raw_api_response")
        parsed.raw_api_source = str(result.get("source") or "")
        if parsed.parse_status == "ok":
            return parsed
        last_result = parsed

    if last_result is None:
        return ParseResult(
            scene_objects=None,
            parse_status="fallback",
            warnings=["object parsing failed and no parser output was captured"],
            raw_vlm_output="",
            raw_api_response=None,
            raw_api_source="missing",
        )

    return ParseResult(
        scene_objects=None,
        parse_status="fallback",
        warnings=last_result.warnings,
        raw_vlm_output=last_result.raw_vlm_output,
        raw_api_response=last_result.raw_api_response,
        raw_api_source=last_result.raw_api_source,
    )


def _make_thread_local_captioner_getter(
    *,
    model_name: str,
    use_cache: bool,
    cache_dir: str,
    object_use_cache: bool,
    object_cache_dir: str,
) -> Callable[[], VLMCaptioner]:
    thread_local = threading.local()

    def _get_captioner() -> VLMCaptioner:
        captioner = getattr(thread_local, "captioner", None)
        if captioner is None:
            captioner = VLMCaptioner(
                model_name=model_name,
                use_cache=use_cache,
                cache_dir=cache_dir,
                object_use_cache=object_use_cache,
                object_cache_dir=object_cache_dir,
            )
            thread_local.captioner = captioner
        return captioner

    return _get_captioner


def _summarize_geometry_outcomes_from_object_groups(
    object_groups_by_entry_id: Mapping[int, Sequence[ObjectGroupItem]],
) -> Tuple[int, int]:
    geometry_ok_count = 0
    geometry_fallback_count = 0
    for _entry_id, groups in object_groups_by_entry_id.items():
        rows = [dict(item[0]) for item in list(groups or []) if item]
        if not rows:
            continue
        sources = {str(row.get("geometry_source") or "").strip() for row in rows if isinstance(row, dict)}
        sources.discard("")
        if not sources:
            continue
        if any(source != "vlm_fallback" for source in sources):
            geometry_ok_count += 1
        else:
            geometry_fallback_count += 1
    return int(geometry_ok_count), int(geometry_fallback_count)


def _run_parallel_vlm_stage(
    *,
    jobs: Sequence[Dict[str, Any]],
    max_in_flight: int,
    worker: Callable[[Dict[str, Any], VLMCaptioner], Any],
    captioner_getter: Callable[[], VLMCaptioner],
    stage_name: str,
) -> Dict[int, Dict[str, Any]]:
    if not jobs:
        return {}

    max_workers = max(1, int(max_in_flight))
    results_by_frame_idx: Dict[int, Dict[str, Any]] = {}

    def _wrapped(job: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = worker(job, captioner_getter())
        return {
            "result": result,
            "elapsed_sec": float(time.perf_counter() - t0),
        }

    _builder_log(
        f"{stage_name}_batch_start count={len(jobs)} max_in_flight={max_workers}"
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame_idx = {
            executor.submit(_wrapped, dict(job)): int(job["frame_idx"])
            for job in jobs
        }
        for future in as_completed(future_to_frame_idx):
            frame_idx = int(future_to_frame_idx[future])
            results_by_frame_idx[frame_idx] = dict(future.result())
    _builder_log(
        f"{stage_name}_batch_done count={len(jobs)} max_in_flight={max_workers}"
    )
    return results_by_frame_idx


def _apply_batched_description_result_to_geometry(
    geometry_result: Any,
    *,
    description_result: Optional[Mapping[str, Any]],
    description_total_sec: float,
) -> Any:
    default_payload = VLMCaptioner._default_object_crop_description(include_occlusion=False)
    request_by_local_id = {
        str(item.get("object_local_id") or "").strip(): dict(item)
        for item in list(getattr(geometry_result, "description_requests", []) or [])
        if str(item.get("object_local_id") or "").strip()
    }
    description_by_local_id: Dict[str, Dict[str, Any]] = {}
    for item in list((description_result or {}).get("objects") or []):
        if not isinstance(item, Mapping):
            continue
        object_local_id = str(item.get("object_local_id") or "").strip()
        if not object_local_id or object_local_id in description_by_local_id:
            continue
        label_hint = str(request_by_local_id.get(object_local_id, {}).get("detector_label") or "")
        description_by_local_id[object_local_id] = VLMCaptioner._normalize_object_description_payload(
            item,
            default_payload=default_payload,
            label_hint=label_hint,
            include_occlusion=False,
        )

    request_count = max(len(request_by_local_id), 1)
    description_per_object_sec = float(description_total_sec / request_count)
    for row in list(getattr(geometry_result, "object_rows", []) or []):
        object_local_id = str(row.get("object_local_id") or "").strip()
        label_hint = str(
            request_by_local_id.get(object_local_id, {}).get("detector_label")
            or row.get("detector_label")
            or row.get("label")
            or "unknown"
        )
        payload = description_by_local_id.get(object_local_id)
        if payload is None:
            payload = VLMCaptioner._normalize_object_description_payload(
                {},
                default_payload=default_payload,
                label_hint=label_hint,
                include_occlusion=False,
            )
        short_description = str(payload.get("short_description") or label_hint).strip() or label_hint
        long_description = (
            str(payload.get("long_description") or payload.get("short_description") or label_hint).strip()
            or label_hint
        )
        detector_label_raw = str(row.get("detector_label_raw") or row.get("detector_label") or label_hint).strip() or "unknown"
        detector_label_norm = canonicalize_household_object_label(
            detector_label_raw,
            default=detector_label_raw,
        )
        vlm_label = canonicalize_household_object_label(
            payload.get("label"),
            default="unknown",
        )
        vlm_label_usable = bool(str(vlm_label or "").strip()) and str(vlm_label).strip().lower() != "unknown"
        final_label = vlm_label if vlm_label_usable else detector_label_raw
        row["label"] = final_label
        row["detector_label"] = detector_label_raw
        row["detector_label_raw"] = detector_label_raw
        row["vlm_label"] = vlm_label
        row["crop_vlm_label"] = vlm_label
        row["final_label"] = final_label
        row["label_source"] = "vlm" if vlm_label_usable else "detector"
        row["label_conflict"] = bool(
            vlm_label_usable
            and str(vlm_label).strip().lower() != str(detector_label_norm).strip().lower()
        )
        row["description"] = short_description
        row["long_form_open_description"] = long_description
        row["attributes"] = [str(v).strip() for v in list(payload.get("attributes") or []) if str(v).strip()]
        row["object_text_short"] = short_description
        row["object_text_long"] = long_description
        row["text_input_for_clip_short"] = short_description
        row["text_input_for_clip_long"] = long_description
        row["vlm_distance_from_camera_m"] = payload.get("distance_from_camera_m")
        row["timing_crop_vlm_description_sec"] = float(description_per_object_sec)

    timings = dict(getattr(geometry_result, "timings", {}) or {})
    object_count = int(len(list(getattr(geometry_result, "object_rows", []) or [])))
    timings["crop_vlm_description_total_sec"] = float(description_total_sec)
    timings["crop_vlm_description_per_object_sec"] = [float(description_per_object_sec) for _ in range(object_count)]
    timings["crop_vlm_description_avg_sec"] = float(description_per_object_sec if object_count > 0 else 0.0)
    timings["object_description_call_count"] = int(1 if object_count > 0 else 0)
    timings["total_sec"] = float(timings.get("total_sec") or 0.0) + float(description_total_sec)
    geometry_result.timings = timings
    return geometry_result


def _frame_text_from_object_rows(rows: Sequence[Dict[str, Any]], mode: str = "short") -> str:
    key = "object_text_short" if str(mode or "short").strip().lower() == "short" else "object_text_long"
    values = [str(row.get(key) or "").strip() for row in list(rows or []) if str(row.get(key) or "").strip()]
    if not values:
        return UNKNOWN_TEXT_TOKEN
    return " | ".join(values)


def _view_attribute_from_selector_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    selector_payload = dict(payload or {})
    return {
        "view_type": str(selector_payload.get("view_type") or "unknown").strip() or "unknown",
        "room_function": str(selector_payload.get("room_function") or "unknown").strip() or "unknown",
        "style_hint": str(selector_payload.get("style_hint") or "unknown").strip() or "unknown",
        "clutter_level": str(selector_payload.get("clutter_level") or "unknown").strip() or "unknown",
        "floor_pattern": str(selector_payload.get("floor_pattern") or "unknown").strip() or "unknown",
        "lighting_ceiling": str(selector_payload.get("lighting_ceiling") or "unknown").strip() or "unknown",
        "wall_color": str(selector_payload.get("wall_color") or "unknown").strip() or "unknown",
        "scene_attributes": [
            str(v).strip() for v in list(selector_payload.get("scene_attributes") or []) if str(v).strip()
        ],
        "additional_notes": str(selector_payload.get("additional_notes") or "").strip(),
        "image_summary": str(selector_payload.get("image_summary") or "").strip(),
    }


def _normalize_angle_bucket(laterality: Optional[str], angle_split_enable: bool) -> str:
    if not angle_split_enable:
        return "center"
    value = str(laterality or "").strip().lower()
    if value in {"left", "center", "right"}:
        return value
    return "center"


def _compute_object_orientation(
    frame_orientation: int,
    laterality: Optional[str],
    angle_split_enable: bool,
    angle_step: int,
) -> int:
    bucket = _normalize_angle_bucket(laterality, angle_split_enable=angle_split_enable)
    orientation = int(frame_orientation) % 360
    if bucket == "left":
        return (orientation - int(angle_step)) % 360
    if bucket == "right":
        return (orientation + int(angle_step)) % 360
    return orientation


def _format_object_text_long(text: str, angle_bucket: str, builder_variant: str) -> str:
    if str(builder_variant) != "angle_split":
        return text
    normalized = str(text or "").strip() or UNKNOWN_TEXT_TOKEN
    bucket = _normalize_angle_bucket(angle_bucket, angle_split_enable=True)
    return f"{bucket} sector | {normalized}"


def _normalize_bearing_deg(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        bearing = float(value)
    except Exception:
        return None
    if not math.isfinite(bearing):
        return None
    if bearing < -90.0 or bearing > 90.0:
        return None
    return bearing


def _project_global_xy(
    origin_x: float,
    origin_y: float,
    camera_orientation_deg: float,
    relative_bearing_deg: Optional[float],
    distance_m: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if relative_bearing_deg is None or distance_m is None:
        return None, None
    dist = float(distance_m)
    if not math.isfinite(dist) or dist < 0.0:
        return None, None
    global_bearing = (float(camera_orientation_deg) - float(relative_bearing_deg)) % 360.0
    yaw = math.radians(global_bearing)
    projected_x = float(origin_x - math.sin(yaw) * dist)
    projected_y = float(origin_y - math.cos(yaw) * dist)
    return projected_x, projected_y


def _fallback_relative_bearing_from_laterality(laterality: Optional[str], angle_step: int) -> float:
    bucket = _normalize_angle_bucket(laterality, angle_split_enable=True)
    if bucket == "left":
        return float(-int(angle_step))
    if bucket == "right":
        return float(int(angle_step))
    return 0.0


def _round_location_coord(value: Optional[float]) -> str:
    if value is None:
        return "na"
    rounded = round(float(value) * 2.0) / 2.0
    return f"{rounded:.1f}"


def _round_location_dist(value: Optional[float]) -> str:
    if value is None:
        return "na"
    return f"{round(float(value), 1):.1f}"


def _build_location_summary_from_surroundings(surrounding_context: Sequence[Dict[str, Any]]) -> str:
    if not surrounding_context:
        return ""
    rendered: List[str] = []
    for item in list(surrounding_context)[: int(OBJECT_SURROUNDING_MAX)]:
        label = str(item.get("label") or "").strip() or "unknown"
        relation = str(item.get("relation_to_primary") or "").strip() or "unknown"
        rendered.append(
            f"{label} relation={relation} d={_round_location_dist(item.get('distance_from_primary_m'))}m "
            f"anchor=({_round_location_coord(item.get('estimated_global_x'))},{_round_location_coord(item.get('estimated_global_y'))})"
        )
    return "; ".join(rendered)


def _serialize_surrounding_context(ctx_items: Any) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for item in list(ctx_items or [])[: int(OBJECT_SURROUNDING_MAX)]:
        serialized.append(
            {
                "label": str(getattr(item, "label", "") or "unknown").strip() or "unknown",
                "attributes": [str(v).strip() for v in list(getattr(item, "attributes", []) or []) if str(v).strip()],
                "distance_from_primary_m": getattr(item, "distance_from_primary_m", None),
                "distance_from_camera_m": getattr(item, "distance_from_camera_m", None),
                "relative_height_from_camera_m": getattr(item, "relative_height_from_camera_m", None),
                "relative_bearing_deg": getattr(item, "relative_bearing_deg", None),
                "estimated_global_x": getattr(item, "estimated_global_x", None),
                "estimated_global_y": getattr(item, "estimated_global_y", None),
                "estimated_global_z": getattr(item, "estimated_global_z", None),
                "relation_to_primary": str(getattr(item, "relation_to_primary", "") or "").strip(),
            }
        )
    return serialized


def _obs_id_for_object_global_id(object_global_id: Any) -> str:
    return f"obs_{int(object_global_id):06d}"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _classify_view_aligned_direction(
    dx: float,
    dy: float,
    view_orientation_deg: float,
    same_axis_eps: float = 0.25,
) -> Optional[str]:
    distance = float(math.hypot(float(dx), float(dy)))
    if distance <= float(same_axis_eps):
        return None

    yaw = math.radians(float(view_orientation_deg))
    forward_x = -math.sin(yaw)
    forward_y = -math.cos(yaw)
    right_x = math.cos(yaw)
    right_y = -math.sin(yaw)

    local_forward = float(dx) * forward_x + float(dy) * forward_y
    local_right = float(dx) * right_x + float(dy) * right_y
    if abs(local_right) >= abs(local_forward):
        if local_right > float(same_axis_eps):
            return "right"
        if local_right < -float(same_axis_eps):
            return "left"
        return None
    if local_forward > float(same_axis_eps):
        return "in front"
    if local_forward < -float(same_axis_eps):
        return "behind"
    return None


def _entry_camera_z(entry: Dict[str, Any]) -> float:
    world_position = entry.get("world_position")
    if isinstance(world_position, (list, tuple)) and len(world_position) >= 3:
        z = _safe_float(world_position[2])
        if z is not None:
            return float(z)
    return 0.0


def _estimated_global_z(camera_z: float, relative_height_from_camera_m: Any) -> Optional[float]:
    relative_height = _safe_float(relative_height_from_camera_m)
    if relative_height is None:
        return None
    return float(camera_z + relative_height)


def _classify_vertical_direction(dy: float, vertical_eps: float = OBJECT_VERTICAL_REL_EPS_M) -> str:
    if float(dy) > float(vertical_eps):
        return "above"
    if float(dy) < -float(vertical_eps):
        return "below"
    return "level"


def _build_view_object_relations(
    metadata_records: Sequence[Dict[str, Any]],
    object_metadata_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entry_by_id = {int(row["id"]): dict(row) for row in metadata_records if row.get("id") is not None}
    relations: List[Dict[str, Any]] = []
    for row in object_metadata_records:
        entry_id = int(row.get("entry_id", -1))
        entry = entry_by_id.get(entry_id)
        if entry is None:
            continue
        object_x = _safe_float(row.get("estimated_global_x"))
        object_y = _safe_float(row.get("estimated_global_y"))
        if object_x is None or object_y is None:
            continue

        view_x = _safe_float(entry.get("x"))
        view_y = _safe_float(entry.get("y"))
        if view_x is None or view_y is None:
            continue
        view_z = _entry_camera_z(entry)
        object_z = _safe_float(row.get("estimated_global_z"))
        dx = float(object_x - view_x)
        dy = float(object_y - view_y)
        dz = float(object_z - view_z) if object_z is not None else 0.0
        relations.append(
            {
                "entry_id": entry_id,
                "view_id": f"view_{entry_id:05d}",
                "object_global_id": int(row["object_global_id"]),
                "obs_id": _obs_id_for_object_global_id(row["object_global_id"]),
                "label": str(row.get("label") or "unknown"),
                "view_x": view_x,
                "view_y": view_y,
                "view_z": view_z,
                "object_x": object_x,
                "object_y": object_y,
                "object_z": object_z,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "distance_m": float(math.hypot(dx, dy)),
                "distance_3d_m": float(math.sqrt(dx * dx + dy * dy + dz * dz)),
                "direction": "in",
                "direction_frame": "view_aligned",
                "vertical_direction": _classify_vertical_direction(dy=dz),
                "relation_type": "ViewObject",
            }
        )
    return relations


def _build_object_object_relations(
    metadata_records: Sequence[Dict[str, Any]],
    object_metadata_records: Sequence[Dict[str, Any]],
    same_axis_eps: float = 0.25,
) -> List[Dict[str, Any]]:
    entry_by_id = {int(row["id"]): dict(row) for row in metadata_records if row.get("id") is not None}
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in object_metadata_records:
        object_x = _safe_float(row.get("estimated_global_x"))
        object_y = _safe_float(row.get("estimated_global_y"))
        if object_x is None or object_y is None:
            continue
        grouped.setdefault(int(row["entry_id"]), []).append(dict(row))

    relations: List[Dict[str, Any]] = []
    for entry_id, rows in grouped.items():
        entry = entry_by_id.get(int(entry_id))
        if entry is None:
            continue
        view_orientation = _safe_float(entry.get("orientation"))
        if view_orientation is None:
            continue
        view_id = f"view_{int(entry_id):05d}"
        ordered = sorted(rows, key=lambda item: int(item["object_global_id"]))
        for source in ordered:
            source_x = _safe_float(source.get("estimated_global_x"))
            source_y = _safe_float(source.get("estimated_global_y"))
            if source_x is None or source_y is None:
                continue
            source_z = _safe_float(source.get("estimated_global_z"))
            for target in ordered:
                if int(source["object_global_id"]) == int(target["object_global_id"]):
                    continue
                target_x = _safe_float(target.get("estimated_global_x"))
                target_y = _safe_float(target.get("estimated_global_y"))
                if target_x is None or target_y is None:
                    continue
                target_z = _safe_float(target.get("estimated_global_z"))
                dx = float(target_x - source_x)
                dy = float(target_y - source_y)
                dz = float(target_z - source_z) if source_z is not None and target_z is not None else 0.0
                direction = _classify_view_aligned_direction(
                    dx=dx,
                    dy=dy,
                    view_orientation_deg=view_orientation,
                    same_axis_eps=same_axis_eps,
                )
                if direction is None:
                    continue
                relations.append(
                    {
                        "entry_id": int(entry_id),
                        "view_id": view_id,
                        "source_object_global_id": int(source["object_global_id"]),
                        "target_object_global_id": int(target["object_global_id"]),
                        "source_obs_id": _obs_id_for_object_global_id(source["object_global_id"]),
                        "target_obs_id": _obs_id_for_object_global_id(target["object_global_id"]),
                        "source_label": str(source.get("label") or "unknown"),
                        "target_label": str(target.get("label") or "unknown"),
                        "source_x": source_x,
                        "source_y": source_y,
                        "source_z": source_z,
                        "target_x": target_x,
                        "target_y": target_y,
                        "target_z": target_z,
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "distance_m": float(math.hypot(dx, dy)),
                        "distance_3d_m": float(math.sqrt(dx * dx + dy * dy + dz * dz)),
                        "direction": direction,
                        "direction_frame": "view_aligned",
                        "vertical_direction": _classify_vertical_direction(dy=dz),
                        "relation_type": "ObjectObject",
                        "relation_source": "geometry_postprocess",
                    }
                )
    return relations


def _get_polar_surroundings_builder():
    from spatial_rag.polar_surrounding_postprocess import build_polar_surroundings

    return build_polar_surroundings


def _run_optional_polar_surrounding_postprocess(
    output_root: Path,
    *,
    enabled: bool,
) -> Dict[str, Any]:
    if not bool(enabled):
        return {
            "enabled": False,
            "ran": False,
            "ok": False,
        }

    build_polar_surroundings = _get_polar_surroundings_builder()
    summary = build_polar_surroundings(str(output_root))
    return {
        "enabled": True,
        "ran": True,
        "ok": True,
        **dict(summary),
    }


def _enrich_scene_objects_geometry(
    scene_objects,
    camera_x: float,
    camera_y: float,
    camera_z: float,
    camera_orientation_deg: float,
    angle_step: int,
) -> None:
    for obj in list(scene_objects.visual_feature or []):
        bearing = _normalize_bearing_deg(getattr(obj, "relative_bearing_deg", None))
        if bearing is None and getattr(obj, "distance_from_camera_m", None) is not None:
            bearing = _fallback_relative_bearing_from_laterality(
                getattr(obj, "relative_position_laterality", "center"),
                angle_step=angle_step,
            )
        obj.relative_bearing_deg = bearing
        projected_x, projected_y = _project_global_xy(
            origin_x=camera_x,
            origin_y=camera_y,
            camera_orientation_deg=camera_orientation_deg,
            relative_bearing_deg=bearing,
            distance_m=getattr(obj, "distance_from_camera_m", None),
        )
        obj.estimated_global_x = projected_x
        obj.estimated_global_y = projected_y
        obj.estimated_global_z = _estimated_global_z(
            camera_z=camera_z,
            relative_height_from_camera_m=getattr(obj, "relative_height_from_camera_m", None),
        )

        enriched_ctx = []
        for ctx in list(getattr(obj, "surrounding_context", []) or [])[: int(OBJECT_SURROUNDING_MAX)]:
            ctx_bearing = _normalize_bearing_deg(getattr(ctx, "relative_bearing_deg", None))
            ctx.relative_bearing_deg = ctx_bearing
            ctx_x, ctx_y = _project_global_xy(
                origin_x=camera_x,
                origin_y=camera_y,
                camera_orientation_deg=camera_orientation_deg,
                relative_bearing_deg=ctx_bearing,
                distance_m=getattr(ctx, "distance_from_camera_m", None),
            )
            ctx.estimated_global_x = ctx_x
            ctx.estimated_global_y = ctx_y
            ctx.estimated_global_z = _estimated_global_z(
                camera_z=camera_z,
                relative_height_from_camera_m=getattr(ctx, "relative_height_from_camera_m", None),
            )
            enriched_ctx.append(ctx)
        obj.surrounding_context = enriched_ctx
        obj.location_relative_to_other_objects = _build_location_summary_from_surroundings(
            _serialize_surrounding_context(enriched_ctx)
        )


def _make_object_record(
    *,
    object_global_id: int,
    frame_id: int,
    entry_id: int,
    file_name: str,
    x: float,
    y: float,
    z: float,
    world_position: List[float],
    orientation: int,
    parse_status: str,
    builder_variant: str,
    angle_split_enable: bool,
    angle_step: int,
    scene_objects=None,
    obj=None,
    object_local_id: str,
    label: str,
    object_confidence: float,
    description: str = "",
    long_form_open_description: str = "",
    attributes: Optional[List[str]] = None,
    laterality: str = "center",
    distance_bin: str = "middle",
    verticality: str = "middle",
    distance_from_camera_m=None,
    relative_height_from_camera_m=None,
    relative_bearing_deg=None,
    estimated_global_x=None,
    estimated_global_y=None,
    estimated_global_z=None,
    any_text: str = "",
    location_relative_to_other_objects: str = "",
    surrounding_context: Optional[List[Dict[str, Any]]] = None,
    scene_attributes: Optional[List[str]] = None,
    object_text_short: str = UNKNOWN_TEXT_TOKEN,
    object_text_long: str = UNKNOWN_TEXT_TOKEN,
    precise_orientation_from_bearing: bool = False,
    geometry_source: str = "vlm_fallback",
    geometry_fallback_reason: Optional[str] = None,
    detector_label: Optional[str] = None,
    detector_label_raw: Optional[str] = None,
    detector_confidence: Optional[float] = None,
    vlm_label: Optional[str] = None,
    final_label: Optional[str] = None,
    label_source: Optional[str] = None,
    label_conflict: Optional[bool] = None,
    occlusion_source: Optional[str] = None,
    occlusion_level: Optional[str] = None,
    occlusion_penalty_p_o: Optional[float] = None,
    reweighted_detection_score_r: Optional[float] = None,
    visible_occlusion_ratio: Optional[float] = None,
    occluded_boundary_ratio: Optional[float] = None,
    nearer_ring_overlap_ratio: Optional[float] = None,
    object_depth_median: Optional[float] = None,
    boundary_pixel_count: Optional[int] = None,
    occluded_boundary_pixel_count: Optional[int] = None,
    ring_pixel_count: Optional[int] = None,
    nearer_ring_pixel_count: Optional[int] = None,
    depth_margin_delta: Optional[float] = None,
    occluding_overlap_pixel_count: Optional[int] = None,
    foreground_occluder_count: Optional[int] = None,
    occlusion_target_overlap_threshold: Optional[float] = None,
    bbox_xywh_norm: Optional[Sequence[float]] = None,
    bbox_xyxy: Optional[Sequence[float]] = None,
    mask_area_px: Optional[int] = None,
    mask_area_ratio: Optional[float] = None,
    mask_centroid_x_px: Optional[float] = None,
    mask_centroid_y_px: Optional[float] = None,
    mask_centroid_x_norm: Optional[float] = None,
    mask_centroid_y_norm: Optional[float] = None,
    depth_stat_median_m: Optional[float] = None,
    depth_stat_p10_m: Optional[float] = None,
    depth_stat_p90_m: Optional[float] = None,
    projected_planar_distance_m: Optional[float] = None,
    vertical_angle_deg: Optional[float] = None,
    vlm_distance_from_camera_m: Optional[float] = None,
    vlm_relative_bearing_deg: Optional[float] = None,
    crop_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    mask_overlay_path: Optional[str] = None,
    depth_map_path: Optional[str] = None,
    crop_vlm_label: Optional[str] = None,
    dinov3_embedding_row_index: Optional[int] = None,
    dinov3_model_name: Optional[str] = None,
    dinov3_embedding_dim: Optional[int] = None,
    dinov3_input_type: Optional[str] = None,
    dinov3_normalized: Optional[bool] = None,
    dinov3_status: Optional[str] = None,
    dinov3_failure_reason: Optional[str] = None,
) -> Dict:
    angle_bucket = _normalize_angle_bucket(laterality, angle_split_enable=angle_split_enable)
    if precise_orientation_from_bearing and relative_bearing_deg is not None:
        object_orientation_deg = int(round((int(orientation) - float(relative_bearing_deg)) % 360.0)) % 360
    else:
        object_orientation_deg = _compute_object_orientation(
            frame_orientation=orientation,
            laterality=angle_bucket,
            angle_split_enable=angle_split_enable,
            angle_step=angle_step,
        )
    record = {
        "object_global_id": object_global_id,
        "frame_id": int(frame_id),
        "entry_id": int(entry_id),
        "file_name": file_name,
        "x": x,
        "y": y,
        "z": z,
        "world_position": world_position,
        "orientation": int(orientation),
        "frame_orientation": int(orientation),
        "object_orientation_deg": int(object_orientation_deg),
        "angle_bucket": angle_bucket,
        "angle_split_step_deg": int(angle_step),
        "builder_variant": str(builder_variant),
        "object_local_id": object_local_id,
        "label": label,
        "object_confidence": float(object_confidence),
        "bbox_xywh_norm": [float(v) for v in list(bbox_xywh_norm or [0.0, 0.0, 0.0, 0.0])[:4]],
        "bbox_xyxy": [float(v) for v in list(bbox_xyxy or [])[:4]],
        "facing": "unknown",
        "orientation_confidence": 0.0,
        "description": description,
        "long_form_open_description": long_form_open_description,
        "attributes": [str(v).strip() for v in list(attributes or []) if str(v).strip()],
        "laterality": laterality,
        "distance_bin": distance_bin,
        "verticality": verticality,
        "distance_from_camera_m": distance_from_camera_m,
        "relative_height_from_camera_m": relative_height_from_camera_m,
        "relative_bearing_deg": relative_bearing_deg,
        "estimated_global_x": estimated_global_x,
        "estimated_global_y": estimated_global_y,
        "estimated_global_z": estimated_global_z,
        "any_text": any_text,
        "location_relative_to_other_objects": location_relative_to_other_objects,
        "surrounding_context": list(surrounding_context or []),
        "scene_attributes": [str(v).strip() for v in list(scene_attributes or []) if str(v).strip()],
        "object_text_short": object_text_short,
        "object_text_long": object_text_long,
        "text_input_for_clip_short": object_text_short,
        "text_input_for_clip_long": object_text_long,
        "parse_status": parse_status,
        "geometry_source": str(geometry_source or "vlm_fallback"),
        "geometry_fallback_reason": geometry_fallback_reason,
        "detector_label": detector_label_raw if detector_label_raw is not None else detector_label,
        "detector_label_raw": detector_label_raw if detector_label_raw is not None else detector_label,
        "detector_confidence": detector_confidence,
        "vlm_label": vlm_label if vlm_label is not None else crop_vlm_label,
        "final_label": final_label if final_label is not None else label,
        "label_source": label_source,
        "label_conflict": None if label_conflict is None else bool(label_conflict),
        "occlusion_source": None if occlusion_source is None else str(occlusion_source),
        "occlusion_level": occlusion_level,
        "occlusion_penalty_p_o": occlusion_penalty_p_o,
        "reweighted_detection_score_r": reweighted_detection_score_r,
        "visible_occlusion_ratio": visible_occlusion_ratio,
        "occluded_boundary_ratio": occluded_boundary_ratio,
        "nearer_ring_overlap_ratio": nearer_ring_overlap_ratio,
        "object_depth_median": object_depth_median,
        "boundary_pixel_count": boundary_pixel_count,
        "occluded_boundary_pixel_count": occluded_boundary_pixel_count,
        "ring_pixel_count": ring_pixel_count,
        "nearer_ring_pixel_count": nearer_ring_pixel_count,
        "depth_margin_delta": depth_margin_delta,
        "occluding_overlap_pixel_count": occluding_overlap_pixel_count,
        "foreground_occluder_count": foreground_occluder_count,
        "occlusion_target_overlap_threshold": occlusion_target_overlap_threshold,
        "mask_area_px": mask_area_px,
        "mask_area_ratio": mask_area_ratio,
        "mask_centroid_x_px": mask_centroid_x_px,
        "mask_centroid_y_px": mask_centroid_y_px,
        "mask_centroid_x_norm": mask_centroid_x_norm,
        "mask_centroid_y_norm": mask_centroid_y_norm,
        "depth_stat_median_m": depth_stat_median_m,
        "depth_stat_p10_m": depth_stat_p10_m,
        "depth_stat_p90_m": depth_stat_p90_m,
        "projected_planar_distance_m": projected_planar_distance_m,
        "vertical_angle_deg": vertical_angle_deg,
        "vlm_distance_from_camera_m": vlm_distance_from_camera_m,
        "vlm_relative_bearing_deg": vlm_relative_bearing_deg,
        "crop_path": crop_path,
        "mask_path": mask_path,
        "mask_overlay_path": mask_overlay_path,
        "depth_map_path": depth_map_path,
        "crop_vlm_label": crop_vlm_label if crop_vlm_label is not None else vlm_label,
        "dinov3_embedding_row_index": dinov3_embedding_row_index,
        "dinov3_model_name": dinov3_model_name,
        "dinov3_embedding_dim": dinov3_embedding_dim,
        "dinov3_input_type": dinov3_input_type,
        "dinov3_normalized": dinov3_normalized,
        "dinov3_status": dinov3_status,
        "dinov3_failure_reason": dinov3_failure_reason,
    }
    if scene_objects is not None:
        record.update(
            {
                "view_type": scene_objects.view_type,
                "room_function": scene_objects.room_function,
                "style_hint": scene_objects.style_hint,
                "clutter_level": scene_objects.clutter_level,
            }
        )
    return record


def _build_spatial_database_core(
    scene_path: str,
    meters_per_step: float,
    max_positions: Optional[int],
    output_dir: str,
    vlm_model: str,
    use_cache: bool,
    object_max_per_frame: int,
    object_parse_retries: int,
    object_use_cache: bool,
    object_cache_dir: Optional[str],
    tour_mode: str,
    random_num_steps: int,
    random_step_size: float,
    random_scan_angles: Sequence[int],
    random_seed: Optional[int],
    random_max_attempts_per_step: int,
    random_include_start_scan: bool,
    object_prompt_variant: str,
    object_orientation_mode: str,
    report_builder_variant: str,
    angle_split_enable: bool,
    angle_step: int,
    run_polar_surrounding_postprocess: bool,
    execution_mode: str = "capture_then_parallel_vlm",
    vlm_max_in_flight: int = 4,
    legacy_per_frame: bool = False,
    bbox_conf_threshold: float = float(BBOX_CONF_THRESHOLD),
    occlusion_reweight_w1: float = float(OCCLUSION_REWEIGHT_W1),
    occlusion_reweight_w2: float = float(OCCLUSION_REWEIGHT_W2),
    occlusion_reweight_b: float = float(OCCLUSION_REWEIGHT_B),
    occlusion_source: str = str(OCCLUSION_SOURCE),
    occlusion_target_overlap_threshold: float = float(OCCLUSION_TARGET_OVERLAP_THRESHOLD),
    visible_occ_boundary_width: int = int(VISIBLE_OCC_BOUNDARY_WIDTH),
    visible_occ_ring_radius: int = int(VISIBLE_OCC_RING_RADIUS),
    visible_occ_depth_margin_delta: float = float(VISIBLE_OCC_DEPTH_MARGIN_DELTA),
    visible_occ_boundary_neighbor_radius: int = int(VISIBLE_OCC_BOUNDARY_NEIGHBOR_RADIUS),
    enable_dinov3_embedding: bool = bool(ENABLE_DINOV3_EMBEDDING),
    store_dinov3_embedding: bool = bool(STORE_DINOV3_EMBEDDING),
    dinov3_model_name: str = str(DINOV3_MODEL_NAME),
    dinov3_batch_size: int = int(DINOV3_BATCH_SIZE),
    dinov3_normalize: bool = bool(DINOV3_NORMALIZE),
    r_threshold: Optional[float] = None,
    export_object_crops_by_global_id_dir: Optional[str] = None,
) -> Dict:
    try:
        from spatial_rag.embedder import DINOv3Embedder, Embedder
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import Embedder dependencies. "
            "Ensure the current Python environment has torch plus either open_clip or clip installed."
        ) from exc

    try:
        from spatial_rag.explorer import Explorer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import Habitat dependencies. "
            "Please run inside your Habitat environment (e.g. `conda activate habitat`) "
            "and ensure `habitat_sim` is installed."
        ) from exc

    normalized_scan_angles = _normalize_scan_angles(random_scan_angles)
    selected_builder_variant = str(report_builder_variant or "standard").strip().lower()
    selected_prompt_variant = str(object_prompt_variant or "standard").strip().lower()
    angle_split_active = bool(angle_split_enable and object_orientation_mode == "laterality_offset")
    selected_occlusion_source = str(occlusion_source or OCCLUSION_SOURCE).strip().lower()
    dinov3_enabled = bool(enable_dinov3_embedding)
    dinov3_store_enabled = bool(store_dinov3_embedding or dinov3_enabled)
    if selected_occlusion_source not in {"visible_mask", "vlm"}:
        raise ValueError(f"Unsupported occlusion_source: {occlusion_source!r}")

    output_root = Path(output_dir)
    images_dir = output_root / "images"
    cache_dir = output_root / "vlm_cache"
    export_object_crops_dir = None
    if export_object_crops_by_global_id_dir:
        raw_export_dir = Path(export_object_crops_by_global_id_dir)
        export_object_crops_dir = (
            raw_export_dir if raw_export_dir.is_absolute() else (output_root / raw_export_dir)
        )
    object_cache_root = (
        Path(object_cache_dir)
        if object_cache_dir is not None
        else output_root / str(OBJECT_CACHE_DIR)
    )

    output_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if object_use_cache:
        object_cache_root.mkdir(parents=True, exist_ok=True)

    report: Dict = {
        "started_at": _now_iso(),
        "scene_path": scene_path,
        "meters_per_step": meters_per_step,
        "max_positions": max_positions,
        "output_dir": str(output_root),
        "vlm_model": vlm_model,
        "use_cache": use_cache,
        "builder_variant": selected_builder_variant,
        "object_prompt_variant": selected_prompt_variant,
        "angle_split_enabled": bool(angle_split_active),
        "angle_step_deg": int(angle_step),
        "object_orientation_mode": str(object_orientation_mode),
        "object_config": {
            "max_per_frame": int(object_max_per_frame),
            "bbox_conf_threshold": float(bbox_conf_threshold),
            "stored_text_modes": ["short", "long"],
            "stored_visual_modes": ["dinov3"] if bool(dinov3_store_enabled) else [],
            "parse_retries": int(object_parse_retries),
            "use_cache": bool(object_use_cache),
            "cache_dir": str(object_cache_root),
            "r_threshold": None if r_threshold is None else float(r_threshold),
            "r_threshold_enabled": bool(r_threshold is not None),
            "export_object_crops_by_global_id_enabled": bool(export_object_crops_dir),
            "export_object_crops_by_global_id_dir": (
                str(export_object_crops_dir)
                if export_object_crops_dir is not None
                else str(output_root / _OBJECT_CROPS_BY_GLOBAL_ID_DIRNAME)
            ),
            "occlusion_reweight": {
                "formula_version": OCCLUSION_SCORE_FORMULA_VERSION,
                "w1": float(occlusion_reweight_w1),
                "w2": float(occlusion_reweight_w2),
                "b": float(occlusion_reweight_b),
                "eps": float(OCCLUSION_REWEIGHT_EPS),
            },
            "occlusion_source": str(selected_occlusion_source),
            "dinov3": {
                "enabled": bool(dinov3_enabled),
                "store_embedding": bool(dinov3_store_enabled),
                "model_name": str(dinov3_model_name),
                "batch_size": int(dinov3_batch_size),
                "normalized": bool(dinov3_normalize),
            },
        },
        "geometry_config": {
            "pipeline_enabled": bool(OBJECT_GEOMETRY_PIPELINE_ENABLE),
            "horizontal_fov_deg": float(FOV),
            "image_width_px": int(IMAGE_WIDTH),
            "image_height_px": int(IMAGE_HEIGHT),
            "taxonomy_path": str(OBJECT_PRELIST_TAXONOMY_PATH),
            "save_artifacts": bool(SAVE_GEOMETRY_ARTIFACTS),
            "nanosam_encoder_path": str(NANOSAM_ENCODER_PATH),
            "nanosam_decoder_path": str(NANOSAM_DECODER_PATH),
            "depth_pro_model_path": str(DEPTH_PRO_MODEL_PATH),
            "occlusion_target_overlap_threshold": float(occlusion_target_overlap_threshold),
            "visible_occ_boundary_width": int(visible_occ_boundary_width),
            "visible_occ_ring_radius": int(visible_occ_ring_radius),
            "visible_occ_depth_margin_delta": float(visible_occ_depth_margin_delta),
            "visible_occ_boundary_neighbor_radius": int(visible_occ_boundary_neighbor_radius),
        },
        "tour_mode": tour_mode,
        "random_config": {
            "num_steps": int(random_num_steps),
            "step_size": float(random_step_size),
            "scan_angles": [int(a) for a in normalized_scan_angles],
            "seed": None if random_seed is None else int(random_seed),
            "max_attempts_per_step": int(random_max_attempts_per_step),
            "include_start_scan": bool(random_include_start_scan),
        },
        "scan_angles": [int(a) for a in normalized_scan_angles],
        "execution_mode": str(execution_mode),
        "legacy_per_frame": bool(legacy_per_frame),
        "vlm_max_in_flight": int(max(1, int(vlm_max_in_flight))),
        "total_frames_raw": 0,
        "total_frames_processed": 0,
        "total_entries": 0,
        "failed_entries": 0,
        "image_index_ntotal": 0,
        "text_index_ntotal_short": 0,
        "text_index_ntotal_long": 0,
        "object_index_ntotal_short": 0,
        "object_index_ntotal_long": 0,
        "object_dinov3_ntotal": 0,
        "total_objects": 0,
        "avg_objects_per_frame": 0.0,
        "parse_ok_count": 0,
        "parse_fallback_count": 0,
        "parse_failed_count": 0,
        "geometry_ok_count": 0,
        "geometry_fallback_count": 0,
        "total_left_bucket_objects": 0,
        "total_center_bucket_objects": 0,
        "total_right_bucket_objects": 0,
        "failure_examples": [],
        "overview_outputs": {},
        "resumed_entry_count": 0,
        "regenerated_length_entry_count": 0,
        "generated_entry_count": 0,
        "capture_phase_total_sec": 0.0,
        "selector_batch_total_sec": 0.0,
        "object_description_batch_total_sec": 0.0,
        "fallback_batch_total_sec": 0.0,
        "r_threshold": None if r_threshold is None else float(r_threshold),
        "r_threshold_enabled": bool(r_threshold is not None),
        "geometry_objects_before_r_threshold": 0,
        "geometry_objects_after_r_threshold": 0,
        "geometry_objects_filtered_by_r_threshold": 0,
        "frames_all_geometry_objects_filtered": 0,
        "object_r_scores_pre_threshold_csv_path": "",
        "object_r_scores_csv_path": "",
        "object_r_scores_pre_threshold_count": 0,
        "object_r_scores_count": 0,
        "object_crops_by_global_id": {
            "enabled": bool(export_object_crops_dir),
            "dir": "",
            "manifest_path": "",
            "manifest_count": 0,
            "exported_count": 0,
            "copied_count": 0,
            "regenerated_count": 0,
            "skipped_count": 0,
        },
        "polar_surrounding_postprocess": {
            "enabled": bool(run_polar_surrounding_postprocess),
            "ran": False,
            "ok": False,
        },
    }

    try:
        embedder = Embedder()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize CLIP embedder. "
            "See the nested error for network/cache checkpoint details."
        ) from exc
    dino_embedder = None
    if dinov3_enabled:
        try:
            dino_embedder = DINOv3Embedder(
                model_name=str(dinov3_model_name),
                batch_size=int(dinov3_batch_size),
                normalize=bool(dinov3_normalize),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize DINOv3 embedder. "
                "Check transformers/PyTorch availability and model checkpoint access."
            ) from exc
    captioner = VLMCaptioner(
        model_name=vlm_model,
        use_cache=use_cache,
        cache_dir=str(cache_dir),
        object_use_cache=object_use_cache,
        object_cache_dir=str(object_cache_root),
    )
    geometry_pipeline = (
        ObjectGeometryPipeline(
            captioner=captioner,
            output_root=str(output_root),
            horizontal_fov_deg=float(FOV),
            image_width_px=int(IMAGE_WIDTH),
            image_height_px=int(IMAGE_HEIGHT),
            save_artifacts=bool(SAVE_GEOMETRY_ARTIFACTS),
            bbox_conf_threshold=float(bbox_conf_threshold),
            occlusion_reweight_w1=float(occlusion_reweight_w1),
            occlusion_reweight_w2=float(occlusion_reweight_w2),
            occlusion_reweight_b=float(occlusion_reweight_b),
            occlusion_reweight_eps=float(OCCLUSION_REWEIGHT_EPS),
            occlusion_source=str(selected_occlusion_source),
            occlusion_target_overlap_threshold=float(occlusion_target_overlap_threshold),
            visible_occ_boundary_width=int(visible_occ_boundary_width),
            visible_occ_ring_radius=int(visible_occ_ring_radius),
            visible_occ_depth_margin_delta=float(visible_occ_depth_margin_delta),
            visible_occ_boundary_neighbor_radius=int(visible_occ_boundary_neighbor_radius),
            dino_embedder=dino_embedder,
            enable_dinov3_embedding=bool(dinov3_enabled),
        )
        if bool(OBJECT_GEOMETRY_PIPELINE_ENABLE)
        else None
    )

    try:
        explorer = Explorer(scene_path=scene_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Habitat simulator. "
            "Check OpenGL/windowless context availability and scene path validity."
        ) from exc

    emb_dim = int(getattr(embedder.model.visual, "output_dim", 512))
    resume_state = _load_resume_state(output_root=output_root, emb_dim=emb_dim)
    metadata_records: List[Dict] = list(resume_state["metadata_records"])
    image_embs: List[np.ndarray] = list(resume_state["image_embs"])
    text_embs_short: List[np.ndarray] = list(resume_state["text_embs_short"])
    text_embs_long: List[np.ndarray] = list(resume_state["text_embs_long"])
    raw_api_records: List[Dict] = list(resume_state["raw_api_records"])
    object_groups_by_entry_id: Dict[int, List[ObjectGroupItem]] = {
        int(entry_id): list(groups)
        for entry_id, groups in dict(resume_state["object_groups_by_entry_id"]).items()
    }
    file_name_to_entry_id: Dict[str, int] = dict(resume_state["file_name_to_entry_id"])
    failures: List[Dict] = []
    timing_records: List[Dict] = []
    pre_threshold_r_score_rows_by_entry_id: Dict[int, List[Dict[str, Any]]] = _load_object_r_scores_pre_threshold_rows(
        output_root / "object_r_scores_pre_threshold.csv"
    )

    try:
        if tour_mode == "full_house":
            frames, poses = explorer.explore_full_house(
                meters_per_step=meters_per_step,
                scan_angles=normalized_scan_angles,
            )
        elif tour_mode == "random":
            frames, poses = explorer.explore_custom_tour(
                num_steps=int(random_num_steps),
                step_size=float(random_step_size),
                scan_angles=normalized_scan_angles,
                seed=random_seed,
                max_attempts_per_step=int(random_max_attempts_per_step),
                include_start_scan=bool(random_include_start_scan),
            )
        else:
            raise ValueError(f"Unsupported tour_mode: {tour_mode}")

        report["total_frames_raw"] = len(frames)
        num_angles_per_position = max(1, len(normalized_scan_angles))

        if max_positions is not None and max_positions >= 0:
            max_frames = int(max_positions) * num_angles_per_position
            frames = frames[:max_frames]
            poses = poses[:max_frames]

        report["total_frames_processed"] = len(frames)
        normalized_execution_mode = str(execution_mode or "capture_then_parallel_vlm").strip().lower()
        legacy_mode = bool(legacy_per_frame) or normalized_execution_mode == "legacy_per_frame"
        report["execution_mode"] = "legacy_per_frame" if legacy_mode else normalized_execution_mode
        valid_orientation_set = set(normalized_scan_angles)

        if not legacy_mode:
            capture_t0 = time.perf_counter()
            frame_jobs: List[Dict[str, Any]] = []
            next_new_entry_id = int(len(metadata_records))
            for frame_idx, (rgb_image, pose) in enumerate(zip(frames, poses)):
                capture_job_t0 = time.perf_counter()
                pos = np.asarray(pose["position"], dtype=np.float32).reshape(-1)
                if pos.shape[0] != 3:
                    raise ValueError(f"Position must be length 3, got {pos.tolist()}")

                world_position = [float(pos[0]), float(pos[1]), float(pos[2])]
                x = float(world_position[0])
                y = float(world_position[1])
                z = float(world_position[2])
                orientation = _rotation_to_orientation_deg(pose.get("rotation"))
                orientation = _nearest_scan_angle(orientation, normalized_scan_angles)
                if orientation not in valid_orientation_set:
                    raise ValueError(
                        f"Invalid orientation value: {orientation}; expected one of {list(normalized_scan_angles)}"
                    )
                position_id = frame_idx // num_angles_per_position
                file_name = f"images/pose_{position_id:05d}_o{orientation:03d}_{frame_idx:06d}.jpg"
                image_path = output_root / file_name
                existing_entry_id = file_name_to_entry_id.get(file_name)
                existing_meta = (
                    metadata_records[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(metadata_records)
                    else None
                )
                existing_raw_api = (
                    raw_api_records[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(raw_api_records)
                    else None
                )
                existing_image_emb = (
                    image_embs[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(image_embs)
                    else None
                )
                existing_text_emb_short = (
                    text_embs_short[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(text_embs_short)
                    else None
                )
                existing_text_emb_long = (
                    text_embs_long[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(text_embs_long)
                    else None
                )
                existing_object_group = (
                    object_groups_by_entry_id.get(existing_entry_id)
                    if existing_entry_id is not None
                    else None
                )
                reusable = _should_reuse_existing_entry(
                    existing_meta=existing_meta,
                    existing_raw_api=existing_raw_api,
                    existing_image_emb=existing_image_emb,
                    existing_text_emb_short=existing_text_emb_short,
                    existing_text_emb_long=existing_text_emb_long,
                    existing_object_group=existing_object_group,
                    expected_file_name=file_name,
                    require_geometry_fields=bool(OBJECT_GEOMETRY_PIPELINE_ENABLE),
                )
                planned_entry_id = int(existing_entry_id) if existing_entry_id is not None else int(next_new_entry_id)
                if existing_entry_id is None:
                    next_new_entry_id += 1

                image_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(image_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f"Failed to save image to {image_path}")

                frame_jobs.append(
                    {
                        "frame_idx": int(frame_idx),
                        "rgb_image": rgb_image,
                        "pose": pose,
                        "world_position": list(world_position),
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "orientation": int(orientation),
                        "position_id": int(position_id),
                        "file_name": file_name,
                        "image_path": str(image_path),
                        "camera_context": {
                            "camera_x": float(x),
                            "camera_y": float(y),
                            "camera_orientation_deg": float(orientation),
                        },
                        "capture_sec": float(time.perf_counter() - capture_job_t0),
                        "planned_entry_id": int(planned_entry_id),
                        "existing_entry_id": None if existing_entry_id is None else int(existing_entry_id),
                        "reused": bool(reusable),
                        "existing_meta": existing_meta,
                        "existing_raw_api": existing_raw_api,
                    }
                )
            report["capture_phase_total_sec"] = float(time.perf_counter() - capture_t0)

            for job in frame_jobs:
                if not job["reused"]:
                    continue
                report["resumed_entry_count"] += 1
                timing_records.append(
                    {
                        "frame_idx": int(job["frame_idx"]),
                        "entry_id": int(job["planned_entry_id"]),
                        "file_name": str(job["file_name"]),
                        "route": "resumed",
                        "resumed": True,
                        "frame_total_sec": float(job["capture_sec"]),
                    }
                )
                _builder_log(
                    "frame_resume "
                    f"frame_idx={int(job['frame_idx'])} "
                    f"entry_id={int(job['planned_entry_id'])} "
                    f"file={job['file_name']}"
                )

            active_jobs = [job for job in frame_jobs if not job["reused"]]
            parallel_captioner_getter = _make_thread_local_captioner_getter(
                model_name=vlm_model,
                use_cache=use_cache,
                cache_dir=str(cache_dir),
                object_use_cache=object_use_cache,
                object_cache_dir=str(object_cache_root),
            )

            selector_results_by_frame_idx: Dict[int, Dict[str, Any]] = {}
            if geometry_pipeline is not None and active_jobs:
                selector_results_by_frame_idx = _run_parallel_vlm_stage(
                    jobs=active_jobs,
                    max_in_flight=int(vlm_max_in_flight),
                    captioner_getter=parallel_captioner_getter,
                    stage_name="selector",
                    worker=lambda job, stage_captioner: stage_captioner.select_object_types_with_meta(
                        image_path=str(job["image_path"]),
                        camera_context=dict(job["camera_context"]),
                    ),
                )
                report["selector_batch_total_sec"] = float(
                    sum(float(item.get("elapsed_sec") or 0.0) for item in selector_results_by_frame_idx.values())
                )
                for job in active_jobs:
                    selector_stage = dict(selector_results_by_frame_idx.get(int(job["frame_idx"])) or {})
                    job["selector_result"] = selector_stage.get("result")
                    job["selector_sec"] = float(selector_stage.get("elapsed_sec") or 0.0)
            else:
                for job in active_jobs:
                    job["selector_result"] = None
                    job["selector_sec"] = 0.0

            description_jobs: List[Dict[str, Any]] = []
            fallback_jobs: List[Dict[str, Any]] = []
            for job in active_jobs:
                geometry_result = None
                if geometry_pipeline is not None:
                    geometry_result = geometry_pipeline.run_for_view(
                        entry_id=int(job["planned_entry_id"]),
                        image_path=str(job["image_path"]),
                        image_rgb=job["rgb_image"],
                        camera_x=float(job["x"]),
                        camera_y=float(job["y"]),
                        camera_z=float(job["z"]),
                        camera_orientation_deg=float(job["orientation"]),
                        max_objects=int(object_max_per_frame),
                        selector_result_override=job.get("selector_result"),
                        defer_object_descriptions=True,
                    )
                    geometry_timings = dict(geometry_result.timings or {})
                    geometry_timings["selector_sec"] = float(job.get("selector_sec") or 0.0)
                    geometry_timings["total_sec"] = float(geometry_timings.get("total_sec") or 0.0) + float(
                        job.get("selector_sec") or 0.0
                    )
                    geometry_result.timings = geometry_timings
                    if geometry_result.ok:
                        report["geometry_ok_count"] = int(report.get("geometry_ok_count", 0)) + 1
                        description_jobs.append(job)
                        _builder_log(
                            "geometry_ok "
                            f"frame_idx={int(job['frame_idx'])} "
                            f"file={job['file_name']} "
                            f"selected_types={len(list(geometry_result.selector_payload.get('selected_object_types') or []))} "
                            f"objects={len(list(geometry_result.object_rows or []))} "
                            f"selector_source={str(geometry_result.selector_source or 'unknown')} "
                            f"selector_sec={float(job.get('selector_sec') or 0.0):.2f} "
                            f"depth_sec={float(geometry_timings.get('depth_sec') or 0.0):.2f} "
                            f"angle_sec={float(geometry_timings.get('angle_geometry_total_sec') or 0.0):.2f}"
                        )
                    else:
                        report["geometry_fallback_count"] = int(report.get("geometry_fallback_count", 0)) + 1
                        fallback_jobs.append(job)
                        _builder_log(
                            "geometry_fallback "
                            f"frame_idx={int(job['frame_idx'])} "
                            f"file={job['file_name']} "
                            f"reason={geometry_result.failure_reason or 'unknown'} "
                            f"selector_sec={float(job.get('selector_sec') or 0.0):.2f} "
                            f"detector_sec={float(geometry_timings.get('detector_sec') or 0.0):.2f} "
                            f"depth_sec={float(geometry_timings.get('depth_sec') or 0.0):.2f}"
                        )
                else:
                    fallback_jobs.append(job)
                job["geometry_result"] = geometry_result

            if description_jobs:
                description_results_by_frame_idx = _run_parallel_vlm_stage(
                    jobs=description_jobs,
                    max_in_flight=int(vlm_max_in_flight),
                    captioner_getter=parallel_captioner_getter,
                    stage_name="object_description",
                    worker=lambda job, stage_captioner: stage_captioner.describe_detected_objects_with_meta(
                        image_path=str(job["image_path"]),
                        detections=list(getattr(job["geometry_result"], "description_requests", []) or []),
                    ),
                )
                report["object_description_batch_total_sec"] = float(
                    sum(float(item.get("elapsed_sec") or 0.0) for item in description_results_by_frame_idx.values())
                )
                for job in description_jobs:
                    description_stage = dict(description_results_by_frame_idx.get(int(job["frame_idx"])) or {})
                    job["geometry_result"] = _apply_batched_description_result_to_geometry(
                        job["geometry_result"],
                        description_result=description_stage.get("result"),
                        description_total_sec=float(description_stage.get("elapsed_sec") or 0.0),
                    )

            if fallback_jobs:
                fallback_results_by_frame_idx = _run_parallel_vlm_stage(
                    jobs=fallback_jobs,
                    max_in_flight=int(vlm_max_in_flight),
                    captioner_getter=parallel_captioner_getter,
                    stage_name="fallback",
                    worker=lambda job, stage_captioner: _parse_objects_with_retry(
                        captioner=stage_captioner,
                        image_path=str(job["image_path"]),
                        image_id=str(job["file_name"]),
                        max_objects=int(object_max_per_frame),
                        retries=int(object_parse_retries),
                        prompt_variant=selected_prompt_variant,
                        camera_context=dict(job["camera_context"]),
                    ),
                )
                report["fallback_batch_total_sec"] = float(
                    sum(float(item.get("elapsed_sec") or 0.0) for item in fallback_results_by_frame_idx.values())
                )
                for job in fallback_jobs:
                    fallback_stage = dict(fallback_results_by_frame_idx.get(int(job["frame_idx"])) or {})
                    job["parse_result"] = fallback_stage.get("result")
                    job["fallback_parse_sec"] = float(fallback_stage.get("elapsed_sec") or 0.0)
            else:
                fallback_results_by_frame_idx = {}

            for job in frame_jobs:
                if job["reused"]:
                    continue
                frame_idx = int(job["frame_idx"])
                rgb_image = job["rgb_image"]
                world_position = list(job["world_position"])
                x = float(job["x"])
                y = float(job["y"])
                z = float(job["z"])
                orientation = int(job["orientation"])
                file_name = str(job["file_name"])
                image_path = Path(str(job["image_path"]))
                entry_id = int(job["planned_entry_id"])
                geometry_result = job.get("geometry_result")
                geometry_object_rows: List[Dict[str, Any]] = []
                geometry_timing: Dict[str, Any] = {}
                geometry_objects_before_r_threshold = 0
                geometry_objects_after_r_threshold = 0
                geometry_objects_filtered_by_r_threshold = 0
                geometry_all_objects_filtered_by_r_threshold = False
                fallback_parse_sec = float(job.get("fallback_parse_sec") or 0.0)
                fallback_angle_geometry_sec = 0.0
                view_embedding_sec = 0.0
                object_embedding_total_sec = 0.0
                parse_warnings: List[str] = []
                scene_objects = None
                object_text_pairs: List[Tuple] = []
                raw_vlm_output = ""
                raw_api_source = "missing"
                raw_api_response = None
                view_attribute = _build_view_attribute(scene_objects=None)
                parse_status = "fallback"
                object_texts_short = [UNKNOWN_TEXT_TOKEN]
                object_texts_long = [UNKNOWN_TEXT_TOKEN]
                frame_text_short = UNKNOWN_TEXT_TOKEN
                frame_text_long = UNKNOWN_TEXT_TOKEN

                if geometry_result is not None and getattr(geometry_result, "ok", False):
                    geometry_object_rows = list(geometry_result.object_rows or [])
                    geometry_timing = dict(geometry_result.timings or {})
                    view_attribute = _view_attribute_from_selector_payload(geometry_result.selector_payload)
                    raw_vlm_output = str(geometry_result.selector_raw_json or "")
                    raw_api_source = str(geometry_result.selector_source or "")
                    raw_api_response = geometry_result.selector_raw_api_response
                    _store_object_r_scores_pre_threshold_rows(
                        pre_threshold_r_score_rows_by_entry_id,
                        entry_id=int(entry_id),
                        rows=[
                            _build_object_r_scores_pre_threshold_row(
                                row,
                                entry_id=int(entry_id),
                                frame_id=int(frame_idx),
                                file_name=file_name,
                                r_threshold=r_threshold,
                            )
                            for row in geometry_object_rows
                        ],
                    )
                    (
                        geometry_object_rows,
                        geometry_objects_before_r_threshold,
                        geometry_objects_after_r_threshold,
                        geometry_objects_filtered_by_r_threshold,
                    ) = _filter_geometry_rows_by_r_threshold(
                        geometry_object_rows,
                        r_threshold=r_threshold,
                    )
                    report["geometry_objects_before_r_threshold"] = int(
                        report.get("geometry_objects_before_r_threshold", 0)
                    ) + int(geometry_objects_before_r_threshold)
                    report["geometry_objects_after_r_threshold"] = int(
                        report.get("geometry_objects_after_r_threshold", 0)
                    ) + int(geometry_objects_after_r_threshold)
                    report["geometry_objects_filtered_by_r_threshold"] = int(
                        report.get("geometry_objects_filtered_by_r_threshold", 0)
                    ) + int(geometry_objects_filtered_by_r_threshold)
                    geometry_all_objects_filtered_by_r_threshold = bool(
                        geometry_objects_before_r_threshold > 0
                        and geometry_objects_after_r_threshold == 0
                        and geometry_objects_filtered_by_r_threshold == geometry_objects_before_r_threshold
                    )
                    if geometry_all_objects_filtered_by_r_threshold:
                        report["frames_all_geometry_objects_filtered"] = int(
                            report.get("frames_all_geometry_objects_filtered", 0)
                        ) + 1
                        object_texts_short = [UNKNOWN_TEXT_TOKEN]
                        object_texts_long = [UNKNOWN_TEXT_TOKEN]
                        frame_text_short = UNKNOWN_TEXT_TOKEN
                        frame_text_long = UNKNOWN_TEXT_TOKEN
                    else:
                        object_texts_short = [
                            str(row.get("object_text_short") or UNKNOWN_TEXT_TOKEN).strip() or UNKNOWN_TEXT_TOKEN
                            for row in geometry_object_rows
                        ] or [UNKNOWN_TEXT_TOKEN]
                        object_texts_long = [
                            _format_object_text_long(
                                str(row.get("object_text_long") or UNKNOWN_TEXT_TOKEN).strip() or UNKNOWN_TEXT_TOKEN,
                                angle_bucket=_normalize_angle_bucket(
                                    row.get("laterality"),
                                    angle_split_enable=angle_split_active,
                                ),
                                builder_variant=selected_builder_variant,
                            )
                            for row in geometry_object_rows
                        ] or [UNKNOWN_TEXT_TOKEN]
                        frame_text_short = " | ".join(object_texts_short) if object_texts_short else UNKNOWN_TEXT_TOKEN
                        frame_text_long = " | ".join(object_texts_long) if object_texts_long else UNKNOWN_TEXT_TOKEN
                    parse_status = "ok"
                else:
                    parse_result = job.get("parse_result")
                    if geometry_result is not None:
                        geometry_timing = dict(geometry_result.timings or {})
                        parse_warnings.append(
                            f"geometry_pipeline_fallback:{geometry_result.failure_reason or 'unknown'}"
                        )
                    if parse_result is not None:
                        parse_status = parse_result.parse_status
                        parse_warnings.extend(list(parse_result.warnings))
                        raw_vlm_output = parse_result.raw_vlm_output
                        raw_api_source = parse_result.raw_api_source
                        raw_api_response = parse_result.raw_api_response
                        scene_objects = parse_result.scene_objects
                    if scene_objects is not None:
                        parse_status = "ok"
                        angle_enrich_t0 = time.perf_counter()
                        view_attribute = _build_view_attribute(scene_objects=scene_objects)
                        _enrich_scene_objects_geometry(
                            scene_objects,
                            camera_x=float(x),
                            camera_y=float(y),
                            camera_z=float(z),
                            camera_orientation_deg=float(orientation),
                            angle_step=int(angle_step),
                        )
                        fallback_angle_geometry_sec = float(time.perf_counter() - angle_enrich_t0)
                        frame_text_short = compose_frame_text(
                            scene_objects,
                            max_objects=int(object_max_per_frame),
                            mode="short",
                        )
                        frame_text_long = compose_frame_text(
                            scene_objects,
                            max_objects=int(object_max_per_frame),
                            mode="long",
                        )
                        objs = sorted_objects(scene_objects, max_objects=int(object_max_per_frame))
                        for obj in objs:
                            obj_text_short = select_object_text(obj, mode="short", scene_objects=scene_objects)
                            raw_long = select_object_text(obj, mode="long", scene_objects=scene_objects)
                            if obj_text_short and raw_long:
                                angle_bucket = _normalize_angle_bucket(
                                    obj.relative_position_laterality,
                                    angle_split_enable=angle_split_active,
                                )
                                obj_text_long = _format_object_text_long(
                                    raw_long,
                                    angle_bucket=angle_bucket,
                                    builder_variant=selected_builder_variant,
                                )
                                object_text_pairs.append((obj, obj_text_short, raw_long, obj_text_long))
                        if object_text_pairs:
                            object_texts_short = [short_text for _, short_text, _, _ in object_text_pairs]
                            object_texts_long = [long_text for _, _, _, long_text in object_text_pairs]

                embed_view_t0 = time.perf_counter()
                image_emb = embedder.embed_image(rgb_image).astype("float32")
                text_emb_short = embedder.embed_text(frame_text_short).astype("float32")
                text_emb_long = embedder.embed_text(frame_text_long).astype("float32")
                view_embedding_sec = float(time.perf_counter() - embed_view_t0)

                if image_emb.ndim != 1 or text_emb_short.ndim != 1 or text_emb_long.ndim != 1:
                    raise ValueError("Embedding must be a 1D vector")
                if (
                    image_emb.shape[0] != emb_dim
                    or text_emb_short.shape[0] != emb_dim
                    or text_emb_long.shape[0] != emb_dim
                ):
                    raise ValueError(
                        f"Embedding dim mismatch: image={image_emb.shape[0]}, "
                        f"text_short={text_emb_short.shape[0]}, text_long={text_emb_long.shape[0]}, "
                        f"expected={emb_dim}"
                    )
                if (
                    not np.isclose(x, world_position[0])
                    or not np.isclose(y, world_position[1])
                    or not np.isclose(z, world_position[2])
                ):
                    raise ValueError("2D/3D coordinate consistency check failed")

                entry_object_count = 0
                metadata_record = {
                    "id": entry_id,
                    "frame_id": int(frame_idx),
                    "x": x,
                    "y": y,
                    "z": z,
                    "world_position": world_position,
                    "orientation": orientation,
                    "file_name": file_name,
                    "text": frame_text_short,
                    "frame_text_short": frame_text_short,
                    "frame_text_long": frame_text_long,
                    "parse_status": parse_status,
                    "parse_warnings": parse_warnings,
                    "raw_vlm_output": raw_vlm_output,
                    "raw_api_source": raw_api_source,
                    "text_input_for_clip_short": frame_text_short,
                    "text_input_for_clip_long": frame_text_long,
                    "object_text_inputs_short": object_texts_short,
                    "object_text_inputs_long": object_texts_long,
                    "builder_variant": selected_builder_variant,
                    "object_prompt_variant": selected_prompt_variant,
                    "attribute": dict(view_attribute),
                }
                raw_api_record = {
                    "entry_id": int(entry_id),
                    "frame_id": int(frame_idx),
                    "file_name": file_name,
                    "raw_api_source": raw_api_source,
                    "raw_api_response": raw_api_response,
                    "object_prompt_variant": selected_prompt_variant,
                    "geometry_pipeline_used": bool(geometry_result is not None and getattr(geometry_result, "ok", False)),
                    "geometry_fallback_reason": None
                    if geometry_result is None or geometry_result.ok
                    else geometry_result.failure_reason,
                    "selected_object_types": []
                    if geometry_result is None
                    else list(geometry_result.selector_payload.get("selected_object_types") or []),
                    "geometry_artifacts": {}
                    if geometry_result is None
                    else {
                        "detections_path": geometry_result.artifacts.detections_path,
                        "detection_overlay_path": geometry_result.artifacts.detection_overlay_path,
                        "filtered_detections_path": geometry_result.artifacts.filtered_detections_path,
                        "filtered_detection_overlay_path": geometry_result.artifacts.filtered_detection_overlay_path,
                        "depth_map_path": geometry_result.artifacts.depth_map_path,
                        "depth_preview_path": geometry_result.artifacts.depth_preview_path,
                    },
                    "timing": {
                        "frame_total_sec": 0.0,
                        "geometry_pipeline_total_sec": float(geometry_timing.get("total_sec") or 0.0),
                        "selector_sec": float(geometry_timing.get("selector_sec") or 0.0),
                        "dependency_setup_sec": float(geometry_timing.get("dependency_setup_sec") or 0.0),
                        "detector_sec": float(geometry_timing.get("detector_sec") or 0.0),
                        "depth_sec": float(geometry_timing.get("depth_sec") or 0.0),
                        "mask_total_sec": float(geometry_timing.get("mask_total_sec") or 0.0),
                        "angle_geometry_total_sec": float(geometry_timing.get("angle_geometry_total_sec") or 0.0),
                        "crop_vlm_description_total_sec": float(
                            geometry_timing.get("crop_vlm_description_total_sec") or 0.0
                        ),
                        "crop_vlm_description_avg_sec": float(
                            geometry_timing.get("crop_vlm_description_avg_sec") or 0.0
                        ),
                        "object_description_call_count": int(geometry_timing.get("object_description_call_count") or 0),
                        "detection_count_raw": int(geometry_timing.get("detection_count_raw") or 0),
                        "detection_count_class_matched": int(geometry_timing.get("detection_count_class_matched") or 0),
                        "detection_count_filtered_by_bbox_conf": int(
                            geometry_timing.get("detection_count_filtered_by_bbox_conf") or 0
                        ),
                        "detection_count_kept": int(geometry_timing.get("detection_count_kept") or 0),
                        "detection_count_truncated_by_max_objects": int(
                            geometry_timing.get("detection_count_truncated_by_max_objects") or 0
                        ),
                        "geometry_objects_before_r_threshold": int(geometry_objects_before_r_threshold),
                        "geometry_objects_after_r_threshold": int(geometry_objects_after_r_threshold),
                        "geometry_objects_filtered_by_r_threshold": int(geometry_objects_filtered_by_r_threshold),
                        "geometry_all_objects_filtered_by_r_threshold": bool(
                            geometry_all_objects_filtered_by_r_threshold
                        ),
                        "vlm_fallback_object_parse_sec": float(fallback_parse_sec),
                        "fallback_angle_geometry_sec": float(fallback_angle_geometry_sec),
                        "view_embedding_sec": float(view_embedding_sec),
                        "object_embedding_total_sec": float(object_embedding_total_sec),
                    },
                }
                existing_entry_id = job.get("existing_entry_id")
                if existing_entry_id is None:
                    metadata_records.append(metadata_record)
                    raw_api_records.append(raw_api_record)
                    image_embs.append(image_emb)
                    text_embs_short.append(text_emb_short)
                    text_embs_long.append(text_emb_long)
                    report["generated_entry_count"] += 1
                else:
                    if _response_has_length_finish_reason((job.get("existing_raw_api") or {}).get("raw_api_response")):
                        report["regenerated_length_entry_count"] += 1
                    metadata_records[entry_id] = metadata_record
                    raw_api_records[entry_id] = raw_api_record
                    image_embs[entry_id] = image_emb
                    text_embs_short[entry_id] = text_emb_short
                    text_embs_long[entry_id] = text_emb_long
                file_name_to_entry_id[file_name] = int(entry_id)
                entry_object_records: List[ObjectGroupItem] = []

                if geometry_all_objects_filtered_by_r_threshold:
                    entry_object_records = []
                elif geometry_object_rows:
                    for geo_row, line_short, line_long in zip(geometry_object_rows, object_texts_short, object_texts_long):
                        object_embed_t0 = time.perf_counter()
                        obj_emb_short = embedder.embed_text(line_short).astype("float32")
                        obj_emb_long = embedder.embed_text(line_long).astype("float32")
                        object_embedding_total_sec += float(time.perf_counter() - object_embed_t0)
                        record = _make_object_record(
                            object_global_id=0,
                            frame_id=frame_idx,
                            entry_id=entry_id,
                            file_name=file_name,
                            x=x,
                            y=y,
                            z=z,
                            world_position=world_position,
                            orientation=orientation,
                            parse_status=parse_status,
                            builder_variant=selected_builder_variant,
                            angle_split_enable=angle_split_active,
                            angle_step=angle_step,
                            object_local_id=str(geo_row.get("object_local_id") or "det_000"),
                            label=str(geo_row.get("label") or "unknown"),
                            object_confidence=float(geo_row.get("object_confidence") or 0.0),
                            description=line_short,
                            long_form_open_description=str(
                                geo_row.get("long_form_open_description") or geo_row.get("object_text_long") or line_short
                            ),
                            attributes=list(geo_row.get("attributes") or []),
                            laterality=str(geo_row.get("laterality") or "center"),
                            distance_bin=str(geo_row.get("distance_bin") or "middle"),
                            verticality=str(geo_row.get("verticality") or "middle"),
                            distance_from_camera_m=geo_row.get("distance_from_camera_m"),
                            relative_height_from_camera_m=geo_row.get("relative_height_from_camera_m"),
                            relative_bearing_deg=geo_row.get("relative_bearing_deg"),
                            estimated_global_x=geo_row.get("estimated_global_x"),
                            estimated_global_y=geo_row.get("estimated_global_y"),
                            estimated_global_z=geo_row.get("estimated_global_z"),
                            any_text=str(geo_row.get("any_text") or ""),
                            location_relative_to_other_objects=str(
                                geo_row.get("location_relative_to_other_objects") or ""
                            ),
                            surrounding_context=list(geo_row.get("surrounding_context") or []),
                            scene_attributes=list(view_attribute.get("scene_attributes") or []),
                            object_text_short=line_short,
                            object_text_long=line_long,
                            precise_orientation_from_bearing=True,
                            geometry_source=str(geo_row.get("geometry_source") or "mask_depth"),
                            geometry_fallback_reason=geo_row.get("geometry_fallback_reason"),
                            detector_label=geo_row.get("detector_label_raw") or geo_row.get("detector_label"),
                            detector_label_raw=geo_row.get("detector_label_raw"),
                            detector_confidence=geo_row.get("detector_confidence"),
                            vlm_label=geo_row.get("vlm_label"),
                            final_label=geo_row.get("final_label"),
                            label_source=geo_row.get("label_source"),
                            label_conflict=geo_row.get("label_conflict"),
                            occlusion_source=geo_row.get("occlusion_source"),
                            occlusion_level=geo_row.get("occlusion_level"),
                            occlusion_penalty_p_o=geo_row.get("occlusion_penalty_p_o"),
                            reweighted_detection_score_r=geo_row.get("reweighted_detection_score_r"),
                            visible_occlusion_ratio=geo_row.get("visible_occlusion_ratio"),
                            occluded_boundary_ratio=geo_row.get("occluded_boundary_ratio"),
                            nearer_ring_overlap_ratio=geo_row.get("nearer_ring_overlap_ratio"),
                            object_depth_median=geo_row.get("object_depth_median"),
                            boundary_pixel_count=geo_row.get("boundary_pixel_count"),
                            occluded_boundary_pixel_count=geo_row.get("occluded_boundary_pixel_count"),
                            ring_pixel_count=geo_row.get("ring_pixel_count"),
                            nearer_ring_pixel_count=geo_row.get("nearer_ring_pixel_count"),
                            depth_margin_delta=geo_row.get("depth_margin_delta"),
                            occluding_overlap_pixel_count=geo_row.get("occluding_overlap_pixel_count"),
                            foreground_occluder_count=geo_row.get("foreground_occluder_count"),
                            occlusion_target_overlap_threshold=geo_row.get("occlusion_target_overlap_threshold"),
                            bbox_xywh_norm=geo_row.get("bbox_xywh_norm"),
                            bbox_xyxy=geo_row.get("bbox_xyxy"),
                            mask_area_px=geo_row.get("mask_area_px"),
                            mask_area_ratio=geo_row.get("mask_area_ratio"),
                            mask_centroid_x_px=geo_row.get("mask_centroid_x_px"),
                            mask_centroid_y_px=geo_row.get("mask_centroid_y_px"),
                            mask_centroid_x_norm=geo_row.get("mask_centroid_x_norm"),
                            mask_centroid_y_norm=geo_row.get("mask_centroid_y_norm"),
                            depth_stat_median_m=geo_row.get("depth_stat_median_m"),
                            depth_stat_p10_m=geo_row.get("depth_stat_p10_m"),
                            depth_stat_p90_m=geo_row.get("depth_stat_p90_m"),
                            projected_planar_distance_m=geo_row.get("projected_planar_distance_m"),
                            vertical_angle_deg=geo_row.get("vertical_angle_deg"),
                            vlm_distance_from_camera_m=geo_row.get("vlm_distance_from_camera_m"),
                            vlm_relative_bearing_deg=geo_row.get("vlm_relative_bearing_deg"),
                            crop_path=geo_row.get("crop_path"),
                            mask_path=geo_row.get("mask_path"),
                            mask_overlay_path=geo_row.get("mask_overlay_path"),
                            depth_map_path=geo_row.get("depth_map_path"),
                            crop_vlm_label=geo_row.get("crop_vlm_label"),
                            dinov3_model_name=geo_row.get("dinov3_model_name"),
                            dinov3_embedding_dim=geo_row.get("dinov3_embedding_dim"),
                            dinov3_input_type=geo_row.get("dinov3_input_type"),
                            dinov3_normalized=geo_row.get("dinov3_normalized"),
                            dinov3_status=geo_row.get("dinov3_status"),
                            dinov3_failure_reason=geo_row.get("dinov3_failure_reason"),
                        )
                        record["view_type"] = str(view_attribute.get("view_type") or "unknown")
                        record["room_function"] = str(view_attribute.get("room_function") or "unknown")
                        record["style_hint"] = str(view_attribute.get("style_hint") or "unknown")
                        record["clutter_level"] = str(view_attribute.get("clutter_level") or "unknown")
                        dino_embedding = geo_row.get("dinov3_embedding")
                        entry_object_records.append(
                            (
                                record,
                                obj_emb_short,
                                obj_emb_long,
                                None
                                if dino_embedding is None
                                else np.asarray(dino_embedding, dtype=np.float32).reshape(-1),
                            )
                        )
                        bucket_key = f"total_{record['angle_bucket']}_bucket_objects"
                        report[bucket_key] = int(report.get(bucket_key, 0)) + 1
                        entry_object_count += 1
                elif scene_objects is not None and object_text_pairs:
                    for obj, line_short, raw_long, line_long in object_text_pairs:
                        fallback_occlusion_level, fallback_occlusion_penalty, fallback_reweighted_detection_score = (
                            _compute_object_reweight_fields(
                                detector_confidence=None,
                                object_confidence=1.0,
                                occlusion_level="uncertain",
                                occlusion_reweight_w1=float(occlusion_reweight_w1),
                                occlusion_reweight_w2=float(occlusion_reweight_w2),
                                occlusion_reweight_b=float(occlusion_reweight_b),
                                occlusion_reweight_eps=float(OCCLUSION_REWEIGHT_EPS),
                            )
                        )
                        object_embed_t0 = time.perf_counter()
                        obj_emb_short = embedder.embed_text(line_short).astype("float32")
                        obj_emb_long = embedder.embed_text(line_long).astype("float32")
                        object_embedding_total_sec += float(time.perf_counter() - object_embed_t0)
                        record = _make_object_record(
                            object_global_id=0,
                            frame_id=frame_idx,
                            entry_id=entry_id,
                            file_name=file_name,
                            x=x,
                            y=y,
                            z=z,
                            world_position=world_position,
                            orientation=orientation,
                            parse_status=parse_status,
                            builder_variant=selected_builder_variant,
                            angle_split_enable=angle_split_active,
                            angle_step=angle_step,
                            scene_objects=scene_objects,
                            obj=obj,
                            object_local_id=obj.feature_id,
                            label=obj.type,
                            object_confidence=1.0,
                            description=line_short,
                            long_form_open_description=raw_long,
                            attributes=list(obj.attributes or []),
                            laterality=obj.relative_position_laterality,
                            distance_bin=obj.relative_position_distance,
                            verticality=obj.relative_position_verticality,
                            distance_from_camera_m=obj.distance_from_camera_m,
                            relative_height_from_camera_m=getattr(obj, "relative_height_from_camera_m", None),
                            relative_bearing_deg=obj.relative_bearing_deg,
                            estimated_global_x=obj.estimated_global_x,
                            estimated_global_y=getattr(obj, "estimated_global_y", None),
                            estimated_global_z=obj.estimated_global_z,
                            any_text=obj.any_text,
                            location_relative_to_other_objects=obj.location_relative_to_other_objects,
                            surrounding_context=_serialize_surrounding_context(obj.surrounding_context),
                            scene_attributes=list(scene_objects.scene_attributes or []),
                            object_text_short=line_short,
                            object_text_long=line_long,
                            precise_orientation_from_bearing=True,
                            geometry_source="vlm_fallback",
                            geometry_fallback_reason=None
                            if geometry_result is None
                            else geometry_result.failure_reason,
                            detector_label=None,
                            detector_label_raw=None,
                            detector_confidence=None,
                            vlm_label=obj.type,
                            final_label=obj.type,
                            label_source="vlm",
                            label_conflict=False,
                            occlusion_source=selected_occlusion_source,
                            occlusion_level=fallback_occlusion_level,
                            occlusion_penalty_p_o=fallback_occlusion_penalty,
                            reweighted_detection_score_r=fallback_reweighted_detection_score,
                            visible_occlusion_ratio=None,
                            occluded_boundary_ratio=None,
                            nearer_ring_overlap_ratio=None,
                            object_depth_median=None,
                            boundary_pixel_count=None,
                            occluded_boundary_pixel_count=None,
                            ring_pixel_count=None,
                            nearer_ring_pixel_count=None,
                            depth_margin_delta=None,
                            vlm_distance_from_camera_m=obj.distance_from_camera_m,
                            vlm_relative_bearing_deg=obj.relative_bearing_deg,
                            dinov3_status="missing" if dinov3_enabled else "disabled",
                        )
                        entry_object_records.append((record, obj_emb_short, obj_emb_long, None))
                        pre_threshold_r_score_rows_by_entry_id.setdefault(int(entry_id), []).append(
                            _build_object_r_scores_pre_threshold_row(
                                record,
                                entry_id=int(entry_id),
                                frame_id=int(frame_idx),
                                file_name=file_name,
                                r_threshold=r_threshold,
                            )
                        )
                        bucket_key = f"total_{record['angle_bucket']}_bucket_objects"
                        report[bucket_key] = int(report.get(bucket_key, 0)) + 1
                        entry_object_count += 1
                else:
                    fallback_occlusion_level, fallback_occlusion_penalty, fallback_reweighted_detection_score = (
                        _compute_object_reweight_fields(
                            detector_confidence=None,
                            object_confidence=0.0,
                            occlusion_level="uncertain",
                            occlusion_reweight_w1=float(occlusion_reweight_w1),
                            occlusion_reweight_w2=float(occlusion_reweight_w2),
                            occlusion_reweight_b=float(occlusion_reweight_b),
                            occlusion_reweight_eps=float(OCCLUSION_REWEIGHT_EPS),
                        )
                    )
                    angle_bucket = _normalize_angle_bucket("center", angle_split_enable=angle_split_active)
                    line_short = UNKNOWN_TEXT_TOKEN
                    line_long = _format_object_text_long(
                        UNKNOWN_TEXT_TOKEN,
                        angle_bucket=angle_bucket,
                        builder_variant=selected_builder_variant,
                    )
                    object_embed_t0 = time.perf_counter()
                    obj_emb_short = embedder.embed_text(line_short).astype("float32")
                    obj_emb_long = embedder.embed_text(line_long).astype("float32")
                    object_embedding_total_sec += float(time.perf_counter() - object_embed_t0)
                    record = _make_object_record(
                        object_global_id=0,
                        frame_id=frame_idx,
                        entry_id=entry_id,
                        file_name=file_name,
                        x=x,
                        y=y,
                        z=z,
                        world_position=world_position,
                        orientation=orientation,
                        parse_status=parse_status,
                        builder_variant=selected_builder_variant,
                        angle_split_enable=angle_split_active,
                        angle_step=angle_step,
                        object_local_id="none_000",
                        label="none",
                        object_confidence=0.0,
                        laterality=angle_bucket,
                        object_text_short=line_short,
                        object_text_long=line_long,
                        geometry_source="vlm_fallback",
                        geometry_fallback_reason=None if geometry_result is None else geometry_result.failure_reason,
                        detector_label_raw=None,
                        vlm_label="unknown",
                        final_label="none",
                        label_source="placeholder",
                        label_conflict=False,
                        occlusion_source=selected_occlusion_source,
                        occlusion_level=fallback_occlusion_level,
                        occlusion_penalty_p_o=fallback_occlusion_penalty,
                        reweighted_detection_score_r=fallback_reweighted_detection_score,
                        visible_occlusion_ratio=None,
                        occluded_boundary_ratio=None,
                        nearer_ring_overlap_ratio=None,
                        object_depth_median=None,
                        boundary_pixel_count=None,
                        occluded_boundary_pixel_count=None,
                        ring_pixel_count=None,
                        nearer_ring_pixel_count=None,
                        depth_margin_delta=None,
                        dinov3_status="missing" if dinov3_enabled else "disabled",
                    )
                    entry_object_records.append((record, obj_emb_short, obj_emb_long, None))
                    pre_threshold_r_score_rows_by_entry_id.setdefault(int(entry_id), []).append(
                        _build_object_r_scores_pre_threshold_row(
                            record,
                            entry_id=int(entry_id),
                            frame_id=int(frame_idx),
                            file_name=file_name,
                            r_threshold=r_threshold,
                        )
                    )
                    report["total_center_bucket_objects"] = int(report.get("total_center_bucket_objects", 0)) + 1
                    entry_object_count += 1
                object_groups_by_entry_id[int(entry_id)] = entry_object_records

                if parse_status == "ok":
                    report["parse_ok_count"] += 1
                elif parse_status == "fallback":
                    report["parse_fallback_count"] += 1
                else:
                    report["parse_failed_count"] += 1

                metadata_records[entry_id]["object_count"] = int(entry_object_count)
                frame_total_sec = float(
                    float(job.get("capture_sec") or 0.0)
                    + float(job.get("selector_sec") or 0.0)
                    + float(geometry_timing.get("dependency_setup_sec") or 0.0)
                    + float(geometry_timing.get("detector_sec") or 0.0)
                    + float(geometry_timing.get("depth_sec") or 0.0)
                    + float(geometry_timing.get("mask_total_sec") or 0.0)
                    + float(geometry_timing.get("angle_geometry_total_sec") or 0.0)
                    + float(geometry_timing.get("crop_vlm_description_total_sec") or 0.0)
                    + float(fallback_parse_sec)
                    + float(fallback_angle_geometry_sec)
                    + float(view_embedding_sec)
                    + float(object_embedding_total_sec)
                )
                raw_api_record["timing"]["view_embedding_sec"] = float(view_embedding_sec)
                raw_api_record["timing"]["object_embedding_total_sec"] = float(object_embedding_total_sec)
                raw_api_record["timing"]["frame_total_sec"] = frame_total_sec
                route_label = _frame_route_label(
                    geometry_object_rows=geometry_object_rows,
                    geometry_all_objects_filtered_by_r_threshold=geometry_all_objects_filtered_by_r_threshold,
                )
                timing_records.append(
                    {
                        "frame_idx": int(frame_idx),
                        "entry_id": int(entry_id),
                        "file_name": file_name,
                        "route": route_label,
                        "parse_status": parse_status,
                        "object_count": int(entry_object_count),
                        "raw_api_source": raw_api_source,
                        "frame_total_sec": frame_total_sec,
                        "geometry_pipeline_total_sec": float(geometry_timing.get("total_sec") or 0.0),
                        "selector_sec": float(geometry_timing.get("selector_sec") or 0.0),
                        "dependency_setup_sec": float(geometry_timing.get("dependency_setup_sec") or 0.0),
                        "detector_sec": float(geometry_timing.get("detector_sec") or 0.0),
                        "depth_sec": float(geometry_timing.get("depth_sec") or 0.0),
                        "mask_total_sec": float(geometry_timing.get("mask_total_sec") or 0.0),
                        "angle_geometry_total_sec": float(geometry_timing.get("angle_geometry_total_sec") or 0.0),
                        "crop_vlm_description_total_sec": float(geometry_timing.get("crop_vlm_description_total_sec") or 0.0),
                        "crop_vlm_description_avg_sec": float(geometry_timing.get("crop_vlm_description_avg_sec") or 0.0),
                        "object_description_call_count": int(geometry_timing.get("object_description_call_count") or 0),
                        "detection_count_raw": int(geometry_timing.get("detection_count_raw") or 0),
                        "detection_count_class_matched": int(geometry_timing.get("detection_count_class_matched") or 0),
                        "detection_count_filtered_by_bbox_conf": int(
                            geometry_timing.get("detection_count_filtered_by_bbox_conf") or 0
                        ),
                        "detection_count_kept": int(geometry_timing.get("detection_count_kept") or 0),
                        "detection_count_truncated_by_max_objects": int(
                            geometry_timing.get("detection_count_truncated_by_max_objects") or 0
                        ),
                        "geometry_objects_before_r_threshold": int(geometry_objects_before_r_threshold),
                        "geometry_objects_after_r_threshold": int(geometry_objects_after_r_threshold),
                        "geometry_objects_filtered_by_r_threshold": int(geometry_objects_filtered_by_r_threshold),
                        "geometry_all_objects_filtered_by_r_threshold": bool(
                            geometry_all_objects_filtered_by_r_threshold
                        ),
                        "vlm_fallback_object_parse_sec": float(fallback_parse_sec),
                        "fallback_angle_geometry_sec": float(fallback_angle_geometry_sec),
                        "view_embedding_sec": float(view_embedding_sec),
                        "object_embedding_total_sec": float(object_embedding_total_sec),
                    }
                )
                _builder_log(
                    "frame_done "
                    f"frame_idx={int(frame_idx)} "
                    f"entry_id={int(entry_id)} "
                    f"file={file_name} "
                    f"route={route_label} "
                    f"parse_status={parse_status} "
                    f"object_count={int(entry_object_count)} "
                    f"raw_api_source={raw_api_source} "
                    f"frame_total_sec={frame_total_sec:.2f} "
                    f"depth_sec={float(geometry_timing.get('depth_sec') or 0.0):.2f} "
                    f"angle_sec={float(geometry_timing.get('angle_geometry_total_sec') or 0.0):.2f} "
                    f"crop_vlm_sec={float(geometry_timing.get('crop_vlm_description_total_sec') or 0.0):.2f} "
                    f"fallback_vlm_sec={fallback_parse_sec:.2f}"
                )

        frame_iter = tqdm(
            zip(frames, poses),
            total=len(frames),
            desc="Building spatial DB",
        ) if legacy_mode else []

        for frame_idx, (rgb_image, pose) in enumerate(frame_iter):
            frame_t0 = time.perf_counter()
            try:
                pos = np.asarray(pose["position"], dtype=np.float32).reshape(-1)
                if pos.shape[0] != 3:
                    raise ValueError(f"Position must be length 3, got {pos.tolist()}")

                world_position = [float(pos[0]), float(pos[1]), float(pos[2])]
                x = float(world_position[0])
                y = float(world_position[1])
                z = float(world_position[2])
                orientation = _rotation_to_orientation_deg(pose.get("rotation"))
                orientation = _nearest_scan_angle(orientation, normalized_scan_angles)
                position_id = frame_idx // num_angles_per_position

                file_name = f"images/pose_{position_id:05d}_o{orientation:03d}_{frame_idx:06d}.jpg"
                image_path = output_root / file_name
                existing_entry_id = file_name_to_entry_id.get(file_name)
                existing_meta = (
                    metadata_records[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(metadata_records)
                    else None
                )
                existing_raw_api = (
                    raw_api_records[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(raw_api_records)
                    else None
                )
                existing_image_emb = (
                    image_embs[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(image_embs)
                    else None
                )
                existing_text_emb_short = (
                    text_embs_short[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(text_embs_short)
                    else None
                )
                existing_text_emb_long = (
                    text_embs_long[existing_entry_id]
                    if existing_entry_id is not None and 0 <= existing_entry_id < len(text_embs_long)
                    else None
                )
                existing_object_group = (
                    object_groups_by_entry_id.get(existing_entry_id)
                    if existing_entry_id is not None
                    else None
                )
                geometry_timing: Dict[str, Any] = {}
                geometry_objects_before_r_threshold = 0
                geometry_objects_after_r_threshold = 0
                geometry_objects_filtered_by_r_threshold = 0
                geometry_all_objects_filtered_by_r_threshold = False
                fallback_parse_sec = 0.0
                fallback_angle_geometry_sec = 0.0
                view_embedding_sec = 0.0
                object_embedding_total_sec = 0.0
                _builder_log(
                    "frame_start "
                    f"frame_idx={int(frame_idx)} "
                    f"position_id={int(position_id)} "
                    f"orientation_deg={int(orientation)} "
                    f"file={file_name} "
                    f"camera_xyz=({x:.3f},{y:.3f},{z:.3f})"
                )
                if _should_reuse_existing_entry(
                    existing_meta=existing_meta,
                    existing_raw_api=existing_raw_api,
                    existing_image_emb=existing_image_emb,
                    existing_text_emb_short=existing_text_emb_short,
                    existing_text_emb_long=existing_text_emb_long,
                    existing_object_group=existing_object_group,
                    expected_file_name=file_name,
                    require_geometry_fields=bool(OBJECT_GEOMETRY_PIPELINE_ENABLE),
                ):
                    report["resumed_entry_count"] += 1
                    _builder_log(
                        "frame_resume "
                        f"frame_idx={int(frame_idx)} "
                        f"entry_id={int(existing_entry_id)} "
                        f"file={file_name}"
                    )
                    timing_records.append(
                        {
                            "frame_idx": int(frame_idx),
                            "entry_id": int(existing_entry_id),
                            "file_name": file_name,
                            "route": "resumed",
                            "resumed": True,
                            "frame_total_sec": float(time.perf_counter() - frame_t0),
                        }
                    )
                    continue

                image_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(image_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f"Failed to save image to {image_path}")

                camera_context = {
                    "camera_x": float(x),
                    "camera_y": float(y),
                    "camera_orientation_deg": float(orientation),
                }
                geometry_result = None
                parse_warnings: List[str] = []
                scene_objects = None
                object_text_pairs: List[Tuple] = []
                geometry_object_rows: List[Dict[str, Any]] = []
                raw_vlm_output = ""
                raw_api_source = "missing"
                raw_api_response = None
                view_attribute = _build_view_attribute(scene_objects=None)

                if geometry_pipeline is not None:
                    _builder_log(
                        "geometry_attempt "
                        f"frame_idx={int(frame_idx)} "
                        f"file={file_name} "
                        f"max_objects={int(object_max_per_frame)}"
                    )
                    geometry_result = geometry_pipeline.run_for_view(
                        entry_id=int(existing_entry_id) if existing_entry_id is not None else int(len(metadata_records)),
                        image_path=str(image_path),
                        image_rgb=rgb_image,
                        camera_x=float(x),
                        camera_y=float(y),
                        camera_z=float(z),
                        camera_orientation_deg=float(orientation),
                        max_objects=int(object_max_per_frame),
                    )
                    if geometry_result.ok:
                        report["geometry_ok_count"] = int(report.get("geometry_ok_count", 0)) + 1
                        geometry_object_rows = list(geometry_result.object_rows)
                        geometry_timing = dict(geometry_result.timings or {})
                        _builder_log(
                            "geometry_ok "
                            f"frame_idx={int(frame_idx)} "
                            f"file={file_name} "
                            f"selected_types={len(list(geometry_result.selector_payload.get('selected_object_types') or []))} "
                            f"objects={len(geometry_object_rows)} "
                            f"selector_source={str(geometry_result.selector_source or 'unknown')} "
                            f"selector_sec={float(geometry_timing.get('selector_sec') or 0.0):.2f} "
                            f"depth_sec={float(geometry_timing.get('depth_sec') or 0.0):.2f} "
                            f"angle_sec={float(geometry_timing.get('angle_geometry_total_sec') or 0.0):.2f} "
                            f"crop_vlm_sec={float(geometry_timing.get('crop_vlm_description_total_sec') or 0.0):.2f}"
                        )
                        view_attribute = _view_attribute_from_selector_payload(geometry_result.selector_payload)
                        raw_vlm_output = str(geometry_result.selector_raw_json or "")
                        raw_api_source = str(geometry_result.selector_source or "")
                        raw_api_response = geometry_result.selector_raw_api_response
                        planned_entry_id = (
                            int(existing_entry_id) if existing_entry_id is not None else int(len(metadata_records))
                        )
                        _store_object_r_scores_pre_threshold_rows(
                            pre_threshold_r_score_rows_by_entry_id,
                            entry_id=planned_entry_id,
                            rows=[
                                _build_object_r_scores_pre_threshold_row(
                                    row,
                                    entry_id=planned_entry_id,
                                    frame_id=int(frame_idx),
                                    file_name=file_name,
                                    r_threshold=r_threshold,
                                )
                                for row in geometry_object_rows
                            ],
                        )
                        (
                            geometry_object_rows,
                            geometry_objects_before_r_threshold,
                            geometry_objects_after_r_threshold,
                            geometry_objects_filtered_by_r_threshold,
                        ) = _filter_geometry_rows_by_r_threshold(
                            geometry_object_rows,
                            r_threshold=r_threshold,
                        )
                        report["geometry_objects_before_r_threshold"] = int(
                            report.get("geometry_objects_before_r_threshold", 0)
                        ) + int(geometry_objects_before_r_threshold)
                        report["geometry_objects_after_r_threshold"] = int(
                            report.get("geometry_objects_after_r_threshold", 0)
                        ) + int(geometry_objects_after_r_threshold)
                        report["geometry_objects_filtered_by_r_threshold"] = int(
                            report.get("geometry_objects_filtered_by_r_threshold", 0)
                        ) + int(geometry_objects_filtered_by_r_threshold)
                        geometry_all_objects_filtered_by_r_threshold = bool(
                            geometry_objects_before_r_threshold > 0
                            and geometry_objects_after_r_threshold == 0
                            and geometry_objects_filtered_by_r_threshold == geometry_objects_before_r_threshold
                        )
                        if geometry_all_objects_filtered_by_r_threshold:
                            report["frames_all_geometry_objects_filtered"] = int(
                                report.get("frames_all_geometry_objects_filtered", 0)
                            ) + 1
                            object_texts_short = [UNKNOWN_TEXT_TOKEN]
                            object_texts_long = [UNKNOWN_TEXT_TOKEN]
                            frame_text_short = UNKNOWN_TEXT_TOKEN
                            frame_text_long = UNKNOWN_TEXT_TOKEN
                        else:
                            object_texts_short = [
                                str(row.get("object_text_short") or UNKNOWN_TEXT_TOKEN).strip() or UNKNOWN_TEXT_TOKEN
                                for row in geometry_object_rows
                            ] or [UNKNOWN_TEXT_TOKEN]
                            object_texts_long = [
                                _format_object_text_long(
                                    str(row.get("object_text_long") or UNKNOWN_TEXT_TOKEN).strip() or UNKNOWN_TEXT_TOKEN,
                                    angle_bucket=_normalize_angle_bucket(
                                        row.get("laterality"),
                                        angle_split_enable=angle_split_active,
                                    ),
                                    builder_variant=selected_builder_variant,
                                )
                                for row in geometry_object_rows
                            ] or [UNKNOWN_TEXT_TOKEN]
                            frame_text_short = " | ".join(object_texts_short) if object_texts_short else UNKNOWN_TEXT_TOKEN
                            frame_text_long = " | ".join(object_texts_long) if object_texts_long else UNKNOWN_TEXT_TOKEN
                        parse_status = "ok"
                    else:
                        report["geometry_fallback_count"] = int(report.get("geometry_fallback_count", 0)) + 1
                        geometry_timing = dict(geometry_result.timings or {})
                        _builder_log(
                            "geometry_fallback "
                            f"frame_idx={int(frame_idx)} "
                            f"file={file_name} "
                            f"reason={geometry_result.failure_reason or 'unknown'} "
                            f"selector_sec={float(geometry_timing.get('selector_sec') or 0.0):.2f} "
                            f"detector_sec={float(geometry_timing.get('detector_sec') or 0.0):.2f} "
                            f"depth_sec={float(geometry_timing.get('depth_sec') or 0.0):.2f}"
                        )
                        parse_warnings.append(
                            f"geometry_pipeline_fallback:{geometry_result.failure_reason or 'unknown'}"
                        )
                        geometry_object_rows = []
                else:
                    _builder_log(
                        "geometry_disabled "
                        f"frame_idx={int(frame_idx)} "
                        f"file={file_name}"
                    )

                if not geometry_object_rows and not geometry_all_objects_filtered_by_r_threshold:
                    _builder_log(
                        "vlm_fallback_start "
                        f"frame_idx={int(frame_idx)} "
                        f"file={file_name} "
                        f"reason={'geometry_disabled' if geometry_result is None and geometry_pipeline is None else (geometry_result.failure_reason if geometry_result is not None else 'geometry_not_used')}"
                    )
                    fallback_t0 = time.perf_counter()
                    parse_result = _parse_objects_with_retry(
                        captioner=captioner,
                        image_path=str(image_path),
                        image_id=file_name,
                        max_objects=int(object_max_per_frame),
                        retries=int(object_parse_retries),
                        prompt_variant=selected_prompt_variant,
                        camera_context=camera_context,
                    )
                    fallback_parse_sec = float(time.perf_counter() - fallback_t0)
                    parse_status = parse_result.parse_status
                    parse_warnings.extend(list(parse_result.warnings))
                    raw_vlm_output = parse_result.raw_vlm_output
                    raw_api_source = parse_result.raw_api_source
                    raw_api_response = parse_result.raw_api_response
                    scene_objects = parse_result.scene_objects
                    if scene_objects is not None:
                        parse_status = "ok"
                        angle_enrich_t0 = time.perf_counter()
                        parsed_count = len(sorted_objects(scene_objects, max_objects=int(object_max_per_frame)))
                        _builder_log(
                            "vlm_fallback_result "
                            f"frame_idx={int(frame_idx)} "
                            f"file={file_name} "
                            f"parse_status={parse_status} "
                            f"objects={int(parsed_count)} "
                            f"raw_api_source={raw_api_source} "
                            f"object_vlm_sec={fallback_parse_sec:.2f}"
                        )
                        view_attribute = _build_view_attribute(scene_objects=scene_objects)
                        _enrich_scene_objects_geometry(
                            scene_objects,
                            camera_x=float(x),
                            camera_y=float(y),
                            camera_z=float(z),
                            camera_orientation_deg=float(orientation),
                            angle_step=int(angle_step),
                        )
                        fallback_angle_geometry_sec = float(time.perf_counter() - angle_enrich_t0)
                        frame_text_short = compose_frame_text(
                            scene_objects,
                            max_objects=int(object_max_per_frame),
                            mode="short",
                        )
                        frame_text_long = compose_frame_text(
                            scene_objects,
                            max_objects=int(object_max_per_frame),
                            mode="long",
                        )
                        objs = sorted_objects(scene_objects, max_objects=int(object_max_per_frame))
                        for obj in objs:
                            obj_text_short = select_object_text(obj, mode="short", scene_objects=scene_objects)
                            raw_long = select_object_text(obj, mode="long", scene_objects=scene_objects)
                            if obj_text_short and raw_long:
                                angle_bucket = _normalize_angle_bucket(
                                    obj.relative_position_laterality,
                                    angle_split_enable=angle_split_active,
                                )
                                obj_text_long = _format_object_text_long(
                                    raw_long,
                                    angle_bucket=angle_bucket,
                                    builder_variant=selected_builder_variant,
                                )
                                object_text_pairs.append((obj, obj_text_short, raw_long, obj_text_long))
                        if object_text_pairs:
                            object_texts_short = [short_text for _, short_text, _, _ in object_text_pairs]
                            object_texts_long = [long_text for _, _, _, long_text in object_text_pairs]
                        else:
                            object_texts_short = [UNKNOWN_TEXT_TOKEN]
                            object_texts_long = [UNKNOWN_TEXT_TOKEN]
                    else:
                        parse_status = "fallback"
                        _builder_log(
                            "vlm_fallback_result "
                            f"frame_idx={int(frame_idx)} "
                            f"file={file_name} "
                            f"parse_status={parse_status} "
                            f"objects=0 "
                            f"raw_api_source={raw_api_source} "
                            f"object_vlm_sec={fallback_parse_sec:.2f}"
                        )
                        frame_text_short = UNKNOWN_TEXT_TOKEN
                        frame_text_long = UNKNOWN_TEXT_TOKEN
                        object_texts_short = [UNKNOWN_TEXT_TOKEN]
                        object_texts_long = [UNKNOWN_TEXT_TOKEN]

                embed_view_t0 = time.perf_counter()
                image_emb = embedder.embed_image(rgb_image).astype("float32")
                text_emb_short = embedder.embed_text(frame_text_short).astype("float32")
                text_emb_long = embedder.embed_text(frame_text_long).astype("float32")
                view_embedding_sec = float(time.perf_counter() - embed_view_t0)

                if image_emb.ndim != 1 or text_emb_short.ndim != 1 or text_emb_long.ndim != 1:
                    raise ValueError("Embedding must be a 1D vector")
                if (
                    image_emb.shape[0] != emb_dim
                    or text_emb_short.shape[0] != emb_dim
                    or text_emb_long.shape[0] != emb_dim
                ):
                    raise ValueError(
                        f"Embedding dim mismatch: image={image_emb.shape[0]}, "
                        f"text_short={text_emb_short.shape[0]}, text_long={text_emb_long.shape[0]}, "
                        f"expected={emb_dim}"
                    )
                if (
                    not np.isclose(x, world_position[0])
                    or not np.isclose(y, world_position[1])
                    or not np.isclose(z, world_position[2])
                ):
                    raise ValueError("2D/3D coordinate consistency check failed")
                if orientation not in valid_orientation_set:
                    raise ValueError(
                        f"Invalid orientation value: {orientation}; "
                        f"expected one of {list(normalized_scan_angles)}"
                    )

                entry_id = int(existing_entry_id) if existing_entry_id is not None else len(metadata_records)
                entry_object_count = 0
                metadata_record = {
                    "id": entry_id,
                    "frame_id": int(frame_idx),
                    "x": x,
                    "y": y,
                    "z": z,
                    "world_position": world_position,
                    "orientation": orientation,
                    "file_name": file_name,
                    "text": frame_text_short,
                    "frame_text_short": frame_text_short,
                    "frame_text_long": frame_text_long,
                    "parse_status": parse_status,
                    "parse_warnings": parse_warnings,
                    "raw_vlm_output": raw_vlm_output,
                    "raw_api_source": raw_api_source,
                    "text_input_for_clip_short": frame_text_short,
                    "text_input_for_clip_long": frame_text_long,
                    "object_text_inputs_short": object_texts_short,
                    "object_text_inputs_long": object_texts_long,
                    "builder_variant": selected_builder_variant,
                    "object_prompt_variant": selected_prompt_variant,
                    "attribute": dict(view_attribute),
                }
                raw_api_record = {
                    "entry_id": int(entry_id),
                    "frame_id": int(frame_idx),
                    "file_name": file_name,
                    "raw_api_source": raw_api_source,
                    "raw_api_response": raw_api_response,
                    "object_prompt_variant": selected_prompt_variant,
                    "geometry_pipeline_used": bool(geometry_result is not None and getattr(geometry_result, "ok", False)),
                    "geometry_fallback_reason": None
                    if geometry_result is None or geometry_result.ok
                    else geometry_result.failure_reason,
                    "selected_object_types": []
                    if geometry_result is None
                    else list(geometry_result.selector_payload.get("selected_object_types") or []),
                    "geometry_artifacts": {}
                    if geometry_result is None
                    else {
                        "detections_path": geometry_result.artifacts.detections_path,
                        "detection_overlay_path": geometry_result.artifacts.detection_overlay_path,
                        "filtered_detections_path": geometry_result.artifacts.filtered_detections_path,
                        "filtered_detection_overlay_path": geometry_result.artifacts.filtered_detection_overlay_path,
                        "depth_map_path": geometry_result.artifacts.depth_map_path,
                        "depth_preview_path": geometry_result.artifacts.depth_preview_path,
                    },
                    "timing": {
                        "frame_total_sec": 0.0,
                        "geometry_pipeline_total_sec": float(geometry_timing.get("total_sec") or 0.0),
                        "selector_sec": float(geometry_timing.get("selector_sec") or 0.0),
                        "dependency_setup_sec": float(geometry_timing.get("dependency_setup_sec") or 0.0),
                        "detector_sec": float(geometry_timing.get("detector_sec") or 0.0),
                        "depth_sec": float(geometry_timing.get("depth_sec") or 0.0),
                        "mask_total_sec": float(geometry_timing.get("mask_total_sec") or 0.0),
                        "angle_geometry_total_sec": float(geometry_timing.get("angle_geometry_total_sec") or 0.0),
                        "crop_vlm_description_total_sec": float(
                            geometry_timing.get("crop_vlm_description_total_sec") or 0.0
                        ),
                        "crop_vlm_description_avg_sec": float(
                            geometry_timing.get("crop_vlm_description_avg_sec") or 0.0
                        ),
                        "object_description_call_count": int(geometry_timing.get("object_description_call_count") or 0),
                        "detection_count_raw": int(geometry_timing.get("detection_count_raw") or 0),
                        "detection_count_class_matched": int(geometry_timing.get("detection_count_class_matched") or 0),
                        "detection_count_filtered_by_bbox_conf": int(
                            geometry_timing.get("detection_count_filtered_by_bbox_conf") or 0
                        ),
                        "detection_count_kept": int(geometry_timing.get("detection_count_kept") or 0),
                        "detection_count_truncated_by_max_objects": int(
                            geometry_timing.get("detection_count_truncated_by_max_objects") or 0
                        ),
                        "geometry_objects_before_r_threshold": int(geometry_objects_before_r_threshold),
                        "geometry_objects_after_r_threshold": int(geometry_objects_after_r_threshold),
                        "geometry_objects_filtered_by_r_threshold": int(geometry_objects_filtered_by_r_threshold),
                        "geometry_all_objects_filtered_by_r_threshold": bool(
                            geometry_all_objects_filtered_by_r_threshold
                        ),
                        "vlm_fallback_object_parse_sec": float(fallback_parse_sec),
                        "fallback_angle_geometry_sec": float(fallback_angle_geometry_sec),
                        "view_embedding_sec": float(view_embedding_sec),
                        "object_embedding_total_sec": float(object_embedding_total_sec),
                    },
                }
                if existing_entry_id is None:
                    metadata_records.append(metadata_record)
                    raw_api_records.append(raw_api_record)
                    image_embs.append(image_emb)
                    text_embs_short.append(text_emb_short)
                    text_embs_long.append(text_emb_long)
                    report["generated_entry_count"] += 1
                else:
                    if _response_has_length_finish_reason((existing_raw_api or {}).get("raw_api_response")):
                        report["regenerated_length_entry_count"] += 1
                    metadata_records[entry_id] = metadata_record
                    raw_api_records[entry_id] = raw_api_record
                    image_embs[entry_id] = image_emb
                    text_embs_short[entry_id] = text_emb_short
                    text_embs_long[entry_id] = text_emb_long
                file_name_to_entry_id[file_name] = int(entry_id)
                entry_object_records: List[ObjectGroupItem] = []

                if geometry_all_objects_filtered_by_r_threshold:
                    entry_object_records = []
                elif geometry_object_rows:
                    for geo_row, line_short, line_long in zip(geometry_object_rows, object_texts_short, object_texts_long):
                        object_embed_t0 = time.perf_counter()
                        obj_emb_short = embedder.embed_text(line_short).astype("float32")
                        obj_emb_long = embedder.embed_text(line_long).astype("float32")
                        object_embedding_total_sec += float(time.perf_counter() - object_embed_t0)
                        record = _make_object_record(
                            object_global_id=0,
                            frame_id=frame_idx,
                            entry_id=entry_id,
                            file_name=file_name,
                            x=x,
                            y=y,
                            z=z,
                            world_position=world_position,
                            orientation=orientation,
                            parse_status=parse_status,
                            builder_variant=selected_builder_variant,
                            angle_split_enable=angle_split_active,
                            angle_step=angle_step,
                            object_local_id=str(geo_row.get("object_local_id") or "det_000"),
                            label=str(geo_row.get("label") or "unknown"),
                            object_confidence=float(geo_row.get("object_confidence") or 0.0),
                            description=line_short,
                            long_form_open_description=str(
                                geo_row.get("long_form_open_description") or geo_row.get("object_text_long") or line_short
                            ),
                            attributes=list(geo_row.get("attributes") or []),
                            laterality=str(geo_row.get("laterality") or "center"),
                            distance_bin=str(geo_row.get("distance_bin") or "middle"),
                            verticality=str(geo_row.get("verticality") or "middle"),
                            distance_from_camera_m=geo_row.get("distance_from_camera_m"),
                            relative_height_from_camera_m=geo_row.get("relative_height_from_camera_m"),
                            relative_bearing_deg=geo_row.get("relative_bearing_deg"),
                            estimated_global_x=geo_row.get("estimated_global_x"),
                            estimated_global_y=geo_row.get("estimated_global_y"),
                            estimated_global_z=geo_row.get("estimated_global_z"),
                            any_text=str(geo_row.get("any_text") or ""),
                            location_relative_to_other_objects=str(
                                geo_row.get("location_relative_to_other_objects") or ""
                            ),
                            surrounding_context=list(geo_row.get("surrounding_context") or []),
                            scene_attributes=list(view_attribute.get("scene_attributes") or []),
                            object_text_short=line_short,
                            object_text_long=line_long,
                            precise_orientation_from_bearing=True,
                            geometry_source=str(geo_row.get("geometry_source") or "mask_depth"),
                            geometry_fallback_reason=geo_row.get("geometry_fallback_reason"),
                            detector_label=geo_row.get("detector_label_raw") or geo_row.get("detector_label"),
                            detector_label_raw=geo_row.get("detector_label_raw"),
                            detector_confidence=geo_row.get("detector_confidence"),
                            vlm_label=geo_row.get("vlm_label"),
                            final_label=geo_row.get("final_label"),
                            label_source=geo_row.get("label_source"),
                            label_conflict=geo_row.get("label_conflict"),
                            occlusion_source=geo_row.get("occlusion_source"),
                            occlusion_level=geo_row.get("occlusion_level"),
                            occlusion_penalty_p_o=geo_row.get("occlusion_penalty_p_o"),
                            reweighted_detection_score_r=geo_row.get("reweighted_detection_score_r"),
                            visible_occlusion_ratio=geo_row.get("visible_occlusion_ratio"),
                            occluded_boundary_ratio=geo_row.get("occluded_boundary_ratio"),
                            nearer_ring_overlap_ratio=geo_row.get("nearer_ring_overlap_ratio"),
                            object_depth_median=geo_row.get("object_depth_median"),
                            boundary_pixel_count=geo_row.get("boundary_pixel_count"),
                            occluded_boundary_pixel_count=geo_row.get("occluded_boundary_pixel_count"),
                            ring_pixel_count=geo_row.get("ring_pixel_count"),
                            nearer_ring_pixel_count=geo_row.get("nearer_ring_pixel_count"),
                            depth_margin_delta=geo_row.get("depth_margin_delta"),
                            occluding_overlap_pixel_count=geo_row.get("occluding_overlap_pixel_count"),
                            foreground_occluder_count=geo_row.get("foreground_occluder_count"),
                            occlusion_target_overlap_threshold=geo_row.get("occlusion_target_overlap_threshold"),
                            bbox_xywh_norm=geo_row.get("bbox_xywh_norm"),
                            bbox_xyxy=geo_row.get("bbox_xyxy"),
                            mask_area_px=geo_row.get("mask_area_px"),
                            mask_area_ratio=geo_row.get("mask_area_ratio"),
                            mask_centroid_x_px=geo_row.get("mask_centroid_x_px"),
                            mask_centroid_y_px=geo_row.get("mask_centroid_y_px"),
                            mask_centroid_x_norm=geo_row.get("mask_centroid_x_norm"),
                            mask_centroid_y_norm=geo_row.get("mask_centroid_y_norm"),
                            depth_stat_median_m=geo_row.get("depth_stat_median_m"),
                            depth_stat_p10_m=geo_row.get("depth_stat_p10_m"),
                            depth_stat_p90_m=geo_row.get("depth_stat_p90_m"),
                            projected_planar_distance_m=geo_row.get("projected_planar_distance_m"),
                            vertical_angle_deg=geo_row.get("vertical_angle_deg"),
                            vlm_distance_from_camera_m=geo_row.get("vlm_distance_from_camera_m"),
                            vlm_relative_bearing_deg=geo_row.get("vlm_relative_bearing_deg"),
                            crop_path=geo_row.get("crop_path"),
                            mask_path=geo_row.get("mask_path"),
                            mask_overlay_path=geo_row.get("mask_overlay_path"),
                            depth_map_path=geo_row.get("depth_map_path"),
                            crop_vlm_label=geo_row.get("crop_vlm_label"),
                            dinov3_model_name=geo_row.get("dinov3_model_name"),
                            dinov3_embedding_dim=geo_row.get("dinov3_embedding_dim"),
                            dinov3_input_type=geo_row.get("dinov3_input_type"),
                            dinov3_normalized=geo_row.get("dinov3_normalized"),
                            dinov3_status=geo_row.get("dinov3_status"),
                            dinov3_failure_reason=geo_row.get("dinov3_failure_reason"),
                        )
                        record["view_type"] = str(view_attribute.get("view_type") or "unknown")
                        record["room_function"] = str(view_attribute.get("room_function") or "unknown")
                        record["style_hint"] = str(view_attribute.get("style_hint") or "unknown")
                        record["clutter_level"] = str(view_attribute.get("clutter_level") or "unknown")
                        dino_embedding = geo_row.get("dinov3_embedding")
                        entry_object_records.append(
                            (
                                record,
                                obj_emb_short,
                                obj_emb_long,
                                None
                                if dino_embedding is None
                                else np.asarray(dino_embedding, dtype=np.float32).reshape(-1),
                            )
                        )
                        bucket_key = f"total_{record['angle_bucket']}_bucket_objects"
                        report[bucket_key] = int(report.get(bucket_key, 0)) + 1
                        entry_object_count += 1
                elif scene_objects is not None and object_text_pairs:
                    for obj, line_short, raw_long, line_long in object_text_pairs:
                        fallback_occlusion_level, fallback_occlusion_penalty, fallback_reweighted_detection_score = (
                            _compute_object_reweight_fields(
                                detector_confidence=None,
                                object_confidence=1.0,
                                occlusion_level="uncertain",
                                occlusion_reweight_w1=float(occlusion_reweight_w1),
                                occlusion_reweight_w2=float(occlusion_reweight_w2),
                                occlusion_reweight_b=float(occlusion_reweight_b),
                                occlusion_reweight_eps=float(OCCLUSION_REWEIGHT_EPS),
                            )
                        )
                        object_embed_t0 = time.perf_counter()
                        obj_emb_short = embedder.embed_text(line_short).astype("float32")
                        obj_emb_long = embedder.embed_text(line_long).astype("float32")
                        object_embedding_total_sec += float(time.perf_counter() - object_embed_t0)
                        record = _make_object_record(
                            object_global_id=0,
                            frame_id=frame_idx,
                            entry_id=entry_id,
                            file_name=file_name,
                            x=x,
                            y=y,
                            z=z,
                            world_position=world_position,
                            orientation=orientation,
                            parse_status=parse_status,
                            builder_variant=selected_builder_variant,
                            angle_split_enable=angle_split_active,
                            angle_step=angle_step,
                            scene_objects=scene_objects,
                            obj=obj,
                            object_local_id=obj.feature_id,
                            label=obj.type,
                            object_confidence=1.0,
                            description=line_short,
                            long_form_open_description=raw_long,
                            attributes=list(obj.attributes or []),
                            laterality=obj.relative_position_laterality,
                            distance_bin=obj.relative_position_distance,
                            verticality=obj.relative_position_verticality,
                            distance_from_camera_m=obj.distance_from_camera_m,
                            relative_height_from_camera_m=getattr(obj, "relative_height_from_camera_m", None),
                            relative_bearing_deg=obj.relative_bearing_deg,
                            estimated_global_x=obj.estimated_global_x,
                            estimated_global_y=getattr(obj, "estimated_global_y", None),
                            estimated_global_z=obj.estimated_global_z,
                            any_text=obj.any_text,
                            location_relative_to_other_objects=obj.location_relative_to_other_objects,
                            surrounding_context=_serialize_surrounding_context(obj.surrounding_context),
                            scene_attributes=list(scene_objects.scene_attributes or []),
                            object_text_short=line_short,
                            object_text_long=line_long,
                            precise_orientation_from_bearing=True,
                            geometry_source="vlm_fallback",
                            geometry_fallback_reason=None
                            if geometry_result is None
                            else geometry_result.failure_reason,
                            detector_label=None,
                            detector_label_raw=None,
                            detector_confidence=None,
                            vlm_label=obj.type,
                            final_label=obj.type,
                            label_source="vlm",
                            label_conflict=False,
                            occlusion_source=selected_occlusion_source,
                            occlusion_level=fallback_occlusion_level,
                            occlusion_penalty_p_o=fallback_occlusion_penalty,
                            reweighted_detection_score_r=fallback_reweighted_detection_score,
                            visible_occlusion_ratio=None,
                            occluded_boundary_ratio=None,
                            nearer_ring_overlap_ratio=None,
                            object_depth_median=None,
                            boundary_pixel_count=None,
                            occluded_boundary_pixel_count=None,
                            ring_pixel_count=None,
                            nearer_ring_pixel_count=None,
                            depth_margin_delta=None,
                            vlm_distance_from_camera_m=obj.distance_from_camera_m,
                            vlm_relative_bearing_deg=obj.relative_bearing_deg,
                            dinov3_status="missing" if dinov3_enabled else "disabled",
                        )
                        entry_object_records.append((record, obj_emb_short, obj_emb_long, None))
                        pre_threshold_r_score_rows_by_entry_id.setdefault(int(entry_id), []).append(
                            _build_object_r_scores_pre_threshold_row(
                                record,
                                entry_id=int(entry_id),
                                frame_id=int(frame_idx),
                                file_name=file_name,
                                r_threshold=r_threshold,
                            )
                        )
                        bucket_key = f"total_{record['angle_bucket']}_bucket_objects"
                        report[bucket_key] = int(report.get(bucket_key, 0)) + 1
                        entry_object_count += 1
                else:
                    fallback_occlusion_level, fallback_occlusion_penalty, fallback_reweighted_detection_score = (
                        _compute_object_reweight_fields(
                            detector_confidence=None,
                            object_confidence=0.0,
                            occlusion_level="uncertain",
                            occlusion_reweight_w1=float(occlusion_reweight_w1),
                            occlusion_reweight_w2=float(occlusion_reweight_w2),
                            occlusion_reweight_b=float(occlusion_reweight_b),
                            occlusion_reweight_eps=float(OCCLUSION_REWEIGHT_EPS),
                        )
                    )
                    angle_bucket = _normalize_angle_bucket("center", angle_split_enable=angle_split_active)
                    line_short = UNKNOWN_TEXT_TOKEN
                    line_long = _format_object_text_long(
                        UNKNOWN_TEXT_TOKEN,
                        angle_bucket=angle_bucket,
                        builder_variant=selected_builder_variant,
                    )
                    object_embed_t0 = time.perf_counter()
                    obj_emb_short = embedder.embed_text(line_short).astype("float32")
                    obj_emb_long = embedder.embed_text(line_long).astype("float32")
                    object_embedding_total_sec += float(time.perf_counter() - object_embed_t0)
                    record = _make_object_record(
                        object_global_id=0,
                        frame_id=frame_idx,
                        entry_id=entry_id,
                        file_name=file_name,
                        x=x,
                        y=y,
                        z=z,
                        world_position=world_position,
                        orientation=orientation,
                        parse_status=parse_status,
                        builder_variant=selected_builder_variant,
                        angle_split_enable=angle_split_active,
                        angle_step=angle_step,
                        object_local_id="none_000",
                        label="none",
                        object_confidence=0.0,
                        laterality=angle_bucket,
                        object_text_short=line_short,
                        object_text_long=line_long,
                        geometry_source="vlm_fallback",
                        geometry_fallback_reason=None if geometry_result is None else geometry_result.failure_reason,
                        detector_label_raw=None,
                        vlm_label="unknown",
                        final_label="none",
                        label_source="placeholder",
                        label_conflict=False,
                        occlusion_source=selected_occlusion_source,
                        occlusion_level=fallback_occlusion_level,
                        occlusion_penalty_p_o=fallback_occlusion_penalty,
                        reweighted_detection_score_r=fallback_reweighted_detection_score,
                        visible_occlusion_ratio=None,
                        occluded_boundary_ratio=None,
                        nearer_ring_overlap_ratio=None,
                        object_depth_median=None,
                        boundary_pixel_count=None,
                        occluded_boundary_pixel_count=None,
                        ring_pixel_count=None,
                        nearer_ring_pixel_count=None,
                        depth_margin_delta=None,
                        dinov3_status="missing" if dinov3_enabled else "disabled",
                    )
                    entry_object_records.append((record, obj_emb_short, obj_emb_long, None))
                    pre_threshold_r_score_rows_by_entry_id.setdefault(int(entry_id), []).append(
                        _build_object_r_scores_pre_threshold_row(
                            record,
                            entry_id=int(entry_id),
                            frame_id=int(frame_idx),
                            file_name=file_name,
                            r_threshold=r_threshold,
                        )
                    )
                    report["total_center_bucket_objects"] = int(report.get("total_center_bucket_objects", 0)) + 1
                    entry_object_count += 1
                object_groups_by_entry_id[int(entry_id)] = entry_object_records

                if parse_status == "ok":
                    report["parse_ok_count"] += 1
                elif parse_status == "fallback":
                    report["parse_fallback_count"] += 1
                else:
                    report["parse_failed_count"] += 1

                metadata_records[entry_id]["object_count"] = int(entry_object_count)
                frame_total_sec = float(time.perf_counter() - frame_t0)
                raw_api_record["timing"]["view_embedding_sec"] = float(view_embedding_sec)
                raw_api_record["timing"]["object_embedding_total_sec"] = float(object_embedding_total_sec)
                raw_api_record["timing"]["frame_total_sec"] = frame_total_sec
                route_label = _frame_route_label(
                    geometry_object_rows=geometry_object_rows,
                    geometry_all_objects_filtered_by_r_threshold=geometry_all_objects_filtered_by_r_threshold,
                )
                timing_records.append(
                    {
                        "frame_idx": int(frame_idx),
                        "entry_id": int(entry_id),
                        "file_name": file_name,
                        "route": route_label,
                        "parse_status": parse_status,
                        "object_count": int(entry_object_count),
                        "raw_api_source": raw_api_source,
                        "frame_total_sec": frame_total_sec,
                        "geometry_pipeline_total_sec": float(geometry_timing.get("total_sec") or 0.0),
                        "selector_sec": float(geometry_timing.get("selector_sec") or 0.0),
                        "dependency_setup_sec": float(geometry_timing.get("dependency_setup_sec") or 0.0),
                        "detector_sec": float(geometry_timing.get("detector_sec") or 0.0),
                        "depth_sec": float(geometry_timing.get("depth_sec") or 0.0),
                        "mask_total_sec": float(geometry_timing.get("mask_total_sec") or 0.0),
                        "angle_geometry_total_sec": float(geometry_timing.get("angle_geometry_total_sec") or 0.0),
                        "crop_vlm_description_total_sec": float(geometry_timing.get("crop_vlm_description_total_sec") or 0.0),
                        "crop_vlm_description_avg_sec": float(geometry_timing.get("crop_vlm_description_avg_sec") or 0.0),
                        "object_description_call_count": int(geometry_timing.get("object_description_call_count") or 0),
                        "detection_count_raw": int(geometry_timing.get("detection_count_raw") or 0),
                        "detection_count_class_matched": int(geometry_timing.get("detection_count_class_matched") or 0),
                        "detection_count_filtered_by_bbox_conf": int(
                            geometry_timing.get("detection_count_filtered_by_bbox_conf") or 0
                        ),
                        "detection_count_kept": int(geometry_timing.get("detection_count_kept") or 0),
                        "detection_count_truncated_by_max_objects": int(
                            geometry_timing.get("detection_count_truncated_by_max_objects") or 0
                        ),
                        "geometry_objects_before_r_threshold": int(geometry_objects_before_r_threshold),
                        "geometry_objects_after_r_threshold": int(geometry_objects_after_r_threshold),
                        "geometry_objects_filtered_by_r_threshold": int(geometry_objects_filtered_by_r_threshold),
                        "geometry_all_objects_filtered_by_r_threshold": bool(
                            geometry_all_objects_filtered_by_r_threshold
                        ),
                        "vlm_fallback_object_parse_sec": float(fallback_parse_sec),
                        "fallback_angle_geometry_sec": float(fallback_angle_geometry_sec),
                        "view_embedding_sec": float(view_embedding_sec),
                        "object_embedding_total_sec": float(object_embedding_total_sec),
                    }
                )
                _builder_log(
                    "frame_done "
                    f"frame_idx={int(frame_idx)} "
                    f"entry_id={int(entry_id)} "
                    f"file={file_name} "
                    f"route={route_label} "
                    f"parse_status={parse_status} "
                    f"object_count={int(entry_object_count)} "
                    f"raw_api_source={raw_api_source} "
                    f"frame_total_sec={frame_total_sec:.2f} "
                    f"depth_sec={float(geometry_timing.get('depth_sec') or 0.0):.2f} "
                    f"angle_sec={float(geometry_timing.get('angle_geometry_total_sec') or 0.0):.2f} "
                    f"crop_vlm_sec={float(geometry_timing.get('crop_vlm_description_total_sec') or 0.0):.2f} "
                    f"fallback_vlm_sec={fallback_parse_sec:.2f}"
                )

            except Exception as exc:
                timing_records.append(
                    {
                        "frame_idx": int(frame_idx),
                        "entry_id": None,
                        "file_name": locals().get("file_name", "unknown"),
                        "route": "error",
                        "frame_total_sec": float(time.perf_counter() - frame_t0),
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
                _builder_log(
                    "frame_error "
                    f"frame_idx={int(frame_idx)} "
                    f"file={locals().get('file_name', 'unknown')} "
                    f"error={type(exc).__name__}:{exc}"
                )
                failures.append({"frame_index": frame_idx, "error": str(exc)})

        image_arr = (
            np.vstack(image_embs).astype("float32")
            if image_embs
            else np.zeros((0, emb_dim), dtype="float32")
        )
        text_arr_short = (
            np.vstack(text_embs_short).astype("float32")
            if text_embs_short
            else np.zeros((0, emb_dim), dtype="float32")
        )
        text_arr_long = (
            np.vstack(text_embs_long).astype("float32")
            if text_embs_long
            else np.zeros((0, emb_dim), dtype="float32")
        )
        object_metadata_records: List[Dict] = []
        object_text_embs_short: List[np.ndarray] = []
        object_text_embs_long: List[np.ndarray] = []
        object_dinov3_embs: List[np.ndarray] = []
        for entry_id, _meta in enumerate(metadata_records):
            for record, obj_emb_short, obj_emb_long, obj_emb_dinov3 in list(object_groups_by_entry_id.get(entry_id, [])):
                out_record = dict(record)
                out_record["entry_id"] = int(entry_id)
                out_record["object_global_id"] = int(len(object_metadata_records))
                if bool(dinov3_store_enabled) and obj_emb_dinov3 is not None:
                    out_record["dinov3_embedding_row_index"] = int(len(object_dinov3_embs))
                    object_dinov3_embs.append(np.asarray(obj_emb_dinov3, dtype=np.float32).reshape(-1))
                else:
                    out_record["dinov3_embedding_row_index"] = None
                object_metadata_records.append(out_record)
                object_text_embs_short.append(np.asarray(obj_emb_short, dtype=np.float32).reshape(-1))
                object_text_embs_long.append(np.asarray(obj_emb_long, dtype=np.float32).reshape(-1))

        object_arr_short = (
            np.vstack(object_text_embs_short).astype("float32")
            if object_text_embs_short
            else np.zeros((0, emb_dim), dtype="float32")
        )
        object_arr_long = (
            np.vstack(object_text_embs_long).astype("float32")
            if object_text_embs_long
            else np.zeros((0, emb_dim), dtype="float32")
        )
        if object_dinov3_embs:
            dino_dim = int(object_dinov3_embs[0].shape[0])
            object_arr_dinov3 = np.vstack(object_dinov3_embs).astype("float32")
        else:
            dino_dim = int(getattr(dino_embedder, "embedding_dim", 0) or 0)
            object_arr_dinov3 = np.zeros((0, dino_dim), dtype="float32")
        view_object_relations = _build_view_object_relations(
            metadata_records=metadata_records,
            object_metadata_records=object_metadata_records,
        )
        object_object_relations = _build_object_object_relations(
            metadata_records=metadata_records,
            object_metadata_records=object_metadata_records,
        )

        _write_jsonl(output_root / "meta.jsonl", metadata_records)
        _write_jsonl(output_root / "metadata.jsonl", metadata_records)
        _write_jsonl(output_root / "raw_api_responses.jsonl", raw_api_records)
        _write_jsonl(output_root / "per_image_timings.jsonl", timing_records)
        np.save(output_root / "image_emb.npy", image_arr)
        np.save(output_root / "text_emb_short.npy", text_arr_short)
        np.save(output_root / "text_emb_long.npy", text_arr_long)

        _write_jsonl(output_root / "object_meta.jsonl", object_metadata_records)
        _write_jsonl(output_root / "view_object_relations.jsonl", view_object_relations)
        _write_jsonl(output_root / "object_object_relations.jsonl", object_object_relations)
        np.save(output_root / "object_text_emb_short.npy", object_arr_short)
        np.save(output_root / "object_text_emb_long.npy", object_arr_long)
        np.save(output_root / "object_dinov3_emb.npy", object_arr_dinov3)
        pre_threshold_r_scores_path = output_root / "object_r_scores_pre_threshold.csv"
        final_r_scores_path = output_root / "object_r_scores.csv"
        final_pre_threshold_r_score_rows = _flatten_object_r_scores_pre_threshold_rows(
            pre_threshold_r_score_rows_by_entry_id
        )
        if not final_pre_threshold_r_score_rows and object_metadata_records:
            final_pre_threshold_r_score_rows = _rebuild_object_r_scores_pre_threshold_rows_from_records(
                object_metadata_records,
                r_threshold=r_threshold,
            )
        report["object_r_scores_pre_threshold_count"] = _write_object_r_scores_pre_threshold_csv(
            pre_threshold_r_scores_path,
            final_pre_threshold_r_score_rows,
        )
        report["object_r_scores_count"] = _write_object_r_scores_csv(
            final_r_scores_path,
            object_metadata_records,
        )
        report["object_r_scores_pre_threshold_csv_path"] = str(pre_threshold_r_scores_path)
        report["object_r_scores_csv_path"] = str(final_r_scores_path)
        if export_object_crops_dir is not None:
            report["object_crops_by_global_id"] = export_object_crops_by_global_id(
                db_root=output_root,
                object_rows=object_metadata_records,
                output_dir=export_object_crops_dir,
            )

        report["image_index_ntotal"] = _save_faiss_index(image_arr, output_root / "image_index.faiss")
        report["text_index_ntotal_short"] = _save_faiss_index(text_arr_short, output_root / "text_index_short.faiss")
        report["text_index_ntotal_long"] = _save_faiss_index(text_arr_long, output_root / "text_index_long.faiss")
        report["object_index_ntotal_short"] = _save_faiss_index(
            object_arr_short,
            output_root / "object_index_short.faiss",
        )
        report["object_index_ntotal_long"] = _save_faiss_index(
            object_arr_long,
            output_root / "object_index_long.faiss",
        )
        report["object_dinov3_ntotal"] = int(object_arr_dinov3.shape[0]) if object_arr_dinov3.ndim == 2 else 0

        overview_dir = output_root / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)

        overview_outputs = {}
        try:
            center_view = explorer.render_center_highest_view(hfov=120.0)
            center_path = overview_dir / "center_highest_view.jpg"
            if cv2.imwrite(str(center_path), center_view):
                overview_outputs["center_highest_view"] = str(center_path)
        except Exception as exc:
            failures.append({"overview": "center_highest_view", "error": str(exc)})

        try:
            traj_view = explorer.render_center_highest_view_with_trajectory(poses, hfov=120.0)
            traj_path = overview_dir / "trajectory_on_center_highest_view.jpg"
            if cv2.imwrite(str(traj_path), traj_view):
                overview_outputs["trajectory_on_center_highest_view"] = str(traj_path)
        except Exception as exc:
            failures.append({"overview": "trajectory_on_center_highest_view", "error": str(exc)})

        try:
            textured = explorer.render_textured_floor_plan()
            textured_path = overview_dir / "textured_floor_plan.jpg"
            if cv2.imwrite(str(textured_path), textured):
                overview_outputs["textured_floor_plan"] = str(textured_path)
                projection_path = _write_floor_plan_projection(
                    overview_dir / "floor_plan_projection.json",
                    getattr(explorer, "_last_top_down_projection", None),
                )
                if projection_path:
                    overview_outputs["floor_plan_projection"] = projection_path
        except Exception as exc:
            failures.append({"overview": "textured_floor_plan", "error": str(exc)})

        report["parse_ok_count"] = sum(1 for row in metadata_records if str(row.get("parse_status") or "") == "ok")
        report["parse_fallback_count"] = sum(
            1 for row in metadata_records if str(row.get("parse_status") or "") == "fallback"
        )
        report["parse_failed_count"] = sum(
            1
            for row in metadata_records
            if str(row.get("parse_status") or "") not in {"ok", "fallback"}
        )
        geometry_ok_count, geometry_fallback_count = _summarize_geometry_outcomes_from_object_groups(
            object_groups_by_entry_id
        )
        report["geometry_ok_count"] = int(geometry_ok_count)
        report["geometry_fallback_count"] = int(geometry_fallback_count)
        report["geometry_objects_before_r_threshold"] = sum(
            1 for row in final_pre_threshold_r_score_rows if str(row.get("object_route") or "") == "geometry"
        )
        report["geometry_objects_after_r_threshold"] = sum(
            1 for row in object_metadata_records if str(row.get("geometry_source") or "") != "vlm_fallback"
        )
        report["geometry_objects_filtered_by_r_threshold"] = max(
            0,
            int(report["geometry_objects_before_r_threshold"]) - int(report["geometry_objects_after_r_threshold"]),
        )
        geometry_before_by_entry: Dict[int, int] = {}
        for row in final_pre_threshold_r_score_rows:
            if str(row.get("object_route") or "") != "geometry":
                continue
            try:
                entry_id = int(row.get("entry_id", -1))
            except Exception:
                continue
            geometry_before_by_entry[entry_id] = int(geometry_before_by_entry.get(entry_id, 0)) + 1
        geometry_after_by_entry: Dict[int, int] = {}
        for row in object_metadata_records:
            if str(row.get("geometry_source") or "") == "vlm_fallback":
                continue
            try:
                entry_id = int(row.get("entry_id", -1))
            except Exception:
                continue
            geometry_after_by_entry[entry_id] = int(geometry_after_by_entry.get(entry_id, 0)) + 1
        report["frames_all_geometry_objects_filtered"] = sum(
            1
            for entry_id, before_count in geometry_before_by_entry.items()
            if int(before_count) > 0 and int(geometry_after_by_entry.get(entry_id, 0)) == 0
        )
        report["total_left_bucket_objects"] = sum(
            1 for row in object_metadata_records if str(row.get("angle_bucket") or "") == "left"
        )
        report["total_center_bucket_objects"] = sum(
            1 for row in object_metadata_records if str(row.get("angle_bucket") or "") == "center"
        )
        report["total_right_bucket_objects"] = sum(
            1 for row in object_metadata_records if str(row.get("angle_bucket") or "") == "right"
        )
        report["total_entries"] = len(metadata_records)
        report["failed_entries"] = len(failures)
        report["failure_examples"] = failures[:20]
        report["overview_outputs"] = overview_outputs
        report["total_objects"] = len(object_metadata_records)
        report["total_view_object_relations"] = len(view_object_relations)
        report["total_object_object_relations"] = len(object_object_relations)
        try:
            report["polar_surrounding_postprocess"] = _run_optional_polar_surrounding_postprocess(
                output_root,
                enabled=bool(run_polar_surrounding_postprocess),
            )
        except Exception as exc:
            failures.append({"postprocess": "polar_surrounding", "error": str(exc)})
            report["polar_surrounding_postprocess"] = {
                "enabled": bool(run_polar_surrounding_postprocess),
                "ran": bool(run_polar_surrounding_postprocess),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if len(metadata_records) > 0:
            report["avg_objects_per_frame"] = float(len(object_metadata_records) / len(metadata_records))
        report["finished_at"] = _now_iso()

        with (output_root / "build_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)

        return report
    finally:
        explorer.close()


def build_spatial_database(
    scene_path: str = SCENE_PATH,
    meters_per_step: float = 1.5,
    max_positions: Optional[int] = None,
    output_dir: str = SPATIAL_DB_DIR,
    vlm_model: str = SPATIAL_DB_VLM_MODEL,
    use_cache: bool = True,
    object_max_per_frame: int = OBJECT_MAX_PER_FRAME,
    object_parse_retries: int = OBJECT_PARSE_RETRIES,
    object_use_cache: bool = OBJECT_USE_CACHE,
    object_cache_dir: Optional[str] = None,
    tour_mode: str = "full_house",
    random_num_steps: int = 50,
    random_step_size: float = 1.0,
    random_scan_angles: Sequence[int] = SCAN_ANGLES,
    random_seed: Optional[int] = None,
    random_max_attempts_per_step: int = 32,
    random_include_start_scan: bool = True,
    run_polar_surrounding_postprocess: bool = False,
    execution_mode: str = "capture_then_parallel_vlm",
    vlm_max_in_flight: int = 4,
    legacy_per_frame: bool = False,
    bbox_conf_threshold: float = float(BBOX_CONF_THRESHOLD),
    occlusion_reweight_w1: float = float(OCCLUSION_REWEIGHT_W1),
    occlusion_reweight_w2: float = float(OCCLUSION_REWEIGHT_W2),
    occlusion_reweight_b: float = float(OCCLUSION_REWEIGHT_B),
    occlusion_source: str = str(OCCLUSION_SOURCE),
    occlusion_target_overlap_threshold: float = float(OCCLUSION_TARGET_OVERLAP_THRESHOLD),
    visible_occ_boundary_width: int = int(VISIBLE_OCC_BOUNDARY_WIDTH),
    visible_occ_ring_radius: int = int(VISIBLE_OCC_RING_RADIUS),
    visible_occ_depth_margin_delta: float = float(VISIBLE_OCC_DEPTH_MARGIN_DELTA),
    visible_occ_boundary_neighbor_radius: int = int(VISIBLE_OCC_BOUNDARY_NEIGHBOR_RADIUS),
    enable_dinov3_embedding: bool = bool(ENABLE_DINOV3_EMBEDDING),
    store_dinov3_embedding: bool = bool(STORE_DINOV3_EMBEDDING),
    dinov3_model_name: str = str(DINOV3_MODEL_NAME),
    dinov3_batch_size: int = int(DINOV3_BATCH_SIZE),
    dinov3_normalize: bool = bool(DINOV3_NORMALIZE),
    r_threshold: Optional[float] = None,
    export_object_crops_by_global_id_dir: Optional[str] = None,
) -> Dict:
    return _build_spatial_database_core(
        scene_path=scene_path,
        meters_per_step=meters_per_step,
        max_positions=max_positions,
        output_dir=output_dir,
        vlm_model=vlm_model,
        use_cache=use_cache,
        object_max_per_frame=object_max_per_frame,
        object_parse_retries=object_parse_retries,
        object_use_cache=object_use_cache,
        object_cache_dir=object_cache_dir,
        tour_mode=tour_mode,
        random_num_steps=random_num_steps,
        random_step_size=random_step_size,
        random_scan_angles=random_scan_angles,
        random_seed=random_seed,
        random_max_attempts_per_step=random_max_attempts_per_step,
        random_include_start_scan=random_include_start_scan,
        object_prompt_variant="standard",
        object_orientation_mode="frame",
        report_builder_variant="standard",
        angle_split_enable=False,
        angle_step=int(VLM_ANGLE_STEP),
        run_polar_surrounding_postprocess=bool(run_polar_surrounding_postprocess),
        execution_mode=str(execution_mode),
        vlm_max_in_flight=int(vlm_max_in_flight),
        legacy_per_frame=bool(legacy_per_frame),
        bbox_conf_threshold=float(bbox_conf_threshold),
        occlusion_reweight_w1=float(occlusion_reweight_w1),
        occlusion_reweight_w2=float(occlusion_reweight_w2),
        occlusion_reweight_b=float(occlusion_reweight_b),
        occlusion_source=str(occlusion_source),
        occlusion_target_overlap_threshold=float(occlusion_target_overlap_threshold),
        visible_occ_boundary_width=int(visible_occ_boundary_width),
        visible_occ_ring_radius=int(visible_occ_ring_radius),
        visible_occ_depth_margin_delta=float(visible_occ_depth_margin_delta),
        visible_occ_boundary_neighbor_radius=int(visible_occ_boundary_neighbor_radius),
        enable_dinov3_embedding=bool(enable_dinov3_embedding),
        store_dinov3_embedding=bool(store_dinov3_embedding),
        dinov3_model_name=str(dinov3_model_name),
        dinov3_batch_size=int(dinov3_batch_size),
        dinov3_normalize=bool(dinov3_normalize),
        r_threshold=None if r_threshold is None else float(r_threshold),
        export_object_crops_by_global_id_dir=export_object_crops_by_global_id_dir,
    )


def build_spatial_database_angle_split(
    scene_path: str = SCENE_PATH,
    meters_per_step: float = 1.5,
    max_positions: Optional[int] = None,
    output_dir: str = SPATIAL_DB_DIR,
    vlm_model: str = SPATIAL_DB_VLM_MODEL,
    use_cache: bool = True,
    object_max_per_frame: int = OBJECT_MAX_PER_FRAME,
    object_parse_retries: int = OBJECT_PARSE_RETRIES,
    object_use_cache: bool = OBJECT_USE_CACHE,
    object_cache_dir: Optional[str] = None,
    tour_mode: str = "full_house",
    random_num_steps: int = 50,
    random_step_size: float = 1.0,
    random_scan_angles: Sequence[int] = SCAN_ANGLES,
    random_seed: Optional[int] = None,
    random_max_attempts_per_step: int = 32,
    random_include_start_scan: bool = True,
    angle_split_enable: bool = VLM_ANGLE_SPLIT_ENABLE,
    angle_step: int = VLM_ANGLE_STEP,
    run_polar_surrounding_postprocess: bool = False,
    execution_mode: str = "capture_then_parallel_vlm",
    vlm_max_in_flight: int = 4,
    legacy_per_frame: bool = False,
    bbox_conf_threshold: float = float(BBOX_CONF_THRESHOLD),
    occlusion_reweight_w1: float = float(OCCLUSION_REWEIGHT_W1),
    occlusion_reweight_w2: float = float(OCCLUSION_REWEIGHT_W2),
    occlusion_reweight_b: float = float(OCCLUSION_REWEIGHT_B),
    occlusion_source: str = str(OCCLUSION_SOURCE),
    occlusion_target_overlap_threshold: float = float(OCCLUSION_TARGET_OVERLAP_THRESHOLD),
    visible_occ_boundary_width: int = int(VISIBLE_OCC_BOUNDARY_WIDTH),
    visible_occ_ring_radius: int = int(VISIBLE_OCC_RING_RADIUS),
    visible_occ_depth_margin_delta: float = float(VISIBLE_OCC_DEPTH_MARGIN_DELTA),
    visible_occ_boundary_neighbor_radius: int = int(VISIBLE_OCC_BOUNDARY_NEIGHBOR_RADIUS),
    enable_dinov3_embedding: bool = bool(ENABLE_DINOV3_EMBEDDING),
    store_dinov3_embedding: bool = bool(STORE_DINOV3_EMBEDDING),
    dinov3_model_name: str = str(DINOV3_MODEL_NAME),
    dinov3_batch_size: int = int(DINOV3_BATCH_SIZE),
    dinov3_normalize: bool = bool(DINOV3_NORMALIZE),
    r_threshold: Optional[float] = None,
    export_object_crops_by_global_id_dir: Optional[str] = None,
) -> Dict:
    return _build_spatial_database_core(
        scene_path=scene_path,
        meters_per_step=meters_per_step,
        max_positions=max_positions,
        output_dir=output_dir,
        vlm_model=vlm_model,
        use_cache=use_cache,
        object_max_per_frame=object_max_per_frame,
        object_parse_retries=object_parse_retries,
        object_use_cache=object_use_cache,
        object_cache_dir=object_cache_dir,
        tour_mode=tour_mode,
        random_num_steps=random_num_steps,
        random_step_size=random_step_size,
        random_scan_angles=random_scan_angles,
        random_seed=random_seed,
        random_max_attempts_per_step=random_max_attempts_per_step,
        random_include_start_scan=random_include_start_scan,
        object_prompt_variant="angle_split",
        object_orientation_mode="laterality_offset",
        report_builder_variant="angle_split",
        angle_split_enable=bool(angle_split_enable),
        angle_step=int(angle_step),
        run_polar_surrounding_postprocess=bool(run_polar_surrounding_postprocess),
        execution_mode=str(execution_mode),
        vlm_max_in_flight=int(vlm_max_in_flight),
        legacy_per_frame=bool(legacy_per_frame),
        bbox_conf_threshold=float(bbox_conf_threshold),
        occlusion_reweight_w1=float(occlusion_reweight_w1),
        occlusion_reweight_w2=float(occlusion_reweight_w2),
        occlusion_reweight_b=float(occlusion_reweight_b),
        occlusion_source=str(occlusion_source),
        occlusion_target_overlap_threshold=float(occlusion_target_overlap_threshold),
        visible_occ_boundary_width=int(visible_occ_boundary_width),
        visible_occ_ring_radius=int(visible_occ_ring_radius),
        visible_occ_depth_margin_delta=float(visible_occ_depth_margin_delta),
        visible_occ_boundary_neighbor_radius=int(visible_occ_boundary_neighbor_radius),
        enable_dinov3_embedding=bool(enable_dinov3_embedding),
        store_dinov3_embedding=bool(store_dinov3_embedding),
        dinov3_model_name=str(dinov3_model_name),
        dinov3_batch_size=int(dinov3_batch_size),
        dinov3_normalize=bool(dinov3_normalize),
        r_threshold=None if r_threshold is None else float(r_threshold),
        export_object_crops_by_global_id_dir=export_object_crops_by_global_id_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a spatial database from Habitat exploration.")
    parser.add_argument("--scene_path", type=str, default=SCENE_PATH, help="Path to Habitat scene .glb")
    parser.add_argument(
        "--meters_per_step",
        type=float,
        default=1.5,
        help="Waypoint spacing in meters for full-house exploration",
    )
    parser.add_argument(
        "--max_positions",
        "--max_position",
        type=int,
        default=None,
        help="Limit number of positions (each position has len(scan_angles) orientation frames)",
    )
    parser.add_argument("--output_dir", type=str, default=SPATIAL_DB_DIR, help="Output directory")
    parser.add_argument("--vlm_model", type=str, default=SPATIAL_DB_VLM_MODEL, help="OpenAI VLM model")
    parser.add_argument(
        "--use_cache",
        type=_str_to_bool,
        default=True,
        help="Whether to cache VLM captions (true/false)",
    )
    parser.add_argument(
        "--object_max_per_frame",
        type=int,
        default=OBJECT_MAX_PER_FRAME,
        help="Max extracted objects per frame",
    )
    parser.add_argument(
        "--object_parse_retries",
        type=int,
        default=OBJECT_PARSE_RETRIES,
        help="Retries after object JSON parse failure",
    )
    parser.add_argument(
        "--object_use_cache",
        type=_str_to_bool,
        default=OBJECT_USE_CACHE,
        help="Whether to cache VLM object outputs (true/false)",
    )
    parser.add_argument(
        "--object_cache_dir",
        type=str,
        default=None,
        help="Object cache directory (default: <output_dir>/vlm_object_cache)",
    )
    parser.add_argument(
        "--tour_mode",
        type=str,
        default="full_house",
        choices=["full_house", "random"],
        help="Exploration mode for DB creation",
    )
    parser.add_argument(
        "--random_num_steps",
        type=int,
        default=50,
        help="Number of move steps when --tour_mode random",
    )
    parser.add_argument(
        "--random_step_size",
        type=float,
        default=1.0,
        help="Step size in meters when --tour_mode random",
    )
    parser.add_argument(
        "--scan_angles",
        "--random_scan_angles",
        type=_parse_scan_angles,
        default=SCAN_ANGLES,
        help="Comma-separated scan angles for both full_house and random tours, e.g. '0,30,60,...,330'",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for random tour",
    )
    parser.add_argument(
        "--random_max_attempts_per_step",
        type=int,
        default=32,
        help="Max attempts per step for random tour",
    )
    parser.add_argument(
        "--random_include_start_scan",
        type=_str_to_bool,
        default=True,
        help="Whether to capture scan at start position in random tour (true/false)",
    )
    parser.add_argument(
        "--run_polar_surrounding_postprocess",
        type=_str_to_bool,
        default=False,
        help="Whether to rebuild polar surrounding context after DB build (true/false)",
    )
    parser.add_argument(
        "--builder_variant",
        type=str,
        default="standard",
        choices=["standard", "angle_split"],
        help="Database builder variant to run",
    )
    parser.add_argument(
        "--angle_split_enable",
        type=_str_to_bool,
        default=VLM_ANGLE_SPLIT_ENABLE,
        help="Whether to offset object orientation by left/center/right angle buckets in angle-split mode",
    )
    parser.add_argument(
        "--angle_step",
        type=int,
        default=VLM_ANGLE_STEP,
        help="Angle offset in degrees for left/right buckets in angle-split mode",
    )
    parser.add_argument(
        "--execution_mode",
        type=str,
        default="capture_then_parallel_vlm",
        choices=["capture_then_parallel_vlm", "legacy_per_frame"],
        help="Builder execution mode",
    )
    parser.add_argument(
        "--vlm_max_in_flight",
        type=int,
        default=4,
        help="Maximum number of concurrent VLM requests in staged execution mode",
    )
    parser.add_argument(
        "--legacy_per_frame",
        type=_str_to_bool,
        default=False,
        help="Force legacy per-frame execution even if execution_mode is capture_then_parallel_vlm",
    )
    parser.add_argument(
        "--bbox_conf_threshold",
        type=float,
        default=BBOX_CONF_THRESHOLD,
        help="Minimum YOLO bbox confidence required for a detection to enter the geometry pipeline.",
    )
    parser.add_argument(
        "--enable_dinov3_embedding",
        type=_str_to_bool,
        default=ENABLE_DINOV3_EMBEDDING,
        help="Whether to encode DINOv3 embeddings for YOLO-detected object crops.",
    )
    parser.add_argument(
        "--store_dinov3_embedding",
        type=_str_to_bool,
        default=STORE_DINOV3_EMBEDDING,
        help="Whether to persist DINOv3 embeddings into object_dinov3_emb.npy sidecar.",
    )
    parser.add_argument(
        "--dinov3_model_name",
        type=str,
        default=DINOV3_MODEL_NAME,
        help="Hugging Face model name for DINOv3 crop encoding.",
    )
    parser.add_argument(
        "--dinov3_batch_size",
        type=int,
        default=DINOV3_BATCH_SIZE,
        help="Batch size used by the DINOv3 crop encoder.",
    )
    parser.add_argument(
        "--dinov3_normalize",
        type=_str_to_bool,
        default=DINOV3_NORMALIZE,
        help="Whether to L2-normalize stored DINOv3 embeddings.",
    )
    parser.add_argument(
        "--occlusion_reweight_w1",
        type=float,
        default=OCCLUSION_REWEIGHT_W1,
        help="Weight for logit(detector_confidence) in the stored occlusion reweight score.",
    )
    parser.add_argument(
        "--occlusion_reweight_w2",
        type=float,
        default=OCCLUSION_REWEIGHT_W2,
        help="Weight for occlusion penalty p(o) in the stored occlusion reweight score.",
    )
    parser.add_argument(
        "--occlusion_reweight_b",
        type=float,
        default=OCCLUSION_REWEIGHT_B,
        help="Bias term for the stored occlusion reweight score.",
    )
    parser.add_argument(
        "--occlusion_source",
        type=str,
        choices=["visible_mask", "vlm"],
        default=str(OCCLUSION_SOURCE),
        help="Source of occlusion labels used for object reweighting. `visible_mask` is the deterministic bbox-overlap+depth path.",
    )
    parser.add_argument(
        "--occlusion_target_overlap_threshold",
        type=float,
        default=OCCLUSION_TARGET_OVERLAP_THRESHOLD,
        help="Minimum intersection_area / target_bbox_area required for another bbox to become an occlusion candidate.",
    )
    parser.add_argument(
        "--visible_occ_boundary_width",
        type=int,
        default=VISIBLE_OCC_BOUNDARY_WIDTH,
        help="Deprecated compatibility flag; no-op for deterministic bbox-overlap occlusion.",
    )
    parser.add_argument(
        "--visible_occ_ring_radius",
        type=int,
        default=VISIBLE_OCC_RING_RADIUS,
        help="Deprecated compatibility flag; no-op for deterministic bbox-overlap occlusion.",
    )
    parser.add_argument(
        "--visible_occ_depth_margin_delta",
        type=float,
        default=VISIBLE_OCC_DEPTH_MARGIN_DELTA,
        help="Depth margin in meters required for another overlapping bbox to count as a nearer occluder.",
    )
    parser.add_argument(
        "--visible_occ_boundary_neighbor_radius",
        type=int,
        default=VISIBLE_OCC_BOUNDARY_NEIGHBOR_RADIUS,
        help="Deprecated compatibility flag; no-op for deterministic bbox-overlap occlusion.",
    )
    parser.add_argument(
        "--r_threshold",
        type=float,
        default=None,
        help="If set, drop geometry-derived objects whose reweighted_detection_score_r is strictly below this threshold.",
    )
    parser.add_argument(
        "--export_object_crops_by_global_id_dir",
        type=str,
        default=None,
        help="If set, export one crop image per final object into this directory, named by object_global_id.",
    )
    args = parser.parse_args()

    common_kwargs = dict(
        scene_path=args.scene_path,
        meters_per_step=args.meters_per_step,
        max_positions=args.max_positions,
        output_dir=args.output_dir,
        vlm_model=args.vlm_model,
        use_cache=args.use_cache,
        object_max_per_frame=args.object_max_per_frame,
        object_parse_retries=args.object_parse_retries,
        object_use_cache=args.object_use_cache,
        object_cache_dir=args.object_cache_dir,
        tour_mode=args.tour_mode,
        random_num_steps=args.random_num_steps,
        random_step_size=args.random_step_size,
        random_scan_angles=args.scan_angles,
        random_seed=args.random_seed,
        random_max_attempts_per_step=args.random_max_attempts_per_step,
        random_include_start_scan=args.random_include_start_scan,
        run_polar_surrounding_postprocess=args.run_polar_surrounding_postprocess,
        execution_mode=args.execution_mode,
        vlm_max_in_flight=args.vlm_max_in_flight,
        legacy_per_frame=args.legacy_per_frame,
        bbox_conf_threshold=args.bbox_conf_threshold,
        enable_dinov3_embedding=args.enable_dinov3_embedding,
        store_dinov3_embedding=args.store_dinov3_embedding,
        dinov3_model_name=args.dinov3_model_name,
        dinov3_batch_size=args.dinov3_batch_size,
        dinov3_normalize=args.dinov3_normalize,
        occlusion_reweight_w1=args.occlusion_reweight_w1,
        occlusion_reweight_w2=args.occlusion_reweight_w2,
        occlusion_reweight_b=args.occlusion_reweight_b,
        occlusion_source=args.occlusion_source,
        occlusion_target_overlap_threshold=args.occlusion_target_overlap_threshold,
        visible_occ_boundary_width=args.visible_occ_boundary_width,
        visible_occ_ring_radius=args.visible_occ_ring_radius,
        visible_occ_depth_margin_delta=args.visible_occ_depth_margin_delta,
        visible_occ_boundary_neighbor_radius=args.visible_occ_boundary_neighbor_radius,
        r_threshold=args.r_threshold,
        export_object_crops_by_global_id_dir=args.export_object_crops_by_global_id_dir,
    )
    if args.builder_variant == "angle_split":
        report = build_spatial_database_angle_split(
            **common_kwargs,
            angle_split_enable=args.angle_split_enable,
            angle_step=args.angle_step,
        )
    else:
        report = build_spatial_database(**common_kwargs)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
