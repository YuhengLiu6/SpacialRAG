import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from spatial_rag.config import (
    DEPTH_PRO_MODEL_PATH,
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
    OBJECT_SURROUNDING_MAX,
    OBJECT_USE_CACHE,
    OBJECT_VERTICAL_REL_EPS_M,
    SAVE_GEOMETRY_ARTIFACTS,
    SCAN_ANGLES,
    SCENE_PATH,
    SPATIAL_DB_DIR,
    SPATIAL_DB_VLM_MODEL,
    VLM_ANGLE_SPLIT_ENABLE,
    VLM_ANGLE_STEP,
)
from spatial_rag.object_canonicalizer import (
    UNKNOWN_TEXT_TOKEN,
    compose_frame_text,
    select_object_text,
    sorted_objects,
)
from spatial_rag.object_geometry_pipeline import ObjectGeometryPipeline
from spatial_rag.object_parser import ParseResult, parse_scene_objects
from spatial_rag.vlm_captioner import VLMCaptioner


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


def _serialize_floor_plan_projection(projection: Any) -> Optional[Dict[str, float]]:
    if not isinstance(projection, dict):
        return None
    required = ("view_min_x", "view_max_x", "view_min_z", "view_max_z")
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

    object_groups_by_entry_id: Dict[int, List[Tuple[Dict, np.ndarray, np.ndarray]]] = {}
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
                object_groups_by_entry_id.setdefault(entry_id, []).append(
                    (
                        dict(row),
                        object_arr_short[idx].astype("float32"),
                        object_arr_long[idx].astype("float32"),
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
    existing_object_group: Optional[Sequence[Tuple[Dict, np.ndarray, np.ndarray]]],
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


def _project_global_xz(
    origin_x: float,
    origin_z: float,
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
    projected_z = float(origin_z - math.cos(yaw) * dist)
    return projected_x, projected_z


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
            f"anchor=({_round_location_coord(item.get('estimated_global_x'))},{_round_location_coord(item.get('estimated_global_z'))})"
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
    dz: float,
    view_orientation_deg: float,
    same_axis_eps: float = 0.25,
) -> Optional[str]:
    distance = float(math.hypot(float(dx), float(dz)))
    if distance <= float(same_axis_eps):
        return None

    yaw = math.radians(float(view_orientation_deg))
    forward_x = -math.sin(yaw)
    forward_z = -math.cos(yaw)
    right_x = math.cos(yaw)
    right_z = -math.sin(yaw)

    local_forward = float(dx) * forward_x + float(dz) * forward_z
    local_right = float(dx) * right_x + float(dz) * right_z
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


def _entry_camera_y(entry: Dict[str, Any]) -> float:
    world_position = entry.get("world_position")
    if isinstance(world_position, (list, tuple)) and len(world_position) >= 2:
        y = _safe_float(world_position[1])
        if y is not None:
            return float(y)
    return 0.0


def _estimated_global_y(camera_y: float, relative_height_from_camera_m: Any) -> Optional[float]:
    relative_height = _safe_float(relative_height_from_camera_m)
    if relative_height is None:
        return None
    return float(camera_y + relative_height)


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
        object_z = _safe_float(row.get("estimated_global_z"))
        if object_x is None or object_z is None:
            continue

        view_x = _safe_float(entry.get("x"))
        view_z = _safe_float(entry.get("y"))
        if view_x is None or view_z is None:
            continue
        view_y = _entry_camera_y(entry)
        object_y = _safe_float(row.get("estimated_global_y"))
        dx = float(object_x - view_x)
        dy = float(object_y - view_y) if object_y is not None else 0.0
        dz = float(object_z - view_z)
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
                "distance_m": float(math.hypot(dx, dz)),
                "distance_3d_m": float(math.sqrt(dx * dx + dy * dy + dz * dz)),
                "direction": "in",
                "direction_frame": "view_aligned",
                "vertical_direction": _classify_vertical_direction(dy=dy),
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
        object_z = _safe_float(row.get("estimated_global_z"))
        if object_x is None or object_z is None:
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
            source_z = _safe_float(source.get("estimated_global_z"))
            if source_x is None or source_z is None:
                continue
            source_y = _safe_float(source.get("estimated_global_y"))
            for target in ordered:
                if int(source["object_global_id"]) == int(target["object_global_id"]):
                    continue
                target_x = _safe_float(target.get("estimated_global_x"))
                target_z = _safe_float(target.get("estimated_global_z"))
                if target_x is None or target_z is None:
                    continue
                target_y = _safe_float(target.get("estimated_global_y"))
                dx = float(target_x - source_x)
                dy = float(target_y - source_y) if source_y is not None and target_y is not None else 0.0
                dz = float(target_z - source_z)
                direction = _classify_view_aligned_direction(
                    dx=dx,
                    dz=dz,
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
                        "distance_m": float(math.hypot(dx, dz)),
                        "distance_3d_m": float(math.sqrt(dx * dx + dy * dy + dz * dz)),
                        "direction": direction,
                        "direction_frame": "view_aligned",
                        "vertical_direction": _classify_vertical_direction(dy=dy),
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
        projected_x, projected_z = _project_global_xz(
            origin_x=camera_x,
            origin_z=camera_z,
            camera_orientation_deg=camera_orientation_deg,
            relative_bearing_deg=bearing,
            distance_m=getattr(obj, "distance_from_camera_m", None),
        )
        obj.estimated_global_x = projected_x
        obj.estimated_global_y = _estimated_global_y(
            camera_y=camera_y,
            relative_height_from_camera_m=getattr(obj, "relative_height_from_camera_m", None),
        )
        obj.estimated_global_z = projected_z

        enriched_ctx = []
        for ctx in list(getattr(obj, "surrounding_context", []) or [])[: int(OBJECT_SURROUNDING_MAX)]:
            ctx_bearing = _normalize_bearing_deg(getattr(ctx, "relative_bearing_deg", None))
            ctx.relative_bearing_deg = ctx_bearing
            ctx_x, ctx_z = _project_global_xz(
                origin_x=camera_x,
                origin_z=camera_z,
                camera_orientation_deg=camera_orientation_deg,
                relative_bearing_deg=ctx_bearing,
                distance_m=getattr(ctx, "distance_from_camera_m", None),
            )
            ctx.estimated_global_x = ctx_x
            ctx.estimated_global_y = _estimated_global_y(
                camera_y=camera_y,
                relative_height_from_camera_m=getattr(ctx, "relative_height_from_camera_m", None),
            )
            ctx.estimated_global_z = ctx_z
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
    detector_confidence: Optional[float] = None,
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
        "detector_label": detector_label,
        "detector_confidence": detector_confidence,
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
        "crop_vlm_label": crop_vlm_label,
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
) -> Dict:
    try:
        from spatial_rag.embedder import Embedder
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import Embedder dependencies. "
            "Ensure the current Python environment has torch/open_clip installed."
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

    output_root = Path(output_dir)
    images_dir = output_root / "images"
    cache_dir = output_root / "vlm_cache"
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
            "stored_text_modes": ["short", "long"],
            "parse_retries": int(object_parse_retries),
            "use_cache": bool(object_use_cache),
            "cache_dir": str(object_cache_root),
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
        "total_frames_raw": 0,
        "total_frames_processed": 0,
        "total_entries": 0,
        "failed_entries": 0,
        "image_index_ntotal": 0,
        "text_index_ntotal_short": 0,
        "text_index_ntotal_long": 0,
        "object_index_ntotal_short": 0,
        "object_index_ntotal_long": 0,
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
    object_groups_by_entry_id: Dict[int, List[Tuple[Dict, np.ndarray, np.ndarray]]] = {
        int(entry_id): list(groups)
        for entry_id, groups in dict(resume_state["object_groups_by_entry_id"]).items()
    }
    file_name_to_entry_id: Dict[str, int] = dict(resume_state["file_name_to_entry_id"])
    failures: List[Dict] = []
    timing_records: List[Dict] = []

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

        frame_iter = tqdm(
            zip(frames, poses),
            total=len(frames),
            desc="Building spatial DB",
        )

        valid_orientation_set = set(normalized_scan_angles)

        for frame_idx, (rgb_image, pose) in enumerate(frame_iter):
            frame_t0 = time.perf_counter()
            try:
                pos = np.asarray(pose["position"], dtype=np.float32).reshape(-1)
                if pos.shape[0] != 3:
                    raise ValueError(f"Position must be length 3, got {pos.tolist()}")

                world_position = [float(pos[0]), float(pos[1]), float(pos[2])]
                x = float(world_position[0])
                y = float(world_position[2])
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
                    f"camera_xyz=({x:.3f},{float(world_position[1]):.3f},{y:.3f})"
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
                    "camera_z": float(y),
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
                        camera_y=float(world_position[1]),
                        camera_z=float(y),
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

                if not geometry_object_rows:
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
                            camera_y=float(world_position[1]),
                            camera_z=float(y),
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
                if not np.isclose(x, world_position[0]) or not np.isclose(y, world_position[2]):
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
                    "geometry_pipeline_used": bool(geometry_object_rows),
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
                entry_object_records: List[Tuple[Dict, np.ndarray, np.ndarray]] = []

                if geometry_object_rows:
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
                            detector_label=geo_row.get("detector_label"),
                            detector_confidence=geo_row.get("detector_confidence"),
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
                        )
                        record["view_type"] = str(view_attribute.get("view_type") or "unknown")
                        record["room_function"] = str(view_attribute.get("room_function") or "unknown")
                        record["style_hint"] = str(view_attribute.get("style_hint") or "unknown")
                        record["clutter_level"] = str(view_attribute.get("clutter_level") or "unknown")
                        entry_object_records.append((record, obj_emb_short, obj_emb_long))
                        bucket_key = f"total_{record['angle_bucket']}_bucket_objects"
                        report[bucket_key] = int(report.get(bucket_key, 0)) + 1
                        entry_object_count += 1
                elif scene_objects is not None and object_text_pairs:
                    for obj, line_short, raw_long, line_long in object_text_pairs:
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
                            detector_confidence=None,
                            vlm_distance_from_camera_m=obj.distance_from_camera_m,
                            vlm_relative_bearing_deg=obj.relative_bearing_deg,
                        )
                        entry_object_records.append((record, obj_emb_short, obj_emb_long))
                        bucket_key = f"total_{record['angle_bucket']}_bucket_objects"
                        report[bucket_key] = int(report.get(bucket_key, 0)) + 1
                        entry_object_count += 1
                else:
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
                    )
                    entry_object_records.append((record, obj_emb_short, obj_emb_long))
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
                timing_records.append(
                    {
                        "frame_idx": int(frame_idx),
                        "entry_id": int(entry_id),
                        "file_name": file_name,
                        "route": "mask_depth" if geometry_object_rows else "vlm_fallback",
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
                    f"route={'mask_depth' if geometry_object_rows else 'vlm_fallback'} "
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
        for entry_id, _meta in enumerate(metadata_records):
            for record, obj_emb_short, obj_emb_long in list(object_groups_by_entry_id.get(entry_id, [])):
                out_record = dict(record)
                out_record["entry_id"] = int(entry_id)
                out_record["object_global_id"] = int(len(object_metadata_records))
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
    args = parser.parse_args()

    report = build_spatial_database(
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
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
