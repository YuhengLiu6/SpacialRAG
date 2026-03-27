import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from spatial_rag.config import OBJECT_SURROUNDING_MAX
from spatial_rag.household_taxonomy import canonicalize_household_object_label, household_label_enum_values
from spatial_rag.object_schema import SceneObjects


ParseStatus = Literal["ok", "fallback", "failed"]


@dataclass
class ParseResult:
    scene_objects: Optional[SceneObjects]
    parse_status: ParseStatus
    warnings: List[str]
    raw_vlm_output: str
    raw_api_response: Optional[Dict[str, Any]] = None
    raw_api_source: Optional[str] = None


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = raw_text.strip()
    if not text:
        return None

    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        loaded = json.loads(text[start : end + 1])
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return None
    return None


def _normalize_attributes(raw_attrs: Any) -> List[str]:
    if isinstance(raw_attrs, list):
        return [str(x).strip() for x in raw_attrs if str(x).strip()]
    if isinstance(raw_attrs, str):
        parts = [p.strip() for p in raw_attrs.split(",")]
        return [p for p in parts if p]
    return []


_VIEW_TYPE_ALLOWED = {
    "living room",
    "bedroom",
    "kitchen",
    "bathroom",
    "dining room",
    "hallway",
    "entryway",
    "balcony",
    "laundry room",
    "staircase",
    "study",
    "utility room",
    "unknown",
    "other",
}
_ROOM_FUNCTION_ALLOWED = {
    "resting",
    "cooking",
    "dining",
    "bathing",
    "working",
    "storage",
    "circulation",
    "mixed",
    "unknown",
}
_STYLE_HINT_ALLOWED = {
    "modern",
    "minimalist",
    "traditional",
    "rustic",
    "industrial",
    "scandinavian",
    "eclectic",
    "unknown",
    "other",
}
_CLUTTER_LEVEL_ALLOWED = {"low", "medium", "high", "unknown"}
_FEATURE_TYPE_ALLOWED = set(household_label_enum_values(include_unknown=True, include_other=True))
_BACKGROUND_FEATURE_TYPES = {"wall feature", "floor pattern"}
_LATERALITY_ALLOWED = {"left", "right", "center"}
_DISTANCE_ALLOWED = {"near", "middle", "far"}
_VERTICALITY_ALLOWED = {"high", "middle", "low"}
_SUPPORT_RELATION_ALLOWED = {"on", "under", "inside", "hanging_on", "attached_to", "freestanding", "unknown"}
_FLOOR_PATTERN_ALLOWED = {"wood", "laminate", "tile", "carpet", "rug", "stone", "concrete", "unknown", "other"}
_LIGHTING_ALLOWED = {
    "recessed lights",
    "chandelier",
    "pendant lights",
    "ceiling lamp",
    "ceiling fan light",
    "natural light source",
    "mixed lighting",
    "unknown",
    "other",
}


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _normalize_choice(value: Any, allowed: set, default: str, aliases: Optional[Dict[str, str]] = None) -> str:
    raw = _norm_token(value)
    if aliases and raw in aliases:
        raw = aliases[raw]
    if raw in allowed:
        return raw
    return default


def _normalize_feature_type(value: Any) -> str:
    token = _canonical_feature_token(value)
    if token in _BACKGROUND_FEATURE_TYPES:
        return token
    return canonicalize_household_object_label(token, default="other")


def _canonical_feature_token(value: Any) -> str:
    token = _norm_token(value)
    alias = {
        "wall_feature": "wall feature",
        "floor_pattern": "floor pattern",
    }
    return alias.get(token, token)


def _normalize_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _normalize_scene_attributes(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        return [text]
    return []


def _scene_attribute_from_background_feature(
    feature_type: str,
    description: str,
    attributes: List[str],
    long_form_open_description: str,
) -> Optional[str]:
    prefix = "wall detail" if feature_type == "wall feature" else "floor detail"
    parts: List[str] = []
    desc = _clean_text(description, default="")
    if desc and desc != "unknown":
        parts.append(desc)
    if attributes:
        parts.append(", ".join(attributes[:4]))
    long_text = _clean_text(long_form_open_description, default="")
    if long_text and long_text != "unknown":
        parts.append(long_text)
    if not parts:
        return None
    return f"{prefix}: {' | '.join(parts[:2])}"


def _normalize_surrounding_context(raw_value: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_value, list):
        return []
    out: List[Dict[str, Any]] = []
    for raw_item in raw_value[: int(OBJECT_SURROUNDING_MAX)]:
        if not isinstance(raw_item, dict):
            continue
        out.append(
            {
                "label": _clean_text(raw_item.get("label"), default="unknown"),
                "attributes": _normalize_attributes(raw_item.get("attributes", [])),
                "distance_from_primary_m": _normalize_float_or_none(raw_item.get("distance_from_primary_m")),
                "distance_from_camera_m": _normalize_float_or_none(raw_item.get("distance_from_camera_m")),
                "relative_height_from_camera_m": _normalize_float_or_none(raw_item.get("relative_height_from_camera_m")),
                "relative_bearing_deg": _normalize_float_or_none(raw_item.get("relative_bearing_deg")),
                "estimated_global_x": _normalize_float_or_none(raw_item.get("estimated_global_x")),
                "estimated_global_y": _normalize_float_or_none(raw_item.get("estimated_global_y")),
                "estimated_global_z": _normalize_float_or_none(raw_item.get("estimated_global_z")),
                "relation_to_primary": _clean_text(raw_item.get("relation_to_primary"), default=""),
            }
        )
    return out


def _normalize_bbox(raw_bbox: Any, warnings: List[str], idx: int) -> List[float]:
    if isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
        return [_clip01(_to_float(raw_bbox[i], 0.0)) for i in range(4)]
    warnings.append(f"legacy object[{idx}] invalid bbox; using zeros")
    return [0.0, 0.0, 0.0, 0.0]


def _infer_relative_bins_from_bbox(bbox_xywh_norm: List[float]) -> Dict[str, str]:
    x = float(bbox_xywh_norm[0])
    y = float(bbox_xywh_norm[1])
    w = float(bbox_xywh_norm[2])
    h = float(bbox_xywh_norm[3])
    cx = _clip01(x + 0.5 * w)
    cy = _clip01(y + 0.5 * h)
    area = _clip01(w) * _clip01(h)

    laterality = "left" if cx < 0.34 else ("center" if cx < 0.67 else "right")
    verticality = "high" if cy < 0.34 else ("middle" if cy < 0.67 else "low")
    distance = "near" if area >= 0.18 else ("middle" if area >= 0.06 else "far")

    return {
        "laterality": laterality,
        "distance": distance,
        "verticality": verticality,
    }


def _scene_payload_signature_exists(payload: Dict[str, Any]) -> bool:
    return any(
        key in payload
        for key in {
            "visual_feature",
            "visual_features",
            "view_type",
            "room_function",
            "style_hint",
            "clutter_level",
            "image_summary",
            "objects",
        }
    )


def _build_scene_from_legacy_object_v1(
    payload: Dict[str, Any],
    image_context: Optional[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    image_id = None
    if image_context and image_context.get("image_id") is not None:
        image_id = str(image_context["image_id"])
    elif payload.get("image_id") is not None:
        image_id = str(payload.get("image_id"))

    objects_raw = payload.get("objects", [])
    if not isinstance(objects_raw, list):
        warnings.append("legacy `objects` is not a list; forcing empty list")
        objects_raw = []

    features: List[Dict[str, Any]] = []
    for idx, raw_obj in enumerate(objects_raw):
        if not isinstance(raw_obj, dict):
            warnings.append(f"legacy object[{idx}] is not a dict; skipped")
            continue

        raw_meta = raw_obj.get("meta") if isinstance(raw_obj.get("meta"), dict) else {}
        raw_loc = raw_obj.get("location") if isinstance(raw_obj.get("location"), dict) else {}

        label = _clean_text(raw_meta.get("label"), default="unknown")
        attrs = _normalize_attributes(raw_meta.get("attributes", []))
        bbox = _normalize_bbox(raw_loc.get("bbox_xywh_norm"), warnings, idx)
        rel = _infer_relative_bins_from_bbox(bbox)

        desc = label if not attrs else f"{label} ({', '.join(attrs[:4])})"
        features.append(
            {
                "feature_id": f"feat_{idx:03d}",
                "type": _normalize_feature_type(label),
                "description": desc,
                "attributes": attrs,
                "relative_position_laterality": rel["laterality"],
                "relative_position_distance": rel["distance"],
                "relative_position_verticality": rel["verticality"],
                "distance_from_camera_m": None,
                "relative_height_from_camera_m": None,
                "relative_bearing_deg": None,
                "estimated_global_x": None,
                "estimated_global_y": None,
                "estimated_global_z": None,
                "support_relation": "unknown",
                "any_text": "",
                "long_form_open_description": desc,
                "location_relative_to_other_objects": "",
                "surrounding_context": [],
            }
        )

    warnings.append("legacy object_v1 payload converted to home schema")
    return {
        "image_id": image_id,
        "view_type": "unknown",
        "room_function": "unknown",
        "style_hint": "unknown",
        "clutter_level": "unknown",
        "scene_attributes": [],
        "visual_feature": features,
        "floor_pattern": "unknown",
        "lighting_ceiling": "unknown",
        "wall_color": "unknown",
        "additional_notes": "",
        "image_summary": "",
    }


def _build_scene_payload(
    payload: Dict[str, Any],
    image_context: Optional[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    if "objects" in payload and "visual_feature" not in payload and "visual_features" not in payload:
        return _build_scene_from_legacy_object_v1(
            payload=payload,
            image_context=image_context,
            warnings=warnings,
        )

    image_id = None
    if image_context and image_context.get("image_id") is not None:
        image_id = str(image_context["image_id"])
    elif payload.get("image_id") is not None:
        image_id = str(payload.get("image_id"))

    view_type = _normalize_choice(
        payload.get("view_type", "unknown"),
        allowed=_VIEW_TYPE_ALLOWED,
        default="unknown",
    )
    room_function = _normalize_choice(
        payload.get("room_function", "unknown"),
        allowed=_ROOM_FUNCTION_ALLOWED,
        default="unknown",
    )
    style_hint = _normalize_choice(
        payload.get("style_hint", "unknown"),
        allowed=_STYLE_HINT_ALLOWED,
        default="unknown",
    )
    clutter_level = _normalize_choice(
        payload.get("clutter_level", "unknown"),
        allowed=_CLUTTER_LEVEL_ALLOWED,
        default="unknown",
    )

    raw_features = payload.get("visual_feature", payload.get("visual_features", []))
    if not isinstance(raw_features, list):
        warnings.append("top-level `visual_feature` is not a list; forcing empty list")
        raw_features = []

    scene_attributes = _normalize_scene_attributes(payload.get("scene_attributes", []))
    visual_feature: List[Dict[str, Any]] = []
    for idx, raw_feat in enumerate(raw_features):
        if not isinstance(raw_feat, dict):
            warnings.append(f"visual_feature[{idx}] is not a dict; skipped")
            continue

        long_form = raw_feat.get("long_form_open_description", raw_feat.get("Long form open description", ""))
        rel_text = raw_feat.get(
            "location_relative_to_other_objects",
            raw_feat.get("Location relative to other objects in the environment", ""),
        )
        attributes = _normalize_attributes(raw_feat.get("attributes", []))
        raw_feature_type = _canonical_feature_token(raw_feat.get("type", "unknown"))

        if raw_feature_type in _BACKGROUND_FEATURE_TYPES:
            migrated = _scene_attribute_from_background_feature(
                feature_type=raw_feature_type,
                description=_clean_text(raw_feat.get("description"), default="unknown"),
                attributes=attributes,
                long_form_open_description=_clean_text(long_form, default=""),
            )
            if migrated:
                scene_attributes.append(migrated)
            warnings.append(f"visual_feature[{idx}] background feature migrated to scene_attributes")
            continue

        laterality = _normalize_choice(
            raw_feat.get("relative_position_laterality", raw_feat.get("laterality", "center")),
            allowed=_LATERALITY_ALLOWED,
            default="center",
        )
        distance = _normalize_choice(
            raw_feat.get("relative_position_distance", raw_feat.get("distance", "middle")),
            allowed=_DISTANCE_ALLOWED,
            default="middle",
        )
        verticality = _normalize_choice(
            raw_feat.get("relative_position_verticality", raw_feat.get("verticality", "middle")),
            allowed=_VERTICALITY_ALLOWED,
            default="middle",
        )
        support_relation = _normalize_choice(
            raw_feat.get("support_relation", "unknown"),
            allowed=_SUPPORT_RELATION_ALLOWED,
            default="unknown",
            aliases={"hanging on": "hanging_on", "attached to": "attached_to"},
        )

        visual_feature.append(
            {
                "feature_id": _clean_text(raw_feat.get("feature_id"), default=f"feat_{idx:03d}"),
                "type": _normalize_feature_type(raw_feature_type),
                "description": _clean_text(raw_feat.get("description"), default="unknown"),
                "attributes": attributes,
                "relative_position_laterality": laterality,
                "relative_position_distance": distance,
                "relative_position_verticality": verticality,
                "distance_from_camera_m": _normalize_float_or_none(raw_feat.get("distance_from_camera_m")),
                "relative_height_from_camera_m": _normalize_float_or_none(raw_feat.get("relative_height_from_camera_m")),
                "relative_bearing_deg": _normalize_float_or_none(raw_feat.get("relative_bearing_deg")),
                "estimated_global_x": _normalize_float_or_none(raw_feat.get("estimated_global_x")),
                "estimated_global_y": _normalize_float_or_none(raw_feat.get("estimated_global_y")),
                "estimated_global_z": _normalize_float_or_none(raw_feat.get("estimated_global_z")),
                "support_relation": support_relation,
                "any_text": _clean_text(raw_feat.get("any_text"), default=""),
                "long_form_open_description": _clean_text(long_form, default=""),
                "location_relative_to_other_objects": _clean_text(rel_text, default=""),
                "surrounding_context": _normalize_surrounding_context(raw_feat.get("surrounding_context", [])),
            }
        )

    floor_pattern = _normalize_choice(
        payload.get("floor_pattern", "unknown"),
        allowed=_FLOOR_PATTERN_ALLOWED,
        default="unknown",
    )
    lighting_ceiling = _normalize_choice(
        payload.get("lighting_ceiling", "unknown"),
        allowed=_LIGHTING_ALLOWED,
        default="unknown",
    )

    wall_color = _clean_text(payload.get("wall_color", payload.get("wall color", "unknown")), default="unknown")
    additional_notes = _clean_text(payload.get("additional_notes", ""), default="")
    image_summary = _clean_text(payload.get("image_summary", ""), default="")

    return {
        "image_id": image_id,
        "view_type": view_type,
        "room_function": room_function,
        "style_hint": style_hint,
        "clutter_level": clutter_level,
        "scene_attributes": scene_attributes,
        "visual_feature": visual_feature,
        "floor_pattern": floor_pattern,
        "lighting_ceiling": lighting_ceiling,
        "wall_color": wall_color,
        "additional_notes": additional_notes,
        "image_summary": image_summary,
    }


def _validate_scene_objects(payload: Dict[str, Any]) -> SceneObjects:
    if hasattr(SceneObjects, "model_validate"):
        return SceneObjects.model_validate(payload)  # pydantic v2
    return SceneObjects.parse_obj(payload)  # pydantic v1


def parse_scene_objects(raw_output: Any, image_context: Optional[Dict[str, Any]] = None) -> ParseResult:
    if isinstance(raw_output, dict):
        raw_text = json.dumps(raw_output, ensure_ascii=True)
        payload = raw_output
    else:
        raw_text = str(raw_output or "")
        payload = _extract_json_object(raw_text)

    if payload is None:
        return ParseResult(
            scene_objects=None,
            parse_status="failed",
            warnings=["failed to parse JSON payload from VLM output"],
            raw_vlm_output=raw_text,
        )

    if not _scene_payload_signature_exists(payload):
        return ParseResult(
            scene_objects=None,
            parse_status="failed",
            warnings=["JSON payload does not match expected scene schema keys"],
            raw_vlm_output=raw_text,
        )

    warnings: List[str] = []
    normalized = _build_scene_payload(payload, image_context=image_context, warnings=warnings)

    try:
        scene_objects = _validate_scene_objects(normalized)
    except Exception as exc:
        warnings.append(f"pydantic validation failed: {exc}")
        return ParseResult(
            scene_objects=None,
            parse_status="failed",
            warnings=warnings,
            raw_vlm_output=raw_text,
        )

    scene_objects.raw_vlm_text = raw_text
    return ParseResult(
        scene_objects=scene_objects,
        parse_status="ok",
        warnings=warnings,
        raw_vlm_output=raw_text,
    )
