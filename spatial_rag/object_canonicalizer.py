from typing import Any, List, Optional

from spatial_rag.config import OBJECT_SURROUNDING_MAX
from spatial_rag.object_schema import SceneObjects, SurroundingObject, VisualFeature

UNKNOWN_TEXT_TOKEN = "unknown"

EMPTY_OBJECT_SENTINEL = (
    "feature=none;type=unknown;lat=center;dist=far;vert=middle;support=unknown;"
    "dcam=na;bearing=na;gx=na;gz=na;attrs=none;ctx=none;text=;desc=none"
)


def _sanitize_text(value: str, max_chars: int = 180) -> str:
    text = str(value or "").strip().replace("\n", " ").replace(";", ",").replace("|", "/")
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _laterality_rank(value: str) -> int:
    order = {"left": 0, "center": 1, "right": 2}
    return order.get(str(value), 9)


def _distance_rank(value: str) -> int:
    order = {"near": 0, "middle": 1, "far": 2}
    return order.get(str(value), 9)


def _verticality_rank(value: str) -> int:
    order = {"high": 0, "middle": 1, "low": 2}
    return order.get(str(value), 9)


def _normalize_embed_text(value: str) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return " ".join(text.split())


def _round_coord(value: Optional[float]) -> str:
    if value is None:
        return "na"
    rounded = round(float(value) * 2.0) / 2.0
    return f"{rounded:.1f}"


def _round_dist(value: Optional[float]) -> str:
    if value is None:
        return "na"
    return f"{round(float(value), 1):.1f}"


def _clean_items(items: List[str], limit: int) -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text.lower() == "unknown":
            continue
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _top_attributes(attrs: List[str], limit: int = 4) -> List[str]:
    return _clean_items(list(attrs or []), limit=limit)


def _ctx_sort_key(ctx: SurroundingObject) -> tuple:
    primary_dist = ctx.distance_from_primary_m
    return (
        1 if primary_dist is None else 0,
        float(primary_dist or 0.0),
        str(ctx.label or ""),
    )


def _top_surroundings(ctx_list: List[SurroundingObject], limit: int) -> List[SurroundingObject]:
    items = [ctx for ctx in list(ctx_list or []) if str(ctx.label or "").strip()]
    items.sort(key=_ctx_sort_key)
    return items[: max(0, int(limit))]


def _format_anchor(x: Optional[float], z: Optional[float]) -> str:
    return f"x={_round_coord(x)}, z={_round_coord(z)}"


def _format_nearby_item(ctx: SurroundingObject) -> str:
    return f"{ctx.label}@({_round_coord(ctx.estimated_global_x)},{_round_coord(ctx.estimated_global_z)})"


def _format_nearby_list(ctx_list: List[SurroundingObject], limit: int) -> str:
    items = _top_surroundings(ctx_list, limit=limit)
    if not items:
        return "none"
    return ", ".join(_format_nearby_item(item) for item in items)


def _format_long_surroundings(ctx_list: List[SurroundingObject], limit: int = OBJECT_SURROUNDING_MAX) -> str:
    items = _top_surroundings(ctx_list, limit=limit)
    if not items:
        return "none"
    rendered: List[str] = []
    for item in items:
        rendered.append(
            f"{item.label} | relation={str(item.relation_to_primary or '').strip() or 'unknown'} "
            f"| primary_dist={_round_dist(item.distance_from_primary_m)} "
            f"| global=({_round_coord(item.estimated_global_x)},{_round_coord(item.estimated_global_z)})"
        )
    return "; ".join(rendered)


def _scene_context_text(scene_objects: Optional[SceneObjects]) -> str:
    if scene_objects is None:
        return "floor_pattern=unknown; scene_attributes=none; wall_color=unknown"
    scene_attrs = _clean_items(list(scene_objects.scene_attributes or []), limit=6)
    return (
        f"floor_pattern={scene_objects.floor_pattern}; "
        f"scene_attributes={', '.join(scene_attrs) if scene_attrs else 'none'}; "
        f"wall_color={str(scene_objects.wall_color or 'unknown').strip() or 'unknown'}"
    )


def _compose_short_object_text(obj: VisualFeature) -> str:
    attrs = ", ".join(_top_attributes(obj.attributes, limit=4)) or "none"
    nearby = _format_nearby_list(obj.surrounding_context, limit=2)
    return (
        f"object: {obj.type} | "
        f"attrs: {attrs} | "
        f"anchor: {_format_anchor(obj.estimated_global_x, obj.estimated_global_z)} | "
        f"nearby: {nearby}"
    )


def _compose_long_object_text(obj: VisualFeature, scene_objects: Optional[SceneObjects] = None) -> str:
    attrs = ", ".join(_top_attributes(obj.attributes, limit=6)) or "none"
    return (
        f"object: {obj.type} | "
        f"attributes: {attrs} | "
        f"camera_relation: distance={_round_dist(obj.distance_from_camera_m)}, "
        f"bearing={_round_dist(obj.relative_bearing_deg)}, "
        f"laterality={obj.relative_position_laterality}, "
        f"verticality={obj.relative_position_verticality} | "
        f"global_anchor: {_format_anchor(obj.estimated_global_x, obj.estimated_global_z)} | "
        f"surroundings: {_format_long_surroundings(obj.surrounding_context, limit=int(OBJECT_SURROUNDING_MAX))} | "
        f"scene_context: {_scene_context_text(scene_objects)}"
    )


def select_object_text(obj: VisualFeature, mode: str = "short", scene_objects: Optional[SceneObjects] = None) -> str:
    selected_mode = str(mode or "short").strip().lower()
    if selected_mode == "short":
        return _normalize_embed_text(_compose_short_object_text(obj))
    if selected_mode == "long":
        return _normalize_embed_text(_compose_long_object_text(obj, scene_objects=scene_objects))
    raise ValueError(f"Unsupported object_text_mode: {mode}")


def _select_object_desc(obj: VisualFeature, object_text_mode: str = "short", scene_objects: Optional[SceneObjects] = None) -> str:
    mode = str(object_text_mode or "short").strip().lower()
    if mode == "short":
        return select_object_text(obj, mode="short", scene_objects=scene_objects)
    if mode == "long":
        return select_object_text(obj, mode="long", scene_objects=scene_objects)
    raise ValueError(f"Unsupported object_text_mode: {object_text_mode}")


def sorted_objects(scene_objects: SceneObjects, max_objects: int = 24) -> List[VisualFeature]:
    items = list(scene_objects.visual_feature)
    items.sort(
        key=lambda obj: (
            _laterality_rank(obj.relative_position_laterality),
            _distance_rank(obj.relative_position_distance),
            _verticality_rank(obj.relative_position_verticality),
            obj.type,
            obj.feature_id,
        )
    )
    if max_objects > 0:
        items = items[:max_objects]
    return items


def collect_object_texts(
    scene_objects: SceneObjects,
    max_objects: int = 24,
    mode: str = "short",
) -> List[str]:
    texts: List[str] = []
    for obj in sorted_objects(scene_objects, max_objects=max_objects):
        text = select_object_text(obj, mode=mode, scene_objects=scene_objects)
        if text:
            texts.append(text)
    if not texts:
        return [UNKNOWN_TEXT_TOKEN]
    return texts


def compose_frame_text(
    scene_objects: SceneObjects,
    max_objects: int = 24,
    mode: str = "short",
    joiner: str = " | ",
) -> str:
    return str(joiner).join(collect_object_texts(scene_objects, max_objects=max_objects, mode=mode))


def _canonical_attrs(obj: VisualFeature) -> str:
    attrs = _top_attributes(obj.attributes, limit=6)
    return ",".join(_sanitize_text(attr, max_chars=40) for attr in attrs) if attrs else "none"


def _canonical_ctx(obj: VisualFeature) -> str:
    items = _top_surroundings(obj.surrounding_context, limit=int(OBJECT_SURROUNDING_MAX))
    if not items:
        return "none"
    rendered: List[str] = []
    for item in items:
        relation = _sanitize_text(str(item.relation_to_primary or "").strip() or "unknown", max_chars=40)
        rendered.append(
            f"{_sanitize_text(item.label, max_chars=40)}@{_round_coord(item.estimated_global_x)},{_round_coord(item.estimated_global_z)}"
            f"~{_round_dist(item.distance_from_primary_m)}#{relation}"
        )
    return ",".join(rendered)


def canonical_object_line(obj: VisualFeature, object_text_mode: str = "short") -> str:
    dcam = "na" if obj.distance_from_camera_m is None else f"{float(obj.distance_from_camera_m):.2f}"
    bearing = "na" if obj.relative_bearing_deg is None else f"{float(obj.relative_bearing_deg):.1f}"
    desc_text = _select_object_desc(obj, object_text_mode=object_text_mode)
    return (
        f"feature={obj.feature_id};"
        f"type={obj.type};"
        f"lat={obj.relative_position_laterality};"
        f"dist={obj.relative_position_distance};"
        f"vert={obj.relative_position_verticality};"
        f"support={obj.support_relation};"
        f"dcam={dcam};"
        f"bearing={bearing};"
        f"gx={_round_coord(obj.estimated_global_x)};"
        f"gz={_round_coord(obj.estimated_global_z)};"
        f"attrs={_canonical_attrs(obj)};"
        f"ctx={_canonical_ctx(obj)};"
        f"text={_sanitize_text(obj.any_text, max_chars=120)};"
        f"desc={_sanitize_text(desc_text, max_chars=220)}"
    )


def canonicalize_scene_objects(
    scene_objects: SceneObjects,
    max_objects: int = 24,
    object_text_mode: str = "short",
) -> List[str]:
    objs = sorted_objects(scene_objects, max_objects=max_objects)
    if not objs:
        return [EMPTY_OBJECT_SENTINEL]
    return [canonical_object_line(obj, object_text_mode=object_text_mode) for obj in objs]


def canonical_scene_text(
    scene_objects: SceneObjects,
    max_objects: int = 24,
    object_text_mode: str = "short",
) -> str:
    scene_attrs = _clean_items(list(scene_objects.scene_attributes or []), limit=8)
    scene_prefix = (
        f"view_type={scene_objects.view_type};"
        f"room_function={scene_objects.room_function};"
        f"style_hint={scene_objects.style_hint};"
        f"clutter_level={scene_objects.clutter_level};"
        f"floor_pattern={scene_objects.floor_pattern};"
        f"lighting={scene_objects.lighting_ceiling};"
        f"scene_attributes={_sanitize_text(', '.join(scene_attrs) if scene_attrs else 'none', max_chars=120)};"
        f"wall_color={_sanitize_text(scene_objects.wall_color, max_chars=80)};"
        f"notes={_sanitize_text(scene_objects.additional_notes, max_chars=160)};"
        f"summary={_sanitize_text(scene_objects.image_summary, max_chars=220)}"
    )
    return scene_prefix + " | " + " | ".join(
        canonicalize_scene_objects(
            scene_objects,
            max_objects=max_objects,
            object_text_mode=object_text_mode,
        )
    )
