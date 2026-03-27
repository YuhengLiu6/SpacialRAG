import argparse
import importlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from spatial_rag.config import OBJECT_VERTICAL_REL_EPS_M


DIRECTION_RELATIONSHIPS = ("NORTH_OF", "SOUTH_OF", "EAST_OF", "WEST_OF")
OBJECT_OBJECT_NEO4J_RELATIONSHIPS = {
    "left": "LEFT_OF",
    "right": "RIGHT_OF",
    "in front": "IN_FRONT_OF",
    "behind": "BEHIND",
}
OBJECT_OBJECT_VERTICAL_NEO4J_RELATIONSHIPS = {
    "above": "ABOVE",
    "below": "BELOW",
}
REVERSE_DIRECTION = {
    "NORTH_OF": "SOUTH_OF",
    "SOUTH_OF": "NORTH_OF",
    "EAST_OF": "WEST_OF",
    "WEST_OF": "EAST_OF",
}


@dataclass(frozen=True)
class PlaceRecord:
    place_id: str
    x: float
    y: float
    z: float
    point: Dict[str, Any]
    room_function: str
    view_type: str
    source_entry_ids: List[int]
    scan_angles: List[int]


@dataclass(frozen=True)
class ViewRecord:
    view_id: str
    place_id: str
    entry_id: int
    orientation_deg: int
    file_name: str
    parse_status: str
    frame_text_short: str
    frame_text_long: str


@dataclass(frozen=True)
class ObjectObservationRecord:
    obs_id: str
    place_id: str
    view_id: str
    entry_id: int
    object_global_id: int
    object_class: str
    label: str
    description: str
    long_form_open_description: str
    laterality: str
    distance_bin: str
    verticality: str
    distance_from_camera_m: Optional[float]
    object_orientation_deg: Optional[float]
    projected_x: Optional[float]
    projected_y: Optional[float]
    projected_z: Optional[float]
    location_relative_to_other_objects: str
    parse_status: str


@dataclass(frozen=True)
class ObjectClassRecord:
    name: str


@dataclass(frozen=True)
class PlaceObjectEdgeRecord:
    place_id: str
    obs_id: str
    view_id: str
    object_orientation_deg: Optional[float]
    distance_from_camera_m: Optional[float]
    projected_x: Optional[float]
    projected_z: Optional[float]


@dataclass(frozen=True)
class DirectionEdgeRecord:
    source_place_id: str
    target_place_id: str
    relation_type: str
    dx: float
    dz: float
    distance_m: float


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return _load_jsonl(path)


def _load_npy_if_exists(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


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


def _embedding_row_to_list(arr: Optional[np.ndarray], idx: int) -> Optional[List[float]]:
    if arr is None:
        return None
    if idx < 0 or idx >= int(arr.shape[0]):
        return None
    return [float(v) for v in np.asarray(arr[idx], dtype=np.float32).reshape(-1)]


def _obs_id_for_object_global_id(object_global_id: Any) -> str:
    return f"obs_{int(object_global_id):06d}"


def _load_build_report(db_dir: str) -> Dict[str, Any]:
    path = Path(db_dir) / "build_report.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_scan_angles(scan_angles: Iterable[int]) -> List[int]:
    return sorted({int(angle) % 360 for angle in scan_angles})


def _resolve_scan_angles(meta_rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> List[int]:
    candidates: List[int] = []
    top_level = report.get("scan_angles")
    if isinstance(top_level, list):
        candidates.extend(int(angle) for angle in top_level)
    random_cfg = report.get("random_config")
    if isinstance(random_cfg, dict) and isinstance(random_cfg.get("scan_angles"), list):
        candidates.extend(int(angle) for angle in random_cfg["scan_angles"])
    if not candidates:
        candidates.extend(int(row.get("orientation", 0)) for row in meta_rows if row.get("orientation") is not None)
    normalized = _normalize_scan_angles(candidates)
    if not normalized:
        raise ValueError("Unable to infer scan_angles from build_report.json or meta.jsonl")
    return normalized


def _safe_world_position(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    world_position = row.get("world_position")
    if isinstance(world_position, (list, tuple)) and len(world_position) == 3:
        return float(world_position[0]), float(world_position[1]), float(world_position[2])
    return float(row["x"]), 0.0, float(row["y"])


def _place_group_key(row: Mapping[str, Any], decimals: int = 4) -> Tuple[float, float, float]:
    wx, wy, wz = _safe_world_position(row)
    return (round(wx, decimals), round(wy, decimals), round(wz, decimals))


def _majority_value(rows: Sequence[Mapping[str, Any]], key: str, default: str = "unknown") -> str:
    counts = Counter(str(row.get(key) or default) for row in rows)
    if not counts:
        return default
    best_count = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == best_count)
    return winners[0]


def _project_object_position(
    origin_x: float,
    origin_z: float,
    orientation_deg: Optional[float],
    distance_m: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if orientation_deg is None or distance_m is None:
        return None, None
    dist = float(distance_m)
    if not math.isfinite(dist) or dist < 0.0:
        return None, None
    yaw = math.radians(float(orientation_deg))
    projected_x = float(origin_x - math.sin(yaw) * dist)
    projected_z = float(origin_z - math.cos(yaw) * dist)
    return projected_x, projected_z


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


def _classify_vertical_direction(dy: float, vertical_eps: float = OBJECT_VERTICAL_REL_EPS_M) -> str:
    if float(dy) > float(vertical_eps):
        return "above"
    if float(dy) < -float(vertical_eps):
        return "below"
    return "level"


def _classify_direction(dx: float, dz: float, same_axis_eps: float) -> Optional[str]:
    if abs(dx) <= float(same_axis_eps) and abs(dz) <= float(same_axis_eps):
        return None
    if abs(dz) >= abs(dx):
        return "NORTH_OF" if dz > 0.0 else "SOUTH_OF"
    return "EAST_OF" if dx > 0.0 else "WEST_OF"


def _serialize_records(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "__dataclass_fields__"):
            serialized.append(asdict(row))
        elif isinstance(row, dict):
            serialized.append(dict(row))
        else:
            raise TypeError(f"Unsupported record type for serialization: {type(row)!r}")
    return serialized


def _build_place_records_with_map(
    meta_rows: Sequence[Mapping[str, Any]],
    scan_angles: Sequence[int],
) -> Tuple[List[PlaceRecord], Dict[int, str]]:
    grouped: Dict[Tuple[float, float, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in meta_rows:
        grouped[_place_group_key(row)].append(row)

    places: List[PlaceRecord] = []
    entry_to_place: Dict[int, str] = {}
    ordered_groups = sorted(grouped.values(), key=lambda rows: min(int(row["id"]) for row in rows))

    for place_index, rows in enumerate(ordered_groups):
        wx, wy, wz = _safe_world_position(rows[0])
        place_id = f"place_{place_index:05d}"
        source_entry_ids = sorted(int(row["id"]) for row in rows)
        observed_angles = _normalize_scan_angles(int(row["orientation"]) for row in rows)
        place = PlaceRecord(
            place_id=place_id,
            x=float(wx),
            y=float(wy),
            z=float(wz),
            point={"x": float(wx), "y": float(wz), "crs": "cartesian"},
            room_function=_majority_value(rows, "room_function"),
            view_type=_majority_value(rows, "view_type"),
            source_entry_ids=source_entry_ids,
            scan_angles=observed_angles or _normalize_scan_angles(scan_angles),
        )
        places.append(place)
        for entry_id in source_entry_ids:
            entry_to_place[entry_id] = place_id

    return places, entry_to_place


def build_place_records(
    meta_rows: Sequence[Mapping[str, Any]],
    scan_angles: Sequence[int],
) -> List[PlaceRecord]:
    places, _entry_to_place = _build_place_records_with_map(meta_rows=meta_rows, scan_angles=scan_angles)
    return places


def build_view_records(
    meta_rows: Sequence[Mapping[str, Any]],
    place_map: Mapping[int, str],
) -> List[ViewRecord]:
    views: List[ViewRecord] = []
    ordered_rows = sorted(meta_rows, key=lambda row: int(row["id"]))

    for row in ordered_rows:
        entry_id = int(row["id"])
        view_id = f"view_{entry_id:05d}"
        views.append(
            ViewRecord(
                view_id=view_id,
                place_id=str(place_map[entry_id]),
                entry_id=entry_id,
                orientation_deg=int(row["orientation"]),
                file_name=str(row.get("file_name") or ""),
                parse_status=str(row.get("parse_status") or "unknown"),
                frame_text_short=str(row.get("frame_text_short") or row.get("text") or ""),
                frame_text_long=str(row.get("frame_text_long") or ""),
            )
        )

    return views


def _build_object_records_full(
    object_rows: Sequence[Mapping[str, Any]],
    place_map: Mapping[int, str],
    view_map: Mapping[int, str],
    place_rows: Sequence[PlaceRecord],
) -> Tuple[List[ObjectObservationRecord], List[ObjectClassRecord], List[PlaceObjectEdgeRecord]]:
    place_by_id = {place.place_id: place for place in place_rows}
    observations: List[ObjectObservationRecord] = []
    object_classes: Dict[str, ObjectClassRecord] = {}
    place_object_edges: List[PlaceObjectEdgeRecord] = []

    for row in sorted(object_rows, key=lambda item: int(item["object_global_id"])):
        entry_id = int(row["entry_id"])
        place_id = str(place_map[entry_id])
        view_id = str(view_map[entry_id])
        place = place_by_id[place_id]
        orientation = row.get("object_orientation_deg")
        distance = row.get("distance_from_camera_m")
        projected_x = _safe_float(row.get("estimated_global_x"))
        projected_y = _safe_float(row.get("estimated_global_y"))
        projected_z = _safe_float(row.get("estimated_global_z"))
        if projected_x is None or projected_z is None:
            projected_x, projected_z = _project_object_position(
                origin_x=place.x,
                origin_z=place.z,
                orientation_deg=float(orientation) if orientation is not None else None,
                distance_m=float(distance) if distance is not None else None,
            )
        label = str(row.get("label") or "unknown")
        obs_id = f"obs_{int(row['object_global_id']):06d}"

        observation = ObjectObservationRecord(
            obs_id=obs_id,
            place_id=place_id,
            view_id=view_id,
            entry_id=entry_id,
            object_global_id=int(row["object_global_id"]),
            object_class=label,
            label=label,
            description=str(row.get("description") or ""),
            long_form_open_description=str(row.get("long_form_open_description") or ""),
            laterality=str(row.get("laterality") or "center"),
            distance_bin=str(row.get("distance_bin") or "middle"),
            verticality=str(row.get("verticality") or "middle"),
            distance_from_camera_m=float(distance) if distance is not None else None,
            object_orientation_deg=float(orientation) if orientation is not None else None,
            projected_x=projected_x,
            projected_y=projected_y,
            projected_z=projected_z,
            location_relative_to_other_objects=str(row.get("location_relative_to_other_objects") or ""),
            parse_status=str(row.get("parse_status") or "unknown"),
        )
        observations.append(observation)
        object_classes.setdefault(label, ObjectClassRecord(name=label))
        place_object_edges.append(
            PlaceObjectEdgeRecord(
                place_id=place_id,
                obs_id=obs_id,
                view_id=view_id,
                object_orientation_deg=observation.object_orientation_deg,
                distance_from_camera_m=observation.distance_from_camera_m,
                projected_x=projected_x,
                projected_z=projected_z,
            )
        )

    return observations, sorted(object_classes.values(), key=lambda item: item.name), place_object_edges


def build_object_records(
    object_rows: Sequence[Mapping[str, Any]],
    place_map: Mapping[int, str],
    view_map: Mapping[int, str],
    place_rows: Sequence[PlaceRecord],
) -> List[ObjectObservationRecord]:
    observations, _object_classes, _place_object_edges = _build_object_records_full(
        object_rows=object_rows,
        place_map=place_map,
        view_map=view_map,
        place_rows=place_rows,
    )
    return observations


def build_direction_edges(
    place_rows: Sequence[PlaceRecord],
    k_neighbors: int = 4,
    same_axis_eps: float = 0.25,
    radius_m: Optional[float] = None,
) -> List[DirectionEdgeRecord]:
    places = list(place_rows)
    edges: Dict[Tuple[str, str, str], DirectionEdgeRecord] = {}

    for source in places:
        candidates: List[Tuple[float, PlaceRecord]] = []
        for target in places:
            if source.place_id == target.place_id:
                continue
            dx = float(target.x - source.x)
            dz = float(target.z - source.z)
            distance = float(math.hypot(dx, dz))
            if radius_m is not None and distance > float(radius_m):
                continue
            candidates.append((distance, target))

        candidates.sort(key=lambda item: (item[0], item[1].place_id))
        for distance, target in candidates[: max(0, int(k_neighbors))]:
            dx = float(target.x - source.x)
            dz = float(target.z - source.z)
            direction = _classify_direction(dx=dx, dz=dz, same_axis_eps=same_axis_eps)
            if direction is None:
                continue
            forward_key = (source.place_id, target.place_id, direction)
            if forward_key not in edges:
                edges[forward_key] = DirectionEdgeRecord(
                    source_place_id=source.place_id,
                    target_place_id=target.place_id,
                    relation_type=direction,
                    dx=dx,
                    dz=dz,
                    distance_m=distance,
                )

            reverse = REVERSE_DIRECTION[direction]
            reverse_key = (target.place_id, source.place_id, reverse)
            if reverse_key not in edges:
                edges[reverse_key] = DirectionEdgeRecord(
                    source_place_id=target.place_id,
                    target_place_id=source.place_id,
                    relation_type=reverse,
                    dx=-dx,
                    dz=-dz,
                    distance_m=distance,
                )

    return sorted(
        edges.values(),
        key=lambda edge: (edge.source_place_id, edge.relation_type, edge.target_place_id),
    )


def _build_view_view_edges(view_rows: Sequence[ViewRecord]) -> List[Dict[str, Any]]:
    by_place: Dict[str, List[ViewRecord]] = defaultdict(list)
    for view in view_rows:
        by_place[str(view.place_id)].append(view)

    rows: List[Dict[str, Any]] = []
    for place_id, views in by_place.items():
        ordered = sorted(views, key=lambda item: (item.orientation_deg, item.view_id))
        for source in ordered:
            for target in ordered:
                if source.view_id == target.view_id:
                    continue
                rows.append(
                    {
                        "source_view_id": source.view_id,
                        "target_view_id": target.view_id,
                        "place_id": place_id,
                        "direction": "neighbor",
                        "relation_type": "ViewView",
                        "dx": 0.0,
                        "dy": 0.0,
                        "dz": 0.0,
                        "distance_m": 0.0,
                    }
                )
    return rows


def _build_view_object_edges_from_rows(
    meta_rows: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    entry_by_id = {int(row["id"]): dict(row) for row in meta_rows if row.get("id") is not None}
    rows: List[Dict[str, Any]] = []
    for row in sorted(object_rows, key=lambda item: int(item["object_global_id"])):
        entry_id = int(row["entry_id"])
        entry = entry_by_id.get(entry_id)
        if entry is None:
            continue
        object_x = _safe_float(row.get("estimated_global_x"))
        object_z = _safe_float(row.get("estimated_global_z"))
        if object_x is None or object_z is None:
            object_x = _safe_float(row.get("projected_x"))
            object_z = _safe_float(row.get("projected_z"))
        view_x = _safe_float(entry.get("x"))
        _view_y = _safe_world_position(entry)[1]
        view_z = _safe_float(entry.get("y"))
        if object_x is None or object_z is None or view_x is None or view_z is None:
            continue
        object_y = _safe_float(row.get("estimated_global_y"))
        dx = float(object_x - view_x)
        dy = float(object_y - _view_y) if object_y is not None else 0.0
        dz = float(object_z - view_z)
        rows.append(
            {
                "entry_id": entry_id,
                "view_id": f"view_{entry_id:05d}",
                "object_global_id": int(row["object_global_id"]),
                "obs_id": _obs_id_for_object_global_id(row["object_global_id"]),
                "label": str(row.get("label") or "unknown"),
                "view_x": view_x,
                "view_y": _view_y,
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
    return rows


def _build_object_object_edges_from_rows(
    meta_rows: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
    same_axis_eps: float,
) -> List[Dict[str, Any]]:
    entry_by_id = {int(row["id"]): dict(row) for row in meta_rows if row.get("id") is not None}
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in object_rows:
        object_x = _safe_float(row.get("estimated_global_x"))
        object_z = _safe_float(row.get("estimated_global_z"))
        if object_x is None or object_z is None:
            object_x = _safe_float(row.get("projected_x"))
            object_z = _safe_float(row.get("projected_z"))
        if object_x is None or object_z is None:
            continue
        cloned = dict(row)
        cloned["_resolved_x"] = object_x
        cloned["_resolved_z"] = object_z
        grouped[int(row["entry_id"])].append(cloned)

    rows: List[Dict[str, Any]] = []
    for entry_id, group_rows in grouped.items():
        entry = entry_by_id.get(entry_id)
        if entry is None:
            continue
        view_orientation = _safe_float(entry.get("orientation"))
        if view_orientation is None:
            continue
        ordered = sorted(group_rows, key=lambda item: int(item["object_global_id"]))
        for source in ordered:
            source_x = float(source["_resolved_x"])
            source_y = _safe_float(source.get("estimated_global_y"))
            source_z = float(source["_resolved_z"])
            for target in ordered:
                if int(source["object_global_id"]) == int(target["object_global_id"]):
                    continue
                target_x = float(target["_resolved_x"])
                target_y = _safe_float(target.get("estimated_global_y"))
                target_z = float(target["_resolved_z"])
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
                rows.append(
                    {
                        "entry_id": int(entry_id),
                        "view_id": f"view_{int(entry_id):05d}",
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
    return rows


def build_graph_payload(
    db_dir: str,
    k_neighbors: int = 4,
    same_axis_eps: float = 0.25,
    radius_m: Optional[float] = None,
) -> Dict[str, Any]:
    db_path = Path(db_dir)
    meta_rows = _load_jsonl(db_path / "meta.jsonl")
    object_rows = _load_jsonl(db_path / "object_meta.jsonl")
    view_object_relation_rows = _load_jsonl_if_exists(db_path / "view_object_relations.jsonl")
    object_object_relation_rows = _load_jsonl_if_exists(db_path / "object_object_relations.jsonl")
    report = _load_build_report(db_dir)
    scan_angles = _resolve_scan_angles(meta_rows, report)
    image_embs = _load_npy_if_exists(db_path / "image_emb.npy")
    text_embs_long = _load_npy_if_exists(db_path / "text_emb_long.npy")
    object_text_embs_long = _load_npy_if_exists(db_path / "object_text_emb_long.npy")
    entry_meta_by_id = {int(row["id"]): dict(row) for row in meta_rows if row.get("id") is not None}

    place_rows, place_map = _build_place_records_with_map(meta_rows, scan_angles=scan_angles)
    view_rows = build_view_records(meta_rows, place_map=place_map)
    view_map = {view.entry_id: view.view_id for view in view_rows}
    object_rows_built, object_classes, place_object_edges = _build_object_records_full(
        object_rows=object_rows,
        place_map=place_map,
        view_map=view_map,
        place_rows=place_rows,
    )
    direction_edges = build_direction_edges(
        place_rows=place_rows,
        k_neighbors=k_neighbors,
        same_axis_eps=same_axis_eps,
        radius_m=radius_m,
    )
    view_view_edges = _build_view_view_edges(view_rows)
    if not view_object_relation_rows:
        view_object_relation_rows = _build_view_object_edges_from_rows(meta_rows=meta_rows, object_rows=object_rows)
    if not object_object_relation_rows:
        object_object_relation_rows = _build_object_object_edges_from_rows(
            meta_rows=meta_rows,
            object_rows=object_rows,
            same_axis_eps=same_axis_eps,
        )

    view_nodes: List[Dict[str, Any]] = []
    for view in view_rows:
        entry = entry_meta_by_id[int(view.entry_id)]
        view_world = _safe_world_position(entry)
        view_nodes.append(
            {
                "node_type": "View",
                "view_id": view.view_id,
                "place_id": view.place_id,
                "entry_id": view.entry_id,
                "orientation_deg": view.orientation_deg,
                "file_name": view.file_name,
                "parse_status": view.parse_status,
                "frame_text_short": view.frame_text_short,
                "frame_text_long": view.frame_text_long,
                "x": float(view_world[0]),
                "y": float(view_world[1]),
                "z": float(view_world[2]),
                "desc_emb": _embedding_row_to_list(text_embs_long, view.entry_id),
                "image_emb": _embedding_row_to_list(image_embs, view.entry_id),
            }
        )

    object_nodes: List[Dict[str, Any]] = []
    for obs in object_rows_built:
        object_nodes.append(
            {
                "node_type": "Object",
                "obs_id": obs.obs_id,
                "place_id": obs.place_id,
                "view_id": obs.view_id,
                "entry_id": obs.entry_id,
                "object_global_id": obs.object_global_id,
                "object_class": obs.object_class,
                "label": obs.label,
                "description": obs.description,
                "long_form_open_description": obs.long_form_open_description,
                "x": obs.projected_x,
                "y": obs.projected_y,
                "z": obs.projected_z,
                "desc_emb": _embedding_row_to_list(object_text_embs_long, obs.object_global_id),
                "image_emb": None,
            }
        )

    payload = {
        "scan_angles": scan_angles,
        "places": _serialize_records(place_rows),
        "views": _serialize_records(view_rows),
        "objects": _serialize_records(object_rows_built),
        "object_classes": _serialize_records(object_classes),
        "place_object_edges": _serialize_records(place_object_edges),
        "direction_edges": _serialize_records(direction_edges),
        "view_nodes": view_nodes,
        "object_nodes": object_nodes,
        "view_view_edges": view_view_edges,
        "view_object_edges": list(view_object_relation_rows),
        "object_object_edges": list(object_object_relation_rows),
    }
    payload["summary"] = {
        "num_places": len(place_rows),
        "num_views": len(view_rows),
        "num_objects": len(object_rows_built),
        "num_object_classes": len(object_classes),
        "num_direction_edges": len(direction_edges),
        "num_view_view_edges": len(view_view_edges),
        "num_view_object_edges": len(view_object_relation_rows),
        "num_object_object_edges": len(object_object_relation_rows),
        "k_neighbors": int(k_neighbors),
        "same_axis_eps": float(same_axis_eps),
    }
    return payload


def _batched(rows: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    size = max(1, int(batch_size))
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _neo4j_driver(uri: str, auth: Tuple[str, str]):
    placeholders = []
    for module_name in ("pandas", "pyarrow"):
        if module_name not in sys.modules:
            sys.modules[module_name] = None
            placeholders.append(module_name)
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("neo4j package is required for load_graph_to_neo4j/query helpers") from exc
    finally:
        for module_name in placeholders:
            sys.modules.pop(module_name, None)
        importlib.invalidate_caches()
    return GraphDatabase.driver(uri, auth=auth)


def create_neo4j_driver(uri: str, auth: Tuple[str, str]):
    return _neo4j_driver(uri=uri, auth=auth)


def load_graph_to_neo4j(
    payload: Mapping[str, Any],
    uri: str,
    auth: Tuple[str, str],
    batch_size: int = 500,
    database: Optional[str] = None,
) -> Dict[str, int]:
    driver = _neo4j_driver(uri=uri, auth=auth)
    places = list(payload.get("places", []))
    views = list(payload.get("views", []))
    objects = list(payload.get("objects", []))
    object_classes = list(payload.get("object_classes", []))
    place_object_edges = list(payload.get("place_object_edges", []))
    direction_edges = list(payload.get("direction_edges", []))
    view_nodes = list(payload.get("view_nodes", []))
    object_nodes = list(payload.get("object_nodes", []))
    view_view_edges = list(payload.get("view_view_edges", []))
    view_object_edges = list(payload.get("view_object_edges", []))
    object_object_edges = list(payload.get("object_object_edges", []))
    view_node_by_id = {str(row["view_id"]): dict(row) for row in view_nodes if row.get("view_id") is not None}
    object_node_by_id = {str(row["obs_id"]): dict(row) for row in object_nodes if row.get("obs_id") is not None}

    with driver.session(database=database) as session:
        for statement in (
            "CREATE CONSTRAINT place_place_id IF NOT EXISTS FOR (p:Place) REQUIRE p.place_id IS UNIQUE",
            "CREATE CONSTRAINT view_view_id IF NOT EXISTS FOR (v:View) REQUIRE v.view_id IS UNIQUE",
            "CREATE CONSTRAINT obj_obs_id IF NOT EXISTS FOR (o:ObjectObservation) REQUIRE o.obs_id IS UNIQUE",
            "CREATE CONSTRAINT obj_class_name IF NOT EXISTS FOR (c:ObjectClass) REQUIRE c.name IS UNIQUE",
        ):
            session.run(statement)

        for batch in _batched(places, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (p:Place {place_id: row.place_id})
                SET p.x = row.x,
                    p.y = row.y,
                    p.z = row.z,
                    p.point = point({x: row.x, y: row.z, crs: 'cartesian'}),
                    p.room_function = row.room_function,
                    p.view_type = row.view_type,
                    p.source_entry_ids = row.source_entry_ids,
                    p.scan_angles = row.scan_angles
                """,
                rows=batch,
            )

        for batch in _batched(views, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (p:Place {place_id: row.place_id})
                MERGE (v:View {view_id: row.view_id})
                SET v.entry_id = row.entry_id,
                    v.orientation_deg = row.orientation_deg,
                    v.file_name = row.file_name,
                    v.parse_status = row.parse_status,
                    v.frame_text_short = row.frame_text_short,
                    v.frame_text_long = row.frame_text_long,
                    v.node_type = coalesce(row.node_type, 'View'),
                    v.x = row.x,
                    v.y = row.y,
                    v.z = row.z,
                    v.desc_emb = row.desc_emb,
                    v.image_emb = row.image_emb
                MERGE (p)-[:HAS_VIEW]->(v)
                """,
                rows=[{**dict(row), **dict(view_node_by_id.get(str(row["view_id"]), {}))} for row in batch],
            )

        for batch in _batched(object_classes, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (:ObjectClass {name: row.name})
                """,
                rows=batch,
            )

        for batch in _batched(objects, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (v:View {view_id: row.view_id})
                MATCH (p:Place {place_id: row.place_id})
                MATCH (c:ObjectClass {name: row.object_class})
                MERGE (o:ObjectObservation {obs_id: row.obs_id})
                SET o.entry_id = row.entry_id,
                    o.object_global_id = row.object_global_id,
                    o.label = row.label,
                    o.description = row.description,
                    o.long_form_open_description = row.long_form_open_description,
                    o.laterality = row.laterality,
                    o.distance_bin = row.distance_bin,
                    o.verticality = row.verticality,
                    o.distance_from_camera_m = row.distance_from_camera_m,
                    o.object_orientation_deg = row.object_orientation_deg,
                    o.projected_x = row.projected_x,
                    o.projected_y = row.projected_y,
                    o.projected_z = row.projected_z,
                    o.location_relative_to_other_objects = row.location_relative_to_other_objects,
                    o.parse_status = row.parse_status,
                    o.node_type = coalesce(row.node_type, 'Object'),
                    o.x = row.x,
                    o.y = row.y,
                    o.z = row.z,
                    o.desc_emb = row.desc_emb,
                    o.image_emb = row.image_emb
                MERGE (v)-[:OBSERVES]->(o)
                MERGE (o)-[:INSTANCE_OF]->(c)
                MERGE (p)-[:HAS_OBJECT {view_id: row.view_id}]->(o)
                """,
                rows=[{**dict(row), **dict(object_node_by_id.get(str(row["obs_id"]), {}))} for row in batch],
            )

        for batch in _batched(place_object_edges, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (p:Place {place_id: row.place_id})
                MATCH (o:ObjectObservation {obs_id: row.obs_id})
                MERGE (p)-[r:HAS_OBJECT {view_id: row.view_id}]->(o)
                SET r.object_orientation_deg = row.object_orientation_deg,
                    r.distance_from_camera_m = row.distance_from_camera_m,
                    r.projected_x = row.projected_x,
                    r.projected_z = row.projected_z
                """,
                rows=batch,
            )

        grouped_direction_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in direction_edges:
            grouped_direction_edges[str(edge["relation_type"])].append(edge)

        for relation_type, rows in grouped_direction_edges.items():
            for batch in _batched(rows, batch_size):
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (src:Place {{place_id: row.source_place_id}})
                    MATCH (dst:Place {{place_id: row.target_place_id}})
                    MERGE (src)-[r:{relation_type}]->(dst)
                    SET r.dx = row.dx,
                        r.dz = row.dz,
                        r.distance_m = row.distance_m
                    """,
                    rows=batch,
                )

        for batch in _batched(view_view_edges, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (src:View {view_id: row.source_view_id})
                MATCH (dst:View {view_id: row.target_view_id})
                MERGE (src)-[r:NEIGHBOR_VIEW]->(dst)
                SET r.direction = row.direction,
                    r.relation_type = row.relation_type,
                    r.dx = row.dx,
                    r.dy = row.dy,
                    r.dz = row.dz,
                    r.distance_m = row.distance_m
                """,
                rows=batch,
            )

        for batch in _batched(view_object_edges, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (v:View {view_id: row.view_id})
                MATCH (o:ObjectObservation {obs_id: row.obs_id})
                MERGE (v)-[r:IN_VIEW]->(o)
                SET r.direction = row.direction,
                    r.direction_frame = row.direction_frame,
                    r.relation_type = row.relation_type,
                    r.vertical_direction = row.vertical_direction,
                    r.dx = row.dx,
                    r.dy = row.dy,
                    r.dz = row.dz,
                    r.distance_m = row.distance_m,
                    r.distance_3d_m = row.distance_3d_m
                """,
                rows=batch,
            )

        grouped_object_object_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in object_object_edges:
            neo4j_rel = OBJECT_OBJECT_NEO4J_RELATIONSHIPS.get(str(edge.get("direction") or "").strip().lower())
            if not neo4j_rel:
                continue
            grouped_object_object_edges[neo4j_rel].append(dict(edge))

        for relation_type, rows in grouped_object_object_edges.items():
            for batch in _batched(rows, batch_size):
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (src:ObjectObservation {{obs_id: row.source_obs_id}})
                    MATCH (dst:ObjectObservation {{obs_id: row.target_obs_id}})
                    MERGE (src)-[r:{relation_type}]->(dst)
                    SET r.direction = row.direction,
                        r.direction_frame = row.direction_frame,
                        r.relation_type = row.relation_type,
                        r.vertical_direction = row.vertical_direction,
                        r.dx = row.dx,
                        r.dy = row.dy,
                        r.dz = row.dz,
                        r.distance_m = row.distance_m,
                        r.distance_3d_m = row.distance_3d_m,
                        r.relation_source = row.relation_source
                    """,
                    rows=batch,
                )

        grouped_vertical_object_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in object_object_edges:
            neo4j_rel = OBJECT_OBJECT_VERTICAL_NEO4J_RELATIONSHIPS.get(
                str(edge.get("vertical_direction") or "").strip().lower()
            )
            if not neo4j_rel:
                continue
            grouped_vertical_object_edges[neo4j_rel].append(dict(edge))

        for relation_type, rows in grouped_vertical_object_edges.items():
            for batch in _batched(rows, batch_size):
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (src:ObjectObservation {{obs_id: row.source_obs_id}})
                    MATCH (dst:ObjectObservation {{obs_id: row.target_obs_id}})
                    MERGE (src)-[r:{relation_type}]->(dst)
                    SET r.direction = row.direction,
                        r.direction_frame = row.direction_frame,
                        r.relation_type = row.relation_type,
                        r.vertical_direction = row.vertical_direction,
                        r.dx = row.dx,
                        r.dy = row.dy,
                        r.dz = row.dz,
                        r.distance_m = row.distance_m,
                        r.distance_3d_m = row.distance_3d_m,
                        r.relation_source = row.relation_source
                    """,
                    rows=batch,
                )

    driver.close()
    return {
        "places": len(places),
        "views": len(views),
        "objects": len(objects),
        "object_classes": len(object_classes),
        "place_object_edges": len(place_object_edges),
        "direction_edges": len(direction_edges),
        "view_view_edges": len(view_view_edges),
        "view_object_edges": len(view_object_edges),
        "object_object_edges": len(object_object_edges),
    }


def query_same_node(driver, place_id: str, database: Optional[str] = None) -> Dict[str, Any]:
    with driver.session(database=database) as session:
        place_rows = [record.data() for record in session.run(
            "MATCH (p:Place {place_id: $place_id}) RETURN p.place_id AS place_id, p.scan_angles AS scan_angles",
            place_id=place_id,
        )]
        view_rows = [record.data() for record in session.run(
            """
            MATCH (p:Place {place_id: $place_id})-[:HAS_VIEW]->(v:View)
            RETURN v.view_id AS view_id, v.entry_id AS entry_id, v.orientation_deg AS orientation_deg, v.file_name AS file_name
            ORDER BY v.orientation_deg
            """,
            place_id=place_id,
        )]
        object_rows = [record.data() for record in session.run(
            """
            MATCH (p:Place {place_id: $place_id})-[:HAS_OBJECT]->(o:ObjectObservation)
            RETURN o.obs_id AS obs_id, o.label AS label, o.description AS description, o.object_orientation_deg AS object_orientation_deg
            ORDER BY o.obs_id
            """,
            place_id=place_id,
        )]
    return {
        "place": place_rows[0] if place_rows else None,
        "views": view_rows,
        "objects": object_rows,
    }


def query_direction_neighbors(
    driver,
    place_id: str,
    direction: str,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    relation_type = _normalize_direction(direction)
    with driver.session(database=database) as session:
        result = session.run(
            f"""
            MATCH (p:Place {{place_id: $place_id}})-[r:{relation_type}]->(n:Place)
            RETURN n.place_id AS place_id,
                   n.x AS x,
                   n.y AS y,
                   n.z AS z,
                   n.room_function AS room_function,
                   n.view_type AS view_type,
                   r.dx AS dx,
                   r.dz AS dz,
                   r.distance_m AS distance_m
            ORDER BY r.distance_m ASC, n.place_id ASC
            """,
            place_id=place_id,
        )
        return [record.data() for record in result]


def query_place_objects(
    driver,
    place_id: str,
    object_label: Optional[str] = None,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = """
        MATCH (p:Place {place_id: $place_id})-[r:HAS_OBJECT]->(o:ObjectObservation)
    """
    if object_label:
        query += "\nWHERE o.label = $object_label"
    query += """
        RETURN o.obs_id AS obs_id,
               o.label AS label,
               o.description AS description,
               r.view_id AS view_id,
               r.object_orientation_deg AS object_orientation_deg,
               r.distance_from_camera_m AS distance_from_camera_m,
               r.projected_x AS projected_x,
               r.projected_z AS projected_z
        ORDER BY o.obs_id
    """
    params = {"place_id": place_id}
    if object_label:
        params["object_label"] = object_label
    with driver.session(database=database) as session:
        return [record.data() for record in session.run(query, **params)]


def query_places_for_object(
    driver,
    object_label: str,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    label = str(object_label or "").strip()
    if not label:
        raise ValueError("object_label must be a non-empty string")
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (p:Place)-[:HAS_OBJECT]->(o:ObjectObservation)<-[:OBSERVES]-(v:View)
            WHERE toLower(o.label) = toLower($object_label)
            RETURN p.place_id AS place_id,
                   p.x AS x,
                   p.y AS y,
                   p.z AS z,
                   p.room_function AS room_function,
                   p.view_type AS view_type,
                   v.view_id AS view_id,
                   v.orientation_deg AS orientation_deg,
                   v.file_name AS file_name,
                   o.obs_id AS obs_id,
                   o.label AS label,
                   o.description AS description
            ORDER BY p.place_id ASC, v.orientation_deg ASC, o.obs_id ASC
            """,
            object_label=label,
        )
        return [record.data() for record in result]


def query_direction_objects(
    driver,
    place_id: str,
    direction: str,
    object_label: Optional[str] = None,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    relation_type = _normalize_direction(direction)
    query = f"""
        MATCH (p:Place {{place_id: $place_id}})-[d:{relation_type}]->(n:Place)-[r:HAS_OBJECT]->(o:ObjectObservation)
    """
    if object_label:
        query += "\nWHERE o.label = $object_label"
    query += """
        RETURN n.place_id AS place_id,
               o.obs_id AS obs_id,
               o.label AS label,
               o.description AS description,
               d.dx AS dx,
               d.dz AS dz,
               d.distance_m AS distance_m,
               r.view_id AS view_id,
               r.object_orientation_deg AS object_orientation_deg,
               r.distance_from_camera_m AS distance_from_camera_m
        ORDER BY d.distance_m ASC, n.place_id ASC, o.obs_id ASC
    """
    params = {"place_id": place_id}
    if object_label:
        params["object_label"] = object_label
    with driver.session(database=database) as session:
        return [record.data() for record in session.run(query, **params)]


def _normalize_direction(direction: str) -> str:
    value = str(direction or "").strip().upper()
    aliases = {
        "NORTH": "NORTH_OF",
        "SOUTH": "SOUTH_OF",
        "EAST": "EAST_OF",
        "WEST": "WEST_OF",
    }
    relation_type = aliases.get(value, value)
    if relation_type not in DIRECTION_RELATIONSHIPS:
        raise ValueError(f"Unsupported direction: {direction}")
    return relation_type


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Neo4j-friendly spatial graph payload from spatial_db artifacts.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory containing meta.jsonl/object_meta.jsonl")
    parser.add_argument("--payload_out", type=str, default=None, help="Optional JSON path for the graph payload")
    parser.add_argument("--k_neighbors", type=int, default=4, help="Number of nearest place neighbors to consider")
    parser.add_argument("--same_axis_eps", type=float, default=0.25, help="Tolerance for treating points as the same place")
    parser.add_argument("--radius_m", type=float, default=None, help="Optional max distance for place adjacency")
    parser.add_argument("--neo4j_uri", type=str, default=None, help="Optional Neo4j bolt URI for direct ingestion")
    parser.add_argument("--neo4j_user", type=str, default=None, help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default=None, help="Neo4j password")
    parser.add_argument("--neo4j_database", type=str, default=None, help="Neo4j database name")
    parser.add_argument("--batch_size", type=int, default=500, help="UNWIND batch size for Neo4j ingestion")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_graph_payload(
        db_dir=args.db_dir,
        k_neighbors=args.k_neighbors,
        same_axis_eps=args.same_axis_eps,
        radius_m=args.radius_m,
    )
    if args.payload_out:
        output_path = Path(args.payload_out)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.neo4j_uri:
        if not args.neo4j_user or not args.neo4j_password:
            raise ValueError("--neo4j_user and --neo4j_password are required when --neo4j_uri is set")
        counts = load_graph_to_neo4j(
            payload=payload,
            uri=args.neo4j_uri,
            auth=(args.neo4j_user, args.neo4j_password),
            batch_size=args.batch_size,
            database=args.neo4j_database,
        )
        print(json.dumps({"payload": payload["summary"], "loaded": counts}, indent=2, ensure_ascii=True))
        return
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
