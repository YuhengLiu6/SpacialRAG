import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from spatial_rag.config import OBJECT_SURROUNDING_MAX, OBJECT_VERTICAL_REL_EPS_M
from spatial_rag.object_instance_clustering import (
    _load_jsonl,
    _safe_float,
    _safe_int,
    _safe_text,
    _to_serializable,
    _write_jsonl,
)


DEFAULT_MAX_NEIGHBOR_DISTANCE_M = 2.0
DEFAULT_DEPTH_REL_NONE_EPS_M = 0.05
DEFAULT_DEPTH_REL_SLIGHT_EPS_M = 0.75
DEFAULT_LATERAL_NONE_EPS_DEG = 5.0
DEFAULT_LATERAL_SLIGHT_EPS_DEG = 20.0
DEFAULT_SURROUNDING_SOURCE = "polar_postprocess_v1"
EXCLUDED_LABELS = {"", "unknown", "other", "none"}
COMPASS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _view_id_for_entry(entry_id: int) -> str:
    return f"view_{int(entry_id):05d}"


def _normalize_label(value: Any) -> str:
    return _safe_text(value).strip().lower()


def _wrap_delta_angle_deg(delta_deg: float) -> float:
    wrapped = (float(delta_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def pair_distance_from_polar(distance_a_m: float, distance_b_m: float, delta_theta_deg: float) -> float:
    angle_rad = math.radians(abs(float(delta_theta_deg)))
    squared = (
        float(distance_a_m) * float(distance_a_m)
        + float(distance_b_m) * float(distance_b_m)
        - 2.0 * float(distance_a_m) * float(distance_b_m) * math.cos(angle_rad)
    )
    return math.sqrt(max(0.0, squared))


def classify_local_semantic_relation(
    delta_theta_deg: float,
    delta_depth_m: Optional[float],
    delta_height_m: Optional[float],
) -> str:
    parts: List[str] = []
    abs_theta = abs(float(delta_theta_deg))
    if abs_theta >= DEFAULT_LATERAL_NONE_EPS_DEG:
        side = "right" if float(delta_theta_deg) > 0.0 else "left"
        if abs_theta < DEFAULT_LATERAL_SLIGHT_EPS_DEG:
            parts.append(f"slightly {side}")
        else:
            parts.append(side)

    if delta_depth_m is not None:
        abs_depth = abs(float(delta_depth_m))
        if abs_depth >= DEFAULT_DEPTH_REL_NONE_EPS_M:
            depth_rel = "behind" if float(delta_depth_m) > 0.0 else "in front"
            if abs_depth < DEFAULT_DEPTH_REL_SLIGHT_EPS_M:
                parts.append(f"slightly {depth_rel}")
            else:
                parts.append(depth_rel)

    if delta_height_m is not None and abs(float(delta_height_m)) >= float(OBJECT_VERTICAL_REL_EPS_M):
        parts.append("above" if float(delta_height_m) > 0.0 else "below")

    return ", ".join(parts)


def _bearing_from_global_positions(
    source_x: Optional[float],
    source_z: Optional[float],
    target_x: Optional[float],
    target_z: Optional[float],
) -> Optional[float]:
    if source_x is None or source_z is None or target_x is None or target_z is None:
        return None
    dx = float(target_x) - float(source_x)
    dz = float(target_z) - float(source_z)
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None
    return (math.degrees(math.atan2(dx, -dz)) + 360.0) % 360.0


def _bearing_from_camera_fallback(
    orientation_deg: Optional[float],
    relative_bearing_deg: Optional[float],
) -> Optional[float]:
    if orientation_deg is None or relative_bearing_deg is None:
        return None
    return (float(orientation_deg) - float(relative_bearing_deg)) % 360.0


def _bearing_to_direction_8(bearing_deg: Optional[float]) -> Optional[str]:
    if bearing_deg is None:
        return None
    index = int(((float(bearing_deg) % 360.0) + 22.5) // 45.0) % 8
    return COMPASS_8[index]


def _is_eligible_object(row: Mapping[str, Any]) -> bool:
    if _normalize_label(row.get("label")) in EXCLUDED_LABELS:
        return False
    if _safe_float(row.get("distance_from_camera_m")) is None:
        return False
    if _safe_float(row.get("relative_bearing_deg")) is None:
        return False
    return True


def _serialize_location_summary(neighbors: Sequence[Mapping[str, Any]]) -> str:
    parts: List[str] = []
    for row in neighbors:
        label = _safe_text(row.get("label"), "unknown")
        distance = _safe_float(row.get("distance_from_primary_m"))
        relation = _safe_text(row.get("semantic_relation_local"))
        if not relation:
            relation = "nearby"
        direction = _safe_text(row.get("allocentric_direction_8"))
        if direction:
            parts.append(f"{label}@{float(distance or 0.0):.2f}m[{relation}|{direction}]")
        else:
            parts.append(f"{label}@{float(distance or 0.0):.2f}m[{relation}]")
    return " ; ".join(parts)


def _meta_by_entry(db_dir: Path) -> Dict[int, Dict[str, Any]]:
    meta_rows = _load_jsonl(db_dir / "meta.jsonl")
    by_entry: Dict[int, Dict[str, Any]] = {}
    for row in meta_rows:
        entry_id = _safe_int(row.get("id"), -1)
        if entry_id < 0:
            continue
        copy_row = dict(row)
        copy_row.setdefault("view_id", _view_id_for_entry(entry_id))
        by_entry[entry_id] = copy_row
    return by_entry


def _relation_record(
    *,
    entry_id: int,
    view_id: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    pair_distance_m: float,
    delta_theta_deg: float,
    delta_depth_m: float,
    delta_height_m: Optional[float],
    semantic_relation_local: str,
    allocentric_bearing_deg: Optional[float],
    allocentric_direction_8: Optional[str],
) -> Dict[str, Any]:
    return {
        "entry_id": int(entry_id),
        "view_id": str(view_id),
        "source_object_global_id": _safe_int(source.get("object_global_id"), -1),
        "target_object_global_id": _safe_int(target.get("object_global_id"), -1),
        "source_label": _safe_text(source.get("label"), "unknown"),
        "target_label": _safe_text(target.get("label"), "unknown"),
        "source_distance_from_camera_m": _safe_float(source.get("distance_from_camera_m")),
        "target_distance_from_camera_m": _safe_float(target.get("distance_from_camera_m")),
        "source_relative_bearing_deg": _safe_float(source.get("relative_bearing_deg")),
        "target_relative_bearing_deg": _safe_float(target.get("relative_bearing_deg")),
        "distance_from_primary_m": float(pair_distance_m),
        "delta_angle_deg": float(delta_theta_deg),
        "delta_depth_m": float(delta_depth_m),
        "delta_height_m": None if delta_height_m is None else float(delta_height_m),
        "semantic_relation_local": semantic_relation_local,
        "allocentric_bearing_deg": None if allocentric_bearing_deg is None else float(allocentric_bearing_deg),
        "allocentric_direction_8": allocentric_direction_8,
        "estimated_global_x": _safe_float(target.get("estimated_global_x")),
        "estimated_global_y": _safe_float(target.get("estimated_global_y")),
        "estimated_global_z": _safe_float(target.get("estimated_global_z")),
        "surrounding_source": DEFAULT_SURROUNDING_SOURCE,
    }


def build_polar_surroundings(
    db_dir: str,
    *,
    max_neighbor_distance_m: float = DEFAULT_MAX_NEIGHBOR_DISTANCE_M,
    max_neighbors: int = OBJECT_SURROUNDING_MAX,
    relation_output_name: str = "object_polar_relations.jsonl",
    object_meta_output_name: str = "object_meta_with_polar_surroundings.jsonl",
) -> Dict[str, Any]:
    root = Path(db_dir)
    object_rows = _load_jsonl(root / "object_meta.jsonl")
    meta_by_entry = _meta_by_entry(root)

    indexed_rows: List[Dict[str, Any]] = []
    rows_by_entry: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for index, raw_row in enumerate(object_rows):
        row = dict(raw_row)
        entry_id = _safe_int(row.get("entry_id"), -1)
        row["entry_id"] = entry_id
        row["view_id"] = _safe_text(row.get("view_id")) or _view_id_for_entry(entry_id)
        row["__row_index"] = index
        row["__eligible"] = _is_eligible_object(row)
        indexed_rows.append(row)
        rows_by_entry[entry_id].append(row)

    pair_rows: List[Dict[str, Any]] = []
    contexts_by_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for entry_id, entry_rows in rows_by_entry.items():
        meta_row = meta_by_entry.get(entry_id, {})
        view_id = _safe_text(meta_row.get("view_id")) or _view_id_for_entry(entry_id)
        orientation_deg = _safe_float(meta_row.get("orientation"))
        eligible_rows = [row for row in entry_rows if bool(row.get("__eligible"))]
        for source in eligible_rows:
            source_distance = _safe_float(source.get("distance_from_camera_m"))
            source_angle = _safe_float(source.get("relative_bearing_deg"))
            source_height = _safe_float(source.get("relative_height_from_camera_m"))
            for target in eligible_rows:
                if int(source["__row_index"]) == int(target["__row_index"]):
                    continue
                target_distance = _safe_float(target.get("distance_from_camera_m"))
                target_angle = _safe_float(target.get("relative_bearing_deg"))
                target_height = _safe_float(target.get("relative_height_from_camera_m"))
                if source_distance is None or target_distance is None or source_angle is None or target_angle is None:
                    continue
                delta_theta_deg = _wrap_delta_angle_deg(target_angle - source_angle)
                pair_distance_m = pair_distance_from_polar(source_distance, target_distance, delta_theta_deg)
                if pair_distance_m >= float(max_neighbor_distance_m):
                    continue
                delta_depth_m = float(target_distance) - float(source_distance)
                delta_height_m = None
                if source_height is not None and target_height is not None:
                    delta_height_m = float(target_height) - float(source_height)
                semantic_relation_local = classify_local_semantic_relation(
                    delta_theta_deg=delta_theta_deg,
                    delta_depth_m=delta_depth_m,
                    delta_height_m=delta_height_m,
                )
                allocentric_bearing_deg = _bearing_from_global_positions(
                    _safe_float(source.get("estimated_global_x")),
                    _safe_float(source.get("estimated_global_z")),
                    _safe_float(target.get("estimated_global_x")),
                    _safe_float(target.get("estimated_global_z")),
                )
                if allocentric_bearing_deg is None:
                    allocentric_bearing_deg = _bearing_from_camera_fallback(
                        orientation_deg=orientation_deg,
                        relative_bearing_deg=target_angle,
                    )
                allocentric_direction_8 = _bearing_to_direction_8(allocentric_bearing_deg)
                relation = _relation_record(
                    entry_id=entry_id,
                    view_id=view_id,
                    source=source,
                    target=target,
                    pair_distance_m=pair_distance_m,
                    delta_theta_deg=delta_theta_deg,
                    delta_depth_m=delta_depth_m,
                    delta_height_m=delta_height_m,
                    semantic_relation_local=semantic_relation_local,
                    allocentric_bearing_deg=allocentric_bearing_deg,
                    allocentric_direction_8=allocentric_direction_8,
                )
                pair_rows.append(relation)
                contexts_by_index[int(source["__row_index"])].append(
                    {
                        "target_object_global_id": relation["target_object_global_id"],
                        "label": relation["target_label"],
                        "distance_from_primary_m": relation["distance_from_primary_m"],
                        "delta_angle_deg": relation["delta_angle_deg"],
                        "delta_depth_m": relation["delta_depth_m"],
                        "delta_height_m": relation["delta_height_m"],
                        "semantic_relation_local": relation["semantic_relation_local"],
                        "relation_to_primary": relation["semantic_relation_local"],
                        "allocentric_bearing_deg": relation["allocentric_bearing_deg"],
                        "allocentric_direction_8": relation["allocentric_direction_8"],
                        "estimated_global_x": relation["estimated_global_x"],
                        "estimated_global_y": relation["estimated_global_y"],
                        "estimated_global_z": relation["estimated_global_z"],
                    }
                )

    updated_rows: List[Dict[str, Any]] = []
    for row in indexed_rows:
        row_index = int(row["__row_index"])
        neighbors = contexts_by_index.get(row_index, [])
        neighbors = sorted(
            neighbors,
            key=lambda item: (
                float(item.get("distance_from_primary_m") or 0.0),
                _safe_int(item.get("target_object_global_id"), 10**9),
            ),
        )[: int(max_neighbors)]
        updated = dict(row)
        updated.pop("__row_index", None)
        updated.pop("__eligible", None)
        updated["surrounding_context"] = neighbors
        updated["location_relative_to_other_objects"] = _serialize_location_summary(neighbors)
        updated["surrounding_source"] = DEFAULT_SURROUNDING_SOURCE
        updated_rows.append(updated)

    relation_path = root / relation_output_name
    meta_output_path = root / object_meta_output_name
    _write_jsonl(relation_path, pair_rows)
    _write_jsonl(meta_output_path, updated_rows)
    return {
        "db_dir": str(root),
        "relation_output_path": str(relation_path),
        "object_meta_output_path": str(meta_output_path),
        "num_objects": len(updated_rows),
        "num_pair_relations": len(pair_rows),
        "max_neighbor_distance_m": float(max_neighbor_distance_m),
        "max_neighbors": int(max_neighbors),
        "surrounding_source": DEFAULT_SURROUNDING_SOURCE,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild surrounding_context from polar geometry offline.")
    parser.add_argument("--db_dir", required=True, help="Existing spatial DB directory")
    parser.add_argument(
        "--max_neighbor_distance_m",
        type=float,
        default=DEFAULT_MAX_NEIGHBOR_DISTANCE_M,
        help="Only keep same-frame neighbors within this polar pair distance",
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=OBJECT_SURROUNDING_MAX,
        help="Maximum serialized surrounding neighbors per object",
    )
    parser.add_argument(
        "--relation_output_name",
        default="object_polar_relations.jsonl",
        help="Output filename inside db_dir for pairwise polar relations",
    )
    parser.add_argument(
        "--object_meta_output_name",
        default="object_meta_with_polar_surroundings.jsonl",
        help="Output filename inside db_dir for updated object metadata",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = _parse_args(argv)
    summary = build_polar_surroundings(
        args.db_dir,
        max_neighbor_distance_m=float(args.max_neighbor_distance_m),
        max_neighbors=int(args.max_neighbors),
        relation_output_name=str(args.relation_output_name),
        object_meta_output_name=str(args.object_meta_output_name),
    )
    print(json.dumps(_to_serializable(summary), ensure_ascii=True))
    return summary


if __name__ == "__main__":
    main()
