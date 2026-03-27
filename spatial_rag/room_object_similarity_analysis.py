import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from spatial_rag.graph_builder import build_graph_payload
from spatial_rag.object_instance_clustering import (
    DEFAULT_SAME_VIEW_PENALTY,
    DEFAULT_SAME_VIEW_POLICY,
    _normalize_same_view_policy,
    _same_view_mask,
)
from spatial_rag.object_index import load_object_db
from spatial_rag.object_instance_eval import build_graph_context_strings, embed_graph_contexts


@dataclass(frozen=True)
class SemanticRoomRecord:
    room_id: str
    room_function: str
    view_type: str
    place_ids: List[str]
    object_ids: List[int]
    num_objects: int
    num_unique_labels: int
    unique_label_ratio: float
    duplicate_objects: int
    max_label_count: int
    repeated_labels: Dict[str, int]


def _safe_text(value: Any, default: str = "unknown") -> str:
    text = " ".join(str(value or "").strip().split())
    return text or default


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def _dense_embeddings_by_object_id(
    meta_rows: Sequence[Mapping[str, Any]],
    emb: np.ndarray,
) -> np.ndarray:
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
    if emb.shape[0] != len(meta_rows):
        raise ValueError(
            f"Embedding/meta mismatch: meta={len(meta_rows)} rows, embeddings={emb.shape[0]}"
        )
    if not meta_rows:
        return np.zeros((0, emb.shape[1]), dtype=np.float32)
    max_obj_id = max(_safe_int(row.get("object_global_id"), -1) for row in meta_rows)
    if max_obj_id < 0:
        return np.zeros((0, emb.shape[1]), dtype=np.float32)
    out = np.zeros((max_obj_id + 1, emb.shape[1]), dtype=np.float32)
    for idx, row in enumerate(meta_rows):
        obj_id = _safe_int(row.get("object_global_id"), -1)
        if obj_id < 0:
            continue
        out[obj_id] = emb[idx]
    return _l2_normalize_rows(out)


def _load_representation_embeddings(
    db_dir: str,
    graph_payload: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    for mode in ("short", "long"):
        loaded = load_object_db(db_dir, text_mode=mode)
        if loaded is None:
            raise FileNotFoundError(f"Missing {mode} object DB artifacts in {db_dir}")
        meta_rows, emb, _entry_map = loaded
        embeddings[mode] = _dense_embeddings_by_object_id(meta_rows, emb.astype(np.float32))

    graph_context_by_obj_id = build_graph_context_strings(db_dir=db_dir, graph_payload=graph_payload)
    embeddings["graph"] = embed_graph_contexts(graph_context_by_obj_id)
    return embeddings


def _merge_object_rows(
    graph_payload: Mapping[str, Any],
    raw_object_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    raw_by_id = {
        _safe_int(row.get("object_global_id"), idx): dict(row)
        for idx, row in enumerate(raw_object_rows)
    }
    merged: List[Dict[str, Any]] = []
    for row in graph_payload.get("objects", []):
        obj_id = _safe_int(row.get("object_global_id"), -1)
        combined = dict(raw_by_id.get(obj_id, {}))
        combined.update(dict(row))
        combined["object_global_id"] = obj_id
        combined["label"] = _safe_text(combined.get("label") or combined.get("object_class"))
        combined["description"] = str(combined.get("description") or "")
        combined["long_form_open_description"] = str(combined.get("long_form_open_description") or "")
        combined["room_function"] = _safe_text(combined.get("room_function"))
        combined["view_type"] = _safe_text(combined.get("view_type"))
        merged.append(combined)
    return merged


def _valid_analysis_object(row: Mapping[str, Any]) -> bool:
    label = _safe_text(row.get("label"), default="")
    if not label or label.lower() in {"unknown", "none", "other"}:
        return False
    parse_status = _safe_text(row.get("parse_status"), default="unknown").lower()
    if parse_status not in {"ok", "api"}:
        return False
    return True


def group_semantic_rooms(
    graph_payload: Mapping[str, Any],
    raw_object_rows: Sequence[Mapping[str, Any]],
    min_objects: int = 3,
) -> List[SemanticRoomRecord]:
    places = {str(row["place_id"]): dict(row) for row in graph_payload.get("places", [])}
    merged_objects = _merge_object_rows(graph_payload=graph_payload, raw_object_rows=raw_object_rows)
    grouped_place_ids: Dict[Tuple[str, str], set] = defaultdict(set)
    grouped_object_ids: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    grouped_labels: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for row in merged_objects:
        if not _valid_analysis_object(row):
            continue
        place_id = str(row.get("place_id") or "")
        place = places.get(place_id, {})
        room_function = _safe_text(place.get("room_function"))
        view_type = _safe_text(place.get("view_type"))
        key = (room_function, view_type)
        grouped_place_ids[key].add(place_id)
        grouped_object_ids[key].append(_safe_int(row.get("object_global_id"), -1))
        grouped_labels[key].append(_safe_text(row.get("label")))

    rooms: List[SemanticRoomRecord] = []
    ordered_keys = sorted(grouped_object_ids.keys(), key=lambda item: (item[0], item[1]))
    for index, key in enumerate(ordered_keys):
        object_ids = [obj_id for obj_id in grouped_object_ids[key] if obj_id >= 0]
        if len(object_ids) < int(min_objects):
            continue
        counts = Counter(grouped_labels[key])
        unique_labels = len(counts)
        num_objects = len(object_ids)
        duplicate_objects = num_objects - unique_labels
        max_label_count = max(counts.values()) if counts else 0
        repeated = {label: count for label, count in counts.items() if count > 1}
        rooms.append(
            SemanticRoomRecord(
                room_id=f"room_{index:03d}",
                room_function=key[0],
                view_type=key[1],
                place_ids=sorted(grouped_place_ids[key]),
                object_ids=sorted(object_ids),
                num_objects=num_objects,
                num_unique_labels=unique_labels,
                unique_label_ratio=(float(unique_labels) / float(num_objects)) if num_objects else 0.0,
                duplicate_objects=duplicate_objects,
                max_label_count=max_label_count,
                repeated_labels=dict(sorted(repeated.items())),
            )
        )
    return rooms


def _parse_view_ids(value: Any) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for item in raw.split(","):
        view_id = str(item or "").strip()
        if not view_id:
            continue
        if view_id not in out:
            out.append(view_id)
    return out


def build_selected_view_group(
    *,
    view_ids: Sequence[str],
    objects_by_id: Mapping[int, Mapping[str, Any]],
    min_objects: int = 3,
) -> Tuple[SemanticRoomRecord, List[Dict[str, Any]]]:
    ordered_view_ids = [str(view_id) for view_id in view_ids if str(view_id).strip()]
    if not ordered_view_ids:
        raise ValueError("selected view analysis requires at least one view_id")
    view_rank = {view_id: idx for idx, view_id in enumerate(ordered_view_ids)}
    selected_rows = [
        dict(row)
        for row in objects_by_id.values()
        if _valid_analysis_object(row) and _row_view_id(row) in view_rank
    ]
    selected_rows.sort(
        key=lambda row: (
            view_rank.get(_row_view_id(row), 10**9),
            _safe_text(row.get("label")),
            _safe_int(row.get("object_global_id")),
        )
    )
    if len(selected_rows) < int(min_objects):
        raise ValueError(
            f"Selected views only have {len(selected_rows)} usable objects; need at least {int(min_objects)}"
        )
    label_counts = Counter(_safe_text(row.get("label")) for row in selected_rows)
    room_function_counts = Counter(
        _safe_text(row.get("room_function"))
        for row in selected_rows
        if _safe_text(row.get("room_function")) != "unknown"
    )
    view_type_counts = Counter(
        _safe_text(row.get("view_type"))
        for row in selected_rows
        if _safe_text(row.get("view_type")) != "unknown"
    )
    num_objects = len(selected_rows)
    unique_labels = len(label_counts)
    repeated = {label: count for label, count in label_counts.items() if count > 1}
    room = SemanticRoomRecord(
        room_id="selected_views",
        room_function=room_function_counts.most_common(1)[0][0] if room_function_counts else "selected_views",
        view_type=view_type_counts.most_common(1)[0][0] if view_type_counts else "selected_views",
        place_ids=sorted({str(row.get("place_id") or "") for row in selected_rows if str(row.get("place_id") or "")}),
        object_ids=sorted(_safe_int(row.get("object_global_id"), -1) for row in selected_rows if _safe_int(row.get("object_global_id"), -1) >= 0),
        num_objects=num_objects,
        num_unique_labels=unique_labels,
        unique_label_ratio=(float(unique_labels) / float(num_objects)) if num_objects else 0.0,
        duplicate_objects=num_objects - unique_labels,
        max_label_count=max(label_counts.values()) if label_counts else 0,
        repeated_labels=dict(sorted(repeated.items())),
    )
    return room, selected_rows


def select_simple_and_complex_rooms(
    rooms: Sequence[SemanticRoomRecord],
) -> Tuple[SemanticRoomRecord, SemanticRoomRecord]:
    if len(rooms) < 2:
        raise ValueError("Need at least two semantic room candidates to select simple and complex rooms")

    simple_ranked = sorted(
        rooms,
        key=lambda room: (
            -float(room.unique_label_ratio),
            int(room.duplicate_objects),
            -int(room.num_unique_labels),
            room.room_function,
            room.view_type,
            room.room_id,
        ),
    )
    complex_ranked = sorted(
        rooms,
        key=lambda room: (
            -int(room.duplicate_objects),
            -int(room.max_label_count),
            -int(room.num_objects),
            room.room_function,
            room.view_type,
            room.room_id,
        ),
    )
    simple_room = simple_ranked[0]
    complex_room = next(
        (
            room
            for room in complex_ranked
            if room.room_id != simple_room.room_id and int(room.duplicate_objects) > 0
        ),
        None,
    )
    if complex_room is None:
        raise ValueError("Unable to select a distinct complex room with repeated labels")
    return simple_room, complex_room


def _objects_by_id(
    graph_payload: Mapping[str, Any],
    raw_object_rows: Sequence[Mapping[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    merged = _merge_object_rows(graph_payload=graph_payload, raw_object_rows=raw_object_rows)
    out: Dict[int, Dict[str, Any]] = {}
    for row in merged:
        obj_id = _safe_int(row.get("object_global_id"), -1)
        if obj_id >= 0:
            out[obj_id] = row
    return out


def sample_room_objects(
    room: SemanticRoomRecord,
    objects_by_id: Mapping[int, Mapping[str, Any]],
    room_kind: str,
    min_objects: int = 3,
    max_objects: int = 8,
) -> List[Dict[str, Any]]:
    candidates = [
        dict(objects_by_id[obj_id])
        for obj_id in room.object_ids
        if obj_id in objects_by_id and _valid_analysis_object(objects_by_id[obj_id])
    ]
    candidates.sort(key=lambda row: (_safe_text(row.get("label")), _safe_int(row.get("object_global_id"))))
    if len(candidates) < int(min_objects):
        raise ValueError(f"Room {room.room_id} only has {len(candidates)} usable objects")

    label_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        label_groups[_safe_text(row.get("label"))].append(row)
    for rows in label_groups.values():
        rows.sort(key=lambda row: (_safe_int(row.get("entry_id")), _safe_int(row.get("object_global_id"))))

    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()

    def add_row(row: Mapping[str, Any]) -> None:
        obj_id = _safe_int(row.get("object_global_id"), -1)
        if obj_id < 0 or obj_id in selected_ids or len(selected) >= int(max_objects):
            return
        selected.append(dict(row))
        selected_ids.add(obj_id)

    groups = sorted(label_groups.items(), key=lambda item: (_safe_text(item[0]), _safe_int(item[1][0].get("object_global_id"))))
    repeated_groups = [(label, rows) for label, rows in groups if len(rows) > 1]
    unique_groups = [(label, rows) for label, rows in groups if len(rows) == 1]

    if room_kind == "simple":
        for _label, rows in unique_groups:
            add_row(rows[0])
        for _label, rows in repeated_groups:
            add_row(rows[0])
    elif room_kind == "complex":
        for _label, rows in sorted(repeated_groups, key=lambda item: (-len(item[1]), item[0])):
            for row in rows[:2]:
                add_row(row)
        for _label, rows in unique_groups:
            add_row(rows[0])
        for _label, rows in sorted(repeated_groups, key=lambda item: (-len(item[1]), item[0])):
            for row in rows[2:]:
                add_row(row)
    else:
        raise ValueError(f"Unsupported room_kind: {room_kind}")

    if len(selected) < int(min_objects):
        for row in candidates:
            add_row(row)
            if len(selected) >= int(min_objects):
                break

    selected.sort(key=lambda row: (_safe_text(row.get("label")), _safe_int(row.get("object_global_id"))))
    return selected[: int(max_objects)]


def compute_similarity_matrix(embeddings: np.ndarray, object_ids: Sequence[int]) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    indices = np.asarray([_safe_int(obj_id) for obj_id in object_ids], dtype=np.int64)
    if indices.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if int(indices.max()) >= int(embeddings.shape[0]):
        raise KeyError(f"Object id {int(indices.max())} outside embedding range {embeddings.shape[0]}")
    mat = embeddings[indices] @ embeddings[indices].T
    mat = ((mat + mat.T) / 2.0).astype(np.float32)
    if mat.size:
        np.fill_diagonal(mat, 1.0)
    return mat


def _parse_fused_text_modes(value: Any) -> List[str]:
    raw = str(value or "short,long").strip()
    modes = []
    for item in raw.split(","):
        mode = str(item or "").strip().lower()
        if not mode:
            continue
        if mode not in {"short", "long"}:
            raise ValueError(f"Unsupported fused_text_mode: {mode}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("At least one fused_text_mode must be provided")
    return modes


def _normalize_fused_weights(weight_text: float, weight_geo: float) -> Tuple[float, float]:
    total = float(weight_text) + float(weight_geo)
    if total <= 0.0:
        raise ValueError("weight_text + weight_geo must be > 0")
    return float(weight_text) / total, float(weight_geo) / total


def _row_view_id(row: Mapping[str, Any]) -> str:
    return str(row.get("view_id") or row.get("synthetic_view_id") or row.get("entry_id") or "")


def _same_view_pair(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> bool:
    view_a = _row_view_id(row_a)
    view_b = _row_view_id(row_b)
    return bool(view_a and view_a == view_b)


def _geometry_details(
    rows: Sequence[Mapping[str, Any]],
    sigma_2d: float = 2.0,
    sigma_3d: float = 2.5,
) -> Dict[str, np.ndarray]:
    if float(sigma_2d) <= 0.0 or float(sigma_3d) <= 0.0:
        raise ValueError("sigma_2d and sigma_3d must be > 0")
    size = len(rows)
    sim = np.zeros((size, size), dtype=np.float32)
    dist_2d = np.full((size, size), np.nan, dtype=np.float32)
    dist_3d = np.full((size, size), np.nan, dtype=np.float32)
    used_3d = np.zeros((size, size), dtype=bool)
    available = np.zeros((size, size), dtype=bool)
    if size == 0:
        return {
            "similarity": sim,
            "distance_2d": dist_2d,
            "distance_3d": dist_3d,
            "used_3d": used_3d,
            "available": available,
        }

    for i in range(size):
        sim[i, i] = 1.0
        dist_2d[i, i] = 0.0
        dist_3d[i, i] = 0.0
        used_3d[i, i] = True
        available[i, i] = True

    for i in range(size):
        x1 = _safe_float(rows[i].get("estimated_global_x"))
        y1 = _safe_float(rows[i].get("estimated_global_y"))
        z1 = _safe_float(rows[i].get("estimated_global_z"))
        for j in range(i + 1, size):
            x2 = _safe_float(rows[j].get("estimated_global_x"))
            y2 = _safe_float(rows[j].get("estimated_global_y"))
            z2 = _safe_float(rows[j].get("estimated_global_z"))
            if x1 is None or z1 is None or x2 is None or z2 is None:
                continue
            dx = float(x2 - x1)
            dz = float(z2 - z1)
            d2 = float(np.hypot(dx, dz))
            if y1 is not None and y2 is not None:
                dy = float(y2 - y1)
                d3 = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                score = float(np.exp(-((d3 * d3) / (float(sigma_3d) * float(sigma_3d)))))
                used_3d[i, j] = True
                used_3d[j, i] = True
            else:
                d3 = d2
                score = float(np.exp(-((d2 * d2) / (float(sigma_2d) * float(sigma_2d)))))
            sim[i, j] = score
            sim[j, i] = score
            dist_2d[i, j] = d2
            dist_2d[j, i] = d2
            dist_3d[i, j] = d3
            dist_3d[j, i] = d3
            available[i, j] = True
            available[j, i] = True

    return {
        "similarity": sim.astype(np.float32),
        "distance_2d": dist_2d.astype(np.float32),
        "distance_3d": dist_3d.astype(np.float32),
        "used_3d": used_3d,
        "available": available,
    }


def compute_geometry_similarity_matrix(
    rows: Sequence[Mapping[str, Any]],
    sigma_2d: float = 2.0,
    sigma_3d: float = 2.5,
) -> np.ndarray:
    return _geometry_details(rows=rows, sigma_2d=sigma_2d, sigma_3d=sigma_3d)["similarity"]


def _apply_same_view_policy(
    matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    same_view_policy: str = DEFAULT_SAME_VIEW_POLICY,
    same_view_penalty: float = DEFAULT_SAME_VIEW_PENALTY,
) -> np.ndarray:
    policy = _normalize_same_view_policy(same_view_policy)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square similarity matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(rows):
        raise ValueError(f"Row/matrix mismatch: rows={len(rows)} similarity_shape={tuple(matrix.shape)}")
    adjusted = np.asarray(matrix, dtype=np.float32).copy()
    same_view = _same_view_mask(rows)
    if policy == "hard_block":
        adjusted[same_view] = 0.0
    elif policy == "soft_penalty":
        adjusted[same_view] = adjusted[same_view] * float(same_view_penalty)
    adjusted = ((adjusted + adjusted.T) / 2.0).astype(np.float32)
    if adjusted.size:
        np.fill_diagonal(adjusted, 1.0)
    return adjusted


def compute_fused_similarity_matrix(
    text_matrix: np.ndarray,
    geometry_matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    geometry_available: Optional[np.ndarray] = None,
    same_view_policy: str = DEFAULT_SAME_VIEW_POLICY,
    same_view_penalty: float = DEFAULT_SAME_VIEW_PENALTY,
    weight_text: float = 0.7,
    weight_geo: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    if text_matrix.shape != geometry_matrix.shape:
        raise ValueError(
            f"text/geometry shape mismatch: text={tuple(text_matrix.shape)} geometry={tuple(geometry_matrix.shape)}"
        )
    if text_matrix.ndim != 2 or text_matrix.shape[0] != text_matrix.shape[1]:
        raise ValueError(f"Expected square text matrix, got shape {text_matrix.shape}")
    w_text, w_geo = _normalize_fused_weights(weight_text=weight_text, weight_geo=weight_geo)
    fused_before_penalty = np.asarray(text_matrix, dtype=np.float32).copy()
    if geometry_available is None:
        geometry_available = np.ones_like(text_matrix, dtype=bool)
    if geometry_available.shape != text_matrix.shape:
        raise ValueError(
            f"geometry availability shape mismatch: availability={tuple(geometry_available.shape)} text={tuple(text_matrix.shape)}"
        )
    for i in range(text_matrix.shape[0]):
        for j in range(text_matrix.shape[1]):
            if i == j:
                fused_before_penalty[i, j] = 1.0
                continue
            if bool(geometry_available[i, j]):
                fused_before_penalty[i, j] = (
                    float(w_text) * float(text_matrix[i, j]) + float(w_geo) * float(geometry_matrix[i, j])
                )
            else:
                fused_before_penalty[i, j] = float(text_matrix[i, j])
    fused_before_penalty = ((fused_before_penalty + fused_before_penalty.T) / 2.0).astype(np.float32)
    if fused_before_penalty.size:
        np.fill_diagonal(fused_before_penalty, 1.0)
    fused_after_penalty = _apply_same_view_policy(
        fused_before_penalty,
        rows,
        same_view_policy=same_view_policy,
        same_view_penalty=same_view_penalty,
    )
    return fused_before_penalty, fused_after_penalty


def _label_for_axis(row: Mapping[str, Any]) -> str:
    return f"{_safe_text(row.get('label'))}#{_safe_int(row.get('object_global_id'))}"


def _matrix_pairs(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pairs.append(
                {
                    "obj_a_id": _safe_int(rows[i].get("object_global_id")),
                    "obj_b_id": _safe_int(rows[j].get("object_global_id")),
                    "label_a": _safe_text(rows[i].get("label")),
                    "label_b": _safe_text(rows[j].get("label")),
                    "same_label": _safe_text(rows[i].get("label")) == _safe_text(rows[j].get("label")),
                    "score": float(matrix[i, j]),
                    "cosine": float(matrix[i, j]),
                }
            )
    return pairs


def summarize_room_matrix(
    room: SemanticRoomRecord,
    room_kind: str,
    rows: Sequence[Mapping[str, Any]],
    matrices: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    label_counts = Counter(_safe_text(row.get("label")) for row in rows)
    by_representation: Dict[str, Any] = {}
    for mode, matrix in matrices.items():
        pairs = _matrix_pairs(rows, matrix)
        top_pair = max(pairs, key=lambda item: (item["score"], item["obj_a_id"], item["obj_b_id"])) if pairs else None
        bottom_pair = min(pairs, key=lambda item: (item["score"], item["obj_a_id"], item["obj_b_id"])) if pairs else None
        same_label = [item["score"] for item in pairs if item["same_label"]]
        different_label = [item["score"] for item in pairs if not item["same_label"]]
        by_representation[mode] = {
            "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "most_similar_pair": top_pair,
            "least_similar_pair": bottom_pair,
            "same_label_mean_score": float(np.mean(same_label)) if same_label else None,
            "different_label_mean_score": float(np.mean(different_label)) if different_label else None,
            "same_label_mean_cosine": float(np.mean(same_label)) if same_label else None,
            "different_label_mean_cosine": float(np.mean(different_label)) if different_label else None,
        }
    return {
        "room_kind": room_kind,
        "room_id": room.room_id,
        "room_function": room.room_function,
        "view_type": room.view_type,
        "place_ids": list(room.place_ids),
        "num_selected_objects": int(len(rows)),
        "label_histogram": dict(sorted(label_counts.items())),
        "representation_stats": by_representation,
    }


def _room_object_payload(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "object_global_id": _safe_int(row.get("object_global_id")),
                "label": _safe_text(row.get("label")),
                "description": str(row.get("description") or ""),
                "long_form_open_description": str(row.get("long_form_open_description") or ""),
                "place_id": str(row.get("place_id") or ""),
                "view_id": str(row.get("view_id") or ""),
                "entry_id": _safe_int(row.get("entry_id")),
                "file_name": str(row.get("file_name") or ""),
                "estimated_global_x": _safe_float(row.get("estimated_global_x")),
                "estimated_global_y": _safe_float(row.get("estimated_global_y")),
                "estimated_global_z": _safe_float(row.get("estimated_global_z")),
            }
        )
    return payload


def _write_matrix_csv(path: Path, labels: Sequence[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object"] + list(labels))
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{float(value):.6f}" for value in row])


def _plot_heatmap(path: Path, labels: Sequence[str], matrix: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_size = max(4.5, 1.1 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{float(matrix[i, j]):.2f}",
                ha="center",
                va="center",
                color="white" if float(matrix[i, j]) < 0.45 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_pair_breakdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    text_mode: str,
    text_matrix: np.ndarray,
    geometry_matrix: np.ndarray,
    geometry_available: np.ndarray,
    fused_before_penalty: np.ndarray,
    fused_after_penalty: np.ndarray,
    distance_2d: np.ndarray,
    distance_3d: np.ndarray,
    used_3d: np.ndarray,
) -> List[Dict[str, Any]]:
    breakdown: List[Dict[str, Any]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            geo_available = bool(geometry_available[i, j])
            breakdown.append(
                {
                    "obj_a_id": _safe_int(rows[i].get("object_global_id")),
                    "obj_b_id": _safe_int(rows[j].get("object_global_id")),
                    "label_a": _safe_text(rows[i].get("label")),
                    "label_b": _safe_text(rows[j].get("label")),
                    "same_label": _safe_text(rows[i].get("label")) == _safe_text(rows[j].get("label")),
                    "same_view": _same_view_pair(rows[i], rows[j]),
                    "text_mode": str(text_mode),
                    "text_similarity": float(text_matrix[i, j]),
                    "geometry_similarity": float(geometry_matrix[i, j]) if geo_available else None,
                    "fused_similarity_before_penalty": float(fused_before_penalty[i, j]),
                    "fused_similarity_after_penalty": float(fused_after_penalty[i, j]),
                    "distance_m_2d": float(distance_2d[i, j]) if geo_available and np.isfinite(distance_2d[i, j]) else None,
                    "distance_m_3d": float(distance_3d[i, j]) if geo_available and np.isfinite(distance_3d[i, j]) else None,
                    "used_3d_geometry": bool(used_3d[i, j]) if geo_available else False,
                }
            )
    breakdown.sort(key=lambda item: (-float(item["fused_similarity_after_penalty"]), item["obj_a_id"], item["obj_b_id"]))
    return breakdown


def analyze_rooms(
    db_dir: str,
    output_dir: str,
    min_objects: int = 3,
    max_objects: int = 8,
    k_neighbors: int = 4,
    same_axis_eps: float = 0.25,
    fused_text_modes: Sequence[str] = ("short", "long"),
    weight_text: float = 0.7,
    weight_geo: float = 0.3,
    same_view_policy: str = DEFAULT_SAME_VIEW_POLICY,
    same_view_penalty: float = DEFAULT_SAME_VIEW_PENALTY,
    view_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    selected_fused_modes = _parse_fused_text_modes(",".join(fused_text_modes) if isinstance(fused_text_modes, (list, tuple)) else fused_text_modes)
    selected_view_ids = _parse_view_ids(",".join(view_ids) if isinstance(view_ids, (list, tuple)) else view_ids)
    normalized_weight_text, normalized_weight_geo = _normalize_fused_weights(
        weight_text=weight_text,
        weight_geo=weight_geo,
    )
    payload = build_graph_payload(db_dir, k_neighbors=k_neighbors, same_axis_eps=same_axis_eps)
    raw_object_rows = _load_jsonl(Path(db_dir) / "object_meta.jsonl")
    embeddings = _load_representation_embeddings(db_dir=db_dir, graph_payload=payload)
    objects_by_id = _objects_by_id(graph_payload=payload, raw_object_rows=raw_object_rows)
    rooms: List[SemanticRoomRecord] = []
    selected_rooms: List[Tuple[str, SemanticRoomRecord, str, List[Dict[str, Any]]]] = []
    analysis_mode = "selected_views" if selected_view_ids else "auto_rooms"
    if selected_view_ids:
        selected_room, selected_rows = build_selected_view_group(
            view_ids=selected_view_ids,
            objects_by_id=objects_by_id,
            min_objects=min_objects,
        )
        if len(selected_rows) > int(max_objects):
            selected_rows = selected_rows[: int(max_objects)]
        selected_rooms = [("selected_views", selected_room, "selected", selected_rows)]
        rooms = [selected_room]
    else:
        rooms = group_semantic_rooms(graph_payload=payload, raw_object_rows=raw_object_rows, min_objects=min_objects)
        simple_room, complex_room = select_simple_and_complex_rooms(rooms)
        selected_rooms = [
            (
                "simple_room",
                simple_room,
                "simple",
                sample_room_objects(
                    room=simple_room,
                    objects_by_id=objects_by_id,
                    room_kind="simple",
                    min_objects=min_objects,
                    max_objects=max_objects,
                ),
            ),
            (
                "complex_room",
                complex_room,
                "complex",
                sample_room_objects(
                    room=complex_room,
                    objects_by_id=objects_by_id,
                    room_kind="complex",
                    min_objects=min_objects,
                    max_objects=max_objects,
                ),
            ),
        ]

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    room_summaries: Dict[str, Any] = {}
    for folder_name, room, room_kind, selected_rows in selected_rooms:
        object_ids = [_safe_int(row.get("object_global_id")) for row in selected_rows]
        labels = [_label_for_axis(row) for row in selected_rows]
        matrices = {
            mode: compute_similarity_matrix(embeddings[mode], object_ids)
            for mode in ("short", "long", "graph")
        }
        geometry_details = _geometry_details(selected_rows)
        matrices["geometry"] = geometry_details["similarity"]
        pair_breakdowns: Dict[str, List[Dict[str, Any]]] = {}
        for text_mode in selected_fused_modes:
            fused_before_penalty, fused_after_penalty = compute_fused_similarity_matrix(
                text_matrix=matrices[text_mode],
                geometry_matrix=geometry_details["similarity"],
                rows=selected_rows,
                geometry_available=geometry_details["available"],
                same_view_policy=same_view_policy,
                same_view_penalty=same_view_penalty,
                weight_text=normalized_weight_text,
                weight_geo=normalized_weight_geo,
            )
            matrices[f"{text_mode}_geo"] = fused_after_penalty
            pair_breakdowns[text_mode] = build_pair_breakdown(
                selected_rows,
                text_mode=text_mode,
                text_matrix=matrices[text_mode],
                geometry_matrix=geometry_details["similarity"],
                geometry_available=geometry_details["available"],
                fused_before_penalty=fused_before_penalty,
                fused_after_penalty=fused_after_penalty,
                distance_2d=geometry_details["distance_2d"],
                distance_3d=geometry_details["distance_3d"],
                used_3d=geometry_details["used_3d"],
            )
        room_dir = root / folder_name
        room_dir.mkdir(parents=True, exist_ok=True)
        (room_dir / "objects.json").write_text(
            json.dumps(_room_object_payload(selected_rows), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        for mode, matrix in matrices.items():
            _write_matrix_csv(room_dir / f"{mode}_similarity.csv", labels=labels, matrix=matrix)
            _plot_heatmap(
                room_dir / f"{mode}_similarity_heatmap.png",
                labels=labels,
                matrix=matrix,
                title=f"{folder_name.replace('_', ' ')} {mode} similarity",
            )
        for text_mode, breakdown in pair_breakdowns.items():
            (room_dir / f"pair_breakdown_{text_mode}_geo.json").write_text(
                json.dumps(breakdown, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        room_summary = summarize_room_matrix(room=room, room_kind=room_kind, rows=selected_rows, matrices=matrices)
        room_summary["objects"] = _room_object_payload(selected_rows)
        room_summary["matrices"] = {
            mode: matrix.astype(float).tolist()
            for mode, matrix in matrices.items()
        }
        room_summary["fused_config"] = {
            "fused_text_modes": list(selected_fused_modes),
            "weight_text": float(normalized_weight_text),
            "weight_geo": float(normalized_weight_geo),
            "same_view_policy": str(same_view_policy),
            "same_view_penalty": float(same_view_penalty),
        }
        (room_dir / "summary.json").write_text(
            json.dumps(room_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        room_summaries[folder_name] = room_summary

    report: Dict[str, Any] = {
        "db_dir": str(db_dir),
        "output_dir": str(root),
        "analysis_mode": analysis_mode,
        "selected_view_ids": list(selected_view_ids),
        "candidate_rooms": [asdict(room) for room in rooms],
        "fused_config": {
            "fused_text_modes": list(selected_fused_modes),
            "weight_text": float(normalized_weight_text),
            "weight_geo": float(normalized_weight_geo),
            "same_view_policy": str(same_view_policy),
            "same_view_penalty": float(same_view_penalty),
        },
    }
    if selected_view_ids:
        report["selected_views"] = room_summaries["selected_views"]
    else:
        report["simple_room"] = room_summaries["simple_room"]
        report["complex_room"] = room_summaries["complex_room"]
    (root / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze object similarity matrices inside automatically selected semantic rooms.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for room-level similarity artifacts")
    parser.add_argument("--min_objects", type=int, default=3, help="Minimum object count for a semantic room candidate")
    parser.add_argument("--max_objects", type=int, default=8, help="Maximum number of sampled objects per selected room")
    parser.add_argument("--k_neighbors", type=int, default=4, help="Neighbor count for graph payload construction")
    parser.add_argument("--same_axis_eps", type=float, default=0.25, help="Axis tolerance for graph direction edges")
    parser.add_argument(
        "--fused_text_modes",
        type=str,
        default="short,long",
        help="Comma-separated fused text modes to generate: short, long, or short,long",
    )
    parser.add_argument("--weight_text", type=float, default=0.7, help="Weight for the text similarity branch")
    parser.add_argument("--weight_geo", type=float, default=0.3, help="Weight for the geometry similarity branch")
    parser.add_argument(
        "--same_view_policy",
        type=str,
        default=DEFAULT_SAME_VIEW_POLICY,
        choices=["soft_penalty", "hard_block", "none"],
        help="Policy for same-view pairs in fused similarity matrices",
    )
    parser.add_argument(
        "--same_view_penalty",
        type=float,
        default=DEFAULT_SAME_VIEW_PENALTY,
        help="Penalty multiplier for same-view pairs when using soft_penalty",
    )
    parser.add_argument(
        "--view_ids",
        type=str,
        default="",
        help="Optional comma-separated view_ids to analyze as a single selected-view group",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = analyze_rooms(
        db_dir=args.db_dir,
        output_dir=args.output_dir,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        k_neighbors=args.k_neighbors,
        same_axis_eps=args.same_axis_eps,
        fused_text_modes=_parse_fused_text_modes(args.fused_text_modes),
        weight_text=args.weight_text,
        weight_geo=args.weight_geo,
        same_view_policy=args.same_view_policy,
        same_view_penalty=args.same_view_penalty,
        view_ids=_parse_view_ids(args.view_ids),
    )
    if report.get("analysis_mode") == "selected_views":
        payload = {
            "output_dir": report["output_dir"],
            "analysis_mode": "selected_views",
            "selected_view_ids": report.get("selected_view_ids", []),
            "selected_group": report["selected_views"]["room_id"],
        }
    else:
        payload = {
            "output_dir": report["output_dir"],
            "analysis_mode": "auto_rooms",
            "simple_room": report["simple_room"]["room_id"],
            "complex_room": report["complex_room"]["room_id"],
        }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
