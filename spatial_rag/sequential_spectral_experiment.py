import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian

from spatial_rag.object_index import load_object_db
from spatial_rag.object_instance_clustering import (
    _alpha_blit,
    _l2_normalize_vec,
    _load_jsonl,
    _make_rotated_text_image,
    _safe_float,
    _safe_int,
    _safe_text,
    _truncate_heatmap_label,
    _to_serializable,
    _write_json,
    plot_similarity_heatmap,
    run_spectral_clustering,
)


DEFAULT_DB_DIR = "/Users/liuyuheng/Desktop/antigravityTest/spatial_db_nd"
DEFAULT_VIEW_IDS = ("view_00019", "view_00024", "view_00058", "view_00065")
DEFAULT_WEIGHT_TEXT = 0.70
DEFAULT_WEIGHT_GLOBAL_GEO = 0.20
DEFAULT_WEIGHT_POLAR = 0.10
DEFAULT_GLOBAL_SIGMA_M = 2.0
DEFAULT_CROSS_AFFINITY_MIN = 0.35
DEFAULT_CURRENT_ONLY_REATTACH_MIN_AFFINITY = 0.75
EXCLUDED_LABELS = {"", "unknown", "other", "none"}
SEQUENTIAL_PROGRESS_COLORS_BGR = (
    (32, 119, 238),
    (70, 190, 255),
    (90, 214, 108),
    (166, 104, 255),
    (88, 88, 240),
    (60, 200, 200),
)


def _normalize_entry_ids(entry_ids: Optional[Sequence[Any]]) -> List[int]:
    if entry_ids is None:
        return []
    out: List[int] = []
    for item in entry_ids:
        if item is None:
            continue
        for token in str(item).split(","):
            cleaned = _safe_text(token)
            if not cleaned:
                continue
            out.append(int(cleaned))
    return out


def _normalize_view_ids(view_ids: Optional[Sequence[str]]) -> List[str]:
    if view_ids is None:
        return list(DEFAULT_VIEW_IDS)
    out: List[str] = []
    for item in view_ids:
        if item is None:
            continue
        for token in str(item).split(","):
            cleaned = _safe_text(token)
            if cleaned:
                out.append(cleaned)
    return out or list(DEFAULT_VIEW_IDS)


def _view_id_for_entry(entry_id: int) -> str:
    return f"view_{int(entry_id):05d}"


def _normalize_label(value: Any) -> str:
    return _safe_text(value).strip().lower()


def _is_valid_object_row(row: Mapping[str, Any]) -> bool:
    if _normalize_label(row.get("label")) in EXCLUDED_LABELS:
        return False
    return _safe_int(row.get("object_global_id"), -1) >= 0


def _wrap_delta_angle_deg(delta_deg: float) -> float:
    wrapped = (float(delta_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _normalize_weight_triplet(weight_text: float, weight_global_geo: float, weight_polar: float) -> Dict[str, float]:
    weights = {
        "text": max(0.0, float(weight_text)),
        "global_geo": max(0.0, float(weight_global_geo)),
        "polar": max(0.0, float(weight_polar)),
    }
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("At least one affinity weight must be positive")
    return {key: value / total for key, value in weights.items()}


def _row_xyz(row: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    return (
        _safe_float(row.get("estimated_global_x")),
        _safe_float(row.get("estimated_global_y")),
        _safe_float(row.get("estimated_global_z")),
    )


def _row_polar(row: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    return (
        _safe_float(row.get("distance_from_camera_m")),
        _safe_float(row.get("relative_bearing_deg")),
        _safe_float(row.get("relative_height_from_camera_m")),
    )


def _median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not numeric:
        return None
    return float(np.median(np.asarray(numeric, dtype=np.float32)))


def _proto_xyz(members: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    return {
        "x": _median_or_none(_safe_float(row.get("estimated_global_x")) for row in members),
        "y": _median_or_none(_safe_float(row.get("estimated_global_y")) for row in members),
        "z": _median_or_none(_safe_float(row.get("estimated_global_z")) for row in members),
    }


def _proto_polar(members: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    return {
        "distance_from_camera_m": _median_or_none(_safe_float(row.get("distance_from_camera_m")) for row in members),
        "relative_bearing_deg": _median_or_none(_safe_float(row.get("relative_bearing_deg")) for row in members),
        "relative_height_from_camera_m": _median_or_none(
            _safe_float(row.get("relative_height_from_camera_m")) for row in members
        ),
    }


def _proto_embedding(members: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    vectors: List[np.ndarray] = []
    for row in members:
        vec = row.get("embedding")
        if vec is None:
            continue
        vectors.append(np.asarray(vec, dtype=np.float32).reshape(-1))
    if not vectors:
        return None
    stacked = np.vstack(vectors)
    mean_vec = np.mean(stacked, axis=0)
    return _l2_normalize_vec(np.asarray(mean_vec, dtype=np.float32))


def _build_cluster(cluster_id: int, members: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    member_rows = [dict(row) for row in members]
    histogram = Counter(_safe_text(row.get("label"), "unknown") for row in member_rows)
    return {
        "cluster_id": int(cluster_id),
        "member_rows": member_rows,
        "member_object_ids": [_safe_int(row.get("object_global_id"), -1) for row in member_rows],
        "member_view_ids": [_safe_text(row.get("view_id")) for row in member_rows],
        "label_histogram": dict(sorted(histogram.items())),
        "prototype_embedding": _proto_embedding(member_rows),
        "prototype_xyz": _proto_xyz(member_rows),
        "prototype_polar": _proto_polar(member_rows),
    }


def _append_member(cluster: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    members = list(cluster.get("member_rows", []))
    members.append(dict(row))
    return _build_cluster(int(cluster.get("cluster_id", -1)), members)


def _merge_clusters(clusters: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not clusters:
        raise ValueError("Cannot merge an empty cluster list")
    ordered = sorted((dict(cluster) for cluster in clusters), key=lambda item: int(item.get("cluster_id", 10**9)))
    merged_members: List[Dict[str, Any]] = []
    for cluster in ordered:
        merged_members.extend([dict(row) for row in cluster.get("member_rows", [])])
    return _build_cluster(int(ordered[0].get("cluster_id", -1)), merged_members)


def _cluster_view_id_set(cluster: Mapping[str, Any]) -> set[str]:
    values = {
        _safe_text(view_id)
        for view_id in list(cluster.get("member_view_ids", []))
        if _safe_text(view_id)
    }
    if values:
        return values
    return {
        _safe_text(row.get("view_id"))
        for row in list(cluster.get("member_rows", []))
        if _safe_text(row.get("view_id"))
    }


def _same_view_collision_pairs(
    indexed_clusters: Sequence[Tuple[int, Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    collisions: List[Dict[str, Any]] = []
    for left_pos, (left_index, left_cluster) in enumerate(indexed_clusters):
        left_views = _cluster_view_id_set(left_cluster)
        if not left_views:
            continue
        for right_index, right_cluster in indexed_clusters[left_pos + 1 :]:
            shared_view_ids = sorted(left_views & _cluster_view_id_set(right_cluster))
            if not shared_view_ids:
                continue
            collisions.append(
                {
                    "left_slot_index": int(left_index),
                    "right_slot_index": int(right_index),
                    "left_cluster_id": _safe_int(left_cluster.get("cluster_id"), -1),
                    "right_cluster_id": _safe_int(right_cluster.get("cluster_id"), -1),
                    "shared_view_ids": shared_view_ids,
                }
            )
    return collisions


def _cross_detail_for_pair(
    cur_idx: int,
    mem_idx: int,
    *,
    cross_details: Sequence[Sequence[Mapping[str, Any]]],
    current_rows: Sequence[Mapping[str, Any]],
    cluster: Mapping[str, Any],
    weight_text: float,
    weight_global_geo: float,
    weight_polar: float,
    global_sigma_m: float,
) -> Dict[str, Any]:
    if 0 <= int(cur_idx) < len(cross_details):
        row_details = cross_details[int(cur_idx)]
        if 0 <= int(mem_idx) < len(row_details):
            detail = dict(row_details[int(mem_idx)] or {})
            if "combined_similarity" in detail:
                return detail
    return _pair_affinity_detail(
        current_rows[int(cur_idx)],
        cluster,
        weights=_normalize_weight_triplet(weight_text, weight_global_geo, weight_polar),
        global_sigma_m=global_sigma_m,
    )


def _best_live_memory_match(
    row: Mapping[str, Any],
    slots: Sequence[Optional[Mapping[str, Any]]],
    *,
    weight_text: float,
    weight_global_geo: float,
    weight_polar: float,
    global_sigma_m: float,
) -> Optional[Tuple[float, int, Dict[str, Any]]]:
    best: Optional[Tuple[float, int, Dict[str, Any]]] = None
    weights = _normalize_weight_triplet(weight_text, weight_global_geo, weight_polar)
    for mem_idx, cluster in enumerate(slots):
        if cluster is None:
            continue
        detail = _pair_affinity_detail(
            row,
            cluster,
            weights=weights,
            global_sigma_m=global_sigma_m,
        )
        score = float(detail.get("combined_similarity") or 0.0)
        candidate = (score, int(mem_idx), detail)
        if best is None or candidate[0] > best[0] or (
            math.isclose(candidate[0], best[0]) and int(mem_idx) < int(best[1])
        ):
            best = candidate
    return best


def _cluster_summary(cluster: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "cluster_id": int(cluster.get("cluster_id", -1)),
        "member_object_ids": list(cluster.get("member_object_ids", [])),
        "member_view_ids": list(cluster.get("member_view_ids", [])),
        "label_histogram": dict(cluster.get("label_histogram", {})),
        "prototype_embedding": cluster.get("prototype_embedding"),
        "prototype_xyz": cluster.get("prototype_xyz"),
        "prototype_polar": cluster.get("prototype_polar"),
    }


def _cluster_output_summary(cluster: Mapping[str, Any]) -> Dict[str, Any]:
    member_rows = list(cluster.get("member_rows", []))
    members = [
        f"{_safe_text(row.get('label'), 'unknown')} ({_safe_int(row.get('object_global_id'), -1)})"
        for row in member_rows
    ]
    return {
        "cluster_id": int(cluster.get("cluster_id", -1)),
        "members": members,
        "member_view_ids": list(cluster.get("member_view_ids", [])),
        "label_histogram": dict(cluster.get("label_histogram", {})),
    }


def _detail_summary(detail: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not detail:
        return None
    return {
        "combined_similarity": _safe_float(detail.get("combined_similarity")),
        "text_similarity": _safe_float(detail.get("text_similarity")),
        "global_geo_similarity": _safe_float(detail.get("global_geo_similarity")),
        "polar_similarity": _safe_float(detail.get("polar_similarity")),
        "global_geo_distance_m": _safe_float(detail.get("global_geo_distance_m")),
        "used_3d_global_geo": bool(detail.get("used_3d_global_geo")),
    }


def _append_case_summary(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "step_index": _safe_int(case.get("step_index"), -1),
        "cluster_id": _safe_int(case.get("cluster_id"), -1),
        "appended_object_id": _safe_int(case.get("appended_object_id"), -1),
        "view_id": _safe_text(case.get("view_id")),
        "reason": _safe_text(case.get("reason")),
        "score": _safe_float(case.get("score")),
        "detail": _detail_summary(case.get("detail")),
    }


def _merge_case_summary(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "step_index": _safe_int(case.get("step_index"), -1),
        "merged_cluster_ids": [_safe_int(value, -1) for value in case.get("merged_cluster_ids", [])],
        "into_cluster_id": _safe_int(case.get("into_cluster_id"), -1),
    }


def _same_view_block_case_summary(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "step_index": _safe_int(case.get("step_index"), -1),
        "blocked_merge_cluster_ids": [_safe_int(value, -1) for value in case.get("blocked_merge_cluster_ids", [])],
        "competing_object_ids": [_safe_int(value, -1) for value in case.get("competing_object_ids", [])],
        "unassigned_object_ids": [_safe_int(value, -1) for value in case.get("unassigned_object_ids", [])],
        "collision_pairs": [
            {
                "left_cluster_id": _safe_int(item.get("left_cluster_id"), -1),
                "right_cluster_id": _safe_int(item.get("right_cluster_id"), -1),
                "shared_view_ids": [_safe_text(value) for value in item.get("shared_view_ids", [])],
            }
            for item in list(case.get("collision_pairs", []))
        ],
        "assignments": [
            {
                "cluster_id": _safe_int(item.get("cluster_id"), -1),
                "object_id": _safe_int(item.get("object_id"), -1),
                "score": _safe_float(item.get("score")),
            }
            for item in list(case.get("assignments", []))
        ],
    }


def _tail_spawn_case_summary(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "step_index": _safe_int(case.get("step_index"), -1),
        "new_cluster_id": _safe_int(case.get("new_cluster_id"), -1),
        "reason": _safe_text(case.get("reason")),
        "object_ids": [_safe_int(value, -1) for value in case.get("object_ids", [])],
        "view_ids": [_safe_text(value) for value in case.get("view_ids", [])],
        "detail": _detail_summary(case.get("detail")),
    }


def _spectral_result_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    labels = result.get("labels")
    return {
        "n_clusters": _safe_int(result.get("n_clusters"), -1),
        "cluster_count_mode": _safe_text(result.get("cluster_count_mode")),
        "backend": _safe_text(result.get("backend")),
        "fallback_reason": result.get("fallback_reason"),
        "num_nodes": int(len(labels) if labels is not None else 0),
    }


def _view_summary(view: Mapping[str, Any], stored_image: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    summary = {
        "view_id": _safe_text(view.get("view_id")),
        "entry_id": _safe_int(view.get("entry_id"), -1),
        "file_name": _safe_text(view.get("file_name")),
        "num_objects": len(view.get("objects", [])),
    }
    if stored_image is not None:
        summary["stored_image_path"] = _safe_text(stored_image.get("stored_image_path"))
        summary["image_status"] = _safe_text(stored_image.get("status"))
        summary["stored_detection_overlay_path"] = _safe_text(stored_image.get("stored_detection_overlay_path"))
        summary["detection_overlay_status"] = _safe_text(stored_image.get("detection_overlay_status"))
    return summary


def _step_report_summary(step_report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "step_index": _safe_int(step_report.get("step_index"), -1),
        "view_id": _safe_text(step_report.get("view_id")),
        "num_current_objects": _safe_int(step_report.get("num_current_objects"), 0),
        "num_existing_clusters": _safe_int(step_report.get("num_existing_clusters"), 0),
        "num_appended": _safe_int(step_report.get("num_appended"), 0),
        "num_current_only_reattached": _safe_int(step_report.get("num_current_only_reattached"), 0),
        "num_merged_clusters": _safe_int(step_report.get("num_merged_clusters"), 0),
        "num_same_view_blocked_components": _safe_int(step_report.get("num_same_view_blocked_components"), 0),
        "num_new_tail_clusters": _safe_int(step_report.get("num_new_tail_clusters"), 0),
        "cross_affinity_shape": list(step_report.get("cross_affinity_shape", [])),
    }


def _label_jitter_summary(cluster: Mapping[str, Any]) -> Dict[str, Any]:
    histogram = dict(cluster.get("label_histogram", {}))
    total = sum(int(value) for value in histogram.values()) or 1
    dominant_label = ""
    dominant_count = 0
    if histogram:
        dominant_label, dominant_count = max(histogram.items(), key=lambda item: (int(item[1]), item[0]))
    return {
        "cluster_id": int(cluster.get("cluster_id", -1)),
        "dominant_label": dominant_label,
        "dominant_label_count": int(dominant_count),
        "purity": float(dominant_count) / float(total),
        "label_histogram": histogram,
        "member_object_ids": list(cluster.get("member_object_ids", [])),
        "member_view_ids": list(cluster.get("member_view_ids", [])),
    }


def _make_run_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base_dir / stamp
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _resolve_view_image_path(db_root: Path, file_name: str) -> Optional[Path]:
    raw = _safe_text(file_name)
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None
    candidate = db_root / raw
    return candidate if candidate.exists() else None


def _resolve_detection_overlay_path(db_root: Path, view_id: str) -> Optional[Path]:
    cleaned_view_id = _safe_text(view_id)
    if not cleaned_view_id:
        return None
    candidates = [
        db_root / "geometry" / cleaned_view_id / "detection_overlay.jpg",
        db_root / "geometry" / cleaned_view_id / "detection_overlay.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _store_selected_view_images(root: Path, db_dir: str, views: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out_dir = root / "selected_view_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_root = Path(db_dir)
    stored: List[Dict[str, Any]] = []
    for view in views:
        view_id = _safe_text(view.get("view_id"))
        source_path = _resolve_view_image_path(db_root, _safe_text(view.get("file_name")))
        if source_path is None:
            stored.append(
                {
                    "view_id": view_id,
                    "entry_id": _safe_int(view.get("entry_id"), -1),
                    "source_file_name": _safe_text(view.get("file_name")),
                    "status": "missing_source_image",
                }
            )
            continue
        destination = out_dir / f"{view_id}{source_path.suffix or '.jpg'}"
        shutil.copy2(source_path, destination)
        overlay_path = _resolve_detection_overlay_path(db_root, view_id)
        stored_overlay_path = None
        overlay_status = "missing_detection_overlay"
        if overlay_path is not None:
            overlay_destination = out_dir / f"{view_id}_yolo_overlay{overlay_path.suffix or '.jpg'}"
            shutil.copy2(overlay_path, overlay_destination)
            stored_overlay_path = str(overlay_destination)
            overlay_status = "copied"
        stored.append(
            {
                "view_id": view_id,
                "entry_id": _safe_int(view.get("entry_id"), -1),
                "source_file_name": _safe_text(view.get("file_name")),
                "source_image_path": str(source_path),
                "stored_image_path": str(destination),
                "status": "copied",
                "source_detection_overlay_path": str(overlay_path) if overlay_path is not None else None,
                "stored_detection_overlay_path": stored_overlay_path,
                "detection_overlay_status": overlay_status,
            }
        )
    return stored


def load_sequence_objects(
    db_dir: str,
    *,
    entry_ids: Optional[Sequence[Any]] = None,
    view_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    root = Path(db_dir)
    manual_entry_ids = _normalize_entry_ids(entry_ids)
    if manual_entry_ids:
        selected_view_ids = [_view_id_for_entry(entry_id) for entry_id in manual_entry_ids]
    else:
        selected_view_ids = _normalize_view_ids(view_ids)
    selected_set = set(selected_view_ids)
    meta_rows = _load_jsonl(root / "meta.jsonl")
    meta_by_view_id: Dict[str, Dict[str, Any]] = {}
    for row in meta_rows:
        entry_id = _safe_int(row.get("id"), -1)
        if entry_id < 0:
            continue
        view_id = _view_id_for_entry(entry_id)
        if view_id in selected_set:
            meta_by_view_id[view_id] = dict(row)

    loaded = load_object_db(str(root), text_mode="long")
    if loaded is None:
        raise FileNotFoundError(f"Missing object_meta.jsonl or object_text_emb_long.npy in {root}")
    object_rows, long_emb, _entry_to_indices = loaded
    if long_emb.shape[0] != len(object_rows):
        raise ValueError("Object long embeddings are misaligned with object_meta.jsonl")

    by_view: Dict[str, List[Dict[str, Any]]] = {view_id: [] for view_id in selected_view_ids}
    for index, row in enumerate(object_rows):
        prepared = dict(row)
        entry_id = _safe_int(prepared.get("entry_id"), -1)
        view_id = _safe_text(prepared.get("view_id")) or _view_id_for_entry(entry_id)
        if view_id not in selected_set:
            continue
        if not _is_valid_object_row(prepared):
            continue
        prepared["view_id"] = view_id
        prepared["entry_id"] = entry_id
        prepared["embedding"] = _l2_normalize_vec(np.asarray(long_emb[index], dtype=np.float32))
        meta_row = meta_by_view_id.get(view_id)
        if meta_row is not None:
            prepared["file_name"] = prepared.get("file_name") or meta_row.get("file_name")
            prepared["orientation"] = _safe_float(meta_row.get("orientation"))
        by_view[view_id].append(prepared)

    ordered_views: List[Dict[str, Any]] = []
    for view_id in selected_view_ids:
        rows = sorted(
            by_view.get(view_id, []),
            key=lambda row: (_safe_int(row.get("object_global_id"), 10**9), _safe_text(row.get("label"))),
        )
        meta_row = meta_by_view_id.get(view_id, {})
        ordered_views.append(
            {
                "view_id": view_id,
                "entry_id": _safe_int(meta_row.get("id"), -1),
                "file_name": _safe_text(meta_row.get("file_name")),
                "orientation": _safe_float(meta_row.get("orientation")),
                "objects": rows,
            }
        )
    return {
        "db_dir": str(root),
        "selected_view_ids": selected_view_ids,
        "views": ordered_views,
    }


def _gaussian_similarity(distance: float, sigma: float) -> float:
    sigma_value = max(float(sigma), 1e-6)
    return float(math.exp(-((float(distance) ** 2) / (sigma_value ** 2))))


def _text_similarity(row: Mapping[str, Any], cluster: Mapping[str, Any]) -> Optional[float]:
    row_vec = row.get("embedding")
    proto_vec = cluster.get("prototype_embedding")
    if row_vec is None or proto_vec is None:
        return None
    return float(np.clip(np.dot(np.asarray(row_vec, dtype=np.float32), np.asarray(proto_vec, dtype=np.float32)), -1.0, 1.0))


def _global_geo_similarity(
    row: Mapping[str, Any],
    cluster: Mapping[str, Any],
    *,
    sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
) -> Tuple[Optional[float], Optional[float], bool]:
    row_x, row_y, row_z = _row_xyz(row)
    proto_xyz = dict(cluster.get("prototype_xyz") or {})
    cluster_x = _safe_float(proto_xyz.get("x"))
    cluster_y = _safe_float(proto_xyz.get("y"))
    cluster_z = _safe_float(proto_xyz.get("z"))
    if row_x is None or row_z is None or cluster_x is None or cluster_z is None:
        return None, None, False
    if row_y is not None and cluster_y is not None:
        delta = np.asarray([row_x - cluster_x, row_y - cluster_y, row_z - cluster_z], dtype=np.float32)
        return _gaussian_similarity(float(np.linalg.norm(delta)), sigma_m), float(np.linalg.norm(delta)), True
    delta = np.asarray([row_x - cluster_x, row_z - cluster_z], dtype=np.float32)
    return _gaussian_similarity(float(np.linalg.norm(delta)), sigma_m), float(np.linalg.norm(delta)), False


def _polar_similarity(row: Mapping[str, Any], cluster: Mapping[str, Any]) -> Optional[float]:
    row_distance, row_bearing, row_height = _row_polar(row)
    proto_polar = dict(cluster.get("prototype_polar") or {})
    cluster_distance = _safe_float(proto_polar.get("distance_from_camera_m"))
    cluster_bearing = _safe_float(proto_polar.get("relative_bearing_deg"))
    cluster_height = _safe_float(proto_polar.get("relative_height_from_camera_m"))
    dims: List[float] = []
    if row_distance is not None and cluster_distance is not None:
        dims.append((float(row_distance) - float(cluster_distance)) / 2.0)
    if row_bearing is not None and cluster_bearing is not None:
        dims.append(_wrap_delta_angle_deg(float(row_bearing) - float(cluster_bearing)) / 45.0)
    if row_height is not None and cluster_height is not None:
        dims.append((float(row_height) - float(cluster_height)) / 1.0)
    if not dims:
        return None
    normalized_distance = float(np.linalg.norm(np.asarray(dims, dtype=np.float32)))
    return _gaussian_similarity(normalized_distance, 1.0)


def _pair_affinity_detail(
    row: Mapping[str, Any],
    cluster: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    global_sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
) -> Dict[str, Any]:
    text_sim = _text_similarity(row, cluster)
    geo_sim, geo_distance, used_3d_geo = _global_geo_similarity(row, cluster, sigma_m=global_sigma_m)
    polar_sim = _polar_similarity(row, cluster)
    raw_scores = {
        "text_similarity": text_sim,
        "global_geo_similarity": geo_sim,
        "polar_similarity": polar_sim,
    }
    available_weights = {
        "text": weights["text"] if text_sim is not None else 0.0,
        "global_geo": weights["global_geo"] if geo_sim is not None else 0.0,
        "polar": weights["polar"] if polar_sim is not None else 0.0,
    }
    weight_total = sum(available_weights.values())
    if weight_total <= 0.0:
        combined = 0.0
        normalized_weights = {key: 0.0 for key in available_weights}
    else:
        normalized_weights = {key: value / weight_total for key, value in available_weights.items()}
        combined = 0.0
        if text_sim is not None:
            combined += normalized_weights["text"] * float(text_sim)
        if geo_sim is not None:
            combined += normalized_weights["global_geo"] * float(geo_sim)
        if polar_sim is not None:
            combined += normalized_weights["polar"] * float(polar_sim)
    return {
        "combined_similarity": float(combined),
        "text_similarity": text_sim,
        "global_geo_similarity": geo_sim,
        "polar_similarity": polar_sim,
        "global_geo_distance_m": geo_distance,
        "used_3d_global_geo": bool(used_3d_geo),
        "normalized_weights": normalized_weights,
    }


def build_cross_affinity_matrix(
    memory_clusters: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    *,
    weight_text: float = DEFAULT_WEIGHT_TEXT,
    weight_global_geo: float = DEFAULT_WEIGHT_GLOBAL_GEO,
    weight_polar: float = DEFAULT_WEIGHT_POLAR,
    global_sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
    weights = _normalize_weight_triplet(weight_text, weight_global_geo, weight_polar)
    matrix = np.zeros((len(current_rows), len(memory_clusters)), dtype=np.float32)
    details: List[List[Dict[str, Any]]] = []
    for row in current_rows:
        row_details: List[Dict[str, Any]] = []
        for cluster in memory_clusters:
            detail = _pair_affinity_detail(row, cluster, weights=weights, global_sigma_m=global_sigma_m)
            row_details.append(detail)
        details.append(row_details)
    for row_index, row_details in enumerate(details):
        for col_index, detail in enumerate(row_details):
            matrix[row_index, col_index] = float(detail["combined_similarity"])
    return matrix, details


def _full_bipartite_affinity(cross_affinity: np.ndarray, *, min_cross_affinity: float = DEFAULT_CROSS_AFFINITY_MIN) -> np.ndarray:
    current_count, memory_count = cross_affinity.shape
    pruned = np.asarray(cross_affinity, dtype=np.float32).copy()
    if current_count and memory_count:
        pruned[pruned < float(min_cross_affinity)] = 0.0
    total = memory_count + current_count
    affinity = np.eye(total, dtype=np.float32)
    if current_count and memory_count:
        affinity[memory_count:, :memory_count] = pruned
        affinity[:memory_count, memory_count:] = pruned.T
    return affinity


def _connectivity_labels(affinity: np.ndarray) -> np.ndarray:
    if affinity.size == 0:
        return np.zeros((0,), dtype=np.int32)
    adjacency = np.asarray(affinity > 0.0, dtype=np.int8)
    np.fill_diagonal(adjacency, 0)
    graph = csr_matrix(adjacency)
    _n_components, labels = connected_components(graph, directed=False, connection="weak")
    return np.asarray(labels, dtype=np.int32)


def _node_labels(memory_clusters: Sequence[Mapping[str, Any]], current_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    labels: List[str] = []
    for cluster in memory_clusters:
        labels.append(f"mem:c{int(cluster.get('cluster_id', -1))}")
    for row in current_rows:
        labels.append(f"cur:obj{_safe_int(row.get('object_global_id'), -1)}@{_safe_text(row.get('view_id'))}")
    return labels


def _write_affinity_csv(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *[f"{float(value):.6f}" for value in row]])


def _write_rect_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix, dtype=np.float32)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *list(col_labels)])
        for label, row in zip(list(row_labels), values.tolist()):
            writer.writerow([label, *[f"{float(value):.6f}" for value in row]])


def _normalized_laplacian_matrix(affinity_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(affinity_matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected square affinity matrix for Laplacian, got shape {values.shape}")
    if values.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    lap = laplacian(values, normed=True)
    lap = np.asarray(lap, dtype=np.float32)
    lap = np.nan_to_num(lap, nan=0.0, posinf=0.0, neginf=0.0)
    return lap


def _write_laplacian_artifacts(
    root: Path,
    *,
    stem: str,
    affinity_matrix: np.ndarray,
    axis_labels: Sequence[str],
    title: str,
) -> np.ndarray:
    lap_matrix = _normalized_laplacian_matrix(affinity_matrix)
    np.save(root / f"{stem}.npy", lap_matrix)
    _write_affinity_csv(root / f"{stem}.csv", lap_matrix, axis_labels)
    plot_similarity_heatmap(
        lap_matrix,
        root / f"{stem}.png",
        title=title,
        axis_labels=axis_labels,
        annotate_values=False,
        vmin=float(np.min(lap_matrix)) if lap_matrix.size else 0.0,
        vmax=float(np.max(lap_matrix)) if lap_matrix.size else 1.0,
    )
    return lap_matrix


def _parse_member_object_id(member: Any) -> Optional[int]:
    text = _safe_text(member)
    if not text:
        return None
    match = re.search(r"\((\d+)\)\s*$", text)
    if match is None:
        return None
    return int(match.group(1))


def _snapshot_cluster_member_ids(cluster: Mapping[str, Any]) -> List[int]:
    explicit = cluster.get("member_object_ids")
    if isinstance(explicit, list) and explicit:
        return [_safe_int(value, -1) for value in explicit if _safe_int(value, -1) >= 0]
    members = cluster.get("members")
    if isinstance(members, list):
        parsed = [_parse_member_object_id(value) for value in members]
        return [int(value) for value in parsed if value is not None and int(value) >= 0]
    return []


def _load_snapshot_clusters(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "clusters_after_step" in payload:
        clusters = payload.get("clusters_after_step")
    else:
        clusters = payload.get("clusters")
    return [dict(item) for item in list(clusters or [])]


def _load_selected_object_rows(db_dir: str, selected_view_ids: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    selected = {_safe_text(view_id) for view_id in selected_view_ids if _safe_text(view_id)}
    rows_by_id: Dict[int, Dict[str, Any]] = {}
    for row in _load_jsonl(Path(db_dir) / "object_meta.jsonl"):
        object_id = _safe_int(row.get("object_global_id"), -1)
        if object_id < 0:
            continue
        view_id = _safe_text(row.get("view_id")) or _view_id_for_entry(_safe_int(row.get("entry_id"), -1))
        if selected and view_id not in selected:
            continue
        rows_by_id[object_id] = dict(row)
    return rows_by_id


def _progression_snapshots(run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[int, Dict[str, Any]]]:
    manifest_path = run_dir / "sequence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_view_ids = list(manifest.get("selected_view_ids") or [])
    object_rows = _load_selected_object_rows(str(manifest["db_dir"]), selected_view_ids)

    snapshots: List[Dict[str, Any]] = []
    initial_path = run_dir / "step_00_initial_registry.json"
    initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
    snapshots.append(
        {
            "stage_index": 0,
            "view_id": _safe_text(initial_payload.get("initial_view_id")),
            "clusters": [dict(item) for item in list(initial_payload.get("clusters") or [])],
            "source_path": str(initial_path),
        }
    )
    step_index = 1
    while True:
        step_path = run_dir / f"step_{step_index:02d}_cluster_update.json"
        if not step_path.exists():
            break
        payload = json.loads(step_path.read_text(encoding="utf-8"))
        snapshots.append(
            {
                "stage_index": int(step_index),
                "view_id": _safe_text(payload.get("view_id")),
                "clusters": [dict(item) for item in list(payload.get("clusters_after_step") or [])],
                "source_path": str(step_path),
            }
        )
        step_index += 1
    return snapshots, manifest, object_rows


def _cluster_sort_key(cluster: Mapping[str, Any]) -> Tuple[int, int]:
    cluster_id = _safe_int(cluster.get("cluster_id"), 10**9)
    member_ids = _snapshot_cluster_member_ids(cluster)
    return cluster_id, min(member_ids) if member_ids else 10**9


def _ordered_snapshot_members(
    clusters: Sequence[Mapping[str, Any]],
    *,
    object_rows: Mapping[int, Mapping[str, Any]],
    first_seen_step_by_object: Mapping[int, int],
    view_order_by_id: Mapping[str, int],
) -> Tuple[List[int], List[int], List[str], List[int]]:
    ordered_ids: List[int] = []
    boundaries: List[int] = []
    axis_labels: List[str] = []
    ordered_first_seen: List[int] = []

    for cluster in sorted(clusters, key=_cluster_sort_key):
        member_ids = _snapshot_cluster_member_ids(cluster)
        sorted_members = sorted(
            member_ids,
            key=lambda object_id: (
                _safe_int(first_seen_step_by_object.get(int(object_id)), 10**9),
                _safe_int(
                    view_order_by_id.get(
                        _safe_text(object_rows.get(int(object_id), {}).get("view_id"))
                        or _view_id_for_entry(_safe_int(object_rows.get(int(object_id), {}).get("entry_id"), -1))
                    ),
                    10**9,
                ),
                _safe_text(object_rows.get(int(object_id), {}).get("label"), "unknown"),
                int(object_id),
            ),
        )
        if not sorted_members:
            continue
        for object_id in sorted_members:
            row = object_rows.get(int(object_id), {})
            label = _safe_text(row.get("label"), "unknown")
            first_seen = int(first_seen_step_by_object.get(int(object_id), 0))
            ordered_ids.append(int(object_id))
            ordered_first_seen.append(first_seen)
            axis_labels.append(f"obj{int(object_id)}|{label}")
        boundaries.append(len(ordered_ids) - 1)
    if boundaries:
        boundaries = [index for index in boundaries[:-1] if index >= 0]
    return ordered_ids, boundaries, axis_labels, ordered_first_seen


def _cumulative_cocluster_matrix(
    clusters: Sequence[Mapping[str, Any]],
    *,
    ordered_ids: Sequence[int],
) -> np.ndarray:
    size = len(ordered_ids)
    matrix = np.zeros((size, size), dtype=np.float32)
    if size == 0:
        return matrix
    index_by_object = {int(object_id): idx for idx, object_id in enumerate(ordered_ids)}
    np.fill_diagonal(matrix, 1.0)
    for cluster in clusters:
        indices = [index_by_object[object_id] for object_id in _snapshot_cluster_member_ids(cluster) if object_id in index_by_object]
        if not indices:
            continue
        idx = np.asarray(indices, dtype=np.int64)
        matrix[np.ix_(idx, idx)] = 1.0
    return matrix


def _step_color(step_index: int) -> Tuple[int, int, int]:
    palette = list(SEQUENTIAL_PROGRESS_COLORS_BGR)
    if not palette:
        return 128, 128, 128
    return tuple(int(value) for value in palette[int(step_index) % len(palette)])


def _plot_cumulative_cluster_heatmap(
    matrix: np.ndarray,
    output_path: Path,
    *,
    title: str,
    axis_labels: Sequence[str],
    first_seen_steps: Sequence[int],
    current_stage_index: int,
    stage_labels: Sequence[str],
    boundary_after_indices: Optional[Sequence[int]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix, dtype=np.float32)
    count = int(values.shape[0])
    size = max(280, count * 20) if count > 0 else 280
    strip = 14 if count > 0 else 0
    display_labels = [_truncate_heatmap_label(label, max_chars=26) for label in list(axis_labels or [])]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32 if count >= 72 else (0.36 if count >= 48 else 0.42)
    thickness = 1
    label_widths = []
    for label in display_labels:
        (text_w, _text_h), _baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_widths.append(int(text_w))

    left_margin = max(84, max(label_widths, default=0) + 26)
    top_margin = max(100, max(label_widths, default=0) + 36)
    right_margin = 26
    bottom_margin = 32
    canvas_w = max(560, left_margin + strip + size + right_margin)
    canvas_h = max(560, top_margin + strip + size + bottom_margin)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, str(title)[:96], (18, 28), font, 0.70, (25, 25, 25), 2, cv2.LINE_AA)

    if count == 0:
        cv2.putText(canvas, "No objects", (190, 280), font, 0.85, (80, 80, 80), 2, cv2.LINE_AA)
        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"Failed to save cumulative heatmap: {output_path}")
        return

    heat_u8 = np.asarray(np.clip(values, 0.0, 1.0) * 255.0, dtype=np.uint8)
    heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_NEAREST)
    x0 = left_margin + strip
    y0 = top_margin + strip
    x1 = x0 + size
    y1 = y0 + size
    canvas[y0:y1, x0:x1] = heatmap
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 40), 1)

    cell_w = float(size) / float(count)
    cell_h = float(size) / float(count)
    for idx, step_index in enumerate(list(first_seen_steps)[:count]):
        color = _step_color(int(step_index))
        x_start = int(round(x0 + idx * cell_w))
        x_end = int(round(x0 + (idx + 1) * cell_w))
        y_start = int(round(y0 + idx * cell_h))
        y_end = int(round(y0 + (idx + 1) * cell_h))
        cv2.rectangle(canvas, (x_start, top_margin), (x_end, top_margin + strip), color, -1)
        cv2.rectangle(canvas, (left_margin, y_start), (left_margin + strip, y_end), color, -1)
        if int(step_index) == int(current_stage_index):
            cv2.rectangle(canvas, (x_start, top_margin), (x_end, top_margin + strip), (255, 255, 255), 2)
            cv2.rectangle(canvas, (left_margin, y_start), (left_margin + strip, y_end), (255, 255, 255), 2)

    cv2.rectangle(canvas, (x0, top_margin), (x1, top_margin + strip), (55, 55, 55), 1)
    cv2.rectangle(canvas, (left_margin, y0), (left_margin + strip, y1), (55, 55, 55), 1)

    if boundary_after_indices:
        boundary_color = (250, 250, 250)
        shadow_color = (35, 35, 35)
        valid_boundaries = sorted({int(idx) for idx in boundary_after_indices if 0 <= int(idx) < count - 1})
        for boundary_index in valid_boundaries:
            x_boundary = int(round(x0 + (boundary_index + 1) * cell_w))
            y_boundary = int(round(y0 + (boundary_index + 1) * cell_h))
            cv2.line(canvas, (x_boundary, y0), (x_boundary, y1), shadow_color, 3, cv2.LINE_AA)
            cv2.line(canvas, (x_boundary, y0), (x_boundary, y1), boundary_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (x0, y_boundary), (x1, y_boundary), shadow_color, 3, cv2.LINE_AA)
            cv2.line(canvas, (x0, y_boundary), (x1, y_boundary), boundary_color, 1, cv2.LINE_AA)

    if display_labels:
        for row_index, label in enumerate(display_labels[:count]):
            text_y = int(round(y0 + (row_index + 0.5) * cell_h + 4))
            cv2.putText(
                canvas,
                label,
                (12, text_y),
                font,
                font_scale,
                (25, 25, 25),
                thickness,
                cv2.LINE_AA,
            )
        for col_index, label in enumerate(display_labels[:count]):
            center_x = int(round(x0 + (col_index + 0.5) * cell_w))
            rotated = _make_rotated_text_image(
                label,
                font_scale=font_scale,
                thickness=thickness,
                angle_deg=-90,
            )
            rx = center_x - rotated.shape[1] // 2
            ry = max(34, y0 - rotated.shape[0] - 6)
            _alpha_blit(canvas, rotated, rx, ry)

    legend_x = 18
    legend_y = 48
    for stage_index, label in enumerate(stage_labels):
        color = _step_color(stage_index)
        top = legend_y + stage_index * 18
        cv2.rectangle(canvas, (legend_x, top), (legend_x + 12, top + 12), color, -1)
        border_color = (255, 255, 255) if stage_index == int(current_stage_index) else (55, 55, 55)
        cv2.rectangle(canvas, (legend_x, top), (legend_x + 12, top + 12), border_color, 1)
        cv2.putText(canvas, str(label)[:46], (legend_x + 18, top + 11), font, 0.42, (35, 35, 35), 1, cv2.LINE_AA)

    footer = f"Current step highlighted in axis strips | n={count}"
    cv2.putText(canvas, footer[:80], (18, canvas.shape[0] - 10), font, 0.45, (55, 55, 55), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to save cumulative heatmap: {output_path}")


def _build_progression_overview(
    image_paths: Sequence[Path],
    *,
    output_path: Path,
    cols: int = 2,
) -> Optional[Path]:
    paths = [Path(path) for path in image_paths if Path(path).exists()]
    if not paths:
        return None
    images = []
    max_w = 0
    max_h = 0
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images.append(image)
        max_h = max(max_h, int(image.shape[0]))
        max_w = max(max_w, int(image.shape[1]))
    if not images:
        return None
    cols = max(1, int(cols))
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = np.full((rows * max_h, cols * max_w, 3), 255, dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        y0 = row * max_h
        x0 = col * max_w
        canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to save progression overview: {output_path}")
    return output_path


def generate_cumulative_cluster_progression_artifacts(run_dir: str) -> Dict[str, Any]:
    root = Path(run_dir)
    snapshots, manifest, object_rows = _progression_snapshots(root)
    selected_view_ids = list(manifest.get("selected_view_ids") or [])
    view_order_by_id = {str(view_id): index for index, view_id in enumerate(selected_view_ids)}
    first_seen_step_by_object: Dict[int, int] = {}
    for object_id, row in object_rows.items():
        view_id = _safe_text(row.get("view_id")) or _view_id_for_entry(_safe_int(row.get("entry_id"), -1))
        first_seen_step_by_object[int(object_id)] = int(view_order_by_id.get(view_id, 0))

    stage_labels = [f"S{index}: {_safe_text(view_id) or f'view_{index:02d}'}" for index, view_id in enumerate(selected_view_ids)]
    progression_entries: List[Dict[str, Any]] = []
    heatmap_paths: List[Path] = []
    for snapshot in snapshots:
        stage_index = int(snapshot["stage_index"])
        ordered_ids, boundaries, axis_labels, ordered_first_seen = _ordered_snapshot_members(
            snapshot["clusters"],
            object_rows=object_rows,
            first_seen_step_by_object=first_seen_step_by_object,
            view_order_by_id=view_order_by_id,
        )
        matrix = _cumulative_cocluster_matrix(snapshot["clusters"], ordered_ids=ordered_ids)
        stage_stem = f"cumulative_cluster_matrix_step_{stage_index:02d}"
        np.save(root / f"{stage_stem}.npy", matrix)
        _write_affinity_csv(root / f"{stage_stem}.csv", matrix, axis_labels)
        title = f"Cumulative Co-Cluster Step {stage_index:02d} (+ {_safe_text(snapshot.get('view_id')) or 'initial'})"
        heatmap_path = root / f"{stage_stem}.png"
        _plot_cumulative_cluster_heatmap(
            matrix,
            heatmap_path,
            title=title,
            axis_labels=axis_labels,
            first_seen_steps=ordered_first_seen,
            current_stage_index=stage_index,
            stage_labels=stage_labels,
            boundary_after_indices=boundaries,
        )
        heatmap_paths.append(heatmap_path)
        current_step_object_ids = [object_id for object_id in ordered_ids if int(first_seen_step_by_object.get(int(object_id), -1)) == stage_index]
        entry = {
            "stage_index": stage_index,
            "view_id": _safe_text(snapshot.get("view_id")),
            "num_objects": len(ordered_ids),
            "ordered_object_ids": [int(object_id) for object_id in ordered_ids],
            "axis_labels": list(axis_labels),
            "ordered_first_seen_steps": [int(value) for value in ordered_first_seen],
            "boundary_after_indices": [int(value) for value in boundaries],
            "new_object_ids": [int(object_id) for object_id in current_step_object_ids],
            "matrix_path": str(root / f"{stage_stem}.npy"),
            "csv_path": str(root / f"{stage_stem}.csv"),
            "heatmap_path": str(heatmap_path),
        }
        progression_entries.append(entry)
        _write_json(root / f"{stage_stem}.json", entry)

    overview_path = _build_progression_overview(
        heatmap_paths,
        output_path=root / "cumulative_cluster_progression_overview.png",
        cols=2,
    )
    manifest_payload = {
        "run_dir": str(root),
        "selected_view_ids": selected_view_ids,
        "stage_labels": stage_labels,
        "steps": progression_entries,
        "overview_path": None if overview_path is None else str(overview_path),
    }
    _write_json(root / "cumulative_cluster_progression_manifest.json", manifest_payload)
    return manifest_payload


def _group_component_nodes(
    spectral_labels: Sequence[int],
    connectivity_labels: Sequence[int],
    *,
    num_memory: int,
) -> List[Dict[str, List[int]]]:
    grouped: Dict[Tuple[int, int], Dict[str, List[int]]] = {}
    for node_index, (spectral_label, cc_label) in enumerate(zip(spectral_labels, connectivity_labels)):
        key = (int(spectral_label), int(cc_label))
        bucket = grouped.setdefault(key, {"memory_indices": [], "current_indices": []})
        if node_index < num_memory:
            bucket["memory_indices"].append(int(node_index))
        else:
            bucket["current_indices"].append(int(node_index - num_memory))
    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            min(item["memory_indices"]) if item["memory_indices"] else 10**9,
            min(item["current_indices"]) if item["current_indices"] else 10**9,
        ),
    )
    for item in ordered:
        item["memory_indices"].sort()
        item["current_indices"].sort()
    return ordered


def _co_cluster_matrix(
    *,
    spectral_result: Mapping[str, Any],
    full_affinity: np.ndarray,
    num_memory: int,
) -> Tuple[np.ndarray, List[Dict[str, List[int]]]]:
    spectral_labels = np.asarray(spectral_result.get("labels", []), dtype=np.int32)
    connectivity_labels = _connectivity_labels(full_affinity)
    components = _group_component_nodes(spectral_labels, connectivity_labels, num_memory=num_memory)
    total = int(full_affinity.shape[0])
    matrix = np.zeros((total, total), dtype=np.float32)
    if total == 0:
        return matrix, components
    np.fill_diagonal(matrix, 1.0)
    for component in components:
        node_indices = list(component.get("memory_indices", [])) + [
            num_memory + int(index) for index in component.get("current_indices", [])
        ]
        for idx in node_indices:
            matrix[idx, node_indices] = 1.0
    return matrix, components


def _step5_block_order(
    memory_clusters: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    *,
    spectral_result: Mapping[str, Any],
    full_affinity: np.ndarray,
) -> Tuple[List[int], List[int]]:
    spectral_labels = np.asarray(spectral_result.get("labels", []), dtype=np.int32)
    if spectral_labels.size == 0:
        return [], []
    connectivity_labels = _connectivity_labels(full_affinity)
    components = _group_component_nodes(
        spectral_labels,
        connectivity_labels,
        num_memory=len(memory_clusters),
    )
    order: List[int] = []
    boundaries: List[int] = []
    for component in components:
        memory_indices = sorted(
            [int(index) for index in component.get("memory_indices", [])],
            key=lambda index: int(memory_clusters[index].get("cluster_id", 10**9)),
        )
        current_indices = sorted(
            [int(index) for index in component.get("current_indices", [])],
            key=lambda index: (
                _safe_text(current_rows[index].get("label"), "unknown"),
                _safe_int(current_rows[index].get("object_global_id"), 10**9),
                index,
            ),
        )
        component_nodes = memory_indices + [len(memory_clusters) + index for index in current_indices]
        if not component_nodes:
            continue
        order.extend(component_nodes)
        boundaries.append(len(order) - 1)
    if boundaries:
        boundaries = boundaries[:-1]
    return order, boundaries


def _create_new_cluster_from_rows(rows: Sequence[Mapping[str, Any]], next_cluster_id: int) -> Dict[str, Any]:
    return _build_cluster(int(next_cluster_id), rows)


def apply_incremental_step(
    memory_clusters: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    *,
    cross_affinity: np.ndarray,
    cross_details: Sequence[Sequence[Mapping[str, Any]]],
    full_affinity: np.ndarray,
    spectral_result: Mapping[str, Any],
    step_index: int,
    next_cluster_id: int,
    weight_text: float = DEFAULT_WEIGHT_TEXT,
    weight_global_geo: float = DEFAULT_WEIGHT_GLOBAL_GEO,
    weight_polar: float = DEFAULT_WEIGHT_POLAR,
    global_sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
    current_only_reattach_min_affinity: float = DEFAULT_CURRENT_ONLY_REATTACH_MIN_AFFINITY,
) -> Dict[str, Any]:
    slots: List[Optional[Dict[str, Any]]] = [deepcopy(dict(cluster)) for cluster in memory_clusters]
    tail_clusters: List[Dict[str, Any]] = []
    append_cases: List[Dict[str, Any]] = []
    merge_cases: List[Dict[str, Any]] = []
    current_only_reattach_cases: List[Dict[str, Any]] = []
    same_view_block_cases: List[Dict[str, Any]] = []
    tail_spawn_cases: List[Dict[str, Any]] = []
    processed_current_indices: set[int] = set()

    spectral_labels = np.asarray(spectral_result.get("labels", []), dtype=np.int32)
    connectivity_labels = _connectivity_labels(full_affinity)
    components = _group_component_nodes(spectral_labels, connectivity_labels, num_memory=len(memory_clusters))

    running_cluster_id = int(next_cluster_id)

    for component in components:
        mem_indices = list(component["memory_indices"])
        cur_indices = list(component["current_indices"])
        if not mem_indices and not cur_indices:
            continue

        base_cluster: Optional[Dict[str, Any]] = None
        base_slot_index: Optional[int] = None
        if mem_indices:
            indexed_source_clusters = [
                (int(idx), slots[idx]) for idx in mem_indices if 0 <= int(idx) < len(slots) and slots[idx] is not None
            ]
            if not indexed_source_clusters:
                continue
            collision_pairs = _same_view_collision_pairs(
                [(idx, cluster) for idx, cluster in indexed_source_clusters if cluster is not None]
            )
            if collision_pairs:
                cluster_id_by_slot = {
                    int(idx): _safe_int(cluster.get("cluster_id"), 10**9)
                    for idx, cluster in indexed_source_clusters
                    if cluster is not None
                }
                candidate_edges: List[Tuple[float, int, int, Dict[str, Any]]] = []
                for cur_idx in cur_indices:
                    for mem_idx, cluster in indexed_source_clusters:
                        if cluster is None:
                            continue
                        detail = _cross_detail_for_pair(
                            int(cur_idx),
                            int(mem_idx),
                            cross_details=cross_details,
                            current_rows=current_rows,
                            cluster=cluster,
                            weight_text=weight_text,
                            weight_global_geo=weight_global_geo,
                            weight_polar=weight_polar,
                            global_sigma_m=global_sigma_m,
                        )
                        edge_score = float(detail.get("combined_similarity") or 0.0)
                        if (
                            0 <= int(cur_idx) < int(cross_affinity.shape[0])
                            and 0 <= int(mem_idx) < int(cross_affinity.shape[1])
                        ):
                            edge_score = float(cross_affinity[int(cur_idx), int(mem_idx)])
                        if edge_score <= 0.0:
                            continue
                        candidate_edges.append(
                            (
                                float(detail.get("combined_similarity") or 0.0),
                                int(cur_idx),
                                int(mem_idx),
                                detail,
                            )
                        )
                candidate_edges.sort(
                    key=lambda item: (
                        -float(item[0]),
                        int(cluster_id_by_slot.get(int(item[2]), 10**9)),
                        int(item[1]),
                    )
                )

                assigned_cur_indices: set[int] = set()
                assigned_mem_indices: set[int] = set()
                assignments: List[Tuple[float, int, int, Dict[str, Any]]] = []
                for score, cur_idx, mem_idx, detail in candidate_edges:
                    if int(cur_idx) in assigned_cur_indices or int(mem_idx) in assigned_mem_indices:
                        continue
                    assigned_cur_indices.add(int(cur_idx))
                    assigned_mem_indices.add(int(mem_idx))
                    assignments.append((float(score), int(cur_idx), int(mem_idx), detail))

                for score, cur_idx, mem_idx, detail in assignments:
                    slot_cluster = slots[mem_idx]
                    if slot_cluster is None:
                        continue
                    best_row = dict(current_rows[cur_idx])
                    updated_cluster = _append_member(slot_cluster, best_row)
                    slots[mem_idx] = updated_cluster
                    processed_current_indices.add(int(cur_idx))
                    append_cases.append(
                        {
                            "step_index": int(step_index),
                            "cluster_id": int(updated_cluster["cluster_id"]),
                            "appended_object_id": _safe_int(best_row.get("object_global_id"), -1),
                            "view_id": _safe_text(best_row.get("view_id")),
                            "reason": "same_view_hard_block_competition_append",
                            "score": float(score),
                            "detail": detail,
                        }
                    )

                unassigned_cur_indices = [int(cur_idx) for cur_idx in cur_indices if int(cur_idx) not in assigned_cur_indices]
                for cur_idx in unassigned_cur_indices:
                    new_cluster = _create_new_cluster_from_rows([current_rows[cur_idx]], running_cluster_id)
                    running_cluster_id += 1
                    tail_clusters.append(new_cluster)
                    processed_current_indices.add(int(cur_idx))
                    tail_spawn_cases.append(
                        {
                            "step_index": int(step_index),
                            "new_cluster_id": int(new_cluster["cluster_id"]),
                            "reason": "tail_after_same_view_hard_block_competition",
                            "object_ids": [_safe_int(current_rows[cur_idx].get("object_global_id"), -1)],
                            "view_ids": [_safe_text(current_rows[cur_idx].get("view_id"))],
                        }
                    )

                same_view_block_cases.append(
                    {
                        "step_index": int(step_index),
                        "blocked_merge_cluster_ids": [
                            _safe_int(cluster.get("cluster_id"), -1)
                            for _idx, cluster in indexed_source_clusters
                            if cluster is not None
                        ],
                        "collision_pairs": collision_pairs,
                        "competing_object_ids": [
                            _safe_int(current_rows[cur_idx].get("object_global_id"), -1) for cur_idx in cur_indices
                        ],
                        "assignments": [
                            {
                                "cluster_id": _safe_int(slots[mem_idx].get("cluster_id") if slots[mem_idx] is not None else -1, -1),
                                "object_id": _safe_int(current_rows[cur_idx].get("object_global_id"), -1),
                                "score": float(score),
                            }
                            for score, cur_idx, mem_idx, _detail in assignments
                        ],
                        "unassigned_object_ids": [
                            _safe_int(current_rows[cur_idx].get("object_global_id"), -1)
                            for cur_idx in unassigned_cur_indices
                        ],
                    }
                )
                continue

            merged_source_clusters = [cluster for _idx, cluster in indexed_source_clusters if cluster is not None]
            base_cluster = _merge_clusters(merged_source_clusters)
            base_slot_index = int(indexed_source_clusters[0][0])
            if len(indexed_source_clusters) > 1:
                merge_cases.append(
                    {
                        "step_index": int(step_index),
                        "merged_cluster_ids": [int(cluster.get("cluster_id", -1)) for cluster in merged_source_clusters],
                        "into_cluster_id": int(base_cluster.get("cluster_id", -1)),
                    }
                )
            slots[base_slot_index] = base_cluster
            for idx, _cluster in indexed_source_clusters[1:]:
                slots[idx] = None

        if not cur_indices:
            continue

        if base_cluster is None:
            reattached_cur_indices: set[int] = set()
            reattach_candidates: List[Tuple[float, int, int, Dict[str, Any]]] = []
            for cur_idx in cur_indices:
                row = dict(current_rows[cur_idx])
                best_match = _best_live_memory_match(
                    row,
                    slots,
                    weight_text=weight_text,
                    weight_global_geo=weight_global_geo,
                    weight_polar=weight_polar,
                    global_sigma_m=global_sigma_m,
                )
                if best_match is None:
                    continue
                best_score, best_mem_idx, best_detail = best_match
                if float(best_score) < float(current_only_reattach_min_affinity):
                    continue
                reattach_candidates.append((float(best_score), int(cur_idx), int(best_mem_idx), best_detail))
            reattach_candidates.sort(key=lambda item: (-item[0], item[2], item[1]))

            for score, cur_idx, mem_idx, detail in reattach_candidates:
                if int(cur_idx) in reattached_cur_indices:
                    continue
                slot_cluster = slots[mem_idx]
                if slot_cluster is None:
                    continue
                row = dict(current_rows[cur_idx])
                updated_cluster = _append_member(slot_cluster, row)
                slots[mem_idx] = updated_cluster
                reattached_cur_indices.add(int(cur_idx))
                processed_current_indices.add(int(cur_idx))
                append_cases.append(
                    {
                        "step_index": int(step_index),
                        "cluster_id": int(updated_cluster["cluster_id"]),
                        "appended_object_id": _safe_int(row.get("object_global_id"), -1),
                        "view_id": _safe_text(row.get("view_id")),
                        "reason": "current_only_high_score_reattach",
                        "score": float(score),
                        "detail": detail,
                    }
                )
                current_only_reattach_cases.append(
                    {
                        "step_index": int(step_index),
                        "cluster_id": int(updated_cluster["cluster_id"]),
                        "appended_object_id": _safe_int(row.get("object_global_id"), -1),
                        "view_id": _safe_text(row.get("view_id")),
                        "reason": "current_only_high_score_reattach",
                        "score": float(score),
                        "detail": detail,
                    }
                )

            residual_rows = [dict(current_rows[idx]) for idx in cur_indices if int(idx) not in reattached_cur_indices]
            if residual_rows:
                for idx in cur_indices:
                    if int(idx) not in reattached_cur_indices:
                        processed_current_indices.add(int(idx))
                new_cluster = _create_new_cluster_from_rows(residual_rows, running_cluster_id)
                running_cluster_id += 1
                tail_clusters.append(new_cluster)
                tail_spawn_cases.append(
                    {
                        "step_index": int(step_index),
                        "new_cluster_id": int(new_cluster["cluster_id"]),
                        "reason": "current_only_component",
                        "object_ids": [_safe_int(row.get("object_global_id"), -1) for row in residual_rows],
                        "view_ids": sorted({_safe_text(row.get("view_id")) for row in residual_rows}),
                    }
                )
            continue

        candidate_rows = [dict(current_rows[idx]) for idx in cur_indices]
        scored_candidates: List[Tuple[float, int, Dict[str, Any]]] = []
        for cur_idx, candidate in zip(cur_indices, candidate_rows):
            detail = _pair_affinity_detail(
                candidate,
                base_cluster,
                weights=_normalize_weight_triplet(weight_text, weight_global_geo, weight_polar),
                global_sigma_m=global_sigma_m,
            )
            scored_candidates.append((float(detail["combined_similarity"]), int(cur_idx), detail))
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        best_score, best_cur_idx, best_detail = scored_candidates[0]
        best_row = dict(current_rows[best_cur_idx])
        base_cluster = _append_member(base_cluster, best_row)
        if base_slot_index is None:
            raise RuntimeError("Missing base slot index while appending current object")
        slots[base_slot_index] = base_cluster
        processed_current_indices.add(int(best_cur_idx))
        append_cases.append(
            {
                "step_index": int(step_index),
                "cluster_id": int(base_cluster["cluster_id"]),
                "appended_object_id": _safe_int(best_row.get("object_global_id"), -1),
                "view_id": _safe_text(best_row.get("view_id")),
                "reason": "component_best_append",
                "score": float(best_score),
                "detail": best_detail,
            }
        )

        for _score, cur_idx, detail in scored_candidates[1:]:
            new_cluster = _create_new_cluster_from_rows([current_rows[cur_idx]], running_cluster_id)
            running_cluster_id += 1
            tail_clusters.append(new_cluster)
            processed_current_indices.add(int(cur_idx))
            tail_spawn_cases.append(
                {
                    "step_index": int(step_index),
                    "new_cluster_id": int(new_cluster["cluster_id"]),
                    "reason": "tail_after_competing_for_existing_cluster",
                    "object_ids": [_safe_int(current_rows[cur_idx].get("object_global_id"), -1)],
                    "view_ids": [_safe_text(current_rows[cur_idx].get("view_id"))],
                    "detail": detail,
                }
            )

    for cur_index, row in enumerate(current_rows):
        if cur_index in processed_current_indices:
            continue
        new_cluster = _create_new_cluster_from_rows([row], running_cluster_id)
        running_cluster_id += 1
        tail_clusters.append(new_cluster)
        tail_spawn_cases.append(
            {
                "step_index": int(step_index),
                "new_cluster_id": int(new_cluster["cluster_id"]),
                "reason": "unprocessed_current_singleton",
                "object_ids": [_safe_int(row.get("object_global_id"), -1)],
                "view_ids": [_safe_text(row.get("view_id"))],
            }
        )

    next_memory = [cluster for cluster in slots if cluster is not None]
    next_memory.extend(tail_clusters)
    next_memory = sorted(next_memory, key=lambda item: int(item.get("cluster_id", 10**9)))

    return {
        "memory_clusters": next_memory,
        "next_cluster_id": int(running_cluster_id),
        "append_cases": append_cases,
        "current_only_reattach_cases": current_only_reattach_cases,
        "merge_cases": merge_cases,
        "same_view_block_cases": same_view_block_cases,
        "tail_spawn_cases": tail_spawn_cases,
        "num_appended": len(append_cases),
        "num_current_only_reattached": len(current_only_reattach_cases),
        "num_merged_clusters": len(merge_cases),
        "num_same_view_blocked_components": len(same_view_block_cases),
        "num_new_tail_clusters": len(tail_clusters),
    }


def _cluster_histogram(clusters: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    histogram: Dict[str, int] = defaultdict(int)
    for cluster in clusters:
        size = len(cluster.get("member_object_ids", []))
        histogram[str(size)] += 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def _label_jitter_examples(clusters: Sequence[Mapping[str, Any]], max_examples: int = 8) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for cluster in clusters:
        histogram = dict(cluster.get("label_histogram", {}))
        if len(histogram) <= 1:
            continue
        examples.append(_label_jitter_summary(cluster))
    return examples[:max_examples]


def run_sequential_spectral_experiment(
    db_dir: str = DEFAULT_DB_DIR,
    *,
    output_dir: Optional[str] = None,
    entry_ids: Optional[Sequence[Any]] = None,
    view_ids: Optional[Sequence[str]] = None,
    weight_text: float = DEFAULT_WEIGHT_TEXT,
    weight_global_geo: float = DEFAULT_WEIGHT_GLOBAL_GEO,
    weight_polar: float = DEFAULT_WEIGHT_POLAR,
    global_sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
    min_cross_affinity: float = DEFAULT_CROSS_AFFINITY_MIN,
    current_only_reattach_min_affinity: float = DEFAULT_CURRENT_ONLY_REATTACH_MIN_AFFINITY,
) -> Dict[str, Any]:
    sequence = load_sequence_objects(db_dir, entry_ids=entry_ids, view_ids=view_ids)
    selected_views = sequence["views"]
    if len(selected_views) < 2:
        raise ValueError("Need at least two selected views for sequential experiment")

    base_root = Path(output_dir) if output_dir else Path(db_dir) / "sequential_spectral_experiment"
    root = _make_run_output_dir(base_root)
    stored_view_images = _store_selected_view_images(root, db_dir, selected_views)

    stored_image_by_view_id = {
        _safe_text(item.get("view_id")): dict(item) for item in stored_view_images if _safe_text(item.get("view_id"))
    }

    manifest = {
        "db_dir": str(db_dir),
        "selected_view_ids": list(sequence["selected_view_ids"]),
        "views": [
            _view_summary(view, stored_image_by_view_id.get(_safe_text(view.get("view_id"))))
            for view in selected_views
        ],
        "stored_view_images": stored_view_images,
    }
    _write_json(root / "sequence_manifest.json", manifest)

    initial_view = selected_views[0]
    memory_clusters: List[Dict[str, Any]] = []
    next_cluster_id = 0
    for row in initial_view["objects"]:
        memory_clusters.append(_create_new_cluster_from_rows([row], next_cluster_id))
        next_cluster_id += 1

    _write_json(
        root / "step_00_initial_registry.json",
        {
            "initial_view_id": initial_view["view_id"],
            "clusters": [_cluster_summary(cluster) for cluster in memory_clusters],
        },
    )

    step_reports: List[Dict[str, Any]] = []
    all_append_cases: List[Dict[str, Any]] = []
    all_same_view_block_cases: List[Dict[str, Any]] = []
    all_tail_spawn_cases: List[Dict[str, Any]] = []

    for step_offset, current_view in enumerate(selected_views[1:], start=1):
        cross_affinity, cross_details = build_cross_affinity_matrix(
            memory_clusters,
            current_view["objects"],
            weight_text=weight_text,
            weight_global_geo=weight_global_geo,
            weight_polar=weight_polar,
            global_sigma_m=global_sigma_m,
        )
        full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=min_cross_affinity)
        axis_labels = _node_labels(memory_clusters, current_view["objects"])
        memory_labels = [f"mem:c{int(cluster.get('cluster_id', -1))}" for cluster in memory_clusters]
        current_labels = [
            f"cur:obj{_safe_int(row.get('object_global_id'), -1)}@{_safe_text(row.get('view_id'))}"
            for row in current_view["objects"]
        ]
        np.save(root / f"step_{step_offset:02d}_cross_affinity_matrix.npy", cross_affinity)
        _write_rect_matrix_csv(
            root / f"step_{step_offset:02d}_cross_affinity_matrix.csv",
            cross_affinity,
            row_labels=current_labels,
            col_labels=memory_labels,
        )
        _write_laplacian_artifacts(
            root,
            stem=f"cross_affinity_laplacian_step_{step_offset:02d}",
            affinity_matrix=full_affinity,
            axis_labels=axis_labels,
            title=f"Step {step_offset:02d} cross-affinity normalized Laplacian",
        )
        np.save(root / f"step_{step_offset:02d}_affinity_matrix.npy", full_affinity)
        _write_affinity_csv(root / f"step_{step_offset:02d}_affinity_matrix.csv", full_affinity, axis_labels)
        plot_similarity_heatmap(
            full_affinity,
            root / f"affinity_heatmap_step_{step_offset:02d}.png",
            title=f"Step {step_offset:02d} affinity",
            axis_labels=axis_labels,
            annotate_values=False,
        )

        spectral_result = run_spectral_clustering(
            full_affinity,
            object_ids=list(range(full_affinity.shape[0])),
            cluster_count_mode="eigengap",
        )
        co_cluster_matrix, component_groups = _co_cluster_matrix(
            spectral_result=spectral_result,
            full_affinity=full_affinity,
            num_memory=len(memory_clusters),
        )
        step5_order, step5_boundaries = _step5_block_order(
            memory_clusters,
            current_view["objects"],
            spectral_result=spectral_result,
            full_affinity=full_affinity,
        )
        np.save(root / f"step_{step_offset:02d}_cocluster_matrix.npy", co_cluster_matrix)
        _write_affinity_csv(root / f"step_{step_offset:02d}_cocluster_matrix.csv", co_cluster_matrix, axis_labels)
        plot_similarity_heatmap(
            full_affinity,
            root / f"spectral_block_heatmap_step_{step_offset:02d}.png",
            title=f"Step {step_offset:02d} spectral block view",
            order=step5_order,
            boundary_after_indices=step5_boundaries,
            axis_labels=axis_labels,
            annotate_values=False,
            vmin=0.0,
            vmax=1.0,
        )
        plot_similarity_heatmap(
            co_cluster_matrix,
            root / f"cocluster_heatmap_step_{step_offset:02d}.png",
            title=f"Step {step_offset:02d} co-cluster",
            order=step5_order,
            boundary_after_indices=step5_boundaries,
            axis_labels=axis_labels,
            annotate_values=False,
            vmin=0.0,
            vmax=1.0,
        )
        _write_laplacian_artifacts(
            root,
            stem=f"cocluster_laplacian_step_{step_offset:02d}",
            affinity_matrix=co_cluster_matrix,
            axis_labels=axis_labels,
            title=f"Step {step_offset:02d} co-cluster normalized Laplacian",
        )

        update = apply_incremental_step(
            memory_clusters,
            current_view["objects"],
            cross_affinity=cross_affinity,
            cross_details=cross_details,
            full_affinity=full_affinity,
            spectral_result=spectral_result,
            step_index=step_offset,
            next_cluster_id=next_cluster_id,
            weight_text=weight_text,
            weight_global_geo=weight_global_geo,
            weight_polar=weight_polar,
            global_sigma_m=global_sigma_m,
            current_only_reattach_min_affinity=current_only_reattach_min_affinity,
        )
        memory_clusters = update["memory_clusters"]
        next_cluster_id = int(update["next_cluster_id"])
        all_append_cases.extend(update["append_cases"])
        all_same_view_block_cases.extend(update["same_view_block_cases"])
        all_tail_spawn_cases.extend(update["tail_spawn_cases"])

        step_report = {
            "step_index": int(step_offset),
            "view_id": current_view["view_id"],
            "num_current_objects": len(current_view["objects"]),
            "num_existing_clusters": full_affinity.shape[0] - len(current_view["objects"]),
            "num_appended": int(update["num_appended"]),
            "num_current_only_reattached": int(update["num_current_only_reattached"]),
            "num_merged_clusters": int(update["num_merged_clusters"]),
            "num_same_view_blocked_components": int(update["num_same_view_blocked_components"]),
            "num_new_tail_clusters": int(update["num_new_tail_clusters"]),
            "spectral_summary": _spectral_result_summary(spectral_result),
            "num_connected_components_after_spectral": len(component_groups),
            "clusters_after_step": [_cluster_output_summary(cluster) for cluster in memory_clusters],
            "append_cases": [_append_case_summary(case) for case in update["append_cases"]],
            "current_only_reattach_cases": [_append_case_summary(case) for case in update["current_only_reattach_cases"]],
            "merge_cases": [_merge_case_summary(case) for case in update["merge_cases"]],
            "same_view_block_cases": [_same_view_block_case_summary(case) for case in update["same_view_block_cases"]],
            "tail_spawn_cases": [_tail_spawn_case_summary(case) for case in update["tail_spawn_cases"]],
            "cross_affinity_shape": list(cross_affinity.shape),
            "cocluster_shape": list(co_cluster_matrix.shape),
        }
        step_reports.append(step_report)
        _write_json(root / f"step_{step_offset:02d}_cluster_update.json", step_report)

    final_registry = [_cluster_output_summary(cluster) for cluster in memory_clusters]
    _write_json(root / "global_object_list_final.json", final_registry)

    report = {
        "db_dir": str(db_dir),
        "output_dir": str(root),
        "selected_view_ids": list(sequence["selected_view_ids"]),
        "views": [_view_summary(view, stored_image_by_view_id.get(_safe_text(view.get("view_id")))) for view in selected_views],
        "view_object_counts": {view["view_id"]: len(view["objects"]) for view in selected_views},
        "weights": {
            "text": float(weight_text),
            "global_geo": float(weight_global_geo),
            "polar": float(weight_polar),
            "normalized": _normalize_weight_triplet(weight_text, weight_global_geo, weight_polar),
        },
        "global_sigma_m": float(global_sigma_m),
        "min_cross_affinity": float(min_cross_affinity),
        "current_only_reattach_min_affinity": float(current_only_reattach_min_affinity),
        "step_summaries": [_step_report_summary(step_report) for step_report in step_reports],
        "final_cluster_count": len(memory_clusters),
        "cluster_size_histogram": _cluster_histogram(memory_clusters),
        "label_jitter_examples": _label_jitter_examples(memory_clusters),
        "final_clusters": [_cluster_output_summary(cluster) for cluster in memory_clusters],
        "total_appended": len(all_append_cases),
        "total_current_only_reattached": int(
            sum(_safe_int(step.get("num_current_only_reattached"), 0) for step in step_reports)
        ),
        "total_new_tail_clusters": len(all_tail_spawn_cases),
        "total_merged_clusters": int(sum(_safe_int(step.get("num_merged_clusters"), 0) for step in step_reports)),
        "total_same_view_blocked_components": int(
            sum(_safe_int(step.get("num_same_view_blocked_components"), 0) for step in step_reports)
        ),
        "append_case_examples": [_append_case_summary(case) for case in all_append_cases[:12]],
        "same_view_block_case_examples": [_same_view_block_case_summary(case) for case in all_same_view_block_cases[:12]],
        "tail_spawn_case_examples": [_tail_spawn_case_summary(case) for case in all_tail_spawn_cases[:12]],
    }
    progression_manifest = generate_cumulative_cluster_progression_artifacts(str(root))
    report["cumulative_cluster_progression_manifest"] = str(root / "cumulative_cluster_progression_manifest.json")
    report["cumulative_cluster_progression_overview"] = progression_manifest.get("overview_path")
    _write_json(root / "experiment_report.json", report)
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-view sequential spectral clustering experiment.")
    parser.add_argument("--db_dir", default=DEFAULT_DB_DIR, help="Spatial DB directory")
    parser.add_argument("--output_dir", default=None, help="Output directory for experiment artifacts")
    parser.add_argument(
        "--view_ids",
        default=None,
        help="Comma-separated ordered view ids to include in the sequential experiment",
    )
    parser.add_argument(
        "--entry_ids",
        default=None,
        help="Comma-separated ordered entry ids to include; overrides --view_ids when provided",
    )
    parser.add_argument("--weight_text", type=float, default=DEFAULT_WEIGHT_TEXT, help="Text branch weight")
    parser.add_argument("--weight_global_geo", type=float, default=DEFAULT_WEIGHT_GLOBAL_GEO, help="Global xyz branch weight")
    parser.add_argument("--weight_polar", type=float, default=DEFAULT_WEIGHT_POLAR, help="Polar branch weight")
    parser.add_argument("--global_sigma_m", type=float, default=DEFAULT_GLOBAL_SIGMA_M, help="Global xyz gaussian sigma")
    parser.add_argument(
        "--min_cross_affinity",
        type=float,
        default=DEFAULT_CROSS_AFFINITY_MIN,
        help="Hard threshold applied to cross-affinity before bipartite spectral clustering",
    )
    parser.add_argument(
        "--current_only_reattach_min_affinity",
        type=float,
        default=DEFAULT_CURRENT_ONLY_REATTACH_MIN_AFFINITY,
        help="High-score threshold for reattaching current-only spectral components back to an existing cluster",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = _parse_args(argv)
    report = run_sequential_spectral_experiment(
        db_dir=str(args.db_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        entry_ids=_normalize_entry_ids([args.entry_ids]) if args.entry_ids else None,
        view_ids=_normalize_view_ids([str(args.view_ids)]) if args.view_ids else None,
        weight_text=float(args.weight_text),
        weight_global_geo=float(args.weight_global_geo),
        weight_polar=float(args.weight_polar),
        global_sigma_m=float(args.global_sigma_m),
        min_cross_affinity=float(args.min_cross_affinity),
        current_only_reattach_min_affinity=float(args.current_only_reattach_min_affinity),
    )
    print(json.dumps(_to_serializable(report), ensure_ascii=True))
    return report


if __name__ == "__main__":
    main()
