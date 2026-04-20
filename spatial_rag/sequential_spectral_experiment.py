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

from spatial_rag.config import (
    DEFAULT_CROSS_AFFINITY_MIN,
    DEFAULT_DISTANCE_GATE_DSQ0,
    DEFAULT_ENFORCE_SAME_VIEW_UNIQUENESS,
    ENABLE_DINOV2_SCORING,
    SCORE_WEIGHT_M1,
    SCORE_WEIGHT_M2,
)
from spatial_rag.object_index import load_object_db, load_object_dinov2_db
from spatial_rag.object_instance_clustering import (
    _estimate_dbscan_eps,
    _alpha_blit,
    _l2_normalize_vec,
    _load_jsonl,
    _make_rotated_text_image,
    _run_dbscan,
    _safe_float,
    _safe_int,
    _safe_text,
    _truncate_heatmap_label,
    _to_serializable,
    _write_json,
    estimate_cluster_count_eigengap,
    plot_similarity_heatmap,
    run_spectral_clustering,
)


DEFAULT_DB_DIR = "/Users/liuyuheng/Desktop/antigravityTest/spatial_db_nd"
DEFAULT_VIEW_IDS = ("view_00019", "view_00024", "view_00058", "view_00065")
DEFAULT_WEIGHT_TEXT = float(SCORE_WEIGHT_M1)
DEFAULT_WEIGHT_DINOV2 = float(SCORE_WEIGHT_M2)
DEFAULT_WEIGHT_GLOBAL_GEO = 0.20
DEFAULT_WEIGHT_POLAR = 0.10
DEFAULT_GLOBAL_SIGMA_M = 2.0
DEFAULT_CURRENT_ONLY_REATTACH_MIN_AFFINITY = 0.75
DEFAULT_SIMILARITY_MODE = "cosine_geo_gate"
DEFAULT_DISTANCE_GATE_DSQ0_SWEEP = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_SPECTRAL_MAX_EXTRA_CLUSTERS = 2
DEFAULT_DBSCAN_MIN_SAMPLES = 2
EXCLUDED_LABELS = {"", "unknown", "other", "none"}
OBJECT_CLUSTER_SIMILARITY_TABLE_COLUMNS = (
    "object_global_id",
    "label",
    "view_id",
    "entry_id",
    "step_index",
    "assignment_reason",
    "final_cluster_id",
    "cluster_id_at_assignment",
    "similarity_reference_cluster_id",
    "estimated_global_x",
    "estimated_global_y",
    "estimated_global_z",
    "term1_cosine",
    "term2_dinov2",
    "semantic_visual_similarity",
    "xy_distance_m",
    "dsq",
    "distance_gate_dsq0",
    "distance_gate",
    "distance_gate_exponent",
    "combined_similarity",
    "similarity_detail_status",
    "same_view_split_applied",
)
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


def _normalize_float_list(values: Optional[Sequence[Any]]) -> List[float]:
    if values is None:
        return []
    out: List[float] = []
    for item in values:
        if item is None:
            continue
        for token in str(item).split(","):
            cleaned = _safe_text(token)
            if not cleaned:
                continue
            out.append(float(cleaned))
    return out


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


def _normalize_weight_triplet(
    weight_text: float,
    weight_global_geo: float,
    weight_polar: float,
    weight_dinov2: float = 0.0,
) -> Dict[str, float]:
    weights = {
        "text": max(0.0, float(weight_text)),
        "dinov2": max(0.0, float(weight_dinov2)),
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


def _proto_dinov2_embedding(members: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    vectors: List[np.ndarray] = []
    for row in members:
        vec = row.get("dinov2_embedding")
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
        "prototype_dinov2_embedding": _proto_dinov2_embedding(member_rows),
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


def _cluster_existing_cluster_id(cluster: Mapping[str, Any]) -> Optional[int]:
    cluster_id = _safe_int(cluster.get("cluster_id"), -1)
    return None if cluster_id < 0 else int(cluster_id)


def _synthetic_singleton_cluster(row: Mapping[str, Any]) -> Dict[str, Any]:
    view_id = _safe_text(row.get("view_id"))
    object_id = _safe_int(row.get("object_global_id"), -1)
    return {
        "cluster_id": None,
        "member_rows": [dict(row)],
        "member_view_ids": [view_id] if view_id else [],
        "member_object_ids": [object_id] if object_id >= 0 else [],
    }


def _same_view_conflicting_view_ids(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        view_id = _safe_text(row.get("view_id"))
        if view_id:
            counts[view_id] += 1
    return sorted(view_id for view_id, count in counts.items() if count > 1)


def _apply_same_view_hard_mask_to_cross_affinity(
    memory_clusters: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    *,
    cross_affinity: np.ndarray,
    cross_details: Sequence[Sequence[Mapping[str, Any]]],
) -> Tuple[np.ndarray, List[List[Dict[str, Any]]], int]:
    masked_affinity = np.asarray(cross_affinity, dtype=np.float32).copy()
    masked_details: List[List[Dict[str, Any]]] = [
        [dict(detail) for detail in row_details]
        for row_details in cross_details
    ]
    num_masked_edges = 0
    for cur_idx, row in enumerate(current_rows):
        row_view_id = _safe_text(row.get("view_id"))
        if not row_view_id:
            continue
        for mem_idx, cluster in enumerate(memory_clusters):
            if row_view_id not in _cluster_view_id_set(cluster):
                continue
            num_masked_edges += 1
            masked_affinity[int(cur_idx), int(mem_idx)] = 0.0
            if cur_idx < len(masked_details) and mem_idx < len(masked_details[cur_idx]):
                detail = dict(masked_details[cur_idx][mem_idx])
            else:
                detail = {}
            detail["same_view_masked"] = True
            detail["same_view_mask_reason"] = "shared_view_id"
            masked_details[cur_idx][mem_idx] = detail
    return masked_affinity, masked_details, int(num_masked_edges)


def _cross_detail_for_pair(
    cur_idx: int,
    mem_idx: int,
    *,
    cross_details: Sequence[Sequence[Mapping[str, Any]]],
    current_rows: Sequence[Mapping[str, Any]],
    cluster: Mapping[str, Any],
    weight_text: float,
    weight_dinov2: float,
    weight_global_geo: float,
    weight_polar: float,
    global_sigma_m: float,
    similarity_mode: str,
    distance_gate_dsq0: float,
    enable_dinov2_scoring: bool,
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
        weights=_normalize_weight_triplet(weight_text, weight_global_geo, weight_polar, weight_dinov2),
        global_sigma_m=global_sigma_m,
        similarity_mode=similarity_mode,
        distance_gate_dsq0=distance_gate_dsq0,
        enable_dinov2_scoring=enable_dinov2_scoring,
    )


def _best_live_memory_match(
    row: Mapping[str, Any],
    slots: Sequence[Optional[Mapping[str, Any]]],
    *,
    weight_text: float,
    weight_dinov2: float,
    weight_global_geo: float,
    weight_polar: float,
    global_sigma_m: float,
    similarity_mode: str,
    distance_gate_dsq0: float,
    enable_dinov2_scoring: bool,
) -> Optional[Tuple[float, int, Dict[str, Any]]]:
    best: Optional[Tuple[float, int, Dict[str, Any]]] = None
    weights = _normalize_weight_triplet(weight_text, weight_global_geo, weight_polar, weight_dinov2)
    for mem_idx, cluster in enumerate(slots):
        if cluster is None:
            continue
        detail = _pair_affinity_detail(
            row,
            cluster,
            weights=weights,
            global_sigma_m=global_sigma_m,
            similarity_mode=similarity_mode,
            distance_gate_dsq0=distance_gate_dsq0,
            enable_dinov2_scoring=enable_dinov2_scoring,
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
        "prototype_dinov2_embedding": cluster.get("prototype_dinov2_embedding"),
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
        "similarity_mode": _safe_text(detail.get("similarity_mode")),
        "combined_similarity": _safe_float(detail.get("combined_similarity")),
        "text_similarity": _safe_float(detail.get("text_similarity")),
        "dinov2_similarity": _safe_float(detail.get("dinov2_similarity")),
        "semantic_visual_similarity": _safe_float(detail.get("semantic_visual_similarity")),
        "global_geo_similarity": _safe_float(detail.get("global_geo_similarity")),
        "polar_similarity": _safe_float(detail.get("polar_similarity")),
        "global_geo_distance_m": _safe_float(detail.get("global_geo_distance_m")),
        "used_3d_global_geo": bool(detail.get("used_3d_global_geo")),
        "distance_gate": _safe_float(detail.get("distance_gate")),
        "xy_distance_m": _safe_float(detail.get("xy_distance_m")),
        "xy_distance_sq_m2": _safe_float(detail.get("xy_distance_sq_m2")),
        "distance_gate_dsq0": _safe_float(detail.get("distance_gate_dsq0")),
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
        "raw_dbscan_label": _safe_int(case.get("raw_dbscan_label"), -1),
        "connectivity_label": _safe_int(case.get("connectivity_label"), -1),
        "blocked_merge_cluster_ids": [_safe_int(value, -1) for value in case.get("blocked_merge_cluster_ids", [])],
        "competing_object_ids": [_safe_int(value, -1) for value in case.get("competing_object_ids", [])],
        "unassigned_object_ids": [_safe_int(value, -1) for value in case.get("unassigned_object_ids", [])],
        "conflicting_view_ids": [_safe_text(value) for value in case.get("conflicting_view_ids", [])],
        "original_group": {
            "memory_cluster_ids": [
                _safe_int(value, -1) for value in case.get("original_group", {}).get("memory_cluster_ids", [])
            ],
            "current_object_ids": [
                _safe_int(value, -1) for value in case.get("original_group", {}).get("current_object_ids", [])
            ],
            "view_ids": [_safe_text(value) for value in case.get("original_group", {}).get("view_ids", [])],
            "conflicting_view_ids": [
                _safe_text(value) for value in case.get("original_group", {}).get("conflicting_view_ids", [])
            ],
        },
        "resolved_subgroups": [
            {
                "cluster_id": _safe_int(item.get("cluster_id"), -1),
                "memory_cluster_ids": [_safe_int(value, -1) for value in item.get("memory_cluster_ids", [])],
                "current_object_ids": [_safe_int(value, -1) for value in item.get("current_object_ids", [])],
                "view_ids": [_safe_text(value) for value in item.get("view_ids", [])],
            }
            for item in list(case.get("resolved_subgroups", []))
        ],
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


def _assignment_detail_status(reason: str, detail: Optional[Mapping[str, Any]]) -> str:
    reason_clean = _safe_text(reason)
    if reason_clean == "initial_seed":
        return "initial_seed"
    if not detail:
        return "no_candidate_detail"
    if reason_clean in {
        "component_best_append",
        "same_view_hard_block_competition_append",
        "current_only_high_score_reattach",
        "dbscan_attach",
    }:
        return "assigned_match"
    return "best_rejected_candidate"


def _object_assignment_record(
    row: Mapping[str, Any],
    *,
    step_index: int,
    assignment_reason: str,
    cluster_id_at_assignment: Optional[int],
    similarity_reference_cluster_id: Optional[int],
    detail: Optional[Mapping[str, Any]],
    similarity_detail_status: Optional[str] = None,
    same_view_split_applied: bool = False,
) -> Dict[str, Any]:
    status = similarity_detail_status or _assignment_detail_status(assignment_reason, detail)
    term1_cosine = _safe_float(detail.get("text_similarity")) if detail else None
    term2_dinov2 = _safe_float(detail.get("dinov2_similarity")) if detail else None
    semantic_visual_similarity = _safe_float(detail.get("semantic_visual_similarity")) if detail else None
    xy_distance_m = _safe_float(detail.get("xy_distance_m")) if detail else None
    dsq = _safe_float(detail.get("xy_distance_sq_m2")) if detail else None
    distance_gate_dsq0 = _safe_float(detail.get("distance_gate_dsq0")) if detail else None
    distance_gate = _safe_float(detail.get("distance_gate")) if detail else None
    exponent = None
    if dsq is not None and distance_gate_dsq0 not in (None, 0.0):
        exponent = -float(dsq) / (2.0 * float(distance_gate_dsq0))
    return {
        "object_global_id": _safe_int(row.get("object_global_id"), -1),
        "label": _safe_text(row.get("label")),
        "view_id": _safe_text(row.get("view_id")),
        "entry_id": _safe_int(row.get("entry_id"), -1),
        "step_index": int(step_index),
        "assignment_reason": _safe_text(assignment_reason),
        "final_cluster_id": None,
        "cluster_id_at_assignment": None if cluster_id_at_assignment is None else int(cluster_id_at_assignment),
        "similarity_reference_cluster_id": None
        if similarity_reference_cluster_id is None
        else int(similarity_reference_cluster_id),
        "estimated_global_x": _safe_float(row.get("estimated_global_x")),
        "estimated_global_y": _safe_float(row.get("estimated_global_y")),
        "estimated_global_z": _safe_float(row.get("estimated_global_z")),
        "term1_cosine": term1_cosine,
        "term2_dinov2": term2_dinov2,
        "semantic_visual_similarity": semantic_visual_similarity,
        "xy_distance_m": xy_distance_m,
        "dsq": dsq,
        "distance_gate_dsq0": distance_gate_dsq0,
        "distance_gate": distance_gate,
        "distance_gate_exponent": exponent,
        "combined_similarity": _safe_float(detail.get("combined_similarity")) if detail else None,
        "similarity_detail_status": status,
        "same_view_split_applied": bool(same_view_split_applied),
    }


def _final_cluster_id_by_object(clusters: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for cluster in clusters:
        cluster_id = _safe_int(cluster.get("cluster_id"), -1)
        if cluster_id < 0:
            continue
        for row in list(cluster.get("member_rows", [])):
            object_id = _safe_int(row.get("object_global_id"), -1)
            if object_id >= 0:
                mapping[int(object_id)] = int(cluster_id)
    return mapping


def _materialize_object_cluster_similarity_rows(
    assignment_records: Sequence[Mapping[str, Any]],
    *,
    final_cluster_id_by_object: Mapping[int, int],
) -> List[Dict[str, Any]]:
    ordered_records = sorted(
        (dict(record) for record in assignment_records),
        key=lambda item: (
            _safe_int(item.get("step_index"), 10**9),
            _safe_int(item.get("object_global_id"), 10**9),
        ),
    )
    materialized: List[Dict[str, Any]] = []
    for record in ordered_records:
        object_id = _safe_int(record.get("object_global_id"), -1)
        final_cluster_id = final_cluster_id_by_object.get(int(object_id))
        row = dict(record)
        row["final_cluster_id"] = None if final_cluster_id is None else int(final_cluster_id)
        materialized.append(row)
    return materialized


def _write_object_cluster_similarity_table(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OBJECT_CLUSTER_SIMILARITY_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
                    for key in OBJECT_CLUSTER_SIMILARITY_TABLE_COLUMNS
                }
            )
    return str(path)


def _write_object_cluster_similarity_tables_by_step(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_safe_int(row.get("step_index"), -1)].append(dict(row))
    outputs: List[Dict[str, Any]] = []
    for step_index in sorted(grouped):
        step_rows = sorted(
            grouped[step_index],
            key=lambda item: _safe_int(item.get("object_global_id"), 10**9),
        )
        path = root / f"step_{int(step_index):02d}_object_assignment_table.csv"
        _write_object_cluster_similarity_table(path, step_rows)
        outputs.append(
            {
                "step_index": int(step_index),
                "view_id": _safe_text(step_rows[0].get("view_id")) if step_rows else "",
                "path": str(path),
                "num_rows": len(step_rows),
            }
        )
    return outputs


def _spectral_result_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    labels = result.get("labels")
    return {
        "n_clusters": _safe_int(result.get("n_clusters"), -1),
        "cluster_count_mode": _safe_text(result.get("cluster_count_mode")),
        "requested_n_clusters": _safe_int(result.get("requested_n_clusters"), -1),
        "eigengap_n_clusters": _safe_int(result.get("eigengap_n_clusters"), -1),
        "connected_component_count": _safe_int(result.get("connected_component_count"), -1),
        "max_allowed_n_clusters": _safe_int(result.get("max_allowed_n_clusters"), -1),
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
        "enforce_same_view_uniqueness": bool(step_report.get("enforce_same_view_uniqueness")),
        "num_current_objects": _safe_int(step_report.get("num_current_objects"), 0),
        "num_existing_clusters": _safe_int(step_report.get("num_existing_clusters"), 0),
        "num_appended": _safe_int(step_report.get("num_appended"), 0),
        "num_current_only_reattached": _safe_int(step_report.get("num_current_only_reattached"), 0),
        "num_merged_clusters": _safe_int(step_report.get("num_merged_clusters"), 0),
        "num_same_view_masked_edges": _safe_int(step_report.get("num_same_view_masked_edges"), 0),
        "num_same_view_blocked_components": _safe_int(step_report.get("num_same_view_blocked_components"), 0),
        "num_new_tail_clusters": _safe_int(step_report.get("num_new_tail_clusters"), 0),
        "num_dbscan_clusters": _safe_int(step_report.get("num_dbscan_clusters"), 0),
        "num_noise_singletons": _safe_int(step_report.get("num_noise_singletons"), 0),
        "num_merged_memory_groups": _safe_int(step_report.get("num_merged_memory_groups"), 0),
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


def _load_image_bgr(path: Path) -> Optional[np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def _overlay_text_for_object(row: Mapping[str, Any]) -> Optional[str]:
    object_id = _safe_int(row.get("object_global_id"), -1)
    if object_id < 0:
        return None
    label = _safe_text(row.get("label"))
    if label:
        return f"obj{object_id} {_truncate_heatmap_label(label, max_chars=24)}"
    return f"obj{object_id}"


def _render_selected_view_detection_overlay(
    source_image_path: Path,
    output_path: Path,
    objects: Sequence[Mapping[str, Any]],
) -> bool:
    canvas = _load_image_bgr(source_image_path)
    if canvas is None:
        return False
    height, width = canvas.shape[:2]
    for row in objects:
        bbox = row.get("bbox_xyxy")
        if bbox is None:
            bbox = row.get("bbox")
        if bbox is None:
            continue
        bbox_values = np.asarray(bbox).reshape(-1)
        if bbox_values.size < 4:
            continue
        label_text = _overlay_text_for_object(row)
        if not label_text:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_values[:4]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 200, 40), 2)

        text = label_text[:40]
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        text_x = max(0, min(width - text_w - 6, x1))
        text_y = y1 - 8
        if text_y - text_h - baseline < 0:
            text_y = min(height - baseline - 4, y1 + text_h + 8)
        bg_left = max(0, text_x - 3)
        bg_top = max(0, text_y - text_h - baseline - 3)
        bg_right = min(width - 1, text_x + text_w + 3)
        bg_bottom = min(height - 1, text_y + baseline + 3)
        cv2.rectangle(canvas, (bg_left, bg_top), (bg_right, bg_bottom), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save detection overlay to {output_path}")
    return True


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
        overlay_status = "overlay_source_unreadable"
        overlay_destination = out_dir / f"{view_id}_yolo_overlay{source_path.suffix or '.jpg'}"
        if _render_selected_view_detection_overlay(source_path, overlay_destination, view.get("objects") or []):
            stored_overlay_path = str(overlay_destination)
            overlay_status = "rendered"
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
    dino_loaded = load_object_dinov2_db(str(root))
    dino_embedding_by_object_id: Dict[int, np.ndarray] = {}
    if dino_loaded is not None:
        _dino_meta, dino_emb, object_id_to_sidecar_row = dino_loaded
        for object_id, sidecar_row in object_id_to_sidecar_row.items():
            dino_embedding_by_object_id[int(object_id)] = _l2_normalize_vec(
                np.asarray(dino_emb[int(sidecar_row)], dtype=np.float32)
            )

    by_view: Dict[str, List[Dict[str, Any]]] = {view_id: [] for view_id in selected_view_ids}
    for index, row in enumerate(object_rows):
        prepared = dict(row)
        entry_id = _safe_int(prepared.get("entry_id"), -1)
        object_id = _safe_int(prepared.get("object_global_id"), -1)
        view_id = _safe_text(prepared.get("view_id")) or _view_id_for_entry(entry_id)
        if view_id not in selected_set:
            continue
        if not _is_valid_object_row(prepared):
            continue
        prepared["view_id"] = view_id
        prepared["entry_id"] = entry_id
        prepared["embedding"] = _l2_normalize_vec(np.asarray(long_emb[index], dtype=np.float32))
        prepared["text_embedding"] = prepared["embedding"]
        prepared["dinov2_embedding"] = dino_embedding_by_object_id.get(int(object_id))
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


def _distance_gate_from_dsq(distance_sq: float, dsq0: float) -> float:
    dsq0_value = max(float(dsq0), 1e-6)
    return float(math.exp(-(float(distance_sq) / (2.0 * dsq0_value))))


def _text_similarity(row: Mapping[str, Any], cluster: Mapping[str, Any]) -> Optional[float]:
    row_vec = row.get("embedding")
    proto_vec = cluster.get("prototype_embedding")
    if row_vec is None or proto_vec is None:
        return None
    return float(np.clip(np.dot(np.asarray(row_vec, dtype=np.float32), np.asarray(proto_vec, dtype=np.float32)), -1.0, 1.0))


def _dinov2_similarity(row: Mapping[str, Any], cluster: Mapping[str, Any]) -> Optional[float]:
    row_vec = row.get("dinov2_embedding")
    proto_vec = cluster.get("prototype_dinov2_embedding")
    if row_vec is None or proto_vec is None:
        return None
    return float(
        np.clip(
            np.dot(np.asarray(row_vec, dtype=np.float32), np.asarray(proto_vec, dtype=np.float32)),
            -1.0,
            1.0,
        )
    )


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
    if row_x is None or row_y is None or cluster_x is None or cluster_y is None:
        return None, None, False
    if row_z is not None and cluster_z is not None:
        delta = np.asarray([row_x - cluster_x, row_y - cluster_y, row_z - cluster_z], dtype=np.float32)
        return _gaussian_similarity(float(np.linalg.norm(delta)), sigma_m), float(np.linalg.norm(delta)), True
    delta = np.asarray([row_x - cluster_x, row_y - cluster_y], dtype=np.float32)
    return _gaussian_similarity(float(np.linalg.norm(delta)), sigma_m), float(np.linalg.norm(delta)), False


def _xy_distance(row: Mapping[str, Any], cluster: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    row_x, row_y, _row_z = _row_xyz(row)
    proto_xyz = dict(cluster.get("prototype_xyz") or {})
    cluster_x = _safe_float(proto_xyz.get("x"))
    cluster_y = _safe_float(proto_xyz.get("y"))
    if row_x is None or row_y is None or cluster_x is None or cluster_y is None:
        return None, None
    dx = float(row_x) - float(cluster_x)
    dy = float(row_y) - float(cluster_y)
    distance_sq = float(dx * dx + dy * dy)
    return float(math.sqrt(distance_sq)), distance_sq


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
    similarity_mode: str = DEFAULT_SIMILARITY_MODE,
    distance_gate_dsq0: float = DEFAULT_DISTANCE_GATE_DSQ0,
    enable_dinov2_scoring: bool = bool(ENABLE_DINOV2_SCORING),
) -> Dict[str, Any]:
    text_sim = _text_similarity(row, cluster)
    dinov2_sim = _dinov2_similarity(row, cluster) if enable_dinov2_scoring else None
    similarity_mode_clean = _safe_text(similarity_mode) or DEFAULT_SIMILARITY_MODE
    if similarity_mode_clean == "legacy_weighted_fusion":
        geo_sim, geo_distance, used_3d_geo = _global_geo_similarity(row, cluster, sigma_m=global_sigma_m)
        polar_sim = _polar_similarity(row, cluster)
        available_weights = {
            "text": weights["text"] if text_sim is not None else 0.0,
            "dinov2": weights.get("dinov2", 0.0) if dinov2_sim is not None else 0.0,
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
            if dinov2_sim is not None:
                combined += normalized_weights["dinov2"] * float(dinov2_sim)
            if geo_sim is not None:
                combined += normalized_weights["global_geo"] * float(geo_sim)
            if polar_sim is not None:
                combined += normalized_weights["polar"] * float(polar_sim)
        return {
            "similarity_mode": similarity_mode_clean,
            "combined_similarity": float(combined),
            "text_similarity": text_sim,
            "dinov2_similarity": dinov2_sim,
            "semantic_visual_similarity": None,
            "global_geo_similarity": geo_sim,
            "polar_similarity": polar_sim,
            "global_geo_distance_m": geo_distance,
            "used_3d_global_geo": bool(used_3d_geo),
            "normalized_weights": normalized_weights,
            "distance_gate": None,
            "xy_distance_m": None,
            "xy_distance_sq_m2": None,
            "distance_gate_dsq0": None,
        }

    xy_distance_m, xy_distance_sq_m2 = _xy_distance(row, cluster)
    geo_sim = None
    polar_sim = None
    used_3d_geo = False
    semantic_weights = {
        "text": weights["text"] if text_sim is not None else 0.0,
        "dinov2": weights.get("dinov2", 0.0) if dinov2_sim is not None else 0.0,
    }
    semantic_weight_total = sum(semantic_weights.values())
    semantic_visual_similarity = None
    if semantic_weight_total > 0.0:
        semantic_visual_similarity = 0.0
        if text_sim is not None:
            semantic_visual_similarity += (semantic_weights["text"] / semantic_weight_total) * float(text_sim)
        if dinov2_sim is not None:
            semantic_visual_similarity += (semantic_weights["dinov2"] / semantic_weight_total) * float(dinov2_sim)
    if semantic_visual_similarity is None:
        combined = 0.0
        distance_gate = None
        geo_distance = xy_distance_m
        normalized_weights = {"text": 0.0, "dinov2": 0.0, "global_geo": 0.0, "polar": 0.0}
    elif xy_distance_sq_m2 is None:
        distance_gate = 1.0
        combined = float(semantic_visual_similarity)
        geo_distance = None
        normalized_weights = {
            "text": float(semantic_weights["text"] / semantic_weight_total) if semantic_weight_total > 0.0 else 0.0,
            "dinov2": float(semantic_weights["dinov2"] / semantic_weight_total) if semantic_weight_total > 0.0 else 0.0,
            "global_geo": 0.0,
            "polar": 0.0,
        }
    else:
        distance_gate = _distance_gate_from_dsq(xy_distance_sq_m2, distance_gate_dsq0)
        combined = float(semantic_visual_similarity) * float(distance_gate)
        geo_distance = xy_distance_m
        normalized_weights = {
            "text": float(semantic_weights["text"] / semantic_weight_total) if semantic_weight_total > 0.0 else 0.0,
            "dinov2": float(semantic_weights["dinov2"] / semantic_weight_total) if semantic_weight_total > 0.0 else 0.0,
            "global_geo": 0.0,
            "polar": 0.0,
        }
    return {
        "similarity_mode": similarity_mode_clean,
        "combined_similarity": float(combined),
        "text_similarity": text_sim,
        "dinov2_similarity": dinov2_sim,
        "semantic_visual_similarity": semantic_visual_similarity,
        "global_geo_similarity": geo_sim,
        "polar_similarity": polar_sim,
        "global_geo_distance_m": geo_distance,
        "used_3d_global_geo": bool(used_3d_geo),
        "normalized_weights": normalized_weights,
        "distance_gate": None if distance_gate is None else float(distance_gate),
        "xy_distance_m": xy_distance_m,
        "xy_distance_sq_m2": xy_distance_sq_m2,
        "distance_gate_dsq0": float(distance_gate_dsq0),
    }


def build_cross_affinity_matrix(
    memory_clusters: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    *,
    weight_text: float = DEFAULT_WEIGHT_TEXT,
    weight_dinov2: float = DEFAULT_WEIGHT_DINOV2,
    weight_global_geo: float = DEFAULT_WEIGHT_GLOBAL_GEO,
    weight_polar: float = DEFAULT_WEIGHT_POLAR,
    global_sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
    similarity_mode: str = DEFAULT_SIMILARITY_MODE,
    distance_gate_dsq0: float = DEFAULT_DISTANCE_GATE_DSQ0,
    enable_dinov2_scoring: bool = bool(ENABLE_DINOV2_SCORING),
) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
    weights = _normalize_weight_triplet(weight_text, weight_global_geo, weight_polar, weight_dinov2)
    matrix = np.zeros((len(current_rows), len(memory_clusters)), dtype=np.float32)
    details: List[List[Dict[str, Any]]] = []
    for row in current_rows:
        row_details: List[Dict[str, Any]] = []
        for cluster in memory_clusters:
            detail = _pair_affinity_detail(
                row,
                cluster,
                weights=weights,
                global_sigma_m=global_sigma_m,
                similarity_mode=similarity_mode,
                distance_gate_dsq0=distance_gate_dsq0,
                enable_dinov2_scoring=enable_dinov2_scoring,
            )
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


def _run_capped_sequential_spectral_clustering(
    affinity_matrix: np.ndarray,
    *,
    object_ids: Optional[Sequence[int]] = None,
    random_state: int = 0,
    max_extra_clusters: int = DEFAULT_SPECTRAL_MAX_EXTRA_CLUSTERS,
) -> Dict[str, Any]:
    size = int(affinity_matrix.shape[0])
    connectivity_labels = _connectivity_labels(affinity_matrix)
    connected_component_count = int(len(set(connectivity_labels.tolist()))) if connectivity_labels.size else 0
    eigengap_n_clusters = int(estimate_cluster_count_eigengap(affinity_matrix, max_clusters=size))
    max_allowed_n_clusters = max(
        1,
        min(size, int(connected_component_count) + max(0, int(max_extra_clusters))),
    )
    requested_n_clusters = max(1, min(int(eigengap_n_clusters), int(max_allowed_n_clusters)))
    spectral_result = run_spectral_clustering(
        affinity_matrix,
        object_ids=object_ids,
        cluster_count_mode="fixed",
        n_clusters=requested_n_clusters,
        random_state=random_state,
    )
    updated_result = dict(spectral_result)
    updated_result["cluster_count_mode"] = "eigengap_capped"
    updated_result["requested_n_clusters"] = int(requested_n_clusters)
    updated_result["eigengap_n_clusters"] = int(eigengap_n_clusters)
    updated_result["connected_component_count"] = int(connected_component_count)
    updated_result["max_allowed_n_clusters"] = int(max_allowed_n_clusters)
    return updated_result


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


def _resolved_step_spectral_embedding(
    spectral_result: Mapping[str, Any],
    *,
    num_nodes: int,
) -> Optional[np.ndarray]:
    raw_embedding = spectral_result.get("spectral_embedding")
    if raw_embedding is None:
        return None
    embedding = np.asarray(raw_embedding, dtype=np.float32)
    if embedding.ndim == 1:
        embedding = embedding.reshape(-1, 1)
    if embedding.ndim != 2 or embedding.shape[0] != int(num_nodes) or embedding.size == 0:
        return None
    return embedding.astype(np.float32)


def _run_dbscan_over_step_graph(
    spectral_result: Mapping[str, Any],
    *,
    full_affinity: np.ndarray,
    dbscan_eps: Optional[float],
    dbscan_min_samples: int,
) -> Dict[str, Any]:
    num_nodes = int(full_affinity.shape[0])
    connectivity_labels = _connectivity_labels(full_affinity)
    resolved_embedding = _resolved_step_spectral_embedding(spectral_result, num_nodes=num_nodes)
    if num_nodes == 0:
        return {
            "labels": np.zeros((0,), dtype=np.int32),
            "connectivity_labels": connectivity_labels,
            "dbscan_eps": None,
            "dbscan_min_samples": 0,
            "used_auto_eps": False,
            "fallback_reason": "empty_graph",
            "spectral_embedding_dim": 0,
        }

    if resolved_embedding is None:
        spectral_labels = np.asarray(spectral_result.get("labels", []), dtype=np.int32)
        if spectral_labels.size != num_nodes:
            spectral_labels = np.arange(num_nodes, dtype=np.int32)
        n_clusters = _safe_int(spectral_result.get("n_clusters"), -1)
        if num_nodes == 1 or n_clusters <= 1:
            labels = np.zeros((num_nodes,), dtype=np.int32)
        elif n_clusters >= num_nodes:
            labels = np.arange(num_nodes, dtype=np.int32)
        else:
            labels = spectral_labels.astype(np.int32)
        return {
            "labels": labels,
            "connectivity_labels": connectivity_labels,
            "dbscan_eps": None if dbscan_eps is None else float(dbscan_eps),
            "dbscan_min_samples": max(1, min(int(dbscan_min_samples), num_nodes)),
            "used_auto_eps": False,
            "fallback_reason": "missing_spectral_embedding",
            "spectral_embedding_dim": 0,
        }

    resolved_min_samples = max(1, min(int(dbscan_min_samples), num_nodes))
    resolved_eps = (
        float(dbscan_eps)
        if dbscan_eps is not None
        else _estimate_dbscan_eps(resolved_embedding, min_samples=resolved_min_samples)
    )
    labels = _run_dbscan(
        resolved_embedding,
        eps=float(resolved_eps),
        min_samples=resolved_min_samples,
    )
    return {
        "labels": np.asarray(labels, dtype=np.int32),
        "connectivity_labels": connectivity_labels,
        "dbscan_eps": float(resolved_eps),
        "dbscan_min_samples": int(resolved_min_samples),
        "used_auto_eps": bool(dbscan_eps is None),
        "fallback_reason": None,
        "spectral_embedding_dim": int(resolved_embedding.shape[1]),
    }


def _group_dbscan_nodes(
    dbscan_labels: Sequence[int],
    connectivity_labels: Sequence[int],
    *,
    num_memory: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for node_index, raw_label in enumerate(dbscan_labels):
        cc_label = _safe_int(connectivity_labels[node_index], 0) if node_index < len(connectivity_labels) else 0
        if int(raw_label) == -1:
            key: Tuple[Any, ...] = ("noise", int(node_index))
        else:
            key = ("cluster", int(raw_label), int(cc_label))
        bucket = grouped.setdefault(
            key,
            {
                "raw_dbscan_label": int(raw_label),
                "connectivity_label": int(cc_label),
                "memory_indices": [],
                "current_indices": [],
            },
        )
        if node_index < int(num_memory):
            bucket["memory_indices"].append(int(node_index))
        else:
            bucket["current_indices"].append(int(node_index - num_memory))

    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            min(item["memory_indices"]) if item["memory_indices"] else 10**9,
            min(item["current_indices"]) if item["current_indices"] else 10**9,
            int(item["raw_dbscan_label"]),
            int(item["connectivity_label"]),
        ),
    )
    for item in ordered:
        item["memory_indices"].sort()
        item["current_indices"].sort()
    return ordered


def _subgroup_existing_cluster_id(
    subgroup: Mapping[str, Any],
    *,
    slots: Sequence[Mapping[str, Any]],
) -> Optional[int]:
    candidate_ids: List[int] = []
    for mem_idx in subgroup.get("memory_indices", []):
        if not (0 <= int(mem_idx) < len(slots)):
            continue
        cluster_id = _cluster_existing_cluster_id(slots[int(mem_idx)])
        if cluster_id is not None:
            candidate_ids.append(int(cluster_id))
    if not candidate_ids:
        return None
    return int(min(candidate_ids))


def _current_best_score_to_memory_indices(
    cur_idx: int,
    mem_indices: Sequence[int],
    *,
    cross_affinity: np.ndarray,
) -> Optional[float]:
    if cross_affinity.ndim != 2 or not mem_indices:
        return None
    if not (0 <= int(cur_idx) < int(cross_affinity.shape[0])):
        return None
    values: List[float] = []
    for mem_idx in mem_indices:
        if 0 <= int(mem_idx) < int(cross_affinity.shape[1]):
            values.append(float(cross_affinity[int(cur_idx), int(mem_idx)]))
    if not values:
        return None
    return float(max(values))


def _resolve_same_view_constrained_group(
    group: Mapping[str, Any],
    *,
    slots: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    cross_affinity: np.ndarray,
) -> List[Dict[str, Any]]:
    mem_indices_raw = [int(index) for index in group.get("memory_indices", [])]
    cur_indices_raw = [int(index) for index in group.get("current_indices", [])]
    resolved: List[Dict[str, Any]] = []
    next_creation_order = 0

    def _new_subgroup(
        *,
        memory_indices: Optional[Sequence[int]] = None,
        current_indices: Optional[Sequence[int]] = None,
        view_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        nonlocal next_creation_order
        subgroup = {
            "memory_indices": [int(index) for index in (memory_indices or [])],
            "current_indices": [int(index) for index in (current_indices or [])],
            "view_ids": {
                _safe_text(view_id)
                for view_id in list(view_ids or [])
                if _safe_text(view_id)
            },
            "creation_order": int(next_creation_order),
        }
        next_creation_order += 1
        resolved.append(subgroup)
        return subgroup

    ordered_mem_indices = sorted(
        mem_indices_raw,
        key=lambda mem_idx: (
            -len(_cluster_view_id_set(slots[int(mem_idx)])),
            _safe_int(slots[int(mem_idx)].get("cluster_id"), 10**9),
        ),
    )
    for mem_idx in ordered_mem_indices:
        cluster = slots[int(mem_idx)]
        cluster_views = _cluster_view_id_set(cluster)
        placed = False
        for subgroup in resolved:
            if subgroup["view_ids"] & cluster_views:
                continue
            subgroup["memory_indices"].append(int(mem_idx))
            subgroup["view_ids"].update(cluster_views)
            placed = True
            break
        if not placed:
            _new_subgroup(memory_indices=[int(mem_idx)], view_ids=sorted(cluster_views))

    if mem_indices_raw:
        def _current_sort_key(cur_idx: int) -> Tuple[float, int]:
            score = _current_best_score_to_memory_indices(
                int(cur_idx),
                mem_indices_raw,
                cross_affinity=cross_affinity,
            )
            return (
                -float(score if score is not None else float("-inf")),
                _safe_int(current_rows[int(cur_idx)].get("object_global_id"), 10**9),
            )

        ordered_cur_indices = sorted(
            cur_indices_raw,
            key=_current_sort_key,
        )
    else:
        ordered_cur_indices = sorted(
            cur_indices_raw,
            key=lambda cur_idx: _safe_int(current_rows[int(cur_idx)].get("object_global_id"), 10**9),
        )

    for cur_idx in ordered_cur_indices:
        row = current_rows[int(cur_idx)]
        row_view_id = _safe_text(row.get("view_id"))
        compatible_subgroups = [
            subgroup
            for subgroup in resolved
            if not row_view_id or row_view_id not in subgroup["view_ids"]
        ]
        chosen_subgroup: Optional[Dict[str, Any]] = None
        chosen_score = float("-inf")
        chosen_cluster_sort = 10**9
        chosen_creation = 10**9
        for subgroup in compatible_subgroups:
            score = _current_best_score_to_memory_indices(
                int(cur_idx),
                subgroup.get("memory_indices", []),
                cross_affinity=cross_affinity,
            )
            score_value = float(score) if score is not None else float("-inf")
            reused_cluster_id = _subgroup_existing_cluster_id(subgroup, slots=slots)
            cluster_sort = 10**9 if reused_cluster_id is None else int(reused_cluster_id)
            creation_order = int(subgroup.get("creation_order", 10**9))
            if (
                chosen_subgroup is None
                or score_value > chosen_score
                or (
                    math.isclose(score_value, chosen_score)
                    and (
                        cluster_sort < chosen_cluster_sort
                        or (
                            cluster_sort == chosen_cluster_sort
                            and creation_order < chosen_creation
                        )
                    )
                )
            ):
                chosen_subgroup = subgroup
                chosen_score = float(score_value)
                chosen_cluster_sort = int(cluster_sort)
                chosen_creation = int(creation_order)
        if chosen_subgroup is None:
            chosen_subgroup = _new_subgroup(current_indices=[int(cur_idx)], view_ids=[row_view_id] if row_view_id else [])
        else:
            chosen_subgroup["current_indices"].append(int(cur_idx))
            if row_view_id:
                chosen_subgroup["view_ids"].add(row_view_id)

    return resolved


def _make_same_view_block_case(
    *,
    step_index: int,
    group: Mapping[str, Any],
    resolved_subgroups: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    cross_affinity: np.ndarray,
) -> Dict[str, Any]:
    original_memory_cluster_ids = [
        _safe_int(slots[int(mem_idx)].get("cluster_id"), -1)
        for mem_idx in group.get("memory_indices", [])
        if 0 <= int(mem_idx) < len(slots)
    ]
    original_current_object_ids = [
        _safe_int(current_rows[int(cur_idx)].get("object_global_id"), -1)
        for cur_idx in group.get("current_indices", [])
        if 0 <= int(cur_idx) < len(current_rows)
    ]
    original_rows: List[Dict[str, Any]] = []
    indexed_clusters: List[Tuple[int, Mapping[str, Any]]] = []
    for mem_idx in group.get("memory_indices", []):
        if not (0 <= int(mem_idx) < len(slots)):
            continue
        cluster = slots[int(mem_idx)]
        original_rows.extend([dict(row) for row in cluster.get("member_rows", [])])
        indexed_clusters.append((int(mem_idx), cluster))
    for cur_idx in group.get("current_indices", []):
        if not (0 <= int(cur_idx) < len(current_rows)):
            continue
        row = dict(current_rows[int(cur_idx)])
        original_rows.append(row)
        indexed_clusters.append((len(slots) + int(cur_idx), _synthetic_singleton_cluster(row)))
    conflicting_view_ids = _same_view_conflicting_view_ids(original_rows)

    resolved_subgroup_summaries: List[Dict[str, Any]] = []
    assignment_summaries: List[Dict[str, Any]] = []
    for subgroup in resolved_subgroups:
        memory_cluster_ids = [
            _safe_int(slots[int(mem_idx)].get("cluster_id"), -1)
            for mem_idx in subgroup.get("memory_indices", [])
            if 0 <= int(mem_idx) < len(slots)
        ]
        current_object_ids = [
            _safe_int(current_rows[int(cur_idx)].get("object_global_id"), -1)
            for cur_idx in subgroup.get("current_indices", [])
            if 0 <= int(cur_idx) < len(current_rows)
        ]
        resolved_subgroup_summaries.append(
            {
                "cluster_id": _safe_int(subgroup.get("cluster_id"), -1),
                "memory_cluster_ids": memory_cluster_ids,
                "current_object_ids": current_object_ids,
                "view_ids": sorted(
                    {
                        _safe_text(view_id)
                        for view_id in subgroup.get("view_ids", [])
                        if _safe_text(view_id)
                    }
                ),
            }
        )
        for cur_idx in subgroup.get("current_indices", []):
            if not (0 <= int(cur_idx) < len(current_rows)):
                continue
            assignment_summaries.append(
                {
                    "cluster_id": _safe_int(subgroup.get("cluster_id"), -1),
                    "object_id": _safe_int(current_rows[int(cur_idx)].get("object_global_id"), -1),
                    "score": _current_best_score_to_memory_indices(
                        int(cur_idx),
                        subgroup.get("memory_indices", []),
                        cross_affinity=cross_affinity,
                    ),
                }
            )

    return {
        "step_index": int(step_index),
        "raw_dbscan_label": _safe_int(group.get("raw_dbscan_label"), -1),
        "connectivity_label": _safe_int(group.get("connectivity_label"), -1),
        "blocked_merge_cluster_ids": original_memory_cluster_ids,
        "competing_object_ids": original_current_object_ids,
        "unassigned_object_ids": [],
        "conflicting_view_ids": conflicting_view_ids,
        "collision_pairs": _same_view_collision_pairs(indexed_clusters),
        "assignments": assignment_summaries,
        "original_group": {
            "memory_cluster_ids": original_memory_cluster_ids,
            "current_object_ids": original_current_object_ids,
            "view_ids": sorted({_safe_text(row.get("view_id")) for row in original_rows if _safe_text(row.get("view_id"))}),
            "conflicting_view_ids": conflicting_view_ids,
        },
        "resolved_subgroups": resolved_subgroup_summaries,
    }


def _best_memory_detail_in_group(
    cur_idx: int,
    mem_indices: Sequence[int],
    *,
    memory_clusters: Sequence[Mapping[str, Any]],
    cross_affinity: np.ndarray,
    cross_details: Sequence[Sequence[Mapping[str, Any]]],
    current_rows: Sequence[Mapping[str, Any]],
    weight_text: float,
    weight_dinov2: float,
    weight_global_geo: float,
    weight_polar: float,
    global_sigma_m: float,
    similarity_mode: str,
    distance_gate_dsq0: float,
    enable_dinov2_scoring: bool,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    best: Optional[Tuple[float, int, int, Dict[str, Any]]] = None
    for mem_idx in mem_indices:
        cluster = memory_clusters[int(mem_idx)]
        detail = _cross_detail_for_pair(
            int(cur_idx),
            int(mem_idx),
            cross_details=cross_details,
            current_rows=current_rows,
            cluster=cluster,
            weight_text=weight_text,
            weight_dinov2=weight_dinov2,
            weight_global_geo=weight_global_geo,
            weight_polar=weight_polar,
            global_sigma_m=global_sigma_m,
            similarity_mode=similarity_mode,
            distance_gate_dsq0=distance_gate_dsq0,
            enable_dinov2_scoring=enable_dinov2_scoring,
        )
        score = float(detail.get("combined_similarity") or 0.0)
        if 0 <= int(cur_idx) < int(cross_affinity.shape[0]) and 0 <= int(mem_idx) < int(cross_affinity.shape[1]):
            score = float(cross_affinity[int(cur_idx), int(mem_idx)])
        cluster_id = _safe_int(cluster.get("cluster_id"), 10**9)
        candidate = (float(score), int(cluster_id), int(mem_idx), detail)
        if best is None or candidate[0] > best[0] or (
            math.isclose(candidate[0], best[0]) and (candidate[1], candidate[2]) < (best[1], best[2])
        ):
            best = candidate
    if best is None:
        return None
    return int(best[2]), dict(best[3])


def _dbscan_summary(
    *,
    dbscan_result: Mapping[str, Any],
    materialized_groups: Sequence[Mapping[str, Any]],
    num_noise_singletons: int,
) -> Dict[str, Any]:
    labels = np.asarray(dbscan_result.get("labels", []), dtype=np.int32)
    non_noise = {int(value) for value in labels.tolist() if int(value) >= 0}
    return {
        "dbscan_eps": _safe_float(dbscan_result.get("dbscan_eps")),
        "dbscan_min_samples": _safe_int(dbscan_result.get("dbscan_min_samples"), 0),
        "used_auto_eps": bool(dbscan_result.get("used_auto_eps")),
        "fallback_reason": dbscan_result.get("fallback_reason"),
        "spectral_embedding_dim": _safe_int(dbscan_result.get("spectral_embedding_dim"), 0),
        "raw_cluster_count": int(len(non_noise)),
        "noise_count": int(np.count_nonzero(labels == -1)),
        "materialized_cluster_count": int(len(materialized_groups)),
        "num_noise_singletons": int(num_noise_singletons),
    }


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
    weight_dinov2: float = DEFAULT_WEIGHT_DINOV2,
    weight_global_geo: float = DEFAULT_WEIGHT_GLOBAL_GEO,
    weight_polar: float = DEFAULT_WEIGHT_POLAR,
    global_sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
    similarity_mode: str = DEFAULT_SIMILARITY_MODE,
    distance_gate_dsq0: float = DEFAULT_DISTANCE_GATE_DSQ0,
    current_only_reattach_min_affinity: float = DEFAULT_CURRENT_ONLY_REATTACH_MIN_AFFINITY,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
    enforce_same_view_uniqueness: bool = DEFAULT_ENFORCE_SAME_VIEW_UNIQUENESS,
    enable_dinov2_scoring: bool = bool(ENABLE_DINOV2_SCORING),
) -> Dict[str, Any]:
    del current_only_reattach_min_affinity

    slots: List[Dict[str, Any]] = [deepcopy(dict(cluster)) for cluster in memory_clusters]
    append_cases: List[Dict[str, Any]] = []
    merge_cases: List[Dict[str, Any]] = []
    current_only_reattach_cases: List[Dict[str, Any]] = []
    same_view_block_cases: List[Dict[str, Any]] = []
    tail_spawn_cases: List[Dict[str, Any]] = []
    assignment_diagnostics: List[Dict[str, Any]] = []
    running_cluster_id = int(next_cluster_id)
    dbscan_result = _run_dbscan_over_step_graph(
        spectral_result,
        full_affinity=full_affinity,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    groups = _group_dbscan_nodes(
        dbscan_result["labels"],
        dbscan_result["connectivity_labels"],
        num_memory=len(slots),
    )

    next_memory: List[Dict[str, Any]] = []
    all_materialized_groups: List[Dict[str, Any]] = []
    num_noise_singletons = 0
    num_merged_memory_groups = 0

    for group in groups:
        raw_dbscan_label = _safe_int(group.get("raw_dbscan_label"), -1)

        if raw_dbscan_label == -1:
            num_noise_singletons += 1
        resolved_groups = (
            _resolve_same_view_constrained_group(
                group,
                slots=slots,
                current_rows=current_rows,
                cross_affinity=cross_affinity,
            )
            if enforce_same_view_uniqueness
            else [
                {
                    "memory_indices": [int(index) for index in group.get("memory_indices", [])],
                    "current_indices": [int(index) for index in group.get("current_indices", [])],
                    "view_ids": {
                        _safe_text(view_id)
                        for index in group.get("memory_indices", [])
                        for view_id in _cluster_view_id_set(slots[int(index)])
                        if _safe_text(view_id)
                    }
                    | {
                        _safe_text(current_rows[int(index)].get("view_id"))
                        for index in group.get("current_indices", [])
                        if 0 <= int(index) < len(current_rows) and _safe_text(current_rows[int(index)].get("view_id"))
                    },
                    "creation_order": 0,
                }
            ]
        )
        split_applied = bool(enforce_same_view_uniqueness and len(resolved_groups) > 1)
        materialized_subgroups: List[Dict[str, Any]] = []

        for resolved_group in resolved_groups:
            mem_indices = [int(index) for index in resolved_group.get("memory_indices", [])]
            cur_indices = [int(index) for index in resolved_group.get("current_indices", [])]
            if len(mem_indices) > 1:
                num_merged_memory_groups += 1

            if mem_indices:
                cluster_id = min(_safe_int(slots[mem_idx].get("cluster_id"), 10**9) for mem_idx in mem_indices)
            else:
                cluster_id = int(running_cluster_id)
                running_cluster_id += 1

            member_rows: List[Dict[str, Any]] = []
            for mem_idx in mem_indices:
                member_rows.extend([dict(row) for row in list(slots[mem_idx].get("member_rows", []))])
            for cur_idx in cur_indices:
                member_rows.append(dict(current_rows[cur_idx]))

            cluster = _build_cluster(int(cluster_id), member_rows)
            next_memory.append(cluster)

            materialized_group = dict(resolved_group)
            materialized_group["cluster_id"] = int(cluster_id)
            materialized_subgroups.append(materialized_group)
            all_materialized_groups.append(materialized_group)

            for cur_idx in cur_indices:
                if mem_indices:
                    best_match = _best_memory_detail_in_group(
                        int(cur_idx),
                        mem_indices,
                        memory_clusters=slots,
                        cross_affinity=cross_affinity,
                        cross_details=cross_details,
                        current_rows=current_rows,
                        weight_text=weight_text,
                        weight_dinov2=weight_dinov2,
                        weight_global_geo=weight_global_geo,
                        weight_polar=weight_polar,
                        global_sigma_m=global_sigma_m,
                        similarity_mode=similarity_mode,
                        distance_gate_dsq0=distance_gate_dsq0,
                        enable_dinov2_scoring=enable_dinov2_scoring,
                    )
                    if best_match is None:
                        similarity_reference_cluster_id = None
                        detail = None
                    else:
                        best_mem_idx, detail = best_match
                        similarity_reference_cluster_id = _safe_int(slots[best_mem_idx].get("cluster_id"), -1)
                    assignment_reason = "dbscan_attach"
                else:
                    similarity_reference_cluster_id = None
                    detail = None
                    assignment_reason = "dbscan_new_cluster"
                assignment_diagnostics.append(
                    _object_assignment_record(
                        current_rows[cur_idx],
                        step_index=int(step_index),
                        assignment_reason=assignment_reason,
                        cluster_id_at_assignment=int(cluster_id),
                        similarity_reference_cluster_id=similarity_reference_cluster_id,
                        detail=detail,
                        same_view_split_applied=split_applied,
                    )
                )

        if split_applied:
            same_view_block_cases.append(
                _make_same_view_block_case(
                    step_index=int(step_index),
                    group=group,
                    resolved_subgroups=materialized_subgroups,
                    slots=slots,
                    current_rows=current_rows,
                    cross_affinity=cross_affinity,
                )
            )

    next_memory = sorted(next_memory, key=lambda item: int(item.get("cluster_id", 10**9)))
    dbscan_summary = _dbscan_summary(
        dbscan_result=dbscan_result,
        materialized_groups=all_materialized_groups,
        num_noise_singletons=num_noise_singletons,
    )

    return {
        "memory_clusters": next_memory,
        "next_cluster_id": int(running_cluster_id),
        "append_cases": append_cases,
        "current_only_reattach_cases": current_only_reattach_cases,
        "merge_cases": merge_cases,
        "same_view_block_cases": same_view_block_cases,
        "tail_spawn_cases": tail_spawn_cases,
        "assignment_diagnostics": assignment_diagnostics,
        "num_appended": len(append_cases),
        "num_current_only_reattached": len(current_only_reattach_cases),
        "num_merged_clusters": len(merge_cases),
        "num_same_view_blocked_components": len(same_view_block_cases),
        "num_new_tail_clusters": len(tail_spawn_cases),
        "num_dbscan_clusters": len(next_memory),
        "num_noise_singletons": int(num_noise_singletons),
        "num_merged_memory_groups": int(num_merged_memory_groups),
        "dbscan_summary": dbscan_summary,
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


def _run_single_sequential_spectral_experiment(
    db_dir: str = DEFAULT_DB_DIR,
    *,
    output_dir: Optional[str] = None,
    entry_ids: Optional[Sequence[Any]] = None,
    view_ids: Optional[Sequence[str]] = None,
    weight_text: float = DEFAULT_WEIGHT_TEXT,
    weight_dinov2: float = DEFAULT_WEIGHT_DINOV2,
    weight_global_geo: float = DEFAULT_WEIGHT_GLOBAL_GEO,
    weight_polar: float = DEFAULT_WEIGHT_POLAR,
    global_sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
    similarity_mode: str = DEFAULT_SIMILARITY_MODE,
    distance_gate_dsq0: float = DEFAULT_DISTANCE_GATE_DSQ0,
    min_cross_affinity: float = DEFAULT_CROSS_AFFINITY_MIN,
    current_only_reattach_min_affinity: float = DEFAULT_CURRENT_ONLY_REATTACH_MIN_AFFINITY,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
    enforce_same_view_uniqueness: bool = DEFAULT_ENFORCE_SAME_VIEW_UNIQUENESS,
    enable_dinov2_scoring: bool = bool(ENABLE_DINOV2_SCORING),
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
    assignment_records: List[Dict[str, Any]] = [
        _object_assignment_record(
            row,
            step_index=0,
            assignment_reason="initial_seed",
            cluster_id_at_assignment=_safe_int(cluster.get("cluster_id"), -1),
            similarity_reference_cluster_id=None,
            detail=None,
            similarity_detail_status="initial_seed",
        )
        for row, cluster in zip(initial_view["objects"], memory_clusters)
    ]

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
    total_same_view_masked_edges = 0

    for step_offset, current_view in enumerate(selected_views[1:], start=1):
        cross_affinity, cross_details = build_cross_affinity_matrix(
            memory_clusters,
            current_view["objects"],
            weight_text=weight_text,
            weight_dinov2=weight_dinov2,
            weight_global_geo=weight_global_geo,
            weight_polar=weight_polar,
            global_sigma_m=global_sigma_m,
            similarity_mode=similarity_mode,
            distance_gate_dsq0=distance_gate_dsq0,
            enable_dinov2_scoring=enable_dinov2_scoring,
        )
        num_same_view_masked_edges = 0
        if enforce_same_view_uniqueness:
            cross_affinity, cross_details, num_same_view_masked_edges = _apply_same_view_hard_mask_to_cross_affinity(
                memory_clusters,
                current_view["objects"],
                cross_affinity=cross_affinity,
                cross_details=cross_details,
            )
            total_same_view_masked_edges += int(num_same_view_masked_edges)
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

        spectral_result = _run_capped_sequential_spectral_clustering(
            full_affinity,
            object_ids=list(range(full_affinity.shape[0])),
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
            weight_dinov2=weight_dinov2,
            weight_global_geo=weight_global_geo,
            weight_polar=weight_polar,
            global_sigma_m=global_sigma_m,
            similarity_mode=similarity_mode,
            distance_gate_dsq0=distance_gate_dsq0,
            current_only_reattach_min_affinity=current_only_reattach_min_affinity,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            enforce_same_view_uniqueness=enforce_same_view_uniqueness,
            enable_dinov2_scoring=enable_dinov2_scoring,
        )
        memory_clusters = update["memory_clusters"]
        next_cluster_id = int(update["next_cluster_id"])
        all_append_cases.extend(update["append_cases"])
        all_same_view_block_cases.extend(update["same_view_block_cases"])
        all_tail_spawn_cases.extend(update["tail_spawn_cases"])
        assignment_records.extend(update["assignment_diagnostics"])

        step_report = {
            "step_index": int(step_offset),
            "view_id": current_view["view_id"],
            "enforce_same_view_uniqueness": bool(enforce_same_view_uniqueness),
            "num_current_objects": len(current_view["objects"]),
            "num_existing_clusters": full_affinity.shape[0] - len(current_view["objects"]),
            "num_appended": int(update["num_appended"]),
            "num_current_only_reattached": int(update["num_current_only_reattached"]),
            "num_merged_clusters": int(update["num_merged_clusters"]),
            "num_same_view_masked_edges": int(num_same_view_masked_edges),
            "num_same_view_blocked_components": int(update["num_same_view_blocked_components"]),
            "num_new_tail_clusters": int(update["num_new_tail_clusters"]),
            "num_dbscan_clusters": int(update["num_dbscan_clusters"]),
            "num_noise_singletons": int(update["num_noise_singletons"]),
            "num_merged_memory_groups": int(update["num_merged_memory_groups"]),
            "spectral_summary": _spectral_result_summary(spectral_result),
            "dbscan_summary": dict(update["dbscan_summary"]),
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
    materialized_assignment_rows = _materialize_object_cluster_similarity_rows(
        assignment_records,
        final_cluster_id_by_object=_final_cluster_id_by_object(memory_clusters),
    )
    object_cluster_similarity_table_path = _write_object_cluster_similarity_table(
        root / "object_cluster_similarity_table.csv",
        materialized_assignment_rows,
    )
    object_cluster_similarity_tables_by_step = _write_object_cluster_similarity_tables_by_step(
        root,
        materialized_assignment_rows,
    )

    report = {
        "db_dir": str(db_dir),
        "output_dir": str(root),
        "selected_view_ids": list(sequence["selected_view_ids"]),
        "views": [_view_summary(view, stored_image_by_view_id.get(_safe_text(view.get("view_id")))) for view in selected_views],
        "view_object_counts": {view["view_id"]: len(view["objects"]) for view in selected_views},
        "weights": {
            "text": float(weight_text),
            "dinov2": float(weight_dinov2),
            "global_geo": float(weight_global_geo),
            "polar": float(weight_polar),
            "normalized": _normalize_weight_triplet(weight_text, weight_global_geo, weight_polar, weight_dinov2),
        },
        "enable_dinov2_scoring": bool(enable_dinov2_scoring),
        "global_sigma_m": float(global_sigma_m),
        "similarity_mode": str(similarity_mode),
        "distance_gate_dsq0": float(distance_gate_dsq0),
        "min_cross_affinity": float(min_cross_affinity),
        "current_only_reattach_min_affinity": float(current_only_reattach_min_affinity),
        "enforce_same_view_uniqueness": bool(enforce_same_view_uniqueness),
        "dbscan_eps": None if dbscan_eps is None else float(dbscan_eps),
        "dbscan_min_samples": int(dbscan_min_samples),
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
        "total_same_view_masked_edges": int(total_same_view_masked_edges),
        "total_same_view_blocked_components": int(
            sum(_safe_int(step.get("num_same_view_blocked_components"), 0) for step in step_reports)
        ),
        "append_case_examples": [_append_case_summary(case) for case in all_append_cases[:12]],
        "same_view_block_case_examples": [_same_view_block_case_summary(case) for case in all_same_view_block_cases[:12]],
        "tail_spawn_case_examples": [_tail_spawn_case_summary(case) for case in all_tail_spawn_cases[:12]],
        "object_cluster_similarity_table": str(object_cluster_similarity_table_path),
        "object_cluster_similarity_tables_by_step": object_cluster_similarity_tables_by_step,
    }
    progression_manifest = generate_cumulative_cluster_progression_artifacts(str(root))
    report["cumulative_cluster_progression_manifest"] = str(root / "cumulative_cluster_progression_manifest.json")
    report["cumulative_cluster_progression_overview"] = progression_manifest.get("overview_path")
    _write_json(root / "experiment_report.json", report)
    return report


def _distance_gate_dir_token(value: float) -> str:
    token = f"{float(value):g}"
    token = token.replace("-", "neg_").replace(".", "p")
    return token


def run_sequential_spectral_experiment(
    db_dir: str = DEFAULT_DB_DIR,
    *,
    output_dir: Optional[str] = None,
    entry_ids: Optional[Sequence[Any]] = None,
    view_ids: Optional[Sequence[str]] = None,
    weight_text: float = DEFAULT_WEIGHT_TEXT,
    weight_dinov2: float = DEFAULT_WEIGHT_DINOV2,
    weight_global_geo: float = DEFAULT_WEIGHT_GLOBAL_GEO,
    weight_polar: float = DEFAULT_WEIGHT_POLAR,
    global_sigma_m: float = DEFAULT_GLOBAL_SIGMA_M,
    similarity_mode: str = DEFAULT_SIMILARITY_MODE,
    distance_gate_dsq0: float = DEFAULT_DISTANCE_GATE_DSQ0,
    distance_gate_dsq0_values: Optional[Sequence[Any]] = None,
    min_cross_affinity: float = DEFAULT_CROSS_AFFINITY_MIN,
    current_only_reattach_min_affinity: float = DEFAULT_CURRENT_ONLY_REATTACH_MIN_AFFINITY,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
    enforce_same_view_uniqueness: bool = DEFAULT_ENFORCE_SAME_VIEW_UNIQUENESS,
    enable_dinov2_scoring: bool = bool(ENABLE_DINOV2_SCORING),
) -> Dict[str, Any]:
    normalized_dsq0_values = _normalize_float_list(distance_gate_dsq0_values)
    if distance_gate_dsq0_values is not None and not normalized_dsq0_values:
        normalized_dsq0_values = [float(v) for v in DEFAULT_DISTANCE_GATE_DSQ0_SWEEP]
    if not normalized_dsq0_values:
        return _run_single_sequential_spectral_experiment(
            db_dir=db_dir,
            output_dir=output_dir,
            entry_ids=entry_ids,
            view_ids=view_ids,
            weight_text=weight_text,
            weight_dinov2=weight_dinov2,
            weight_global_geo=weight_global_geo,
            weight_polar=weight_polar,
            global_sigma_m=global_sigma_m,
            similarity_mode=similarity_mode,
            distance_gate_dsq0=distance_gate_dsq0,
            min_cross_affinity=min_cross_affinity,
            current_only_reattach_min_affinity=current_only_reattach_min_affinity,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            enforce_same_view_uniqueness=enforce_same_view_uniqueness,
            enable_dinov2_scoring=enable_dinov2_scoring,
        )
    if len(normalized_dsq0_values) == 1:
        return _run_single_sequential_spectral_experiment(
            db_dir=db_dir,
            output_dir=output_dir,
            entry_ids=entry_ids,
            view_ids=view_ids,
            weight_text=weight_text,
            weight_dinov2=weight_dinov2,
            weight_global_geo=weight_global_geo,
            weight_polar=weight_polar,
            global_sigma_m=global_sigma_m,
            similarity_mode=similarity_mode,
            distance_gate_dsq0=float(normalized_dsq0_values[0]),
            min_cross_affinity=min_cross_affinity,
            current_only_reattach_min_affinity=current_only_reattach_min_affinity,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            enforce_same_view_uniqueness=enforce_same_view_uniqueness,
            enable_dinov2_scoring=enable_dinov2_scoring,
        )

    base_root = Path(output_dir) if output_dir else Path(db_dir) / "sequential_spectral_experiment_sweep"
    sweep_root = _make_run_output_dir(base_root)
    sweep_runs: List[Dict[str, Any]] = []
    for dsq0 in normalized_dsq0_values:
        run_report = _run_single_sequential_spectral_experiment(
            db_dir=db_dir,
            output_dir=str(sweep_root / f"dsq0_{_distance_gate_dir_token(float(dsq0))}"),
            entry_ids=entry_ids,
            view_ids=view_ids,
            weight_text=weight_text,
            weight_dinov2=weight_dinov2,
            weight_global_geo=weight_global_geo,
            weight_polar=weight_polar,
            global_sigma_m=global_sigma_m,
            similarity_mode=similarity_mode,
            distance_gate_dsq0=float(dsq0),
            min_cross_affinity=min_cross_affinity,
            current_only_reattach_min_affinity=current_only_reattach_min_affinity,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            enforce_same_view_uniqueness=enforce_same_view_uniqueness,
            enable_dinov2_scoring=enable_dinov2_scoring,
        )
        sweep_runs.append(
            {
                "dsq0": float(dsq0),
                "output_dir": str(run_report.get("output_dir") or ""),
                "final_cluster_count": int(run_report.get("final_cluster_count") or 0),
                "enforce_same_view_uniqueness": bool(run_report.get("enforce_same_view_uniqueness")),
                "dbscan_eps": run_report.get("dbscan_eps"),
                "dbscan_min_samples": int(run_report.get("dbscan_min_samples") or 0),
                "total_appended": int(run_report.get("total_appended") or 0),
                "total_current_only_reattached": int(run_report.get("total_current_only_reattached") or 0),
                "total_new_tail_clusters": int(run_report.get("total_new_tail_clusters") or 0),
                "total_merged_clusters": int(run_report.get("total_merged_clusters") or 0),
                "total_same_view_masked_edges": int(run_report.get("total_same_view_masked_edges") or 0),
                "total_same_view_blocked_components": int(
                    run_report.get("total_same_view_blocked_components") or 0
                ),
                "object_cluster_similarity_table": str(run_report.get("object_cluster_similarity_table") or ""),
            }
        )
    summary = {
        "db_dir": str(db_dir),
        "output_dir": str(sweep_root),
        "similarity_mode": str(similarity_mode),
        "enable_dinov2_scoring": bool(enable_dinov2_scoring),
        "distance_gate_dsq0_values": [float(v) for v in normalized_dsq0_values],
        "enforce_same_view_uniqueness": bool(enforce_same_view_uniqueness),
        "dbscan_eps": None if dbscan_eps is None else float(dbscan_eps),
        "dbscan_min_samples": int(dbscan_min_samples),
        "runs": sweep_runs,
    }
    _write_json(sweep_root / "sweep_summary.json", summary)
    return summary


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
    parser.add_argument("--weight_dinov2", type=float, default=DEFAULT_WEIGHT_DINOV2, help="DINOv2 branch weight")
    parser.add_argument("--weight_global_geo", type=float, default=DEFAULT_WEIGHT_GLOBAL_GEO, help="Global xyz branch weight")
    parser.add_argument("--weight_polar", type=float, default=DEFAULT_WEIGHT_POLAR, help="Polar branch weight")
    parser.add_argument(
        "--enable_dinov2_scoring",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_DINOV2_SCORING,
        help="Whether to fuse DINOv2 visual consistency into the sequential affinity score.",
    )
    parser.add_argument("--global_sigma_m", type=float, default=DEFAULT_GLOBAL_SIGMA_M, help="Global xyz gaussian sigma")
    parser.add_argument(
        "--similarity_mode",
        default=DEFAULT_SIMILARITY_MODE,
        choices=["cosine_geo_gate", "legacy_weighted_fusion"],
        help="Similarity computation mode",
    )
    parser.add_argument(
        "--distance_gate_dsq0",
        type=float,
        default=DEFAULT_DISTANCE_GATE_DSQ0,
        help="Distance-gating constant for cosine_geo_gate mode",
    )
    parser.add_argument(
        "--distance_gate_dsq0_values",
        default=None,
        help="Comma-separated dsq0 values for sweep mode",
    )
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
        help="Deprecated compatibility flag; retained but ignored by the DBSCAN materialization stage",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=None,
        help="Optional DBSCAN epsilon on the step spectral embedding; auto-estimated when omitted",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=DEFAULT_DBSCAN_MIN_SAMPLES,
        help="DBSCAN min_samples on the step spectral embedding",
    )
    parser.add_argument(
        "--enforce_same_view_uniqueness",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENFORCE_SAME_VIEW_UNIQUENESS,
        help="Whether final materialized clusters may contain at most one object from each view_id",
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
        weight_dinov2=float(args.weight_dinov2),
        weight_global_geo=float(args.weight_global_geo),
        weight_polar=float(args.weight_polar),
        enable_dinov2_scoring=bool(args.enable_dinov2_scoring),
        global_sigma_m=float(args.global_sigma_m),
        similarity_mode=str(args.similarity_mode),
        distance_gate_dsq0=float(args.distance_gate_dsq0),
        distance_gate_dsq0_values=_normalize_float_list([args.distance_gate_dsq0_values])
        if args.distance_gate_dsq0_values
        else None,
        min_cross_affinity=float(args.min_cross_affinity),
        current_only_reattach_min_affinity=float(args.current_only_reattach_min_affinity),
        dbscan_eps=None if args.dbscan_eps is None else float(args.dbscan_eps),
        dbscan_min_samples=int(args.dbscan_min_samples),
        enforce_same_view_uniqueness=bool(args.enforce_same_view_uniqueness),
    )
    print(json.dumps(_to_serializable(report), ensure_ascii=True))
    return report


if __name__ == "__main__":
    main()
