import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from spatial_rag.config import (
    OBJECT_CACHE_DIR,
    OBJECT_MAX_PER_FRAME,
    OBJECT_PARSE_RETRIES,
    OBJECT_SURROUNDING_MAX,
    OBJECT_TEXT_MODE,
    OBJECT_USE_CACHE,
    SCENE_PATH,
    SPATIAL_DB_VLM_MODEL,
    VLM_ANGLE_SPLIT_ENABLE,
    VLM_ANGLE_STEP,
)
from spatial_rag.graph_builder import build_graph_payload
from spatial_rag.object_canonicalizer import UNKNOWN_TEXT_TOKEN, compose_frame_text, select_object_text, sorted_objects
from spatial_rag.object_index import load_object_db
from spatial_rag.spatial_db_builder import (
    _build_view_attribute,
    _enrich_scene_objects_geometry,
    _format_object_text_long,
    _make_object_record,
    _normalize_angle_bucket,
    _parse_objects_with_retry,
    _serialize_surrounding_context,
)
from spatial_rag.vlm_captioner import VLMCaptioner
from spatial_rag.vpr_query import _set_agent_pose_2d


DEFAULT_TEXT_MODE = "long"
DEFAULT_GROUP_MODE = "place"
DEFAULT_SAME_VIEW_POLICY = "soft_penalty"
DEFAULT_SAME_VIEW_PENALTY = 0.25
DEFAULT_CLUSTER_COUNT_MODE = "eigengap"
DEFAULT_MULTI_VIEW_SIMILARITY_THRESHOLD = 0.60
CLUSTERED_SIMILARITY_HEATMAP_TITLE = "Clustered Similarity Matrix (Multi-view Object Deduplication)"
CLUSTERED_SIMILARITY_OFFDIAG_HEATMAP_TITLE = "Clustered Similarity Matrix (No Self-Similarity)"
REFINED_GRAPH_HEATMAP_TITLE = "Refined Graph-Based Clustered Similarity Matrix"
REFINED_GRAPH_OFFDIAG_HEATMAP_TITLE = "Refined Graph-Based Clustered Similarity Matrix (No Self-Similarity)"
REFINED_GRAPH_DIAG1_HEATMAP_TITLE = "Refined Graph-Based Clustered Similarity Matrix (Diag=1)"
DEFAULT_REFINED_GRAPH_KNN_K = 6
DEFAULT_REFINED_GRAPH_SPECTRAL_DIM = 6
DEFAULT_REFINED_GRAPH_DBSCAN_MIN_SAMPLES = 2
DEFAULT_MANUAL_ANCHOR_RADIUS_M = 1.5
DEFAULT_MAX_SNAP_DISTANCE_M = 0.75
DEFAULT_MIN_POSE_SEPARATION_M = 0.1


def _safe_text(value: Any, default: str = "") -> str:
    text = " ".join(str(value or "").strip().split())
    return text or default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_serializable(row), ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serializable(payload), ensure_ascii=True, indent=2), encoding="utf-8")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _normalize_text_mode(text_mode: str) -> str:
    mode = str(text_mode or DEFAULT_TEXT_MODE).strip().lower()
    if mode not in {"short", "long", "long_neighbors"}:
        raise ValueError(f"Unsupported text_mode: {text_mode}")
    return mode


def _normalize_group_mode(group_mode: str) -> str:
    mode = str(group_mode or DEFAULT_GROUP_MODE).strip().lower()
    if mode not in {"place", "selected_views", "manual_anchor_center"}:
        raise ValueError(f"Unsupported group_mode: {group_mode}")
    return mode


def _normalize_same_view_policy(policy: str) -> str:
    normalized = str(policy or DEFAULT_SAME_VIEW_POLICY).strip().lower()
    if normalized not in {"soft_penalty", "hard_block", "none"}:
        raise ValueError(f"Unsupported same_view_policy: {policy}")
    return normalized


def _normalize_cluster_count_mode(mode: str) -> str:
    normalized = str(mode or DEFAULT_CLUSTER_COUNT_MODE).strip().lower()
    if normalized not in {"fixed", "eigengap"}:
        raise ValueError(f"Unsupported cluster_count_mode: {mode}")
    return normalized


def _parse_entry_ids_arg(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    tokens = [part.strip() for part in str(value).split(",")]
    out: List[int] = []
    for token in tokens:
        if not token:
            continue
        out.append(int(token))
    return out or None


def _has_usable_text(text: str) -> bool:
    normalized = _safe_text(text).lower()
    return bool(normalized) and normalized not in {"unknown", "none", "null", "n/a", "na"}


def _sanitize_neighbor_text_fragment(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    return text.replace("|", "/").replace(";", ",").replace("[", "(").replace("]", ")")


def _serialize_surrounding_context_for_embedding(
    surrounding_context: Any,
    *,
    limit: int = OBJECT_SURROUNDING_MAX,
) -> str:
    serialized: List[str] = []
    for item in list(surrounding_context or [])[: int(limit)]:
        if isinstance(item, Mapping):
            label_value = item.get("label")
            relation_value = item.get("relation_to_primary")
            distance_value = item.get("distance_from_primary_m")
        else:
            label_value = getattr(item, "label", None)
            relation_value = getattr(item, "relation_to_primary", None)
            distance_value = getattr(item, "distance_from_primary_m", None)

        label = _sanitize_neighbor_text_fragment(label_value)
        if not _has_usable_text(label):
            continue

        detail_parts: List[str] = []
        relation = _sanitize_neighbor_text_fragment(relation_value)
        if _has_usable_text(relation):
            detail_parts.append(relation)
        distance = _safe_float(distance_value)
        if distance is not None:
            detail_parts.append(f"{round(float(distance), 1):.1f}m")

        entry = label
        if detail_parts:
            entry = f"{entry} [{', '.join(detail_parts)}]"
        serialized.append(entry)

    return "; ".join(serialized) if serialized else "none"


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def _l2_normalize_vec(vec: np.ndarray) -> np.ndarray:
    flat = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = max(float(np.linalg.norm(flat)), 1e-12)
    return flat / norm


def _slugify_group_id(group_id: str) -> str:
    raw = str(group_id or "group").strip()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if slug:
        return slug
    return "group"


def _filter_observations_by_entry_ids(
    observations: Sequence[Mapping[str, Any]],
    entry_ids: Optional[Sequence[int]],
) -> List[Mapping[str, Any]]:
    if not entry_ids:
        return list(observations)
    allowed = {int(entry_id) for entry_id in entry_ids}
    return [row for row in observations if _safe_int(row.get("entry_id"), -1) in allowed]


def _observation_sort_key(row: Mapping[str, Any]) -> Tuple[int, str, int]:
    return (
        _safe_int(row.get("object_global_id"), 10**9),
        str(row.get("view_id") or ""),
        _safe_int(row.get("entry_id"), 10**9),
    )


def _view_sort_key(row: Mapping[str, Any]) -> Tuple[int, str]:
    return (_safe_int(row.get("entry_id"), 10**9), str(row.get("synthetic_view_id") or row.get("view_id") or ""))


def _resolve_object_text(row: Mapping[str, Any], text_mode: str = DEFAULT_TEXT_MODE) -> str:
    mode = _normalize_text_mode(text_mode)
    if mode in {"long", "long_neighbors"}:
        candidates = (
            row.get("text_input_for_clip_long"),
            row.get("object_text_long"),
            row.get("long_form_open_description"),
            row.get("description"),
        )
    else:
        candidates = (
            row.get("text_input_for_clip_short"),
            row.get("object_text_short"),
            row.get("description"),
            row.get("long_form_open_description"),
        )
    for value in candidates:
        text = _safe_text(value)
        if _has_usable_text(text):
            if mode == "long_neighbors":
                neighbor_text = _serialize_surrounding_context_for_embedding(row.get("surrounding_context"))
                return f"{text} | neighbors: {neighbor_text}"
            return text
    return ""


def _load_precomputed_object_embeddings(
    db_dir: str,
    text_mode: str = DEFAULT_TEXT_MODE,
) -> Dict[int, np.ndarray]:
    mode = _normalize_text_mode(text_mode)
    if mode == "long_neighbors":
        return {}
    loaded = load_object_db(db_dir, text_mode=mode)
    if loaded is None:
        return {}
    meta_rows, emb, _entry_to_indices = loaded
    if emb.ndim != 2 or emb.shape[0] != len(meta_rows):
        raise ValueError(
            f"Invalid object embedding payload in {db_dir}: meta={len(meta_rows)} emb_shape={tuple(emb.shape)}"
        )
    out: Dict[int, np.ndarray] = {}
    for idx, row in enumerate(meta_rows):
        obj_id = _safe_int(row.get("object_global_id"), idx)
        out[obj_id] = _l2_normalize_vec(emb[idx])
    return out


def _build_place_observation_rows(
    db_dir: str,
    graph_payload: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    payload = dict(graph_payload) if graph_payload is not None else build_graph_payload(db_dir)
    raw_object_rows = _load_jsonl(Path(db_dir) / "object_meta.jsonl")
    raw_by_id = {
        _safe_int(row.get("object_global_id"), idx): dict(row)
        for idx, row in enumerate(raw_object_rows)
    }
    views_by_id = {
        str(row["view_id"]): dict(row)
        for row in payload.get("views", [])
        if row.get("view_id") is not None
    }
    places_by_id = {
        str(row["place_id"]): dict(row)
        for row in payload.get("places", [])
        if row.get("place_id") is not None
    }
    observations: List[Dict[str, Any]] = []
    for row in payload.get("objects", []):
        obj_id = _safe_int(row.get("object_global_id"), -1)
        combined = dict(raw_by_id.get(obj_id, {}))
        combined.update(dict(row))
        view = views_by_id.get(str(combined.get("view_id") or ""))
        if view is not None:
            combined.setdefault("file_name", view.get("file_name"))
            combined["view_orientation_deg"] = _safe_float(view.get("orientation_deg"))
        place = places_by_id.get(str(combined.get("place_id") or ""))
        if place is not None:
            combined["group_type"] = "place"
            combined["group_id"] = str(place["place_id"])
            combined.setdefault("room_function", place.get("room_function"))
            combined.setdefault("view_type", place.get("view_type"))
        combined["object_global_id"] = obj_id
        combined["observation_id"] = str(combined.get("obs_id") or f"obs_{obj_id:06d}")
        combined["anchor_id"] = None
        combined["synthetic_view_id"] = combined.get("view_id")
        combined["description"] = _safe_text(combined.get("description"))
        combined["long_form_open_description"] = _safe_text(combined.get("long_form_open_description"))
        combined["label"] = _safe_text(combined.get("label") or combined.get("object_class") or "unknown", "unknown")
        observations.append(combined)
    observations.sort(key=_observation_sort_key)
    return observations


def _attach_object_embeddings(
    rows: Sequence[Mapping[str, Any]],
    *,
    db_dir: Optional[str],
    text_mode: str,
    embedder=None,
) -> List[Dict[str, Any]]:
    mode = _normalize_text_mode(text_mode)
    prepared: List[Dict[str, Any]] = [dict(row) for row in rows]
    emb_by_obj_id: Dict[int, np.ndarray] = {}
    if db_dir:
        emb_by_obj_id = _load_precomputed_object_embeddings(db_dir, text_mode=mode)
    missing_rows: List[int] = []
    if embedder is None:
        embedder_instance = None
    else:
        embedder_instance = embedder

    for index, row in enumerate(prepared):
        obj_id = _safe_int(row.get("object_global_id"), -1)
        embedding_text = _resolve_object_text(row, text_mode=mode)
        row["embedding_text"] = embedding_text
        precomputed = emb_by_obj_id.get(obj_id)
        if precomputed is not None:
            row["embedding"] = precomputed.astype(np.float32)
            continue
        if _has_usable_text(embedding_text):
            missing_rows.append(index)
        else:
            row["embedding"] = None

    if missing_rows and embedder_instance is None:
        from spatial_rag.embedder import Embedder

        embedder_instance = Embedder()

    for index in missing_rows:
        vec = embedder_instance.embed_text(str(prepared[index]["embedding_text"]))
        prepared[index]["embedding"] = _l2_normalize_vec(np.asarray(vec, dtype=np.float32))

    return prepared


def load_object_observations(
    db_dir: str,
    text_mode: str = DEFAULT_TEXT_MODE,
    graph_payload: Optional[Mapping[str, Any]] = None,
    embedder=None,
) -> List[Dict[str, Any]]:
    """Load object observations from an existing spatial DB.

    Each row is still an object observation, not a resolved physical instance.
    v1 only attaches description-text embeddings. Later versions can fuse image or spatial similarity here.
    """

    observations = _build_place_observation_rows(db_dir=db_dir, graph_payload=graph_payload)
    return _attach_object_embeddings(
        observations,
        db_dir=db_dir,
        text_mode=text_mode,
        embedder=embedder,
    )


def group_objects_by_scope(
    observations: Sequence[Mapping[str, Any]],
    group_mode: str = DEFAULT_GROUP_MODE,
) -> Dict[str, List[Dict[str, Any]]]:
    mode = _normalize_group_mode(group_mode)
    if mode == "selected_views":
        grouped = {"selected_views": [dict(row) for row in observations]}
        grouped["selected_views"].sort(key=_observation_sort_key)
        return grouped
    key_field = "place_id" if mode == "place" else "anchor_id"
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in observations:
        group_id = _safe_text(row.get(key_field))
        if not group_id:
            continue
        grouped[group_id].append(dict(row))
    for rows in grouped.values():
        rows.sort(key=_observation_sort_key)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def compute_text_similarity(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    normed = _l2_normalize_rows(np.asarray(embeddings, dtype=np.float32))
    matrix = normed @ normed.T
    matrix = np.clip(matrix, -1.0, 1.0).astype(np.float32)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def build_similarity_matrix_from_descriptions(
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    embeddings = []
    for row in rows:
        vec = row.get("embedding")
        if vec is None:
            raise ValueError("Missing embedding on observation row while building similarity matrix")
        embeddings.append(np.asarray(vec, dtype=np.float32).reshape(-1))
    matrix = compute_text_similarity(np.vstack(embeddings))
    return matrix


def _same_view_mask(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    size = len(rows)
    mask = np.zeros((size, size), dtype=bool)
    view_ids = [str(row.get("view_id") or row.get("synthetic_view_id") or row.get("entry_id") or "") for row in rows]
    for i in range(size):
        for j in range(i + 1, size):
            if view_ids[i] and view_ids[i] == view_ids[j]:
                mask[i, j] = True
                mask[j, i] = True
    return mask


def _same_view_mask_from_view_ids(view_ids: Sequence[Any]) -> np.ndarray:
    size = len(view_ids)
    mask = np.zeros((size, size), dtype=bool)
    normalized = [_safe_text(view_id) for view_id in view_ids]
    for i in range(size):
        for j in range(i + 1, size):
            if normalized[i] and normalized[i] == normalized[j]:
                mask[i, j] = True
                mask[j, i] = True
    return mask


def _apply_top_k_filter(affinity: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    if top_k is None:
        return affinity
    keep = max(0, int(top_k))
    if affinity.shape[0] <= 1 or keep <= 0:
        out = np.eye(affinity.shape[0], dtype=np.float32)
        return out
    size = affinity.shape[0]
    kept = np.zeros_like(affinity, dtype=np.float32)
    for row_index in range(size):
        row = affinity[row_index].copy()
        row[row_index] = -np.inf
        order = np.argsort(-row)
        for col_index in order[:keep]:
            if not np.isfinite(row[col_index]) or row[col_index] <= 0.0:
                continue
            kept[row_index, col_index] = affinity[row_index, col_index]
    kept = np.maximum(kept, kept.T)
    np.fill_diagonal(kept, 1.0)
    return kept.astype(np.float32)


def build_multiview_affinity_matrix(
    similarity_matrix: np.ndarray,
    view_ids: Sequence[Any],
    *,
    same_view_penalty: float = DEFAULT_SAME_VIEW_PENALTY,
    similarity_threshold: Optional[float] = DEFAULT_MULTI_VIEW_SIMILARITY_THRESHOLD,
) -> np.ndarray:
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(f"Expected square similarity matrix, got shape {similarity_matrix.shape}")
    if similarity_matrix.shape[0] != len(view_ids):
        raise ValueError(
            f"view_ids length mismatch: len(view_ids)={len(view_ids)} similarity_shape={tuple(similarity_matrix.shape)}"
        )
    base_similarity = np.asarray(similarity_matrix, dtype=np.float32)
    affinity = np.clip(base_similarity, 0.0, 1.0).astype(np.float32)
    same_view = _same_view_mask_from_view_ids(view_ids)
    affinity[same_view] = affinity[same_view] * float(same_view_penalty)
    if similarity_threshold is not None:
        threshold = float(similarity_threshold)
        off_diag = ~np.eye(affinity.shape[0], dtype=bool)
        affinity[np.logical_and(off_diag, base_similarity < threshold)] = 0.0
    affinity = np.maximum(affinity, affinity.T)
    np.fill_diagonal(affinity, 1.0)
    return affinity.astype(np.float32)


def apply_constraints(
    similarity_matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    same_view_policy: str = DEFAULT_SAME_VIEW_POLICY,
    same_view_penalty: float = DEFAULT_SAME_VIEW_PENALTY,
    min_similarity: Optional[float] = None,
    top_k: Optional[int] = None,
) -> np.ndarray:
    policy = _normalize_same_view_policy(same_view_policy)
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(f"Expected square similarity matrix, got shape {similarity_matrix.shape}")
    if similarity_matrix.shape[0] != len(rows):
        raise ValueError(
            f"Row/matrix mismatch: rows={len(rows)} similarity_shape={tuple(similarity_matrix.shape)}"
        )
    base_similarity = np.asarray(similarity_matrix, dtype=np.float32)
    adjusted = base_similarity.copy()
    same_view = _same_view_mask(rows)
    if policy == "hard_block":
        adjusted[same_view] = 0.0
    elif policy == "soft_penalty":
        adjusted[same_view] = adjusted[same_view] * float(same_view_penalty)
    affinity = np.clip(adjusted, 0.0, 1.0).astype(np.float32)
    if min_similarity is not None:
        threshold = float(min_similarity)
        off_diag = ~np.eye(affinity.shape[0], dtype=bool)
        affinity[np.logical_and(off_diag, base_similarity < threshold)] = 0.0
    affinity = _apply_top_k_filter(affinity, top_k=top_k)
    affinity = np.maximum(affinity, affinity.T)
    np.fill_diagonal(affinity, 1.0)
    return affinity.astype(np.float32)


def _positive_off_diagonal_count(affinity: np.ndarray) -> int:
    if affinity.size == 0:
        return 0
    mask = ~np.eye(affinity.shape[0], dtype=bool)
    return int(np.count_nonzero(affinity[mask] > 0.0))


def _connected_component_count(affinity: np.ndarray) -> int:
    if affinity.shape[0] == 0:
        return 0
    adjacency = np.asarray(affinity > 0.0, dtype=np.int8)
    np.fill_diagonal(adjacency, 0)
    graph = csr_matrix(adjacency)
    n_components, _labels = connected_components(graph, directed=False, connection="weak")
    return int(n_components)


def build_normalized_laplacian(affinity_matrix: np.ndarray) -> np.ndarray:
    if affinity_matrix.ndim != 2 or affinity_matrix.shape[0] != affinity_matrix.shape[1]:
        raise ValueError(f"Expected square affinity matrix, got shape {affinity_matrix.shape}")
    values = np.asarray(affinity_matrix, dtype=np.float64)
    size = values.shape[0]
    if size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    degrees = np.sum(values, axis=1)
    inv_sqrt = np.zeros_like(degrees)
    valid = degrees > 1e-12
    inv_sqrt[valid] = 1.0 / np.sqrt(degrees[valid])
    normalized_affinity = (values * inv_sqrt[:, None]) * inv_sqrt[None, :]
    lap = np.eye(size, dtype=np.float64) - normalized_affinity
    lap = 0.5 * (lap + lap.T)
    return lap.astype(np.float32)


def estimate_cluster_count_eigengap(
    affinity_matrix: np.ndarray,
    *,
    max_clusters: Optional[int] = None,
) -> int:
    if affinity_matrix.ndim != 2 or affinity_matrix.shape[0] != affinity_matrix.shape[1]:
        raise ValueError(f"Expected square affinity matrix, got shape {affinity_matrix.shape}")
    size = affinity_matrix.shape[0]
    if size == 0:
        return 0
    if size == 1:
        return 1
    if _positive_off_diagonal_count(affinity_matrix) == 0:
        return size

    component_lb = max(1, _connected_component_count(affinity_matrix))
    upper = size if max_clusters is None else max(1, min(size, int(max_clusters)))
    if component_lb >= upper:
        return int(component_lb)

    lap = build_normalized_laplacian(affinity_matrix)
    eigvals = np.linalg.eigvalsh(lap)
    best_k = int(component_lb)
    best_gap = -np.inf
    for k in range(int(component_lb), int(upper)):
        gap = float(eigvals[k] - eigvals[k - 1])
        if gap > best_gap:
            best_gap = gap
            best_k = int(k)
    best_k = max(int(component_lb), min(int(best_k), int(upper)))
    return int(best_k)


def _choose_cluster_count(
    affinity_matrix: np.ndarray,
    *,
    cluster_count_mode: str,
    n_clusters: Optional[int],
) -> int:
    mode = _normalize_cluster_count_mode(cluster_count_mode)
    size = affinity_matrix.shape[0]
    if size == 0:
        return 0
    if size == 1:
        return 1
    if _positive_off_diagonal_count(affinity_matrix) == 0:
        return size
    component_lb = max(1, _connected_component_count(affinity_matrix))
    if mode == "fixed":
        if n_clusters is None:
            raise ValueError("n_clusters must be provided when cluster_count_mode='fixed'")
        requested = max(1, min(size, int(n_clusters)))
        return max(component_lb, requested)
    estimated = estimate_cluster_count_eigengap(affinity_matrix, max_clusters=size)
    return max(component_lb, min(size, int(estimated)))


def _spectral_embedding(affinity_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    lap = build_normalized_laplacian(affinity_matrix)
    _eigvals, eigvecs = np.linalg.eigh(lap)
    embedding = np.asarray(eigvecs[:, :n_clusters], dtype=np.float32)
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    return embedding / row_norms


def _init_kmeans_pp(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    num_rows = data.shape[0]
    centers = np.zeros((k, data.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, num_rows))
    centers[0] = data[first]
    closest_sq = np.sum((data - centers[0]) ** 2, axis=1)
    for center_index in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 1e-12:
            centers[center_index] = data[int(rng.integers(0, num_rows))]
            continue
        probs = closest_sq / total
        pick = int(rng.choice(num_rows, p=probs))
        centers[center_index] = data[pick]
        dist_sq = np.sum((data - centers[center_index]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)
    return centers


def _run_kmeans(data: np.ndarray, k: int, random_state: int = 0, n_init: int = 8, max_iter: int = 100) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D kmeans input, got shape {data.shape}")
    num_rows = data.shape[0]
    if num_rows == 0:
        return np.zeros((0,), dtype=np.int32)
    if k <= 1:
        return np.zeros((num_rows,), dtype=np.int32)
    if k >= num_rows:
        return np.arange(num_rows, dtype=np.int32)

    best_labels: Optional[np.ndarray] = None
    best_inertia = np.inf
    for init_index in range(max(1, int(n_init))):
        rng = np.random.default_rng(int(random_state) + init_index)
        centers = _init_kmeans_pp(data, k=k, rng=rng)
        labels = np.zeros((num_rows,), dtype=np.int32)
        for _iteration in range(max(1, int(max_iter))):
            dist_sq = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dist_sq, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for center_index in range(k):
                mask = labels == center_index
                if np.any(mask):
                    centers[center_index] = np.mean(data[mask], axis=0)
                else:
                    farthest = int(np.argmax(np.min(dist_sq, axis=1)))
                    centers[center_index] = data[farthest]
        inertia = float(np.sum((data - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    if best_labels is None:
        return np.zeros((num_rows,), dtype=np.int32)
    return best_labels.astype(np.int32)


def _relabel_clusters(labels: np.ndarray, object_ids: Sequence[int]) -> np.ndarray:
    if labels.size == 0:
        return labels.astype(np.int32)
    members: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        members[int(label)].append(int(object_ids[index]))
    ordered_labels = sorted(members.keys(), key=lambda label: (min(members[label]), label))
    remap = {old_label: new_index for new_index, old_label in enumerate(ordered_labels)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int32)


def _cluster_boundary_after_indices(labels: Sequence[int], order: Sequence[int]) -> List[int]:
    ordered_labels = [int(labels[int(index)]) for index in order]
    boundaries: List[int] = []
    for index in range(len(ordered_labels) - 1):
        if ordered_labels[index] != ordered_labels[index + 1]:
            boundaries.append(index)
    return boundaries


def reorder_similarity_matrix_by_cluster(
    similarity_matrix: np.ndarray,
    labels: Sequence[int],
    *,
    object_ids: Optional[Sequence[int]] = None,
    noise_last: bool = False,
) -> Dict[str, Any]:
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(f"Expected square similarity matrix, got shape {similarity_matrix.shape}")
    size = similarity_matrix.shape[0]
    if len(labels) != size:
        raise ValueError(f"labels length mismatch: len(labels)={len(labels)} size={size}")
    ids = list(object_ids) if object_ids is not None else list(range(size))
    if len(ids) != size:
        raise ValueError(f"object_ids length mismatch: len(ids)={len(ids)} size={size}")
    members_by_cluster: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members_by_cluster[int(label)].append(index)
    ordered_clusters = sorted(
        members_by_cluster.items(),
        key=lambda item: (
            1 if noise_last and int(item[0]) == -1 else 0,
            -len(item[1]),
            min(int(ids[idx]) for idx in item[1]),
            int(item[0]),
        ),
    )
    order: List[int] = []
    for _cluster_id, member_indices in ordered_clusters:
        order.extend(sorted(member_indices, key=lambda idx: (int(ids[idx]), idx)))
    reordered = (
        np.asarray(similarity_matrix, dtype=np.float32)[np.ix_(np.asarray(order, dtype=np.int64), np.asarray(order, dtype=np.int64))]
        if size
        else np.zeros((0, 0), dtype=np.float32)
    )
    return {
        "order": order,
        "reordered_matrix": reordered.astype(np.float32),
        "boundary_after_indices": _cluster_boundary_after_indices(labels, order),
        "ordered_cluster_labels": [int(labels[index]) for index in order],
        "ordered_object_ids": [int(ids[index]) for index in order],
    }


def deduplicate_multi_view_embeddings(
    embeddings: np.ndarray,
    view_ids: Sequence[Any],
    *,
    object_ids: Optional[Sequence[int]] = None,
    cluster_count_mode: str = DEFAULT_CLUSTER_COUNT_MODE,
    n_clusters: Optional[int] = None,
    same_view_penalty: float = DEFAULT_SAME_VIEW_PENALTY,
    similarity_threshold: Optional[float] = DEFAULT_MULTI_VIEW_SIMILARITY_THRESHOLD,
    random_state: int = 0,
) -> Dict[str, Any]:
    normalized_embeddings = _l2_normalize_rows(np.asarray(embeddings, dtype=np.float32))
    similarity_matrix = compute_text_similarity(normalized_embeddings)
    affinity_matrix = build_multiview_affinity_matrix(
        similarity_matrix,
        view_ids,
        same_view_penalty=same_view_penalty,
        similarity_threshold=similarity_threshold,
    )
    spectral_result = run_spectral_clustering(
        affinity_matrix,
        object_ids=object_ids,
        cluster_count_mode=cluster_count_mode,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    reordered = reorder_similarity_matrix_by_cluster(
        similarity_matrix,
        spectral_result["labels"].tolist(),
        object_ids=object_ids,
    )
    return {
        "normalized_embeddings": normalized_embeddings,
        "similarity_matrix": similarity_matrix,
        "affinity_matrix": affinity_matrix,
        "cluster_labels": np.asarray(spectral_result["labels"], dtype=np.int32),
        "reordered_similarity_matrix": reordered["reordered_matrix"],
        "order": reordered["order"],
        "boundary_after_indices": reordered["boundary_after_indices"],
        "ordered_cluster_labels": reordered["ordered_cluster_labels"],
        "ordered_object_ids": reordered["ordered_object_ids"],
        "laplacian": spectral_result.get("laplacian"),
        "eigenvalues": spectral_result.get("eigenvalues"),
        "spectral_embedding": spectral_result.get("spectral_embedding"),
        "n_clusters": int(spectral_result["n_clusters"]),
        "cluster_count_mode": spectral_result["cluster_count_mode"],
        "backend": spectral_result["backend"],
        "fallback_reason": spectral_result["fallback_reason"],
    }


def build_knn_affinity_matrix(
    similarity_matrix: np.ndarray,
    *,
    k: int = DEFAULT_REFINED_GRAPH_KNN_K,
) -> np.ndarray:
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(f"Expected square similarity matrix, got shape {similarity_matrix.shape}")
    size = similarity_matrix.shape[0]
    if size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if size == 1:
        return np.eye(1, dtype=np.float32)
    keep = max(1, min(int(k), size - 1))
    base = np.asarray(similarity_matrix, dtype=np.float32)
    knn = np.zeros_like(base, dtype=np.float32)
    for row_index in range(size):
        row = base[row_index].copy()
        row[row_index] = -np.inf
        order = np.argsort(-row)
        for col_index in order[:keep]:
            if not np.isfinite(row[col_index]):
                continue
            knn[row_index, col_index] = base[row_index, col_index]
    knn = 0.5 * (knn + knn.T)
    knn = np.clip(knn, 0.0, 1.0)
    np.fill_diagonal(knn, 0.0)
    return knn.astype(np.float32)


def _pairwise_euclidean_distances(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {values.shape}")
    if values.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0)).astype(np.float32)


def _estimate_dbscan_eps(spectral_embedding: np.ndarray, min_samples: int) -> float:
    values = np.asarray(spectral_embedding, dtype=np.float32)
    size = values.shape[0]
    if size <= 1:
        return 0.0
    min_pts = max(1, min(int(min_samples), size - 1))
    distances = _pairwise_euclidean_distances(values)
    np.fill_diagonal(distances, np.inf)
    sorted_distances = np.sort(distances, axis=1)
    kth = sorted_distances[:, min_pts - 1]
    finite = kth[np.isfinite(kth)]
    if finite.size == 0:
        return 0.0
    return float(np.clip(np.median(finite), 1e-6, np.max(finite)))


def _run_dbscan(data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    num_rows = values.shape[0]
    if num_rows == 0:
        return np.zeros((0,), dtype=np.int32)
    if num_rows == 1:
        return np.zeros((1,), dtype=np.int32)

    radius = max(float(eps), 0.0)
    min_pts = max(1, int(min_samples))
    distances = _pairwise_euclidean_distances(values)
    neighborhoods = [np.where(distances[row_index] <= radius)[0].tolist() for row_index in range(num_rows)]
    is_core = [len(neighbors) >= min_pts for neighbors in neighborhoods]
    labels = np.full((num_rows,), -1, dtype=np.int32)
    visited = np.zeros((num_rows,), dtype=bool)
    cluster_id = 0

    for start_index in range(num_rows):
        if visited[start_index]:
            continue
        visited[start_index] = True
        if not is_core[start_index]:
            continue
        labels[start_index] = cluster_id
        queue = list(neighborhoods[start_index])
        head = 0
        while head < len(queue):
            neighbor_index = int(queue[head])
            head += 1
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                if is_core[neighbor_index]:
                    for expanded in neighborhoods[neighbor_index]:
                        if expanded not in queue:
                            queue.append(int(expanded))
            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id
        cluster_id += 1
    return labels.astype(np.int32)


def run_refined_graph_visualization_pipeline(
    rows: Sequence[Mapping[str, Any]],
    *,
    similarity_matrix: Optional[np.ndarray] = None,
    object_ids: Optional[Sequence[int]] = None,
    knn_k: int = DEFAULT_REFINED_GRAPH_KNN_K,
    spectral_dim: Optional[int] = None,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = DEFAULT_REFINED_GRAPH_DBSCAN_MIN_SAMPLES,
) -> Dict[str, Any]:
    prepared_rows = [dict(row) for row in rows]
    if not prepared_rows:
        return {
            "similarity_matrix": np.zeros((0, 0), dtype=np.float32),
            "visual_similarity_matrix": np.zeros((0, 0), dtype=np.float32),
            "knn_affinity_matrix": np.zeros((0, 0), dtype=np.float32),
            "laplacian": np.zeros((0, 0), dtype=np.float32),
            "spectral_embedding": np.zeros((0, 0), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
            "order": [],
            "boundary_after_indices": [],
            "ordered_cluster_labels": [],
            "ordered_object_ids": [],
            "reordered_similarity_matrix": np.zeros((0, 0), dtype=np.float32),
            "reordered_visual_similarity_matrix": np.zeros((0, 0), dtype=np.float32),
            "knn_k": 0,
            "spectral_dim": 0,
            "dbscan_eps": 0.0,
            "dbscan_min_samples": int(dbscan_min_samples),
            "noise_count": 0,
            "num_clusters_excluding_noise": 0,
        }

    embeddings = np.vstack([np.asarray(row["embedding"], dtype=np.float32).reshape(-1) for row in prepared_rows])
    normalized_embeddings = _l2_normalize_rows(embeddings)
    similarity = (
        compute_text_similarity(normalized_embeddings)
        if similarity_matrix is None
        else np.asarray(similarity_matrix, dtype=np.float32)
    )
    visual_similarity = similarity.copy()
    np.fill_diagonal(visual_similarity, 0.0)

    size = similarity.shape[0]
    ids = list(object_ids) if object_ids is not None else [
        _safe_int(row.get("object_global_id"), index) for index, row in enumerate(prepared_rows)
    ]
    if len(ids) != size:
        raise ValueError(f"object_ids length mismatch: len(ids)={len(ids)} size={size}")

    resolved_k = max(1, min(int(knn_k), max(1, size - 1)))
    knn_affinity = build_knn_affinity_matrix(similarity, k=resolved_k)
    laplacian_matrix = build_normalized_laplacian(knn_affinity)

    resolved_dim = max(1, min(int(spectral_dim or DEFAULT_REFINED_GRAPH_SPECTRAL_DIM), size))
    eigvals, eigvecs = np.linalg.eigh(np.asarray(laplacian_matrix, dtype=np.float64))
    raw_embedding = np.asarray(eigvecs[:, :resolved_dim], dtype=np.float32)
    row_norms = np.linalg.norm(raw_embedding, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    spectral_embedding = raw_embedding / row_norms

    resolved_min_samples = max(1, min(int(dbscan_min_samples), size))
    resolved_eps = float(dbscan_eps) if dbscan_eps is not None else _estimate_dbscan_eps(
        spectral_embedding,
        min_samples=resolved_min_samples,
    )
    labels = _run_dbscan(
        spectral_embedding,
        eps=resolved_eps,
        min_samples=resolved_min_samples,
    )

    reordered = reorder_similarity_matrix_by_cluster(
        similarity,
        labels.tolist(),
        object_ids=ids,
        noise_last=True,
    )
    reordered_vis = visual_similarity[
        np.ix_(np.asarray(reordered["order"], dtype=np.int64), np.asarray(reordered["order"], dtype=np.int64))
    ]
    non_noise = sorted({int(label) for label in labels.tolist() if int(label) != -1})
    return {
        "similarity_matrix": similarity.astype(np.float32),
        "visual_similarity_matrix": visual_similarity.astype(np.float32),
        "knn_affinity_matrix": knn_affinity.astype(np.float32),
        "laplacian": laplacian_matrix.astype(np.float32),
        "eigenvalues": np.asarray(eigvals, dtype=np.float32),
        "spectral_embedding": spectral_embedding.astype(np.float32),
        "labels": labels.astype(np.int32),
        "order": reordered["order"],
        "boundary_after_indices": reordered["boundary_after_indices"],
        "ordered_cluster_labels": reordered["ordered_cluster_labels"],
        "ordered_object_ids": reordered["ordered_object_ids"],
        "reordered_similarity_matrix": reordered["reordered_matrix"].astype(np.float32),
        "reordered_visual_similarity_matrix": reordered_vis.astype(np.float32),
        "knn_k": int(resolved_k),
        "spectral_dim": int(resolved_dim),
        "dbscan_eps": float(resolved_eps),
        "dbscan_min_samples": int(resolved_min_samples),
        "noise_count": int(np.count_nonzero(labels == -1)),
        "num_clusters_excluding_noise": int(len(non_noise)),
    }


def run_spectral_clustering(
    affinity_matrix: np.ndarray,
    *,
    object_ids: Optional[Sequence[int]] = None,
    cluster_count_mode: str = DEFAULT_CLUSTER_COUNT_MODE,
    n_clusters: Optional[int] = None,
    random_state: int = 0,
) -> Dict[str, Any]:
    if affinity_matrix.ndim != 2 or affinity_matrix.shape[0] != affinity_matrix.shape[1]:
        raise ValueError(f"Expected square affinity matrix, got shape {affinity_matrix.shape}")
    size = affinity_matrix.shape[0]
    ids = list(object_ids) if object_ids is not None else list(range(size))
    if len(ids) != size:
        raise ValueError(f"object_ids length mismatch: len(ids)={len(ids)} size={size}")
    if size == 0:
        return {
            "labels": np.zeros((0,), dtype=np.int32),
            "n_clusters": 0,
            "cluster_count_mode": _normalize_cluster_count_mode(cluster_count_mode),
            "backend": "empty",
            "fallback_reason": "empty_group",
        }
    if size == 1:
        return {
            "labels": np.zeros((1,), dtype=np.int32),
            "n_clusters": 1,
            "cluster_count_mode": _normalize_cluster_count_mode(cluster_count_mode),
            "backend": "singleton",
            "fallback_reason": None,
        }
    if _positive_off_diagonal_count(affinity_matrix) == 0:
        singleton_labels = np.arange(size, dtype=np.int32)
        return {
            "labels": _relabel_clusters(singleton_labels, ids),
            "n_clusters": size,
            "cluster_count_mode": _normalize_cluster_count_mode(cluster_count_mode),
            "backend": "singleton_fallback",
            "fallback_reason": "zero_affinity_graph",
        }

    chosen_k = _choose_cluster_count(
        affinity_matrix,
        cluster_count_mode=cluster_count_mode,
        n_clusters=n_clusters,
    )
    if chosen_k <= 1:
        zero_labels = np.zeros((size,), dtype=np.int32)
        return {
            "labels": zero_labels,
            "n_clusters": 1,
            "cluster_count_mode": _normalize_cluster_count_mode(cluster_count_mode),
            "backend": "single_cluster",
            "fallback_reason": None,
        }
    if chosen_k >= size:
        singleton_labels = np.arange(size, dtype=np.int32)
        return {
            "labels": _relabel_clusters(singleton_labels, ids),
            "n_clusters": size,
            "cluster_count_mode": _normalize_cluster_count_mode(cluster_count_mode),
            "backend": "singleton_fallback",
            "fallback_reason": "cluster_count_equals_group_size",
        }

    laplacian_matrix = build_normalized_laplacian(affinity_matrix)
    eigvals, eigvecs = np.linalg.eigh(np.asarray(laplacian_matrix, dtype=np.float64))
    raw_embedding = np.asarray(eigvecs[:, : int(chosen_k)], dtype=np.float32)
    row_norms = np.linalg.norm(raw_embedding, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    embedding = raw_embedding / row_norms
    labels = _run_kmeans(embedding, k=int(chosen_k), random_state=int(random_state))

    labels = _relabel_clusters(np.asarray(labels, dtype=np.int32), ids)
    return {
        "labels": labels,
        "n_clusters": int(len(set(labels.tolist()))),
        "cluster_count_mode": _normalize_cluster_count_mode(cluster_count_mode),
        "backend": "explicit_numpy",
        "fallback_reason": None,
        "laplacian": laplacian_matrix,
        "eigenvalues": np.asarray(eigvals, dtype=np.float32),
        "spectral_embedding": embedding.astype(np.float32),
    }


def _cluster_order(labels: Sequence[int], object_ids: Sequence[int]) -> List[int]:
    return sorted(range(len(object_ids)), key=lambda index: (int(labels[index]), int(object_ids[index]), index))


def _cluster_color(cluster_id: int) -> Tuple[int, int, int]:
    palette = [
        (46, 125, 50),
        (33, 150, 243),
        (255, 152, 0),
        (156, 39, 176),
        (229, 57, 53),
        (0, 121, 107),
        (121, 85, 72),
        (63, 81, 181),
        (124, 179, 66),
        (255, 87, 34),
    ]
    return palette[int(cluster_id) % len(palette)]


def _format_heatmap_axis_label(row: Mapping[str, Any]) -> str:
    object_id = _safe_int(row.get("object_global_id"), -1)
    view_id = _safe_text(row.get("view_id") or row.get("synthetic_view_id") or row.get("entry_id"))
    label = _safe_text(row.get("label"), "unknown")
    parts: List[str] = []
    if object_id >= 0:
        parts.append(f"obj{object_id}")
    if view_id:
        parts.append(view_id)
    if label:
        parts.append(label)
    return "|".join(parts) or "unknown"


def _truncate_heatmap_label(label: str, max_chars: int = 30) -> str:
    text = _safe_text(label, default="unknown")
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _make_rotated_text_image(
    text: str,
    *,
    font_scale: float,
    thickness: int = 1,
    angle_deg: int = -90,
    color: Tuple[int, int, int] = (25, 25, 25),
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    height = max(1, text_h + baseline + pad * 2)
    width = max(1, text_w + pad * 2)
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(
        bgr,
        text,
        (pad, text_h + pad),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        alpha,
        text,
        (pad, text_h + pad),
        font,
        font_scale,
        255,
        thickness,
        cv2.LINE_AA,
    )
    rgba = np.dstack([bgr, alpha])
    if angle_deg == -90:
        return cv2.rotate(rgba, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle_deg == 90:
        return cv2.rotate(rgba, cv2.ROTATE_90_CLOCKWISE)
    if angle_deg == 180:
        return cv2.rotate(rgba, cv2.ROTATE_180)
    return rgba


def _alpha_blit(canvas: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    if overlay.ndim != 3 or overlay.shape[2] != 4:
        raise ValueError(f"Expected RGBA overlay, got shape {overlay.shape}")
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(canvas.shape[1], x0 + overlay.shape[1])
    y1 = min(canvas.shape[0], y0 + overlay.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    clipped = overlay[: y1 - y0, : x1 - x0]
    alpha = clipped[:, :, 3:4].astype(np.float32) / 255.0
    dst = canvas[y0:y1, x0:x1].astype(np.float32)
    src = clipped[:, :, :3].astype(np.float32)
    blended = alpha * src + (1.0 - alpha) * dst
    canvas[y0:y1, x0:x1] = np.asarray(blended, dtype=np.uint8)


def _draw_label_box(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    *,
    bg_color: Tuple[int, int, int],
    fg_color: Tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(org[0])
    y = int(org[1])
    pad_x = 7
    pad_y = 5
    box_x1 = max(0, x)
    box_y1 = max(0, y - th - baseline - pad_y * 2)
    box_x2 = min(image.shape[1], x + tw + pad_x * 2)
    box_y2 = min(image.shape[0], y)
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), 1)
    text_x = box_x1 + pad_x
    text_y = min(image.shape[0] - 2, box_y2 - pad_y - baseline)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, fg_color, thickness, cv2.LINE_AA)


def _resolve_image_path(file_name: str, candidate_roots: Sequence[Path]) -> Optional[Path]:
    raw = _safe_text(file_name)
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None
    for root in candidate_roots:
        candidate = Path(root) / raw
        if candidate.exists():
            return candidate
    return None


def _valid_bbox_xywh_norm(row: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = row.get("bbox_xywh_norm")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(bbox[i]) for i in range(4)]
    except Exception:
        return None
    if max(w, h) <= 1e-6:
        return None
    return (
        min(max(x, 0.0), 1.0),
        min(max(y, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0),
    )


def _fallback_anchor_point(
    row: Mapping[str, Any],
    image_w: int,
    image_h: int,
    bucket_rank: int,
) -> Tuple[int, int]:
    laterality = _safe_text(row.get("laterality"), "center").lower()
    verticality = _safe_text(row.get("verticality"), "middle").lower()
    x_ratio = {"left": 0.18, "center": 0.50, "right": 0.82}.get(laterality, 0.50)
    y_ratio = {"high": 0.18, "middle": 0.50, "low": 0.82}.get(verticality, 0.50)
    base_x = int(round(image_w * x_ratio))
    base_y = int(round(image_h * y_ratio))
    row_offset = bucket_rank % 3
    col_offset = bucket_rank // 3
    anchor_x = int(np.clip(base_x + col_offset * 42, 16, max(16, image_w - 16)))
    anchor_y = int(np.clip(base_y + row_offset * 26, 16, max(16, image_h - 16)))
    return anchor_x, anchor_y


def _overlay_view_annotations(
    image: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    *,
    title: str,
) -> np.ndarray:
    canvas = image.copy()
    image_h, image_w = canvas.shape[:2]
    bucket_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    items = list(zip(rows, labels))
    items.sort(
        key=lambda item: (
            _safe_text(item[0].get("verticality")),
            _safe_text(item[0].get("laterality")),
            _safe_int(item[0].get("object_global_id"), 10**9),
        )
    )

    for row, cluster_id in items:
        cluster_color = _cluster_color(int(cluster_id))
        bbox = _valid_bbox_xywh_norm(row)
        if bbox is not None:
            x, y, w, h = bbox
            x1 = int(round(x * image_w))
            y1 = int(round(y * image_h))
            x2 = int(round((x + w) * image_w))
            y2 = int(round((y + h) * image_h))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), cluster_color, 2)
            anchor_x = x1
            anchor_y = max(28, y1)
        else:
            bucket_key = (
                _safe_text(row.get("laterality"), "center").lower(),
                _safe_text(row.get("verticality"), "middle").lower(),
            )
            rank = bucket_counts[bucket_key]
            bucket_counts[bucket_key] += 1
            anchor_x, anchor_y = _fallback_anchor_point(row, image_w, image_h, rank)
            cv2.circle(canvas, (anchor_x, anchor_y), 6, cluster_color, -1)
            cv2.circle(canvas, (anchor_x, anchor_y), 8, (25, 25, 25), 1)

        text = f"obj{_safe_int(row.get('object_global_id'), -1)} { _safe_text(row.get('label'), 'unknown') } c{int(cluster_id)}"
        fg_color = (255, 255, 255) if sum(cluster_color) < 360 else (20, 20, 20)
        _draw_label_box(
            canvas,
            text,
            (anchor_x + 8, anchor_y),
            bg_color=cluster_color,
            fg_color=fg_color,
            font_scale=0.50,
            thickness=1,
        )

    header = f"{title} | {len(rows)} objects"
    _draw_label_box(
        canvas,
        header,
        (14, 34),
        bg_color=(245, 245, 245),
        fg_color=(20, 20, 20),
        font_scale=0.65,
        thickness=2,
    )
    return canvas


def _build_view_annotation_grid(
    annotated_paths: Sequence[Path],
    output_path: Path,
) -> Optional[Path]:
    if not annotated_paths:
        return None
    images: List[np.ndarray] = []
    valid_paths: List[Path] = []
    target_w = 640
    target_h = 360
    for path in annotated_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        images.append(resized)
        valid_paths.append(path)
    if not images:
        return None
    cols = 2 if len(images) > 1 else 1
    rows = int(math.ceil(len(images) / float(cols)))
    grid = np.full((rows * target_h, cols * target_w, 3), 255, dtype=np.uint8)
    for index, image in enumerate(images):
        row_idx = index // cols
        col_idx = index % cols
        y0 = row_idx * target_h
        x0 = col_idx * target_w
        grid[y0:y0 + target_h, x0:x0 + target_w] = image
        cv2.rectangle(grid, (x0, y0), (x0 + target_w - 1, y0 + target_h - 1), (35, 35, 35), 2)
        label = valid_paths[index].stem
        cv2.putText(grid, label, (x0 + 14, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), grid)
    if not ok:
        raise RuntimeError(f"Failed to save view annotation grid: {output_path}")
    return output_path


def save_view_annotation_results(
    group_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    image_roots: Sequence[Path],
) -> Dict[str, Any]:
    annotations_dir = group_dir / "view_annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    by_view: Dict[str, List[Tuple[Mapping[str, Any], int]]] = defaultdict(list)
    for row, cluster_id in zip(rows, labels):
        view_id = _safe_text(row.get("view_id") or row.get("synthetic_view_id") or row.get("entry_id"))
        if not view_id:
            continue
        by_view[view_id].append((row, int(cluster_id)))

    manifest: List[Dict[str, Any]] = []
    saved_images: List[Path] = []
    candidate_roots = [Path(root) for root in image_roots if root is not None]
    if group_dir not in candidate_roots:
        candidate_roots.insert(0, group_dir)

    for view_id, items in sorted(by_view.items(), key=lambda item: item[0]):
        first_row = items[0][0]
        source_path = _resolve_image_path(_safe_text(first_row.get("file_name")), candidate_roots)
        if source_path is None:
            manifest.append(
                {
                    "view_id": view_id,
                    "source_file_name": _safe_text(first_row.get("file_name")),
                    "status": "missing_image",
                    "objects": [
                        {
                            "object_global_id": _safe_int(row.get("object_global_id"), -1),
                            "label": _safe_text(row.get("label"), "unknown"),
                            "cluster_id": int(cluster_id),
                        }
                        for row, cluster_id in items
                    ],
                }
            )
            continue
        image = cv2.imread(str(source_path))
        if image is None:
            manifest.append(
                {
                    "view_id": view_id,
                    "source_image_path": str(source_path),
                    "status": "image_read_failed",
                }
            )
            continue
        overlay = _overlay_view_annotations(
            image,
            [row for row, _cluster_id in items],
            [cluster_id for _row, cluster_id in items],
            title=view_id,
        )
        output_path = annotations_dir / f"{view_id}_annotated.jpg"
        ok = cv2.imwrite(str(output_path), overlay)
        if not ok:
            raise RuntimeError(f"Failed to save annotated view image: {output_path}")
        saved_images.append(output_path)
        manifest.append(
            {
                "view_id": view_id,
                "source_image_path": str(source_path),
                "annotated_image_path": str(output_path),
                "source_file_name": _safe_text(first_row.get("file_name")),
                "num_objects": len(items),
                "objects": [
                    {
                        "object_global_id": _safe_int(row.get("object_global_id"), -1),
                        "label": _safe_text(row.get("label"), "unknown"),
                        "cluster_id": int(cluster_id),
                        "laterality": _safe_text(row.get("laterality")),
                        "verticality": _safe_text(row.get("verticality")),
                    }
                    for row, cluster_id in items
                ],
                "status": "ok",
            }
        )

    manifest_path = annotations_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "views": manifest,
        },
    )
    grid_path = _build_view_annotation_grid(saved_images, annotations_dir / "annotated_views_grid.jpg")
    return {
        "annotations_dir": annotations_dir,
        "manifest_path": manifest_path,
        "grid_path": grid_path,
        "annotated_images": saved_images,
    }


def _compute_heatmap_display_values(
    matrix: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    display_transform: str = "linear",
    display_log_gain: float = 24.0,
    suppress_diagonal_display: bool = False,
    diagonal_brightness_budget_ratio: Optional[float] = None,
    diagonal_display_cap: float = 0.82,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected square heatmap matrix, got shape {values.shape}")
    if values.size == 0:
        return values.astype(np.float32)

    transform = str(display_transform or "linear").strip().lower()
    if transform in {"linear", "log"}:
        lower = float(np.min(values) if vmin is None else vmin)
        upper = float(np.max(values) if vmax is None else vmax)
        if upper - lower < 1e-12:
            normalized = np.zeros_like(values, dtype=np.float32)
        else:
            normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
        if transform == "linear":
            display_values = normalized
        else:
            gain = max(float(display_log_gain), 1e-6)
            denom = float(np.log1p(gain))
            if denom <= 1e-12:
                display_values = normalized
            else:
                # Apply a log stretch to the distance from the top end so high-similarity
                # blocks separate more clearly without changing the underlying values.
                tail_distance = np.clip(1.0 - normalized, 0.0, 1.0)
                display_values = np.clip(1.0 - np.log1p(gain * tail_distance) / denom, 0.0, 1.0)
    elif transform == "offdiag_extreme":
        size = values.shape[0]
        offdiag_mask = ~np.eye(size, dtype=bool)
        offdiag = values[offdiag_mask]
        if offdiag.size == 0:
            display_values = np.zeros_like(values, dtype=np.float32)
        else:
            low = float(np.percentile(offdiag, 14.0))
            high = float(np.percentile(offdiag, 88.0))
            if high - low < 1e-12:
                normalized = np.clip(values, 0.0, 1.0)
            else:
                normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
            boosted = np.power(normalized, 1.4)
            levels = np.asarray([0.0, 0.06, 0.16, 0.32, 0.52, 0.74, 1.0], dtype=np.float32)
            bins = np.asarray([0.03, 0.10, 0.24, 0.42, 0.64, 0.87], dtype=np.float32)
            level_indices = np.digitize(boosted, bins, right=False)
            display_values = levels[level_indices]
            display_values[boosted < 0.10] = 0.0
    else:
        raise ValueError(f"Unsupported heatmap display_transform={display_transform!r}")

    display_values = np.asarray(display_values, dtype=np.float32)
    if display_values.size > 0:
        if suppress_diagonal_display:
            np.fill_diagonal(display_values, 0.0)
        elif diagonal_brightness_budget_ratio is not None:
            size = display_values.shape[0]
            offdiag_mask = ~np.eye(size, dtype=bool)
            offdiag = display_values[offdiag_mask]
            if offdiag.size > 0:
                budget_ratio = max(0.0, float(diagonal_brightness_budget_ratio))
                budget_based_level = budget_ratio * float(np.mean(offdiag)) * float(max(1, size - 1))
                quantile_based_level = float(np.percentile(offdiag, 90.0))
                diag_level = min(
                    max(float(diagonal_display_cap), 0.0),
                    max(budget_based_level, quantile_based_level),
                )
                np.fill_diagonal(display_values, diag_level)
            else:
                np.fill_diagonal(display_values, min(max(float(diagonal_display_cap), 0.0), 1.0))
    return np.clip(display_values, 0.0, 1.0).astype(np.float32)


def plot_similarity_heatmap(
    matrix: np.ndarray,
    output_path: Path,
    *,
    title: str,
    order: Optional[Sequence[int]] = None,
    boundary_after_indices: Optional[Sequence[int]] = None,
    axis_labels: Optional[Sequence[str]] = None,
    annotate_values: Optional[bool] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    display_transform: str = "linear",
    display_log_gain: float = 24.0,
    suppress_diagonal_display: bool = False,
    diagonal_brightness_budget_ratio: Optional[float] = None,
    diagonal_display_cap: float = 0.82,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = np.asarray(matrix, dtype=np.float32)
    ordered_labels = list(axis_labels or [])
    if order is not None and ordered.size > 0:
        idx = np.asarray(list(order), dtype=np.int64)
        ordered = ordered[np.ix_(idx, idx)]
        if ordered_labels:
            ordered_labels = [ordered_labels[int(index)] for index in idx.tolist()]
    display_labels = [_truncate_heatmap_label(label) for label in ordered_labels]
    count = int(ordered.shape[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.40 if count >= 16 else 0.45
    thickness = 1
    label_widths = []
    for label in display_labels:
        (text_w, _text_h), _baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_widths.append(int(text_w))

    left_margin = max(24, max(label_widths, default=0) + 22)
    top_margin = max(50, max(label_widths, default=0) + 26) if display_labels else 50
    right_margin = 20
    bottom_margin = 24
    size = max(280, count * 32) if count > 0 else 280
    canvas_w = max(520, left_margin + size + right_margin)
    canvas_h = max(520, top_margin + size + bottom_margin)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        str(title)[:80],
        (18, 28),
        font,
        0.65,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    if ordered.size == 0:
        cv2.putText(
            canvas,
            "No objects",
            (180, 270),
            font,
            0.8,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )
        ok = cv2.imwrite(str(output_path), canvas)
        if not ok:
            raise RuntimeError(f"Failed to save heatmap: {output_path}")
        return

    display_values = _compute_heatmap_display_values(
        ordered,
        vmin=vmin,
        vmax=vmax,
        display_transform=display_transform,
        display_log_gain=display_log_gain,
        suppress_diagonal_display=suppress_diagonal_display,
        diagonal_brightness_budget_ratio=diagonal_brightness_budget_ratio,
        diagonal_display_cap=diagonal_display_cap,
    )

    heat_u8 = np.asarray(display_values * 255.0, dtype=np.uint8)
    heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_NEAREST)
    x0 = left_margin
    y0 = top_margin
    x1 = min(x0 + size, canvas.shape[1] - right_margin)
    y1 = min(y0 + size, canvas.shape[0] - bottom_margin)
    heatmap = cv2.resize(heatmap, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    canvas[y0:y1, x0:x1] = heatmap
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 40), 1)

    if count > 0 and boundary_after_indices:
        cell_h = float(y1 - y0) / float(count)
        cell_w = float(x1 - x0) / float(count)
        boundary_color = (250, 250, 250)
        shadow_color = (35, 35, 35)
        valid_boundaries = sorted(
            {int(idx) for idx in boundary_after_indices if 0 <= int(idx) < count - 1}
        )
        for boundary_index in valid_boundaries:
            x_boundary = int(round(x0 + (boundary_index + 1) * cell_w))
            y_boundary = int(round(y0 + (boundary_index + 1) * cell_h))
            cv2.line(canvas, (x_boundary, y0), (x_boundary, y1), shadow_color, 3, cv2.LINE_AA)
            cv2.line(canvas, (x_boundary, y0), (x_boundary, y1), boundary_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (x0, y_boundary), (x1, y_boundary), shadow_color, 3, cv2.LINE_AA)
            cv2.line(canvas, (x0, y_boundary), (x1, y_boundary), boundary_color, 1, cv2.LINE_AA)

    if display_labels:
        cell_h = float(y1 - y0) / float(len(display_labels))
        cell_w = float(x1 - x0) / float(len(display_labels))
        for row_index, label in enumerate(display_labels):
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
        for col_index, label in enumerate(display_labels):
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

    should_annotate = bool(annotate_values) if annotate_values is not None else (count <= 16)
    if should_annotate and count > 0:
        cell_h = float(y1 - y0) / float(count)
        cell_w = float(x1 - x0) / float(count)
        value_font_scale = min(0.42, max(0.26, min(cell_h, cell_w) / 85.0))
        for row_index in range(count):
            for col_index in range(count):
                value = float(ordered[row_index, col_index])
                value_text = f"{value:.2f}"
                center_x = int(round(x0 + (col_index + 0.5) * cell_w))
                center_y = int(round(y0 + (row_index + 0.5) * cell_h))
                (text_w, text_h), baseline = cv2.getTextSize(
                    value_text,
                    font,
                    value_font_scale,
                    1,
                )
                display_value = float(display_values[row_index, col_index])
                text_color = (245, 245, 245) if display_value < 0.55 else (20, 20, 20)
                text_x = int(round(center_x - text_w / 2.0))
                text_y = int(round(center_y + text_h / 2.0 - baseline))
                cv2.putText(
                    canvas,
                    value_text,
                    (text_x, text_y),
                    font,
                    value_font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save heatmap: {output_path}")


def _representative_member_indices(
    member_indices: Sequence[int],
    similarity_matrix: np.ndarray,
) -> List[int]:
    if not member_indices:
        return []
    if len(member_indices) == 1:
        return [int(member_indices[0])]
    scores = []
    for index in member_indices:
        score = float(np.mean(similarity_matrix[index, member_indices]))
        scores.append((score, int(index)))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [index for _score, index in scores]


def summarize_clusters(
    rows: Sequence[Mapping[str, Any]],
    similarity_matrix: np.ndarray,
    labels: Sequence[int],
    *,
    group_id: str,
    group_type: str,
) -> Dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int32)
    grouped_indices: Dict[int, List[int]] = defaultdict(list)
    for index, cluster_id in enumerate(label_array.tolist()):
        grouped_indices[int(cluster_id)].append(index)

    clusters: List[Dict[str, Any]] = []
    for cluster_id in sorted(grouped_indices):
        member_indices = grouped_indices[cluster_id]
        members = [dict(rows[index]) for index in member_indices]
        member_object_ids = [_safe_int(row.get("object_global_id"), -1) for row in members]
        member_view_ids = [str(row.get("view_id") or row.get("synthetic_view_id") or row.get("entry_id") or "") for row in members]
        member_labels = [_safe_text(row.get("label"), "unknown") for row in members]
        member_descriptions = [_safe_text(row.get("description")) for row in members]
        member_long_descriptions = [_safe_text(row.get("long_form_open_description")) for row in members]
        view_counts = Counter(member_view_ids)
        offending_views = sorted(view_id for view_id, count in view_counts.items() if count > 1 and view_id)
        representative_index = _representative_member_indices(member_indices, similarity_matrix)[0]
        representative_row = dict(rows[representative_index])
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "candidate_instance_id": f"{group_type}:{group_id}:cluster_{int(cluster_id):03d}",
                "group_id": str(group_id),
                "group_type": str(group_type),
                "num_members": len(member_indices),
                "member_object_ids": member_object_ids,
                "member_observation_ids": [str(row.get("observation_id") or "") for row in members],
                "member_view_ids": member_view_ids,
                "member_labels": member_labels,
                "member_descriptions": member_descriptions,
                "member_long_descriptions": member_long_descriptions,
                "label_histogram": dict(sorted(Counter(member_labels).items())),
                "representative_object_id": _safe_int(representative_row.get("object_global_id"), -1),
                "representative_label": _safe_text(representative_row.get("label"), "unknown"),
                "representative_description": _safe_text(
                    representative_row.get("long_form_open_description")
                    or representative_row.get("description")
                ),
                "same_view_collision": bool(offending_views),
                "offending_view_ids": offending_views,
            }
        )

    return {
        "group_id": str(group_id),
        "group_type": str(group_type),
        "n_objects": len(rows),
        "n_clusters": len(clusters),
        "clusters": clusters,
    }


def _serialize_object_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    serialized = dict(row)
    embedding = serialized.pop("embedding", None)
    if embedding is not None:
        serialized["embedding_available"] = True
        serialized["embedding_dim"] = int(np.asarray(embedding).reshape(-1).shape[0])
    else:
        serialized["embedding_available"] = False
        serialized["embedding_dim"] = 0
    return _to_serializable(serialized)


def _cluster_labels_payload(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    *,
    group_id: str,
    group_type: str,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for row, cluster_id in zip(rows, labels):
        items.append(
            {
                "group_id": str(group_id),
                "group_type": str(group_type),
                "cluster_id": int(cluster_id),
                "object_global_id": _safe_int(row.get("object_global_id"), -1),
                "observation_id": str(row.get("observation_id") or ""),
                "view_id": str(row.get("view_id") or row.get("synthetic_view_id") or ""),
                "entry_id": _safe_int(row.get("entry_id"), -1),
                "label": _safe_text(row.get("label"), "unknown"),
                "description": _safe_text(row.get("description")),
            }
        )
    return {
        "group_id": str(group_id),
        "group_type": str(group_type),
        "labels": items,
    }


def _cluster_summary_markdown(cluster_summary: Mapping[str, Any]) -> str:
    lines = [
        f"# Cluster Summary: {cluster_summary.get('group_id')}",
        "",
        f"- group_type: {cluster_summary.get('group_type')}",
        f"- n_objects: {cluster_summary.get('n_objects')}",
        f"- n_clusters: {cluster_summary.get('n_clusters')}",
        "",
    ]
    for cluster in cluster_summary.get("clusters", []):
        lines.append(f"## Cluster {cluster['cluster_id']}")
        lines.append(f"- candidate_instance_id: {cluster['candidate_instance_id']}")
        lines.append(f"- num_members: {cluster['num_members']}")
        lines.append(f"- representative_label: {cluster['representative_label']}")
        lines.append(f"- representative_description: {cluster['representative_description']}")
        lines.append(f"- same_view_collision: {cluster['same_view_collision']}")
        if cluster.get("offending_view_ids"):
            lines.append(f"- offending_view_ids: {', '.join(cluster['offending_view_ids'])}")
        for object_id, view_id, label, description in zip(
            cluster.get("member_object_ids", []),
            cluster.get("member_view_ids", []),
            cluster.get("member_labels", []),
            cluster.get("member_descriptions", []),
        ):
            lines.append(
                f"- member: object_id={object_id} view_id={view_id} label={label} description={description}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_refined_graph_visualization_results(
    group_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    similarity_matrix: np.ndarray,
    object_ids: Sequence[int],
    heatmap_labels: Sequence[str],
) -> Dict[str, Path]:
    refined = run_refined_graph_visualization_pipeline(
        rows,
        similarity_matrix=similarity_matrix,
        object_ids=object_ids,
    )
    similarity_diag1_source = np.asarray(refined["similarity_matrix"], dtype=np.float32).copy()
    _write_json(
        group_dir / "refined_graph_cluster_labels.json",
        {
            "labels": refined["labels"].tolist(),
            "ordered_cluster_labels": list(refined["ordered_cluster_labels"]),
            "ordered_object_ids": list(refined["ordered_object_ids"]),
            "boundary_after_indices": list(refined["boundary_after_indices"]),
            "knn_k": int(refined["knn_k"]),
            "spectral_dim": int(refined["spectral_dim"]),
            "dbscan_eps": float(refined["dbscan_eps"]),
            "dbscan_min_samples": int(refined["dbscan_min_samples"]),
            "noise_count": int(refined["noise_count"]),
            "num_clusters_excluding_noise": int(refined["num_clusters_excluding_noise"]),
        },
    )
    np.save(
        group_dir / "refined_graph_knn_affinity_matrix.npy",
        np.asarray(refined["knn_affinity_matrix"], dtype=np.float32),
    )
    np.save(
        group_dir / "refined_graph_laplacian.npy",
        np.asarray(refined["laplacian"], dtype=np.float32),
    )
    np.save(
        group_dir / "refined_graph_spectral_embedding.npy",
        np.asarray(refined["spectral_embedding"], dtype=np.float32),
    )
    np.save(
        group_dir / "refined_graph_reordered_similarity_matrix.npy",
        np.asarray(refined["reordered_similarity_matrix"], dtype=np.float32),
    )
    np.save(
        group_dir / "refined_graph_reordered_visual_similarity_matrix.npy",
        np.asarray(refined["reordered_visual_similarity_matrix"], dtype=np.float32),
    )
    np.save(
        group_dir / "refined_graph_similarity_diag1_source_matrix.npy",
        similarity_diag1_source,
    )
    plot_similarity_heatmap(
        similarity_diag1_source,
        group_dir / "refined_graph_clustered_similarity_heatmap.png",
        title=REFINED_GRAPH_HEATMAP_TITLE,
        order=refined["order"],
        boundary_after_indices=refined["boundary_after_indices"],
        axis_labels=heatmap_labels,
        annotate_values=False,
        vmin=0.0,
        vmax=1.0,
        display_transform="linear",
        suppress_diagonal_display=False,
    )
    plot_similarity_heatmap(
        refined["visual_similarity_matrix"],
        group_dir / "refined_graph_clustered_similarity_heatmap_offdiag_only.png",
        title=REFINED_GRAPH_OFFDIAG_HEATMAP_TITLE,
        order=refined["order"],
        boundary_after_indices=refined["boundary_after_indices"],
        axis_labels=heatmap_labels,
        annotate_values=False,
        vmin=0.0,
        vmax=1.0,
        display_transform="offdiag_extreme",
        suppress_diagonal_display=True,
    )
    plot_similarity_heatmap(
        similarity_diag1_source,
        group_dir / "refined_graph_clustered_similarity_heatmap_diag1.png",
        title=REFINED_GRAPH_DIAG1_HEATMAP_TITLE,
        order=refined["order"],
        boundary_after_indices=refined["boundary_after_indices"],
        axis_labels=heatmap_labels,
        annotate_values=False,
        vmin=0.0,
        vmax=1.0,
        display_transform="offdiag_extreme",
        suppress_diagonal_display=False,
        diagonal_brightness_budget_ratio=0.18,
        diagonal_display_cap=0.82,
    )
    return {
        "refined_graph_cluster_labels": group_dir / "refined_graph_cluster_labels.json",
        "refined_graph_knn_affinity_matrix": group_dir / "refined_graph_knn_affinity_matrix.npy",
        "refined_graph_laplacian": group_dir / "refined_graph_laplacian.npy",
        "refined_graph_spectral_embedding": group_dir / "refined_graph_spectral_embedding.npy",
        "refined_graph_reordered_similarity_matrix": group_dir / "refined_graph_reordered_similarity_matrix.npy",
        "refined_graph_reordered_visual_similarity_matrix": group_dir / "refined_graph_reordered_visual_similarity_matrix.npy",
        "refined_graph_similarity_diag1_source_matrix": group_dir / "refined_graph_similarity_diag1_source_matrix.npy",
        "refined_graph_clustered_similarity_heatmap": group_dir / "refined_graph_clustered_similarity_heatmap.png",
        "refined_graph_clustered_similarity_heatmap_offdiag_only": group_dir / "refined_graph_clustered_similarity_heatmap_offdiag_only.png",
        "refined_graph_clustered_similarity_heatmap_diag1": group_dir / "refined_graph_clustered_similarity_heatmap_diag1.png",
    }


def save_cluster_results(
    group_dir: Path,
    *,
    group_id: str,
    group_type: str,
    rows: Sequence[Mapping[str, Any]],
    similarity_matrix: np.ndarray,
    affinity_matrix: np.ndarray,
    labels: Sequence[int],
    cluster_summary: Mapping[str, Any],
    group_metadata: Optional[Mapping[str, Any]] = None,
    image_roots: Optional[Sequence[Path]] = None,
) -> Dict[str, Path]:
    group_dir.mkdir(parents=True, exist_ok=True)
    object_ids = [_safe_int(row.get("object_global_id"), index) for index, row in enumerate(rows)]
    reordered_similarity = reorder_similarity_matrix_by_cluster(
        similarity_matrix,
        labels,
        object_ids=object_ids,
    )
    order = reordered_similarity["order"]
    boundary_after_indices = reordered_similarity["boundary_after_indices"]
    heatmap_labels = [_format_heatmap_axis_label(row) for row in rows]

    np.save(group_dir / "similarity_matrix.npy", np.asarray(similarity_matrix, dtype=np.float32))
    np.save(group_dir / "affinity_matrix.npy", np.asarray(affinity_matrix, dtype=np.float32))
    np.save(
        group_dir / "clustered_similarity_matrix.npy",
        np.asarray(reordered_similarity["reordered_matrix"], dtype=np.float32),
    )
    np.savetxt(
        group_dir / "clustered_similarity_matrix.csv",
        np.asarray(reordered_similarity["reordered_matrix"], dtype=np.float32),
        delimiter=",",
        fmt="%.6f",
    )

    _write_json(
        group_dir / "objects.json",
        {
            "group_id": str(group_id),
            "group_type": str(group_type),
            "objects": [_serialize_object_row(row) for row in rows],
        },
    )
    _write_json(
        group_dir / "cluster_labels.json",
        _cluster_labels_payload(rows, labels, group_id=group_id, group_type=group_type),
    )
    _write_json(group_dir / "cluster_summary.json", dict(cluster_summary))
    (group_dir / "cluster_summary.md").write_text(
        _cluster_summary_markdown(cluster_summary),
        encoding="utf-8",
    )
    _write_json(
        group_dir / "clustered_similarity_order.json",
        {
            "group_id": str(group_id),
            "group_type": str(group_type),
            "order": list(order),
            "boundary_after_indices": list(boundary_after_indices),
            "ordered_cluster_labels": list(reordered_similarity["ordered_cluster_labels"]),
            "ordered_object_ids": list(reordered_similarity["ordered_object_ids"]),
        },
    )
    if group_metadata is not None:
        _write_json(group_dir / "group_metadata.json", dict(group_metadata))

    annotation_paths = save_view_annotation_results(
        group_dir,
        rows=rows,
        labels=labels,
        image_roots=list(image_roots or []),
    )
    refined_paths = save_refined_graph_visualization_results(
        group_dir,
        rows=rows,
        similarity_matrix=similarity_matrix,
        object_ids=object_ids,
        heatmap_labels=heatmap_labels,
    )

    plot_similarity_heatmap(
        similarity_matrix,
        group_dir / "similarity_heatmap.png",
        title=f"Text Cosine Similarity: {group_id}",
        order=order,
        axis_labels=heatmap_labels,
        vmin=-1.0,
        vmax=1.0,
    )
    plot_similarity_heatmap(
        affinity_matrix,
        group_dir / "affinity_heatmap.png",
        title=f"Constrained Affinity: {group_id}",
        order=order,
        axis_labels=heatmap_labels,
        vmin=0.0,
        vmax=1.0,
    )
    plot_similarity_heatmap(
        similarity_matrix,
        group_dir / "clustered_similarity_heatmap.png",
        title=CLUSTERED_SIMILARITY_HEATMAP_TITLE,
        order=order,
        boundary_after_indices=boundary_after_indices,
        axis_labels=heatmap_labels,
        vmin=0.0,
        vmax=1.0,
        display_transform="log",
    )
    plot_similarity_heatmap(
        similarity_matrix,
        group_dir / "clustered_similarity_heatmap_offdiag_only.png",
        title=CLUSTERED_SIMILARITY_OFFDIAG_HEATMAP_TITLE,
        order=order,
        boundary_after_indices=boundary_after_indices,
        axis_labels=heatmap_labels,
        annotate_values=False,
        vmin=0.0,
        vmax=1.0,
        display_transform="offdiag_extreme",
        suppress_diagonal_display=True,
    )
    saved = {
        "objects_json": group_dir / "objects.json",
        "similarity_matrix": group_dir / "similarity_matrix.npy",
        "affinity_matrix": group_dir / "affinity_matrix.npy",
        "clustered_similarity_matrix": group_dir / "clustered_similarity_matrix.npy",
        "clustered_similarity_matrix_csv": group_dir / "clustered_similarity_matrix.csv",
        "clustered_similarity_order": group_dir / "clustered_similarity_order.json",
        "cluster_labels": group_dir / "cluster_labels.json",
        "cluster_summary": group_dir / "cluster_summary.json",
        "cluster_summary_md": group_dir / "cluster_summary.md",
        "similarity_heatmap": group_dir / "similarity_heatmap.png",
        "affinity_heatmap": group_dir / "affinity_heatmap.png",
        "clustered_similarity_heatmap": group_dir / "clustered_similarity_heatmap.png",
        "clustered_similarity_heatmap_offdiag_only": group_dir / "clustered_similarity_heatmap_offdiag_only.png",
        "view_annotations_dir": annotation_paths["annotations_dir"],
        "view_annotations_manifest": annotation_paths["manifest_path"],
        "view_annotations_grid": annotation_paths["grid_path"],
    }
    saved.update(refined_paths)
    return saved


def build_manual_anchor_camera_specs(
    anchor_row: Mapping[str, Any],
    *,
    default_radius_m: float = DEFAULT_MANUAL_ANCHOR_RADIUS_M,
) -> List[Dict[str, Any]]:
    anchor_id = _safe_text(anchor_row.get("anchor_id")) or "anchor"
    center_x = float(anchor_row["center_x"])
    center_z = float(anchor_row["center_z"])
    radius_m = _safe_float(anchor_row.get("radius_m"))
    radius = float(default_radius_m if radius_m is None else radius_m)
    return [
        {
            "anchor_id": anchor_id,
            "anchor_direction": "north",
            "requested_x": center_x,
            "requested_z": center_z + radius,
            "orientation_deg": 0,
        },
        {
            "anchor_id": anchor_id,
            "anchor_direction": "east",
            "requested_x": center_x + radius,
            "requested_z": center_z,
            "orientation_deg": 90,
        },
        {
            "anchor_id": anchor_id,
            "anchor_direction": "south",
            "requested_x": center_x,
            "requested_z": center_z - radius,
            "orientation_deg": 180,
        },
        {
            "anchor_id": anchor_id,
            "anchor_direction": "west",
            "requested_x": center_x - radius,
            "requested_z": center_z,
            "orientation_deg": 270,
        },
    ]


def validate_manual_anchor_pose_results(
    pose_results: Sequence[Mapping[str, Any]],
    *,
    max_snap_distance_m: float = DEFAULT_MAX_SNAP_DISTANCE_M,
    min_pose_separation_m: float = DEFAULT_MIN_POSE_SEPARATION_M,
) -> None:
    if len(pose_results) != 4:
        raise ValueError(f"Manual anchor requires exactly 4 views, got {len(pose_results)}")
    resolved_positions: List[Tuple[float, float]] = []
    for row in pose_results:
        requested_x = float(row["requested_x"])
        requested_z = float(row["requested_z"])
        actual_world_position = row.get("actual_world_position")
        if not isinstance(actual_world_position, (list, tuple)) or len(actual_world_position) != 3:
            raise ValueError(f"Invalid actual_world_position for anchor view: {actual_world_position!r}")
        actual_x = float(actual_world_position[0])
        actual_z = float(actual_world_position[2])
        snap_distance = float(math.hypot(actual_x - requested_x, actual_z - requested_z))
        if snap_distance > float(max_snap_distance_m):
            raise ValueError(
                f"Anchor view {row.get('synthetic_view_id') or row.get('anchor_direction')} exceeded snap threshold: "
                f"{snap_distance:.3f}m > {float(max_snap_distance_m):.3f}m"
            )
        resolved_positions.append((actual_x, actual_z))

    for left_index in range(len(resolved_positions)):
        for right_index in range(left_index + 1, len(resolved_positions)):
            dx = float(resolved_positions[left_index][0] - resolved_positions[right_index][0])
            dz = float(resolved_positions[left_index][1] - resolved_positions[right_index][1])
            if math.hypot(dx, dz) < float(min_pose_separation_m):
                raise ValueError("Manual anchor became degenerate after snapping: two views collapsed together")


def _collect_anchor_scene_records(
    *,
    explorer,
    captioner: VLMCaptioner,
    anchor_row: Mapping[str, Any],
    anchor_specs: Sequence[Mapping[str, Any]],
    group_dir: Path,
    starting_entry_id: int,
    starting_object_id: int,
    object_max_per_frame: int,
    object_parse_retries: int,
    object_prompt_variant: str,
    angle_split_enable: bool,
    angle_step: int,
) -> Dict[str, Any]:
    generated_view_dir = group_dir / "generated_views"
    generated_view_dir.mkdir(parents=True, exist_ok=True)
    pose_results: List[Dict[str, Any]] = []
    pending_rows: List[Dict[str, Any]] = []
    current_entry_id = int(starting_entry_id)
    current_object_id = int(starting_object_id)

    for spec in anchor_specs:
        actual_world_position, used_snap = _set_agent_pose_2d(
            explorer,
            x0=float(spec["requested_x"]),
            y0=float(spec["requested_z"]),
            theta0=float(spec["orientation_deg"]),
        )
        synthetic_view_id = f"anchor_{_slugify_group_id(spec['anchor_id'])}_{spec['anchor_direction']}"
        image_name = f"{synthetic_view_id}.jpg"
        image_rel_path = Path("generated_views") / image_name
        image_path = generated_view_dir / image_name

        obs = explorer.sim.get_sensor_observations()
        rgb = obs["color_sensor"]
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        ok = cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to save manual anchor view image: {image_path}")

        camera_context = {
            "camera_x": float(actual_world_position[0]),
            "camera_z": float(actual_world_position[2]),
            "camera_orientation_deg": float(spec["orientation_deg"]),
        }
        parse_result = _parse_objects_with_retry(
            captioner=captioner,
            image_path=str(image_path),
            image_id=synthetic_view_id,
            max_objects=int(object_max_per_frame),
            retries=int(object_parse_retries),
            prompt_variant=object_prompt_variant,
            camera_context=camera_context,
        )
        scene_objects = parse_result.scene_objects
        parse_status = parse_result.parse_status
        if scene_objects is not None:
            parse_status = "ok"
            _enrich_scene_objects_geometry(
                scene_objects,
                camera_x=float(actual_world_position[0]),
                camera_y=float(actual_world_position[1]),
                camera_z=float(actual_world_position[2]),
                camera_orientation_deg=float(spec["orientation_deg"]),
                angle_step=int(angle_step),
            )
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
            object_text_inputs_short: List[str] = []
            object_text_inputs_long: List[str] = []
            object_rows: List[Dict[str, Any]] = []
            for obj in objs:
                obj_text_short = select_object_text(obj, mode="short", scene_objects=scene_objects)
                raw_long = select_object_text(obj, mode="long", scene_objects=scene_objects)
                angle_bucket = _normalize_angle_bucket(
                    obj.relative_position_laterality,
                    angle_split_enable=bool(angle_split_enable),
                )
                obj_text_long = _format_object_text_long(
                    raw_long,
                    angle_bucket=angle_bucket,
                    builder_variant="angle_split" if angle_split_enable else "manual_anchor",
                )
                object_text_inputs_short.append(obj_text_short)
                object_text_inputs_long.append(obj_text_long)
                record = _make_object_record(
                    object_global_id=current_object_id,
                    frame_id=current_entry_id,
                    entry_id=current_entry_id,
                    file_name=str(image_rel_path),
                    x=float(actual_world_position[0]),
                    y=float(actual_world_position[2]),
                    world_position=[
                        float(actual_world_position[0]),
                        float(actual_world_position[1]),
                        float(actual_world_position[2]),
                    ],
                    orientation=int(spec["orientation_deg"]),
                    parse_status=parse_status,
                    builder_variant="manual_anchor_center",
                    angle_split_enable=bool(angle_split_enable),
                    angle_step=int(angle_step),
                    scene_objects=scene_objects,
                    obj=obj,
                    object_local_id=str(obj.feature_id),
                    label=str(obj.type),
                    object_confidence=1.0,
                    description=obj_text_short,
                    long_form_open_description=raw_long,
                    attributes=list(obj.attributes or []),
                    laterality=str(obj.relative_position_laterality),
                    distance_bin=str(obj.relative_position_distance),
                    verticality=str(obj.relative_position_verticality),
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
                    object_text_short=obj_text_short,
                    object_text_long=obj_text_long,
                )
                record["anchor_id"] = str(spec["anchor_id"])
                record["synthetic_view_id"] = synthetic_view_id
                record["anchor_direction"] = str(spec["anchor_direction"])
                object_rows.append(record)
                current_object_id += 1
        else:
            frame_text_short = UNKNOWN_TEXT_TOKEN
            frame_text_long = UNKNOWN_TEXT_TOKEN
            object_text_inputs_short = [UNKNOWN_TEXT_TOKEN]
            object_text_inputs_long = [UNKNOWN_TEXT_TOKEN]
            object_rows = []

        meta_row = {
            "id": int(current_entry_id),
            "frame_id": int(current_entry_id),
            "anchor_id": str(spec["anchor_id"]),
            "synthetic_view_id": synthetic_view_id,
            "anchor_direction": str(spec["anchor_direction"]),
            "x": float(actual_world_position[0]),
            "y": float(actual_world_position[2]),
            "world_position": [
                float(actual_world_position[0]),
                float(actual_world_position[1]),
                float(actual_world_position[2]),
            ],
            "orientation": int(spec["orientation_deg"]),
            "file_name": str(image_rel_path),
            "text": frame_text_short,
            "frame_text_short": frame_text_short,
            "frame_text_long": frame_text_long,
            "parse_status": parse_status,
            "parse_warnings": list(parse_result.warnings),
            "raw_vlm_output": parse_result.raw_vlm_output,
            "raw_api_source": parse_result.raw_api_source,
            "text_input_for_clip_short": frame_text_short,
            "text_input_for_clip_long": frame_text_long,
            "object_text_inputs_short": object_text_inputs_short,
            "object_text_inputs_long": object_text_inputs_long,
            "builder_variant": "manual_anchor_center",
            "object_prompt_variant": object_prompt_variant,
            "attribute": _build_view_attribute(scene_objects=scene_objects, raw_vlm_output=parse_result.raw_vlm_output),
            "object_count": len(object_rows),
            "view_type": _safe_text(getattr(scene_objects, "view_type", "") or "unknown", "unknown"),
            "room_function": _safe_text(getattr(scene_objects, "room_function", "") or "unknown", "unknown"),
            "used_snap": bool(used_snap),
        }

        pose_results.append(
            {
                "anchor_id": str(spec["anchor_id"]),
                "anchor_direction": str(spec["anchor_direction"]),
                "requested_x": float(spec["requested_x"]),
                "requested_z": float(spec["requested_z"]),
                "orientation_deg": int(spec["orientation_deg"]),
                "synthetic_view_id": synthetic_view_id,
                "actual_world_position": [
                    float(actual_world_position[0]),
                    float(actual_world_position[1]),
                    float(actual_world_position[2]),
                ],
                "used_snap": bool(used_snap),
                "meta_row": meta_row,
                "object_rows": object_rows,
            }
        )
        pending_rows.append(
            {
                "meta_row": meta_row,
                "object_rows": object_rows,
            }
        )
        current_entry_id += 1

    meta_rows = [row["meta_row"] for row in pending_rows]
    object_rows = [obj for row in pending_rows for obj in row["object_rows"]]
    _write_jsonl(generated_view_dir / "meta.jsonl", meta_rows)
    _write_jsonl(generated_view_dir / "object_meta.jsonl", object_rows)
    return {
        "group_id": str(anchor_row.get("anchor_id") or "anchor"),
        "group_type": "manual_anchor_center",
        "anchor_row": dict(anchor_row),
        "meta_rows": meta_rows,
        "object_rows": object_rows,
        "pose_results": pose_results,
        "next_entry_id": current_entry_id,
        "next_object_id": current_object_id,
    }


def collect_manual_anchor_groups(
    *,
    anchors_jsonl: str,
    output_dir: str,
    scene_path: str = SCENE_PATH,
    vlm_model: str = SPATIAL_DB_VLM_MODEL,
    use_cache: bool = True,
    object_parse_retries: int = OBJECT_PARSE_RETRIES,
    object_use_cache: bool = OBJECT_USE_CACHE,
    object_cache_dir: Optional[str] = None,
    object_max_per_frame: int = OBJECT_MAX_PER_FRAME,
    angle_split_enable: bool = VLM_ANGLE_SPLIT_ENABLE,
    angle_step: int = VLM_ANGLE_STEP,
    object_prompt_variant: str = "angle_split",
    default_radius_m: float = DEFAULT_MANUAL_ANCHOR_RADIUS_M,
    max_snap_distance_m: float = DEFAULT_MAX_SNAP_DISTANCE_M,
    min_pose_separation_m: float = DEFAULT_MIN_POSE_SEPARATION_M,
    text_mode: str = DEFAULT_TEXT_MODE,
    embedder=None,
) -> List[Dict[str, Any]]:
    anchors = _load_jsonl(Path(anchors_jsonl))
    if not anchors:
        return []
    from spatial_rag.explorer import Explorer

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(object_cache_dir) if object_cache_dir is not None else output_root / str(OBJECT_CACHE_DIR)
    captioner = VLMCaptioner(
        model_name=vlm_model,
        use_cache=use_cache,
        cache_dir=str(output_root / "vlm_cache"),
        object_use_cache=object_use_cache,
        object_cache_dir=str(cache_root),
    )
    explorer = Explorer(scene_path=scene_path)
    try:
        groups: List[Dict[str, Any]] = []
        next_entry_id = 0
        next_object_id = 0
        for anchor_index, anchor_row in enumerate(anchors):
            anchor_id = _safe_text(anchor_row.get("anchor_id")) or f"anchor_{anchor_index:05d}"
            group_id = anchor_id
            group_dir = output_root / _slugify_group_id(group_id)
            anchor_specs = build_manual_anchor_camera_specs(
                {
                    **dict(anchor_row),
                    "anchor_id": anchor_id,
                },
                default_radius_m=default_radius_m,
            )
            try:
                bundle = _collect_anchor_scene_records(
                    explorer=explorer,
                    captioner=captioner,
                    anchor_row={**dict(anchor_row), "anchor_id": anchor_id},
                    anchor_specs=anchor_specs,
                    group_dir=group_dir,
                    starting_entry_id=next_entry_id,
                    starting_object_id=next_object_id,
                    object_max_per_frame=object_max_per_frame,
                    object_parse_retries=object_parse_retries,
                    object_prompt_variant=object_prompt_variant,
                    angle_split_enable=angle_split_enable,
                    angle_step=angle_step,
                )
                validate_manual_anchor_pose_results(
                    bundle["pose_results"],
                    max_snap_distance_m=max_snap_distance_m,
                    min_pose_separation_m=min_pose_separation_m,
                )
                observations = _attach_object_embeddings(
                    _manual_anchor_observations_from_rows(
                        meta_rows=bundle["meta_rows"],
                        object_rows=bundle["object_rows"],
                        anchor_id=anchor_id,
                    ),
                    db_dir=None,
                    text_mode=text_mode,
                    embedder=embedder,
                )
                groups.append(
                    {
                        "group_id": group_id,
                        "group_type": "manual_anchor_center",
                        "objects": observations,
                        "generated_meta_rows": bundle["meta_rows"],
                        "generated_object_rows": bundle["object_rows"],
                        "group_metadata": {
                            "anchor": dict(anchor_row),
                            "pose_results": bundle["pose_results"],
                        },
                    }
                )
                next_entry_id = int(bundle["next_entry_id"])
                next_object_id = int(bundle["next_object_id"])
            except Exception as exc:
                groups.append(
                    {
                        "group_id": group_id,
                        "group_type": "manual_anchor_center",
                        "objects": [],
                        "generated_meta_rows": [],
                        "generated_object_rows": [],
                        "group_metadata": {
                            "anchor": dict(anchor_row),
                            "skip_reason": str(exc),
                        },
                    }
                )
        return groups
    finally:
        try:
            explorer.sim.close()
        except Exception:
            pass


def _manual_anchor_observations_from_rows(
    *,
    meta_rows: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
    anchor_id: str,
) -> List[Dict[str, Any]]:
    meta_by_entry = {
        _safe_int(row.get("id"), -1): dict(row)
        for row in meta_rows
    }
    observations: List[Dict[str, Any]] = []
    for row in sorted(object_rows, key=lambda item: _safe_int(item.get("object_global_id"), 10**9)):
        entry = meta_by_entry.get(_safe_int(row.get("entry_id"), -1), {})
        synthetic_view_id = str(row.get("synthetic_view_id") or entry.get("synthetic_view_id") or "")
        observation = dict(row)
        observation["group_id"] = str(anchor_id)
        observation["group_type"] = "manual_anchor_center"
        observation["anchor_id"] = str(anchor_id)
        observation["place_id"] = None
        observation["view_id"] = synthetic_view_id or f"view_{_safe_int(row.get('entry_id'), -1):05d}"
        observation["synthetic_view_id"] = synthetic_view_id or observation["view_id"]
        observation["observation_id"] = str(row.get("observation_id") or f"obs_{_safe_int(row.get('object_global_id'), 0):06d}")
        observation["file_name"] = str(row.get("file_name") or entry.get("file_name") or "")
        observation["parse_status"] = str(row.get("parse_status") or entry.get("parse_status") or "unknown")
        observation["description"] = _safe_text(row.get("description"))
        observation["long_form_open_description"] = _safe_text(row.get("long_form_open_description"))
        observation["view_orientation_deg"] = _safe_float(entry.get("orientation"))
        observations.append(observation)
    observations.sort(key=_observation_sort_key)
    return observations


def _filter_rows_for_clustering(
    rows: Sequence[Mapping[str, Any]],
    *,
    text_mode: str,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        embedding = row.get("embedding")
        text = _resolve_object_text(row, text_mode=text_mode)
        if embedding is None or not _has_usable_text(text):
            continue
        filtered.append(dict(row))
    filtered.sort(key=_observation_sort_key)
    return filtered


def _average_cluster_size(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    counts = Counter(int(label) for label in labels)
    return float(np.mean(list(counts.values()))) if counts else 0.0


def _build_instance_candidate_graph(
    cluster_summaries: Sequence[Mapping[str, Any]],
    rows_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    labels_by_group: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    candidate_nodes: List[Dict[str, Any]] = []
    membership_edges: List[Dict[str, Any]] = []
    for cluster_summary in cluster_summaries:
        group_id = str(cluster_summary["group_id"])
        rows = list(rows_by_group.get(group_id, []))
        labels = list(labels_by_group.get(group_id, []))
        label_lookup = {
            _safe_int(row.get("object_global_id"), -1): int(cluster_id)
            for row, cluster_id in zip(rows, labels)
        }
        for cluster in cluster_summary.get("clusters", []):
            candidate_nodes.append(
                {
                    "node_type": "InstanceCandidate",
                    "candidate_instance_id": cluster["candidate_instance_id"],
                    "group_id": group_id,
                    "group_type": cluster_summary["group_type"],
                    "cluster_id": int(cluster["cluster_id"]),
                    "num_members": int(cluster["num_members"]),
                    "representative_object_id": int(cluster["representative_object_id"]),
                    "representative_label": str(cluster["representative_label"]),
                    "representative_description": str(cluster["representative_description"]),
                    "same_view_collision": bool(cluster["same_view_collision"]),
                    "member_object_ids": list(cluster["member_object_ids"]),
                    "member_view_ids": list(cluster["member_view_ids"]),
                    "label_histogram": dict(cluster["label_histogram"]),
                }
            )
        for row in rows:
            object_id = _safe_int(row.get("object_global_id"), -1)
            cluster_id = label_lookup.get(object_id)
            if cluster_id is None:
                continue
            membership_edges.append(
                {
                    "edge_type": "OBSERVATION_MEMBER_OF",
                    "candidate_instance_id": f"{cluster_summary['group_type']}:{group_id}:cluster_{int(cluster_id):03d}",
                    "object_global_id": object_id,
                    "observation_id": str(row.get("observation_id") or ""),
                    "view_id": str(row.get("view_id") or ""),
                    "group_id": group_id,
                }
            )
    return {
        "instance_candidates": candidate_nodes,
        "observation_membership_edges": membership_edges,
    }


def _process_group(
    *,
    group_id: str,
    group_type: str,
    rows: Sequence[Mapping[str, Any]],
    group_dir: Path,
    text_mode: str,
    same_view_policy: str,
    same_view_penalty: float,
    min_similarity: Optional[float],
    top_k: Optional[int],
    cluster_count_mode: str,
    n_clusters: Optional[int],
    random_state: int,
    group_metadata: Optional[Mapping[str, Any]] = None,
    image_roots: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    filtered_rows = _filter_rows_for_clustering(rows, text_mode=text_mode)
    similarity_matrix = build_similarity_matrix_from_descriptions(filtered_rows)
    affinity_matrix = apply_constraints(
        similarity_matrix,
        filtered_rows,
        same_view_policy=same_view_policy,
        same_view_penalty=same_view_penalty,
        min_similarity=min_similarity,
        top_k=top_k,
    )
    clustering = run_spectral_clustering(
        affinity_matrix,
        object_ids=[_safe_int(row.get("object_global_id"), index) for index, row in enumerate(filtered_rows)],
        cluster_count_mode=cluster_count_mode,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    labels = clustering["labels"].tolist()
    cluster_summary = summarize_clusters(
        filtered_rows,
        similarity_matrix,
        labels,
        group_id=group_id,
        group_type=group_type,
    )
    if group_metadata and group_metadata.get("skip_reason"):
        cluster_summary = {
            **dict(cluster_summary),
            "skip_reason": str(group_metadata["skip_reason"]),
        }

    saved_paths = save_cluster_results(
        group_dir,
        group_id=group_id,
        group_type=group_type,
        rows=filtered_rows,
        similarity_matrix=similarity_matrix,
        affinity_matrix=affinity_matrix,
        labels=labels,
        cluster_summary=cluster_summary,
        group_metadata=group_metadata,
        image_roots=image_roots,
    )
    return {
        "group_id": str(group_id),
        "group_type": str(group_type),
        "rows": filtered_rows,
        "labels": labels,
        "cluster_summary": cluster_summary,
        "saved_paths": saved_paths,
        "backend": clustering["backend"],
        "fallback_reason": clustering["fallback_reason"],
        "n_objects": len(filtered_rows),
        "n_clusters": int(cluster_summary.get("n_clusters", 0)),
        "average_cluster_size": _average_cluster_size(labels),
    }


def run_object_instance_clustering(
    *,
    output_dir: str,
    group_mode: str = DEFAULT_GROUP_MODE,
    db_dir: Optional[str] = None,
    entry_ids: Optional[Sequence[int]] = None,
    anchors_jsonl: Optional[str] = None,
    scene_path: str = SCENE_PATH,
    text_mode: str = DEFAULT_TEXT_MODE,
    cluster_count_mode: str = DEFAULT_CLUSTER_COUNT_MODE,
    n_clusters: Optional[int] = None,
    same_view_policy: str = DEFAULT_SAME_VIEW_POLICY,
    same_view_penalty: float = DEFAULT_SAME_VIEW_PENALTY,
    min_similarity: Optional[float] = None,
    top_k: Optional[int] = None,
    random_state: int = 0,
    vlm_model: str = SPATIAL_DB_VLM_MODEL,
    use_cache: bool = True,
    object_parse_retries: int = OBJECT_PARSE_RETRIES,
    object_use_cache: bool = OBJECT_USE_CACHE,
    object_cache_dir: Optional[str] = None,
    object_max_per_frame: int = OBJECT_MAX_PER_FRAME,
    default_anchor_radius_m: float = DEFAULT_MANUAL_ANCHOR_RADIUS_M,
    max_snap_distance_m: float = DEFAULT_MAX_SNAP_DISTANCE_M,
    min_pose_separation_m: float = DEFAULT_MIN_POSE_SEPARATION_M,
    export_graph_json: bool = True,
    embedder=None,
) -> Dict[str, Any]:
    mode = _normalize_group_mode(group_mode)
    text_mode = _normalize_text_mode(text_mode)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if mode == "place":
        if not db_dir:
            raise ValueError("db_dir is required when group_mode='place'")
        observations = load_object_observations(db_dir=db_dir, text_mode=text_mode, embedder=embedder)
        observations = _filter_observations_by_entry_ids(observations, entry_ids)
        groups: List[Dict[str, Any]] = []
        for group_id, rows in group_objects_by_scope(observations, group_mode="place").items():
            groups.append(
                {
                    "group_id": group_id,
                    "group_type": "place",
                    "objects": rows,
                    "group_metadata": {
                        "source_db_dir": db_dir,
                        "selected_entry_ids": None if not entry_ids else [int(entry_id) for entry_id in entry_ids],
                    },
                }
            )
    elif mode == "selected_views":
        if not db_dir:
            raise ValueError("db_dir is required when group_mode='selected_views'")
        observations = load_object_observations(db_dir=db_dir, text_mode=text_mode, embedder=embedder)
        observations = _filter_observations_by_entry_ids(observations, entry_ids)
        groups = [
            {
                "group_id": "selected_views",
                "group_type": "selected_views",
                "objects": list(observations),
                "group_metadata": {
                    "source_db_dir": db_dir,
                    "selected_entry_ids": None if not entry_ids else [int(entry_id) for entry_id in entry_ids],
                },
            }
        ]
    else:
        if entry_ids:
            raise ValueError("entry_ids is only supported when group_mode='place' or 'selected_views'")
        if not anchors_jsonl:
            raise ValueError("anchors_jsonl is required when group_mode='manual_anchor_center'")
        groups = collect_manual_anchor_groups(
            anchors_jsonl=anchors_jsonl,
            output_dir=output_dir,
            scene_path=scene_path,
            vlm_model=vlm_model,
            use_cache=use_cache,
            object_parse_retries=object_parse_retries,
            object_use_cache=object_use_cache,
            object_cache_dir=object_cache_dir,
            object_max_per_frame=object_max_per_frame,
            angle_split_enable=VLM_ANGLE_SPLIT_ENABLE,
            angle_step=VLM_ANGLE_STEP,
            default_radius_m=default_anchor_radius_m,
            max_snap_distance_m=max_snap_distance_m,
            min_pose_separation_m=min_pose_separation_m,
            text_mode=text_mode,
            embedder=embedder,
        )

    processed_groups: List[Dict[str, Any]] = []
    rows_by_group: Dict[str, Sequence[Mapping[str, Any]]] = {}
    labels_by_group: Dict[str, Sequence[int]] = {}
    cluster_summaries: List[Mapping[str, Any]] = []

    for group in groups:
        group_id = str(group["group_id"])
        group_type = str(group["group_type"])
        group_dir = output_root / _slugify_group_id(group_id)
        result = _process_group(
            group_id=group_id,
            group_type=group_type,
            rows=list(group.get("objects", [])),
            group_dir=group_dir,
            text_mode=text_mode,
            same_view_policy=same_view_policy,
            same_view_penalty=same_view_penalty,
            min_similarity=min_similarity,
            top_k=top_k,
            cluster_count_mode=cluster_count_mode,
            n_clusters=n_clusters,
            random_state=random_state,
            group_metadata=group.get("group_metadata"),
            image_roots=[Path(db_dir)] if mode == "place" and db_dir else [group_dir],
        )
        processed_groups.append(
            {
                "group_id": result["group_id"],
                "group_type": result["group_type"],
                "n_objects": result["n_objects"],
                "n_clusters": result["n_clusters"],
                "average_cluster_size": result["average_cluster_size"],
                "backend": result["backend"],
                "fallback_reason": result["fallback_reason"],
                "skip_reason": (group.get("group_metadata") or {}).get("skip_reason"),
            }
        )
        rows_by_group[group_id] = result["rows"]
        labels_by_group[group_id] = result["labels"]
        cluster_summaries.append(result["cluster_summary"])

    report = {
        "group_mode": mode,
        "text_mode": text_mode,
        "selected_entry_ids": None if not entry_ids else [int(entry_id) for entry_id in entry_ids],
        "cluster_count_mode": _normalize_cluster_count_mode(cluster_count_mode),
        "same_view_policy": _normalize_same_view_policy(same_view_policy),
        "same_view_penalty": float(same_view_penalty),
        "min_similarity": None if min_similarity is None else float(min_similarity),
        "top_k": None if top_k is None else int(top_k),
        "num_groups": len(processed_groups),
        "groups": processed_groups,
    }
    _write_json(output_root / "summary.json", report)

    if export_graph_json:
        graph_payload = _build_instance_candidate_graph(
            cluster_summaries=cluster_summaries,
            rows_by_group=rows_by_group,
            labels_by_group=labels_by_group,
        )
        _write_json(output_root / "instance_candidate_graph.json", graph_payload)

    return report


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Cluster object observations into same-instance candidates using text-similarity spectral clustering."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for clustering artifacts")
    parser.add_argument(
        "--group_mode",
        type=str,
        default=DEFAULT_GROUP_MODE,
        choices=["place", "selected_views", "manual_anchor_center"],
        help="Grouping scope for object clustering",
    )
    parser.add_argument("--db_dir", type=str, default=None, help="Existing spatial DB directory for place grouping")
    parser.add_argument(
        "--entry_ids",
        type=str,
        default=None,
        help="Optional comma-separated entry ids to restrict place-mode clustering to a subset of views",
    )
    parser.add_argument("--anchors_jsonl", type=str, default=None, help="Manual anchor JSONL for manual anchor mode")
    parser.add_argument("--scene_path", type=str, default=SCENE_PATH, help="Habitat scene path for manual anchor mode")
    parser.add_argument(
        "--text_mode",
        type=str,
        default=DEFAULT_TEXT_MODE,
        choices=["short", "long", "long_neighbors"],
        help="Which object text embedding variant to use",
    )
    parser.add_argument(
        "--cluster_count_mode",
        type=str,
        default=DEFAULT_CLUSTER_COUNT_MODE,
        choices=["fixed", "eigengap"],
        help="How to choose the number of clusters",
    )
    parser.add_argument("--n_clusters", type=int, default=None, help="Explicit number of clusters for fixed mode")
    parser.add_argument(
        "--same_view_policy",
        type=str,
        default=DEFAULT_SAME_VIEW_POLICY,
        choices=["soft_penalty", "hard_block", "none"],
        help="How to handle pairs from the same view",
    )
    parser.add_argument(
        "--same_view_penalty",
        type=float,
        default=DEFAULT_SAME_VIEW_PENALTY,
        help="Penalty multiplier for same-view pairs when using soft_penalty",
    )
    parser.add_argument("--min_similarity", type=float, default=None, help="Optional minimum affinity threshold")
    parser.add_argument("--top_k", type=int, default=None, help="Optional symmetric top-k affinity sparsification")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed for clustering")
    parser.add_argument("--vlm_model", type=str, default=SPATIAL_DB_VLM_MODEL, help="VLM model for manual anchor mode")
    parser.add_argument("--use_cache", action="store_true", help="Enable VLM cache")
    parser.add_argument("--no_use_cache", action="store_false", dest="use_cache", help="Disable VLM cache")
    parser.set_defaults(use_cache=True)
    parser.add_argument("--object_parse_retries", type=int, default=OBJECT_PARSE_RETRIES)
    parser.add_argument("--object_use_cache", action="store_true", help="Enable object cache")
    parser.add_argument("--no_object_use_cache", action="store_false", dest="object_use_cache", help="Disable object cache")
    parser.set_defaults(object_use_cache=OBJECT_USE_CACHE)
    parser.add_argument("--object_cache_dir", type=str, default=None)
    parser.add_argument("--object_max_per_frame", type=int, default=OBJECT_MAX_PER_FRAME)
    parser.add_argument("--default_anchor_radius_m", type=float, default=DEFAULT_MANUAL_ANCHOR_RADIUS_M)
    parser.add_argument("--max_snap_distance_m", type=float, default=DEFAULT_MAX_SNAP_DISTANCE_M)
    parser.add_argument("--min_pose_separation_m", type=float, default=DEFAULT_MIN_POSE_SEPARATION_M)
    parser.add_argument("--skip_graph_export", action="store_true", help="Do not write instance_candidate_graph.json")
    args = parser.parse_args(argv)

    report = run_object_instance_clustering(
        output_dir=args.output_dir,
        group_mode=args.group_mode,
        db_dir=args.db_dir,
        entry_ids=_parse_entry_ids_arg(args.entry_ids),
        anchors_jsonl=args.anchors_jsonl,
        scene_path=args.scene_path,
        text_mode=args.text_mode,
        cluster_count_mode=args.cluster_count_mode,
        n_clusters=args.n_clusters,
        same_view_policy=args.same_view_policy,
        same_view_penalty=args.same_view_penalty,
        min_similarity=args.min_similarity,
        top_k=args.top_k,
        random_state=args.random_state,
        vlm_model=args.vlm_model,
        use_cache=args.use_cache,
        object_parse_retries=args.object_parse_retries,
        object_use_cache=args.object_use_cache,
        object_cache_dir=args.object_cache_dir,
        object_max_per_frame=args.object_max_per_frame,
        default_anchor_radius_m=args.default_anchor_radius_m,
        max_snap_distance_m=args.max_snap_distance_m,
        min_pose_separation_m=args.min_pose_separation_m,
        export_graph_json=not bool(args.skip_graph_export),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return report


if __name__ == "__main__":
    main()
