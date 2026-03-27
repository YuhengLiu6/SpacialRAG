import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_object_db(
    db_dir: str,
    text_mode: str = "short",
) -> Optional[Tuple[List[Dict], np.ndarray, Dict[int, List[int]]]]:
    mode = str(text_mode or "").strip().lower()
    if mode not in {"short", "long"}:
        raise ValueError(f"Unsupported text_mode: {text_mode}")
    root = Path(db_dir)
    meta_path = root / "object_meta.jsonl"
    emb_path = root / f"object_text_emb_{mode}.npy"
    if not meta_path.exists() or not emb_path.exists():
        return None

    meta = _load_jsonl(meta_path)
    emb = np.load(emb_path).astype("float32")
    if emb.ndim != 2:
        raise ValueError(f"Invalid object_text_emb shape: {emb.shape}")
    if emb.shape[0] != len(meta):
        raise ValueError(
            f"Object DB mismatch: meta={len(meta)} rows, object_text_emb={emb.shape[0]}"
        )

    entry_to_indices: Dict[int, List[int]] = {}
    for idx, row in enumerate(meta):
        entry_id = int(row.get("entry_id", -1))
        if entry_id < 0:
            continue
        entry_to_indices.setdefault(entry_id, []).append(idx)

    return meta, emb, entry_to_indices


def load_object_faiss_index(db_dir: str, text_mode: str = "short"):
    mode = str(text_mode or "").strip().lower()
    if mode not in {"short", "long"}:
        raise ValueError(f"Unsupported text_mode: {text_mode}")
    index_path = Path(db_dir) / f"object_index_{mode}.faiss"
    if not index_path.exists():
        return None
    import faiss

    return faiss.read_index(str(index_path))


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def compute_frame_object_scores(
    query_object_embs: np.ndarray,
    candidate_entry_ids: List[int],
    object_embs: np.ndarray,
    entry_to_indices: Dict[int, List[int]],
) -> Dict[int, float]:
    if query_object_embs.ndim != 2:
        raise ValueError("query_object_embs must be a 2D array")
    if object_embs.ndim != 2:
        raise ValueError("object_embs must be a 2D array")

    q = _l2_normalize(query_object_embs, axis=1)
    db = _l2_normalize(object_embs, axis=1)

    scores: Dict[int, float] = {}
    for entry_id in candidate_entry_ids:
        idxs = entry_to_indices.get(int(entry_id), [])
        if not idxs:
            scores[int(entry_id)] = 0.0
            continue
        obj_mat = db[np.asarray(idxs, dtype=np.int64)]
        sim = q @ obj_mat.T  # [num_query_objects, num_candidate_objects]
        max_per_query = np.max(sim, axis=1)
        scores[int(entry_id)] = float(np.mean(max_per_query))
    return scores


def compute_frame_object_scores_faiss(
    query_object_embs: np.ndarray,
    candidate_entry_ids: List[int],
    object_meta: List[Dict],
    object_index,
    top_k_per_query: int = 256,
) -> Dict[int, float]:
    if query_object_embs.ndim != 2:
        raise ValueError("query_object_embs must be a 2D array")
    if len(candidate_entry_ids) == 0:
        return {}
    if object_index is None or int(object_index.ntotal) <= 0:
        return {int(eid): 0.0 for eid in candidate_entry_ids}

    q = _l2_normalize(query_object_embs, axis=1).astype("float32")
    k = max(1, min(int(top_k_per_query), int(object_index.ntotal)))
    sims, inds = object_index.search(q, k)

    candidate_set = {int(eid) for eid in candidate_entry_ids}
    per_entry_scores: Dict[int, List[float]] = {eid: [] for eid in candidate_set}

    for qi in range(q.shape[0]):
        best_for_entry: Dict[int, float] = {eid: 0.0 for eid in candidate_set}
        for score, obj_idx in zip(sims[qi], inds[qi]):
            if int(obj_idx) < 0 or int(obj_idx) >= len(object_meta):
                continue
            entry_id = int(object_meta[int(obj_idx)].get("entry_id", -1))
            if entry_id in candidate_set and float(score) > best_for_entry[entry_id]:
                best_for_entry[entry_id] = float(score)
        for entry_id, score in best_for_entry.items():
            per_entry_scores[entry_id].append(score)

    out: Dict[int, float] = {}
    for entry_id, values in per_entry_scores.items():
        if not values:
            out[entry_id] = 0.0
        else:
            out[entry_id] = float(np.mean(values))
    return out
