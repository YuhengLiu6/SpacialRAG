import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from spatial_rag.graph_builder import build_graph_payload
from spatial_rag.object_index import load_object_db


@dataclass(frozen=True)
class PairGTRecord:
    pair_id: str
    db_dir: str
    obj_a_id: int
    obj_b_id: int
    is_same_instance: bool
    split: str = "dev"
    notes: str = ""
    annotator: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class PairScoreRecord:
    pair_id: str
    db_dir: str
    obj_a_id: int
    obj_b_id: int
    is_same_instance: bool
    split: str
    short_cosine: float
    long_cosine: float
    graph_cosine: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_db_key(db_dir: str) -> str:
    text = str(db_dir or "").strip()
    if not text:
        return ""
    return Path(text).name or Path(text).as_posix()


def _normalize_pair_ids(obj_a_id: int, obj_b_id: int) -> Tuple[int, int]:
    a = int(obj_a_id)
    b = int(obj_b_id)
    if a == b:
        raise ValueError(f"Ground-truth pair cannot reference the same object twice: {a}")
    return (a, b) if a < b else (b, a)


def _object_row_map(meta_rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for idx, row in enumerate(meta_rows):
        obj_id = int(row.get("object_global_id", idx))
        out[obj_id] = dict(row)
    return out


def _object_id_to_row_index(meta_rows: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for idx, row in enumerate(meta_rows):
        out[int(row.get("object_global_id", idx))] = idx
    return out


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
    id_to_row = _object_id_to_row_index(meta_rows)
    if not id_to_row:
        return np.zeros((0, emb.shape[1]), dtype=np.float32)
    max_obj_id = max(id_to_row.keys())
    dense = np.zeros((max_obj_id + 1, emb.shape[1]), dtype=np.float32)
    for obj_id, row_idx in id_to_row.items():
        dense[int(obj_id)] = emb[int(row_idx)]
    return dense


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


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


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _object_snippet(row: Mapping[str, Any], use_long_descriptions: bool = True) -> str:
    label = _safe_text(row.get("label") or row.get("object_class") or "unknown")
    desc = _safe_text(row.get("description"))
    long_desc = _safe_text(row.get("long_form_open_description"))
    if use_long_descriptions and long_desc:
        detail = long_desc
    else:
        detail = desc
    if detail:
        return f"{label}: {detail}"
    return label or "unknown"


def _rank_average(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for i in range(start, end):
            ranks[indexed[i][0]] = avg_rank
        start = end
    return ranks


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(int(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _rank_average(list(scores))
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if int(label) == 1)
    auc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(int(label) for label in labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if int(label) != 1:
            continue
        tp += 1
        precision_sum += tp / rank
    return float(precision_sum / positives)


def _threshold_stats(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        pred = float(score) >= float(threshold)
        actual = int(label) == 1
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif (not pred) and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _best_threshold(labels: Sequence[int], scores: Sequence[float]) -> Optional[Dict[str, float]]:
    if not scores:
        return None
    thresholds = sorted({float(score) for score in scores})
    if thresholds:
        thresholds = [thresholds[0] - 1e-6] + thresholds
    best: Optional[Dict[str, float]] = None
    for threshold in thresholds:
        stats = _threshold_stats(labels, scores, threshold=threshold)
        if best is None:
            best = stats
            continue
        if stats["f1"] > best["f1"]:
            best = stats
            continue
        if stats["f1"] == best["f1"] and stats["accuracy"] > best["accuracy"]:
            best = stats
            continue
        if stats["f1"] == best["f1"] and stats["accuracy"] == best["accuracy"] and stats["threshold"] < best["threshold"]:
            best = stats
    return best


def _retrieval_metrics(pair_scores: Sequence[PairScoreRecord], field_name: str) -> Dict[str, Optional[float]]:
    adjacency: Dict[Tuple[str, int], Dict[int, float]] = {}
    positives: Dict[Tuple[str, int], set] = {}
    for score in pair_scores:
        score_value = float(getattr(score, field_name))
        key_a = (score.db_dir, int(score.obj_a_id))
        key_b = (score.db_dir, int(score.obj_b_id))
        adjacency.setdefault(key_a, {})[int(score.obj_b_id)] = score_value
        adjacency.setdefault(key_b, {})[int(score.obj_a_id)] = score_value
        if score.is_same_instance:
            positives.setdefault(key_a, set()).add(int(score.obj_b_id))
            positives.setdefault(key_b, set()).add(int(score.obj_a_id))

    anchors = sorted(positives.keys())
    if not anchors:
        return {"recall_at_1": None, "recall_at_5": None, "mrr": None, "num_anchors": 0}

    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_sum = 0.0
    for db_key, obj_id in anchors:
        candidates = adjacency.get((db_key, obj_id), {})
        ranked = sorted(candidates.items(), key=lambda item: (-float(item[1]), int(item[0])))
        positive_ids = positives[(db_key, obj_id)]
        first_rank: Optional[int] = None
        for rank, (cand_obj_id, _score) in enumerate(ranked, start=1):
            if int(cand_obj_id) in positive_ids:
                first_rank = rank
                break
        if first_rank is None:
            continue
        if first_rank <= 1:
            hits_at_1 += 1
        if first_rank <= 5:
            hits_at_5 += 1
        reciprocal_sum += 1.0 / float(first_rank)

    denom = float(len(anchors))
    return {
        "recall_at_1": hits_at_1 / denom if denom else None,
        "recall_at_5": hits_at_5 / denom if denom else None,
        "mrr": reciprocal_sum / denom if denom else None,
        "num_anchors": int(len(anchors)),
    }


def _representation_metrics(pair_scores: Sequence[PairScoreRecord], field_name: str) -> Dict[str, Any]:
    labels = [1 if item.is_same_instance else 0 for item in pair_scores]
    scores = [float(getattr(item, field_name)) for item in pair_scores]
    pos_scores = [score for score, label in zip(scores, labels) if label == 1]
    neg_scores = [score for score, label in zip(scores, labels) if label == 0]
    best_threshold = _best_threshold(labels, scores)
    return {
        "num_pairs": int(len(pair_scores)),
        "num_positive_pairs": int(sum(labels)),
        "num_negative_pairs": int(len(labels) - sum(labels)),
        "positive_mean_cosine": float(np.mean(pos_scores)) if pos_scores else None,
        "negative_mean_cosine": float(np.mean(neg_scores)) if neg_scores else None,
        "positive_median_cosine": float(np.median(pos_scores)) if pos_scores else None,
        "negative_median_cosine": float(np.median(neg_scores)) if neg_scores else None,
        "roc_auc": _roc_auc(labels, scores),
        "pr_auc": _average_precision(labels, scores),
        "best_threshold": best_threshold,
        "retrieval": _retrieval_metrics(pair_scores, field_name=field_name),
    }


def load_object_pair_ground_truth(path: str, split: Optional[str] = None) -> List[PairGTRecord]:
    rows = _load_jsonl(Path(path))
    records: List[PairGTRecord] = []
    seen: Dict[Tuple[str, int, int], PairGTRecord] = {}
    for idx, row in enumerate(rows):
        missing = {key for key in ("pair_id", "db_dir", "obj_a_id", "obj_b_id", "is_same_instance") if key not in row}
        if missing:
            raise ValueError(f"Ground-truth row {idx} missing fields: {sorted(missing)}")
        obj_a_id, obj_b_id = _normalize_pair_ids(row["obj_a_id"], row["obj_b_id"])
        record = PairGTRecord(
            pair_id=str(row["pair_id"]),
            db_dir=str(row["db_dir"]),
            obj_a_id=obj_a_id,
            obj_b_id=obj_b_id,
            is_same_instance=bool(row["is_same_instance"]),
            split=str(row.get("split") or "dev"),
            notes=str(row.get("notes") or ""),
            annotator=str(row.get("annotator") or ""),
            created_at=str(row.get("created_at") or ""),
        )
        if split is not None and record.split != str(split):
            continue
        key = (_normalize_db_key(record.db_dir), int(record.obj_a_id), int(record.obj_b_id))
        existing = seen.get(key)
        if existing is not None:
            if bool(existing.is_same_instance) != bool(record.is_same_instance):
                raise ValueError(f"Conflicting duplicate ground-truth entries for {key}")
            continue
        seen[key] = record
        records.append(record)
    return records


def build_graph_context_strings(
    db_dir: str,
    graph_payload: Optional[Mapping[str, Any]] = None,
    same_node_limit: int = 8,
    direction_limit: int = 4,
    use_long_descriptions: bool = True,
) -> Dict[int, str]:
    payload = dict(graph_payload) if graph_payload is not None else build_graph_payload(db_dir)
    places = {str(row["place_id"]): dict(row) for row in payload.get("places", [])}
    objects = [dict(row) for row in payload.get("objects", [])]
    place_to_objects: Dict[str, List[Dict[str, Any]]] = {}
    for row in objects:
        place_to_objects.setdefault(str(row["place_id"]), []).append(row)
    for rows in place_to_objects.values():
        rows.sort(key=lambda item: (str(item.get("view_id") or ""), int(item.get("object_global_id", -1))))

    direction_to_places: Dict[Tuple[str, str], List[str]] = {}
    for edge in payload.get("direction_edges", []):
        key = (str(edge["source_place_id"]), str(edge["relation_type"]))
        direction_to_places.setdefault(key, []).append(str(edge["target_place_id"]))
    for targets in direction_to_places.values():
        targets.sort()

    contexts: Dict[int, str] = {}
    for row in sorted(objects, key=lambda item: int(item["object_global_id"])):
        obj_id = int(row["object_global_id"])
        place_id = str(row["place_id"])
        place = places.get(place_id, {})
        self_parts = [
            f"label={_safe_text(row.get('label')) or 'unknown'}",
            f"description={_safe_text(row.get('description')) or 'unknown'}",
        ]
        long_desc = _safe_text(row.get("long_form_open_description"))
        if use_long_descriptions:
            self_parts.append(f"long={long_desc or 'unknown'}")
        self_parts.extend(
            [
                f"laterality={_safe_text(row.get('laterality')) or 'unknown'}",
                f"distance_bin={_safe_text(row.get('distance_bin')) or 'unknown'}",
                f"verticality={_safe_text(row.get('verticality')) or 'unknown'}",
                f"distance_from_camera_m={row.get('distance_from_camera_m') if row.get('distance_from_camera_m') is not None else 'unknown'}",
                f"object_orientation_deg={row.get('object_orientation_deg') if row.get('object_orientation_deg') is not None else 'unknown'}",
            ]
        )
        same_node_candidates = [
            _object_snippet(other, use_long_descriptions=use_long_descriptions)
            for other in place_to_objects.get(place_id, [])
            if int(other["object_global_id"]) != obj_id
        ][: max(0, int(same_node_limit))]
        directional_sections: List[str] = []
        for relation_type in ("NORTH_OF", "EAST_OF", "SOUTH_OF", "WEST_OF"):
            neighbor_labels: List[str] = []
            for neighbor_place_id in direction_to_places.get((place_id, relation_type), []):
                for other in place_to_objects.get(neighbor_place_id, []):
                    neighbor_labels.append(_object_snippet(other, use_long_descriptions=False))
                    if len(neighbor_labels) >= int(direction_limit):
                        break
                if len(neighbor_labels) >= int(direction_limit):
                    break
            directional_sections.append(
                f"{relation_type.lower().replace('_of', '')}: {', '.join(neighbor_labels) if neighbor_labels else 'none'}"
            )

        place_text = (
            f"place={place_id} | room_function={_safe_text(place.get('room_function')) or 'unknown'} | "
            f"view_type={_safe_text(place.get('view_type')) or 'unknown'}"
        )
        same_node_text = f"same_node: {', '.join(same_node_candidates) if same_node_candidates else 'none'}"
        contexts[obj_id] = " | ".join(
            [
                f"self: {'; '.join(self_parts)}",
                place_text,
                same_node_text,
                *directional_sections,
            ]
        )
    return contexts


def embed_graph_contexts(
    context_by_obj_id: Mapping[int, str],
    embedder=None,
) -> np.ndarray:
    if not context_by_obj_id:
        return np.zeros((0, 0), dtype=np.float32)
    if embedder is None:
        from spatial_rag.embedder import Embedder

        embedder = Embedder()

    obj_ids = sorted(int(obj_id) for obj_id in context_by_obj_id.keys())
    max_obj_id = max(obj_ids)
    rows: List[Optional[np.ndarray]] = [None] * (max_obj_id + 1)
    dim: Optional[int] = None
    for obj_id in obj_ids:
        vec = np.asarray(embedder.embed_text(str(context_by_obj_id[obj_id])), dtype=np.float32).reshape(-1)
        if dim is None:
            dim = int(vec.shape[0])
        rows[obj_id] = vec
    if dim is None:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.zeros((max_obj_id + 1, dim), dtype=np.float32)
    for idx, vec in enumerate(rows):
        if vec is None:
            continue
        out[idx] = vec
    return _l2_normalize_rows(out)


def compute_pairwise_cosines(
    pair_gt: Sequence[PairGTRecord],
    short_emb: np.ndarray,
    long_emb: np.ndarray,
    graph_emb: np.ndarray,
) -> List[PairScoreRecord]:
    short_norm = _l2_normalize_rows(short_emb.astype(np.float32))
    long_norm = _l2_normalize_rows(long_emb.astype(np.float32))
    graph_norm = _l2_normalize_rows(graph_emb.astype(np.float32))
    results: List[PairScoreRecord] = []
    for pair in pair_gt:
        max_required = max(int(pair.obj_a_id), int(pair.obj_b_id))
        for name, arr in (("short", short_norm), ("long", long_norm), ("graph", graph_norm)):
            if max_required >= int(arr.shape[0]):
                raise KeyError(f"Pair {pair.pair_id} references object id {max_required} outside {name} embedding range")
        a = int(pair.obj_a_id)
        b = int(pair.obj_b_id)
        results.append(
            PairScoreRecord(
                pair_id=pair.pair_id,
                db_dir=pair.db_dir,
                obj_a_id=a,
                obj_b_id=b,
                is_same_instance=pair.is_same_instance,
                split=pair.split,
                short_cosine=float(np.dot(short_norm[a], short_norm[b])),
                long_cosine=float(np.dot(long_norm[a], long_norm[b])),
                graph_cosine=float(np.dot(graph_norm[a], graph_norm[b])),
            )
        )
    return results


def summarize_similarity_metrics(pair_scores: Sequence[PairScoreRecord]) -> Dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "num_pairs": int(len(pair_scores)),
        "short": _representation_metrics(pair_scores, field_name="short_cosine"),
        "long": _representation_metrics(pair_scores, field_name="long_cosine"),
        "graph": _representation_metrics(pair_scores, field_name="graph_cosine"),
    }


def export_pair_artifacts(
    pair_scores: Sequence[PairScoreRecord],
    db_dir: str,
    output_dir: str,
    graph_context_by_obj_id: Optional[Mapping[int, str]] = None,
    max_examples_per_bucket: int = 20,
) -> Dict[str, Any]:
    root = Path(output_dir)
    examples_root = root / "examples"
    examples_root.mkdir(parents=True, exist_ok=True)

    meta_path = Path(db_dir) / "object_meta.jsonl"
    meta_rows = _load_jsonl(meta_path)
    meta_by_id = _object_row_map(meta_rows)

    def choose_score(row: PairScoreRecord) -> float:
        return float(row.graph_cosine)

    positives = [row for row in pair_scores if row.is_same_instance]
    negatives = [row for row in pair_scores if not row.is_same_instance]
    buckets = {
        "positives_high_similarity": sorted(positives, key=choose_score, reverse=True)[: max_examples_per_bucket],
        "positives_low_similarity": sorted(positives, key=choose_score)[: max_examples_per_bucket],
        "negatives_high_similarity": sorted(negatives, key=choose_score, reverse=True)[: max_examples_per_bucket],
        "negatives_low_similarity": sorted(negatives, key=choose_score)[: max_examples_per_bucket],
    }

    bucket_counts: Dict[str, int] = {}
    for bucket, items in buckets.items():
        bucket_dir = examples_root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        bucket_counts[bucket] = len(items)
        for row in items:
            pair_dir = bucket_dir / row.pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            a_meta = meta_by_id.get(int(row.obj_a_id), {})
            b_meta = meta_by_id.get(int(row.obj_b_id), {})
            manifest = {
                "pair_score": asdict(row),
                "object_a_meta": a_meta,
                "object_b_meta": b_meta,
                "graph_context_a": str(graph_context_by_obj_id.get(int(row.obj_a_id), "")) if graph_context_by_obj_id else "",
                "graph_context_b": str(graph_context_by_obj_id.get(int(row.obj_b_id), "")) if graph_context_by_obj_id else "",
            }
            (pair_dir / "pair_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
            for prefix, meta in (("a", a_meta), ("b", b_meta)):
                file_name = str(meta.get("file_name") or "").strip()
                if not file_name:
                    continue
                src = Path(file_name)
                if not src.is_absolute():
                    src = Path(db_dir) / src
                if not src.exists():
                    continue
                dst = pair_dir / f"{prefix}_{src.name}"
                shutil.copy2(src, dst)
    return {"examples_root": str(examples_root), "bucket_counts": bucket_counts}


def _write_pairs_csv(path: Path, pair_scores: Sequence[PairScoreRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(pair_scores[0]).keys()) if pair_scores else [
            "pair_id",
            "db_dir",
            "obj_a_id",
            "obj_b_id",
            "is_same_instance",
            "split",
            "short_cosine",
            "long_cosine",
            "graph_cosine",
        ])
        writer.writeheader()
        for row in pair_scores:
            writer.writerow(asdict(row))


def _write_summary_markdown(path: Path, report: Mapping[str, Any]) -> None:
    metrics = report.get("metrics", {})
    lines = [
        "# Object Instance Similarity Evaluation",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- DB: `{report.get('db_dir')}`",
        f"- GT file: `{report.get('gt_pairs_path')}`",
        f"- Number of pairs: `{report.get('num_pairs')}`",
        "",
        "| Representation | Pos Mean | Neg Mean | ROC-AUC | PR-AUC | Best F1 | Recall@1 | Recall@5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("short", "long", "graph"):
        item = metrics.get(key, {})
        best = item.get("best_threshold") or {}
        retrieval = item.get("retrieval") or {}
        lines.append(
            "| {name} | {pos_mean} | {neg_mean} | {roc} | {pr} | {f1} | {r1} | {r5} | {mrr} |".format(
                name=key,
                pos_mean=f"{item.get('positive_mean_cosine'):.4f}" if item.get("positive_mean_cosine") is not None else "n/a",
                neg_mean=f"{item.get('negative_mean_cosine'):.4f}" if item.get("negative_mean_cosine") is not None else "n/a",
                roc=f"{item.get('roc_auc'):.4f}" if item.get("roc_auc") is not None else "n/a",
                pr=f"{item.get('pr_auc'):.4f}" if item.get("pr_auc") is not None else "n/a",
                f1=f"{best.get('f1'):.4f}" if best.get("f1") is not None else "n/a",
                r1=f"{retrieval.get('recall_at_1'):.4f}" if retrieval.get("recall_at_1") is not None else "n/a",
                r5=f"{retrieval.get('recall_at_5'):.4f}" if retrieval.get("recall_at_5") is not None else "n/a",
                mrr=f"{retrieval.get('mrr'):.4f}" if retrieval.get("mrr") is not None else "n/a",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_representation_embeddings(
    db_dir: str,
    graph_context_by_obj_id: Optional[Mapping[int, str]] = None,
    text_modes: Sequence[str] = ("short", "long", "graph"),
    use_long_descriptions: bool = True,
) -> Dict[str, np.ndarray]:
    modes = {str(mode).strip().lower() for mode in text_modes}
    embeddings: Dict[str, np.ndarray] = {}
    if "short" in modes:
        loaded = load_object_db(db_dir, text_mode="short")
        if loaded is None:
            raise FileNotFoundError(f"Missing short object DB artifacts in {db_dir}")
        meta_rows, short_emb, _entry_map = loaded
        embeddings["short"] = _l2_normalize_rows(_dense_embeddings_by_object_id(meta_rows, short_emb))
    if "long" in modes:
        loaded = load_object_db(db_dir, text_mode="long")
        if loaded is None:
            raise FileNotFoundError(f"Missing long object DB artifacts in {db_dir}")
        meta_rows, long_emb, _entry_map = loaded
        embeddings["long"] = _l2_normalize_rows(_dense_embeddings_by_object_id(meta_rows, long_emb))
    if "graph" in modes:
        contexts = graph_context_by_obj_id or build_graph_context_strings(
            db_dir=db_dir,
            use_long_descriptions=use_long_descriptions,
        )
        embeddings["graph"] = embed_graph_contexts(contexts)
    return embeddings


def _select_db_pairs(pair_gt: Sequence[PairGTRecord], db_dir: str) -> List[PairGTRecord]:
    target_key = _normalize_db_key(db_dir)
    selected = [pair for pair in pair_gt if _normalize_db_key(pair.db_dir) == target_key]
    if not selected:
        raise ValueError(f"No ground-truth pairs matched db_dir={db_dir}")
    return selected


def _prepare_pair_scores(
    db_dir: str,
    gt_pairs_path: str,
    split: Optional[str],
    text_modes: Sequence[str],
    use_long_descriptions: bool,
) -> Tuple[List[PairGTRecord], List[PairScoreRecord], Dict[int, str]]:
    pair_gt_all = load_object_pair_ground_truth(gt_pairs_path, split=split)
    pair_gt = _select_db_pairs(pair_gt_all, db_dir=db_dir)
    graph_context_by_obj_id = build_graph_context_strings(
        db_dir=db_dir,
        use_long_descriptions=use_long_descriptions,
    )
    embeddings = _load_representation_embeddings(
        db_dir=db_dir,
        graph_context_by_obj_id=graph_context_by_obj_id,
        text_modes=text_modes,
        use_long_descriptions=use_long_descriptions,
    )
    short_emb = embeddings.get("short")
    long_emb = embeddings.get("long")
    graph_emb = embeddings.get("graph")
    if short_emb is None or long_emb is None or graph_emb is None:
        raise ValueError("First implementation requires short, long, and graph embeddings together")
    pair_scores = compute_pairwise_cosines(pair_gt, short_emb=short_emb, long_emb=long_emb, graph_emb=graph_emb)
    return pair_gt, pair_scores, graph_context_by_obj_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate same-instance object similarity using VLM descriptions and GraphRAG context.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory containing object_meta.jsonl and embeddings")
    parser.add_argument("--gt_pairs", type=str, required=True, help="JSONL file with manual same-instance pair annotations")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for evaluation report and artifacts")
    parser.add_argument("--text_modes", type=str, default="short,long,graph", help="Comma-separated list of representations to evaluate")
    parser.add_argument("--split", type=str, default=None, help="Optional split filter, e.g. dev/test")
    parser.add_argument("--max_examples_per_bucket", type=int, default=20, help="How many example pairs to export per bucket")
    parser.add_argument("--use_long_descriptions", type=str, default="true", help="Whether graph context should include long descriptions")
    return parser.parse_args()


def _str_to_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main() -> None:
    args = _parse_args()
    text_modes = [mode.strip().lower() for mode in str(args.text_modes).split(",") if mode.strip()]
    use_long_descriptions = _str_to_bool(args.use_long_descriptions)
    pair_gt, pair_scores, graph_context_by_obj_id = _prepare_pair_scores(
        db_dir=args.db_dir,
        gt_pairs_path=args.gt_pairs,
        split=args.split,
        text_modes=text_modes,
        use_long_descriptions=use_long_descriptions,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": _now_iso(),
        "db_dir": args.db_dir,
        "gt_pairs_path": args.gt_pairs,
        "split": args.split,
        "num_pairs": len(pair_scores),
        "pair_gt": [asdict(item) for item in pair_gt],
        "metrics": summarize_similarity_metrics(pair_scores),
    }
    artifact_summary = export_pair_artifacts(
        pair_scores=pair_scores,
        db_dir=args.db_dir,
        output_dir=str(output_dir),
        graph_context_by_obj_id=graph_context_by_obj_id,
        max_examples_per_bucket=args.max_examples_per_bucket,
    )
    report["artifacts"] = artifact_summary

    _write_pairs_csv(output_dir / "pairs.csv", pair_scores)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_summary_markdown(output_dir / "summary.md", report)
    print(json.dumps({"output_dir": str(output_dir), "num_pairs": len(pair_scores), "metrics": report["metrics"]}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
