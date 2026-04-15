from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from spatial_rag.config import OCCLUSION_REWEIGHT_B, OCCLUSION_REWEIGHT_W1, OCCLUSION_REWEIGHT_W2
from spatial_rag.object_canonicalizer import UNKNOWN_TEXT_TOKEN
from spatial_rag.occlusion_scoring import (
    compute_reweighted_detection_score_from_penalty,
    map_occlusion_level_to_penalty,
    normalize_occlusion_level,
)
from spatial_rag.spatial_db_builder import (
    _OBJECT_R_SCORES_COLUMNS,
    _OBJECT_R_SCORES_PRE_THRESHOLD_COLUMNS,
    _build_object_object_relations,
    _build_view_object_relations,
    _bbox_xyxy_ints_from_row,
    _ensure_metadata_record_attribute,
    _resolve_existing_path,
    _save_faiss_index,
    _safe_filename_token,
    _write_csv_rows,
    _write_jsonl,
)


_FILTERED_OBJECT_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "object_global_id",
    "entry_id",
    "frame_id",
    "object_local_id",
    "label",
    "reweighted_detection_score_r",
    "threshold",
    "file_name",
    "source_image",
    "source_crop_path",
    "export_status",
    "export_source",
    "export_path",
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _str_to_bool(value: str) -> bool:
    token = _safe_text(value).lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_number_token(value: Optional[float]) -> str:
    if value is None:
        return "none"
    token = f"{float(value):g}"
    return token.replace("-", "neg_").replace(".", "p")


def _config_token(w1: float, w2: float, b: float, threshold: Optional[float]) -> str:
    return (
        f"w1_{_normalize_number_token(w1)}_"
        f"w2_{_normalize_number_token(w2)}_"
        f"b_{_normalize_number_token(b)}_"
        f"t_{_normalize_number_token(threshold)}"
    )


def _parse_float_list(value: Any, *, allow_none: bool = False) -> List[Optional[float]]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        parts = list(value)
    else:
        parts = [value]

    out: List[Optional[float]] = []
    for item in parts:
        token = _safe_text(item).lower()
        if allow_none and token in {"none", "null", ""}:
            out.append(None)
            continue
        parsed = _safe_float(item)
        if parsed is None:
            raise ValueError(f"Invalid numeric value: {item!r}")
        out.append(parsed)
    if not out:
        raise ValueError("Expected at least one numeric value")
    return out


def _parse_selected_tokens(value: Any) -> List[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        parts = [_safe_text(item) for item in value if _safe_text(item)]
    else:
        parts = [_safe_text(value)]
    return parts


def _now_stamp() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _make_run_output_dir(base_dir: Path) -> Path:
    stamp = _now_stamp()
    candidate = base_dir / stamp
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _safe_path_token(value: Any, default: str = "unknown") -> str:
    return _safe_filename_token(value, default=default)


def _parse_bbox_xyxy(value: Any) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in list(value)[:4]]
    text = _safe_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    try:
        return [float(v) for v in parsed[:4]]
    except Exception:
        return []


def _filtered_object_id_token(row: Mapping[str, Any]) -> str:
    object_global_id = row.get("object_global_id")
    try:
        if object_global_id is not None and object_global_id != "":
            return str(int(object_global_id))
    except Exception:
        pass
    entry_id = _safe_int(row.get("entry_id"), -1)
    object_local_id = _safe_path_token(row.get("object_local_id"), default="unknown")
    return f"{entry_id}_{object_local_id}"


def _score_filename_token(value: Any) -> str:
    score = _safe_float(value)
    if score is None:
        return "score_unknown"
    token = f"{float(score):.4f}".rstrip("0").rstrip(".")
    token = token.replace("-", "neg_").replace(".", "p")
    return f"score_{token or '0'}"


def _filtered_object_filename(row: Mapping[str, Any]) -> str:
    object_id = _filtered_object_id_token(row)
    label = _safe_path_token(row.get("label"), default="unknown")
    occlusion_level = _safe_path_token(
        normalize_occlusion_level(row.get("occlusion_level"), default="uncertain"),
        default="uncertain",
    )
    score = _score_filename_token(row.get("reweighted_detection_score_r"))
    return f"{object_id}_{label}_{occlusion_level}_{score}.jpg"


def _row_for_filtered_export(
    row: Mapping[str, Any],
    *,
    base_object_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = dict(row)
    source_index = row.get("source_index")
    try:
        if source_index is not None:
            base_row = dict(base_object_rows[int(source_index)])
            base_row.update(payload)
            payload = base_row
    except Exception:
        pass
    payload["bbox_xyxy"] = _parse_bbox_xyxy(payload.get("bbox_xyxy"))
    return payload


def _export_filtered_objects(
    *,
    db_root: Path,
    base_object_rows: Sequence[Mapping[str, Any]],
    filtered_rows: Sequence[Mapping[str, Any]],
    export_dir: Path,
    threshold: Optional[float],
) -> Dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    exported_count = 0
    copied_count = 0
    regenerated_count = 0
    skipped_count = 0

    for raw_row in list(filtered_rows):
        row = _row_for_filtered_export(raw_row, base_object_rows=base_object_rows)
        target_path = export_dir / _filtered_object_filename(row)
        crop_source_path = _resolve_existing_path(db_root, row.get("crop_path"))
        image_path = _resolve_existing_path(db_root, row.get("file_name"))
        source_kind = ""
        export_status = "skipped"

        if crop_source_path is not None:
            crop_image = cv2.imread(str(crop_source_path), cv2.IMREAD_COLOR)
            if crop_image is not None and cv2.imwrite(str(target_path), crop_image):
                exported_count += 1
                copied_count += 1
                export_status = "exported"
                source_kind = "existing_crop_path"
            else:
                crop_source_path = None

        if crop_source_path is None:
            bbox = _bbox_xyxy_ints_from_row(row)
            if bbox is not None and image_path is not None:
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is not None:
                    h, w = image_bgr.shape[:2]
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(x1 + 1, min(w, x2))
                    y2 = max(y1 + 1, min(h, y2))
                    crop_bgr = image_bgr[y1:y2, x1:x2]
                    if crop_bgr.size > 0 and cv2.imwrite(str(target_path), crop_bgr):
                        exported_count += 1
                        regenerated_count += 1
                        export_status = "exported"
                        source_kind = "reconstructed_from_bbox"
            if not source_kind:
                skipped_count += 1
                if image_path is None:
                    export_status = "missing_source_image"
                elif _bbox_xyxy_ints_from_row(row) is None:
                    export_status = "missing_crop_and_bbox"
                else:
                    export_status = "crop_write_failed"
                source_kind = "unavailable"

        manifest_rows.append(
            {
                "object_global_id": row.get("object_global_id"),
                "entry_id": row.get("entry_id"),
                "frame_id": row.get("frame_id"),
                "object_local_id": row.get("object_local_id"),
                "label": row.get("label"),
                "reweighted_detection_score_r": row.get("reweighted_detection_score_r"),
                "threshold": threshold,
                "file_name": row.get("file_name"),
                "source_image": str(image_path) if image_path is not None else "",
                "source_crop_path": str(crop_source_path) if crop_source_path is not None else "",
                "export_status": export_status,
                "export_source": source_kind,
                "export_path": str(target_path) if export_status == "exported" else "",
            }
        )

    manifest_path = export_dir / "manifest.csv"
    _write_csv_rows(manifest_path, _FILTERED_OBJECT_MANIFEST_COLUMNS, manifest_rows)
    return {
        "dir": str(export_dir),
        "manifest_path": str(manifest_path),
        "manifest_count": int(len(manifest_rows)),
        "filtered_object_count": int(len(filtered_rows)),
        "exported_count": int(exported_count),
        "copied_count": int(copied_count),
        "regenerated_count": int(regenerated_count),
        "skipped_count": int(skipped_count),
    }


def _make_embedder():
    from spatial_rag.embedder import Embedder

    return Embedder()


def _entry_id_from_meta(row: Mapping[str, Any], index: int) -> int:
    if row.get("id") is not None:
        return _safe_int(row.get("id"), index)
    if row.get("entry_id") is not None:
        return _safe_int(row.get("entry_id"), index)
    return int(index)


def _route_for_object_row(row: Mapping[str, Any]) -> str:
    return "vlm_fallback" if _safe_text(row.get("geometry_source")).lower() == "vlm_fallback" else "geometry"


def _normalize_path_for_variant(base_root: Path, raw_path: Any) -> Any:
    raw = _safe_text(raw_path)
    if not raw:
        return raw_path
    path = Path(raw)
    if path.is_absolute():
        try:
            return str(path.relative_to(base_root).as_posix())
        except Exception:
            return str(path)

    candidate_in_db = base_root / path
    if candidate_in_db.exists():
        return str(path.as_posix())

    candidate_from_repo = base_root.parent / path
    if candidate_from_repo.exists():
        try:
            return str(candidate_from_repo.relative_to(base_root).as_posix())
        except Exception:
            pass

    if len(path.parts) > 1 and path.parts[0] == base_root.name:
        return str(Path(*path.parts[1:]).as_posix())

    return str(path.as_posix())


def _symlink_dir(source: Path, dest: Path) -> None:
    if not source.exists():
        return
    if dest.exists() or dest.is_symlink():
        return
    relative_target = os.path.relpath(str(source), start=str(dest.parent))
    dest.symlink_to(relative_target, target_is_directory=True)


def _canonical_base_is_filtered(
    *,
    build_report: Mapping[str, Any],
    object_rows: Sequence[Mapping[str, Any]],
    pre_threshold_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if bool(build_report.get("r_threshold_enabled")):
        return True
    if pre_threshold_rows and len(pre_threshold_rows) > len(list(object_rows or [])):
        return True
    return False


def _build_analysis_rows_from_objects(object_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(object_rows):
        rows.append(
            {
                "source_index": int(index),
                "object_global_id": row.get("object_global_id"),
                "entry_id": _safe_int(row.get("entry_id"), -1),
                "frame_id": _safe_int(row.get("frame_id"), -1),
                "file_name": _safe_text(row.get("file_name")),
                "object_local_id": _safe_text(row.get("object_local_id")),
                "route": _route_for_object_row(row),
                "label": _safe_text(row.get("label")) or "unknown",
                "detector_confidence": row.get("detector_confidence"),
                "object_confidence": row.get("object_confidence"),
                "occlusion_level": row.get("occlusion_level"),
            }
        )
    return rows


def _build_analysis_rows_from_prethreshold(pre_threshold_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in pre_threshold_rows:
        rows.append(
            {
                "source_index": None,
                "object_global_id": None,
                "entry_id": _safe_int(row.get("entry_id"), -1),
                "frame_id": _safe_int(row.get("frame_id"), -1),
                "file_name": _safe_text(row.get("file_name")),
                "object_local_id": _safe_text(row.get("object_local_id")),
                "route": "vlm_fallback" if _safe_text(row.get("object_route")) == "vlm_fallback" else "geometry",
                "label": _safe_text(row.get("label")) or "unknown",
                "detector_confidence": row.get("detector_confidence"),
                "object_confidence": row.get("object_confidence"),
                "occlusion_level": row.get("occlusion_level"),
            }
        )
    return rows


def _score_analysis_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    w1: float,
    w2: float,
    b: float,
    threshold: Optional[float],
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for row in rows:
        normalized_level = normalize_occlusion_level(row.get("occlusion_level"), default="uncertain")
        raw_penalty = _safe_float(row.get("occlusion_penalty_p_o"))
        penalty = float(raw_penalty) if raw_penalty is not None else float(map_occlusion_level_to_penalty(normalized_level))
        detector_confidence = _safe_float(row.get("detector_confidence"))
        object_confidence = _safe_float(row.get("object_confidence"))
        confidence = detector_confidence if detector_confidence is not None else object_confidence
        confidence_value = float(confidence or 0.0)
        score = float(
            compute_reweighted_detection_score_from_penalty(
                confidence_value,
                penalty,
                w1=float(w1),
                w2=float(w2),
                b=float(b),
            )
        )
        route = _safe_text(row.get("route")) or "geometry"
        keep = bool(route != "geometry" or threshold is None or score >= float(threshold))
        payload = dict(row)
        payload.update(
            {
                "occlusion_level": normalized_level,
                "occlusion_penalty_p_o": penalty,
                "reweighted_detection_score_r": score,
                "threshold": None if threshold is None else float(threshold),
                "keep": keep,
            }
        )
        scored.append(payload)
    return scored


def _sorted_counter_dict(counter: Counter[str]) -> Dict[str, int]:
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))
    return {key: int(value) for key, value in ordered}


def _summarize_scored_rows(
    *,
    meta_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
    base_total_objects: int,
    exportable: bool,
    analysis_only: bool,
    config_token: str,
    w1: float,
    w2: float,
    b: float,
    threshold: Optional[float],
) -> Dict[str, Any]:
    kept_rows = [dict(row) for row in scored_rows if bool(row.get("keep"))]
    filtered_rows = [dict(row) for row in scored_rows if not bool(row.get("keep"))]
    geometry_rows = [dict(row) for row in scored_rows if _safe_text(row.get("route")) == "geometry"]
    fallback_rows = [dict(row) for row in scored_rows if _safe_text(row.get("route")) == "vlm_fallback"]
    geometry_kept_rows = [dict(row) for row in kept_rows if _safe_text(row.get("route")) == "geometry"]
    fallback_kept_rows = [dict(row) for row in kept_rows if _safe_text(row.get("route")) == "vlm_fallback"]

    kept_by_entry: Dict[int, int] = Counter(_safe_int(row.get("entry_id"), -1) for row in kept_rows)
    empty_frame_count = 0
    for index, row in enumerate(meta_rows):
        entry_id = _entry_id_from_meta(row, index)
        if kept_by_entry.get(entry_id, 0) == 0:
            empty_frame_count += 1

    filtered_labels = Counter(_safe_text(row.get("label")) or "unknown" for row in filtered_rows)
    keep_rate = float(len(kept_rows) / max(len(scored_rows), 1))
    return {
        "config_token": config_token,
        "w1": float(w1),
        "w2": float(w2),
        "b": float(b),
        "threshold": None if threshold is None else float(threshold),
        "base_total_objects": int(base_total_objects),
        "candidate_total_objects": int(len(scored_rows)),
        "kept_total_objects": int(len(kept_rows)),
        "dropped_total_objects": int(len(filtered_rows)),
        "total_object_delta_vs_base": int(len(kept_rows) - int(base_total_objects)),
        "geometry_candidate_count": int(len(geometry_rows)),
        "geometry_kept_count": int(len(geometry_kept_rows)),
        "geometry_filtered_count": int(len(geometry_rows) - len(geometry_kept_rows)),
        "fallback_candidate_count": int(len(fallback_rows)),
        "fallback_kept_count": int(len(fallback_kept_rows)),
        "empty_frame_count": int(empty_frame_count),
        "keep_rate": keep_rate,
        "filtered_label_counts": _sorted_counter_dict(filtered_labels),
        "analysis_only": bool(analysis_only),
        "exportable": bool(exportable),
    }


def _current_frame_text_short(meta_row: Mapping[str, Any]) -> str:
    return _safe_text(meta_row.get("text_input_for_clip_short")) or _safe_text(meta_row.get("frame_text_short")) or _safe_text(
        meta_row.get("text")
    )


def _current_frame_text_long(meta_row: Mapping[str, Any]) -> str:
    return _safe_text(meta_row.get("text_input_for_clip_long")) or _safe_text(meta_row.get("frame_text_long"))


def _rebuild_frame_text_fields(
    *,
    base_meta_rows: Sequence[Mapping[str, Any]],
    kept_object_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    kept_by_entry: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in kept_object_rows:
        kept_by_entry[_safe_int(row.get("entry_id"), -1)].append(dict(row))

    updated_meta_rows: List[Dict[str, Any]] = []
    per_entry_stats: Dict[int, Dict[str, Any]] = {}
    for index, base_row in enumerate(base_meta_rows):
        entry_id = _entry_id_from_meta(base_row, index)
        rows = kept_by_entry.get(entry_id, [])
        if rows:
            object_texts_short = [
                _safe_text(row.get("object_text_short")) or UNKNOWN_TEXT_TOKEN for row in rows
            ] or [UNKNOWN_TEXT_TOKEN]
            object_texts_long = [
                _safe_text(row.get("object_text_long")) or UNKNOWN_TEXT_TOKEN for row in rows
            ] or [UNKNOWN_TEXT_TOKEN]
            frame_text_short = " | ".join(object_texts_short) if object_texts_short else UNKNOWN_TEXT_TOKEN
            frame_text_long = " | ".join(object_texts_long) if object_texts_long else UNKNOWN_TEXT_TOKEN
        else:
            object_texts_short = [UNKNOWN_TEXT_TOKEN]
            object_texts_long = [UNKNOWN_TEXT_TOKEN]
            frame_text_short = UNKNOWN_TEXT_TOKEN
            frame_text_long = UNKNOWN_TEXT_TOKEN

        geometry_before = sum(1 for row in rows if _route_for_object_row(row) == "geometry")
        base_geometry_before = geometry_before
        # Use kept rows for text/object count; threshold stats get patched separately.
        updated = _ensure_metadata_record_attribute(dict(base_row))
        updated["id"] = int(entry_id)
        updated["text"] = frame_text_short
        updated["frame_text_short"] = frame_text_short
        updated["frame_text_long"] = frame_text_long
        updated["text_input_for_clip_short"] = frame_text_short
        updated["text_input_for_clip_long"] = frame_text_long
        updated["object_text_inputs_short"] = list(object_texts_short)
        updated["object_text_inputs_long"] = list(object_texts_long)
        updated["object_count"] = int(len(rows))
        updated_meta_rows.append(updated)
        per_entry_stats[int(entry_id)] = {
            "object_count": int(len(rows)),
            "frame_text_short": frame_text_short,
            "frame_text_long": frame_text_long,
            "object_text_inputs_short": list(object_texts_short),
            "object_text_inputs_long": list(object_texts_long),
            "kept_rows": rows,
            "geometry_kept_count": sum(1 for row in rows if _route_for_object_row(row) == "geometry"),
            "base_geometry_count": int(base_geometry_before),
        }
    return updated_meta_rows, per_entry_stats


def _subset_object_rows(
    *,
    base_root: Path,
    base_object_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
    scored_by_index = {int(row["source_index"]): dict(row) for row in scored_rows if row.get("source_index") is not None}
    kept_rows: List[Dict[str, Any]] = []
    kept_indices: List[int] = []
    pre_threshold_debug_rows: List[Dict[str, Any]] = []
    for index, base_row in enumerate(base_object_rows):
        scored = scored_by_index.get(int(index))
        if scored is None:
            continue
        updated = dict(base_row)
        updated["occlusion_level"] = scored["occlusion_level"]
        updated["occlusion_penalty_p_o"] = scored["occlusion_penalty_p_o"]
        updated["reweighted_detection_score_r"] = scored["reweighted_detection_score_r"]
        for key in ("file_name", "crop_path", "mask_path", "mask_overlay_path", "depth_map_path"):
            if key in updated:
                updated[key] = _normalize_path_for_variant(base_root, updated.get(key))
        pre_threshold_debug_rows.append(
            {
                "entry_id": _safe_int(updated.get("entry_id"), -1),
                "frame_id": _safe_int(updated.get("frame_id"), -1),
                "file_name": _safe_text(updated.get("file_name")),
                "object_local_id": _safe_text(updated.get("object_local_id")),
                "object_route": _route_for_object_row(updated),
                "label": _safe_text(updated.get("label")),
                "bbox_xyxy": list(updated.get("bbox_xyxy") or []),
                "bbox_xywh_norm": list(updated.get("bbox_xywh_norm") or []),
                "object_confidence": updated.get("object_confidence"),
                "detector_confidence": updated.get("detector_confidence"),
                "occlusion_level": updated.get("occlusion_level"),
                "occlusion_penalty_p_o": updated.get("occlusion_penalty_p_o"),
                "reweighted_detection_score_r": updated.get("reweighted_detection_score_r"),
                "r_threshold_used": scored.get("threshold"),
                "would_be_filtered_by_r_threshold": bool(not scored.get("keep")),
            }
        )
        if bool(scored.get("keep")):
            kept_rows.append(updated)
            kept_indices.append(int(index))
    return kept_rows, kept_indices, pre_threshold_debug_rows


def _rebuild_frame_embeddings(
    *,
    meta_rows: Sequence[Mapping[str, Any]],
    base_meta_rows: Sequence[Mapping[str, Any]],
    base_text_short: np.ndarray,
    base_text_long: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not meta_rows:
        dim = int(base_text_short.shape[1]) if base_text_short.ndim == 2 and base_text_short.shape[1] > 0 else 0
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0, dim), dtype=np.float32)

    short_rows: List[np.ndarray] = []
    long_rows: List[np.ndarray] = []
    text_short_cache: Dict[str, np.ndarray] = {}
    text_long_cache: Dict[str, np.ndarray] = {}
    embedder = None

    for index, row in enumerate(meta_rows):
        desired_short = _safe_text(row.get("text_input_for_clip_short")) or UNKNOWN_TEXT_TOKEN
        desired_long = _safe_text(row.get("text_input_for_clip_long")) or UNKNOWN_TEXT_TOKEN
        current_short = _current_frame_text_short(base_meta_rows[index])
        current_long = _current_frame_text_long(base_meta_rows[index])

        if desired_short == current_short:
            short_rows.append(np.asarray(base_text_short[index], dtype=np.float32).reshape(-1))
        else:
            if desired_short not in text_short_cache:
                if embedder is None:
                    embedder = _make_embedder()
                text_short_cache[desired_short] = np.asarray(embedder.embed_text(desired_short), dtype=np.float32).reshape(-1)
            short_rows.append(text_short_cache[desired_short])

        if desired_long == current_long:
            long_rows.append(np.asarray(base_text_long[index], dtype=np.float32).reshape(-1))
        else:
            if desired_long not in text_long_cache:
                if embedder is None:
                    embedder = _make_embedder()
                text_long_cache[desired_long] = np.asarray(embedder.embed_text(desired_long), dtype=np.float32).reshape(-1)
            long_rows.append(text_long_cache[desired_long])

    return np.vstack(short_rows).astype("float32"), np.vstack(long_rows).astype("float32")


def _rewrite_raw_api_rows(
    *,
    base_root: Path,
    raw_api_rows: Sequence[Mapping[str, Any]],
    per_entry_threshold_stats: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    updated_rows: List[Dict[str, Any]] = []
    for row in raw_api_rows:
        entry_id = _safe_int(row.get("entry_id"), -1)
        stats = dict(per_entry_threshold_stats.get(entry_id, {}))
        updated = dict(row)
        artifacts = dict(updated.get("geometry_artifacts") or {})
        for key in (
            "detections_path",
            "detection_overlay_path",
            "filtered_detections_path",
            "filtered_detection_overlay_path",
            "depth_map_path",
            "depth_preview_path",
        ):
            if key in artifacts:
                artifacts[key] = _normalize_path_for_variant(base_root, artifacts.get(key))
        updated["geometry_artifacts"] = artifacts
        timing = dict(updated.get("timing") or {})
        if stats:
            timing["geometry_objects_before_r_threshold"] = int(stats.get("geometry_before_count", 0))
            timing["geometry_objects_after_r_threshold"] = int(stats.get("geometry_after_count", 0))
            timing["geometry_objects_filtered_by_r_threshold"] = int(stats.get("geometry_filtered_count", 0))
            timing["geometry_all_objects_filtered_by_r_threshold"] = bool(stats.get("geometry_all_filtered", False))
        updated["timing"] = timing
        updated_rows.append(updated)
    return updated_rows


def _rewrite_timing_rows(
    timing_rows: Sequence[Mapping[str, Any]],
    *,
    per_entry_threshold_stats: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    updated_rows: List[Dict[str, Any]] = []
    for row in timing_rows:
        entry_id = _safe_int(row.get("entry_id"), -1)
        stats = dict(per_entry_threshold_stats.get(entry_id, {}))
        updated = dict(row)
        if stats:
            updated["object_count"] = int(stats.get("object_count", 0))
            updated["geometry_objects_before_r_threshold"] = int(stats.get("geometry_before_count", 0))
            updated["geometry_objects_after_r_threshold"] = int(stats.get("geometry_after_count", 0))
            updated["geometry_objects_filtered_by_r_threshold"] = int(stats.get("geometry_filtered_count", 0))
            updated["geometry_all_objects_filtered_by_r_threshold"] = bool(stats.get("geometry_all_filtered", False))
        updated_rows.append(updated)
    return updated_rows


def _rewrite_overview_outputs(base_root: Path, output_root: Path, overview_outputs: Mapping[str, Any]) -> Dict[str, Any]:
    rewritten: Dict[str, Any] = {}
    for key, value in dict(overview_outputs or {}).items():
        raw = _safe_text(value)
        if not raw:
            continue
        normalized = _normalize_path_for_variant(base_root, raw)
        relative = Path(normalized)
        candidate = output_root / relative
        rewritten[key] = str(candidate) if candidate.exists() or "overview" in relative.parts else normalized
    return rewritten


def _rebuild_build_report(
    *,
    base_root: Path,
    output_root: Path,
    base_report: Mapping[str, Any],
    meta_rows: Sequence[Mapping[str, Any]],
    kept_object_rows: Sequence[Mapping[str, Any]],
    pre_threshold_rows: Sequence[Mapping[str, Any]],
    view_object_relations: Sequence[Mapping[str, Any]],
    object_object_relations: Sequence[Mapping[str, Any]],
    image_arr: np.ndarray,
    text_arr_short: np.ndarray,
    text_arr_long: np.ndarray,
    object_arr_short: np.ndarray,
    object_arr_long: np.ndarray,
    per_entry_threshold_stats: Mapping[int, Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    report = dict(base_report)
    object_config = dict(report.get("object_config") or {})
    occlusion_block = dict(object_config.get("occlusion_reweight") or {})
    occlusion_block["w1"] = float(summary["w1"])
    occlusion_block["w2"] = float(summary["w2"])
    occlusion_block["b"] = float(summary["b"])
    object_config["occlusion_reweight"] = occlusion_block
    object_config["r_threshold"] = summary.get("threshold")
    object_config["r_threshold_enabled"] = bool(summary.get("threshold") is not None)
    report["object_config"] = object_config
    report["output_dir"] = str(output_root)
    report["object_r_scores_pre_threshold_csv_path"] = str(output_root / "object_r_scores_pre_threshold.csv")
    report["object_r_scores_csv_path"] = str(output_root / "object_r_scores.csv")
    report["object_r_scores_pre_threshold_count"] = int(len(pre_threshold_rows))
    report["object_r_scores_count"] = int(len(kept_object_rows))
    report["r_threshold"] = summary.get("threshold")
    report["r_threshold_enabled"] = bool(summary.get("threshold") is not None)
    report["geometry_objects_before_r_threshold"] = int(sum(v.get("geometry_before_count", 0) for v in per_entry_threshold_stats.values()))
    report["geometry_objects_after_r_threshold"] = int(sum(v.get("geometry_after_count", 0) for v in per_entry_threshold_stats.values()))
    report["geometry_objects_filtered_by_r_threshold"] = int(
        sum(v.get("geometry_filtered_count", 0) for v in per_entry_threshold_stats.values())
    )
    report["frames_all_geometry_objects_filtered"] = int(
        sum(1 for v in per_entry_threshold_stats.values() if bool(v.get("geometry_all_filtered", False)))
    )
    report["image_index_ntotal"] = int(image_arr.shape[0]) if image_arr.ndim == 2 else 0
    report["text_index_ntotal_short"] = int(text_arr_short.shape[0]) if text_arr_short.ndim == 2 else 0
    report["text_index_ntotal_long"] = int(text_arr_long.shape[0]) if text_arr_long.ndim == 2 else 0
    report["object_index_ntotal_short"] = int(object_arr_short.shape[0]) if object_arr_short.ndim == 2 else 0
    report["object_index_ntotal_long"] = int(object_arr_long.shape[0]) if object_arr_long.ndim == 2 else 0
    report["total_entries"] = int(len(meta_rows))
    report["total_objects"] = int(len(kept_object_rows))
    report["avg_objects_per_frame"] = float(len(kept_object_rows) / max(len(meta_rows), 1))
    report["total_view_object_relations"] = int(len(view_object_relations))
    report["total_object_object_relations"] = int(len(object_object_relations))
    report["total_left_bucket_objects"] = sum(1 for row in kept_object_rows if _safe_text(row.get("angle_bucket")) == "left")
    report["total_center_bucket_objects"] = sum(
        1 for row in kept_object_rows if _safe_text(row.get("angle_bucket")) == "center"
    )
    report["total_right_bucket_objects"] = sum(1 for row in kept_object_rows if _safe_text(row.get("angle_bucket")) == "right")
    report["overview_outputs"] = _rewrite_overview_outputs(base_root, output_root, report.get("overview_outputs") or {})
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["reweight_sweep"] = {
        "base_db_dir": str(base_root),
        "config_token": _safe_text(summary.get("config_token")),
        "analysis_only": bool(summary.get("analysis_only")),
        "exportable": bool(summary.get("exportable")),
        "w1": float(summary["w1"]),
        "w2": float(summary["w2"]),
        "b": float(summary["b"]),
        "threshold": summary.get("threshold"),
    }
    return report


def _export_variant_db(
    *,
    base_root: Path,
    output_root: Path,
    base_meta_rows: Sequence[Mapping[str, Any]],
    base_object_rows: Sequence[Mapping[str, Any]],
    base_raw_api_rows: Sequence[Mapping[str, Any]],
    base_timing_rows: Sequence[Mapping[str, Any]],
    base_report: Mapping[str, Any],
    image_arr: np.ndarray,
    base_text_short: np.ndarray,
    base_text_long: np.ndarray,
    base_object_arr_short: np.ndarray,
    base_object_arr_long: np.ndarray,
    scored_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    for dirname in ("images", "geometry", "overview", "vlm_cache", "vlm_object_cache"):
        _symlink_dir(base_root / dirname, output_root / dirname)

    kept_object_rows, kept_indices, pre_threshold_debug_rows = _subset_object_rows(
        base_root=base_root,
        base_object_rows=base_object_rows,
        scored_rows=scored_rows,
    )
    meta_rows, frame_stats = _rebuild_frame_text_fields(
        base_meta_rows=base_meta_rows,
        kept_object_rows=kept_object_rows,
    )

    per_entry_threshold_stats: Dict[int, Dict[str, Any]] = {}
    scored_by_entry: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        scored_by_entry[_safe_int(row.get("entry_id"), -1)].append(dict(row))

    for index, meta_row in enumerate(meta_rows):
        entry_id = _entry_id_from_meta(meta_row, index)
        entry_rows = scored_by_entry.get(entry_id, [])
        geometry_before = sum(1 for row in entry_rows if _safe_text(row.get("route")) == "geometry")
        geometry_after = sum(
            1 for row in entry_rows if _safe_text(row.get("route")) == "geometry" and bool(row.get("keep"))
        )
        filtered = geometry_before - geometry_after
        per_entry_threshold_stats[entry_id] = {
            "object_count": int(frame_stats[entry_id]["object_count"]),
            "geometry_before_count": int(geometry_before),
            "geometry_after_count": int(geometry_after),
            "geometry_filtered_count": int(filtered),
            "geometry_all_filtered": bool(geometry_before > 0 and geometry_after == 0 and filtered == geometry_before),
        }

    updated_raw_api_rows = _rewrite_raw_api_rows(
        base_root=base_root,
        raw_api_rows=base_raw_api_rows,
        per_entry_threshold_stats=per_entry_threshold_stats,
    )
    updated_timing_rows = _rewrite_timing_rows(
        base_timing_rows,
        per_entry_threshold_stats=per_entry_threshold_stats,
    )

    text_arr_short, text_arr_long = _rebuild_frame_embeddings(
        meta_rows=meta_rows,
        base_meta_rows=base_meta_rows,
        base_text_short=base_text_short,
        base_text_long=base_text_long,
    )
    if kept_indices:
        object_arr_short = np.asarray(base_object_arr_short[kept_indices], dtype=np.float32)
        object_arr_long = np.asarray(base_object_arr_long[kept_indices], dtype=np.float32)
    else:
        dim = int(base_object_arr_short.shape[1]) if base_object_arr_short.ndim == 2 and base_object_arr_short.shape[1] > 0 else 0
        object_arr_short = np.zeros((0, dim), dtype=np.float32)
        object_arr_long = np.zeros((0, dim), dtype=np.float32)

    view_object_relations = _build_view_object_relations(meta_rows, kept_object_rows)
    object_object_relations = _build_object_object_relations(meta_rows, kept_object_rows)

    _write_jsonl(output_root / "meta.jsonl", list(meta_rows))
    _write_jsonl(output_root / "metadata.jsonl", list(meta_rows))
    _write_jsonl(output_root / "object_meta.jsonl", list(kept_object_rows))
    _write_jsonl(output_root / "view_object_relations.jsonl", list(view_object_relations))
    _write_jsonl(output_root / "object_object_relations.jsonl", list(object_object_relations))
    if updated_raw_api_rows:
        _write_jsonl(output_root / "raw_api_responses.jsonl", list(updated_raw_api_rows))
    if updated_timing_rows:
        _write_jsonl(output_root / "per_image_timings.jsonl", list(updated_timing_rows))

    np.save(output_root / "image_emb.npy", np.asarray(image_arr, dtype=np.float32))
    np.save(output_root / "text_emb_short.npy", np.asarray(text_arr_short, dtype=np.float32))
    np.save(output_root / "text_emb_long.npy", np.asarray(text_arr_long, dtype=np.float32))
    np.save(output_root / "object_text_emb_short.npy", np.asarray(object_arr_short, dtype=np.float32))
    np.save(output_root / "object_text_emb_long.npy", np.asarray(object_arr_long, dtype=np.float32))

    _write_csv_rows(output_root / "object_r_scores_pre_threshold.csv", _OBJECT_R_SCORES_PRE_THRESHOLD_COLUMNS, pre_threshold_debug_rows)
    _write_csv_rows(
        output_root / "object_r_scores.csv",
        _OBJECT_R_SCORES_COLUMNS,
        [
            {
                "object_global_id": _safe_int(row.get("object_global_id"), 0),
                "reweighted_detection_score_r": row.get("reweighted_detection_score_r"),
            }
            for row in kept_object_rows
        ],
    )

    image_index_ntotal = _save_faiss_index(np.asarray(image_arr, dtype=np.float32), output_root / "image_index.faiss")
    text_index_ntotal_short = _save_faiss_index(np.asarray(text_arr_short, dtype=np.float32), output_root / "text_index_short.faiss")
    text_index_ntotal_long = _save_faiss_index(np.asarray(text_arr_long, dtype=np.float32), output_root / "text_index_long.faiss")
    object_index_ntotal_short = _save_faiss_index(
        np.asarray(object_arr_short, dtype=np.float32),
        output_root / "object_index_short.faiss",
    )
    object_index_ntotal_long = _save_faiss_index(
        np.asarray(object_arr_long, dtype=np.float32),
        output_root / "object_index_long.faiss",
    )

    report = _rebuild_build_report(
        base_root=base_root,
        output_root=output_root,
        base_report=base_report,
        meta_rows=meta_rows,
        kept_object_rows=kept_object_rows,
        pre_threshold_rows=pre_threshold_debug_rows,
        view_object_relations=view_object_relations,
        object_object_relations=object_object_relations,
        image_arr=np.asarray(image_arr, dtype=np.float32),
        text_arr_short=np.asarray(text_arr_short, dtype=np.float32),
        text_arr_long=np.asarray(text_arr_long, dtype=np.float32),
        object_arr_short=np.asarray(object_arr_short, dtype=np.float32),
        object_arr_long=np.asarray(object_arr_long, dtype=np.float32),
        per_entry_threshold_stats=per_entry_threshold_stats,
        summary=summary,
    )
    report["image_index_ntotal"] = int(image_index_ntotal)
    report["text_index_ntotal_short"] = int(text_index_ntotal_short)
    report["text_index_ntotal_long"] = int(text_index_ntotal_long)
    report["object_index_ntotal_short"] = int(object_index_ntotal_short)
    report["object_index_ntotal_long"] = int(object_index_ntotal_long)
    _write_json(output_root / "build_report.json", report)

    config_summary = dict(summary)
    config_summary["exported_db_dir"] = str(output_root)
    config_summary["object_r_scores_pre_threshold_csv_path"] = str(output_root / "object_r_scores_pre_threshold.csv")
    config_summary["object_r_scores_csv_path"] = str(output_root / "object_r_scores.csv")
    _write_json(output_root / "config_summary.json", config_summary)
    return config_summary


def run_reweight_sweep(
    db_dir: str,
    *,
    w1_values: Sequence[Any] = (OCCLUSION_REWEIGHT_W1,),
    w2_values: Sequence[Any] = (OCCLUSION_REWEIGHT_W2,),
    b_values: Sequence[Any] = (OCCLUSION_REWEIGHT_B,),
    thresholds: Sequence[Any] = (None,),
    export_db_variants: bool = False,
    export_filtered_objects: bool = False,
    filtered_object_dirname: str = "filtered_obj",
    selected_configs: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    base_root = Path(db_dir).expanduser().resolve()
    if not base_root.exists():
        raise FileNotFoundError(f"DB directory does not exist: {base_root}")

    meta_rows = _load_jsonl(base_root / "meta.jsonl")
    object_rows = _load_jsonl(base_root / "object_meta.jsonl")
    raw_api_rows = _load_jsonl(base_root / "raw_api_responses.jsonl")
    timing_rows = _load_jsonl(base_root / "per_image_timings.jsonl")
    pre_threshold_rows = _load_csv_rows(base_root / "object_r_scores_pre_threshold.csv")
    build_report = _load_json(base_root / "build_report.json")

    if not meta_rows or not object_rows:
        raise FileNotFoundError(f"Missing meta.jsonl or object_meta.jsonl in {base_root}")

    image_arr = np.load(base_root / "image_emb.npy").astype("float32")
    text_arr_short = np.load(base_root / "text_emb_short.npy").astype("float32")
    text_arr_long = np.load(base_root / "text_emb_long.npy").astype("float32")
    object_arr_short = np.load(base_root / "object_text_emb_short.npy").astype("float32")
    object_arr_long = np.load(base_root / "object_text_emb_long.npy").astype("float32")

    if image_arr.ndim != 2 or image_arr.shape[0] != len(meta_rows):
        raise ValueError("image_emb.npy is misaligned with meta.jsonl")
    if text_arr_short.ndim != 2 or text_arr_short.shape[0] != len(meta_rows):
        raise ValueError("text_emb_short.npy is misaligned with meta.jsonl")
    if text_arr_long.ndim != 2 or text_arr_long.shape[0] != len(meta_rows):
        raise ValueError("text_emb_long.npy is misaligned with meta.jsonl")
    if object_arr_short.ndim != 2 or object_arr_short.shape[0] != len(object_rows):
        raise ValueError("object_text_emb_short.npy is misaligned with object_meta.jsonl")
    if object_arr_long.ndim != 2 or object_arr_long.shape[0] != len(object_rows):
        raise ValueError("object_text_emb_long.npy is misaligned with object_meta.jsonl")

    is_filtered_base = _canonical_base_is_filtered(
        build_report=build_report,
        object_rows=object_rows,
        pre_threshold_rows=pre_threshold_rows,
    )
    exportable = bool(not is_filtered_base)
    analysis_only = bool(is_filtered_base)

    if export_db_variants and not exportable:
        raise ValueError(
            "Cannot export DB variants from an already filtered DB. "
            "Use a canonical DB built without r_threshold."
        )

    if exportable:
        analysis_rows = _build_analysis_rows_from_objects(object_rows)
    else:
        if not pre_threshold_rows:
            raise ValueError(
                "Filtered DB analysis requires object_r_scores_pre_threshold.csv to be present."
            )
        analysis_rows = _build_analysis_rows_from_prethreshold(pre_threshold_rows)

    normalized_w1 = [float(value) for value in _parse_float_list(w1_values)]
    normalized_w2 = [float(value) for value in _parse_float_list(w2_values)]
    normalized_b = [float(value) for value in _parse_float_list(b_values)]
    normalized_thresholds = _parse_float_list(thresholds, allow_none=True)
    selected_tokens = set(_parse_selected_tokens(selected_configs))

    base_total_objects = int(len(object_rows))
    run_base = Path(output_dir).expanduser().resolve() if output_dir else (base_root / "reweight_sweeps")
    run_root = _make_run_output_dir(run_base)

    summaries: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    sweep_row_by_token: Dict[str, Dict[str, Any]] = {}
    export_queue: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    filtered_export_queue: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    seen_tokens: set[str] = set()
    for w1 in normalized_w1:
        for w2 in normalized_w2:
            for b in normalized_b:
                for threshold in normalized_thresholds:
                    token = _config_token(w1, w2, b, threshold)
                    if token in seen_tokens:
                        continue
                    seen_tokens.add(token)
                    scored_rows = _score_analysis_rows(
                        analysis_rows,
                        w1=w1,
                        w2=w2,
                        b=b,
                        threshold=threshold,
                    )
                    summary = _summarize_scored_rows(
                        meta_rows=meta_rows,
                        scored_rows=scored_rows,
                        base_total_objects=base_total_objects,
                        exportable=exportable,
                        analysis_only=analysis_only,
                        config_token=token,
                        w1=w1,
                        w2=w2,
                        b=b,
                        threshold=threshold,
                    )
                    if analysis_only:
                        summary["warning"] = (
                            "Input DB is already filtered; summary is analysis-only and cannot export full DB variants."
                        )
                    summary["filtered_object_dir"] = ""
                    summary["filtered_manifest_path"] = ""
                    summary["filtered_object_count"] = 0
                    summaries.append(summary)
                    csv_row = dict(summary)
                    csv_row["filtered_label_counts"] = json.dumps(summary.get("filtered_label_counts", {}), ensure_ascii=True)
                    csv_row["exported_db_dir"] = ""
                    csv_row["filtered_object_dir"] = ""
                    csv_row["filtered_manifest_path"] = ""
                    csv_row["filtered_object_count"] = 0
                    sweep_rows.append(csv_row)
                    sweep_row_by_token[token] = csv_row
                    should_export = bool(
                        export_db_variants and (not selected_tokens or token in selected_tokens)
                    )
                    should_export_filtered = bool(
                        export_filtered_objects and (not selected_tokens or token in selected_tokens)
                    )
                    if should_export:
                        export_queue.append((summary, scored_rows))
                    if should_export_filtered:
                        filtered_export_queue.append((summary, scored_rows))

    if selected_tokens:
        unknown = sorted(token for token in selected_tokens if token not in seen_tokens)
        if unknown:
            raise ValueError(f"Unknown selected_configs token(s): {unknown}")

    for summary, scored_rows in export_queue:
        token = _safe_text(summary.get("config_token"))
        config_dir = run_root / f"config_{token}"
        exported_summary = _export_variant_db(
            base_root=base_root,
            output_root=config_dir,
            base_meta_rows=meta_rows,
            base_object_rows=object_rows,
            base_raw_api_rows=raw_api_rows,
            base_timing_rows=timing_rows,
            base_report=build_report,
            image_arr=image_arr,
            base_text_short=text_arr_short,
            base_text_long=text_arr_long,
            base_object_arr_short=object_arr_short,
            base_object_arr_long=object_arr_long,
            scored_rows=scored_rows,
            summary=summary,
        )
        summary["exported_db_dir"] = str(config_dir)
        if token in sweep_row_by_token:
            sweep_row_by_token[token]["exported_db_dir"] = str(config_dir)
        _write_json(config_dir / "config_summary.json", exported_summary)

    for summary, scored_rows in filtered_export_queue:
        token = _safe_text(summary.get("config_token"))
        config_dir = run_root / f"config_{token}"
        config_dir.mkdir(parents=True, exist_ok=True)
        filtered_rows = [dict(row) for row in scored_rows if not bool(row.get("keep"))]
        filtered_export = _export_filtered_objects(
            db_root=base_root,
            base_object_rows=object_rows,
            filtered_rows=filtered_rows,
            export_dir=config_dir / filtered_object_dirname,
            threshold=summary.get("threshold"),
        )
        summary["filtered_object_dir"] = filtered_export["dir"]
        summary["filtered_manifest_path"] = filtered_export["manifest_path"]
        summary["filtered_object_count"] = int(filtered_export["filtered_object_count"])
        summary["filtered_exported_count"] = int(filtered_export["exported_count"])
        summary["filtered_export_skipped_count"] = int(filtered_export["skipped_count"])
        if token in sweep_row_by_token:
            sweep_row_by_token[token]["filtered_object_dir"] = filtered_export["dir"]
            sweep_row_by_token[token]["filtered_manifest_path"] = filtered_export["manifest_path"]
            sweep_row_by_token[token]["filtered_object_count"] = int(filtered_export["filtered_object_count"])
        _write_json(config_dir / "config_summary.json", dict(summary))

    _write_csv_rows(
        run_root / "sweep_results.csv",
        (
            "config_token",
            "w1",
            "w2",
            "b",
            "threshold",
            "base_total_objects",
            "candidate_total_objects",
            "kept_total_objects",
            "dropped_total_objects",
            "total_object_delta_vs_base",
            "geometry_candidate_count",
            "geometry_kept_count",
            "geometry_filtered_count",
            "fallback_candidate_count",
            "fallback_kept_count",
            "empty_frame_count",
            "keep_rate",
            "analysis_only",
            "exportable",
            "filtered_label_counts",
            "exported_db_dir",
            "filtered_object_dir",
            "filtered_manifest_path",
            "filtered_object_count",
            "warning",
        ),
        sweep_rows,
    )

    summary_payload = {
        "db_dir": str(base_root),
        "output_dir": str(run_root),
        "canonical_base_required": True,
        "base_db_filtered": bool(is_filtered_base),
        "exportable": bool(exportable),
        "analysis_only": bool(analysis_only),
        "num_configs": int(len(summaries)),
        "exported_config_count": int(sum(1 for item in summaries if _safe_text(item.get("exported_db_dir")))),
        "filtered_object_exported_config_count": int(sum(1 for item in summaries if _safe_text(item.get("filtered_object_dir")))),
        "runs": summaries,
    }
    _write_json(run_root / "sweep_summary.json", summary_payload)
    return summary_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline reweight sweep for canonical spatial DBs.")
    parser.add_argument("--db_dir", type=str, required=True, help="Canonical base DB directory.")
    parser.add_argument("--w1_values", type=str, default=str(OCCLUSION_REWEIGHT_W1), help="Comma-separated w1 values.")
    parser.add_argument("--w2_values", type=str, default=str(OCCLUSION_REWEIGHT_W2), help="Comma-separated w2 values.")
    parser.add_argument("--b_values", type=str, default=str(OCCLUSION_REWEIGHT_B), help="Comma-separated b values.")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="none",
        help="Comma-separated thresholds. Use 'none' to disable threshold for a config.",
    )
    parser.add_argument(
        "--export_db_variants",
        type=_str_to_bool,
        default=False,
        help="Whether to export filtered DB variants for selected configs.",
    )
    parser.add_argument(
        "--export_filtered_objects",
        type=_str_to_bool,
        default=False,
        help="Whether to export filtered object crops for selected configs.",
    )
    parser.add_argument(
        "--filtered_object_dirname",
        type=str,
        default="filtered_obj",
        help="Directory name used under each config root for exported filtered object crops.",
    )
    parser.add_argument(
        "--selected_configs",
        type=str,
        default="",
        help="Optional comma-separated config tokens to export. Empty means export all configs for enabled export modes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional sweep root directory. Defaults to <db_dir>/reweight_sweeps.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = run_reweight_sweep(
        db_dir=args.db_dir,
        w1_values=_parse_float_list(args.w1_values),
        w2_values=_parse_float_list(args.w2_values),
        b_values=_parse_float_list(args.b_values),
        thresholds=_parse_float_list(args.thresholds, allow_none=True),
        export_db_variants=bool(args.export_db_variants),
        export_filtered_objects=bool(args.export_filtered_objects),
        filtered_object_dirname=args.filtered_object_dirname,
        selected_configs=_parse_selected_tokens(args.selected_configs),
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
