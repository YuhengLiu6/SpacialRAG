from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence as AbcSequence
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


SemanticFrameProvider = Callable[[int, Mapping[str, Any]], np.ndarray]


UNKNOWN_GT_LABEL = "unknown"


def _empty_gt_assignment(status: str, *, include_unknown: bool) -> Dict[str, Any]:
    return {
        "gt_assignment_status": str(status),
        "gt_label": UNKNOWN_GT_LABEL if include_unknown else None,
        "gt_semantic_id_top": None,
        "gt_label_pixel_count": 0,
        "gt_label_pixel_ratio": 0.0,
        "gt_label_histogram": {},
        "gt_semantic_id_histogram": {},
        "gt_bbox_pixel_count": 0,
        "gt_valid_semantic_pixel_count": 0,
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2), encoding="utf-8")


def parse_hm3d_semantic_txt(path: str | Path) -> Dict[int, str]:
    """
    Parse HM3D semantic annotation text files.

    Expected rows after the title line look like:
    21,086BDF,"table lamp",1
    """
    semantic_path = Path(path)
    id_to_label: Dict[int, str] = {}
    with semantic_path.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.readlines()
    if not lines:
        return id_to_label

    for row in csv.reader(lines[1:]):
        if len(row) < 3:
            continue
        try:
            semantic_id = int(str(row[0]).strip())
        except Exception:
            continue
        label = str(row[2]).strip()
        if label:
            id_to_label[semantic_id] = label
    return id_to_label


def _safe_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out


def _source_image_path(spatial_db_dir: Path, raw_path: Any) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [spatial_db_dir / path, path]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _load_image_shape(path: Optional[Path]) -> Optional[Tuple[int, int]]:
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    return int(h), int(w)


def _bbox_to_semantic_slice(
    bbox_xyxy: Sequence[Any],
    *,
    semantic_shape: Tuple[int, int],
    source_image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Tuple[slice, slice]], Dict[str, Any]]:
    bbox_values = list(bbox_xyxy) if bbox_xyxy is not None else []
    if len(bbox_values) < 4:
        return None, {"status": "empty_bbox"}
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_values[:4]]
    except Exception:
        return None, {"status": "empty_bbox"}
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None, {"status": "empty_bbox"}

    sem_h, sem_w = int(semantic_shape[0]), int(semantic_shape[1])
    if sem_h <= 0 or sem_w <= 0:
        return None, {"status": "invalid_semantic_frame"}

    scale_x = 1.0
    scale_y = 1.0
    if source_image_shape is not None:
        src_h, src_w = int(source_image_shape[0]), int(source_image_shape[1])
        if src_h > 0 and src_w > 0 and (src_h != sem_h or src_w != sem_w):
            scale_x = float(sem_w) / float(src_w)
            scale_y = float(sem_h) / float(src_h)
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y

    x1_i = max(0, min(sem_w, int(math.floor(x1))))
    y1_i = max(0, min(sem_h, int(math.floor(y1))))
    x2_i = max(0, min(sem_w, int(math.ceil(x2))))
    y2_i = max(0, min(sem_h, int(math.ceil(y2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None, {
            "status": "empty_bbox_after_clip",
            "bbox_xyxy_semantic": [x1_i, y1_i, x2_i, y2_i],
            "bbox_scale_x": scale_x,
            "bbox_scale_y": scale_y,
        }
    return (
        (slice(y1_i, y2_i), slice(x1_i, x2_i)),
        {
            "status": "ok",
            "bbox_xyxy_semantic": [x1_i, y1_i, x2_i, y2_i],
            "bbox_scale_x": scale_x,
            "bbox_scale_y": scale_y,
        },
    )


def assign_semantic_gt_label(
    *,
    bbox_xyxy: Sequence[Any],
    semantic_frame: np.ndarray,
    id_to_label: Mapping[int, str],
    source_image_shape: Optional[Tuple[int, int]] = None,
    include_unknown: bool = False,
) -> Dict[str, Any]:
    semantic = np.asarray(semantic_frame)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[:, :, 0]
    if semantic.ndim != 2:
        return {"gt_assignment_status": "invalid_semantic_frame"}

    bbox_slice, bbox_info = _bbox_to_semantic_slice(
        bbox_xyxy,
        semantic_shape=(int(semantic.shape[0]), int(semantic.shape[1])),
        source_image_shape=source_image_shape,
    )
    if bbox_slice is None:
        return {
            **_empty_gt_assignment(
                str(bbox_info.get("status") or "empty_bbox"),
                include_unknown=include_unknown,
            ),
            **bbox_info,
        }

    crop = np.asarray(semantic[bbox_slice])
    bbox_pixel_count = int(crop.size)
    ids, counts = np.unique(crop.reshape(-1), return_counts=True)
    id_histogram: Dict[int, int] = {
        int(semantic_id): int(count)
        for semantic_id, count in zip(ids.tolist(), counts.tolist())
    }

    label_histogram: Dict[str, int] = {}
    missing_pixel_count = 0
    zero_pixel_count = 0
    for semantic_id, count in id_histogram.items():
        if int(semantic_id) == 0:
            zero_pixel_count += int(count)
            continue
        label = id_to_label.get(int(semantic_id))
        if label is None:
            missing_pixel_count += int(count)
            continue
        label_histogram[str(label)] = int(label_histogram.get(str(label), 0) + int(count))

    valid_pixel_count = int(sum(label_histogram.values()))
    if not label_histogram:
        return {
            **_empty_gt_assignment("no_valid_semantic_pixels", include_unknown=include_unknown),
            "gt_semantic_id_histogram": {str(k): int(v) for k, v in sorted(id_histogram.items())},
            "gt_bbox_pixel_count": bbox_pixel_count,
            "gt_valid_semantic_pixel_count": 0,
            "gt_zero_semantic_pixel_count": int(zero_pixel_count),
            "gt_missing_semantic_pixel_count": int(missing_pixel_count),
            **bbox_info,
        }

    gt_label, gt_label_count = sorted(
        label_histogram.items(),
        key=lambda item: (-int(item[1]), item[0]),
    )[0]
    candidate_ids = [
        (semantic_id, count)
        for semantic_id, count in id_histogram.items()
        if int(semantic_id) != 0 and id_to_label.get(int(semantic_id)) == gt_label
    ]
    gt_semantic_id_top = None
    if candidate_ids:
        gt_semantic_id_top = int(
            sorted(candidate_ids, key=lambda item: (-int(item[1]), int(item[0])))[0][0]
        )

    return {
        "gt_assignment_status": "ok",
        "gt_label": gt_label,
        "gt_semantic_id_top": gt_semantic_id_top,
        "gt_label_pixel_count": int(gt_label_count),
        "gt_label_pixel_ratio": float(int(gt_label_count) / max(float(valid_pixel_count), 1.0)),
        "gt_label_histogram": {str(k): int(v) for k, v in sorted(label_histogram.items())},
        "gt_semantic_id_histogram": {str(k): int(v) for k, v in sorted(id_histogram.items())},
        "gt_bbox_pixel_count": bbox_pixel_count,
        "gt_valid_semantic_pixel_count": valid_pixel_count,
        "gt_zero_semantic_pixel_count": int(zero_pixel_count),
        "gt_missing_semantic_pixel_count": int(missing_pixel_count),
        **bbox_info,
    }


def _meta_by_entry_id(meta_rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for idx, row in enumerate(meta_rows):
        entry_id = _safe_int(row.get("id"))
        if entry_id is None:
            entry_id = idx
        out[int(entry_id)] = dict(row)
    return out


def _position_from_meta(meta_row: Mapping[str, Any]) -> List[float]:
    raw = meta_row.get("world_position")
    if isinstance(raw, AbcSequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
        return [float(raw[0]), float(raw[1]), float(raw[2])]
    return [
        float(meta_row.get("x")),
        float(meta_row.get("y")),
        float(meta_row.get("z")),
    ]


def _orientation_from_meta(meta_row: Mapping[str, Any]) -> float:
    return float(meta_row.get("orientation"))


class HabitatSemanticFrameProvider:
    def __init__(
        self,
        *,
        scene_path: str,
        scene_dataset_config_file: Optional[str] = None,
    ):
        from spatial_rag.explorer_semantic import SemanticExplorer

        self.explorer = SemanticExplorer(
            scene_path=scene_path,
            scene_dataset_config_file=scene_dataset_config_file,
            require_semantics=True,
        )

    def __call__(self, entry_id: int, meta_row: Mapping[str, Any]) -> np.ndarray:
        del entry_id
        return self.explorer.capture_semantic_at_pose(
            _position_from_meta(meta_row),
            _orientation_from_meta(meta_row),
        )


def _empty_like_rows(arr: np.ndarray, count: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape={arr.shape}")
    return np.zeros((int(count), int(arr.shape[1])), dtype=np.asarray(arr).dtype)


def _save_selected_rows(path: Path, source: np.ndarray, source_indices: Sequence[int]) -> np.ndarray:
    if source.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape={source.shape}")
    selected = (
        source[np.asarray(list(source_indices), dtype=np.int64)]
        if source_indices
        else _empty_like_rows(source, 0)
    )
    np.save(path, selected)
    return selected


def build_semantic_gt_dataset(
    *,
    spatial_db_dir: str | Path,
    semantic_txt_path: str | Path,
    output_dir: str | Path,
    scene_path: Optional[str] = None,
    scene_dataset_config_file: Optional[str] = None,
    include_unknown: bool = False,
    semantic_frame_provider: Optional[SemanticFrameProvider] = None,
) -> Dict[str, Any]:
    spatial_db = Path(spatial_db_dir)
    output_root = Path(output_dir)
    id_to_label = parse_hm3d_semantic_txt(semantic_txt_path)

    object_rows = _read_jsonl(spatial_db / "object_meta.jsonl")
    meta_rows = _read_jsonl(spatial_db / "meta.jsonl")
    meta_by_entry = _meta_by_entry_id(meta_rows)

    text_short = np.load(spatial_db / "object_text_emb_short.npy")
    text_long = np.load(spatial_db / "object_text_emb_long.npy")
    if text_short.shape[0] != len(object_rows) or text_long.shape[0] != len(object_rows):
        raise ValueError(
            "object_text_emb_short.npy and object_text_emb_long.npy must align with object_meta.jsonl"
        )

    image_emb_path = spatial_db / "image_emb.npy"
    image_emb = np.load(image_emb_path) if image_emb_path.exists() else None
    dino_emb_path = spatial_db / "object_dinov3_emb.npy"
    dino_emb = np.load(dino_emb_path) if dino_emb_path.exists() else None

    if semantic_frame_provider is None:
        if not scene_path:
            raise ValueError("scene_path is required when semantic_frame_provider is not provided")
        semantic_frame_provider = HabitatSemanticFrameProvider(
            scene_path=str(scene_path),
            scene_dataset_config_file=scene_dataset_config_file,
        )

    semantic_cache: Dict[int, np.ndarray] = {}
    image_shape_cache: Dict[int, Optional[Tuple[int, int]]] = {}
    output_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    kept_source_indices: List[int] = []
    view_image_rows: List[np.ndarray] = []
    dino_rows: List[np.ndarray] = []
    dino_available_rows: List[bool] = []

    stats: Dict[str, Any] = {
        "source_object_count": int(len(object_rows)),
        "kept_object_count": 0,
        "skipped_object_count": 0,
        "include_unknown": bool(include_unknown),
        "semantic_label_count": int(len(id_to_label)),
        "gt_label_counts": {},
        "assignment_status_counts": {},
        "view_image_emb_written": False,
        "object_dinov3_emb_written": False,
    }

    for source_index, row in enumerate(object_rows):
        entry_id = _safe_int(row.get("entry_id"))
        if entry_id is None or entry_id not in meta_by_entry:
            assignment = _empty_gt_assignment("missing_pose", include_unknown=bool(include_unknown))
            should_keep = bool(include_unknown)
            meta_row = None
        else:
            meta_row = meta_by_entry[int(entry_id)]
            try:
                if int(entry_id) not in semantic_cache:
                    semantic_cache[int(entry_id)] = np.asarray(
                        semantic_frame_provider(int(entry_id), meta_row)
                    )
                semantic_frame = semantic_cache[int(entry_id)]
                if int(entry_id) not in image_shape_cache:
                    image_shape_cache[int(entry_id)] = _load_image_shape(
                        _source_image_path(spatial_db, meta_row.get("file_name"))
                    )
                assignment = assign_semantic_gt_label(
                    bbox_xyxy=list(row.get("bbox_xyxy") or []),
                    semantic_frame=semantic_frame,
                    id_to_label=id_to_label,
                    source_image_shape=image_shape_cache[int(entry_id)],
                    include_unknown=bool(include_unknown),
                )
            except Exception as exc:
                assignment = {
                    **_empty_gt_assignment(
                        f"semantic_provider_failed:{type(exc).__name__}",
                        include_unknown=bool(include_unknown),
                    ),
                }
            should_keep = bool(assignment.get("gt_label"))

        status = str(assignment.get("gt_assignment_status") or "unknown")
        stats["assignment_status_counts"][status] = int(
            dict(stats["assignment_status_counts"]).get(status, 0)
        ) + 1

        if status != "ok":
            skipped = dict(row)
            skipped["source_object_row_index"] = int(source_index)
            skipped.update(assignment)
            skipped["included_in_dataset"] = bool(should_keep)
            skipped_rows.append(skipped)
        if not should_keep:
            continue

        out_row = dict(row)
        out_row["source_object_row_index"] = int(source_index)
        out_row.update(assignment)
        if dino_emb is not None:
            dino_index = _safe_int(row.get("dinov3_embedding_row_index"))
            dino_available = (
                dino_index is not None
                and dino_emb.ndim == 2
                and 0 <= int(dino_index) < int(dino_emb.shape[0])
            )
        else:
            dino_available = False
        out_row["dinov3_embedding_available"] = bool(dino_available)
        output_rows.append(out_row)
        kept_source_indices.append(int(source_index))
        dino_available_rows.append(bool(dino_available))

        label = str(out_row.get("gt_label") or UNKNOWN_GT_LABEL)
        stats["gt_label_counts"][label] = int(dict(stats["gt_label_counts"]).get(label, 0)) + 1

        if image_emb is not None:
            image_row_index = entry_id if entry_id is not None else None
            if image_row_index is not None and 0 <= int(image_row_index) < int(image_emb.shape[0]):
                view_image_rows.append(np.asarray(image_emb[int(image_row_index)]).reshape(-1))
            else:
                view_image_rows.append(np.zeros((int(image_emb.shape[1]),), dtype=image_emb.dtype))

        if dino_emb is not None:
            dino_index = _safe_int(row.get("dinov3_embedding_row_index"))
            if (
                dino_index is not None
                and dino_emb.ndim == 2
                and 0 <= int(dino_index) < int(dino_emb.shape[0])
            ):
                dino_rows.append(np.asarray(dino_emb[int(dino_index)]).reshape(-1))
            else:
                dino_rows.append(np.zeros((int(dino_emb.shape[1]),), dtype=dino_emb.dtype))

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "semantic_gt_object_meta.jsonl", output_rows)
    _write_jsonl(output_root / "semantic_gt_skipped.jsonl", skipped_rows)
    _save_selected_rows(output_root / "object_text_emb_short.npy", text_short, kept_source_indices)
    _save_selected_rows(output_root / "object_text_emb_long.npy", text_long, kept_source_indices)

    if image_emb is not None:
        view_image_arr = (
            np.vstack(view_image_rows).astype(image_emb.dtype)
            if view_image_rows
            else np.zeros((0, int(image_emb.shape[1])), dtype=image_emb.dtype)
        )
        np.save(output_root / "view_image_emb.npy", view_image_arr)
        stats["view_image_emb_written"] = True

    if dino_emb is not None:
        dino_arr = (
            np.vstack(dino_rows).astype(dino_emb.dtype)
            if dino_rows
            else np.zeros((0, int(dino_emb.shape[1])), dtype=dino_emb.dtype)
        )
        np.save(output_root / "object_dinov3_emb.npy", dino_arr)
        stats["object_dinov3_emb_written"] = True
        stats["object_dinov3_available_count"] = int(sum(1 for v in dino_available_rows if v))

    labels = sorted({str(row.get("gt_label") or UNKNOWN_GT_LABEL) for row in output_rows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    gt_label_ids = np.asarray(
        [label_to_id[str(row.get("gt_label") or UNKNOWN_GT_LABEL)] for row in output_rows],
        dtype=np.int64,
    )
    np.save(output_root / "gt_label_ids.npy", gt_label_ids)
    _write_json(output_root / "gt_label_vocab.json", {"label_to_id": label_to_id, "id_to_label": labels})

    stats["kept_object_count"] = int(len(output_rows))
    stats["skipped_object_count"] = int(len(skipped_rows))
    stats["semantic_frame_count"] = int(len(semantic_cache))
    stats["output_dir"] = str(output_root)
    _write_json(output_root / "semantic_gt_stats.json", stats)
    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an object-label GT dataset from Habitat semantic frames.")
    parser.add_argument("--spatial_db_dir", type=str, required=True, help="Existing spatial_db directory")
    parser.add_argument("--scene_path", type=str, required=True, help="Habitat scene .glb path")
    parser.add_argument("--semantic_txt_path", type=str, required=True, help="HM3D .semantic.txt annotation file")
    parser.add_argument(
        "--scene_dataset_config_file",
        type=str,
        default=None,
        help="Habitat scene dataset config with semantic asset settings",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output dataset directory")
    parser.add_argument(
        "--include_unknown",
        action="store_true",
        help="Include rows without a valid semantic majority label as gt_label='unknown'",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    stats = build_semantic_gt_dataset(
        spatial_db_dir=args.spatial_db_dir,
        semantic_txt_path=args.semantic_txt_path,
        output_dir=args.output_dir,
        scene_path=args.scene_path,
        scene_dataset_config_file=args.scene_dataset_config_file,
        include_unknown=bool(args.include_unknown),
    )
    print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
