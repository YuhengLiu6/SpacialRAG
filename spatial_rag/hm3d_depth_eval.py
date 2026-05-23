from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from spatial_rag.depth_stats import mask_depth_stats


DepthFrameProvider = Callable[[int, Mapping[str, Any]], np.ndarray]


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


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
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
        return [float(raw[0]), float(raw[1]), float(raw[2])]
    return [
        float(meta_row.get("x")),
        float(meta_row.get("y")),
        float(meta_row.get("z")),
    ]


def _orientation_from_meta(meta_row: Mapping[str, Any]) -> float:
    return float(meta_row.get("orientation"))


def _resolve_existing_path(spatial_db_dir: Path, raw_path: Any) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    raw = Path(text)
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                spatial_db_dir / raw,
                spatial_db_dir.parent / raw,
                Path.cwd() / raw,
                raw,
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image: {mask_path}")
    return np.asarray(mask > 0, dtype=bool)


def _normalize_depth_frame(depth_frame: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_frame, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth frame, got shape={depth.shape}")
    return depth


class HabitatDepthFrameProvider:
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
            require_semantics=False,
        )

    def __call__(self, entry_id: int, meta_row: Mapping[str, Any]) -> np.ndarray:
        del entry_id
        return self.explorer.capture_depth_at_pose(
            _position_from_meta(meta_row),
            _orientation_from_meta(meta_row),
        )


def _compute_error_record(row: Mapping[str, Any], gt_stats: Mapping[str, Any]) -> Dict[str, Any]:
    pred_trimmed = _safe_float(row.get("distance_from_camera_m"))
    pred_median = _safe_float(row.get("depth_stat_median_m"))
    gt_trimmed = _safe_float(gt_stats.get("trimmed_median_m"))
    gt_median = _safe_float(gt_stats.get("median_m"))
    signed_error_m = None
    abs_error_m = None
    relative_abs_error = None
    median_signed_error_m = None
    median_abs_error_m = None
    median_relative_abs_error = None
    if pred_trimmed is not None and gt_trimmed is not None:
        signed_error_m = float(pred_trimmed - gt_trimmed)
        abs_error_m = float(abs(signed_error_m))
        relative_abs_error = float(abs_error_m / max(gt_trimmed, 1e-6))
    if pred_median is not None and gt_median is not None:
        median_signed_error_m = float(pred_median - gt_median)
        median_abs_error_m = float(abs(median_signed_error_m))
        median_relative_abs_error = float(median_abs_error_m / max(gt_median, 1e-6))
    return {
        "pred_depth_trimmed_m": pred_trimmed,
        "pred_depth_median_m": pred_median,
        "gt_depth_trimmed_m": gt_trimmed,
        "gt_depth_median_m": gt_median,
        "gt_depth_p10_m": _safe_float(gt_stats.get("p10_m")),
        "gt_depth_p90_m": _safe_float(gt_stats.get("p90_m")),
        "gt_depth_num_valid_px": _safe_int(gt_stats.get("num_valid_px")),
        "signed_error_m": signed_error_m,
        "abs_error_m": abs_error_m,
        "relative_abs_error": relative_abs_error,
        "median_signed_error_m": median_signed_error_m,
        "median_abs_error_m": median_abs_error_m,
        "median_relative_abs_error": median_relative_abs_error,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    signed_errors = [
        float(row["signed_error_m"])
        for row in rows
        if _safe_float(row.get("signed_error_m")) is not None
    ]
    abs_errors = [
        float(row["abs_error_m"])
        for row in rows
        if _safe_float(row.get("abs_error_m")) is not None
    ]
    relative_errors = [
        float(row["relative_abs_error"])
        for row in rows
        if _safe_float(row.get("relative_abs_error")) is not None
    ]
    if not abs_errors:
        return {
            "count": 0,
            "mae_m": None,
            "rmse_m": None,
            "bias_m": None,
            "median_abs_error_m": None,
            "mean_relative_abs_error": None,
            "median_relative_abs_error": None,
        }
    mae_m = float(sum(abs_errors) / len(abs_errors))
    rmse_m = float(math.sqrt(sum(err * err for err in signed_errors) / len(signed_errors)))
    bias_m = float(sum(signed_errors) / len(signed_errors))
    mean_relative_abs_error = (
        float(sum(relative_errors) / len(relative_errors)) if relative_errors else None
    )
    median_relative_abs_error = (
        float(median(relative_errors)) if relative_errors else None
    )
    return {
        "count": int(len(abs_errors)),
        "mae_m": mae_m,
        "rmse_m": rmse_m,
        "bias_m": bias_m,
        "median_abs_error_m": float(median(abs_errors)),
        "mean_relative_abs_error": mean_relative_abs_error,
        "median_relative_abs_error": median_relative_abs_error,
    }


def _group_summaries(rows: Sequence[Mapping[str, Any]], field_name: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field_name) or "unknown")
        groups.setdefault(key, []).append(row)
    return {
        key: _summarize_rows(group_rows)
        for key, group_rows in sorted(groups.items(), key=lambda item: item[0])
    }


def evaluate_depth_dataset(
    *,
    spatial_db_dir: str | Path,
    output_dir: str | Path,
    scene_path: Optional[str] = None,
    scene_dataset_config_file: Optional[str] = None,
    depth_frame_provider: Optional[DepthFrameProvider] = None,
    max_objects: Optional[int] = None,
) -> Dict[str, Any]:
    spatial_db = Path(spatial_db_dir)
    output_root = Path(output_dir)
    object_rows = _read_jsonl(spatial_db / "object_meta.jsonl")
    meta_rows = _read_jsonl(spatial_db / "meta.jsonl")
    meta_by_entry = _meta_by_entry_id(meta_rows)

    if depth_frame_provider is None:
        if not scene_path:
            raise ValueError("scene_path is required when depth_frame_provider is not provided")
        depth_frame_provider = HabitatDepthFrameProvider(
            scene_path=str(scene_path),
            scene_dataset_config_file=scene_dataset_config_file,
        )

    evaluated_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    depth_cache: Dict[int, np.ndarray] = {}
    candidate_mask_depth_count = 0

    for source_index, row in enumerate(object_rows):
        geometry_source = str(row.get("geometry_source") or "").strip().lower()
        if geometry_source != "mask_depth":
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": row.get("entry_id"),
                    "file_name": row.get("file_name"),
                    "skip_reason": "geometry_source_not_mask_depth",
                    "geometry_source": row.get("geometry_source"),
                }
            )
            continue

        candidate_mask_depth_count += 1
        if max_objects is not None and len(evaluated_rows) >= int(max_objects):
            break

        entry_id = _safe_int(row.get("entry_id"))
        if entry_id is None:
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": row.get("entry_id"),
                    "file_name": row.get("file_name"),
                    "skip_reason": "missing_entry_id",
                    "geometry_source": row.get("geometry_source"),
                }
            )
            continue
        meta_row = meta_by_entry.get(int(entry_id))
        if meta_row is None:
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": int(entry_id),
                    "file_name": row.get("file_name"),
                    "skip_reason": "missing_meta_row",
                    "geometry_source": row.get("geometry_source"),
                }
            )
            continue

        mask_path = _resolve_existing_path(spatial_db, row.get("mask_path"))
        if mask_path is None:
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": int(entry_id),
                    "file_name": row.get("file_name"),
                    "skip_reason": "missing_mask_path",
                    "geometry_source": row.get("geometry_source"),
                    "mask_path": row.get("mask_path"),
                }
            )
            continue

        try:
            mask = _load_mask(mask_path)
        except Exception as exc:
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": int(entry_id),
                    "file_name": row.get("file_name"),
                    "skip_reason": f"mask_load_failed:{type(exc).__name__}",
                    "geometry_source": row.get("geometry_source"),
                    "mask_path": str(mask_path),
                }
            )
            continue
        if not np.any(mask):
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": int(entry_id),
                    "file_name": row.get("file_name"),
                    "skip_reason": "empty_mask",
                    "geometry_source": row.get("geometry_source"),
                    "mask_path": str(mask_path),
                }
            )
            continue

        try:
            if int(entry_id) not in depth_cache:
                depth_cache[int(entry_id)] = _normalize_depth_frame(
                    depth_frame_provider(int(entry_id), meta_row)
                )
            depth_frame = depth_cache[int(entry_id)]
        except Exception as exc:
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": int(entry_id),
                    "file_name": row.get("file_name"),
                    "skip_reason": f"depth_provider_failed:{type(exc).__name__}",
                    "geometry_source": row.get("geometry_source"),
                }
            )
            continue

        if depth_frame.shape != mask.shape:
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": int(entry_id),
                    "file_name": row.get("file_name"),
                    "skip_reason": "mask_depth_shape_mismatch",
                    "geometry_source": row.get("geometry_source"),
                    "mask_shape": list(mask.shape),
                    "depth_shape": list(depth_frame.shape),
                }
            )
            continue

        gt_stats = mask_depth_stats(depth_frame, mask)
        if _safe_float(gt_stats.get("trimmed_median_m")) is None:
            skipped_rows.append(
                {
                    "source_object_row_index": int(source_index),
                    "object_global_id": row.get("object_global_id"),
                    "entry_id": int(entry_id),
                    "file_name": row.get("file_name"),
                    "skip_reason": "no_valid_gt_depth_pixels",
                    "geometry_source": row.get("geometry_source"),
                    "mask_path": str(mask_path),
                }
            )
            continue

        out_row = {
            "source_object_row_index": int(source_index),
            "object_global_id": _safe_int(row.get("object_global_id")),
            "entry_id": int(entry_id),
            "file_name": str(row.get("file_name") or ""),
            "object_local_id": str(row.get("object_local_id") or ""),
            "final_label": str(row.get("final_label") or row.get("label") or "unknown"),
            "detector_label": row.get("detector_label"),
            "vlm_label": row.get("vlm_label"),
            "distance_bin": row.get("distance_bin"),
            "occlusion_level": row.get("occlusion_level"),
            "geometry_source": str(row.get("geometry_source") or ""),
            "mask_path": str(mask_path),
            "depth_map_path": row.get("depth_map_path"),
        }
        out_row.update(_compute_error_record(row, gt_stats))
        evaluated_rows.append(out_row)

    summary = {
        "source_object_count": int(len(object_rows)),
        "candidate_mask_depth_count": int(candidate_mask_depth_count),
        "evaluated_object_count": int(len(evaluated_rows)),
        "skipped_object_count": int(len(skipped_rows)),
        "depth_frame_count": int(len(depth_cache)),
        "scene_path": None if scene_path is None else str(scene_path),
        "scene_dataset_config_file": None if scene_dataset_config_file is None else str(scene_dataset_config_file),
        "overall": _summarize_rows(evaluated_rows),
        "by_final_label": _group_summaries(evaluated_rows, "final_label"),
        "by_distance_bin": _group_summaries(evaluated_rows, "distance_bin"),
        "by_occlusion_level": _group_summaries(evaluated_rows, "occlusion_level"),
        "output_dir": str(output_root),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "depth_eval_rows.jsonl", evaluated_rows)
    _write_jsonl(output_root / "depth_eval_skipped.jsonl", skipped_rows)
    _write_json(output_root / "depth_eval_summary.json", summary)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare HM3D Habitat sensor depth against stored Depth Pro object depth summaries."
    )
    parser.add_argument("--spatial_db_dir", type=str, required=True, help="Existing spatial_db directory")
    parser.add_argument("--scene_path", type=str, required=True, help="Habitat scene .glb path")
    parser.add_argument(
        "--scene_dataset_config_file",
        type=str,
        default=None,
        help="Habitat scene dataset config with semantic/depth asset settings",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for evaluation artifacts")
    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="Optional cap on the number of mask_depth objects to evaluate",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = evaluate_depth_dataset(
        spatial_db_dir=args.spatial_db_dir,
        output_dir=args.output_dir,
        scene_path=args.scene_path,
        scene_dataset_config_file=args.scene_dataset_config_file,
        max_objects=args.max_objects,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()