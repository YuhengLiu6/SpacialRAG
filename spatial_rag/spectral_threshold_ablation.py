from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from spatial_rag.config import OCCLUSION_REWEIGHT_B, OCCLUSION_REWEIGHT_W1, OCCLUSION_REWEIGHT_W2
from spatial_rag.object_instance_clustering import run_object_instance_clustering
from spatial_rag.reweight_sweep import run_reweight_sweep
from spatial_rag.sequential_spectral_experiment import run_sequential_spectral_experiment

DEFAULT_THRESHOLDS = (None, 0.0, 0.2, 0.4, 0.6, 0.8)
DEFAULT_BATCH_TEXT_MODE = "long"
DEFAULT_BATCH_CLUSTER_COUNT_MODE = "eigengap"
DEFAULT_BATCH_SAME_VIEW_POLICY = "soft_penalty"
DEFAULT_BATCH_SAME_VIEW_PENALTY = 0.25


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return float(out)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_entry_ids(value: Any) -> List[int]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    out: List[int] = []
    for item in parts:
        parsed = _safe_float(item)
        if parsed is None:
            raise ValueError(f"Invalid entry id: {item!r}")
        out.append(int(parsed))
    if not out:
        raise ValueError("Expected at least one entry id")
    return out


def _parse_thresholds(value: Any) -> List[Optional[float]]:
    if value in (None, "", []):
        return list(DEFAULT_THRESHOLDS)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    out: List[Optional[float]] = []
    for item in parts:
        token = _safe_text(item).lower()
        if token in {"none", "null"}:
            out.append(None)
            continue
        parsed = _safe_float(item)
        if parsed is None:
            raise ValueError(f"Invalid threshold value: {item!r}")
        out.append(float(parsed))
    if not out:
        raise ValueError("Expected at least one threshold")
    return out


def _normalize_number_token(value: Optional[float]) -> str:
    if value is None:
        return "none"
    token = f"{float(value):g}"
    return token.replace("-", "neg_").replace(".", "p")


def _threshold_token(value: Optional[float]) -> str:
    return _normalize_number_token(value)


def _make_run_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S")
    candidate = base_dir / stamp
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _mean_label_purity(clusters: Sequence[Mapping[str, Any]]) -> float:
    purities: List[float] = []
    for cluster in clusters:
        histogram = dict(cluster.get("label_histogram") or {})
        total = sum(int(value) for value in histogram.values())
        if total <= 0:
            continue
        purities.append(float(max(int(value) for value in histogram.values()) / float(total)))
    if not purities:
        return 0.0
    return float(sum(purities) / len(purities))


def _same_view_collision_count(clusters: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for cluster in clusters:
        if bool(cluster.get("same_view_collision")):
            count += 1
            continue
        view_ids = [_safe_text(view_id) for view_id in cluster.get("member_view_ids", []) if _safe_text(view_id)]
        if not view_ids:
            continue
        if any(value > 1 for value in Counter(view_ids).values()):
            count += 1
    return int(count)


def _cluster_metrics(
    *,
    total_objects: int,
    clusters: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    num_clusters = int(len(clusters))
    singleton_clusters = int(sum(1 for cluster in clusters if _safe_int(cluster.get("num_members"), len(cluster.get("member_view_ids", []))) == 1))
    same_view_collision_count = _same_view_collision_count(clusters)
    return {
        "total_objects": int(total_objects),
        "num_clusters": num_clusters,
        "avg_cluster_size": float(total_objects / max(num_clusters, 1)) if num_clusters else 0.0,
        "singleton_cluster_rate": float(singleton_clusters / max(num_clusters, 1)) if num_clusters else 0.0,
        "same_view_collision_count": int(same_view_collision_count),
        "same_view_collision_cluster_rate": float(same_view_collision_count / max(num_clusters, 1)) if num_clusters else 0.0,
        "mean_label_purity": _mean_label_purity(clusters),
    }


def _extract_batch_metrics(object_instance_root: Path) -> Dict[str, Any]:
    group_dir = object_instance_root / "selected_views"
    cluster_summary = _load_json(group_dir / "cluster_summary.json")
    clusters = list(cluster_summary.get("clusters") or [])
    metrics = _cluster_metrics(
        total_objects=_safe_int(cluster_summary.get("n_objects"), 0),
        clusters=clusters,
    )
    metrics["group_dir"] = str(group_dir)
    metrics["cluster_summary_path"] = str(group_dir / "cluster_summary.json")
    return metrics


def _extract_sequential_metrics(report: Mapping[str, Any]) -> Dict[str, Any]:
    clusters = list(report.get("final_clusters") or [])
    total_objects = int(sum(sum(int(value) for value in dict(cluster.get("label_histogram") or {}).values()) for cluster in clusters))
    metrics = _cluster_metrics(total_objects=total_objects, clusters=clusters)
    metrics["total_appended"] = int(report.get("total_appended") or 0)
    metrics["total_current_only_reattached"] = int(report.get("total_current_only_reattached") or 0)
    metrics["total_new_tail_clusters"] = int(report.get("total_new_tail_clusters") or 0)
    metrics["total_merged_clusters"] = int(report.get("total_merged_clusters") or 0)
    metrics["total_same_view_blocked_components"] = int(report.get("total_same_view_blocked_components") or 0)
    return metrics


def _symlink_dir(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(destination), target_is_directory=True)


def _infer_reweight_defaults(base_root: Path) -> Dict[str, float]:
    build_report = _load_json(base_root / "build_report.json")
    occlusion = dict((build_report.get("object_config") or {}).get("occlusion_reweight") or {})
    return {
        "w1": float(_safe_float(occlusion.get("w1")) if _safe_float(occlusion.get("w1")) is not None else OCCLUSION_REWEIGHT_W1),
        "w2": float(_safe_float(occlusion.get("w2")) if _safe_float(occlusion.get("w2")) is not None else OCCLUSION_REWEIGHT_W2),
        "b": float(_safe_float(occlusion.get("b")) if _safe_float(occlusion.get("b")) is not None else OCCLUSION_REWEIGHT_B),
    }


def _markdown_summary(results: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Threshold Ablation",
        "",
        "| threshold | kept_objects | batch_clusters | batch_collision_count | batch_purity | seq_clusters | seq_collision_count | seq_purity | seq_merged | seq_reattached |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    _safe_text(row.get("threshold_display")),
                    str(_safe_int(row.get("kept_total_objects"), 0)),
                    str(_safe_int(row.get("batch_num_clusters"), 0)),
                    str(_safe_int(row.get("batch_same_view_collision_count"), 0)),
                    f"{float(row.get('batch_mean_label_purity') or 0.0):.3f}",
                    str(_safe_int(row.get("sequential_num_clusters"), 0)),
                    str(_safe_int(row.get("sequential_same_view_collision_count"), 0)),
                    f"{float(row.get('sequential_mean_label_purity') or 0.0):.3f}",
                    str(_safe_int(row.get("sequential_total_merged_clusters"), 0)),
                    str(_safe_int(row.get("sequential_total_current_only_reattached"), 0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines).strip() + "\n"


def run_spectral_threshold_ablation(
    *,
    db_dir: str,
    entry_ids: Sequence[int],
    thresholds: Sequence[Optional[float]] = DEFAULT_THRESHOLDS,
    output_dir: Optional[str] = None,
    w1: Optional[float] = None,
    w2: Optional[float] = None,
    b: Optional[float] = None,
    export_filtered_objects: bool = False,
    filtered_object_dirname: str = "filtered_obj",
    batch_text_mode: str = DEFAULT_BATCH_TEXT_MODE,
    batch_cluster_count_mode: str = DEFAULT_BATCH_CLUSTER_COUNT_MODE,
    batch_same_view_policy: str = DEFAULT_BATCH_SAME_VIEW_POLICY,
    batch_same_view_penalty: float = DEFAULT_BATCH_SAME_VIEW_PENALTY,
    weight_text: Optional[float] = None,
    weight_dinov3: Optional[float] = None,
    enable_dinov3_scoring: Optional[bool] = None,
    enable_vlm_compress: Optional[bool] = None,
    enable_vlm_member_spatial: Optional[bool] = None,
    distance_gate_dsq0: Optional[float] = None,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: Optional[int] = None,
    enforce_same_view_uniqueness: Optional[bool] = None,
) -> Dict[str, Any]:
    base_root = Path(db_dir).expanduser().resolve()
    if not base_root.exists():
        raise FileNotFoundError(f"DB directory does not exist: {base_root}")
    ordered_entry_ids = [int(entry_id) for entry_id in entry_ids]
    if len(ordered_entry_ids) < 2:
        raise ValueError("Need at least two ordered entry_ids for threshold ablation")

    inferred = _infer_reweight_defaults(base_root)
    resolved_w1 = float(inferred["w1"] if w1 is None else w1)
    resolved_w2 = float(inferred["w2"] if w2 is None else w2)
    resolved_b = float(inferred["b"] if b is None else b)
    normalized_thresholds = list(thresholds)

    output_base = Path(output_dir).expanduser().resolve() if output_dir else (base_root / "spectral_threshold_ablation")
    run_root = _make_run_output_dir(output_base)

    sweep_report = run_reweight_sweep(
        str(base_root),
        w1_values=[resolved_w1],
        w2_values=[resolved_w2],
        b_values=[resolved_b],
        thresholds=normalized_thresholds,
        export_db_variants=True,
        export_filtered_objects=bool(export_filtered_objects),
        filtered_object_dirname=filtered_object_dirname,
        output_dir=str(run_root / "_reweight_sweep"),
    )

    result_rows: List[Dict[str, Any]] = []
    threshold_runs: List[Dict[str, Any]] = []
    for sweep_run in list(sweep_report.get("runs") or []):
        threshold_value = _safe_float(sweep_run.get("threshold"))
        if sweep_run.get("threshold") is None:
            threshold_value = None
        token = _threshold_token(threshold_value)
        threshold_root = run_root / f"threshold_{token}"
        threshold_root.mkdir(parents=True, exist_ok=False)

        exported_db_dir = Path(_safe_text(sweep_run.get("exported_db_dir"))).expanduser().resolve()
        if not exported_db_dir.exists():
            raise FileNotFoundError(f"Exported DB variant missing for threshold {token}: {exported_db_dir}")
        db_variant_link = threshold_root / "db_variant"
        _symlink_dir(exported_db_dir, db_variant_link)

        object_instance_root = threshold_root / "object_instance"
        batch_report = run_object_instance_clustering(
            output_dir=str(object_instance_root),
            group_mode="selected_views",
            db_dir=str(db_variant_link),
            entry_ids=ordered_entry_ids,
            text_mode=batch_text_mode,
            cluster_count_mode=batch_cluster_count_mode,
            same_view_policy=batch_same_view_policy,
            same_view_penalty=float(batch_same_view_penalty),
        )

        sequential_parent = threshold_root / "sequential"
        sequential_kwargs: Dict[str, Any] = {
            "db_dir": str(db_variant_link),
            "output_dir": str(sequential_parent),
            "entry_ids": ordered_entry_ids,
        }
        if weight_text is not None:
            sequential_kwargs["weight_text"] = float(weight_text)
        if weight_dinov3 is not None:
            sequential_kwargs["weight_dinov3"] = float(weight_dinov3)
        if enable_dinov3_scoring is not None:
            sequential_kwargs["enable_dinov3_scoring"] = bool(enable_dinov3_scoring)
        if enable_vlm_compress is not None:
            sequential_kwargs["enable_vlm_compress"] = bool(enable_vlm_compress)
        if enable_vlm_member_spatial is not None:
            sequential_kwargs["enable_vlm_member_spatial"] = bool(enable_vlm_member_spatial)
        if distance_gate_dsq0 is not None:
            sequential_kwargs["distance_gate_dsq0"] = float(distance_gate_dsq0)
        if dbscan_eps is not None:
            sequential_kwargs["dbscan_eps"] = float(dbscan_eps)
        if dbscan_min_samples is not None:
            sequential_kwargs["dbscan_min_samples"] = int(dbscan_min_samples)
        if enforce_same_view_uniqueness is not None:
            sequential_kwargs["enforce_same_view_uniqueness"] = bool(enforce_same_view_uniqueness)
        sequential_report = run_sequential_spectral_experiment(**sequential_kwargs)

        batch_metrics = _extract_batch_metrics(object_instance_root)
        sequential_metrics = _extract_sequential_metrics(sequential_report)
        sequential_output_root = Path(_safe_text(sequential_report.get("output_dir"))).expanduser().resolve()

        threshold_summary = {
            "threshold": threshold_value,
            "threshold_display": "none" if threshold_value is None else f"{float(threshold_value):g}",
            "threshold_token": token,
            "db_variant_root": str(db_variant_link),
            "db_variant_export_root": str(exported_db_dir),
            "filtered_object_dir": _safe_text(sweep_run.get("filtered_object_dir")),
            "filtered_manifest_path": _safe_text(sweep_run.get("filtered_manifest_path")),
            "filtered_object_count": int(sweep_run.get("filtered_object_count") or 0),
            "object_instance_native_output_root": str(object_instance_root),
            "object_instance_group_root": str(object_instance_root / "selected_views"),
            "sequential_native_output_parent": str(sequential_parent),
            "sequential_native_output_root": str(sequential_output_root),
            "kept_total_objects": int(sweep_run.get("kept_total_objects") or 0),
            "dropped_total_objects": int(sweep_run.get("dropped_total_objects") or 0),
            "empty_frame_count": int(sweep_run.get("empty_frame_count") or 0),
            "geometry_filtered_count": int(sweep_run.get("geometry_filtered_count") or 0),
            "keep_rate": float(sweep_run.get("keep_rate") or 0.0),
            "filtered_label_counts": dict(sweep_run.get("filtered_label_counts") or {}),
            "batch": batch_metrics,
            "sequential": sequential_metrics,
            "batch_report": batch_report,
            "sequential_report": {
                "output_dir": str(sequential_output_root),
                "final_cluster_count": int(sequential_report.get("final_cluster_count") or 0),
                "weight_text": _safe_float((sequential_report.get("weights") or {}).get("text")),
                "weight_dinov3": _safe_float((sequential_report.get("weights") or {}).get("dinov3")),
                "enable_dinov3_scoring": bool(sequential_report.get("enable_dinov3_scoring")),
                "enable_vlm_compress": bool(sequential_report.get("enable_vlm_compress")),
                "enable_vlm_member_spatial": bool(sequential_report.get("enable_vlm_member_spatial")),
                "distance_gate_dsq0": _safe_float(sequential_report.get("distance_gate_dsq0")),
                "dbscan_eps": sequential_report.get("dbscan_eps"),
                "dbscan_min_samples": int(sequential_report.get("dbscan_min_samples") or 0),
                "enforce_same_view_uniqueness": bool(sequential_report.get("enforce_same_view_uniqueness")),
            },
        }
        threshold_runs.append(threshold_summary)

        flat_row = {
            "threshold": threshold_summary["threshold_display"],
            "threshold_token": token,
            "kept_total_objects": threshold_summary["kept_total_objects"],
            "dropped_total_objects": threshold_summary["dropped_total_objects"],
            "empty_frame_count": threshold_summary["empty_frame_count"],
            "geometry_filtered_count": threshold_summary["geometry_filtered_count"],
            "keep_rate": threshold_summary["keep_rate"],
            "db_variant_root": threshold_summary["db_variant_root"],
            "filtered_object_dir": threshold_summary["filtered_object_dir"],
            "filtered_manifest_path": threshold_summary["filtered_manifest_path"],
            "filtered_object_count": threshold_summary["filtered_object_count"],
            "object_instance_native_output_root": threshold_summary["object_instance_native_output_root"],
            "sequential_native_output_root": threshold_summary["sequential_native_output_root"],
            "batch_total_objects": batch_metrics["total_objects"],
            "batch_num_clusters": batch_metrics["num_clusters"],
            "batch_avg_cluster_size": batch_metrics["avg_cluster_size"],
            "batch_singleton_cluster_rate": batch_metrics["singleton_cluster_rate"],
            "batch_same_view_collision_count": batch_metrics["same_view_collision_count"],
            "batch_mean_label_purity": batch_metrics["mean_label_purity"],
            "sequential_total_objects": sequential_metrics["total_objects"],
            "sequential_num_clusters": sequential_metrics["num_clusters"],
            "sequential_avg_cluster_size": sequential_metrics["avg_cluster_size"],
            "sequential_singleton_cluster_rate": sequential_metrics["singleton_cluster_rate"],
            "sequential_same_view_collision_count": sequential_metrics["same_view_collision_count"],
            "sequential_mean_label_purity": sequential_metrics["mean_label_purity"],
            "sequential_total_appended": sequential_metrics["total_appended"],
            "sequential_total_current_only_reattached": sequential_metrics["total_current_only_reattached"],
            "sequential_total_new_tail_clusters": sequential_metrics["total_new_tail_clusters"],
            "sequential_total_merged_clusters": sequential_metrics["total_merged_clusters"],
            "sequential_total_same_view_blocked_components": sequential_metrics["total_same_view_blocked_components"],
            "sequential_weight_text": _safe_float((sequential_report.get("weights") or {}).get("text")),
            "sequential_weight_dinov3": _safe_float((sequential_report.get("weights") or {}).get("dinov3")),
            "sequential_enable_dinov3_scoring": bool(sequential_report.get("enable_dinov3_scoring")),
            "sequential_enable_vlm_compress": bool(sequential_report.get("enable_vlm_compress")),
            "sequential_enable_vlm_member_spatial": bool(sequential_report.get("enable_vlm_member_spatial")),
            "sequential_distance_gate_dsq0": _safe_float(sequential_report.get("distance_gate_dsq0")),
            "sequential_dbscan_eps": sequential_report.get("dbscan_eps"),
            "sequential_dbscan_min_samples": int(sequential_report.get("dbscan_min_samples") or 0),
            "sequential_enforce_same_view_uniqueness": bool(sequential_report.get("enforce_same_view_uniqueness")),
        }
        result_rows.append(flat_row)

    summary = {
        "db_dir": str(base_root),
        "output_dir": str(run_root),
        "entry_ids": ordered_entry_ids,
        "thresholds": [None if value is None else float(value) for value in normalized_thresholds],
        "reweight": {"w1": resolved_w1, "w2": resolved_w2, "b": resolved_b},
        "filtered_object_exports": {
            "enabled": bool(export_filtered_objects),
            "dirname": filtered_object_dirname,
        },
        "batch_defaults": {
            "group_mode": "selected_views",
            "text_mode": batch_text_mode,
            "cluster_count_mode": batch_cluster_count_mode,
            "same_view_policy": batch_same_view_policy,
            "same_view_penalty": float(batch_same_view_penalty),
        },
        "sequential_overrides": {
            "weight_text": None if weight_text is None else float(weight_text),
            "weight_dinov3": None if weight_dinov3 is None else float(weight_dinov3),
            "enable_dinov3_scoring": None if enable_dinov3_scoring is None else bool(enable_dinov3_scoring),
            "enable_vlm_compress": None if enable_vlm_compress is None else bool(enable_vlm_compress),
            "enable_vlm_member_spatial": (
                None if enable_vlm_member_spatial is None else bool(enable_vlm_member_spatial)
            ),
            "distance_gate_dsq0": None if distance_gate_dsq0 is None else float(distance_gate_dsq0),
            "dbscan_eps": None if dbscan_eps is None else float(dbscan_eps),
            "dbscan_min_samples": None if dbscan_min_samples is None else int(dbscan_min_samples),
            "enforce_same_view_uniqueness": (
                None if enforce_same_view_uniqueness is None else bool(enforce_same_view_uniqueness)
            ),
        },
        "reweight_sweep_output_dir": str(sweep_report.get("output_dir") or ""),
        "threshold_runs": threshold_runs,
    }
    _write_json(run_root / "summary.json", summary)
    _write_csv_rows(
        run_root / "threshold_results.csv",
        [
            "threshold",
            "threshold_token",
            "kept_total_objects",
            "dropped_total_objects",
            "empty_frame_count",
            "geometry_filtered_count",
            "keep_rate",
            "db_variant_root",
            "filtered_object_dir",
            "filtered_manifest_path",
            "filtered_object_count",
            "object_instance_native_output_root",
            "sequential_native_output_root",
            "batch_total_objects",
            "batch_num_clusters",
            "batch_avg_cluster_size",
            "batch_singleton_cluster_rate",
            "batch_same_view_collision_count",
            "batch_mean_label_purity",
            "sequential_total_objects",
            "sequential_num_clusters",
            "sequential_avg_cluster_size",
            "sequential_singleton_cluster_rate",
            "sequential_same_view_collision_count",
            "sequential_mean_label_purity",
            "sequential_total_appended",
            "sequential_total_current_only_reattached",
            "sequential_total_new_tail_clusters",
            "sequential_total_merged_clusters",
            "sequential_total_same_view_blocked_components",
            "sequential_weight_text",
            "sequential_weight_dinov3",
            "sequential_enable_dinov3_scoring",
            "sequential_enable_vlm_compress",
            "sequential_enable_vlm_member_spatial",
            "sequential_distance_gate_dsq0",
            "sequential_dbscan_eps",
            "sequential_dbscan_min_samples",
            "sequential_enforce_same_view_uniqueness",
        ],
        result_rows,
    )
    (run_root / "threshold_results.md").write_text(_markdown_summary(result_rows), encoding="utf-8")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep object filter thresholds and run native spectral pipelines.")
    parser.add_argument("--db_dir", type=str, required=True, help="Canonical base DB directory.")
    parser.add_argument("--entry_ids", type=str, required=True, help="Comma-separated ordered entry ids.")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="none,0.0,0.2,0.4,0.6,0.8",
        help="Comma-separated thresholds. Use 'none' to disable filtering for a config.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Optional ablation output root.")
    parser.add_argument("--w1", type=float, default=None, help="Override occlusion reweight w1.")
    parser.add_argument("--w2", type=float, default=None, help="Override occlusion reweight w2.")
    parser.add_argument("--b", type=float, default=None, help="Override occlusion reweight bias.")
    parser.add_argument(
        "--export_filtered_objects",
        type=lambda value: str(value).strip().lower() in {"1", "true", "t", "yes", "y"},
        default=False,
        help="Whether to export filtered object crops under each threshold root.",
    )
    parser.add_argument(
        "--filtered_object_dirname",
        type=str,
        default="filtered_obj",
        help="Directory name used under each threshold root for filtered object crops.",
    )
    parser.add_argument(
        "--weight_text",
        type=float,
        default=None,
        help="Optional text weight override passed through to sequential_spectral_experiment.",
    )
    parser.add_argument(
        "--weight_dinov3",
        type=float,
        default=None,
        help="Optional DINOv3 weight override passed through to sequential_spectral_experiment.",
    )
    parser.add_argument(
        "--enable_dinov3_scoring",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional DINOv3 scoring override passed through to sequential_spectral_experiment.",
    )
    parser.add_argument(
        "--enable_vlm_compress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional VLM cluster-text compression override passed through to sequential_spectral_experiment.",
    )
    parser.add_argument(
        "--enable_vlm_member_spatial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional member-only spatial cue override passed through to sequential_spectral_experiment.",
    )
    parser.add_argument(
        "--distance_gate_dsq0",
        type=float,
        default=None,
        help="Optional distance-gate dsq0 override passed through to sequential_spectral_experiment.",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=None,
        help="Optional DBSCAN eps override passed through to sequential_spectral_experiment.",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=None,
        help="Optional DBSCAN min_samples override passed through to sequential_spectral_experiment.",
    )
    parser.add_argument(
        "--enforce_same_view_uniqueness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional same-view uniqueness override passed through to sequential_spectral_experiment.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = run_spectral_threshold_ablation(
        db_dir=args.db_dir,
        entry_ids=_parse_entry_ids(args.entry_ids),
        thresholds=_parse_thresholds(args.thresholds),
        output_dir=args.output_dir,
        w1=args.w1,
        w2=args.w2,
        b=args.b,
        export_filtered_objects=bool(args.export_filtered_objects),
        filtered_object_dirname=args.filtered_object_dirname,
        weight_text=args.weight_text,
        weight_dinov3=args.weight_dinov3,
        enable_dinov3_scoring=args.enable_dinov3_scoring,
        enable_vlm_compress=args.enable_vlm_compress,
        enable_vlm_member_spatial=args.enable_vlm_member_spatial,
        distance_gate_dsq0=args.distance_gate_dsq0,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        enforce_same_view_uniqueness=args.enforce_same_view_uniqueness,
    )
    print(json.dumps(report, ensure_ascii=True))
    return report


if __name__ == "__main__":
    main()
