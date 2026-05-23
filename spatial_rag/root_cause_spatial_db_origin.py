from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from spatial_rag.object_index import load_object_db
from spatial_rag.sequential_spectral_experiment import (
    _apply_same_view_hard_mask_to_cross_affinity,
    _build_cluster,
    _full_bipartite_affinity,
    _run_capped_sequential_spectral_clustering,
    apply_incremental_step,
    build_cross_affinity_matrix,
)


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    run_label: str
    db_dir: Path
    run_dir: Path
    selected_views_root: Path
    weight_text: float
    weight_dinov2: float
    weight_global_geo: float = 0.2
    weight_polar: float = 0.1
    similarity_mode: str = "cosine_geo_gate"
    distance_gate_dsq0: float = 4.0
    min_cross_affinity: float = 0.5
    dbscan_min_samples: int = 2
    current_only_reattach_min_affinity: float = 0.75
    enforce_same_view_uniqueness: bool = True


@dataclass
class CaseRecord:
    case_id: str
    case_bucket: str
    run_name: str
    run_label: str
    step_index: Optional[int]
    view_id: Optional[str]
    query_object_id: Optional[int]
    query_label: str
    canonical_instance_id: str
    canonical_status: str
    final_outcome: str
    assignment_reason: str
    assigned_cluster_id: Optional[int]
    correct_cluster_ids: List[int]
    correct_match_object_id: Optional[int]
    wrong_match_object_id: Optional[int]
    wrong_match_cluster_id: Optional[int]
    first_error_stage: str
    primary_signal_family: str
    root_cause_label: str
    primary_cause: str
    secondary_cause: str
    explanation: str
    question_answer: str
    query_crop_path: Optional[str]
    query_mask_overlay_path: Optional[str]
    query_frame_path: Optional[str]
    wrong_crop_path: Optional[str]
    correct_crop_path: Optional[str]
    panel_path: Optional[str]
    camera_global_x: Optional[float]
    camera_global_y: Optional[float]
    camera_orientation_deg: Optional[float]
    query_global_x: Optional[float]
    query_global_y: Optional[float]
    wrong_match_global_x: Optional[float]
    wrong_match_global_y: Optional[float]
    correct_match_global_x: Optional[float]
    correct_match_global_y: Optional[float]
    wrong_match_view_id: str
    correct_match_view_id: str
    text_similarity_correct: Optional[float]
    text_similarity_wrong: Optional[float]
    dinov2_similarity_correct: Optional[float]
    dinov2_similarity_wrong: Optional[float]
    semantic_similarity_correct: Optional[float]
    semantic_similarity_wrong: Optional[float]
    combined_similarity_correct: Optional[float]
    combined_similarity_wrong: Optional[float]
    distance_gate_correct: Optional[float]
    distance_gate_wrong: Optional[float]
    xy_distance_correct_m: Optional[float]
    xy_distance_wrong_m: Optional[float]
    query_distance_from_camera_m: Optional[float]
    query_relative_bearing_deg: Optional[float]
    query_object_orientation_deg: Optional[float]
    query_angle_bucket: str
    query_occlusion_level: str
    query_foreground_occluder_count: Optional[int]
    query_visible_occlusion_ratio: Optional[float]
    query_depth_p10_m: Optional[float]
    query_depth_p90_m: Optional[float]
    query_depth_spread_m: Optional[float]
    top_semantic_cluster_id: Optional[int]
    top_combined_cluster_id: Optional[int]
    top_combined_score: Optional[float]
    top_correct_combined_score: Optional[float]
    top_correct_semantic_score: Optional[float]
    combined_threshold: float
    correct_candidate_rank_by_semantic: Optional[int]
    correct_candidate_rank_by_combined: Optional[int]
    wrong_candidate_rank_by_semantic: Optional[int]
    wrong_candidate_rank_by_combined: Optional[int]
    candidate_count: int
    notes: str


DEFAULT_OUTPUT_DIR = Path("analysis/root_cause_spatial_db_origin")
SELECTED_VIEW_IDS = ("view_00015", "view_00019", "view_00023", "view_00027")

RUN_SPECS: Dict[str, RunSpec] = {
    "run010": RunSpec(
        run_name="run010",
        run_label="threshold_ablation_010 (text=0.0, DINO=1.0)",
        db_dir=Path("evaluation/threshold_ablation_010/run_20260420_162756/threshold_0p4/db_variant"),
        run_dir=Path("evaluation/threshold_ablation_010/run_20260420_162756/threshold_0p4/sequential/run_20260420_162806"),
        selected_views_root=Path("evaluation/threshold_ablation_010/run_20260420_162756/threshold_0p4/object_instance/selected_views"),
        weight_text=0.0,
        weight_dinov2=1.0,
    ),
    "run55": RunSpec(
        run_name="run55",
        run_label="threshold_ablation_55 (text=0.5, DINO=0.5)",
        db_dir=Path("evaluation/threshold_ablation_55/run_20260420_162635/threshold_0p4/db_variant"),
        run_dir=Path("evaluation/threshold_ablation_55/run_20260420_162635/threshold_0p4/sequential/run_20260420_162646"),
        selected_views_root=Path("evaluation/threshold_ablation_55/run_20260420_162635/threshold_0p4/object_instance/selected_views"),
        weight_text=0.5,
        weight_dinov2=0.5,
    ),
}

DIAGNOSTIC_DB_DIRS: Dict[str, Path] = {
    "run010": RUN_SPECS["run010"].db_dir,
    "run55": RUN_SPECS["run55"].db_dir,
    "spatial_db_origin": Path("spatial_db_origin"),
}

MANUAL_CANONICAL_ENTRIES: Dict[int, Dict[str, str]] = {
    76: {"canonical_instance_id": "oven_main", "status": "clear", "confidence": "high", "notes": "Single oven observation."},
    79: {"canonical_instance_id": "island_main", "status": "clear", "confidence": "high", "notes": "Kitchen island / tabletop instance."},
    80: {"canonical_instance_id": "seat_a", "status": "clear", "confidence": "medium", "notes": "Primary recurring wooden seat / stool."},
    82: {"canonical_instance_id": "fridge_main", "status": "clear", "confidence": "high", "notes": "Recurring double-door refrigerator."},
    83: {"canonical_instance_id": "lamp_main", "status": "clear", "confidence": "high", "notes": "Single foreground table lamp."},
    86: {"canonical_instance_id": "rug_main", "status": "clear", "confidence": "high", "notes": "Single floor mat observation."},
    96: {"canonical_instance_id": "coffee_black", "status": "clear", "confidence": "high", "notes": "Black drip coffee maker."},
    117: {"canonical_instance_id": "fridge_main", "status": "clear", "confidence": "high", "notes": "Recurring refrigerator observation."},
    118: {"canonical_instance_id": "picture_frame_main", "status": "clear", "confidence": "high", "notes": "Single wall picture observation."},
    121: {"canonical_instance_id": "seat_a", "status": "clear", "confidence": "medium", "notes": "Primary recurring wooden seat / stool."},
    122: {"canonical_instance_id": "island_main", "status": "clear", "confidence": "high", "notes": "Kitchen island / tabletop instance."},
    126: {"canonical_instance_id": "clock_main", "status": "clear", "confidence": "high", "notes": "Recurring decorative wall clock."},
    127: {"canonical_instance_id": "coffee_black", "status": "clear", "confidence": "high", "notes": "Black drip coffee maker."},
    128: {"canonical_instance_id": "seat_b", "status": "clear", "confidence": "medium", "notes": "Second recurring chair instance."},
    131: {"canonical_instance_id": "coffee_silver", "status": "clear", "confidence": "high", "notes": "Silver coffee maker on right counter."},
    166: {"canonical_instance_id": "fridge_main", "status": "clear", "confidence": "high", "notes": "Recurring refrigerator observation."},
    167: {"canonical_instance_id": "seat_a", "status": "clear", "confidence": "medium", "notes": "Primary recurring wooden seat / stool."},
    169: {"canonical_instance_id": "island_main", "status": "clear", "confidence": "high", "notes": "Kitchen island / tabletop instance."},
    172: {"canonical_instance_id": "clock_main", "status": "clear", "confidence": "high", "notes": "Recurring decorative wall clock."},
    176: {"canonical_instance_id": "seat_b", "status": "clear", "confidence": "medium", "notes": "Second recurring chair instance."},
    198: {"canonical_instance_id": "fridge_main", "status": "clear", "confidence": "high", "notes": "Recurring refrigerator observation."},
    201: {"canonical_instance_id": "coffee_single", "status": "clear", "confidence": "high", "notes": "Black single-serve coffee maker, should form a new cluster."},
    203: {"canonical_instance_id": "dishwasher_main", "status": "clear", "confidence": "high", "notes": "Single dishwasher observation."},
    205: {"canonical_instance_id": "island_main", "status": "clear", "confidence": "high", "notes": "Kitchen island / tabletop instance."},
    207: {"canonical_instance_id": "toaster_main", "status": "clear", "confidence": "high", "notes": "Single toaster observation."},
}

MAIN_CASE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "run010_obj169": {
        "first_error_stage": "similarity",
        "primary_signal_family": "interaction",
        "root_cause_label": "compounded_error",
        "primary_cause": "compounded_error",
        "secondary_cause": "visual_similarity_error",
        "question_answer": "DINO slightly preferred a cabinet over the correct island, and both candidates then fell below the hard 0.5 cross-affinity threshold.",
    },
    "run55_obj169": {
        "first_error_stage": "geometry",
        "primary_signal_family": "geometry",
        "root_cause_label": "similarity_correct_but_geometry_wrong",
        "primary_cause": "depth_error",
        "secondary_cause": "compounded_error",
        "question_answer": "Text rescued the semantic ranking, but geometry still kept the correct island below the threshold needed to survive into the spectral graph.",
    },
    "run010_obj176": {
        "first_error_stage": "similarity",
        "primary_signal_family": "similarity",
        "root_cause_label": "text_visual_both_wrong",
        "primary_cause": "text_visual_both_wrong",
        "secondary_cause": "compounded_error",
        "question_answer": "Both text and DINO favored the already-seen seat_a cluster over the correct seat_b cluster, then same-view uniqueness blocked the bad merge and left a false split.",
    },
    "run010_obj198": {
        "first_error_stage": "geometry",
        "primary_signal_family": "geometry",
        "root_cause_label": "similarity_correct_but_geometry_wrong",
        "primary_cause": "depth_error",
        "secondary_cause": "",
        "question_answer": "The refrigerator match stayed semantically strongest, but the geometry gate pushed the correct edge below 0.5, so the graph never saw it.",
    },
    "run55_obj198": {
        "first_error_stage": "geometry",
        "primary_signal_family": "geometry",
        "root_cause_label": "similarity_correct_but_geometry_wrong",
        "primary_cause": "depth_error",
        "secondary_cause": "",
        "question_answer": "Even with text included, the fridge edge died at the geometry-threshold stage rather than the similarity stage.",
    },
    "run010_obj201": {
        "first_error_stage": "similarity",
        "primary_signal_family": "similarity",
        "root_cause_label": "text_visual_both_wrong",
        "primary_cause": "text_visual_both_wrong",
        "secondary_cause": "",
        "question_answer": "A new single-serve coffee maker should have started its own cluster, but both similarity branches pulled it toward an existing coffee-maker cluster.",
    },
    "run55_obj201": {
        "first_error_stage": "similarity",
        "primary_signal_family": "similarity",
        "root_cause_label": "text_similarity_error",
        "primary_cause": "text_similarity_error",
        "secondary_cause": "visual_similarity_error",
        "question_answer": "The mixed run flipped the wrong attachment target from the silver coffee maker to the black one because the text branch dominated the semantic score.",
    },
    "run010_obj205": {
        "first_error_stage": "geometry",
        "primary_signal_family": "geometry",
        "root_cause_label": "similarity_correct_but_geometry_wrong",
        "primary_cause": "depth_error",
        "secondary_cause": "",
        "question_answer": "The correct island cluster stayed on top, but geometry kept every correct edge below the 0.5 graph threshold and forced a false new cluster.",
    },
    "run55_obj205": {
        "first_error_stage": "geometry",
        "primary_signal_family": "geometry",
        "root_cause_label": "similarity_correct_but_geometry_wrong",
        "primary_cause": "depth_error",
        "secondary_cause": "",
        "question_answer": "The mixed run raised the island semantic score, but not enough to overcome the geometry thresholding step.",
    },
}

SUPPORT_CASE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "run55_obj176": {
        "case_bucket": "supporting",
        "first_error_stage": "similarity",
        "primary_signal_family": "similarity",
        "root_cause_label": "text_visual_both_wrong",
        "primary_cause": "text_visual_both_wrong",
        "secondary_cause": "compounded_error",
        "question_answer": "This is a near-miss: the semantic score still favored the wrong seat, but the later graph constraint rescued the final assignment.",
    },
    "run010_obj119": {
        "case_bucket": "supporting",
        "first_error_stage": "interaction",
        "primary_signal_family": "interaction",
        "root_cause_label": "unclear",
        "primary_cause": "unclear",
        "secondary_cause": "",
        "question_answer": "Repeated cabinets create broad same-view co-cluster proposals; the final split prevents the merge, but the underlying cause is ambiguous without manual cabinet identity labels.",
    },
    "run010_obj124": {
        "case_bucket": "supporting",
        "first_error_stage": "interaction",
        "primary_signal_family": "interaction",
        "root_cause_label": "unclear",
        "primary_cause": "unclear",
        "secondary_cause": "",
        "question_answer": "Repeated cabinets create broad same-view co-cluster proposals; the final split prevents the merge, but the underlying cause is ambiguous without manual cabinet identity labels.",
    },
    "run010_obj132": {
        "case_bucket": "supporting",
        "first_error_stage": "interaction",
        "primary_signal_family": "interaction",
        "root_cause_label": "unclear",
        "primary_cause": "unclear",
        "secondary_cause": "",
        "question_answer": "Repeated cabinets create broad same-view co-cluster proposals; the final split prevents the merge, but the underlying cause is ambiguous without manual cabinet identity labels.",
    },
    "run55_obj199": {
        "case_bucket": "supporting",
        "first_error_stage": "interaction",
        "primary_signal_family": "interaction",
        "root_cause_label": "unclear",
        "primary_cause": "unclear",
        "secondary_cause": "",
        "question_answer": "Late-step cabinet groups show the same repeated-appearance problem, but cabinet identity remains too ambiguous to use in the main counts.",
    },
}

RETRIEVAL_SUPPORT_EXAMPLES = [
    {
        "case_id": "retrieval_q01",
        "query_json": Path("object_vpr_results/angle_split_batch_yoloworld/query_01_sampled_attempt_01/query_20260312_205038.json"),
        "label": "partial chair edge crop",
        "pattern": "generic chair text + partial crop",
    },
    {
        "case_id": "retrieval_q03",
        "query_json": Path("object_vpr_results/angle_split_batch_yoloworld/query_03_sampled_attempt_01/query_20260312_205115.json"),
        "label": "ornate chair among many similar chairs",
        "pattern": "repeated-seat ambiguity",
    },
    {
        "case_id": "retrieval_q05",
        "query_json": Path("object_vpr_results/angle_split_batch_yoloworld/query_05_sampled_attempt_01/query_20260312_205158.json"),
        "label": "small framed picture",
        "pattern": "generic picture text",
    },
]

FLOORPLAN_EXAMPLE_CASE_IDS = (
    "run010_obj198",
    "run55_obj169",
    "run55_obj201",
)

ANGLE_EXAMPLES = [
    {
        "example_id": "angle_obj172_clock_left",
        "run_name": "run010",
        "object_global_id": 172,
        "title": "Clock from view_00023",
        "notes": "Left-side object with a large negative bearing; useful for visually checking left-bucket consistency.",
    },
    {
        "example_id": "angle_obj169_island_right",
        "run_name": "run010",
        "object_global_id": 169,
        "title": "Island/table from view_00023",
        "notes": "Right-side island crop with a clearly positive bearing.",
    },
    {
        "example_id": "angle_obj198_fridge_center",
        "run_name": "run010",
        "object_global_id": 198,
        "title": "Fridge from view_00027",
        "notes": "Near-center object with a small-magnitude bearing, useful as a sanity-check example.",
    },
    {
        "example_id": "angle90_obj478_picture_center",
        "run_name": "spatial_db_origin",
        "object_global_id": 478,
        "title": "90 deg | picture frame near center",
        "notes": "Orientation 90 deg example with a near-zero bearing.",
    },
    {
        "example_id": "angle90_obj558_chair_left",
        "run_name": "spatial_db_origin",
        "object_global_id": 558,
        "title": "90 deg | chair on the left",
        "notes": "Orientation 90 deg example with a clear negative bearing.",
    },
    {
        "example_id": "angle90_obj579_clock_right",
        "run_name": "spatial_db_origin",
        "object_global_id": 579,
        "title": "90 deg | clock on the right",
        "notes": "Orientation 90 deg example with a clear positive bearing.",
    },
    {
        "example_id": "angle180_obj458_oven_center",
        "run_name": "spatial_db_origin",
        "object_global_id": 458,
        "title": "180 deg | oven near center",
        "notes": "Orientation 180 deg example with a small-magnitude bearing.",
    },
    {
        "example_id": "angle180_obj527_fridge_left",
        "run_name": "spatial_db_origin",
        "object_global_id": 527,
        "title": "180 deg | refrigerator on the left",
        "notes": "Orientation 180 deg example with a large negative bearing.",
    },
    {
        "example_id": "angle180_obj005_picture_right",
        "run_name": "spatial_db_origin",
        "object_global_id": 5,
        "title": "180 deg | picture frame on the right",
        "notes": "Orientation 180 deg example with a positive bearing.",
    },
]

DEPTH_EXAMPLES = [
    {
        "example_id": "depth_obj169_island",
        "run_name": "run010",
        "object_global_id": 169,
        "title": "Island/table depth from view_00023",
        "notes": "Representative island case from the geometry-suppressed failure group.",
    },
    {
        "example_id": "depth_obj198_fridge",
        "run_name": "run010",
        "object_global_id": 198,
        "title": "Fridge depth from view_00027",
        "notes": "Representative refrigerator case where geometry stays the dominant failure source.",
    },
    {
        "example_id": "depth_obj205_island",
        "run_name": "run55",
        "object_global_id": 205,
        "title": "Island/table depth from view_00027",
        "notes": "Another island case where the semantic winner is correct but geometry still blocks the merge.",
    },
    {
        "example_id": "depth90_obj610_picture_near",
        "run_name": "spatial_db_origin",
        "object_global_id": 610,
        "title": "90 deg | near picture frame depth",
        "notes": "Orientation 90 deg near-depth example.",
    },
    {
        "example_id": "depth90_obj579_clock_mid",
        "run_name": "spatial_db_origin",
        "object_global_id": 579,
        "title": "90 deg | mid-range clock depth",
        "notes": "Orientation 90 deg medium-depth example.",
    },
    {
        "example_id": "depth90_obj595_chair_far",
        "run_name": "spatial_db_origin",
        "object_global_id": 595,
        "title": "90 deg | far chair depth",
        "notes": "Orientation 90 deg far-depth example.",
    },
    {
        "example_id": "depth180_obj005_picture_near",
        "run_name": "spatial_db_origin",
        "object_global_id": 5,
        "title": "180 deg | near picture frame depth",
        "notes": "Orientation 180 deg near-depth example.",
    },
    {
        "example_id": "depth180_obj458_oven_mid",
        "run_name": "spatial_db_origin",
        "object_global_id": 458,
        "title": "180 deg | mid-range oven depth",
        "notes": "Orientation 180 deg medium-depth example.",
    },
    {
        "example_id": "depth180_obj491_dishwasher_far",
        "run_name": "spatial_db_origin",
        "object_global_id": 491,
        "title": "180 deg | far dishwasher depth",
        "notes": "Orientation 180 deg far-depth example.",
    },
]


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-12)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=8)
def _load_meta_by_view_id(db_dir_str: str) -> Dict[str, Dict[str, Any]]:
    db_dir = Path(db_dir_str)
    rows = _load_jsonl(db_dir / "meta.jsonl")
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        entry_id = _safe_int(row.get("id"))
        if entry_id is None:
            continue
        out[f"view_{int(entry_id):05d}"] = dict(row)
    return out


@lru_cache(maxsize=8)
def _load_object_rows_by_id(db_dir_str: str) -> Dict[int, Dict[str, Any]]:
    db_dir = Path(db_dir_str)
    rows = _load_jsonl(db_dir / "object_meta.jsonl")
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        object_id = _safe_int(row.get("object_global_id"))
        if object_id is None:
            continue
        out[int(object_id)] = dict(row)
    return out


def _resolve_path(root: Path, relative_or_abs: Optional[str]) -> Optional[Path]:
    if not relative_or_abs:
        return None
    raw = Path(str(relative_or_abs))
    if raw.is_absolute():
        return raw if raw.exists() else None
    candidate = root / raw
    return candidate if candidate.exists() else None


def _read_image(path: Optional[Path], max_side: Optional[int] = None) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if max_side is not None:
        h, w = image.shape[:2]
        scale = min(float(max_side) / max(float(h), float(w)), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return image


def _tile_label(text: str, width: int) -> np.ndarray:
    lines = textwrap.wrap(_safe_text(text), width=max(12, width // 10)) or [""]
    height = 26 + 22 * max(len(lines) - 1, 0)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 20
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
        y += 22
    return canvas


def _image_tile(image: Optional[np.ndarray], label: str, width: int = 360, height: int = 240) -> np.ndarray:
    if image is None:
        canvas = np.full((height, width, 3), 235, dtype=np.uint8)
        cv2.putText(canvas, "missing", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70, 70, 70), 2, cv2.LINE_AA)
        label_img = _tile_label(label, width)
        return np.vstack([canvas, label_img])
    h, w = image.shape[:2]
    scale = min(float(width) / max(float(w), 1.0), float(height) / max(float(h), 1.0))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    label_img = _tile_label(label, width)
    return np.vstack([canvas, label_img])


def _text_tile(label: str, width: int = 360, height: int = 240) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    label_img = _tile_label(label, width)
    return np.vstack([canvas, label_img])


def _load_floor_plan_projection(db_dir: Path) -> Optional[Dict[str, float]]:
    path = db_dir / "overview" / "floor_plan_projection.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    required = ("view_min_x", "view_max_x", "view_min_y", "view_max_y")
    try:
        return {key: float(payload[key]) for key in required}
    except Exception:
        return None


def _world_to_floor_pixel(
    x: float,
    y: float,
    width: int,
    height: int,
    projection: Mapping[str, float],
) -> Tuple[int, int]:
    min_x = float(projection["view_min_x"])
    max_x = float(projection["view_max_x"])
    min_y = float(projection["view_min_y"])
    max_y = float(projection["view_max_y"])
    denom_x = max(max_x - min_x, 1e-6)
    denom_y = max(max_y - min_y, 1e-6)
    px = int(np.clip((float(x) - min_x) / denom_x * (width - 1), 0, width - 1))
    py = int(np.clip((float(y) - min_y) / denom_y * (height - 1), 0, height - 1))
    return px, py


def _orientation_tip_world(
    camera_x: float,
    camera_floor_y: float,
    orientation_deg: float,
    world_len: float,
) -> Tuple[float, float]:
    yaw = math.radians(float(orientation_deg))
    tip_x = float(camera_x - math.sin(yaw) * float(world_len))
    tip_y = float(camera_floor_y - math.cos(yaw) * float(world_len))
    return tip_x, tip_y


def _relative_bearing_from_camera_to_object(
    camera_x: float,
    camera_floor_y: float,
    object_x: float,
    object_floor_y: float,
    camera_orientation_deg: float,
) -> float:
    dx = float(object_x) - float(camera_x)
    dy = float(object_floor_y) - float(camera_floor_y)
    global_bearing = (math.degrees(math.atan2(-dx, -dy)) + 360.0) % 360.0
    return ((float(camera_orientation_deg) - global_bearing + 180.0) % 360.0) - 180.0


def _planar_distance_for_row(row: Mapping[str, Any]) -> Optional[float]:
    planar = _safe_float(row.get("projected_planar_distance_m"))
    if planar is not None:
        return planar
    depth = _safe_float(row.get("distance_from_camera_m"))
    bearing = _safe_float(row.get("relative_bearing_deg"))
    if depth is None:
        return None
    if bearing is None:
        return depth
    cos_h = math.cos(math.radians(float(bearing)))
    if abs(cos_h) < 1e-6:
        return None
    return float(depth / cos_h)


def _floor_pose_from_row(db_dir: Path, row: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    if row is None:
        return {
            "camera_x": None,
            "camera_floor_y": None,
            "camera_orientation_deg": None,
            "object_x": None,
            "object_floor_y": None,
        }
    entry_id = _safe_int(row.get("entry_id"))
    view_id = _safe_text(row.get("view_id"))
    if not view_id and entry_id is not None:
        view_id = f"view_{int(entry_id):05d}"
    meta_row = dict(_load_meta_by_view_id(str(db_dir)).get(view_id) or {})
    camera_x = _safe_float(meta_row.get("x"))
    camera_floor_y = _safe_float(meta_row.get("z"))
    camera_orientation_deg = _safe_float(meta_row.get("orientation"))
    object_x = _safe_float(row.get("estimated_global_x"))
    object_floor_y = None
    planar_distance = _planar_distance_for_row(row)
    relative_bearing_deg = _safe_float(row.get("relative_bearing_deg"))
    if None not in (camera_x, camera_floor_y, camera_orientation_deg, planar_distance, relative_bearing_deg):
        global_bearing = (float(camera_orientation_deg) - float(relative_bearing_deg)) % 360.0
        yaw = math.radians(global_bearing)
        object_x = float(camera_x - math.sin(yaw) * float(planar_distance))
        object_floor_y = float(camera_floor_y - math.cos(yaw) * float(planar_distance))
    return {
        "camera_x": camera_x,
        "camera_floor_y": camera_floor_y,
        "camera_orientation_deg": camera_orientation_deg,
        "object_x": object_x,
        "object_floor_y": object_floor_y,
    }


def _draw_floor_marker(
    image: np.ndarray,
    point: Tuple[int, int],
    label: str,
    color: Tuple[int, int, int],
    *,
    offset_xy: Tuple[int, int] = (8, -8),
) -> None:
    px, py = int(point[0]), int(point[1])
    cv2.circle(image, (px, py), 8, color, -1, cv2.LINE_AA)
    cv2.circle(image, (px, py), 10, (255, 255, 255), 2, cv2.LINE_AA)
    tx = int(np.clip(px + offset_xy[0], 4, max(image.shape[1] - 60, 4)))
    ty = int(np.clip(py + offset_xy[1], 16, max(image.shape[0] - 8, 16)))
    cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _floor_plan_tile(case: CaseRecord, width: int = 360, height: int = 240) -> np.ndarray:
    spec = RUN_SPECS[case.run_name]
    floor_plan_path = spec.db_dir / "overview" / "textured_floor_plan.jpg"
    projection = _load_floor_plan_projection(spec.db_dir)
    base = _read_image(floor_plan_path)
    if base is None or projection is None:
        return _text_tile("Floor plan unavailable for this case.", width=width, height=height)

    canvas = base.copy()
    h, w = canvas.shape[:2]
    camera_point = None
    heading_point = None
    if case.camera_global_x is not None and case.camera_global_y is not None:
        camera_point = _world_to_floor_pixel(case.camera_global_x, case.camera_global_y, w, h, projection)
    if (
        case.camera_global_x is not None
        and case.camera_global_y is not None
        and case.camera_orientation_deg is not None
    ):
        tip_x, tip_y = _orientation_tip_world(
            case.camera_global_x,
            case.camera_global_y,
            case.camera_orientation_deg,
            world_len=1.0,
        )
        heading_point = _world_to_floor_pixel(tip_x, tip_y, w, h, projection)
    query_point = None
    if case.query_global_x is not None and case.query_global_y is not None:
        query_point = _world_to_floor_pixel(case.query_global_x, case.query_global_y, w, h, projection)
    correct_point = None
    if case.correct_match_global_x is not None and case.correct_match_global_y is not None:
        correct_point = _world_to_floor_pixel(case.correct_match_global_x, case.correct_match_global_y, w, h, projection)
    wrong_point = None
    if case.wrong_match_global_x is not None and case.wrong_match_global_y is not None:
        wrong_point = _world_to_floor_pixel(case.wrong_match_global_x, case.wrong_match_global_y, w, h, projection)

    if camera_point and heading_point:
        cv2.line(canvas, camera_point, heading_point, (10, 10, 10), 1, cv2.LINE_AA)
    if camera_point and query_point:
        cv2.line(canvas, camera_point, query_point, (10, 10, 10), 1, cv2.LINE_AA)
    if camera_point and correct_point:
        cv2.line(canvas, camera_point, correct_point, (60, 200, 90), 1, cv2.LINE_AA)
    if camera_point and wrong_point:
        cv2.line(canvas, camera_point, wrong_point, (40, 40, 220), 1, cv2.LINE_AA)

    if camera_point:
        _draw_floor_marker(canvas, camera_point, "V", (20, 20, 240), offset_xy=(8, -10))
    if correct_point:
        _draw_floor_marker(canvas, correct_point, "C", (60, 200, 90), offset_xy=(10, -10))
    if wrong_point:
        _draw_floor_marker(canvas, wrong_point, "W", (40, 40, 220), offset_xy=(10, 18))
    if query_point:
        _draw_floor_marker(canvas, query_point, "Q", (30, 210, 240), offset_xy=(-20, -14))

    cv2.rectangle(canvas, (8, 8), (128, 76), (245, 245, 245), -1)
    cv2.rectangle(canvas, (8, 8), (128, 76), (80, 80, 80), 1)
    legend = [("V", (20, 20, 240)), ("Q", (30, 210, 240)), ("C", (60, 200, 90)), ("W", (40, 40, 220))]
    for index, (token, color) in enumerate(legend):
        y = 20 + index * 13
        cv2.circle(canvas, (20, y), 4, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, token, (30, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    label = (
        f"Floor plan | V=({ _fmt(case.camera_global_x) },{ _fmt(case.camera_global_y) }) | "
        f"Q=({ _fmt(case.query_global_x) },{ _fmt(case.query_global_y) }) | "
        f"C=({ _fmt(case.correct_match_global_x) },{ _fmt(case.correct_match_global_y) }) | "
        f"W=({ _fmt(case.wrong_match_global_x) },{ _fmt(case.wrong_match_global_y) })"
    )
    return _image_tile(canvas, label, width=width, height=height)


def _render_floor_example_panel(
    *,
    db_dir: Path,
    source_name: str,
    object_row: Mapping[str, Any],
    example_id: str,
    title: str,
    notes: str,
    mode: str,
    output_dir: Path,
) -> Dict[str, Any]:
    meta_by_view = _load_meta_by_view_id(str(db_dir))
    view_id = _safe_text(object_row.get("view_id"))
    if not view_id:
        entry_id = _safe_int(object_row.get("entry_id"))
        if entry_id is not None:
            view_id = f"view_{int(entry_id):05d}"
    meta_row = dict(meta_by_view.get(view_id) or {})
    if not meta_row:
        raise KeyError(f"Missing meta row for {view_id} in {db_dir}")

    pose = _floor_pose_from_row(db_dir, object_row)
    camera_x = pose["camera_x"]
    camera_y = pose["camera_floor_y"]
    camera_orientation_deg = pose["camera_orientation_deg"]
    object_x = pose["object_x"]
    object_y = pose["object_floor_y"]
    if None in (camera_x, camera_y, camera_orientation_deg, object_x, object_y):
        raise ValueError(f"Incomplete floor-plan coordinates for example {example_id}")

    floor_plan_path = db_dir / "overview" / "textured_floor_plan.jpg"
    projection = _load_floor_plan_projection(db_dir)
    base = _read_image(floor_plan_path)
    if base is None or projection is None:
        raise FileNotFoundError(f"Missing floor plan artifacts under {db_dir / 'overview'}")

    canvas = base.copy()
    h, w = canvas.shape[:2]
    camera_point = _world_to_floor_pixel(camera_x, camera_y, w, h, projection)
    object_point = _world_to_floor_pixel(object_x, object_y, w, h, projection)
    heading_tip_x, heading_tip_y = _orientation_tip_world(camera_x, camera_y, camera_orientation_deg, world_len=1.2)
    heading_point = _world_to_floor_pixel(heading_tip_x, heading_tip_y, w, h, projection)

    stored_bearing = _safe_float(object_row.get("relative_bearing_deg"))
    geometry_bearing = _relative_bearing_from_camera_to_object(
        camera_x=camera_x,
        camera_floor_y=camera_y,
        object_x=object_x,
        object_floor_y=object_y,
        camera_orientation_deg=camera_orientation_deg,
    )
    angle_delta = None
    if stored_bearing is not None:
        angle_delta = abs(((float(geometry_bearing) - float(stored_bearing) + 180.0) % 360.0) - 180.0)

    depth_pro_m = _safe_float(object_row.get("distance_from_camera_m"))
    planar_distance_m = _safe_float(object_row.get("projected_planar_distance_m"))

    cv2.line(canvas, camera_point, heading_point, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.line(canvas, camera_point, object_point, (10, 10, 10), 1, cv2.LINE_AA)
    _draw_floor_marker(canvas, camera_point, "V", (20, 20, 240), offset_xy=(10, -10))
    _draw_floor_marker(canvas, object_point, "O", (40, 170, 40), offset_xy=(10, 16))
    cv2.putText(
        canvas,
        f"ori={int(round(float(camera_orientation_deg))) if camera_orientation_deg is not None else 'n/a'}",
        (max(camera_point[0] - 20, 6), max(camera_point[1] - 18, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    angle_text = f"{_fmt(stored_bearing)} deg"
    cv2.putText(
        canvas,
        angle_text,
        (max(camera_point[0] + 8, 6), max(camera_point[1] + 24, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    frame_path = _resolve_path(db_dir, object_row.get("file_name"))
    frame = _read_image(frame_path, max_side=400)

    if mode == "angle":
        floor_label = (
            f"{title} | stored_bearing={_fmt(stored_bearing)} deg | "
            f"geom_bearing={_fmt(geometry_bearing)} deg | delta={_fmt(angle_delta)} deg | "
            f"bucket={_safe_text(object_row.get('angle_bucket'))}"
        )
        text_label = (
            f"{notes}\n"
            f"object_id={int(object_row['object_global_id'])} | label={_safe_text(object_row.get('label'))}\n"
            f"view={view_id} | camera=({_fmt(camera_x)},{_fmt(camera_y)}) | object=({_fmt(object_x)},{_fmt(object_y)})\n"
            f"laterality={_safe_text(object_row.get('laterality'))} | "
            f"relative_bearing_deg={_fmt(stored_bearing)}"
        )
    else:
        floor_label = (
            f"{title} | depth_pro={_fmt(depth_pro_m)} m | planar={_fmt(planar_distance_m)} m | "
            f"occlusion={_safe_text(object_row.get('occlusion_level')) or 'unknown'}"
        )
        mid_x = int(round((camera_point[0] + object_point[0]) / 2.0))
        mid_y = int(round((camera_point[1] + object_point[1]) / 2.0))
        cv2.putText(
            canvas,
            f"{_fmt(depth_pro_m)} m",
            (max(mid_x + 6, 6), max(mid_y - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{_fmt(depth_pro_m)} m",
            (max(mid_x + 6, 6), max(mid_y - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
        text_label = (
            f"{notes}\n"
            f"object_id={int(object_row['object_global_id'])} | label={_safe_text(object_row.get('label'))}\n"
            f"view={view_id} | camera=({_fmt(camera_x)},{_fmt(camera_y)}) | object=({_fmt(object_x)},{_fmt(object_y)})\n"
            f"depth_pro={_fmt(depth_pro_m)} m | planar_distance={_fmt(planar_distance_m)} m | "
            f"foreground_occluders={_safe_int(object_row.get('foreground_occluder_count'))}"
        )

    floor_tile = _image_tile(canvas, floor_label, width=560, height=340)
    frame_tile = _image_tile(frame, f"View frame | {view_id}", width=320, height=220)
    text_tile = _text_tile(text_label, width=320, height=220)
    side_column = _stack_v([frame_tile, text_tile])
    panel = _stack_h([floor_tile, side_column])
    output_path = output_dir / f"{example_id}.jpg"
    _write_image(output_path, panel)

    return {
        "example_id": example_id,
        "mode": mode,
        "title": title,
        "notes": notes,
        "run_name": source_name,
        "object_global_id": int(object_row["object_global_id"]),
        "view_id": view_id,
        "panel_path": str(output_path),
        "camera_global_x": camera_x,
        "camera_global_y": camera_y,
        "camera_orientation_deg": camera_orientation_deg,
        "object_global_x": object_x,
        "object_global_y": object_y,
        "stored_bearing_deg": stored_bearing,
        "geometry_bearing_deg": geometry_bearing,
        "bearing_delta_deg": angle_delta,
        "depth_pro_m": depth_pro_m,
        "planar_distance_m": planar_distance_m,
        "angle_bucket": _safe_text(object_row.get("angle_bucket")),
        "occlusion_level": _safe_text(object_row.get("occlusion_level")),
    }


def _render_diagnostic_examples(output_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    examples = {"angle": [], "depth": []}
    for spec_row in ANGLE_EXAMPLES:
        source_name = _safe_text(spec_row["run_name"])
        db_dir = DIAGNOSTIC_DB_DIRS[source_name]
        object_row = _load_object_rows_by_id(str(db_dir))[int(spec_row["object_global_id"])]
        examples["angle"].append(
            _render_floor_example_panel(
                db_dir=db_dir,
                source_name=source_name,
                object_row=object_row,
                example_id=_safe_text(spec_row["example_id"]),
                title=_safe_text(spec_row["title"]),
                notes=_safe_text(spec_row["notes"]),
                mode="angle",
                output_dir=output_dir,
            )
        )
    for spec_row in DEPTH_EXAMPLES:
        source_name = _safe_text(spec_row["run_name"])
        db_dir = DIAGNOSTIC_DB_DIRS[source_name]
        object_row = _load_object_rows_by_id(str(db_dir))[int(spec_row["object_global_id"])]
        examples["depth"].append(
            _render_floor_example_panel(
                db_dir=db_dir,
                source_name=source_name,
                object_row=object_row,
                example_id=_safe_text(spec_row["example_id"]),
                title=_safe_text(spec_row["title"]),
                notes=_safe_text(spec_row["notes"]),
                mode="depth",
                output_dir=output_dir,
            )
        )
    for bucket in ("angle", "depth"):
        examples[bucket].sort(
            key=lambda item: (
                int(round(float(item.get("camera_orientation_deg") or -1))),
                str(item.get("view_id") or ""),
                int(item.get("object_global_id") or -1),
            )
        )
    return examples


def _stack_h(images: Sequence[np.ndarray]) -> np.ndarray:
    arrays = list(images)
    if not arrays:
        raise ValueError("Expected at least one image to stack horizontally.")
    max_height = max(int(image.shape[0]) for image in arrays)
    padded: List[np.ndarray] = []
    for image in arrays:
        if int(image.shape[0]) == max_height:
            padded.append(image)
            continue
        pad = max_height - int(image.shape[0])
        padded.append(cv2.copyMakeBorder(image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)))
    return np.hstack(padded)


def _stack_v(images: Sequence[np.ndarray]) -> np.ndarray:
    arrays = list(images)
    if not arrays:
        raise ValueError("Expected at least one image to stack vertically.")
    max_width = max(int(image.shape[1]) for image in arrays)
    padded: List[np.ndarray] = []
    for image in arrays:
        if int(image.shape[1]) == max_width:
            padded.append(image)
            continue
        pad = max_width - int(image.shape[1])
        padded.append(cv2.copyMakeBorder(image, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(245, 245, 245)))
    return np.vstack(padded)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _sort_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _safe_int(row.get("entry_id")) or 10**9,
            _safe_int(row.get("object_global_id")) or 10**9,
        ),
    )


def _make_canonical_manifest(selected_objects: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    manifest: Dict[int, Dict[str, Any]] = {}
    for row in selected_objects:
        object_id = int(row["object_global_id"])
        base = MANUAL_CANONICAL_ENTRIES.get(object_id)
        if base is None:
            label = _safe_text(row.get("label"))
            if label == "cabinet":
                manifest[object_id] = {
                    "object_global_id": object_id,
                    "label": label,
                    "canonical_instance_id": "",
                    "status": "unclear",
                    "confidence": "low",
                    "notes": "Repeated cabinet bank; not used for main quantitative conclusions.",
                }
            else:
                manifest[object_id] = {
                    "object_global_id": object_id,
                    "label": label,
                    "canonical_instance_id": "",
                    "status": "unclear",
                    "confidence": "low",
                    "notes": "Not part of the high-confidence canonical subset.",
                }
            continue
        manifest[object_id] = {
            "object_global_id": object_id,
            "label": _safe_text(row.get("label")),
            "canonical_instance_id": base["canonical_instance_id"],
            "status": base["status"],
            "confidence": base["confidence"],
            "notes": base["notes"],
        }
    return manifest


def _load_dinov2_embeddings(db_dir: Path) -> Dict[int, np.ndarray]:
    emb_path = db_dir / "object_dinov2_emb.npy"
    if not emb_path.exists():
        legacy_path = db_dir / "object_dinov3_emb.npy"
        if not legacy_path.exists():
            return {}
        emb_path = legacy_path
    emb = np.load(emb_path).astype("float32")
    object_rows = _load_jsonl(db_dir / "object_meta.jsonl")
    out: Dict[int, np.ndarray] = {}
    for row in object_rows:
        object_id = _safe_int(row.get("object_global_id"))
        sidecar_row = row.get("dinov2_embedding_row_index")
        if sidecar_row in (None, ""):
            sidecar_row = row.get("dinov3_embedding_row_index")
        if object_id is None or sidecar_row in (None, ""):
            continue
        sidecar_idx = int(sidecar_row)
        if not (0 <= sidecar_idx < emb.shape[0]):
            continue
        out[int(object_id)] = _normalize_vec(np.asarray(emb[sidecar_idx], dtype=np.float32))
    return out


def _load_long_embeddings(db_dir: Path) -> Dict[int, np.ndarray]:
    loaded = load_object_db(str(db_dir), text_mode="long")
    if loaded is None:
        raise FileNotFoundError(f"Missing object_text_emb_long.npy or object_meta.jsonl under {db_dir}")
    object_rows, long_emb, _ = loaded
    out: Dict[int, np.ndarray] = {}
    for idx, row in enumerate(object_rows):
        object_id = int(row["object_global_id"])
        out[object_id] = _normalize_vec(np.asarray(long_emb[idx], dtype=np.float32))
    return out


def load_selected_sequence(spec: RunSpec) -> Dict[str, Any]:
    selected_payload = _load_json(spec.selected_views_root / "objects.json")
    selected_objects = _sort_rows(selected_payload["objects"])
    long_by_object = _load_long_embeddings(spec.db_dir)
    dinov2_by_object = _load_dinov2_embeddings(spec.db_dir)
    views_by_id: Dict[str, List[Dict[str, Any]]] = {view_id: [] for view_id in SELECTED_VIEW_IDS}
    for row in selected_objects:
        object_id = int(row["object_global_id"])
        enriched = dict(row)
        enriched["embedding"] = long_by_object[object_id]
        enriched["text_embedding"] = enriched["embedding"]
        enriched["dinov3_embedding"] = dinov2_by_object.get(object_id)
        enriched["dinov2_embedding"] = dinov2_by_object.get(object_id)
        enriched["view_id"] = _safe_text(enriched.get("view_id"))
        views_by_id[enriched["view_id"]].append(enriched)
    views: List[Dict[str, Any]] = []
    meta_rows = {f"view_{int(row['id']):05d}": row for row in _load_jsonl(spec.db_dir / "meta.jsonl")}
    for view_id in SELECTED_VIEW_IDS:
        rows = _sort_rows(views_by_id[view_id])
        meta_row = meta_rows.get(view_id, {})
        views.append(
            {
                "view_id": view_id,
                "entry_id": _safe_int(meta_row.get("id")) or _safe_int(rows[0].get("entry_id")) or -1,
                "file_name": _safe_text(meta_row.get("file_name") or (rows[0].get("file_name") if rows else "")),
                "orientation": _safe_float(meta_row.get("orientation")),
                "objects": rows,
            }
        )
    return {
        "views": views,
        "selected_objects": selected_objects,
        "canonical_manifest": _make_canonical_manifest(selected_objects),
    }


def _build_candidate_records(
    current_row: Mapping[str, Any],
    current_index: int,
    memory_clusters: Sequence[Mapping[str, Any]],
    cross_affinity: np.ndarray,
    cross_details: Sequence[Sequence[Mapping[str, Any]]],
    canonical_manifest: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for mem_index, cluster in enumerate(memory_clusters):
        detail = dict(cross_details[current_index][mem_index] or {})
        member_object_ids = [int(value) for value in cluster.get("member_object_ids", [])]
        canonical_ids = sorted(
            {
                _safe_text(canonical_manifest.get(object_id, {}).get("canonical_instance_id"))
                for object_id in member_object_ids
                if _safe_text(canonical_manifest.get(object_id, {}).get("canonical_instance_id"))
            }
        )
        candidates.append(
            {
                "cluster_id": int(cluster.get("cluster_id", -1)),
                "member_object_ids": member_object_ids,
                "member_rows": [dict(row) for row in cluster.get("member_rows", [])],
                "member_labels": sorted({_safe_text(row.get("label")) for row in cluster.get("member_rows", [])}),
                "member_view_ids": sorted({_safe_text(row.get("view_id")) for row in cluster.get("member_rows", [])}),
                "canonical_instance_ids": canonical_ids,
                "combined_similarity": float(cross_affinity[current_index, mem_index]),
                "text_similarity": _safe_float(detail.get("text_similarity")),
                "dinov2_similarity": _safe_float(detail.get("dinov3_similarity")),
                "semantic_visual_similarity": _safe_float(detail.get("semantic_visual_similarity")),
                "xy_distance_m": _safe_float(detail.get("xy_distance_m")),
                "distance_gate": _safe_float(detail.get("distance_gate")),
                "above_threshold": float(cross_affinity[current_index, mem_index]) >= RUN_SPECS["run010"].min_cross_affinity,
                "same_view_conflict": _safe_text(current_row.get("view_id")) in {
                    _safe_text(view_id) for view_id in cluster.get("member_view_ids", [])
                },
            }
        )

    semantic_sorted = sorted(
        candidates,
        key=lambda item: (
            -(item["semantic_visual_similarity"] if item["semantic_visual_similarity"] is not None else -1.0),
            item["cluster_id"],
        ),
    )
    combined_sorted = sorted(
        candidates,
        key=lambda item: (-item["combined_similarity"], item["cluster_id"]),
    )
    semantic_rank = {item["cluster_id"]: rank for rank, item in enumerate(semantic_sorted, start=1)}
    combined_rank = {item["cluster_id"]: rank for rank, item in enumerate(combined_sorted, start=1)}
    for candidate in candidates:
        candidate["rank_by_semantic"] = semantic_rank[candidate["cluster_id"]]
        candidate["rank_by_combined"] = combined_rank[candidate["cluster_id"]]
    return combined_sorted


def _reconstruct_run(spec: RunSpec, canonical_manifest: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    sequence = load_selected_sequence(spec)
    views = deepcopy(sequence["views"])
    memory_clusters: List[Dict[str, Any]] = [_build_cluster(index, [row]) for index, row in enumerate(views[0]["objects"])]
    next_cluster_id = len(memory_clusters)
    step_traces: List[Dict[str, Any]] = []
    computed_assignments: Dict[int, Dict[str, Any]] = {}

    for step_index, current_view in enumerate(views[1:], start=1):
        cross_affinity, cross_details = build_cross_affinity_matrix(
            memory_clusters,
            current_view["objects"],
            weight_text=spec.weight_text,
            weight_dinov3=spec.weight_dinov2,
            weight_global_geo=spec.weight_global_geo,
            weight_polar=spec.weight_polar,
            similarity_mode=spec.similarity_mode,
            distance_gate_dsq0=spec.distance_gate_dsq0,
            enable_dinov3_scoring=True,
        )
        masked_affinity, masked_details, num_same_view_masked_edges = _apply_same_view_hard_mask_to_cross_affinity(
            memory_clusters,
            current_view["objects"],
            cross_affinity=cross_affinity,
            cross_details=cross_details,
        )
        full_affinity = _full_bipartite_affinity(masked_affinity, min_cross_affinity=spec.min_cross_affinity)
        spectral_result = _run_capped_sequential_spectral_clustering(
            full_affinity,
            object_ids=list(range(full_affinity.shape[0])),
        )
        update = apply_incremental_step(
            memory_clusters,
            current_view["objects"],
            cross_affinity=masked_affinity,
            cross_details=masked_details,
            full_affinity=full_affinity,
            spectral_result=spectral_result,
            step_index=step_index,
            next_cluster_id=next_cluster_id,
            weight_text=spec.weight_text,
            weight_dinov3=spec.weight_dinov2,
            weight_global_geo=spec.weight_global_geo,
            weight_polar=spec.weight_polar,
            similarity_mode=spec.similarity_mode,
            distance_gate_dsq0=spec.distance_gate_dsq0,
            current_only_reattach_min_affinity=spec.current_only_reattach_min_affinity,
            dbscan_eps=None,
            dbscan_min_samples=spec.dbscan_min_samples,
            enforce_same_view_uniqueness=spec.enforce_same_view_uniqueness,
            enable_dinov3_scoring=True,
        )

        assignment_by_object: Dict[int, Dict[str, Any]] = {}
        for row in update["assignment_diagnostics"]:
            assignment_by_object[int(row["object_global_id"])] = dict(row)
            computed_assignments[int(row["object_global_id"])] = dict(row)

        candidate_rankings: Dict[int, List[Dict[str, Any]]] = {}
        for current_index, row in enumerate(current_view["objects"]):
            object_id = int(row["object_global_id"])
            candidate_rankings[object_id] = _build_candidate_records(
                row,
                current_index,
                memory_clusters,
                masked_affinity,
                masked_details,
                canonical_manifest,
            )

        step_traces.append(
            {
                "step_index": step_index,
                "view_id": current_view["view_id"],
                "memory_clusters_before": deepcopy(memory_clusters),
                "current_rows": deepcopy(current_view["objects"]),
                "candidate_rankings_by_object": candidate_rankings,
                "assignment_by_object": assignment_by_object,
                "same_view_block_cases": deepcopy(update["same_view_block_cases"]),
                "num_same_view_masked_edges": int(num_same_view_masked_edges),
            }
        )
        memory_clusters = update["memory_clusters"]
        next_cluster_id = int(update["next_cluster_id"])

    final_cluster_by_object: Dict[int, int] = {}
    for cluster in memory_clusters:
        cluster_id = int(cluster.get("cluster_id", -1))
        for object_id in cluster.get("member_object_ids", []):
            final_cluster_by_object[int(object_id)] = cluster_id

    stored_rows = _load_csv(spec.run_dir / "object_cluster_similarity_table.csv")
    validation = _validate_trace_against_stored(spec, step_traces, final_cluster_by_object, stored_rows)
    experiment_report = _load_json(spec.run_dir / "experiment_report.json")
    return {
        "spec": spec,
        "views": views,
        "selected_objects": sequence["selected_objects"],
        "step_traces": step_traces,
        "final_clusters": deepcopy(memory_clusters),
        "final_cluster_by_object": final_cluster_by_object,
        "stored_rows": stored_rows,
        "validation": validation,
        "experiment_report": experiment_report,
    }


def _validate_trace_against_stored(
    spec: RunSpec,
    step_traces: Sequence[Mapping[str, Any]],
    final_cluster_by_object: Mapping[int, int],
    stored_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    stored_by_object = {int(row["object_global_id"]): dict(row) for row in stored_rows}
    numeric_fields = (
        "term1_cosine",
        "term2_dinov2",
        "semantic_visual_similarity",
        "xy_distance_m",
        "distance_gate",
        "combined_similarity",
    )
    max_abs_diff = 0.0
    checked = 0
    step_counts_recomputed: Dict[str, int] = {"0": len(SELECTED_VIEW_IDS) * 0}
    step_counts_recomputed = {}
    mismatch_objects: List[int] = []
    for step_trace in step_traces:
        step_index = int(step_trace["step_index"])
        step_counts_recomputed[str(step_index)] = len(step_trace["current_rows"])
        assignment_by_object = dict(step_trace["assignment_by_object"])
        for object_id, row in assignment_by_object.items():
            stored = stored_by_object.get(int(object_id))
            if stored is None:
                mismatch_objects.append(int(object_id))
                continue
            if int(stored["cluster_id_at_assignment"]) != int(row["cluster_id_at_assignment"]):
                mismatch_objects.append(int(object_id))
                continue
            if _safe_text(stored["assignment_reason"]) != _safe_text(row["assignment_reason"]):
                mismatch_objects.append(int(object_id))
                continue
            for field in numeric_fields:
                expected_value = None
                if field == "term1_cosine":
                    expected_value = _safe_float(row.get("term1_cosine"))
                elif field == "term2_dinov2":
                    expected_value = _safe_float(row.get("term2_dinov3"))
                elif field == "semantic_visual_similarity":
                    expected_value = _safe_float(row.get("semantic_visual_similarity"))
                elif field == "xy_distance_m":
                    expected_value = _safe_float(row.get("xy_distance_m"))
                elif field == "distance_gate":
                    expected_value = _safe_float(row.get("distance_gate"))
                elif field == "combined_similarity":
                    expected_value = _safe_float(row.get("combined_similarity"))
                observed_value = _safe_float(stored.get(field))
                if expected_value is None and observed_value is None:
                    continue
                if expected_value is None or observed_value is None:
                    mismatch_objects.append(int(object_id))
                    break
                max_abs_diff = max(max_abs_diff, abs(float(expected_value) - float(observed_value)))
            checked += 1

    final_cluster_count_matches = len({int(value) for value in final_cluster_by_object.values()}) == int(
        _load_json(spec.run_dir / "experiment_report.json")["final_cluster_count"]
    )
    stored_step_counts = {
        str(step): sum(1 for row in stored_rows if int(row["step_index"]) == step)
        for step in range(4)
    }
    step_counts_recomputed["0"] = sum(1 for row in stored_rows if int(row["step_index"]) == 0)
    step_counts_match = stored_step_counts == step_counts_recomputed
    return {
        "assignment_rows_checked": checked,
        "max_abs_diff": float(max_abs_diff),
        "mismatch_object_ids": sorted({int(object_id) for object_id in mismatch_objects}),
        "final_cluster_count_matches": bool(final_cluster_count_matches),
        "step_counts_match": bool(step_counts_match),
        "stored_step_counts": stored_step_counts,
        "recomputed_step_counts": step_counts_recomputed,
    }


def _candidate_match_status(
    current_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    canonical_manifest: Mapping[int, Mapping[str, Any]],
    assigned_cluster_id: Optional[int],
    assignment_reason: str,
) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    canonical = canonical_manifest.get(int(current_row["object_global_id"]), {})
    canonical_id = _safe_text(canonical.get("canonical_instance_id"))
    canonical_status = _safe_text(canonical.get("status")) or "unclear"
    if canonical_status != "clear" or not canonical_id:
        return "unclear", [], None, None

    correct_candidates = [candidate for candidate in candidates if canonical_id in candidate["canonical_instance_ids"]]
    top_wrong_candidate = next(
        (
            candidate
            for candidate in candidates
            if canonical_id not in candidate["canonical_instance_ids"]
        ),
        None,
    )
    assigned_candidate = None
    if assigned_cluster_id is not None:
        assigned_candidate = next((candidate for candidate in candidates if int(candidate["cluster_id"]) == int(assigned_cluster_id)), None)

    if not correct_candidates:
        if assignment_reason == "dbscan_new_cluster":
            return "correct", [], assigned_candidate, top_wrong_candidate
        return "bad", [], assigned_candidate, top_wrong_candidate

    correct_cluster_ids = {int(candidate["cluster_id"]) for candidate in correct_candidates}
    if assigned_cluster_id in correct_cluster_ids:
        return "correct", correct_candidates, assigned_candidate, top_wrong_candidate
    return "bad", correct_candidates, assigned_candidate, top_wrong_candidate


def _select_representative_member(
    query_row: Mapping[str, Any],
    cluster_candidate: Optional[Mapping[str, Any]],
    canonical_manifest: Mapping[int, Mapping[str, Any]],
    preferred_canonical_id: str = "",
) -> Optional[Dict[str, Any]]:
    if not cluster_candidate:
        return None
    members = [dict(row) for row in cluster_candidate.get("member_rows", [])]
    if not members:
        return None
    if preferred_canonical_id:
        filtered = [
            member
            for member in members
            if _safe_text(canonical_manifest.get(int(member["object_global_id"]), {}).get("canonical_instance_id")) == preferred_canonical_id
        ]
        if filtered:
            members = filtered

    qx = _safe_float(query_row.get("estimated_global_x"))
    qz = _safe_float(query_row.get("estimated_global_z"))

    def sort_key(member: Mapping[str, Any]) -> Tuple[float, int]:
        mx = _safe_float(member.get("estimated_global_x"))
        mz = _safe_float(member.get("estimated_global_z"))
        if qx is None or qz is None or mx is None or mz is None:
            dist = 10**6
        else:
            dist = math.sqrt((qx - mx) ** 2 + (qz - mz) ** 2)
        return (dist, int(member["object_global_id"]))

    return dict(sorted(members, key=sort_key)[0])


def _case_source_paths(db_dir: Path, row: Optional[Mapping[str, Any]]) -> Dict[str, Optional[str]]:
    if row is None:
        return {
            "crop_path": None,
            "mask_overlay_path": None,
            "frame_path": None,
        }
    crop_path = _resolve_path(db_dir, row.get("crop_path"))
    mask_overlay_path = _resolve_path(db_dir, row.get("mask_overlay_path"))
    frame_path = _resolve_path(db_dir, row.get("file_name"))
    return {
        "crop_path": str(crop_path) if crop_path is not None else None,
        "mask_overlay_path": str(mask_overlay_path) if mask_overlay_path is not None else None,
        "frame_path": str(frame_path) if frame_path is not None else None,
    }


def _depth_spread(row: Mapping[str, Any]) -> Optional[float]:
    p10 = _safe_float(row.get("depth_stat_p10_m"))
    p90 = _safe_float(row.get("depth_stat_p90_m"))
    if p10 is None or p90 is None:
        return None
    return float(max(p90 - p10, 0.0))


def _make_case_record(
    *,
    case_id: str,
    case_bucket: str,
    run_trace: Mapping[str, Any],
    step_trace: Mapping[str, Any],
    query_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    correct_candidates: Sequence[Mapping[str, Any]],
    assigned_candidate: Optional[Mapping[str, Any]],
    top_wrong_candidate: Optional[Mapping[str, Any]],
    canonical_manifest: Mapping[int, Mapping[str, Any]],
    override: Mapping[str, str],
) -> CaseRecord:
    spec: RunSpec = run_trace["spec"]
    query_object_id = int(query_row["object_global_id"])
    canonical_entry = canonical_manifest.get(query_object_id, {})
    canonical_instance_id = _safe_text(canonical_entry.get("canonical_instance_id"))
    correct_best = correct_candidates[0] if correct_candidates else None
    correct_member = _select_representative_member(query_row, correct_best, canonical_manifest, canonical_instance_id)
    wrong_member = _select_representative_member(query_row, top_wrong_candidate, canonical_manifest)
    assigned_cluster_id = _safe_int(step_trace["assignment_by_object"].get(query_object_id, {}).get("cluster_id_at_assignment"))
    assignment_reason = _safe_text(step_trace["assignment_by_object"].get(query_object_id, {}).get("assignment_reason"))
    query_paths = _case_source_paths(spec.db_dir, query_row)
    wrong_paths = _case_source_paths(spec.db_dir, wrong_member)
    correct_paths = _case_source_paths(spec.db_dir, correct_member)
    top_semantic = sorted(
        candidates,
        key=lambda item: (
            -(item["semantic_visual_similarity"] if item["semantic_visual_similarity"] is not None else -1.0),
            item["cluster_id"],
        ),
    )[0]
    top_combined = candidates[0]
    top_correct_combined_score = correct_best["combined_similarity"] if correct_best is not None else None
    top_correct_semantic_score = correct_best["semantic_visual_similarity"] if correct_best is not None else None

    explanation = _compose_case_explanation(
        override=override,
        query_row=query_row,
        assignment_reason=assignment_reason,
        correct_best=correct_best,
        top_wrong_candidate=top_wrong_candidate,
    )
    notes = _safe_text(canonical_entry.get("notes"))
    query_pose = _floor_pose_from_row(spec.db_dir, query_row)
    wrong_pose = _floor_pose_from_row(spec.db_dir, wrong_member)
    correct_pose = _floor_pose_from_row(spec.db_dir, correct_member)

    return CaseRecord(
        case_id=case_id,
        case_bucket=case_bucket,
        run_name=spec.run_name,
        run_label=spec.run_label,
        step_index=int(step_trace["step_index"]),
        view_id=_safe_text(query_row.get("view_id")),
        query_object_id=query_object_id,
        query_label=_safe_text(query_row.get("label")),
        canonical_instance_id=canonical_instance_id,
        canonical_status=_safe_text(canonical_entry.get("status")),
        final_outcome="bad" if case_bucket == "main" else "supporting",
        assignment_reason=assignment_reason,
        assigned_cluster_id=assigned_cluster_id,
        correct_cluster_ids=[int(candidate["cluster_id"]) for candidate in correct_candidates],
        correct_match_object_id=_safe_int(correct_member.get("object_global_id") if correct_member else None),
        wrong_match_object_id=_safe_int(wrong_member.get("object_global_id") if wrong_member else None),
        wrong_match_cluster_id=_safe_int(top_wrong_candidate.get("cluster_id") if top_wrong_candidate else None),
        first_error_stage=_safe_text(override.get("first_error_stage")),
        primary_signal_family=_safe_text(override.get("primary_signal_family")),
        root_cause_label=_safe_text(override.get("root_cause_label")),
        primary_cause=_safe_text(override.get("primary_cause")),
        secondary_cause=_safe_text(override.get("secondary_cause")),
        explanation=explanation,
        question_answer=_safe_text(override.get("question_answer")),
        query_crop_path=query_paths["crop_path"],
        query_mask_overlay_path=query_paths["mask_overlay_path"],
        query_frame_path=query_paths["frame_path"],
        wrong_crop_path=wrong_paths["crop_path"],
        correct_crop_path=correct_paths["crop_path"],
        panel_path=None,
        camera_global_x=query_pose["camera_x"],
        camera_global_y=query_pose["camera_floor_y"],
        camera_orientation_deg=query_pose["camera_orientation_deg"],
        query_global_x=query_pose["object_x"],
        query_global_y=query_pose["object_floor_y"],
        wrong_match_global_x=wrong_pose["object_x"],
        wrong_match_global_y=wrong_pose["object_floor_y"],
        correct_match_global_x=correct_pose["object_x"],
        correct_match_global_y=correct_pose["object_floor_y"],
        wrong_match_view_id=_safe_text(wrong_member.get("view_id") if wrong_member else ""),
        correct_match_view_id=_safe_text(correct_member.get("view_id") if correct_member else ""),
        text_similarity_correct=_safe_float(correct_best.get("text_similarity") if correct_best else None),
        text_similarity_wrong=_safe_float(top_wrong_candidate.get("text_similarity") if top_wrong_candidate else None),
        dinov2_similarity_correct=_safe_float(correct_best.get("dinov2_similarity") if correct_best else None),
        dinov2_similarity_wrong=_safe_float(top_wrong_candidate.get("dinov2_similarity") if top_wrong_candidate else None),
        semantic_similarity_correct=_safe_float(correct_best.get("semantic_visual_similarity") if correct_best else None),
        semantic_similarity_wrong=_safe_float(top_wrong_candidate.get("semantic_visual_similarity") if top_wrong_candidate else None),
        combined_similarity_correct=_safe_float(correct_best.get("combined_similarity") if correct_best else None),
        combined_similarity_wrong=_safe_float(top_wrong_candidate.get("combined_similarity") if top_wrong_candidate else None),
        distance_gate_correct=_safe_float(correct_best.get("distance_gate") if correct_best else None),
        distance_gate_wrong=_safe_float(top_wrong_candidate.get("distance_gate") if top_wrong_candidate else None),
        xy_distance_correct_m=_safe_float(correct_best.get("xy_distance_m") if correct_best else None),
        xy_distance_wrong_m=_safe_float(top_wrong_candidate.get("xy_distance_m") if top_wrong_candidate else None),
        query_distance_from_camera_m=_safe_float(query_row.get("distance_from_camera_m")),
        query_relative_bearing_deg=_safe_float(query_row.get("relative_bearing_deg")),
        query_object_orientation_deg=_safe_float(query_row.get("object_orientation_deg")),
        query_angle_bucket=_safe_text(query_row.get("angle_bucket")),
        query_occlusion_level=_safe_text(query_row.get("occlusion_level")),
        query_foreground_occluder_count=_safe_int(query_row.get("foreground_occluder_count")),
        query_visible_occlusion_ratio=_safe_float(query_row.get("visible_occlusion_ratio")),
        query_depth_p10_m=_safe_float(query_row.get("depth_stat_p10_m")),
        query_depth_p90_m=_safe_float(query_row.get("depth_stat_p90_m")),
        query_depth_spread_m=_depth_spread(query_row),
        top_semantic_cluster_id=int(top_semantic["cluster_id"]),
        top_combined_cluster_id=int(top_combined["cluster_id"]),
        top_combined_score=_safe_float(top_combined.get("combined_similarity")),
        top_correct_combined_score=_safe_float(top_correct_combined_score),
        top_correct_semantic_score=_safe_float(top_correct_semantic_score),
        combined_threshold=float(spec.min_cross_affinity),
        correct_candidate_rank_by_semantic=_safe_int(correct_best.get("rank_by_semantic") if correct_best else None),
        correct_candidate_rank_by_combined=_safe_int(correct_best.get("rank_by_combined") if correct_best else None),
        wrong_candidate_rank_by_semantic=_safe_int(top_wrong_candidate.get("rank_by_semantic") if top_wrong_candidate else None),
        wrong_candidate_rank_by_combined=_safe_int(top_wrong_candidate.get("rank_by_combined") if top_wrong_candidate else None),
        candidate_count=len(candidates),
        notes=notes,
    )


def _compose_case_explanation(
    *,
    override: Mapping[str, str],
    query_row: Mapping[str, Any],
    assignment_reason: str,
    correct_best: Optional[Mapping[str, Any]],
    top_wrong_candidate: Optional[Mapping[str, Any]],
) -> str:
    label = _safe_text(query_row.get("label"))
    qid = int(query_row["object_global_id"])
    first = _safe_text(override.get("first_error_stage"))
    primary = _safe_text(override.get("primary_cause"))
    if correct_best is None:
        correct_text = "No valid earlier cluster should have matched this object, so the correct action was to start a new cluster."
    else:
        correct_text = (
            f"The correct cluster {int(correct_best['cluster_id'])} ranked "
            f"#{int(correct_best['rank_by_combined'])} by combined score with "
            f"combined={float(correct_best['combined_similarity']):.3f}."
        )
    wrong_text = ""
    if top_wrong_candidate is not None:
        wrong_text = (
            f" The most misleading competing cluster was {int(top_wrong_candidate['cluster_id'])} "
            f"with combined={float(top_wrong_candidate['combined_similarity']):.3f}, "
            f"text={float(top_wrong_candidate['text_similarity'] or 0.0):.3f}, "
            f"DINO={float(top_wrong_candidate['dinov2_similarity'] or 0.0):.3f}."
        )
    return (
        f"Object {qid} ({label}) first goes wrong at the {first} stage. "
        f"Primary cause: {primary}. {correct_text}{wrong_text} "
        f"The stored assignment reason was {assignment_reason}."
    )


def _mine_case_records(
    run_traces: Mapping[str, Mapping[str, Any]],
    canonical_manifest: Mapping[int, Mapping[str, Any]],
) -> List[CaseRecord]:
    cases: List[CaseRecord] = []

    for run_key, run_trace in run_traces.items():
        spec: RunSpec = run_trace["spec"]
        for step_trace in run_trace["step_traces"]:
            for query_row in step_trace["current_rows"]:
                query_object_id = int(query_row["object_global_id"])
                case_id = f"{spec.run_name}_obj{query_object_id}"
                candidates = list(step_trace["candidate_rankings_by_object"][query_object_id])
                assignment = dict(step_trace["assignment_by_object"][query_object_id])
                assigned_cluster_id = _safe_int(assignment.get("cluster_id_at_assignment"))
                assignment_reason = _safe_text(assignment.get("assignment_reason"))
                match_status, correct_candidates, assigned_candidate, top_wrong_candidate = _candidate_match_status(
                    query_row,
                    candidates,
                    canonical_manifest,
                    assigned_cluster_id,
                    assignment_reason,
                )
                if case_id in MAIN_CASE_OVERRIDES and match_status == "bad":
                    cases.append(
                        _make_case_record(
                            case_id=case_id,
                            case_bucket="main",
                            run_trace=run_trace,
                            step_trace=step_trace,
                            query_row=query_row,
                            candidates=candidates,
                            correct_candidates=correct_candidates,
                            assigned_candidate=assigned_candidate,
                            top_wrong_candidate=top_wrong_candidate,
                            canonical_manifest=canonical_manifest,
                            override=MAIN_CASE_OVERRIDES[case_id],
                        )
                    )
                elif case_id in SUPPORT_CASE_OVERRIDES:
                    override = SUPPORT_CASE_OVERRIDES[case_id]
                    cases.append(
                        _make_case_record(
                            case_id=case_id,
                            case_bucket=_safe_text(override.get("case_bucket")) or "supporting",
                            run_trace=run_trace,
                            step_trace=step_trace,
                            query_row=query_row,
                            candidates=candidates,
                            correct_candidates=correct_candidates,
                            assigned_candidate=assigned_candidate,
                            top_wrong_candidate=top_wrong_candidate,
                            canonical_manifest=canonical_manifest,
                            override=override,
                        )
                    )

    ordering = {"main": 0, "supporting": 1}
    cases.sort(key=lambda case: (ordering.get(case.case_bucket, 9), case.run_name, case.step_index or 99, case.query_object_id or 10**9))
    return cases


def _render_case_panel(
    case: CaseRecord,
    output_dir: Path,
) -> Optional[str]:
    query_frame = _read_image(Path(case.query_frame_path) if case.query_frame_path else None, max_side=640)
    query_crop = _read_image(Path(case.query_crop_path) if case.query_crop_path else None, max_side=320)
    query_mask = _read_image(Path(case.query_mask_overlay_path) if case.query_mask_overlay_path else None, max_side=320)
    wrong_crop = _read_image(Path(case.wrong_crop_path) if case.wrong_crop_path else None, max_side=320)
    correct_crop = _read_image(Path(case.correct_crop_path) if case.correct_crop_path else None, max_side=320)
    floor_tile = _floor_plan_tile(case)
    top_row = _stack_h(
        [
            _image_tile(query_crop, f"Query crop | obj {case.query_object_id} | {case.query_label}"),
            _image_tile(wrong_crop, f"Most misleading candidate | cluster {case.wrong_match_cluster_id or 'n/a'}"),
            _image_tile(correct_crop, f"Correct candidate | clusters {','.join(str(v) for v in case.correct_cluster_ids) or 'new'}"),
            floor_tile,
        ]
    )
    bottom_row = _stack_h(
        [
            _image_tile(query_frame, f"Query frame | {case.view_id}"),
            _image_tile(query_mask, f"Mask / crop overlay | occlusion={case.query_occlusion_level or 'unknown'}"),
            _text_tile(
                (
                    f"{case.root_cause_label}\n"
                    f"first_error={case.first_error_stage} | family={case.primary_signal_family}\n"
                    f"correct_combined={_fmt(case.combined_similarity_correct)} | wrong_combined={_fmt(case.combined_similarity_wrong)}\n"
                    f"correct_semantic={_fmt(case.semantic_similarity_correct)} | wrong_semantic={_fmt(case.semantic_similarity_wrong)}\n"
                    f"correct_gate={_fmt(case.distance_gate_correct)} | wrong_gate={_fmt(case.distance_gate_wrong)}"
                ),
                width=360,
                height=240,
            ),
        ]
    )
    canvas = _stack_v([top_row, bottom_row])
    output_path = output_dir / f"{case.case_id}.jpg"
    _write_image(output_path, canvas)
    return str(output_path)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _render_retrieval_support_panels(output_dir: Path) -> List[Dict[str, Any]]:
    panels: List[Dict[str, Any]] = []
    for example in RETRIEVAL_SUPPORT_EXAMPLES:
        payload = _load_json(example["query_json"])
        query_dir = example["query_json"].parent
        query_crop = _read_image(query_dir / "query_object_crop.jpg", max_side=320)
        top_k_images = [
            _read_image(Path(path), max_side=320)
            for path in payload.get("artifacts", {}).get("top_k_images", [])[:3]
        ]
        tiles = [_image_tile(query_crop, f"Query crop | {example['label']}")]
        for index, image in enumerate(top_k_images, start=1):
            record = payload["top_k"][index - 1]
            tiles.append(
                _image_tile(
                    image,
                    f"Top {index} | entry {record['id']} | score={float(record['object_score']):.3f}",
                )
            )
        canvas = _stack_h(tiles)
        output_path = output_dir / f"{example['case_id']}.jpg"
        _write_image(output_path, canvas)
        panels.append(
            {
                "case_id": example["case_id"],
                "panel_path": str(output_path),
                "pattern": example["pattern"],
                "label": example["label"],
                "metrics": payload.get("metrics", {}),
                "debug": payload.get("debug", {}),
            }
        )
    return panels


def _category_summary(cases: Sequence[CaseRecord]) -> Dict[str, Any]:
    main_cases = [case for case in cases if case.case_bucket == "main"]
    counts_by_root: Dict[str, int] = {}
    counts_by_stage: Dict[str, int] = {}
    counts_by_family: Dict[str, int] = {}
    for case in main_cases:
        counts_by_root[case.root_cause_label] = counts_by_root.get(case.root_cause_label, 0) + 1
        counts_by_stage[case.first_error_stage] = counts_by_stage.get(case.first_error_stage, 0) + 1
        counts_by_family[case.primary_signal_family] = counts_by_family.get(case.primary_signal_family, 0) + 1

    text_branch_count = sum(1 for case in main_cases if case.primary_cause == "text_similarity_error")
    dino_branch_count = sum(1 for case in main_cases if case.primary_cause == "visual_similarity_error")
    multi_branch_count = sum(1 for case in main_cases if case.primary_cause == "text_visual_both_wrong")
    geometry_suppressed_count = sum(
        1 for case in main_cases if case.root_cause_label == "similarity_correct_but_geometry_wrong"
    )
    orientation_first_error_count = sum(1 for case in main_cases if case.primary_cause == "orientation_error")
    depth_error_count = sum(1 for case in main_cases if case.primary_cause == "depth_error")
    contamination_count = sum(1 for case in main_cases if case.primary_cause == "depth_contamination_from_other_object")
    return {
        "main_case_count": len(main_cases),
        "supporting_case_count": sum(1 for case in cases if case.case_bucket != "main"),
        "counts_by_root_cause": dict(sorted(counts_by_root.items())),
        "counts_by_first_error_stage": dict(sorted(counts_by_stage.items())),
        "counts_by_signal_family": dict(sorted(counts_by_family.items())),
        "direct_answers": {
            "geometry_vs_similarity_first_error": {
                "geometry": counts_by_stage.get("geometry", 0),
                "similarity": counts_by_stage.get("similarity", 0),
                "interaction": counts_by_stage.get("interaction", 0),
            },
            "orientation_first_error_count": orientation_first_error_count,
            "depth_wrong_with_orientation_ok_count": depth_error_count,
            "depth_contamination_count": contamination_count,
            "wrong_pick_driver_counts": {
                "text_branch_preferred_it": text_branch_count,
                "dino_branch_preferred_it": dino_branch_count,
                "geometry_suppressed_correct": geometry_suppressed_count,
                "multiple_branches_agreed_on_wrong_one": multi_branch_count,
            },
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_cases_csv(path: Path, cases: Sequence[CaseRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(case) for case in cases]
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report_markdown(
    *,
    cases: Sequence[CaseRecord],
    category_summary: Mapping[str, Any],
    retrieval_panels: Sequence[Mapping[str, Any]],
    diagnostic_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    run_traces: Mapping[str, Mapping[str, Any]],
    canonical_manifest_path: Path,
) -> str:
    main_cases = [case for case in cases if case.case_bucket == "main"]
    supporting_cases = [case for case in cases if case.case_bucket != "main"]
    lines: List[str] = []
    lines.append("# Root-Cause Analysis for `spatial_db_origin`")
    lines.append("")
    lines.append("## Scope And Validation")
    lines.append(
        f"- Investigated the selected-view same-instance clustering problem on `{RUN_SPECS['run010'].db_dir}` using the two stored sequential runs."
    )
    for run_key, run_trace in run_traces.items():
        validation = run_trace["validation"]
        lines.append(
            f"- `{run_key}` validation: checked {validation['assignment_rows_checked']} assignment rows, "
            f"max abs diff {validation['max_abs_diff']:.6f}, "
            f"step-count match={validation['step_counts_match']}, "
            f"final-cluster-count match={validation['final_cluster_count_matches']}."
        )
    lines.append(
        f"- Built an analysis-only canonical manifest for the 53 selected-view objects at "
        f"[canonical_manifest.json]({canonical_manifest_path.resolve().as_posix()})."
    )
    lines.append("")
    lines.append("## Main Findings")
    lines.append(
        f"- Main conclusions are based on {len(main_cases)} high-confidence bad cases. "
        f"{len(supporting_cases)} additional supporting / ambiguous cases are included in the appendix but do not drive the summary."
    )
    lines.append(
        f"- First error stage: geometry {category_summary['direct_answers']['geometry_vs_similarity_first_error']['geometry']}/{len(main_cases)}, "
        f"similarity {category_summary['direct_answers']['geometry_vs_similarity_first_error']['similarity']}/{len(main_cases)}, "
        f"interaction {category_summary['direct_answers']['geometry_vs_similarity_first_error']['interaction']}/{len(main_cases)}."
    )
    lines.append(
        f"- Orientation was not the first error in any main case. All four selected views share the same frame orientation (270 deg), "
        f"and the angle-bucket fields remain internally consistent with the crop laterality."
    )
    lines.append(
        f"- Depth / geometry is the dominant problem: {category_summary['direct_answers']['depth_wrong_with_orientation_ok_count']}/{len(main_cases)} "
        f"main failures have the correct semantic candidate on top (or near the top) but lose it at the geometry-threshold stage."
    )
    lines.append(
        f"- Nearby-object depth contamination is rare in this slice: {category_summary['direct_answers']['depth_contamination_count']}/{len(main_cases)} "
        f"main cases. The dominant geometry failures happen on fully visible fridge / island views, so the issue is position/depth drift rather than occluder leakage."
    )
    lines.append(
        f"- Wrong-pick driver counts: geometry suppressed the correct candidate "
        f"{category_summary['direct_answers']['wrong_pick_driver_counts']['geometry_suppressed_correct']} times, "
        f"text branch led {category_summary['direct_answers']['wrong_pick_driver_counts']['text_branch_preferred_it']} time, "
        f"DINO-only led {category_summary['direct_answers']['wrong_pick_driver_counts']['dino_branch_preferred_it']} time, "
        f"and multiple branches agreed on the wrong candidate "
        f"{category_summary['direct_answers']['wrong_pick_driver_counts']['multiple_branches_agreed_on_wrong_one']} times."
    )
    lines.append("")
    lines.append("## Root-Cause Breakdown")
    lines.append("| root_cause_label | count | representative cases |")
    lines.append("| --- | ---: | --- |")
    grouped: Dict[str, List[CaseRecord]] = {}
    for case in main_cases:
        grouped.setdefault(case.root_cause_label, []).append(case)
    for root_cause, root_cases in sorted(grouped.items()):
        examples = ", ".join(case.case_id for case in root_cases[:3])
        lines.append(f"| `{root_cause}` | {len(root_cases)} | {examples} |")
    lines.append("")
    lines.append("## Direct Answers")
    lines.append(
        f"1. Geometry introduces the first error slightly more often than similarity in this reviewed set: "
        f"{category_summary['direct_answers']['geometry_vs_similarity_first_error']['geometry']} geometry-led vs "
        f"{category_summary['direct_answers']['geometry_vs_similarity_first_error']['similarity']} similarity-led cases."
    )
    lines.append(
        "2. Orientation is not a meaningful first-error source in these selected views. The stored frame orientation is constant and the angle buckets are plausible."
    )
    lines.append(
        "3. Depth is often wrong even when orientation looks fine. The fridge and island cases repeatedly keep the right semantic cluster on top but lose it because the geometry gate or cross-affinity threshold drops the edge."
    )
    lines.append(
        "4. Nearby-object contamination is uncommon in the main failures. The strongest geometry failures occur on fully visible objects with low occlusion, so depth drift is more common than contamination."
    )
    lines.append(
        "5. When the system picks the wrong object here, it is more often because geometry suppresses the correct one than because text or DINO alone pick the wrong one. Text does matter in the coffee-maker confusion, and DINO matters in the island / seat near-misses."
    )
    lines.append(
        "6. In crowded or partially occluded seating cases, the main issue is misleading similarity from partial crops rather than obviously wrong depth. The DINO-heavy run mistakes one seat instance for another, and the mixed run only avoids the final merge because of the later graph constraint."
    )
    lines.append(
        "7. Recurring failure modes are tied to repeated kitchen layouts: the island, fridge, repeated cabinet banks, and multiple similar chairs/stools. Coffee-maker variants also recur as a text-vs-DINO disagreement pattern."
    )
    lines.append("")
    lines.append("## Representative Main Cases")
    for case in main_cases:
        panel_line = f"[panel]({Path(case.panel_path).resolve().as_posix()})" if case.panel_path else "panel missing"
        lines.append(
            f"- `{case.case_id}`: {case.explanation} "
            f"Correct combined={_fmt(case.combined_similarity_correct)}, wrong combined={_fmt(case.combined_similarity_wrong)}, {panel_line}."
        )
    lines.append("")
    lines.append("## Floor-Plan Examples")
    for case_id in FLOORPLAN_EXAMPLE_CASE_IDS:
        case = next((item for item in main_cases if item.case_id == case_id), None)
        if case is None:
            continue
        panel_line = f"[panel]({Path(case.panel_path).resolve().as_posix()})" if case.panel_path else "panel missing"
        correct_note = (
            f"correct=({_fmt(case.correct_match_global_x)},{_fmt(case.correct_match_global_y)}) in {case.correct_match_view_id}"
            if case.correct_match_global_x is not None and case.correct_match_global_y is not None
            else "correct=new cluster"
        )
        wrong_note = (
            f"wrong=({_fmt(case.wrong_match_global_x)},{_fmt(case.wrong_match_global_y)}) in {case.wrong_match_view_id}"
            if case.wrong_match_global_x is not None and case.wrong_match_global_y is not None
            else "wrong=n/a"
        )
        lines.append(
            f"- `{case.case_id}`: camera=({_fmt(case.camera_global_x)},{_fmt(case.camera_global_y)}) in {case.view_id}; "
            f"query=({_fmt(case.query_global_x)},{_fmt(case.query_global_y)}); "
            f"{correct_note}; {wrong_note}. "
            f"This map tile shows whether the failure is a local confusion between nearby instances or a geometry miss on the same physical object. {panel_line}."
        )
    lines.append("")
    lines.append("## Angle Examples")
    angle_examples = list(diagnostic_examples.get("angle", []))
    for orientation_deg in (270, 90, 180):
        orientation_examples = [
            item for item in angle_examples if int(round(float(item.get("camera_orientation_deg") or -1))) == orientation_deg
        ]
        if not orientation_examples:
            continue
        lines.append(f"### {orientation_deg} deg")
        for example in orientation_examples:
            panel_line = f"[panel]({Path(example['panel_path']).resolve().as_posix()})"
            lines.append(
                f"- `{example['example_id']}`: camera=({_fmt(example.get('camera_global_x'))},{_fmt(example.get('camera_global_y'))}) "
                f"orientation={_fmt(example.get('camera_orientation_deg'))} deg; "
                f"object=({_fmt(example.get('object_global_x'))},{_fmt(example.get('object_global_y'))}); "
                f"stored_bearing={_fmt(example.get('stored_bearing_deg'))} deg; "
                f"geometry_bearing={_fmt(example.get('geometry_bearing_deg'))} deg; "
                f"delta={_fmt(example.get('bearing_delta_deg'))} deg. {panel_line}."
            )
    lines.append("")
    lines.append("## Depth Examples")
    depth_examples = list(diagnostic_examples.get("depth", []))
    for orientation_deg in (270, 90, 180):
        orientation_examples = [
            item for item in depth_examples if int(round(float(item.get("camera_orientation_deg") or -1))) == orientation_deg
        ]
        if not orientation_examples:
            continue
        lines.append(f"### {orientation_deg} deg")
        for example in orientation_examples:
            panel_line = f"[panel]({Path(example['panel_path']).resolve().as_posix()})"
            lines.append(
                f"- `{example['example_id']}`: camera=({_fmt(example.get('camera_global_x'))},{_fmt(example.get('camera_global_y'))}) "
                f"to object=({_fmt(example.get('object_global_x'))},{_fmt(example.get('object_global_y'))}); "
                f"depth_pro={_fmt(example.get('depth_pro_m'))} m; "
                f"planar_distance={_fmt(example.get('planar_distance_m'))} m; "
                f"occlusion={example.get('occlusion_level')}. {panel_line}."
            )
    lines.append("")
    lines.append("## Supporting / Ambiguous Cases")
    for case in supporting_cases:
        panel_line = f"[panel]({Path(case.panel_path).resolve().as_posix()})" if case.panel_path else "panel missing"
        lines.append(f"- `{case.case_id}`: {case.question_answer} {panel_line}.")
    lines.append("")
    lines.append("## Retrieval Appendix")
    for example in retrieval_panels:
        panel_line = f"[panel]({Path(example['panel_path']).resolve().as_posix()})"
        metrics = example.get("metrics", {})
        lines.append(
            f"- `{example['case_id']}` ({example['label']}): {example['pattern']}; "
            f"pos_error={metrics.get('pos_error')}, yaw_error={metrics.get('yaw_error')}. {panel_line}."
        )
    lines.append("")
    lines.append("## Conclusion")
    lines.append(
        "The main problem in this reviewed slice is geometry, specifically depth / position drift that pushes otherwise-correct matches below the sequential graph threshold. "
        "The secondary problem is similarity confusion on repeated or partially visible objects, especially seats and coffee-maker variants. "
        "Fix priority should therefore be: 1) diagnose why global XY / depth drift is large enough to kill fridge and island links, 2) reduce partial-view seat confusion in DINO / crop similarity, and 3) tighten text descriptions for visually similar kitchen appliances."
    )
    return "\n".join(lines) + "\n"


def generate_root_cause_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    base_sequence = load_selected_sequence(RUN_SPECS["run010"])
    canonical_manifest = base_sequence["canonical_manifest"]
    canonical_manifest_path = output_dir / "canonical_manifest.json"
    _write_json(canonical_manifest_path, list(canonical_manifest.values()))

    run_traces = {run_key: _reconstruct_run(spec, canonical_manifest) for run_key, spec in RUN_SPECS.items()}
    cases = _mine_case_records(run_traces, canonical_manifest)

    for index, case in enumerate(cases, start=1):
        panel_path = _render_case_panel(case, panels_dir)
        case.panel_path = panel_path

    retrieval_panels = _render_retrieval_support_panels(panels_dir)
    diagnostic_examples = _render_diagnostic_examples(panels_dir)
    summary = _category_summary(cases)
    summary["validation"] = {
        run_key: run_trace["validation"] for run_key, run_trace in run_traces.items()
    }
    summary["retrieval_appendix_count"] = len(retrieval_panels)
    summary["angle_example_count"] = len(diagnostic_examples.get("angle", []))
    summary["depth_example_count"] = len(diagnostic_examples.get("depth", []))
    _write_json(output_dir / "category_summary.json", summary)
    _write_json(output_dir / "angle_examples.json", list(diagnostic_examples.get("angle", [])))
    _write_json(output_dir / "depth_examples.json", list(diagnostic_examples.get("depth", [])))

    bad_cases_json = [asdict(case) for case in cases]
    _write_json(output_dir / "bad_cases.json", bad_cases_json)
    _write_cases_csv(output_dir / "bad_cases.csv", cases)

    report = _report_markdown(
        cases=cases,
        category_summary=summary,
        retrieval_panels=retrieval_panels,
        diagnostic_examples=diagnostic_examples,
        run_traces=run_traces,
        canonical_manifest_path=canonical_manifest_path,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "num_cases": len(cases),
        "num_main_cases": sum(1 for case in cases if case.case_bucket == "main"),
        "num_supporting_cases": sum(1 for case in cases if case.case_bucket != "main"),
        "num_angle_examples": len(diagnostic_examples.get("angle", [])),
        "num_depth_examples": len(diagnostic_examples.get("depth", [])),
        "validation": summary["validation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the root-cause analysis report for spatial_db_origin clustering errors.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for the report and panels")
    args = parser.parse_args()
    result = generate_root_cause_analysis(Path(args.output_dir))
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
