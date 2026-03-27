import argparse
import json
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from spatial_rag.config import OBJECT_TEXT_MODE, OBJECT_USE_CACHE, SCENE_PATH, VLM_ANGLE_STEP
from spatial_rag.detector import Detector
from spatial_rag.object_index import load_object_db, load_object_faiss_index
from spatial_rag.vlm_captioner import VLMCaptioner
from spatial_rag.vpr_query import (
    _entry_world_position,
    _prepare_overlay_base,
    _set_agent_pose_2d,
    _validate_object_text_mode,
    _world_to_pixel_center,
    _world_to_pixel_floor,
    circular_abs_diff_deg,
    load_spatial_db,
)


class NoValidObjectDetectionsError(RuntimeError):
    """Raised when a query image contains no valid object detections."""


def _str_to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _runtime_log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[ObjectQuery][{ts}] {message}", flush=True)


def _safe_text(value: Any) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return " ".join(text.split())


def _serialize_detection(det: Dict[str, Any]) -> Dict[str, Any]:
    bbox = [int(v) for v in det["bbox_xyxy"]]
    payload = {
        "label": str(det["label"]),
        "confidence": float(det["confidence"]),
        "bbox_xyxy": bbox,
        "area_ratio": float(det["area_ratio"]),
    }
    if "crop_bbox_xyxy" in det:
        payload["crop_bbox_xyxy"] = [int(v) for v in det["crop_bbox_xyxy"]]
    return payload


def _clamp_bbox_xyxy(bbox: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    x1 = int(np.clip(round(x1), 0, max(width - 1, 0)))
    y1 = int(np.clip(round(y1), 0, max(height - 1, 0)))
    x2 = int(np.clip(round(x2), 0, width))
    y2 = int(np.clip(round(y2), 0, height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def filter_valid_detections(
    detections: Sequence[Dict[str, Any]],
    image_shape: Sequence[int],
    detector_conf: float = 0.35,
    min_bbox_side_px: int = 32,
    min_bbox_area_ratio: float = 0.005,
    max_bbox_area_ratio: float = 0.60,
) -> List[Dict[str, Any]]:
    height = int(image_shape[0])
    width = int(image_shape[1])
    image_area = float(max(width * height, 1))
    valid: List[Dict[str, Any]] = []

    for det in detections:
        bbox_raw = det.get("bbox")
        if bbox_raw is None or len(bbox_raw) < 4:
            continue
        conf = float(det.get("confidence", 0.0))
        if conf < float(detector_conf):
            continue
        x1, y1, x2, y2 = _clamp_bbox_xyxy(bbox_raw, width=width, height=height)
        bw = int(x2 - x1)
        bh = int(y2 - y1)
        if bw < int(min_bbox_side_px) or bh < int(min_bbox_side_px):
            continue
        area_ratio = float((bw * bh) / image_area)
        if area_ratio < float(min_bbox_area_ratio) or area_ratio > float(max_bbox_area_ratio):
            continue
        valid.append(
            {
                "label": str(det.get("label") or "unknown"),
                "confidence": conf,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "area_ratio": area_ratio,
            }
        )

    valid.sort(
        key=lambda item: (
            str(item["label"]),
            -float(item["confidence"]),
            int(item["bbox_xyxy"][0]),
            int(item["bbox_xyxy"][1]),
            int(item["bbox_xyxy"][2]),
            int(item["bbox_xyxy"][3]),
        )
    )
    return valid


def select_detection(
    valid_detections: Sequence[Dict[str, Any]],
    selection_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if not valid_detections:
        raise NoValidObjectDetectionsError("No valid detections available for selection.")
    rng = random.Random(selection_seed) if selection_seed is not None else random.Random()
    choice = dict(valid_detections[rng.randrange(len(valid_detections))])
    return choice


def crop_with_padding(
    image_rgb: np.ndarray,
    bbox_xyxy: Sequence[int],
    padding_ratio: float = 0.10,
) -> Tuple[np.ndarray, List[int]]:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy[:4]]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * float(padding_ratio)))
    pad_y = int(round(bh * float(padding_ratio)))
    px1 = int(np.clip(x1 - pad_x, 0, width))
    py1 = int(np.clip(y1 - pad_y, 0, height))
    px2 = int(np.clip(x2 + pad_x, 0, width))
    py2 = int(np.clip(y2 + pad_y, 0, height))
    if px2 <= px1 or py2 <= py1:
        raise ValueError("Padded crop is empty.")
    crop = image_rgb[py1:py2, px1:px2].copy()
    if crop.size == 0:
        raise ValueError("Padded crop is empty.")
    return crop, [px1, py1, px2, py2]


def _resolve_query_text(description: Dict[str, Any], yolo_label: str) -> Tuple[str, str, str]:
    resolved_label = _safe_text(description.get("label"))
    short_desc = _safe_text(description.get("short_description"))
    long_desc = _safe_text(description.get("long_description"))
    source = str(description.get("source") or "unknown")
    if (not resolved_label or resolved_label.lower() == "unknown") and yolo_label:
        resolved_label = _safe_text(yolo_label)
    long_desc_clean = long_desc if long_desc and long_desc.lower() != "unknown" else ""
    if short_desc and short_desc.lower() != "unknown":
        return short_desc, long_desc_clean or short_desc or resolved_label, source
    if long_desc_clean:
        return long_desc_clean, long_desc_clean, source
    label = _safe_text(yolo_label) or "unknown"
    return label, long_desc_clean or label, "fallback_label"


def _select_query_text_for_mode(
    query_text_short: str,
    query_text_long: str,
    object_text_mode: str,
) -> str:
    mode = str(object_text_mode or "short").strip().lower()
    short_clean = _safe_text(query_text_short)
    long_clean = _safe_text(query_text_long)
    if mode == "long" and long_clean and long_clean.lower() != "unknown":
        return long_clean
    return short_clean or long_clean or "unknown"


def _aggregate_entry_scores(
    search_scores: np.ndarray,
    search_indices: np.ndarray,
    object_meta: Sequence[Dict[str, Any]],
    entry_ids: Sequence[int],
) -> Dict[int, float]:
    scores: Dict[int, float] = {int(entry_id): 0.0 for entry_id in entry_ids}
    if search_scores.ndim != 2 or search_indices.ndim != 2 or search_scores.shape != search_indices.shape:
        raise ValueError("search_scores/search_indices must be matching 2D arrays")

    for row_scores, row_indices in zip(search_scores, search_indices):
        for score, obj_idx in zip(row_scores, row_indices):
            obj_int = int(obj_idx)
            if obj_int < 0 or obj_int >= len(object_meta):
                continue
            entry_id = int(object_meta[obj_int].get("entry_id", -1))
            if entry_id < 0:
                continue
            score_f = float(score)
            if score_f > scores.get(entry_id, 0.0):
                scores[entry_id] = score_f
    return scores


def _best_object_match_per_entry(
    search_scores: np.ndarray,
    search_indices: np.ndarray,
    object_meta: Sequence[Dict[str, Any]],
    entry_ids: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    best: Dict[int, Dict[str, Any]] = {int(entry_id): {} for entry_id in entry_ids}
    if search_scores.ndim != 2 or search_indices.ndim != 2 or search_scores.shape != search_indices.shape:
        raise ValueError("search_scores/search_indices must be matching 2D arrays")

    for row_scores, row_indices in zip(search_scores, search_indices):
        for score, obj_idx in zip(row_scores, row_indices):
            obj_int = int(obj_idx)
            if obj_int < 0 or obj_int >= len(object_meta):
                continue
            meta = object_meta[obj_int]
            entry_id = int(meta.get("entry_id", -1))
            if entry_id < 0:
                continue
            score_f = float(score)
            current = best.get(entry_id, {})
            if score_f > float(current.get("score", -1.0)):
                best[entry_id] = {
                    "score": score_f,
                    "object_index": obj_int,
                    "object_meta": meta,
                }
    return best


def _laterality_from_bbox(bbox_xyxy: Sequence[int], image_width: int) -> str:
    if image_width <= 0:
        return "center"
    x1, _, x2, _ = [int(v) for v in bbox_xyxy[:4]]
    center_x = 0.5 * float(x1 + x2)
    if center_x < float(image_width) / 3.0:
        return "left"
    if center_x > (2.0 * float(image_width) / 3.0):
        return "right"
    return "center"


def _bucket_orientation_deg(frame_orientation_deg: float, laterality: str, angle_step_deg: int = VLM_ANGLE_STEP) -> float:
    orientation = float(frame_orientation_deg) % 360.0
    bucket = str(laterality or "center").strip().lower()
    if bucket == "left":
        return (orientation + float(angle_step_deg)) % 360.0
    if bucket == "right":
        return (orientation - float(angle_step_deg)) % 360.0
    return orientation


def _project_object_xy(
    origin_x: float,
    origin_y: float,
    orientation_deg: float,
    distance_m: Optional[float],
) -> Optional[Tuple[float, float]]:
    if distance_m is None:
        return None
    dist = float(distance_m)
    if not math.isfinite(dist) or dist < 0.0:
        return None
    yaw = math.radians(float(orientation_deg))
    obj_x = float(origin_x - math.sin(yaw) * dist)
    obj_y = float(origin_y - math.cos(yaw) * dist)
    return obj_x, obj_y


def _rank_entries(
    entries: Sequence[Dict[str, Any]],
    entry_scores: Dict[int, float],
    top_k: int,
) -> Tuple[List[int], List[Tuple[int, float]]]:
    ranked_pairs = [
        (int(entry["id"]), float(entry_scores.get(int(entry["id"]), 0.0)))
        for entry in entries
    ]
    ranked_pairs.sort(key=lambda item: (-float(item[1]), int(item[0])))

    top_idx: List[int] = []
    seen_xy = set()
    k = min(max(1, int(top_k)), 5, len(entries))
    entry_by_id = {int(entry["id"]): idx for idx, entry in enumerate(entries)}
    for entry_id, _score in ranked_pairs:
        idx = int(entry_by_id[entry_id])
        entry = entries[idx]
        key = (round(float(entry["x"]), 3), round(float(entry["y"]), 3))
        if key in seen_xy:
            continue
        seen_xy.add(key)
        top_idx.append(idx)
        if len(top_idx) >= k:
            break
    if not top_idx and entries:
        top_idx = [0]
    return top_idx, ranked_pairs


def _draw_label_box(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    fg_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(org[0])
    y = int(org[1])
    pad_x = 8
    pad_y = 6
    box_x1 = max(0, x)
    box_y1 = max(0, y - th - baseline - pad_y * 2)
    box_x2 = min(image.shape[1], x + tw + pad_x * 2)
    box_y2 = min(image.shape[0], y)
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), 1)
    text_x = box_x1 + pad_x
    text_y = min(image.shape[0] - 2, box_y2 - pad_y - baseline)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, fg_color, thickness, cv2.LINE_AA)


def draw_query_detection_overlay(
    query_rgb: np.ndarray,
    detections: Sequence[Dict[str, Any]],
    selected_detection: Dict[str, Any],
    output_path: str,
) -> str:
    canvas = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)
    selected_bbox = tuple(int(v) for v in selected_detection["bbox_xyxy"])
    for det in detections:
        bbox = tuple(int(v) for v in det["bbox_xyxy"])
        is_selected = bbox == selected_bbox and str(det["label"]) == str(selected_detection["label"])
        color = (0, 0, 255) if is_selected else (140, 140, 140)
        thickness = 4 if is_selected else 2
        cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        label = (
            f"selected object | {det['label']} {float(det['confidence']):.2f}"
            if is_selected
            else f"{det['label']} {float(det['confidence']):.2f}"
        )
        _draw_label_box(
            canvas,
            label,
            (bbox[0], max(28, bbox[1])),
            bg_color=(255, 255, 255) if is_selected else (230, 230, 230),
            fg_color=(0, 0, 0),
            font_scale=0.7 if is_selected else 0.6,
            thickness=2,
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save detection overlay: {out}")
    return str(out)


def _resize_to_fit(image: np.ndarray, target_w: int, target_h: int, fill_color: Tuple[int, int, int]) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Invalid image shape for resize.")
    scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    panel = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    off_x = max(0, (target_w - new_w) // 2)
    off_y = max(0, (target_h - new_h) // 2)
    panel[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return panel


def _wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        add_len = len(word) if not current else len(word) + 1
        if current and current_len + add_len > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += add_len
    if current:
        lines.append(" ".join(current))
    return lines


def _paste_panel(canvas: np.ndarray, panel: np.ndarray, x: int, y: int) -> None:
    h, w = panel.shape[:2]
    canvas[y:y + h, x:x + w] = panel


def _map_rank_color(rank: int) -> Tuple[int, int, int]:
    if rank <= 1:
        return (0, 0, 255)
    if rank == 2:
        return (0, 128, 255)
    if rank == 3:
        return (0, 180, 255)
    if rank == 4:
        return (0, 215, 255)
    return (0, 240, 255)


def _draw_ranked_map(
    explorer,
    entries: Sequence[Dict[str, Any]],
    top_k_entries: Sequence[Dict[str, Any]],
    x0: float,
    y0: float,
    theta0: float,
    query_world_y: float,
    actual_query_world_position: Optional[Sequence[float]] = None,
) -> np.ndarray:
    base, proj = _prepare_overlay_base(explorer, floor_height=query_world_y)
    canvas = base.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]

    def world_to_pixel(wx: float, wy: float, wz: float) -> Optional[Tuple[int, int]]:
        mode, pinfo = proj
        if mode == "floor":
            return _world_to_pixel_floor(wx, wz, w, h, pinfo)
        return _world_to_pixel_center(wx, wy, wz, w, h, pinfo)

    def orientation_arrow_tip(wx: float, wy: float, wz: float, yaw_deg: float, world_len: float) -> Optional[Tuple[int, int]]:
        yaw = np.deg2rad(float(yaw_deg))
        tip_wx = float(wx - np.sin(yaw) * world_len)
        tip_wz = float(wz - np.cos(yaw) * world_len)
        return world_to_pixel(tip_wx, wy, tip_wz)

    query_world_x = float(actual_query_world_position[0]) if actual_query_world_position is not None else float(x0)
    query_world_z = float(actual_query_world_position[2]) if actual_query_world_position is not None else float(y0)
    p_query = world_to_pixel(query_world_x, float(query_world_y), query_world_z)
    if p_query is not None:
        cv2.circle(canvas, p_query, 17, (255, 255, 255), -1)
        cv2.circle(canvas, p_query, 13, (255, 255, 0), -1)
        gt_tip = orientation_arrow_tip(query_world_x, float(query_world_y), query_world_z, float(theta0), world_len=1.25)
        if gt_tip is not None:
            cv2.arrowedLine(canvas, p_query, gt_tip, (255, 255, 0), 4, tipLength=0.35)
        _draw_label_box(canvas, "Q", (p_query[0] + 8, max(36, p_query[1] - 8)), (255, 255, 255), (0, 0, 0))

    used_label_positions: List[Tuple[int, int]] = []
    for rank, entry in enumerate(top_k_entries, start=1):
        wx, wy, wz = _entry_world_position(entry, default_world_y=query_world_y)
        p_rank = world_to_pixel(wx, wy, wz)
        if p_rank is None:
            continue
        color = _map_rank_color(rank)
        tip = orientation_arrow_tip(wx, wy, wz, float(entry["orientation"]), world_len=0.95)
        if tip is not None:
            cv2.arrowedLine(canvas, p_rank, tip, color, 3, tipLength=0.35)
        cv2.circle(canvas, p_rank, 12 if rank == 1 else 9, (255, 255, 255), -1)
        cv2.circle(canvas, p_rank, 9 if rank == 1 else 7, color, -1)
        label_x = int(np.clip(p_rank[0] + 12, 0, max(0, w - 80)))
        label_y = int(np.clip(p_rank[1] - 10, 30, max(30, h - 6)))
        while any(abs(label_x - ox) < 56 and abs(label_y - oy) < 24 for ox, oy in used_label_positions):
            label_y = int(np.clip(label_y + 26, 30, max(30, h - 6)))
        used_label_positions.append((label_x, label_y))
        _draw_label_box(canvas, f"#{rank}", (label_x, label_y), (255, 255, 255), (0, 0, 0))

    cv2.putText(
        canvas,
        "Object Localization Top-K",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return canvas


def compose_object_query_overlay(
    map_canvas_bgr: np.ndarray,
    query_rgb: np.ndarray,
    detections: Sequence[Dict[str, Any]],
    selected_detection: Dict[str, Any],
    crop_rgb: np.ndarray,
    query_text: str,
    top1_score: float,
    top_k_records: Optional[Sequence[Dict[str, Any]]] = None,
    output_size: Tuple[int, int] = (2200, 1200),
) -> np.ndarray:
    total_w, total_h = int(output_size[0]), int(output_size[1])
    canvas = np.full((total_h, total_w, 3), 245, dtype=np.uint8)
    map_w = int(round(total_w * 0.62))
    side_w = total_w - map_w
    margin = 28
    gap = 20

    map_panel = _resize_to_fit(map_canvas_bgr, map_w - margin * 2, total_h - margin * 2, (30, 30, 30))
    _paste_panel(canvas, map_panel, margin, margin)
    cv2.rectangle(canvas, (margin, margin), (margin + map_panel.shape[1], margin + map_panel.shape[0]), (35, 35, 35), 2)

    query_panel_h = int(round((total_h - margin * 2 - gap) * 0.56))
    crop_panel_h = total_h - margin * 2 - gap - query_panel_h
    query_panel_w = side_w - margin * 2
    crop_img_w = int(round(query_panel_w * 0.46))
    text_panel_w = query_panel_w - crop_img_w - gap

    query_overlay_bgr = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)
    selected_bbox = tuple(int(v) for v in selected_detection["bbox_xyxy"])
    for det in detections:
        bbox = tuple(int(v) for v in det["bbox_xyxy"])
        is_selected = bbox == selected_bbox and str(det["label"]) == str(selected_detection["label"])
        color = (0, 0, 255) if is_selected else (140, 140, 140)
        thickness = 4 if is_selected else 2
        cv2.rectangle(query_overlay_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        if is_selected:
            _draw_label_box(
                query_overlay_bgr,
                "selected object",
                (bbox[0], max(30, bbox[1])),
                (255, 255, 255),
                (0, 0, 0),
                font_scale=0.8,
            )

    query_panel = _resize_to_fit(query_overlay_bgr, query_panel_w, query_panel_h, (255, 255, 255))
    side_x = map_w + margin
    _paste_panel(canvas, query_panel, side_x, margin)
    cv2.rectangle(canvas, (side_x, margin), (side_x + query_panel_w, margin + query_panel_h), (35, 35, 35), 2)
    cv2.putText(canvas, "Query Image", (side_x, margin - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)

    crop_panel = _resize_to_fit(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), crop_img_w, crop_panel_h, (255, 255, 255))
    crop_y = margin + query_panel_h + gap
    _paste_panel(canvas, crop_panel, side_x, crop_y)
    cv2.rectangle(canvas, (side_x, crop_y), (side_x + crop_img_w, crop_y + crop_panel_h), (35, 35, 35), 2)
    cv2.putText(canvas, "Selected Crop", (side_x, crop_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)

    text_x = side_x + crop_img_w + gap
    text_panel = np.full((crop_panel_h, text_panel_w, 3), 255, dtype=np.uint8)
    cv2.rectangle(text_panel, (0, 0), (text_panel_w - 1, crop_panel_h - 1), (35, 35, 35), 2)
    info_lines = [
        f"YOLO label: {_safe_text(selected_detection.get('label')) or 'unknown'}",
        f"Detection conf: {float(selected_detection.get('confidence', 0.0)):.2f}",
        f"Top1 score: {float(top1_score):.4f}",
    ]
    if top_k_records:
        distance_tokens = []
        for rec in list(top_k_records)[:5]:
            dist_value = rec.get("distance_to_query_object_m_approx")
            dist_label = "obj"
            if dist_value is None:
                dist_value = rec.get("distance_to_query_entry_m")
                dist_label = "pose"
            if dist_value is None:
                continue
            distance_tokens.append(f"#{int(rec['rank'])} {dist_label} {float(dist_value):.2f}m")
        if distance_tokens:
            info_lines.append("Top-K dist:")
            info_lines.extend(_wrap_text(" | ".join(distance_tokens), max_chars=30)[:3])
    info_lines.extend([
        "Query text:",
    ])
    cursor_y = 40
    for line in info_lines:
        cv2.putText(text_panel, line, (18, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (30, 30, 30), 2, cv2.LINE_AA)
        cursor_y += 40
    for line in _wrap_text(_safe_text(query_text), max_chars=34)[:7]:
        cv2.putText(text_panel, line, (18, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (60, 60, 60), 2, cv2.LINE_AA)
        cursor_y += 34
    _paste_panel(canvas, text_panel, text_x, crop_y)

    return canvas


def draw_object_query_topk_overlay(
    explorer,
    entries: Sequence[Dict[str, Any]],
    top_k_entries: Sequence[Dict[str, Any]],
    pred_entry: Dict[str, Any],
    x0: float,
    y0: float,
    theta0: float,
    query_rgb: np.ndarray,
    detections: Sequence[Dict[str, Any]],
    selected_detection: Dict[str, Any],
    crop_rgb: np.ndarray,
    query_text: str,
    top1_score: float,
    top_k_records: Optional[Sequence[Dict[str, Any]]],
    output_path: str,
    query_world_y: float,
    actual_query_world_position: Optional[Sequence[float]] = None,
) -> str:
    del pred_entry  # top_k_entries already contains rank-1 entry.
    map_canvas = _draw_ranked_map(
        explorer=explorer,
        entries=entries,
        top_k_entries=top_k_entries,
        x0=x0,
        y0=y0,
        theta0=theta0,
        query_world_y=query_world_y,
        actual_query_world_position=actual_query_world_position,
    )
    overlay = compose_object_query_overlay(
        map_canvas_bgr=map_canvas,
        query_rgb=query_rgb,
        detections=detections,
        selected_detection=selected_detection,
        crop_rgb=crop_rgb,
        query_text=query_text,
        top1_score=top1_score,
        top_k_records=top_k_records,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), overlay)
    if not ok:
        raise RuntimeError(f"Failed to save object overlay: {out}")
    return str(out)


def build_top_k_contact_sheet(top_k_records: Sequence[Dict[str, Any]], output_path: str) -> Optional[str]:
    if not top_k_records:
        return None
    tiles: List[np.ndarray] = []
    for rec in top_k_records:
        img_path = rec.get("retrieved_image_path")
        if not img_path or not Path(img_path).exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        tile = _resize_to_fit(img, 420, 250, (255, 255, 255))
        dist_value = rec.get("distance_to_query_object_m_approx")
        dist_label = "obj"
        if dist_value is None:
            dist_value = rec.get("distance_to_query_entry_m")
            dist_label = "pose"
        dist_text = f" | {dist_label} {float(dist_value):.2f}m" if dist_value is not None else ""
        _draw_label_box(
            tile,
            f"#{int(rec['rank'])} {float(rec['object_score']):.4f}{dist_text}",
            (12, 34),
            (255, 255, 255),
            (0, 0, 0),
        )
        tiles.append(tile)
    if not tiles:
        return None

    cols = min(2, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    gap = 16
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.full((rows * tile_h + (rows + 1) * gap, cols * tile_w + (cols + 1) * gap, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y = gap + row * (tile_h + gap)
        x = gap + col * (tile_w + gap)
        _paste_panel(canvas, tile, x, y)
        cv2.rectangle(canvas, (x, y), (x + tile_w, y + tile_h), (35, 35, 35), 2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save contact sheet: {out}")
    return str(out)


def run_object_query(
    x0: float,
    y0: float,
    theta0: float,
    db_dir: str,
    scene_path: str,
    top_k: int,
    results_dir: str,
    vlm_model: str,
    use_cache: bool,
    object_text_mode: str = "short",
    detector_conf: float = 0.35,
    min_bbox_side_px: int = 32,
    min_bbox_area_ratio: float = 0.005,
    max_bbox_area_ratio: float = 0.60,
    crop_padding_ratio: float = 0.25,
    object_candidate_pool: int = 256,
    selection_seed: Optional[int] = None,
    detector_classes: Optional[str] = None,
    embedder=None,
) -> Dict[str, Any]:
    from spatial_rag.embedder import Embedder
    from spatial_rag.explorer import Explorer

    t_run_start = time.perf_counter()
    object_text_mode = _validate_object_text_mode(object_text_mode)
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    entries, _, _ = load_spatial_db(db_dir, object_text_mode=object_text_mode)
    if not entries:
        raise ValueError("Spatial DB is empty.")
    object_db = load_object_db(db_dir, text_mode=object_text_mode)
    if object_db is None:
        raise FileNotFoundError(f"Missing object DB artifacts under {db_dir}")
    object_meta, _object_emb, _entry_to_indices = object_db
    object_index = load_object_faiss_index(db_dir, text_mode=object_text_mode)
    if object_index is None:
        raise FileNotFoundError(f"Missing object FAISS index under {db_dir}")
    if int(object_index.ntotal) <= 0:
        raise ValueError("Object FAISS index is empty.")

    out_root = Path(results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    topk_dir = out_root / "top_k"
    topk_dir.mkdir(parents=True, exist_ok=True)

    if embedder is None:
        _runtime_log("initializing_embedder")
        embedder = Embedder()
    else:
        _runtime_log("using_shared_embedder")

    detector = Detector(class_names=detector_classes)
    captioner = VLMCaptioner(
        model_name=vlm_model,
        use_cache=use_cache,
        cache_dir=str(out_root / "vlm_cache"),
        object_use_cache=bool(use_cache and OBJECT_USE_CACHE),
        object_cache_dir=str(out_root / "vlm_object_cache"),
    )

    _runtime_log(f"initializing_explorer scene_path={scene_path}")
    explorer = Explorer(scene_path=scene_path)
    try:
        actual_world_pos, used_snap = _set_agent_pose_2d(explorer, x0=x0, y0=y0, theta0=theta0)
        obs = explorer.sim.get_sensor_observations()
        query_rgb = obs["color_sensor"]
        if query_rgb.shape[2] == 4:
            query_rgb = query_rgb[:, :, :3]

        query_image_path = out_root / "query_image.jpg"
        ok = cv2.imwrite(str(query_image_path), cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to save query image: {query_image_path}")

        raw_detections = detector.detect(query_rgb)
        valid_detections = filter_valid_detections(
            raw_detections,
            image_shape=query_rgb.shape,
            detector_conf=detector_conf,
            min_bbox_side_px=min_bbox_side_px,
            min_bbox_area_ratio=min_bbox_area_ratio,
            max_bbox_area_ratio=max_bbox_area_ratio,
        )
        if not valid_detections:
            raise NoValidObjectDetectionsError(
                f"No valid detections at pose ({float(x0):.3f}, {float(y0):.3f}, {float(theta0):.1f})"
            )
        selected_detection = select_detection(valid_detections, selection_seed=selection_seed)
        crop_rgb, crop_bbox_xyxy = crop_with_padding(query_rgb, selected_detection["bbox_xyxy"], padding_ratio=crop_padding_ratio)
        selected_detection["crop_bbox_xyxy"] = [int(v) for v in crop_bbox_xyxy]

        detection_overlay_path = draw_query_detection_overlay(
            query_rgb=query_rgb,
            detections=valid_detections,
            selected_detection=selected_detection,
            output_path=str(out_root / "query_detection_overlay.jpg"),
        )

        query_crop_path = out_root / "query_object_crop.jpg"
        ok = cv2.imwrite(str(query_crop_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to save query crop: {query_crop_path}")

        object_description = captioner.describe_object_crop_with_meta(
            str(query_crop_path),
            yolo_label=str(selected_detection["label"]),
            yolo_confidence=float(selected_detection["confidence"]),
        )
        query_text, query_text_long, query_text_source = _resolve_query_text(
            object_description,
            yolo_label=str(selected_detection["label"]),
        )
        query_text_for_retrieval = _select_query_text_for_mode(
            query_text_short=query_text,
            query_text_long=query_text_long,
            object_text_mode=object_text_mode,
        )
        query_text_emb = embedder.embed_text(query_text_for_retrieval).astype("float32").reshape(1, -1)
        k = max(1, min(int(object_candidate_pool), int(object_index.ntotal)))
        search_scores, search_indices = object_index.search(query_text_emb.astype("float32"), k)
        entry_scores = _aggregate_entry_scores(
            search_scores=search_scores,
            search_indices=search_indices,
            object_meta=object_meta,
            entry_ids=[int(entry["id"]) for entry in entries],
        )
        best_object_by_entry = _best_object_match_per_entry(
            search_scores=search_scores,
            search_indices=search_indices,
            object_meta=object_meta,
            entry_ids=[int(entry["id"]) for entry in entries],
        )
        top_idx, ranked_pairs = _rank_entries(entries, entry_scores, top_k=top_k)
        top_k_entries = [entries[idx] for idx in top_idx]
        pred_entry = top_k_entries[0]
        pred_score = float(entry_scores.get(int(pred_entry["id"]), 0.0))
        x_pred = float(pred_entry["x"])
        y_pred = float(pred_entry["y"])
        theta_pred = float(pred_entry["orientation"])
        pos_error = float(math.sqrt((x_pred - float(x0)) ** 2 + (y_pred - float(y0)) ** 2))
        yaw_error = float(circular_abs_diff_deg(theta_pred, float(theta0)))

        top_k_records: List[Dict[str, Any]] = []
        top_k_image_paths: List[str] = []
        query_world_x = float(actual_world_pos[0])
        query_world_z = float(actual_world_pos[2])
        query_bbox_laterality = _laterality_from_bbox(
            selected_detection["bbox_xyxy"],
            image_width=int(query_rgb.shape[1]),
        )
        query_object_orientation_deg = _bucket_orientation_deg(
            frame_orientation_deg=float(theta0),
            laterality=query_bbox_laterality,
            angle_step_deg=VLM_ANGLE_STEP,
        )
        query_object_distance_m = object_description.get("distance_from_camera_m")
        query_object_xy = _project_object_xy(
            origin_x=query_world_x,
            origin_y=query_world_z,
            orientation_deg=query_object_orientation_deg,
            distance_m=query_object_distance_m,
        )
        for rank, idx in enumerate(top_idx, start=1):
            entry = entries[idx]
            distance_to_query_entry_m = float(
                math.sqrt(
                    (float(entry["x"]) - query_world_x) ** 2
                    + (float(entry["y"]) - query_world_z) ** 2
                )
            )
            matched_object_info = best_object_by_entry.get(int(entry["id"]), {})
            matched_object_meta = matched_object_info.get("object_meta")
            matched_object_xy = None
            distance_to_query_object_m_approx = None
            if isinstance(matched_object_meta, dict):
                matched_object_xy = _project_object_xy(
                    origin_x=float(matched_object_meta.get("x", entry["x"])),
                    origin_y=float(matched_object_meta.get("y", entry["y"])),
                    orientation_deg=float(
                        matched_object_meta.get(
                            "object_orientation_deg",
                            matched_object_meta.get("orientation", entry["orientation"]),
                        )
                    ),
                    distance_m=matched_object_meta.get("distance_from_camera_m"),
                )
                if query_object_xy is not None and matched_object_xy is not None:
                    distance_to_query_object_m_approx = float(
                        math.sqrt(
                            (float(matched_object_xy[0]) - float(query_object_xy[0])) ** 2
                            + (float(matched_object_xy[1]) - float(query_object_xy[1])) ** 2
                        )
                    )
            rec = {
                "rank": rank,
                "id": int(entry["id"]),
                "x": float(entry["x"]),
                "y": float(entry["y"]),
                "orientation": int(entry["orientation"]),
                "file_name": entry["file_name"],
                "object_score": float(entry_scores.get(int(entry["id"]), 0.0)),
                "distance_to_query_entry_m": distance_to_query_entry_m,
                "distance_to_query_object_m_approx": distance_to_query_object_m_approx,
            }
            if query_object_xy is not None:
                rec["query_object_xy_approx"] = [float(query_object_xy[0]), float(query_object_xy[1])]
            if matched_object_xy is not None:
                rec["matched_object_xy_approx"] = [float(matched_object_xy[0]), float(matched_object_xy[1])]
            if isinstance(matched_object_meta, dict):
                rec["matched_object"] = {
                    "object_index": int(matched_object_info.get("object_index", -1)),
                    "label": _safe_text(matched_object_meta.get("label")),
                    "description": _safe_text(matched_object_meta.get("description")),
                    "long_form_open_description": _safe_text(
                        matched_object_meta.get("long_form_open_description")
                    ),
                    "laterality": _safe_text(matched_object_meta.get("laterality")),
                    "distance_from_camera_m": matched_object_meta.get("distance_from_camera_m"),
                    "object_orientation_deg": matched_object_meta.get("object_orientation_deg"),
                    "score": float(matched_object_info.get("score", 0.0)),
                }
            src = Path(db_dir) / str(entry["file_name"])
            if src.exists():
                safe_score = f"{rec['object_score']:.4f}".replace(".", "p")
                dst = topk_dir / (
                    f"rank_{rank:02d}_id_{int(entry['id']):06d}_"
                    f"ori_{int(entry['orientation']):03d}_score_{safe_score}.jpg"
                )
                shutil.copy2(src, dst)
                rec["retrieved_image_path"] = str(dst)
                top_k_image_paths.append(str(dst))
            else:
                rec["retrieved_image_path"] = None
            top_k_records.append(rec)

        overlay_path = draw_object_query_topk_overlay(
            explorer=explorer,
            entries=entries,
            top_k_entries=top_k_entries,
            pred_entry=pred_entry,
            x0=float(x0),
            y0=float(y0),
            theta0=float(theta0),
            query_rgb=query_rgb,
            detections=valid_detections,
            selected_detection=selected_detection,
            crop_rgb=crop_rgb,
            query_text=query_text_for_retrieval,
            top1_score=pred_score,
            top_k_records=top_k_records,
            output_path=str(out_root / "retrieval_topk_overlay.jpg"),
            query_world_y=float(actual_world_pos[1]),
            actual_query_world_position=actual_world_pos.tolist(),
        )
        contact_sheet_path = build_top_k_contact_sheet(
            top_k_records=top_k_records,
            output_path=str(out_root / "top_k_contact_sheet.jpg"),
        )

        result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pose": {
                "x0": float(x0),
                "y0": float(y0),
                "theta0": float(theta0),
                "actual_world_position": [float(actual_world_pos[0]), float(actual_world_pos[1]), float(actual_world_pos[2])],
                "used_snap_fallback": bool(used_snap),
            },
            "query": {
                "query_image_path": str(query_image_path),
                "query_object_crop_path": str(query_crop_path),
                "query_object_text": query_text,
                "query_object_text_long": query_text_long,
                "query_text_used_for_retrieval": query_text_for_retrieval,
                "query_object_text_source": query_text_source,
                "yolo_label_hint": _safe_text(object_description.get("yolo_label_hint") or selected_detection["label"]),
                "yolo_confidence_hint": float(
                    object_description.get("yolo_confidence_hint")
                    if object_description.get("yolo_confidence_hint") is not None
                    else selected_detection["confidence"]
                ),
                "query_bbox_laterality": query_bbox_laterality,
                "query_object_orientation_deg_approx": float(query_object_orientation_deg),
                "query_object_distance_from_camera_m": query_object_distance_m,
                "query_object_xy_approx": None
                if query_object_xy is None
                else [float(query_object_xy[0]), float(query_object_xy[1])],
                "object_text_mode": object_text_mode,
                "selection_seed": None if selection_seed is None else int(selection_seed),
            },
            "selected_detection": _serialize_detection(selected_detection),
            "valid_detections": [_serialize_detection(det) for det in valid_detections],
            "prediction": {
                "id": int(pred_entry["id"]),
                "x": x_pred,
                "y": y_pred,
                "orientation": int(theta_pred),
                "file_name": pred_entry["file_name"],
                "object_score": pred_score,
            },
            "metrics": {
                "pos_error": pos_error,
                "yaw_error": yaw_error,
                "num_valid_detections": int(len(valid_detections)),
                "selected_detection_confidence": float(selected_detection["confidence"]),
                "selected_detection_label": str(selected_detection["label"]),
                "selected_detection_bbox_xyxy": [int(v) for v in selected_detection["bbox_xyxy"]],
                "top1_object_score": pred_score,
                "top1_distance_to_query_entry_m": float(top_k_records[0]["distance_to_query_entry_m"]),
                "top1_distance_to_query_object_m_approx": top_k_records[0].get(
                    "distance_to_query_object_m_approx"
                ),
            },
            "top_k": top_k_records,
            "artifacts": {
                "query_detection_overlay": detection_overlay_path,
                "retrieval_topk_overlay": overlay_path,
                "top_k_dir": str(topk_dir),
                "top_k_images": top_k_image_paths,
                "top_k_contact_sheet": contact_sheet_path,
            },
            "debug": {
                "object_candidate_pool": int(object_candidate_pool),
                "ranked_entry_scores_top10": [
                    {"entry_id": int(entry_id), "score": float(score)}
                    for entry_id, score in ranked_pairs[:10]
                ],
                "crop_description": {
                    "label": _safe_text(object_description.get("label")),
                    "short_description": _safe_text(object_description.get("short_description")),
                    "long_description": _safe_text(object_description.get("long_description")),
                    "attributes": list(object_description.get("attributes") or []),
                    "distance_from_camera_m": object_description.get("distance_from_camera_m"),
                    "source": str(object_description.get("source") or "unknown"),
                    "yolo_label_hint": _safe_text(object_description.get("yolo_label_hint")),
                    "yolo_confidence_hint": object_description.get("yolo_confidence_hint"),
                },
            },
        }

        json_path = out_root / f"query_{result['timestamp']}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
        result["artifacts"]["query_json"] = str(json_path)
        _runtime_log(f"run_object_query_done elapsed_sec={time.perf_counter() - t_run_start:.2f}")
        return result
    finally:
        explorer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Object-centric localization query on a spatial DB.")
    parser.add_argument("--x0", type=float, required=True, help="Query 2D x")
    parser.add_argument("--y0", type=float, required=True, help="Query 2D y (Habitat z)")
    parser.add_argument("--theta0", type=float, required=True, help="Query orientation in degrees")
    parser.add_argument("--db_dir", type=str, default="spatial_db", help="Spatial DB directory")
    parser.add_argument("--scene_path", type=str, default=SCENE_PATH, help="Habitat scene .glb")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval results (max 5)")
    parser.add_argument("--results_dir", type=str, default="object_vpr_results/query", help="Output directory")
    parser.add_argument("--vlm_model", type=str, default="gpt-5-mini", help="OpenAI VLM model")
    parser.add_argument("--use_cache", type=_str_to_bool, default=True, help="Use VLM cache")
    parser.add_argument("--object_text_mode", type=str, default=OBJECT_TEXT_MODE, choices=["short", "long"])
    parser.add_argument("--detector_conf", type=float, default=0.35, help="Minimum detector confidence")
    parser.add_argument("--min_bbox_side_px", type=int, default=32, help="Minimum bbox side length in pixels")
    parser.add_argument("--min_bbox_area_ratio", type=float, default=0.005, help="Minimum bbox area ratio")
    parser.add_argument("--max_bbox_area_ratio", type=float, default=0.60, help="Maximum bbox area ratio")
    parser.add_argument("--crop_padding_ratio", type=float, default=0.25, help="Per-side crop padding ratio")
    parser.add_argument("--object_candidate_pool", type=int, default=256, help="FAISS object search pool size")
    parser.add_argument("--selection_seed", type=int, default=None, help="Seed for choosing one valid detection")
    parser.add_argument(
        "--detector_classes",
        type=str,
        default=None,
        help="Comma-separated open-vocabulary classes for YOLO_WORLD",
    )
    args = parser.parse_args()

    result = run_object_query(
        x0=args.x0,
        y0=args.y0,
        theta0=args.theta0,
        db_dir=args.db_dir,
        scene_path=args.scene_path,
        top_k=args.top_k,
        results_dir=args.results_dir,
        vlm_model=args.vlm_model,
        use_cache=args.use_cache,
        object_text_mode=args.object_text_mode,
        detector_conf=args.detector_conf,
        min_bbox_side_px=args.min_bbox_side_px,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        max_bbox_area_ratio=args.max_bbox_area_ratio,
        crop_padding_ratio=args.crop_padding_ratio,
        object_candidate_pool=args.object_candidate_pool,
        selection_seed=args.selection_seed,
        detector_classes=args.detector_classes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
