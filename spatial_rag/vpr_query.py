import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from spatial_rag.config import (
    OBJECT_CACHE_DIR,
    OBJECT_MAX_PER_FRAME,
    OBJECT_PARSE_RETRIES,
    OBJECT_TEXT_MODE,
    OBJECT_USE_CACHE,
    SCAN_ANGLES,
    SCENE_PATH,
)
from spatial_rag.object_canonicalizer import (
    UNKNOWN_TEXT_TOKEN,
    collect_object_texts,
    compose_frame_text,
)
from spatial_rag.object_parser import ParseResult, parse_scene_objects
from spatial_rag.vlm_captioner import VLMCaptioner


def _str_to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _validate_object_text_mode(object_text_mode: str) -> str:
    mode = str(object_text_mode or "").strip().lower()
    if mode not in {"short", "long"}:
        raise ValueError(f"Unsupported object_text_mode: {object_text_mode}")
    return mode


def _runtime_log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[VPR][{ts}] {message}", flush=True)


def circular_abs_diff_deg(a: float, b: float) -> float:
    """Return minimal absolute angular difference in degrees, in [0, 180]."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def _load_meta_jsonl(meta_path: Path) -> List[Dict]:
    records: List[Dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_spatial_db(db_dir: str, object_text_mode: str = "short") -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    mode = _validate_object_text_mode(object_text_mode)
    db_root = Path(db_dir)
    meta_path = db_root / "meta.jsonl"
    image_emb_path = db_root / "image_emb.npy"
    text_emb_path = db_root / f"text_emb_{mode}.npy"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not image_emb_path.exists():
        raise FileNotFoundError(f"Missing {image_emb_path}")
    if not text_emb_path.exists():
        raise FileNotFoundError(f"Missing {text_emb_path}")

    entries = _load_meta_jsonl(meta_path)
    image_emb = np.load(image_emb_path).astype("float32")
    text_emb = np.load(text_emb_path).astype("float32")

    n = len(entries)
    if image_emb.ndim != 2 or text_emb.ndim != 2:
        raise ValueError("image_emb/text_emb must be 2D arrays")
    if image_emb.shape[0] != n or text_emb.shape[0] != n:
        raise ValueError(
            f"DB mismatch: meta={n}, image_emb={image_emb.shape[0]}, text_emb={text_emb.shape[0]}"
        )
    if image_emb.shape[1] != text_emb.shape[1]:
        raise ValueError("Embedding dims mismatch between image_emb and text_emb")

    required_fields = {"id", "x", "y", "orientation", "file_name"}
    for idx, e in enumerate(entries):
        missing = required_fields.difference(e.keys())
        if missing:
            raise ValueError(f"Entry {idx} missing fields: {sorted(missing)}")

    return entries, image_emb, text_emb


def _normalize_scan_angles(scan_angles: Sequence[int]) -> Tuple[int, ...]:
    normalized = sorted({int(angle) % 360 for angle in scan_angles})
    return tuple(normalized)


def _infer_scan_angles_from_entries(entries: Sequence[Dict]) -> Tuple[int, ...]:
    return _normalize_scan_angles(
        int(entry["orientation"])
        for entry in entries
        if entry.get("orientation") is not None
    )


def _load_overlay_scan_angles(db_dir: str, entries: Sequence[Dict]) -> Tuple[int, ...]:
    report_path = Path(db_dir) / "build_report.json"
    if report_path.exists():
        try:
            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
            top_level_angles = report.get("scan_angles")
            if isinstance(top_level_angles, list) and top_level_angles:
                normalized = _normalize_scan_angles(top_level_angles)
                if normalized:
                    return normalized
            random_cfg = report.get("random_config")
            if isinstance(random_cfg, dict):
                angles = random_cfg.get("scan_angles")
                if isinstance(angles, list) and angles:
                    normalized = _normalize_scan_angles(angles)
                    if normalized:
                        return normalized
        except Exception:
            pass

    inferred = _infer_scan_angles_from_entries(entries)
    if inferred:
        return inferred
    return _normalize_scan_angles(SCAN_ANGLES)


def _entry_world_position(entry: Dict, default_world_y: float) -> Tuple[float, float, float]:
    if "world_position" in entry and isinstance(entry["world_position"], (list, tuple)) and len(entry["world_position"]) == 3:
        return (
            float(entry["world_position"][0]),
            float(entry["world_position"][1]),
            float(entry["world_position"][2]),
        )
    return float(entry["x"]), float(default_world_y), float(entry["y"])


def compute_similarities(
    query_image_emb: np.ndarray,
    query_text_emb: np.ndarray,
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    w_img: float = 0.5,
    w_txt: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_img = _l2_normalize(query_image_emb.reshape(1, -1)).reshape(-1)
    q_txt = _l2_normalize(query_text_emb.reshape(1, -1)).reshape(-1)
    db_img = _l2_normalize(image_emb, axis=1)
    db_txt = _l2_normalize(text_emb, axis=1)

    sim_img = db_img @ q_img
    sim_txt = db_txt @ q_txt
    sim_fused = (w_img * sim_img) + (w_txt * sim_txt)
    return sim_img.astype("float32"), sim_txt.astype("float32"), sim_fused.astype("float32")


def _score_to_color(score_norm: float) -> Tuple[int, int, int]:
    """Map normalized score in [0,1] to BGR color."""
    score_norm = float(np.clip(score_norm, 0.0, 1.0))
    b = int(255.0 * (1.0 - score_norm))
    g = int(80.0 + 150.0 * score_norm)
    r = int(255.0 * score_norm)
    return (b, g, r)


def _prepare_overlay_base(explorer, floor_height: Optional[float] = None):
    try:
        base = explorer.render_true_floor_plan(floor_height=floor_height)
        proj = ("floor", explorer._last_top_down_projection)
        return base, proj
    except Exception:
        base, cam, prj = explorer._capture_center_highest_view(hfov=120.0)
        proj = ("center", {"camera_matrix": cam, "projection_matrix": prj})
        return base, proj


def _world_to_pixel_floor(
    x: float,
    z: float,
    width: int,
    height: int,
    projection: Dict,
) -> Tuple[int, int]:
    min_x = float(projection["view_min_x"])
    max_x = float(projection["view_max_x"])
    min_z = float(projection["view_min_z"])
    max_z = float(projection["view_max_z"])

    denom_x = max(max_x - min_x, 1e-6)
    denom_z = max(max_z - min_z, 1e-6)
    px = int(np.clip((x - min_x) / denom_x * (width - 1), 0, width - 1))
    py = int(np.clip((z - min_z) / denom_z * (height - 1), 0, height - 1))
    return px, py


def _world_to_pixel_center(
    x: float,
    y: float,
    z: float,
    width: int,
    height: int,
    projection: Dict,
) -> Optional[Tuple[int, int]]:
    import magnum as mn

    cam_m = projection["camera_matrix"]
    prj_m = projection["projection_matrix"]
    cam = cam_m.transform_point(mn.Vector3(float(x), float(y), float(z)))
    if cam[2] >= 0:
        return None

    ndc = prj_m.transform_point(cam)
    px = int(round((float(ndc[0]) * 0.5 + 0.5) * (width - 1)))
    py = int(round((1.0 - (float(ndc[1]) * 0.5 + 0.5)) * (height - 1)))
    if 0 <= px < width and 0 <= py < height:
        return px, py
    return None


def _group_entries_for_overlay(
    entries: List[Dict],
    sim_fused: np.ndarray,
    default_world_y: float,
) -> List[Dict]:
    groups: Dict[Tuple[float, float], Dict] = {}
    for i, entry in enumerate(entries):
        wx, wy, wz = _entry_world_position(entry, default_world_y=default_world_y)
        key = (round(wx, 3), round(wz, 3))
        grp = groups.get(key)
        if grp is None:
            grp = {
                "world_x": wx,
                "world_y": wy,
                "world_z": wz,
                "x": float(entry["x"]),
                "y": float(entry["y"]),
                "orientation_scores": {},
                "max_sim": -1e9,
                "best_orientation": None,
            }
            groups[key] = grp

        ori = int(entry["orientation"])
        score = float(sim_fused[i])
        prev = grp["orientation_scores"].get(ori)
        if prev is None or score > prev:
            grp["orientation_scores"][ori] = score
        if score > grp["max_sim"]:
            grp["max_sim"] = score
            grp["best_orientation"] = ori

    return list(groups.values())


def draw_query_overlay(
    explorer,
    entries: List[Dict],
    sim_fused: np.ndarray,
    scan_angles: Sequence[int],
    x0: float,
    y0: float,
    theta0: float,
    top_k_entries: List[Dict],
    pred_entry: Dict,
    output_path: str,
    query_world_y: float,
    actual_query_world_position: Optional[Sequence[float]] = None,
) -> str:
    # Use the query's actual navigable floor height to avoid empty top-level slices.
    base, proj = _prepare_overlay_base(explorer, floor_height=query_world_y)
    canvas = base.copy()
    h, w = canvas.shape[:2]

    all_scores = np.asarray(sim_fused, dtype=np.float32)
    s_min = float(np.min(all_scores)) if all_scores.size > 0 else 0.0
    s_max = float(np.max(all_scores)) if all_scores.size > 0 else 1.0
    denom = max(s_max - s_min, 1e-6)

    def normalize_score(s: float) -> float:
        return float(np.clip((s - s_min) / denom, 0.0, 1.0))

    def world_to_pixel(wx: float, wy: float, wz: float) -> Optional[Tuple[int, int]]:
        mode, pinfo = proj
        if mode == "floor":
            return _world_to_pixel_floor(wx, wz, w, h, pinfo)
        return _world_to_pixel_center(wx, wy, wz, w, h, pinfo)

    def orientation_arrow_tip(
        wx: float,
        wy: float,
        wz: float,
        yaw_deg: float,
        world_len: float = 0.9,
    ) -> Optional[Tuple[int, int]]:
        yaw = np.deg2rad(float(yaw_deg))
        # Use -Z as forward to align overlay arrows with Habitat camera heading.
        tip_wx = float(wx - np.sin(yaw) * world_len)
        tip_wz = float(wz - np.cos(yaw) * world_len)
        return world_to_pixel(tip_wx, wy, tip_wz)

    def _orientation_to_sector_center(ori_deg: float) -> float:
        # DB orientation uses Habitat yaw: 0->up, 90->left, 180->down, 270->right.
        # OpenCV ellipse angles use image coordinates: 0->right, 90->down.
        return float((270.0 - float(ori_deg)) % 360.0)

    def _build_sector_bounds() -> Dict[int, Tuple[float, float]]:
        orientations = _normalize_scan_angles(scan_angles)
        if not orientations:
            orientations = _normalize_scan_angles(SCAN_ANGLES)
        if len(orientations) == 1:
            only = int(orientations[0])
            return {only: (0.0, 360.0)}

        bounds: Dict[int, Tuple[float, float]] = {}
        for idx, angle in enumerate(orientations):
            prev_angle = orientations[idx - 1]
            next_angle = orientations[(idx + 1) % len(orientations)]
            prev_gap = (angle - prev_angle) % 360
            next_gap = (next_angle - angle) % 360
            start = float((angle - (prev_gap / 2.0)) % 360)
            end = float((angle + (next_gap / 2.0)) % 360)
            bounds[int(angle)] = (start, end)
        return bounds

    sector_bounds = _build_sector_bounds()

    def _draw_orientation_sector(
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        orientation_deg: int,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        bounds = sector_bounds.get(int(orientation_deg) % 360)
        if bounds is None:
            sector_span = 360.0 / float(max(len(sector_bounds), 1))
            start_ori = float((orientation_deg - sector_span / 2.0) % 360)
            end_ori = float((orientation_deg + sector_span / 2.0) % 360)
        else:
            start_ori, end_ori = bounds

        if abs((end_ori - start_ori) % 360) < 1e-6:
            cv2.circle(img, center, radius, color, thickness)
            return

        start = _orientation_to_sector_center(end_ori)
        end = _orientation_to_sector_center(start_ori)
        if start <= end:
            cv2.ellipse(img, center, (radius, radius), 0.0, start, end, color, thickness)
        else:
            cv2.ellipse(img, center, (radius, radius), 0.0, start, 360.0, color, thickness)
            cv2.ellipse(img, center, (radius, radius), 0.0, 0.0, end, color, thickness)

    groups = _group_entries_for_overlay(entries, sim_fused, default_world_y=query_world_y)

    # Draw orientation sectors for every grouped point.
    for grp in groups:
        p = world_to_pixel(grp["world_x"], grp["world_y"], grp["world_z"])
        if p is None:
            continue
        px, py = p
        # Emphasize high-score (yellow-ish) points with a larger radius so
        # score differences are easier to read at a glance.
        score_n = normalize_score(grp["max_sim"])
        r = int(round(8 + 10 * score_n + 16 * (score_n ** 2)))
        r = int(np.clip(r, 8, 34))

        for ori in sector_bounds:
            s = grp["orientation_scores"].get(ori)
            color = (90, 90, 90) if s is None else _score_to_color(normalize_score(s))
            _draw_orientation_sector(canvas, (px, py), r, ori, color, -1)

        cv2.circle(canvas, (px, py), r, (25, 25, 25), 2)
        bo = grp.get("best_orientation")
        if bo is not None and (int(bo) % 360) in sector_bounds:
            _draw_orientation_sector(canvas, (px, py), r + 2, int(bo), (255, 255, 255), 3)

    # Draw all retrieved Top-K local orientation arrows and rank labels.
    query_world_x = float(actual_query_world_position[0]) if actual_query_world_position is not None else float(x0)
    query_world_z = float(actual_query_world_position[2]) if actual_query_world_position is not None else float(y0)
    p_gt = world_to_pixel(query_world_x, float(query_world_y), query_world_z)
    if top_k_entries:
        for rank, entry in enumerate(top_k_entries, start=1):
            rwx, rwy, rwz = _entry_world_position(entry, default_world_y=query_world_y)
            p_rank = world_to_pixel(rwx, rwy, rwz)
            if p_rank is None:
                continue
            rank_ori = float(entry["orientation"])
            rank_tip = orientation_arrow_tip(rwx, rwy, rwz, rank_ori, world_len=0.95)
            if rank_tip is not None:
                cv2.arrowedLine(
                    canvas,
                    p_rank,
                    rank_tip,
                    (0, 0, 255),
                    2,
                    tipLength=0.35,
                )
            cv2.circle(canvas, p_rank, 5, (0, 0, 255), -1)
            label = f"Top{rank}"
            tx = int(np.clip(p_rank[0] + 5, 0, w - 1))
            ty = int(np.clip(p_rank[1] - 5 - 12 * ((rank - 1) % 3), 0, h - 1))
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 4)

    # Draw GT point (large green circle + orientation arrow).
    if p_gt is not None:
        cv2.circle(canvas, p_gt, 16, (0, 255, 0), 3)
        cv2.circle(canvas, p_gt, 4, (0, 255, 0), -1)
        gt_tip = orientation_arrow_tip(query_world_x, float(query_world_y), query_world_z, float(theta0), world_len=1.25)
        if gt_tip is not None:
            cv2.arrowedLine(canvas, p_gt, gt_tip, (0, 255, 0), 3, tipLength=0.35)
        cv2.putText(canvas, "GT", (p_gt[0] + 8, p_gt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

    cv2.putText(
        canvas,
        f"Query theta0={theta0:.1f} deg",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save overlay: {out}")
    return str(out)


def _set_agent_pose_2d(explorer, x0: float, y0: float, theta0: float) -> Tuple[np.ndarray, bool]:
    """Set agent pose from 2D coordinates. Returns actual world position and whether snap fallback was used."""
    import quaternion

    state = explorer.agent.get_state()
    curr_y = float(state.position[1])
    candidate = np.array([float(x0), curr_y, float(y0)], dtype=np.float32)
    used_snap = False

    if not explorer.sim.pathfinder.is_navigable(candidate):
        snapped = np.array(explorer.sim.pathfinder.snap_point(candidate), dtype=np.float32)
        if not np.isfinite(snapped).all():
            raise ValueError(f"Pose is not navigable and cannot be snapped: ({x0}, {y0})")
        candidate = snapped
        used_snap = True

    state.position = candidate
    state.rotation = quaternion.from_rotation_vector([0.0, np.deg2rad(float(theta0)), 0.0])
    explorer.agent.set_state(state)
    return np.array(state.position, dtype=np.float32), used_snap


def _parse_objects_with_retry(
    captioner: VLMCaptioner,
    image_path: str,
    image_id: str,
    max_objects: int,
    retries: int,
) -> ParseResult:
    retries = max(0, int(retries))
    last_result: Optional[ParseResult] = None
    for attempt in range(retries + 1):
        _runtime_log(
            f"object_parse attempt={attempt + 1}/{retries + 1} "
            f"force_refresh={attempt > 0} image_id={image_id}"
        )
        result = captioner.extract_objects_with_meta(
            image_path=image_path,
            max_objects=max_objects,
            force_refresh=attempt > 0,
        )
        raw_output = result.get("raw_json", "")
        parsed = parse_scene_objects(raw_output, image_context={"image_id": image_id})
        parsed.raw_api_response = result.get("raw_api_response")
        parsed.raw_api_source = str(result.get("source") or "")
        _runtime_log(
            f"object_parse result status={parsed.parse_status} "
            f"warnings={len(parsed.warnings)} image_id={image_id}"
        )
        if parsed.parse_status == "ok":
            return parsed
        last_result = parsed

    if last_result is None:
        return ParseResult(
            scene_objects=None,
            parse_status="fallback",
            warnings=["object parsing failed and no parser output was captured"],
            raw_vlm_output="",
            raw_api_response=None,
            raw_api_source="missing",
        )
    return ParseResult(
        scene_objects=None,
        parse_status="fallback",
        warnings=last_result.warnings,
        raw_vlm_output=last_result.raw_vlm_output,
        raw_api_response=last_result.raw_api_response,
        raw_api_source=last_result.raw_api_source,
    )


def run_query(
    x0: float,
    y0: float,
    theta0: float,
    db_dir: str,
    scene_path: str,
    top_k: int,
    w_img: float,
    w_txt: float,
    results_dir: str,
    vlm_model: str,
    use_cache: bool,
    object_parse_retries: int = OBJECT_PARSE_RETRIES,
    object_use_cache: bool = OBJECT_USE_CACHE,
    object_cache_dir: Optional[str] = None,
    object_max_per_frame: int = OBJECT_MAX_PER_FRAME,
    object_text_mode: str = OBJECT_TEXT_MODE,
    embedder=None,
) -> Dict:
    from spatial_rag.embedder import Embedder
    from spatial_rag.explorer import Explorer

    t_run_start = time.perf_counter()
    _runtime_log(
        f"run_query start x={float(x0):.3f} y={float(y0):.3f} "
        f"theta={float(theta0):.1f} object_text_mode={object_text_mode}"
    )

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if w_img < 0 or w_txt < 0:
        raise ValueError("w_img and w_txt must be >= 0")
    if (w_img + w_txt) <= 0:
        raise ValueError("w_img + w_txt must be > 0")
    object_text_mode = _validate_object_text_mode(object_text_mode)

    # Normalize weights.
    w_sum = float(w_img + w_txt)
    w_img = float(w_img / w_sum)
    w_txt = float(w_txt / w_sum)

    entries, image_emb, text_emb = load_spatial_db(db_dir, object_text_mode=object_text_mode)
    overlay_scan_angles = _load_overlay_scan_angles(db_dir, entries)
    _runtime_log(
        f"loaded_spatial_db entries={len(entries)} "
        f"image_emb_shape={tuple(image_emb.shape)} text_emb_shape={tuple(text_emb.shape)}"
    )
    _runtime_log(f"overlay_scan_angles={overlay_scan_angles}")
    if len(entries) == 0:
        raise ValueError("Spatial DB is empty.")

    out_root = Path(results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    topk_dir = out_root / "top_k"
    topk_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_cache_root = (
        Path(object_cache_dir)
        if object_cache_dir is not None
        else out_root / str(OBJECT_CACHE_DIR)
    )
    if object_use_cache:
        object_cache_root.mkdir(parents=True, exist_ok=True)

    _runtime_log(f"initializing_explorer scene_path={scene_path}")
    t_explorer_start = time.perf_counter()
    explorer = Explorer(scene_path=scene_path)
    _runtime_log(f"explorer_ready elapsed_sec={time.perf_counter() - t_explorer_start:.2f}")
    try:
        actual_world_pos, used_snap = _set_agent_pose_2d(explorer, x0=x0, y0=y0, theta0=theta0)
        _runtime_log(
            "agent_pose_set "
            f"actual_world=({float(actual_world_pos[0]):.3f}, {float(actual_world_pos[1]):.3f}, {float(actual_world_pos[2]):.3f}) "
            f"used_snap={bool(used_snap)}"
        )

        obs = explorer.sim.get_sensor_observations()
        query_rgb = obs["color_sensor"]
        if query_rgb.shape[2] == 4:
            query_rgb = query_rgb[:, :, :3]

        query_image_path = out_root / "query_image.jpg"
        ok = cv2.imwrite(str(query_image_path), cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to save query image to {query_image_path}")
        _runtime_log(
            f"query_image_saved path={query_image_path} shape={tuple(query_rgb.shape)}"
        )

        if embedder is None:
            _runtime_log("initializing_embedder (no shared embedder provided)")
            t_embedder_start = time.perf_counter()
            embedder = Embedder()
            _runtime_log(f"embedder_ready elapsed_sec={time.perf_counter() - t_embedder_start:.2f}")
        else:
            _runtime_log("using_shared_embedder")

        _runtime_log("initializing_vlm_captioner")
        t_captioner_start = time.perf_counter()
        captioner = VLMCaptioner(
            model_name=vlm_model,
            use_cache=use_cache,
            cache_dir=str(out_root / "vlm_cache"),
            object_use_cache=object_use_cache,
            object_cache_dir=str(object_cache_root),
        )
        _runtime_log(f"vlm_captioner_ready elapsed_sec={time.perf_counter() - t_captioner_start:.2f}")

        query_parse_status = "fallback"
        query_parse_warnings: List[str] = []
        query_object_lines: List[str] = []
        query_object_lines_short: List[str] = [UNKNOWN_TEXT_TOKEN]
        query_object_lines_long: List[str] = [UNKNOWN_TEXT_TOKEN]
        query_text = UNKNOWN_TEXT_TOKEN
        query_text_short = UNKNOWN_TEXT_TOKEN
        query_text_long = UNKNOWN_TEXT_TOKEN

        _runtime_log(
            f"query_object_parse_start max_objects={int(object_max_per_frame)} "
            f"retries={int(object_parse_retries)}"
        )
        t_parse_start = time.perf_counter()
        parse_result = _parse_objects_with_retry(
            captioner=captioner,
            image_path=str(query_image_path),
            image_id="query_image",
            max_objects=int(object_max_per_frame),
            retries=int(object_parse_retries),
        )
        query_parse_status = parse_result.parse_status
        query_parse_warnings = list(parse_result.warnings)
        if parse_result.scene_objects is not None:
            query_parse_status = "ok"
            query_object_lines_short = collect_object_texts(
                parse_result.scene_objects,
                max_objects=int(object_max_per_frame),
                mode="short",
            )
            query_object_lines_long = collect_object_texts(
                parse_result.scene_objects,
                max_objects=int(object_max_per_frame),
                mode="long",
            )
            query_text_short = compose_frame_text(
                parse_result.scene_objects,
                max_objects=int(object_max_per_frame),
                mode="short",
            )
            query_text_long = compose_frame_text(
                parse_result.scene_objects,
                max_objects=int(object_max_per_frame),
                mode="long",
            )
        else:
            query_parse_status = "fallback"
            query_object_lines_short = [UNKNOWN_TEXT_TOKEN]
            query_object_lines_long = [UNKNOWN_TEXT_TOKEN]
            query_text_short = UNKNOWN_TEXT_TOKEN
            query_text_long = UNKNOWN_TEXT_TOKEN
        if object_text_mode == "long":
            query_object_lines = query_object_lines_long
            query_text = query_text_long
        else:
            query_object_lines = query_object_lines_short
            query_text = query_text_short
        _runtime_log(
            f"query_object_parse_done status={query_parse_status} "
            f"object_lines={len(query_object_lines)} warnings={len(query_parse_warnings)} "
            f"elapsed_sec={time.perf_counter() - t_parse_start:.2f}"
        )

        _runtime_log("embedding_query_image_start")
        t_img_embed_start = time.perf_counter()
        query_image_emb = embedder.embed_image(query_rgb).astype("float32")
        _runtime_log(
            f"embedding_query_image_done dim={query_image_emb.shape[0]} "
            f"elapsed_sec={time.perf_counter() - t_img_embed_start:.2f}"
        )
        _runtime_log("embedding_query_text_start")
        t_txt_embed_start = time.perf_counter()
        query_text_emb = embedder.embed_text(query_text).astype("float32")
        _runtime_log(
            f"embedding_query_text_done dim={query_text_emb.shape[0]} "
            f"elapsed_sec={time.perf_counter() - t_txt_embed_start:.2f}"
        )

        sim_img, sim_txt, sim_fused = compute_similarities(
            query_image_emb=query_image_emb,
            query_text_emb=query_text_emb,
            image_emb=image_emb,
            text_emb=text_emb,
            w_img=w_img,
            w_txt=w_txt,
        )

        _runtime_log(
            f"similarities_ready sim_shape={tuple(sim_fused.shape)} "
            f"score_range=({float(np.min(sim_fused)):.4f},{float(np.max(sim_fused)):.4f})"
        )

        final_scores = sim_fused.copy()
        order = np.argsort(-sim_fused)
        k = min(int(top_k), 5, len(entries))

        # De-duplicate Top-K by 2D position so overlay points do not overlap.
        # Keep the highest-score candidate for each position.
        top_idx_list: List[int] = []
        seen_xy = set()
        for idx in order:
            entry = entries[int(idx)]
            key = (round(float(entry["x"]), 3), round(float(entry["y"]), 3))
            if key in seen_xy:
                continue
            seen_xy.add(key)
            top_idx_list.append(int(idx))
            if len(top_idx_list) >= k:
                break
        top_idx = np.asarray(top_idx_list, dtype=np.int64)
        if top_idx.size == 0:
            top_idx = np.asarray([int(order[0])], dtype=np.int64)

        pred_idx = int(top_idx[0])
        pred_entry = entries[pred_idx]

        x_pred = float(pred_entry["x"])
        y_pred = float(pred_entry["y"])
        theta_pred = float(pred_entry["orientation"])
        pos_error = float(math.sqrt((x_pred - x0) ** 2 + (y_pred - y0) ** 2))
        yaw_error = circular_abs_diff_deg(theta_pred, theta0)
        _runtime_log(
            f"prediction_ready pred_id={int(pred_entry['id'])} "
            f"pos_error={pos_error:.3f} yaw_error={float(yaw_error):.2f}"
        )

        top_k_records = []
        top_k_image_paths: List[str] = []
        for rank, idx in enumerate(top_idx, start=1):
            entry = entries[int(idx)]
            rec = {
                "rank": rank,
                "id": int(entry["id"]),
                "x": float(entry["x"]),
                "y": float(entry["y"]),
                "orientation": int(entry["orientation"]),
                "file_name": entry["file_name"],
                "sim_img": float(sim_img[int(idx)]),
                "sim_txt": float(sim_txt[int(idx)]),
                "sim_fused": float(sim_fused[int(idx)]),
            }

            # Export retrieved image for Top-K inspection.
            src = Path(db_dir) / str(entry["file_name"])
            if src.exists():
                safe_sim = f"{rec['sim_fused']:.4f}".replace(".", "p")
                dst_name = (
                    f"rank_{rank:02d}_id_{int(entry['id']):06d}_"
                    f"ori_{int(entry['orientation']):03d}_sim_{safe_sim}.jpg"
                )
                dst = topk_dir / dst_name
                try:
                    shutil.copy2(src, dst)
                    rec["retrieved_image_path"] = str(dst)
                    top_k_image_paths.append(str(dst))
                except Exception:
                    rec["retrieved_image_path"] = None
            else:
                rec["retrieved_image_path"] = None

            top_k_records.append(rec)

        top_k_entries = [entries[int(idx)] for idx in top_idx]

        overlay_path = out_root / "query_overlay.jpg"
        _runtime_log("overlay_render_start")
        t_overlay_start = time.perf_counter()
        overlay_path_str = draw_query_overlay(
            explorer=explorer,
            entries=entries,
            sim_fused=sim_fused,
            scan_angles=overlay_scan_angles,
            x0=float(x0),
            y0=float(y0),
            theta0=float(theta0),
            top_k_entries=top_k_entries,
            pred_entry=pred_entry,
            output_path=str(overlay_path),
            query_world_y=float(actual_world_pos[1]),
            actual_query_world_position=actual_world_pos.tolist(),
        )
        _runtime_log(
            f"overlay_render_done path={overlay_path_str} "
            f"elapsed_sec={time.perf_counter() - t_overlay_start:.2f}"
        )

        result = {
            "timestamp": ts,
            "query": {
                "x0": float(x0),
                "y0": float(y0),
                "theta0": float(theta0),
                "query_image_path": str(query_image_path),
                "query_text": query_text,
                "query_text_input_for_clip": query_text,
                "query_text_short": query_text_short,
                "query_text_long": query_text_long,
                "query_object_lines": query_object_lines,
                "query_object_text_inputs_for_clip": query_object_lines,
                "query_object_lines_short": query_object_lines_short,
                "query_object_lines_long": query_object_lines_long,
                "object_text_mode": object_text_mode,
                "actual_world_position": [float(actual_world_pos[0]), float(actual_world_pos[1]), float(actual_world_pos[2])],
                "used_snap_fallback": bool(used_snap),
            },
            "weights": {"w_img": float(w_img), "w_txt": float(w_txt)},
            "query_object_count": int(len(query_object_lines)),
            "parse_status": query_parse_status,
            "parse_warnings": query_parse_warnings,
            "prediction": {
                "id": int(pred_entry["id"]),
                "x": x_pred,
                "y": y_pred,
                "orientation": int(theta_pred),
                "file_name": pred_entry["file_name"],
                "score": float(final_scores[pred_idx]),
            },
            "metrics": {"pos_error": pos_error, "yaw_error": float(yaw_error)},
            "top_k": top_k_records,
            "artifacts": {
                "query_overlay": overlay_path_str,
                "top_k_dir": str(topk_dir),
                "top_k_images": top_k_image_paths,
            },
        }

        json_path = out_root / f"query_{ts}.json"
        _runtime_log(f"writing_result_json path={json_path}")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
        result["artifacts"]["query_json"] = str(json_path)
        _runtime_log(f"run_query_done elapsed_sec={time.perf_counter() - t_run_start:.2f}")

        return result
    finally:
        _runtime_log("closing_explorer")
        t_close_start = time.perf_counter()
        explorer.close()
        _runtime_log(f"explorer_closed elapsed_sec={time.perf_counter() - t_close_start:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual Placement Recognition query on spatial DB.")
    parser.add_argument("--x0", type=float, required=True, help="Query 2D x")
    parser.add_argument("--y0", type=float, required=True, help="Query 2D y (Habitat z)")
    parser.add_argument("--theta0", type=float, required=True, help="Query orientation in degrees")
    parser.add_argument("--db_dir", type=str, default="spatial_db", help="Spatial DB directory")
    parser.add_argument("--scene_path", type=str, default=SCENE_PATH, help="Habitat scene .glb")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval results (max 5)")
    parser.add_argument("--w_img", type=float, default=0.5, help="Image similarity weight")
    parser.add_argument("--w_txt", type=float, default=0.5, help="Text similarity weight")
    parser.add_argument("--results_dir", type=str, default="vpr_results", help="Output results directory")
    parser.add_argument("--vlm_model", type=str, default="gpt-5-mini", help="OpenAI VLM model")
    parser.add_argument("--use_cache", type=_str_to_bool, default=True, help="Use VLM cache (true/false)")
    parser.add_argument(
        "--object_parse_retries",
        type=int,
        default=OBJECT_PARSE_RETRIES,
        help="Retries after object JSON parse failure",
    )
    parser.add_argument(
        "--object_use_cache",
        type=_str_to_bool,
        default=OBJECT_USE_CACHE,
        help="Use object VLM cache (true/false)",
    )
    parser.add_argument(
        "--object_cache_dir",
        type=str,
        default=None,
        help="Object cache directory (default: <results_dir>/vlm_object_cache)",
    )
    parser.add_argument(
        "--object_max_per_frame",
        type=int,
        default=OBJECT_MAX_PER_FRAME,
        help="Maximum objects used for query object extraction",
    )
    parser.add_argument(
        "--object_text_mode",
        type=str,
        default=OBJECT_TEXT_MODE,
        choices=["short", "long"],
        help="Text channel to use: short=description concat, long=long_form_open_description concat",
    )
    args = parser.parse_args()

    result = run_query(
        x0=args.x0,
        y0=args.y0,
        theta0=args.theta0,
        db_dir=args.db_dir,
        scene_path=args.scene_path,
        top_k=args.top_k,
        w_img=args.w_img,
        w_txt=args.w_txt,
        results_dir=args.results_dir,
        vlm_model=args.vlm_model,
        use_cache=args.use_cache,
        object_parse_retries=args.object_parse_retries,
        object_use_cache=args.object_use_cache,
        object_cache_dir=args.object_cache_dir,
        object_max_per_frame=args.object_max_per_frame,
        object_text_mode=args.object_text_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
