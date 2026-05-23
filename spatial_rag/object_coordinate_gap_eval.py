from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_rag.config import FOV, IMAGE_HEIGHT, IMAGE_WIDTH, SCENE_PATH, SENSOR_HEIGHT
from spatial_rag.depth_stats import mask_depth_stats


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


class HabitatDepthFrameProvider:
    def __init__(
        self,
        *,
        scene_path: str,
        scene_dataset_config_file: Optional[str] = None,
    ):
        try:
            import habitat_sim  # type: ignore
            import quaternion  # type: ignore
        except ModuleNotFoundError as exc:
            if exc.name == "habitat_sim":
                raise RuntimeError(
                    "Habitat depth mode requires habitat_sim in the active Python environment."
                ) from exc
            raise RuntimeError(
                "Habitat depth mode requires the quaternion package in the active Python environment."
            ) from exc

        self._habitat_sim = habitat_sim
        self._quaternion = quaternion

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_path)
        if scene_dataset_config_file:
            sim_cfg.scene_dataset_config_file = str(scene_dataset_config_file)

        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                sim_cfg.gpu_device_id = 0
        except Exception:
            pass

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [int(IMAGE_HEIGHT), int(IMAGE_WIDTH)]
        depth_sensor_spec.position = [0.0, float(SENSOR_HEIGHT), 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        depth_sensor_spec.hfov = float(FOV)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [depth_sensor_spec]

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self.agent = self.sim.initialize_agent(0)

    def __call__(self, entry_id: int, meta_row: Mapping[str, Any]) -> np.ndarray:
        del entry_id
        state = self.agent.get_state()
        state.position = np.asarray(_position_from_meta(meta_row), dtype=np.float32)
        yaw = np.deg2rad(float(_orientation_from_meta(meta_row)))
        state.rotation = self._quaternion.from_rotation_vector([0.0, yaw, 0.0])
        self.agent.set_state(state)
        obs = self.sim.get_sensor_observations()
        depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        return depth.copy()


def _read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise ValueError(f"Not a PNG file: {path}")

    offset = len(signature)
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    interlace_method = 0
    palette: Optional[np.ndarray] = None
    idat_parts: List[bytes] = []

    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace_method = struct.unpack(">IIBBBBB", chunk_data
            )
        elif chunk_type == b"PLTE":
            palette = np.frombuffer(chunk_data, dtype=np.uint8).reshape(-1, 3)
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth {bit_depth} in {path}")
    if interlace_method != 0:
        raise ValueError(f"Unsupported interlaced PNG in {path}")

    channels_by_color_type = {
        0: 1,
        2: 3,
        3: 1,
        4: 2,
        6: 4,
    }
    channels = channels_by_color_type.get(color_type)
    if channels is None:
        raise ValueError(f"Unsupported PNG color type {color_type} in {path}")

    compressed = b"".join(idat_parts)
    raw = zlib.decompress(compressed)
    stride = width * channels
    bpp = max(channels, 1)
    expected_len = height * (stride + 1)
    if len(raw) != expected_len:
        raise ValueError(
            f"Unexpected PNG payload length in {path}: got {len(raw)}, expected {expected_len}"
        )

    scanlines = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride + 1)
    recon = np.empty((height, stride), dtype=np.uint8)

    def _paeth_predictor(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        p = a.astype(np.int32) + b.astype(np.int32) - c.astype(np.int32)
        pa = np.abs(p - a.astype(np.int32))
        pb = np.abs(p - b.astype(np.int32))
        pc = np.abs(p - c.astype(np.int32))
        return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c)).astype(np.uint8)

    for row_index in range(height):
        filter_type = int(scanlines[row_index, 0])
        filtered = scanlines[row_index, 1:].copy()
        left = np.zeros_like(filtered)
        left[bpp:] = filtered[:-bpp]
        up = recon[row_index - 1] if row_index > 0 else np.zeros_like(filtered)
        up_left = np.zeros_like(filtered)
        if row_index > 0:
            up_left[bpp:] = recon[row_index - 1, :-bpp]

        if filter_type == 0:
            recon[row_index] = filtered
        elif filter_type == 1:
            recon[row_index] = (filtered + left) & 0xFF
        elif filter_type == 2:
            recon[row_index] = (filtered + up) & 0xFF
        elif filter_type == 3:
            recon[row_index] = (filtered + ((left.astype(np.uint16) + up.astype(np.uint16)) // 2)).astype(np.uint8)
        elif filter_type == 4:
            recon[row_index] = (filtered + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter type {filter_type} in {path}")

    if color_type == 0:
        return recon.reshape(height, width)
    if color_type == 2:
        return recon.reshape(height, width, 3)
    if color_type == 3:
        if palette is None:
            raise ValueError(f"Indexed PNG missing palette in {path}")
        indices = recon.reshape(height, width)
        return palette[indices]
    if color_type == 4:
        return recon.reshape(height, width, 2)
    return recon.reshape(height, width, 4)


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = _read_png(mask_path)
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 2:
        return np.asarray(mask_arr > 0, dtype=bool)
    if mask_arr.shape[-1] == 2:
        return np.asarray(np.logical_or(mask_arr[..., 0] > 0, mask_arr[..., 1] > 0), dtype=bool)
    return np.asarray(np.any(mask_arr > 0, axis=-1), dtype=bool)


def _normalize_depth_frame(depth_frame: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_frame, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth frame, got shape={depth.shape}")
    return depth


def _vertical_fov_deg(horizontal_fov_deg: float, width_px: int, height_px: int) -> float:
    hfov_rad = math.radians(float(horizontal_fov_deg))
    return math.degrees(
        2.0 * math.atan(math.tan(hfov_rad / 2.0) * float(height_px) / max(float(width_px), 1.0))
    )


def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    mask_arr = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask_arr)
    if xs.size == 0 or ys.size == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _pixel_center_to_relative_angles_deg(
    x_px: float,
    y_px: float,
    *,
    width_px: int,
    height_px: int,
    horizontal_fov_deg: float,
) -> Tuple[float, float]:
    width_f = max(float(width_px), 1.0)
    height_f = max(float(height_px), 1.0)
    cx = (width_f - 1.0) / 2.0
    cy = (height_f - 1.0) / 2.0
    fx = width_f / (2.0 * math.tan(math.radians(float(horizontal_fov_deg)) / 2.0))
    vfov_deg = _vertical_fov_deg(
        horizontal_fov_deg=float(horizontal_fov_deg),
        width_px=width_px,
        height_px=height_px,
    )
    fy = height_f / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    horizontal_angle = math.degrees(math.atan((float(x_px) - cx) / max(fx, 1e-6)))
    vertical_angle = math.degrees(math.atan((cy - float(y_px)) / max(fy, 1e-6)))
    return float(horizontal_angle), float(vertical_angle)


def _planar_distance_from_forward_depth_m(
    forward_depth_m: Optional[float],
    relative_bearing_deg: Optional[float],
) -> Optional[float]:
    depth = _safe_float(forward_depth_m)
    bearing = _safe_float(relative_bearing_deg)
    if depth is None or bearing is None or depth <= 0.0:
        return None
    cos_h = math.cos(math.radians(float(bearing)))
    if abs(cos_h) < 1e-6:
        return None
    return float(depth / cos_h)


def _relative_height_from_forward_depth_m(
    forward_depth_m: Optional[float],
    vertical_angle_deg: Optional[float],
) -> Optional[float]:
    depth = _safe_float(forward_depth_m)
    angle = _safe_float(vertical_angle_deg)
    if depth is None or angle is None:
        return None
    return float(depth * math.tan(math.radians(float(angle))))


def _project_global_xyz_from_geometry(
    *,
    camera_x: float,
    camera_y: float,
    camera_z: float,
    camera_orientation_deg: float,
    distance_m: Optional[float],
    relative_bearing_deg: Optional[float],
    relative_height_from_camera_m: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if distance_m is None or relative_bearing_deg is None:
        return None, None, None
    dist = _safe_float(distance_m)
    bearing = _safe_float(relative_bearing_deg)
    if dist is None or bearing is None or dist < 0.0:
        return None, None, None
    global_bearing = (float(camera_orientation_deg) - float(bearing)) % 360.0
    yaw = math.radians(global_bearing)
    projected_x = float(camera_x - math.sin(yaw) * dist)
    projected_y = float(camera_y - math.cos(yaw) * dist)
    projected_z = None
    rel_h = _safe_float(relative_height_from_camera_m)
    if rel_h is not None:
        projected_z = float(camera_z + rel_h)
    return projected_x, projected_y, projected_z


def _canonical_view_id(entry_id: int) -> str:
    return f"view_{int(entry_id):05d}"


def _xyz_payload(x: Optional[float], y: Optional[float], z: Optional[float]) -> Dict[str, Optional[float]]:
    return {"x": _safe_float(x), "y": _safe_float(y), "z": _safe_float(z)}


def _delta_payload(
    left: Tuple[Optional[float], Optional[float], Optional[float]],
    right: Tuple[Optional[float], Optional[float], Optional[float]],
) -> Tuple[Dict[str, Optional[float]], Optional[float]]:
    left_vals = [_safe_float(value) for value in left]
    right_vals = [_safe_float(value) for value in right]
    if any(value is None for value in left_vals + right_vals):
        return _xyz_payload(None, None, None), None
    dx = float(right_vals[0] - left_vals[0])
    dy = float(right_vals[1] - left_vals[1])
    dz = float(right_vals[2] - left_vals[2])
    return _xyz_payload(dx, dy, dz), float(math.sqrt(dx * dx + dy * dy + dz * dz))


def _mean_xyz(rows: Sequence[Mapping[str, Any]], field_name: str) -> Dict[str, Optional[float]]:
    keys = ("x", "y", "z")
    out: Dict[str, Optional[float]] = {}
    for key in keys:
        values: List[float] = []
        for row in rows:
            payload = row.get(field_name)
            if not isinstance(payload, Mapping):
                continue
            numeric = _safe_float(payload.get(key))
            if numeric is not None:
                values.append(numeric)
        out[key] = float(mean(values)) if values else None
    return out


def _infer_identity_status(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = [str(row.get("label") or "") for row in rows]
    view_types = [str(row.get("view_type") or "") for row in rows]
    room_functions = [str(row.get("room_function") or "") for row in rows]
    orientations = [_safe_int(row.get("orientation")) for row in rows]
    camera_xs = [_safe_float(row.get("x")) for row in rows]
    camera_ys = [_safe_float(row.get("y")) for row in rows]
    texts = [
        " ".join(
            str(row.get(key) or "")
            for key in ("description", "long_form_open_description", "object_text_long")
        ).lower()
        for row in rows
    ]
    table_family = all("table" in label.lower() for label in labels if label)
    same_view_type = len({item for item in view_types if item}) <= 1
    same_room_function = len({item for item in room_functions if item}) <= 1
    same_orientation = len({item for item in orientations if item is not None}) <= 1
    same_camera_xy = False
    if camera_xs and camera_ys and all(item is not None for item in camera_xs + camera_ys):
        same_camera_xy = (
            max(camera_xs) - min(camera_xs) <= 1e-6
            and max(camera_ys) - min(camera_ys) <= 1e-6
        )
    island_mentions = sum(1 for text in texts if "island" in text)
    if table_family and same_view_type and same_room_function and same_orientation and same_camera_xy and island_mentions == len(rows):
        status = "same_scene_island_candidate"
    elif table_family and same_view_type and same_room_function:
        status = "same_scene_table_aggregate"
    else:
        status = "cross_target_aggregate"
    return {
        "status": status,
        "labels": labels,
        "view_types": view_types,
        "room_functions": room_functions,
        "orientations": orientations,
        "same_camera_xy": bool(same_camera_xy),
        "island_mentions": int(island_mentions),
    }


def _load_saved_predicted_depth(row: Mapping[str, Any], spatial_db_dir: Path) -> np.ndarray:
    depth_map_path = _resolve_existing_path(spatial_db_dir, row.get("depth_map_path"))
    if depth_map_path is None:
        raise ValueError(f"Missing saved depth map for object_global_id={row.get('object_global_id')}")
    return _normalize_depth_frame(np.load(depth_map_path))


def _compute_sensor_geometry(
    *,
    row: Mapping[str, Any],
    meta_row: Mapping[str, Any],
    spatial_db_dir: Path,
    horizontal_fov_deg: float,
    depth_source: str,
    habitat_provider: Optional[HabitatDepthFrameProvider],
) -> Dict[str, Any]:
    mask_path = _resolve_existing_path(spatial_db_dir, row.get("mask_path"))
    if mask_path is None:
        raise ValueError(f"Missing mask_path for object_global_id={row.get('object_global_id')}")
    mask = _load_mask(mask_path)

    if depth_source == "habitat":
        if habitat_provider is None:
            raise ValueError("Habitat depth provider is required when depth_source=habitat")
        entry_id = _safe_int(row.get("entry_id"))
        if entry_id is None:
            raise ValueError(f"Missing entry_id for object_global_id={row.get('object_global_id')}")
        depth_map_m = _normalize_depth_frame(habitat_provider(int(entry_id), meta_row))
        depth_origin = "habitat_sensor"
        depth_map_path = None
    else:
        depth_map_m = _load_saved_predicted_depth(row, spatial_db_dir)
        depth_origin = "saved_predicted_depth"
        resolved_depth_map = _resolve_existing_path(spatial_db_dir, row.get("depth_map_path"))
        depth_map_path = None if resolved_depth_map is None else str(resolved_depth_map)

    if mask.shape[:2] != depth_map_m.shape[:2]:
        raise ValueError(
            f"Mask/depth shape mismatch for object_global_id={row.get('object_global_id')}: "
            f"mask={mask.shape}, depth={depth_map_m.shape}"
        )

    centroid = _mask_centroid(mask)
    if centroid is None:
        raise ValueError(f"Mask has no positive pixels for object_global_id={row.get('object_global_id')}")
    centroid_x_px, centroid_y_px = centroid
    stats = mask_depth_stats(depth_map_m, mask)
    forward_depth_m = _safe_float(stats.get("trimmed_median_m"))
    if forward_depth_m is None:
        raise ValueError(f"No valid depth pixels for object_global_id={row.get('object_global_id')}")

    relative_bearing_deg, vertical_angle_deg = _pixel_center_to_relative_angles_deg(
        centroid_x_px,
        centroid_y_px,
        width_px=depth_map_m.shape[1],
        height_px=depth_map_m.shape[0],
        horizontal_fov_deg=horizontal_fov_deg,
    )
    planar_distance_m = _planar_distance_from_forward_depth_m(forward_depth_m, relative_bearing_deg)
    relative_height_from_camera_m = _relative_height_from_forward_depth_m(forward_depth_m, vertical_angle_deg)
    if planar_distance_m is None or relative_height_from_camera_m is None:
        raise ValueError(f"Geometry projection failed for object_global_id={row.get('object_global_id')}")

    estimated_global_x, estimated_global_y, estimated_global_z = _project_global_xyz_from_geometry(
        camera_x=float(meta_row.get("x")),
        camera_y=float(meta_row.get("y")),
        camera_z=float(meta_row.get("z")),
        camera_orientation_deg=float(meta_row.get("orientation")),
        distance_m=planar_distance_m,
        relative_bearing_deg=relative_bearing_deg,
        relative_height_from_camera_m=relative_height_from_camera_m,
    )
    return {
        "depth_origin": depth_origin,
        "depth_map_path": depth_map_path,
        "mask_path": str(mask_path),
        "mask_centroid": {"x_px": float(centroid_x_px), "y_px": float(centroid_y_px)},
        "depth_stats": {
            "median_m": _safe_float(stats.get("median_m")),
            "trimmed_median_m": forward_depth_m,
            "p10_m": _safe_float(stats.get("p10_m")),
            "p90_m": _safe_float(stats.get("p90_m")),
            "num_valid_px": _safe_int(stats.get("num_valid_px")),
        },
        "relative_bearing_deg": float(relative_bearing_deg),
        "vertical_angle_deg": float(vertical_angle_deg),
        "planar_distance_m": float(planar_distance_m),
        "relative_height_from_camera_m": float(relative_height_from_camera_m),
        "estimated_global_xyz": _xyz_payload(estimated_global_x, estimated_global_y, estimated_global_z),
    }


def _build_comparison_row(
    *,
    row: Mapping[str, Any],
    meta_row: Mapping[str, Any],
    sensor_geometry: Mapping[str, Any],
) -> Dict[str, Any]:
    stored_xyz = (
        _safe_float(row.get("estimated_global_x")),
        _safe_float(row.get("estimated_global_y")),
        _safe_float(row.get("estimated_global_z")),
    )
    sensor_payload = sensor_geometry.get("estimated_global_xyz")
    if not isinstance(sensor_payload, Mapping):
        raise ValueError("sensor_geometry missing estimated_global_xyz")
    sensor_xyz = (
        _safe_float(sensor_payload.get("x")),
        _safe_float(sensor_payload.get("y")),
        _safe_float(sensor_payload.get("z")),
    )
    delta_xyz, euclidean_error_m = _delta_payload(stored_xyz, sensor_xyz)
    return {
        "object_global_id": _safe_int(row.get("object_global_id")),
        "entry_id": _safe_int(row.get("entry_id")),
        "view_id": _canonical_view_id(int(row.get("entry_id"))),
        "label": row.get("label"),
        "file_name": row.get("file_name"),
        "view_type": row.get("view_type"),
        "room_function": row.get("room_function"),
        "description": row.get("description"),
        "long_form_open_description": row.get("long_form_open_description"),
        "camera_world_position": _xyz_payload(meta_row.get("x"), meta_row.get("y"), meta_row.get("z")),
        "camera_orientation_deg": _safe_float(meta_row.get("orientation")),
        "stored_estimated_global_xyz": _xyz_payload(*stored_xyz),
        "stored_forward_depth_m": _safe_float(row.get("distance_from_camera_m")),
        "stored_projected_planar_distance_m": _safe_float(row.get("projected_planar_distance_m")),
        "stored_relative_bearing_deg": _safe_float(row.get("relative_bearing_deg")),
        "stored_vertical_angle_deg": _safe_float(row.get("vertical_angle_deg")),
        "stored_depth_stat_median_m": _safe_float(row.get("depth_stat_median_m")),
        "sensor_geometry": dict(sensor_geometry),
        "delta_xyz_m": delta_xyz,
        "euclidean_error_m": euclidean_error_m,
        "forward_depth_delta_m": (
            None
            if _safe_float(row.get("distance_from_camera_m")) is None
            else _safe_float(sensor_geometry.get("depth_stats", {}).get("trimmed_median_m"))
            - float(row.get("distance_from_camera_m"))
        ),
        "planar_distance_delta_m": (
            None
            if _safe_float(row.get("projected_planar_distance_m")) is None
            else _safe_float(sensor_geometry.get("planar_distance_m"))
            - float(row.get("projected_planar_distance_m"))
        ),
        "bearing_delta_deg": (
            None
            if _safe_float(row.get("relative_bearing_deg")) is None
            else _safe_float(sensor_geometry.get("relative_bearing_deg"))
            - float(row.get("relative_bearing_deg"))
        ),
        "vertical_angle_delta_deg": (
            None
            if _safe_float(row.get("vertical_angle_deg")) is None
            else _safe_float(sensor_geometry.get("vertical_angle_deg"))
            - float(row.get("vertical_angle_deg"))
        ),
    }


def _build_markdown_report(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = [
        "# Object Coordinate Gap Evaluation",
        "",
        f"Depth source: {summary.get('depth_source')}",
        f"Selected object ids: {', '.join(str(item) for item in summary.get('selected_object_ids', []))}",
        f"Identity status: {summary.get('identity_verification', {}).get('status')}",
        "",
        "## Per-object results",
        "",
        "| object_id | view_id | label | stored_xyz | sensor_xyz | euclidean_error_m |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for row in rows:
        stored_xyz = row.get("stored_estimated_global_xyz", {})
        sensor_xyz = row.get("sensor_geometry", {}).get("estimated_global_xyz", {})
        lines.append(
            "| {object_id} | {view_id} | {label} | ({sx:.4f}, {sy:.4f}, {sz:.4f}) | ({gx:.4f}, {gy:.4f}, {gz:.4f}) | {err:.4f} |".format(
                object_id=row.get("object_global_id"),
                view_id=row.get("view_id"),
                label=row.get("label"),
                sx=float(stored_xyz.get("x") or 0.0),
                sy=float(stored_xyz.get("y") or 0.0),
                sz=float(stored_xyz.get("z") or 0.0),
                gx=float(sensor_xyz.get("x") or 0.0),
                gy=float(sensor_xyz.get("y") or 0.0),
                gz=float(sensor_xyz.get("z") or 0.0),
                err=float(row.get("euclidean_error_m") or 0.0),
            )
        )
    aggregate = summary.get("aggregate", {})
    stored_mean = aggregate.get("stored_mean_xyz", {})
    sensor_mean = aggregate.get("sensor_mean_xyz", {})
    delta_mean = aggregate.get("delta_mean_xyz", {})
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"Stored mean XYZ: ({float(stored_mean.get('x') or 0.0):.4f}, {float(stored_mean.get('y') or 0.0):.4f}, {float(stored_mean.get('z') or 0.0):.4f})",
            f"Sensor mean XYZ: ({float(sensor_mean.get('x') or 0.0):.4f}, {float(sensor_mean.get('y') or 0.0):.4f}, {float(sensor_mean.get('z') or 0.0):.4f})",
            f"Delta mean XYZ: ({float(delta_mean.get('x') or 0.0):.4f}, {float(delta_mean.get('y') or 0.0):.4f}, {float(delta_mean.get('z') or 0.0):.4f})",
            f"Mean-to-mean Euclidean gap: {float(aggregate.get('euclidean_gap_m') or 0.0):.4f} m",
        ]
    )
    note = summary.get("source_note")
    if note:
        lines.extend(["", "## Note", "", str(note)])
    return "\n".join(lines) + "\n"


def evaluate_coordinate_gap(
    *,
    spatial_db_dir: str | Path,
    output_dir: str | Path,
    object_ids: Sequence[int],
    depth_source: str,
    horizontal_fov_deg: float,
    scene_path: Optional[str] = None,
    scene_dataset_config_file: Optional[str] = None,
) -> Dict[str, Any]:
    spatial_db = Path(spatial_db_dir)
    output_root = Path(output_dir)
    object_rows = _read_jsonl(spatial_db / "object_meta.jsonl")
    meta_rows = _read_jsonl(spatial_db / "meta.jsonl")
    meta_by_entry = _meta_by_entry_id(meta_rows)
    row_by_object_id = {
        int(row["object_global_id"]): row
        for row in object_rows
        if _safe_int(row.get("object_global_id")) is not None
    }

    target_rows: List[Dict[str, Any]] = []
    missing_ids: List[int] = []
    for object_id in object_ids:
        row = row_by_object_id.get(int(object_id))
        if row is None:
            missing_ids.append(int(object_id))
            continue
        target_rows.append(dict(row))
    if missing_ids:
        raise ValueError(f"Missing object ids in object_meta.jsonl: {missing_ids}")

    habitat_provider: Optional[HabitatDepthFrameProvider] = None
    source_note = None
    if depth_source == "habitat":
        if not scene_path:
            raise ValueError("scene_path is required when depth_source=habitat")
        habitat_provider = HabitatDepthFrameProvider(
            scene_path=str(scene_path),
            scene_dataset_config_file=scene_dataset_config_file,
        )
        source_note = "Mean-to-mean gap compares stored Depth Pro-derived coordinates against Habitat depth sensor reprojections."
    else:
        source_note = "saved-predicted mode is for debugging only: it compares stored coordinates against reprojections from the saved predicted depth_map.npy, not Habitat sensor ground truth."

    evaluated_rows: List[Dict[str, Any]] = []
    for row in target_rows:
        entry_id = _safe_int(row.get("entry_id"))
        if entry_id is None:
            raise ValueError(f"Missing entry_id for object_global_id={row.get('object_global_id')}")
        meta_row = meta_by_entry.get(int(entry_id))
        if meta_row is None:
            raise ValueError(f"Missing meta row for entry_id={entry_id}")
        sensor_geometry = _compute_sensor_geometry(
            row=row,
            meta_row=meta_row,
            spatial_db_dir=spatial_db,
            horizontal_fov_deg=horizontal_fov_deg,
            depth_source=depth_source,
            habitat_provider=habitat_provider,
        )
        evaluated_rows.append(
            _build_comparison_row(
                row=row,
                meta_row=meta_row,
                sensor_geometry=sensor_geometry,
            )
        )

    identity_verification = _infer_identity_status(target_rows)
    stored_mean_xyz = _mean_xyz(evaluated_rows, "stored_estimated_global_xyz")
    sensor_mean_xyz = _mean_xyz(
        [{"sensor_mean_proxy": row.get("sensor_geometry", {}).get("estimated_global_xyz", {})} for row in evaluated_rows],
        "sensor_mean_proxy",
    )
    delta_mean_xyz, mean_gap_m = _delta_payload(
        (
            _safe_float(stored_mean_xyz.get("x")),
            _safe_float(stored_mean_xyz.get("y")),
            _safe_float(stored_mean_xyz.get("z")),
        ),
        (
            _safe_float(sensor_mean_xyz.get("x")),
            _safe_float(sensor_mean_xyz.get("y")),
            _safe_float(sensor_mean_xyz.get("z")),
        ),
    )
    per_object_errors = [
        float(row["euclidean_error_m"])
        for row in evaluated_rows
        if _safe_float(row.get("euclidean_error_m")) is not None
    ]
    summary = {
        "selected_object_ids": [int(object_id) for object_id in object_ids],
        "depth_source": depth_source,
        "scene_path": None if scene_path is None else str(scene_path),
        "scene_dataset_config_file": None if scene_dataset_config_file is None else str(scene_dataset_config_file),
        "row_count": int(len(evaluated_rows)),
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "identity_verification": identity_verification,
        "aggregate": {
            "stored_mean_xyz": stored_mean_xyz,
            "sensor_mean_xyz": sensor_mean_xyz,
            "delta_mean_xyz": delta_mean_xyz,
            "euclidean_gap_m": mean_gap_m,
        },
        "per_object_mean_euclidean_error_m": (
            float(mean(per_object_errors)) if per_object_errors else None
        ),
        "per_object_max_euclidean_error_m": (
            float(max(per_object_errors)) if per_object_errors else None
        ),
        "output_dir": str(output_root),
        "source_note": source_note,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "coordinate_gap_rows.jsonl", evaluated_rows)
    _write_json(output_root / "coordinate_gap_summary.json", summary)
    _write_text(output_root / "coordinate_gap_report.md", _build_markdown_report(summary, evaluated_rows))
    return summary


def _default_output_dir_for_ids(object_ids: Sequence[int]) -> Path:
    stem = "_".join(str(int(object_id)) for object_id in object_ids)
    return REPO_ROOT / "coordinate_gap_eval_runs" / f"obj_{stem}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare stored object global coordinates against depth-derived reprojections for selected objects."
    )
    parser.add_argument(
        "object_ids",
        nargs="*",
        type=int,
        help="Object ids to evaluate. Example: 118 243",
    )
    parser.add_argument(
        "--spatial_db_dir",
        type=str,
        default=str(REPO_ROOT / "spatial_db_origin"),
        help="Existing spatial_db directory containing object_meta.jsonl and geometry artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. When omitted, a directory is created automatically from the object ids.",
    )
    parser.add_argument(
        "--object_id",
        dest="object_ids_flag",
        action="append",
        type=int,
        default=None,
        help="Repeatable object_global_id to evaluate. This is kept for backward compatibility.",
    )
    parser.add_argument(
        "--depth_source",
        type=str,
        choices=("habitat", "saved-predicted"),
        default="habitat",
        help="Depth source for reprojection. Use habitat for sensor ground truth; saved-predicted only for debugging.",
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        default=str((REPO_ROOT / SCENE_PATH).resolve()) if not Path(SCENE_PATH).is_absolute() else str(Path(SCENE_PATH)),
        help="Habitat scene path used when depth_source=habitat.",
    )
    parser.add_argument(
        "--scene_dataset_config_file",
        type=str,
        default=None,
        help="Optional Habitat dataset config path. Usually auto-resolved from scene_path when omitted.",
    )
    parser.add_argument(
        "--horizontal_fov_deg",
        type=float,
        default=float(FOV),
        help="Horizontal FOV used by the geometry pipeline.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    positional_ids = list(args.object_ids or [])
    flag_ids = list(args.object_ids_flag or [])
    object_ids = positional_ids or flag_ids or [118, 243]
    output_dir = args.output_dir or str(_default_output_dir_for_ids(object_ids))
    summary = evaluate_coordinate_gap(
        spatial_db_dir=args.spatial_db_dir,
        output_dir=output_dir,
        object_ids=object_ids,
        depth_source=args.depth_source,
        horizontal_fov_deg=args.horizontal_fov_deg,
        scene_path=args.scene_path,
        scene_dataset_config_file=args.scene_dataset_config_file,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()