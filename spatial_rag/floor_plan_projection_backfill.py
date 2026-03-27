import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from spatial_rag.spatial_db_builder import _write_floor_plan_projection


def _load_build_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_scene_path(db_path: Path, scene_path: Optional[str]) -> str:
    if scene_path:
        return str(scene_path)
    report = _load_build_report(db_path / "build_report.json")
    candidate = str(report.get("scene_path") or "").strip()
    if candidate:
        return candidate
    raise ValueError("Unable to resolve scene_path from build_report.json; pass --scene_path explicitly")


def _default_explorer_factory(scene_path: str):
    from spatial_rag.explorer import Explorer

    return Explorer(scene_path=scene_path)


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


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _compute_projection_from_existing_floorplan(db_path: Path) -> Optional[Dict[str, float]]:
    floorplan_path = db_path / "overview" / "textured_floor_plan.jpg"
    meta_path = db_path / "meta.jsonl"
    if not floorplan_path.exists() or not meta_path.exists():
        return None
    image = cv2.imread(str(floorplan_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    meta_rows = _load_jsonl(meta_path)
    xs = [_safe_float(row.get("x")) for row in meta_rows]
    zs = [_safe_float(row.get("y")) for row in meta_rows]
    xs = [v for v in xs if v is not None]
    zs = [v for v in zs if v is not None]
    if not xs or not zs:
        return None

    # Background pixels are the uniform dark gray fill used by render_textured_floor_plan.
    mask = np.any(np.abs(image.astype(np.int16) - 70) > 8, axis=2)
    ys_px, xs_px = np.where(mask)
    if xs_px.size == 0 or ys_px.size == 0:
        return None
    px_min = int(xs_px.min())
    px_max = int(xs_px.max())
    py_min = int(ys_px.min())
    py_max = int(ys_px.max())
    width = max(int(image.shape[1]) - 1, 1)
    height = max(int(image.shape[0]) - 1, 1)

    world_min_x = float(min(xs))
    world_max_x = float(max(xs))
    world_min_z = float(min(zs))
    world_max_z = float(max(zs))
    if world_max_x <= world_min_x or world_max_z <= world_min_z:
        return None

    frac_min_x = float(px_min) / float(width)
    frac_max_x = float(px_max) / float(width)
    frac_min_z = float(py_min) / float(height)
    frac_max_z = float(py_max) / float(height)
    if frac_max_x <= frac_min_x or frac_max_z <= frac_min_z:
        return None

    full_min_x = world_min_x - frac_min_x * (world_max_x - world_min_x) / max(frac_max_x - frac_min_x, 1e-6)
    full_max_x = full_min_x + (world_max_x - world_min_x) / max(frac_max_x - frac_min_x, 1e-6)
    full_min_z = world_min_z - frac_min_z * (world_max_z - world_min_z) / max(frac_max_z - frac_min_z, 1e-6)
    full_max_z = full_min_z + (world_max_z - world_min_z) / max(frac_max_z - frac_min_z, 1e-6)

    return {
        "view_min_x": float(full_min_x),
        "view_max_x": float(full_max_x),
        "view_min_z": float(full_min_z),
        "view_max_z": float(full_max_z),
    }


def _compute_projection_from_scene(
    scene_path: str,
    explorer_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, float]:
    factory = explorer_factory or _default_explorer_factory
    explorer = factory(scene_path)
    try:
        explorer.render_true_floor_plan()
        projection = getattr(explorer, "_last_top_down_projection", None)
    finally:
        explorer.close()
    if not isinstance(projection, dict):
        raise RuntimeError("Failed to compute floor plan projection from scene")
    try:
        return {
            "view_min_x": float(projection["view_min_x"]),
            "view_max_x": float(projection["view_max_x"]),
            "view_min_z": float(projection["view_min_z"]),
            "view_max_z": float(projection["view_max_z"]),
        }
    except Exception as exc:
        raise RuntimeError("Failed to compute floor plan projection from scene") from exc


def backfill_floor_plan_projection(
    db_dir: str,
    scene_path: Optional[str] = None,
    overwrite: bool = False,
    explorer_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    db_path = Path(db_dir)
    overview_dir = db_path / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    projection_path = overview_dir / "floor_plan_projection.json"
    if projection_path.exists() and not overwrite:
        return {
            "db_dir": str(db_path),
            "projection_path": str(projection_path),
            "status": "exists",
        }

    resolved_scene_path = _resolve_scene_path(db_path, scene_path)
    try:
        projection = _compute_projection_from_scene(
            scene_path=resolved_scene_path,
            explorer_factory=explorer_factory,
        )
        projection_source = "scene"
    except Exception:
        projection = _compute_projection_from_existing_floorplan(db_path)
        if projection is None:
            raise
        projection_source = "floorplan_bbox_calibration"
    saved_path = _write_floor_plan_projection(projection_path, projection)
    if saved_path is None:
        raise RuntimeError("Failed to persist floor plan projection")

    report_path = db_path / "build_report.json"
    report = _load_build_report(report_path)
    overview_outputs = dict(report.get("overview_outputs") or {})
    overview_outputs["floor_plan_projection"] = str(projection_path)
    report["overview_outputs"] = overview_outputs
    report["floor_plan_projection_source"] = projection_source
    if "scene_path" not in report or not str(report.get("scene_path") or "").strip():
        report["scene_path"] = resolved_scene_path
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "db_dir": str(db_path),
        "projection_path": str(projection_path),
        "scene_path": resolved_scene_path,
        "projection_source": projection_source,
        "status": "written",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill overview/floor_plan_projection.json for an existing spatial DB.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory")
    parser.add_argument("--scene_path", type=str, default=None, help="Optional explicit scene path override")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite projection file if it already exists")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = backfill_floor_plan_projection(
        db_dir=args.db_dir,
        scene_path=args.scene_path,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
