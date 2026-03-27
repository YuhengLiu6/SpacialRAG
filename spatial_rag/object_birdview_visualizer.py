import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


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


def _load_build_report(db_dir: Path) -> Dict[str, Any]:
    path = db_dir / "build_report.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_saved_projection(db_dir: Path) -> Optional[Dict[str, float]]:
    path = db_dir / "overview" / "floor_plan_projection.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    required = ("view_min_x", "view_max_x", "view_min_z", "view_max_z")
    if not isinstance(payload, dict):
        return None
    try:
        return {key: float(payload[key]) for key in required}
    except Exception:
        return None


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


def _world_to_pixel_floor(
    x: float,
    z: float,
    width: int,
    height: int,
    projection: Dict[str, float],
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


def _projection_from_rows(
    meta_rows: Sequence[Dict[str, Any]],
    object_rows: Sequence[Dict[str, Any]],
    pad_m: float = 0.5,
) -> Dict[str, float]:
    xs: List[float] = []
    zs: List[float] = []
    for row in meta_rows:
        x = _safe_float(row.get("x"))
        z = _safe_float(row.get("y"))
        if x is not None and z is not None:
            xs.append(x)
            zs.append(z)
    for row in object_rows:
        x = _safe_float(row.get("estimated_global_x"))
        z = _safe_float(row.get("estimated_global_z"))
        if x is not None and z is not None:
            xs.append(x)
            zs.append(z)
    if not xs or not zs:
        return {"view_min_x": -1.0, "view_max_x": 1.0, "view_min_z": -1.0, "view_max_z": 1.0}
    min_x = min(xs) - float(pad_m)
    max_x = max(xs) + float(pad_m)
    min_z = min(zs) - float(pad_m)
    max_z = max(zs) + float(pad_m)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_z <= min_z:
        max_z = min_z + 1.0
    return {
        "view_min_x": float(min_x),
        "view_max_x": float(max_x),
        "view_min_z": float(min_z),
        "view_max_z": float(max_z),
    }


def _try_scene_projection(db_dir: Path) -> Optional[Dict[str, float]]:
    report = _load_build_report(db_dir)
    scene_path = str(report.get("scene_path") or "").strip()
    if not scene_path:
        return None
    try:
        from spatial_rag.explorer import Explorer

        explorer = Explorer(scene_path=scene_path)
        try:
            explorer.render_true_floor_plan()
            projection = getattr(explorer, "_last_top_down_projection", None)
            if isinstance(projection, dict) and "view_min_x" in projection:
                return {
                    "view_min_x": float(projection["view_min_x"]),
                    "view_max_x": float(projection["view_max_x"]),
                    "view_min_z": float(projection["view_min_z"]),
                    "view_max_z": float(projection["view_max_z"]),
                }
        finally:
            explorer.close()
    except Exception:
        return None
    return None


def _default_floorplan_path(db_dir: Path) -> Optional[Path]:
    candidate = db_dir / "overview" / "textured_floor_plan.jpg"
    if candidate.exists():
        return candidate
    candidate = db_dir / "overview" / "true_floor_plan.jpg"
    if candidate.exists():
        return candidate
    return None


def _resolve_view_image_source(db_path: Path, entry: Dict[str, Any]) -> Optional[Path]:
    file_name = str(entry.get("file_name") or "").strip()
    if not file_name:
        return None
    candidate = Path(file_name)
    if not candidate.is_absolute():
        candidate = db_path / file_name
    return candidate if candidate.exists() else None


def _export_view_image(view_image_source: Optional[Path], out_path: Path) -> Optional[str]:
    if view_image_source is None:
        return None
    suffix = view_image_source.suffix or ".jpg"
    target = out_path.with_name(f"{out_path.stem}_view{suffix}")
    if view_image_source.resolve() != target.resolve():
        shutil.copy2(view_image_source, target)
    return str(target)


def _load_base_and_projection(
    db_path: Path,
    meta_rows: Sequence[Dict[str, Any]],
    object_rows: Sequence[Dict[str, Any]],
    floorplan_path: Optional[str],
) -> Tuple[np.ndarray, Dict[str, float], Optional[Path]]:
    selected_floorplan = Path(floorplan_path) if floorplan_path else _default_floorplan_path(db_path)
    if selected_floorplan and selected_floorplan.exists():
        base = cv2.imread(str(selected_floorplan), cv2.IMREAD_COLOR)
        if base is None:
            raise RuntimeError(f"Failed to read floor plan image: {selected_floorplan}")
    else:
        base = np.full((1024, 1024, 3), 245, dtype=np.uint8)

    projection = _load_saved_projection(db_path)
    if projection is None:
        projection = _try_scene_projection(db_path)
    if projection is None:
        projection = _projection_from_rows(meta_rows, object_rows)
    return base, projection, selected_floorplan


def render_object_nodes_birdview(
    db_dir: str,
    entry_id: int,
    output_path: str,
    draw_object_object_edges: bool = False,
    floorplan_path: Optional[str] = None,
    sample_objects: Optional[int] = None,
    sample_seed: Optional[int] = None,
    show_heights: bool = False,
) -> str:
    db_path = Path(db_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"DB directory does not exist: {db_path}")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = _load_jsonl(db_path / "meta.jsonl")
    object_rows = _load_jsonl(db_path / "object_meta.jsonl")
    relation_rows = _load_jsonl(db_path / "object_object_relations.jsonl") if draw_object_object_edges else []
    entry_by_id = {int(row["id"]): dict(row) for row in meta_rows if row.get("id") is not None}
    entry = entry_by_id.get(int(entry_id))
    if entry is None:
        raise ValueError(f"entry_id not found in meta.jsonl: {entry_id}")
    view_image_source = _resolve_view_image_source(db_path, entry)
    view_image_output_path = _export_view_image(view_image_source, out_path)

    objects = [
        dict(row)
        for row in object_rows
        if int(row.get("entry_id", -1)) == int(entry_id)
        and _safe_float(row.get("estimated_global_x")) is not None
        and _safe_float(row.get("estimated_global_z")) is not None
    ]
    if not objects:
        raise ValueError(f"No plottable objects found for entry_id={entry_id} in {db_path / 'object_meta.jsonl'}")
    total_object_count = len(objects)
    sampled = False
    if sample_objects is not None and int(sample_objects) > 0 and int(sample_objects) < len(objects):
        rng = random.Random(sample_seed)
        objects = rng.sample(objects, int(sample_objects))
        sampled = True

    base, projection, selected_floorplan = _load_base_and_projection(
        db_path=db_path,
        meta_rows=meta_rows,
        object_rows=object_rows,
        floorplan_path=floorplan_path,
    )

    canvas = base.copy()
    height, width = canvas.shape[:2]
    camera_x = float(entry["x"])
    camera_y = float(entry["world_position"][1]) if isinstance(entry.get("world_position"), (list, tuple)) and len(entry.get("world_position")) >= 2 else 0.0
    camera_z = float(entry["y"])
    camera_px, camera_py = _world_to_pixel_floor(camera_x, camera_z, width, height, projection)

    plotted_objects: List[Dict[str, Any]] = []
    object_pixels: Dict[int, Tuple[int, int]] = {}
    for row in objects:
        obj_x = float(row["estimated_global_x"])
        obj_z = float(row["estimated_global_z"])
        px, py = _world_to_pixel_floor(obj_x, obj_z, width, height, projection)
        object_global_id = int(row["object_global_id"])
        object_pixels[object_global_id] = (px, py)
        plotted_objects.append(
            {
                "object_global_id": object_global_id,
                "label": str(row.get("label") or "unknown"),
                "x": obj_x,
                "y": _safe_float(row.get("estimated_global_y")),
                "z": obj_z,
                "pixel": [int(px), int(py)],
            }
        )
        cv2.line(canvas, (camera_px, camera_py), (px, py), (80, 160, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 7, (40, 40, 220), -1, cv2.LINE_AA)
        label = f"{row.get('label') or 'unknown'} #{object_global_id}"
        if show_heights:
            object_y = _safe_float(row.get("estimated_global_y"))
            label = f"{label} y={object_y:.2f}" if object_y is not None else f"{label} y=na"
        cv2.putText(canvas, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)

    if draw_object_object_edges:
        for edge in relation_rows:
            if int(edge.get("entry_id", -1)) != int(entry_id):
                continue
            source_id = int(edge["source_object_global_id"])
            target_id = int(edge["target_object_global_id"])
            source_px = object_pixels.get(source_id)
            target_px = object_pixels.get(target_id)
            if source_px is None or target_px is None:
                continue
            cv2.line(canvas, source_px, target_px, (90, 210, 90), 1, cv2.LINE_AA)

    cv2.circle(canvas, (camera_px, camera_py), 9, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(canvas, (camera_px, camera_py), 6, (0, 215, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"camera entry {int(entry_id)}",
        (camera_px + 10, camera_py + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"camera entry {int(entry_id)}",
        (camera_px + 10, camera_py + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save bird-view render: {out_path}")

    sidecar = {
        "entry_id": int(entry_id),
        "sampled": bool(sampled),
        "num_total_objects": int(total_object_count),
        "num_objects": len(plotted_objects),
        "sample_objects": int(sample_objects) if sample_objects is not None else None,
        "sample_seed": int(sample_seed) if sample_seed is not None else None,
        "camera": {
            "x": camera_x,
            "y": camera_y,
            "z": camera_z,
            "pixel": [int(camera_px), int(camera_py)],
        },
        "objects": plotted_objects,
        "draw_object_object_edges": bool(draw_object_object_edges),
        "show_heights": bool(show_heights),
        "floorplan_path": str(selected_floorplan) if selected_floorplan else None,
        "view_image_source_path": str(view_image_source) if view_image_source else None,
        "view_image_output_path": view_image_output_path,
        "projection": projection,
    }
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(out_path)


def render_all_object_nodes_birdview(
    db_dir: str,
    output_path: str,
    floorplan_path: Optional[str] = None,
    draw_labels: bool = False,
    draw_cameras: bool = False,
    sample_objects: Optional[int] = None,
    sample_seed: Optional[int] = None,
    show_heights: bool = False,
) -> str:
    db_path = Path(db_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"DB directory does not exist: {db_path}")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = _load_jsonl(db_path / "meta.jsonl")
    object_rows = [
        dict(row)
        for row in _load_jsonl(db_path / "object_meta.jsonl")
        if _safe_float(row.get("estimated_global_x")) is not None
        and _safe_float(row.get("estimated_global_z")) is not None
    ]
    if not object_rows:
        raise ValueError(f"No plottable objects found in {db_path / 'object_meta.jsonl'}")
    total_object_count = len(object_rows)
    sampled = False
    if sample_objects is not None and int(sample_objects) > 0 and int(sample_objects) < len(object_rows):
        rng = random.Random(sample_seed)
        object_rows = rng.sample(object_rows, int(sample_objects))
        sampled = True
    effective_draw_labels = bool(draw_labels) or sample_objects is not None
    base, projection, selected_floorplan = _load_base_and_projection(
        db_path=db_path,
        meta_rows=meta_rows,
        object_rows=object_rows,
        floorplan_path=floorplan_path,
    )
    canvas = base.copy()
    height, width = canvas.shape[:2]

    if draw_cameras:
        for entry in meta_rows:
            camera_x = _safe_float(entry.get("x"))
            camera_z = _safe_float(entry.get("y"))
            if camera_x is None or camera_z is None:
                continue
            px, py = _world_to_pixel_floor(camera_x, camera_z, width, height, projection)
            cv2.circle(canvas, (px, py), 3, (0, 215, 255), -1, cv2.LINE_AA)

    plotted_objects: List[Dict[str, Any]] = []
    for row in object_rows:
        obj_x = float(row["estimated_global_x"])
        obj_z = float(row["estimated_global_z"])
        px, py = _world_to_pixel_floor(obj_x, obj_z, width, height, projection)
        object_global_id = int(row["object_global_id"])
        label = str(row.get("label") or "unknown")
        plotted_objects.append(
            {
                "object_global_id": object_global_id,
                "entry_id": int(row.get("entry_id", -1)),
                "label": label,
                "x": obj_x,
                "y": _safe_float(row.get("estimated_global_y")),
                "z": obj_z,
                "pixel": [int(px), int(py)],
            }
        )
        cv2.circle(canvas, (px, py), 4, (40, 40, 220), -1, cv2.LINE_AA)
        if effective_draw_labels:
            text = f"{label} #{object_global_id}"
            if show_heights:
                object_y = _safe_float(row.get("estimated_global_y"))
                text = f"{text} y={object_y:.2f}" if object_y is not None else f"{text} y=na"
            cv2.putText(canvas, text, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1, cv2.LINE_AA)

    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save all-objects bird-view render: {out_path}")

    sidecar = {
        "mode": "all_objects",
        "sampled": bool(sampled),
        "num_total_objects": int(total_object_count),
        "num_objects": len(plotted_objects),
        "sample_objects": int(sample_objects) if sample_objects is not None else None,
        "sample_seed": int(sample_seed) if sample_seed is not None else None,
        "draw_labels": bool(effective_draw_labels),
        "draw_cameras": bool(draw_cameras),
        "show_heights": bool(show_heights),
        "floorplan_path": str(selected_floorplan) if selected_floorplan else None,
        "projection": projection,
        "objects": plotted_objects,
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(out_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render object nodes and view-object links on a bird-view floor plan.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory containing meta/object metadata")
    parser.add_argument("--output_path", type=str, required=True, help="Output PNG path")
    parser.add_argument("--entry_id", type=int, default=None, help="Entry/view id to render")
    parser.add_argument("--all_objects", action="store_true", help="Render all object nodes in the DB on one floor plan")
    parser.add_argument("--floorplan_path", type=str, default=None, help="Optional explicit floor-plan image path")
    parser.add_argument(
        "--draw_object_object_edges",
        action="store_true",
        help="Also draw object-object relation edges for the selected frame",
    )
    parser.add_argument(
        "--sample_objects",
        type=int,
        default=None,
        help="Randomly sample this many objects before rendering; works for both single-view and --all_objects modes",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Optional random seed for --sample_objects",
    )
    parser.add_argument("--show_heights", action="store_true", help="Append estimated y values to labels and sidecar output")
    parser.add_argument("--draw_labels", action="store_true", help="Draw object labels on the all-objects render")
    parser.add_argument("--draw_cameras", action="store_true", help="Draw camera/view locations on the all-objects render")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if bool(args.all_objects):
        out = render_all_object_nodes_birdview(
            db_dir=args.db_dir,
            output_path=args.output_path,
            floorplan_path=args.floorplan_path,
            draw_labels=bool(args.draw_labels),
            draw_cameras=bool(args.draw_cameras),
            sample_objects=args.sample_objects,
            sample_seed=args.sample_seed,
            show_heights=bool(args.show_heights),
        )
        print(json.dumps({"output_path": out}, ensure_ascii=True))
    else:
        if args.entry_id is None:
            raise ValueError("--entry_id is required unless --all_objects is set")
        out = render_object_nodes_birdview(
            db_dir=args.db_dir,
            entry_id=args.entry_id,
            output_path=args.output_path,
            draw_object_object_edges=bool(args.draw_object_object_edges),
            floorplan_path=args.floorplan_path,
            sample_objects=args.sample_objects,
            sample_seed=args.sample_seed,
            show_heights=bool(args.show_heights),
        )
        sidecar = json.loads(Path(out).with_suffix(".json").read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "output_path": out,
                    "view_image_path": sidecar.get("view_image_output_path"),
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
