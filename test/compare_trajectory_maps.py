import argparse
import json
import sys
from pathlib import Path

import cv2


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def _safe_name(scene_path: str) -> str:
    stem = Path(scene_path).stem.strip()
    return stem if stem else "scene"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render only trajectory map and trajectory floor map for a scene."
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        default="",
        help="Path to Habitat scene .glb. Empty means use config default.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test/output",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--meters_per_step",
        type=float,
        default=1.5,
        help="Waypoint spacing for explore_full_house.",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=120.0,
        help="HFOV for center-highest trajectory map.",
    )
    args = parser.parse_args()

    _ensure_repo_root_on_path()
    from spatial_rag.config import SCENE_PATH
    from spatial_rag.explorer import Explorer

    scene_path = args.scene_path.strip() if args.scene_path.strip() else SCENE_PATH
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_tag = _safe_name(scene_path)
    center_out = out_dir / f"{scene_tag}_trajectory_map.jpg"
    floor_out = out_dir / f"{scene_tag}_trajectory_floor_map.jpg"

    explorer = Explorer(scene_path=scene_path)
    try:
        _, poses = explorer.explore_full_house(meters_per_step=float(args.meters_per_step))
        center_img = explorer.render_center_highest_view_with_trajectory(
            poses, hfov=float(args.hfov)
        )
        floor_img = explorer.render_true_floor_plan_with_trajectory(poses)

        ok_center = cv2.imwrite(str(center_out), center_img)
        ok_floor = cv2.imwrite(str(floor_out), floor_img)
        if not ok_center:
            raise RuntimeError(f"Failed to save center trajectory map: {center_out}")
        if not ok_floor:
            raise RuntimeError(f"Failed to save floor trajectory map: {floor_out}")

        print(
            json.dumps(
                {
                    "scene_path": scene_path,
                    "poses_count": len(poses),
                    "trajectory_map": str(center_out),
                    "trajectory_floor_map": str(floor_out),
                },
                ensure_ascii=True,
                indent=2,
            )
        )
    finally:
        explorer.close()


if __name__ == "__main__":
    main()
