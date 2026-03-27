import argparse
import json

from spatial_rag.config import (
    OBJECT_MAX_PER_FRAME,
    OBJECT_PARSE_RETRIES,
    OBJECT_USE_CACHE,
    SCAN_ANGLES,
    SCENE_PATH,
    SPATIAL_DB_DIR,
    SPATIAL_DB_VLM_MODEL,
    VLM_ANGLE_SPLIT_ENABLE,
    VLM_ANGLE_STEP,
)
from spatial_rag.spatial_db_builder import (
    _parse_scan_angles,
    _str_to_bool,
    build_spatial_database_angle_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an angle-split spatial database from Habitat exploration.")
    parser.add_argument("--scene_path", type=str, default=SCENE_PATH, help="Path to Habitat scene .glb")
    parser.add_argument("--meters_per_step", type=float, default=1.5, help="Waypoint spacing in meters")
    parser.add_argument(
        "--max_positions",
        "--max_position",
        type=int,
        default=None,
        help="Limit number of positions (each position has len(scan_angles) orientation frames)",
    )
    parser.add_argument("--output_dir", type=str, default=SPATIAL_DB_DIR, help="Output directory")
    parser.add_argument("--vlm_model", type=str, default=SPATIAL_DB_VLM_MODEL, help="OpenAI VLM model")
    parser.add_argument("--use_cache", type=_str_to_bool, default=True, help="Whether to cache VLM outputs")
    parser.add_argument(
        "--object_max_per_frame",
        type=int,
        default=OBJECT_MAX_PER_FRAME,
        help="Max extracted objects per frame",
    )
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
        help="Whether to cache VLM object outputs",
    )
    parser.add_argument(
        "--object_cache_dir",
        type=str,
        default=None,
        help="Object cache directory (default: <output_dir>/vlm_object_cache)",
    )
    parser.add_argument(
        "--tour_mode",
        type=str,
        default="full_house",
        choices=["full_house", "random"],
        help="Exploration mode for DB creation",
    )
    parser.add_argument("--random_num_steps", type=int, default=50, help="Number of move steps in random mode")
    parser.add_argument("--random_step_size", type=float, default=1.0, help="Step size in meters in random mode")
    parser.add_argument(
        "--scan_angles",
        "--random_scan_angles",
        type=_parse_scan_angles,
        default=SCAN_ANGLES,
        help="Comma-separated scan angles, e.g. '0,90,180,270'",
    )
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for random tour")
    parser.add_argument(
        "--random_max_attempts_per_step",
        type=int,
        default=32,
        help="Max attempts per step for random tour",
    )
    parser.add_argument(
        "--random_include_start_scan",
        type=_str_to_bool,
        default=True,
        help="Whether to capture scan at start position in random tour",
    )
    parser.add_argument(
        "--angle_split_enable",
        type=_str_to_bool,
        default=VLM_ANGLE_SPLIT_ENABLE,
        help="Whether to offset object orientation by left/center/right angle buckets",
    )
    parser.add_argument(
        "--angle_step",
        type=int,
        default=VLM_ANGLE_STEP,
        help="Angle offset in degrees for left/right buckets",
    )
    args = parser.parse_args()

    report = build_spatial_database_angle_split(
        scene_path=args.scene_path,
        meters_per_step=args.meters_per_step,
        max_positions=args.max_positions,
        output_dir=args.output_dir,
        vlm_model=args.vlm_model,
        use_cache=args.use_cache,
        object_max_per_frame=args.object_max_per_frame,
        object_parse_retries=args.object_parse_retries,
        object_use_cache=args.object_use_cache,
        object_cache_dir=args.object_cache_dir,
        tour_mode=args.tour_mode,
        random_num_steps=args.random_num_steps,
        random_step_size=args.random_step_size,
        random_scan_angles=args.scan_angles,
        random_seed=args.random_seed,
        random_max_attempts_per_step=args.random_max_attempts_per_step,
        random_include_start_scan=args.random_include_start_scan,
        angle_split_enable=args.angle_split_enable,
        angle_step=args.angle_step,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
