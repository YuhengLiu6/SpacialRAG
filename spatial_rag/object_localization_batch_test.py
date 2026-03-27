import argparse
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from spatial_rag.config import OBJECT_TEXT_MODE, SCENE_PATH
from spatial_rag.object_localization_query import NoValidObjectDetectionsError, run_object_query
from spatial_rag.vpr_batch_test import _sample_novel_query_poses
from spatial_rag.vpr_query import load_spatial_db


def _str_to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _sample_db_pose(
    entries: Sequence[Dict],
    rng: random.Random,
    used_pose_keys: set,
) -> Optional[Dict]:
    candidates = list(entries)
    rng.shuffle(candidates)
    for entry in candidates:
        key = (
            round(float(entry["x"]), 3),
            round(float(entry["y"]), 3),
            round(float(entry["orientation"]), 1),
        )
        if key in used_pose_keys:
            continue
        used_pose_keys.add(key)
        return {
            "query_id": int(entry["id"]),
            "x0": float(entry["x"]),
            "y0": float(entry["y"]),
            "theta0": float(entry["orientation"]),
        }
    return None


def _sample_novel_pose(
    seed: int,
    scene_path: str,
    db_entries: Sequence[Dict],
    accepted_queries: Sequence[Dict],
    min_db_dist: float,
    min_query_dist: float,
    max_sample_attempts: int,
) -> Dict:
    augmented_entries = list(db_entries) + [
        {
            "id": -100000 - idx,
            "x": float(q["x0"]),
            "y": float(q["y0"]),
            "orientation": float(q["theta0"]),
            "file_name": "synthetic",
        }
        for idx, q in enumerate(accepted_queries)
    ]
    sampled = _sample_novel_query_poses(
        num_queries=1,
        seed=seed,
        scene_path=scene_path,
        db_entries=augmented_entries,
        min_db_dist=min_db_dist,
        min_query_dist=min_query_dist,
        max_sample_attempts=max_sample_attempts,
    )
    query = dict(sampled[0])
    query["query_id"] = None
    return query


def _compute_aggregate_metrics(results: Sequence[Dict]) -> Dict[str, Optional[float]]:
    if not results:
        return {
            "mean_pos_error": None,
            "median_pos_error": None,
            "p90_pos_error": None,
            "mean_yaw_error": None,
        }
    pos = [float(item["metrics"]["pos_error"]) for item in results]
    yaw = [float(item["metrics"]["yaw_error"]) for item in results]
    return {
        "mean_pos_error": float(statistics.fmean(pos)),
        "median_pos_error": float(statistics.median(pos)),
        "p90_pos_error": float(np.percentile(np.asarray(pos, dtype=np.float32), 90)),
        "mean_yaw_error": float(statistics.fmean(yaw)),
    }


def run_batch(
    num_queries: int,
    seed: int,
    db_dir: str,
    scene_path: str,
    top_k: int,
    results_dir: str,
    run_name: str,
    vlm_model: str,
    use_cache: bool,
    object_text_mode: str,
    query_source: str,
    min_db_dist: float,
    min_query_dist: float,
    max_sample_attempts: int,
    detector_conf: float,
    min_bbox_side_px: int,
    min_bbox_area_ratio: float,
    max_bbox_area_ratio: float,
    crop_padding_ratio: float,
    object_candidate_pool: int,
    max_resample_attempts_per_query: int,
    detector_classes: Optional[str],
) -> Dict:
    from spatial_rag.embedder import Embedder

    entries, _, _ = load_spatial_db(db_dir, object_text_mode=object_text_mode)
    if query_source == "db" and num_queries > len(entries):
        raise ValueError(f"num_queries={num_queries} exceeds DB size={len(entries)} in db mode")

    shared_embedder = Embedder()
    rng = random.Random(seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_tag = "novel" if query_source == "novel" else "db"
    run_folder_name = run_name.strip() if run_name.strip() else f"object_batch_{num_queries}_{source_tag}_{ts}"
    run_root = Path(results_dir) / run_folder_name
    run_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    failures: List[Dict] = []
    skips: List[Dict] = []
    selected_queries: List[Dict] = []
    used_pose_keys = set()

    for query_index in tqdm(range(1, num_queries + 1), total=num_queries, desc="Object localization batch"):
        success = False
        for attempt in range(1, max_resample_attempts_per_query + 1):
            if query_source == "db":
                pose = _sample_db_pose(entries, rng=rng, used_pose_keys=used_pose_keys)
                if pose is None:
                    skips.append(
                        {
                            "index": query_index,
                            "reason": "exhausted_db_poses",
                        }
                    )
                    success = True
                    break
            elif query_source == "novel":
                pose = _sample_novel_pose(
                    seed=seed + query_index * 1000 + attempt,
                    scene_path=scene_path,
                    db_entries=entries,
                    accepted_queries=selected_queries,
                    min_db_dist=min_db_dist,
                    min_query_dist=min_query_dist,
                    max_sample_attempts=max_sample_attempts,
                )
            else:
                raise ValueError(f"Unsupported query_source: {query_source}")

            pose["sample_id"] = query_index
            pose["resample_attempt"] = attempt
            query_dir_name = (
                f"query_{query_index:02d}_id_{int(pose['query_id']):06d}_attempt_{attempt:02d}"
                if pose["query_id"] is not None
                else f"query_{query_index:02d}_sampled_attempt_{attempt:02d}"
            )
            query_dir = run_root / query_dir_name
            try:
                result = run_object_query(
                    x0=float(pose["x0"]),
                    y0=float(pose["y0"]),
                    theta0=float(pose["theta0"]),
                    db_dir=db_dir,
                    scene_path=scene_path,
                    top_k=top_k,
                    results_dir=str(query_dir),
                    vlm_model=vlm_model,
                    use_cache=use_cache,
                    object_text_mode=object_text_mode,
                    detector_conf=detector_conf,
                    min_bbox_side_px=min_bbox_side_px,
                    min_bbox_area_ratio=min_bbox_area_ratio,
                    max_bbox_area_ratio=max_bbox_area_ratio,
                    crop_padding_ratio=crop_padding_ratio,
                    object_candidate_pool=object_candidate_pool,
                    selection_seed=seed + query_index * 10000 + attempt,
                    detector_classes=detector_classes,
                    embedder=shared_embedder,
                )
                records.append(
                    {
                        "index": query_index,
                        "query_id": pose["query_id"],
                        "pose": {
                            "x0": float(pose["x0"]),
                            "y0": float(pose["y0"]),
                            "theta0": float(pose["theta0"]),
                        },
                        "resample_attempt": attempt,
                        "query_dir": str(query_dir),
                        "selected_detection": result["selected_detection"],
                        "prediction": result["prediction"],
                        "metrics": result["metrics"],
                        "artifacts": result["artifacts"],
                    }
                )
                selected_queries.append(dict(pose))
                success = True
                break
            except NoValidObjectDetectionsError as exc:
                if attempt >= max_resample_attempts_per_query:
                    skips.append(
                        {
                            "index": query_index,
                            "query_id": pose["query_id"],
                            "pose": {
                                "x0": float(pose["x0"]),
                                "y0": float(pose["y0"]),
                                "theta0": float(pose["theta0"]),
                            },
                            "reason": "skipped_no_valid_object",
                            "error": str(exc),
                        }
                    )
                    success = True
                    break
                continue
            except Exception as exc:
                failures.append(
                    {
                        "index": query_index,
                        "query_id": pose["query_id"],
                        "pose": {
                            "x0": float(pose["x0"]),
                            "y0": float(pose["y0"]),
                            "theta0": float(pose["theta0"]),
                        },
                        "resample_attempt": attempt,
                        "query_dir": str(query_dir),
                        "error": str(exc),
                    }
                )
                success = True
                break
        if not success:
            failures.append(
                {
                    "index": query_index,
                    "query_id": None,
                    "error": "unknown batch control flow failure",
                }
            )

    aggregate_metrics = _compute_aggregate_metrics(records)
    summary = {
        "timestamp": ts,
        "num_queries_requested": int(num_queries),
        "num_succeeded": len(records),
        "num_failed": len(failures),
        "num_skipped": len(skips),
        "config": {
            "seed": int(seed),
            "db_dir": db_dir,
            "scene_path": scene_path,
            "top_k": int(top_k),
            "results_dir": str(run_root),
            "vlm_model": vlm_model,
            "use_cache": bool(use_cache),
            "object_text_mode": object_text_mode,
            "query_source": query_source,
            "min_db_dist": float(min_db_dist),
            "min_query_dist": float(min_query_dist),
            "max_sample_attempts": int(max_sample_attempts),
            "detector_conf": float(detector_conf),
            "min_bbox_side_px": int(min_bbox_side_px),
            "min_bbox_area_ratio": float(min_bbox_area_ratio),
            "max_bbox_area_ratio": float(max_bbox_area_ratio),
            "crop_padding_ratio": float(crop_padding_ratio),
            "object_candidate_pool": int(object_candidate_pool),
            "max_resample_attempts_per_query": int(max_resample_attempts_per_query),
            "detector_classes": detector_classes,
        },
        "selected_queries": selected_queries,
        "results": records,
        "failures": failures,
        "skips": skips,
        "aggregate_metrics": {
            "num_requested": int(num_queries),
            "num_succeeded": int(len(records)),
            "num_failed": int(len(failures)),
            "num_skipped": int(len(skips)),
            **aggregate_metrics,
        },
    }
    summary_path = run_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch object-centric localization queries.")
    parser.add_argument("--num_queries", type=int, default=20, help="How many queries to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--db_dir", type=str, default="spatial_db", help="Spatial DB directory")
    parser.add_argument("--scene_path", type=str, default=SCENE_PATH, help="Habitat scene .glb")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval results")
    parser.add_argument("--results_dir", type=str, default="object_vpr_results", help="Output root directory")
    parser.add_argument("--run_name", type=str, default="", help="Subfolder name under results_dir")
    parser.add_argument("--vlm_model", type=str, default="gpt-5-mini", help="OpenAI VLM model")
    parser.add_argument("--use_cache", type=_str_to_bool, default=True, help="Use VLM cache")
    parser.add_argument("--object_text_mode", type=str, default=OBJECT_TEXT_MODE, choices=["short", "long"])
    parser.add_argument("--query_source", type=str, default="novel", choices=["db", "novel"])
    parser.add_argument("--min_db_dist", type=float, default=0.35, help="Min 2D distance from DB poses in novel mode")
    parser.add_argument("--min_query_dist", type=float, default=0.60, help="Min 2D distance between sampled novel queries")
    parser.add_argument("--max_sample_attempts", type=int, default=20000, help="Max novel sampling attempts")
    parser.add_argument("--detector_conf", type=float, default=0.35, help="Minimum detector confidence")
    parser.add_argument("--min_bbox_side_px", type=int, default=32, help="Minimum bbox side length")
    parser.add_argument("--min_bbox_area_ratio", type=float, default=0.005, help="Minimum bbox area ratio")
    parser.add_argument("--max_bbox_area_ratio", type=float, default=0.60, help="Maximum bbox area ratio")
    parser.add_argument("--crop_padding_ratio", type=float, default=0.25, help="Per-side crop padding ratio")
    parser.add_argument("--object_candidate_pool", type=int, default=256, help="FAISS object search pool size")
    parser.add_argument("--max_resample_attempts_per_query", type=int, default=20, help="Max re-samples when no valid object is found")
    parser.add_argument(
        "--detector_classes",
        type=str,
        default=None,
        help="Comma-separated open-vocabulary classes for YOLO_WORLD",
    )
    args = parser.parse_args()

    summary = run_batch(
        num_queries=args.num_queries,
        seed=args.seed,
        db_dir=args.db_dir,
        scene_path=args.scene_path,
        top_k=args.top_k,
        results_dir=args.results_dir,
        run_name=args.run_name,
        vlm_model=args.vlm_model,
        use_cache=args.use_cache,
        object_text_mode=args.object_text_mode,
        query_source=args.query_source,
        min_db_dist=args.min_db_dist,
        min_query_dist=args.min_query_dist,
        max_sample_attempts=args.max_sample_attempts,
        detector_conf=args.detector_conf,
        min_bbox_side_px=args.min_bbox_side_px,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        max_bbox_area_ratio=args.max_bbox_area_ratio,
        crop_padding_ratio=args.crop_padding_ratio,
        object_candidate_pool=args.object_candidate_pool,
        max_resample_attempts_per_query=args.max_resample_attempts_per_query,
        detector_classes=args.detector_classes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
