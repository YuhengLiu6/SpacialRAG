import argparse
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from spatial_rag.config import (
    OBJECT_MAX_PER_FRAME,
    OBJECT_PARSE_RETRIES,
    OBJECT_TEXT_MODE,
    OBJECT_USE_CACHE,
    SCENE_PATH,
)
from spatial_rag.vpr_query import load_spatial_db, run_query


def _str_to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _choose_entries(entries: List[Dict], num_queries: int, seed: int) -> List[Dict]:
    if num_queries <= 0:
        raise ValueError("num_queries must be > 0")
    if num_queries > len(entries):
        raise ValueError(f"num_queries={num_queries} exceeds DB size={len(entries)}")

    indices = list(range(len(entries)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = [entries[i] for i in indices[:num_queries]]
    return chosen


def _sample_non_cardinal_orientations(num_queries: int, seed: int) -> List[float]:
    candidates = [deg for deg in range(360) if deg not in {0, 90, 180, 270}]
    if num_queries > len(candidates):
        raise ValueError(f"num_queries={num_queries} exceeds available non-cardinal integer headings={len(candidates)}")
    rng = random.Random(seed + 7919)
    rng.shuffle(candidates)
    return [float(v) for v in candidates[:num_queries]]


def _sample_novel_query_poses(
    num_queries: int,
    seed: int,
    scene_path: str,
    db_entries: List[Dict],
    min_db_dist: float,
    min_query_dist: float,
    max_sample_attempts: int,
) -> List[Dict]:
    from spatial_rag.explorer import Explorer

    if min_db_dist < 0 or min_query_dist < 0:
        raise ValueError("min_db_dist and min_query_dist must be >= 0")

    db_xy = np.asarray([[float(e["x"]), float(e["y"])] for e in db_entries], dtype=np.float32)
    orientations = _sample_non_cardinal_orientations(num_queries=num_queries, seed=seed)
    rng = random.Random(seed)
    chosen_xy: List[List[float]] = []
    tries = 0

    explorer = Explorer(scene_path=scene_path)
    try:
        pf = explorer.sim.pathfinder
        agent_y = float(explorer.agent.get_state().position[1])

        while len(chosen_xy) < num_queries and tries < max_sample_attempts:
            tries += 1
            p = np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
            if p.shape[0] != 3 or not np.isfinite(p).all():
                continue

            # Keep sampled points near the agent's operating floor.
            if abs(float(p[1]) - agent_y) > 1.0:
                continue

            x = float(p[0])
            y = float(p[2])

            if db_xy.size > 0:
                d_db = np.linalg.norm(db_xy - np.asarray([x, y], dtype=np.float32), axis=1)
                if float(np.min(d_db)) < float(min_db_dist):
                    continue

            if chosen_xy:
                existing = np.asarray(chosen_xy, dtype=np.float32)
                d_q = np.linalg.norm(existing - np.asarray([x, y], dtype=np.float32), axis=1)
                if float(np.min(d_q)) < float(min_query_dist):
                    continue

            chosen_xy.append([x, y])

        if len(chosen_xy) < num_queries:
            raise RuntimeError(
                f"Only sampled {len(chosen_xy)}/{num_queries} valid novel poses after {tries} attempts. "
                f"Try reducing min_db_dist/min_query_dist."
            )
    finally:
        explorer.close()

    rng.shuffle(chosen_xy)
    out: List[Dict] = []
    for i, (xy, theta) in enumerate(zip(chosen_xy[:num_queries], orientations), start=1):
        out.append(
            {
                "sample_id": i,
                "query_id": None,
                "x0": float(xy[0]),
                "y0": float(xy[1]),
                "theta0": float(theta),
            }
        )
    return out


def run_batch(
    num_queries: int,
    seed: int,
    db_dir: str,
    scene_path: str,
    top_k: int,
    w_img: float,
    w_txt: float,
    results_dir: str,
    run_name: str,
    vlm_model: str,
    use_cache: bool,
    query_source: str,
    min_db_dist: float,
    min_query_dist: float,
    max_sample_attempts: int,
    query_offset_x: float,
    query_offset_y: float,
    query_offset_theta: float,
    object_parse_retries: int,
    object_use_cache: bool,
    object_cache_dir: Optional[str],
    object_max_per_frame: int,
    object_text_mode: str,
) -> Dict:
    from spatial_rag.embedder import Embedder

    entries, _, _ = load_spatial_db(db_dir, object_text_mode=object_text_mode)
    print(f"[Batch] loaded_spatial_db entries={len(entries)} db_dir={db_dir}", flush=True)
    print("[Batch] initializing shared Embedder...", flush=True)
    shared_embedder = Embedder()
    print("[Batch] shared Embedder ready", flush=True)
    if query_source == "db":
        selected_entries = _choose_entries(entries, num_queries=num_queries, seed=seed)
        selected_queries: List[Dict] = [
            {
                "sample_id": i,
                "query_id": int(e["id"]),
                "x0": float(e["x"]),
                "y0": float(e["y"]),
                "theta0": float(e["orientation"]),
            }
            for i, e in enumerate(selected_entries, start=1)
        ]
    elif query_source == "novel":
        selected_queries = _sample_novel_query_poses(
            num_queries=num_queries,
            seed=seed,
            scene_path=scene_path,
            db_entries=entries,
            min_db_dist=min_db_dist,
            min_query_dist=min_query_dist,
            max_sample_attempts=max_sample_attempts,
        )
    else:
        raise ValueError(f"Unsupported query_source: {query_source}")

    # Apply a deterministic 2D offset to all test query points for batch evaluation.
    if (
        abs(float(query_offset_x)) > 0.0
        or abs(float(query_offset_y)) > 0.0
        or abs(float(query_offset_theta)) > 0.0
    ):
        for q in selected_queries:
            q["x0"] = float(q["x0"]) + float(query_offset_x)
            q["y0"] = float(q["y0"]) + float(query_offset_y)
            q["theta0"] = (float(q["theta0"]) + float(query_offset_theta)) % 360.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_tag = "novel" if query_source == "novel" else "db"
    run_folder_name = run_name.strip() if run_name.strip() else f"batch_{num_queries}_queries_{source_tag}_{ts}"
    run_root = Path(results_dir) / run_folder_name
    overlays_dir = run_root / "overlays"
    run_root.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    failures: List[Dict] = []

    for i, q in enumerate(tqdm(selected_queries, total=len(selected_queries), desc="Batch VPR queries"), start=1):
        qid: Optional[int] = q.get("query_id")
        x0 = float(q["x0"])
        y0 = float(q["y0"])
        theta0 = float(q["theta0"])
        query_dir_name = (
            f"query_{i:02d}_id_{int(qid):06d}_ori_{int(theta0):03d}"
            if qid is not None
            else f"query_{i:02d}_sampled_ori_{int(theta0):03d}"
        )
        query_dir = run_root / query_dir_name
        query_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{i:02d}/{num_queries:02d}] "
            f"run_query(id={qid if qid is not None else 'sampled'}, x={x0:.3f}, y={y0:.3f}, theta={theta0:.1f})",
            flush=True,
        )

        try:
            t_query_start = time.perf_counter()
            res = run_query(
                x0=x0,
                y0=y0,
                theta0=theta0,
                db_dir=db_dir,
                scene_path=scene_path,
                top_k=top_k,
                w_img=w_img,
                w_txt=w_txt,
                results_dir=str(query_dir),
                vlm_model=vlm_model,
                use_cache=use_cache,
                object_parse_retries=object_parse_retries,
                object_use_cache=object_use_cache,
                object_cache_dir=object_cache_dir,
                object_max_per_frame=object_max_per_frame,
                object_text_mode=object_text_mode,
                embedder=shared_embedder,
            )
            t_query_sec = time.perf_counter() - t_query_start
            print(f"  -> finished in {t_query_sec:.2f}s", flush=True)

            overlay_src = Path(res["artifacts"]["query_overlay"])
            overlay_name = (
                f"q{i:02d}_id_{int(qid):06d}_ori_{int(theta0):03d}.jpg"
                if qid is not None
                else f"q{i:02d}_sampled_ori_{int(theta0):03d}.jpg"
            )
            overlay_dst = overlays_dir / overlay_name
            if overlay_src.exists():
                shutil.copy2(overlay_src, overlay_dst)
            else:
                overlay_dst = None

            records.append(
                {
                    "index": i,
                    "query_id": qid,
                    "pose": {"x0": x0, "y0": y0, "theta0": theta0},
                    "query_dir": str(query_dir),
                    "overlay_path": str(overlay_dst) if overlay_dst else None,
                    "prediction": res["prediction"],
                    "metrics": res["metrics"],
                    "query_json": res["artifacts"].get("query_json"),
                    "used_snap_fallback": bool(res["query"].get("used_snap_fallback", False)),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "index": i,
                    "query_id": qid,
                    "pose": {"x0": x0, "y0": y0, "theta0": theta0},
                    "query_dir": str(query_dir),
                    "error": str(exc),
                }
            )
            print(f"  -> failed: {exc}", flush=True)

    summary = {
        "timestamp": ts,
        "num_queries_requested": int(num_queries),
        "num_succeeded": len(records),
        "num_failed": len(failures),
        "config": {
            "seed": int(seed),
            "db_dir": db_dir,
            "scene_path": scene_path,
            "top_k": int(top_k),
            "w_img": float(w_img),
            "w_txt": float(w_txt),
            "results_dir": str(run_root),
            "overlays_dir": str(overlays_dir),
            "vlm_model": vlm_model,
            "use_cache": bool(use_cache),
            "query_source": query_source,
            "min_db_dist": float(min_db_dist),
            "min_query_dist": float(min_query_dist),
            "max_sample_attempts": int(max_sample_attempts),
            "query_offset_x": float(query_offset_x),
            "query_offset_y": float(query_offset_y),
            "query_offset_theta": float(query_offset_theta),
            "object_parse_retries": int(object_parse_retries),
            "object_use_cache": bool(object_use_cache),
            "object_cache_dir": object_cache_dir,
            "object_max_per_frame": int(object_max_per_frame),
            "object_text_mode": object_text_mode,
        },
        "selected_queries": selected_queries,
        "results": records,
        "failures": failures,
    }

    summary_path = run_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved batch summary: {summary_path}", flush=True)
    print(f"Saved overlays dir: {overlays_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch VPR queries and save all overlays in one folder.")
    parser.add_argument("--num_queries", type=int, default=20, help="How many queries to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selecting query poses")
    parser.add_argument("--db_dir", type=str, default="spatial_db", help="Spatial DB directory")
    parser.add_argument("--scene_path", type=str, default=SCENE_PATH, help="Habitat scene .glb")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k retrieval results for each query")
    parser.add_argument("--w_img", type=float, default=0.5, help="Image similarity weight")
    parser.add_argument("--w_txt", type=float, default=0.5, help="Text similarity weight")
    parser.add_argument("--results_dir", type=str, default="vpr_results", help="Output root directory")
    parser.add_argument("--run_name", type=str, default="", help="Subfolder name under results_dir")
    parser.add_argument("--vlm_model", type=str, default="gpt-5-mini", help="OpenAI VLM model")
    parser.add_argument("--use_cache", type=_str_to_bool, default=True, help="Use VLM cache (true/false)")
    parser.add_argument(
        "--query_source",
        type=str,
        default="db",
        choices=["db", "novel"],
        help="db: sample query poses from DB entries; novel: sample navigable poses not overlapping DB points",
    )
    parser.add_argument("--min_db_dist", type=float, default=0.35, help="Min 2D distance (meters) from DB poses in novel mode")
    parser.add_argument("--min_query_dist", type=float, default=0.60, help="Min 2D distance (meters) between sampled queries in novel mode")
    parser.add_argument("--max_sample_attempts", type=int, default=20000, help="Max attempts for novel pose sampling")
    parser.add_argument("--query_offset_x", type=float, default=0.0, help="Global x offset (meters) added to each batch query pose")
    parser.add_argument("--query_offset_y", type=float, default=0.0, help="Global y/z offset (meters) added to each batch query pose")
    parser.add_argument("--query_offset_theta", type=float, default=0.0, help="Global theta offset (degrees) added to each batch query pose")
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
        help="Object cache directory for query object extraction",
    )
    parser.add_argument(
        "--object_max_per_frame",
        type=int,
        default=OBJECT_MAX_PER_FRAME,
        help="Maximum objects extracted per query image",
    )
    parser.add_argument(
        "--object_text_mode",
        type=str,
        default=OBJECT_TEXT_MODE,
        choices=["short", "long"],
        help="Text channel to use: short=description concat, long=long_form_open_description concat",
    )
    args = parser.parse_args()

    summary = run_batch(
        num_queries=args.num_queries,
        seed=args.seed,
        db_dir=args.db_dir,
        scene_path=args.scene_path,
        top_k=args.top_k,
        w_img=args.w_img,
        w_txt=args.w_txt,
        results_dir=args.results_dir,
        run_name=args.run_name,
        vlm_model=args.vlm_model,
        use_cache=args.use_cache,
        query_source=args.query_source,
        min_db_dist=args.min_db_dist,
        min_query_dist=args.min_query_dist,
        max_sample_attempts=args.max_sample_attempts,
        query_offset_x=args.query_offset_x,
        query_offset_y=args.query_offset_y,
        query_offset_theta=args.query_offset_theta,
        object_parse_retries=args.object_parse_retries,
        object_use_cache=args.object_use_cache,
        object_cache_dir=args.object_cache_dir,
        object_max_per_frame=args.object_max_per_frame,
        object_text_mode=args.object_text_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
