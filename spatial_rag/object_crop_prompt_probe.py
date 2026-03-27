import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import spatial_rag.config  # noqa: F401  # Ensure project env side effects match runtime scripts.

from spatial_rag.vlm_captioner import VLMCaptioner


def _safe_text(value) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _find_latest_query_json(query_dir: Path) -> Optional[Path]:
    candidates = sorted(query_dir.glob("query_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def _load_query_context(query_json_path: Optional[Path]) -> Dict[str, object]:
    if query_json_path is None or not query_json_path.exists():
        return {}
    try:
        payload = json.loads(query_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_job_for_crop(crop_path: Path) -> Dict[str, object]:
    query_dir = crop_path.parent
    query_json_path = _find_latest_query_json(query_dir)
    ctx = _load_query_context(query_json_path)
    selected = ctx.get("selected_detection") if isinstance(ctx.get("selected_detection"), dict) else {}
    query = ctx.get("query") if isinstance(ctx.get("query"), dict) else {}
    yolo_label = _safe_text(query.get("yolo_label_hint") or selected.get("label") or "unknown")
    yolo_conf = query.get("yolo_confidence_hint")
    try:
        yolo_conf = None if yolo_conf is None else float(yolo_conf)
    except Exception:
        yolo_conf = None

    return {
        "crop_path": str(crop_path),
        "query_dir": str(query_dir),
        "query_json_path": None if query_json_path is None else str(query_json_path),
        "yolo_label": yolo_label,
        "yolo_confidence": yolo_conf,
    }


def collect_jobs(results_dir: Path, limit: Optional[int] = None) -> List[Dict[str, object]]:
    crops = sorted(results_dir.rglob("query_object_crop.jpg"))
    if limit is not None and limit > 0:
        crops = crops[: int(limit)]
    return [_build_job_for_crop(crop_path) for crop_path in crops]


def run_probe(
    results_dir: str,
    output_dir: str,
    vlm_model: str,
    use_cache: bool,
    force_refresh: bool,
    print_prompts: bool,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    results_root = Path(results_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_root}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(results_root, limit=limit)
    captioner = VLMCaptioner(
        model_name=vlm_model,
        use_cache=use_cache,
        cache_dir=str(output_root / "vlm_cache"),
        object_use_cache=use_cache,
        object_cache_dir=str(output_root / "vlm_object_cache"),
    )

    records: List[Dict[str, object]] = []
    for idx, job in enumerate(jobs, start=1):
        crop_path = str(job["crop_path"])
        yolo_label = str(job["yolo_label"])
        yolo_conf = job["yolo_confidence"]
        yolo_conf_text = f"{float(yolo_conf):.3f}" if yolo_conf is not None else "unknown"
        system_prompt = VLMCaptioner._object_crop_system_prompt()
        user_prompt = VLMCaptioner._object_crop_user_prompt(yolo_label, yolo_conf_text)
        if print_prompts:
            print(f"\n=== Crop {idx}: {crop_path} ===")
            print("[System Prompt]")
            print(system_prompt)
            print("[User Prompt]")
            print(user_prompt)
        result = captioner.describe_object_crop_with_meta(
            crop_path,
            force_refresh=force_refresh,
            yolo_label=yolo_label,
            yolo_confidence=yolo_conf,
        )
        record = {
            "index": int(idx),
            "crop_path": crop_path,
            "query_dir": str(job["query_dir"]),
            "query_json_path": job["query_json_path"],
            "yolo_label": yolo_label,
            "yolo_confidence": yolo_conf,
            "prompt_version": VLMCaptioner._object_crop_prompt_version(),
            "model": vlm_model,
            "prompts": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            "result": {
                "label": _safe_text(result.get("label")),
                "short_description": _safe_text(result.get("short_description")),
                "long_description": _safe_text(result.get("long_description")),
                "attributes": list(result.get("attributes") or []),
                "distance_from_camera_m": result.get("distance_from_camera_m"),
                "source": str(result.get("source") or "unknown"),
                "yolo_label_hint": _safe_text(result.get("yolo_label_hint")),
                "yolo_confidence_hint": result.get("yolo_confidence_hint"),
            },
        }
        records.append(record)

    jsonl_path = output_root / "crop_prompt_probe.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results_dir": str(results_root),
        "output_dir": str(output_root),
        "model": vlm_model,
        "prompt_version": VLMCaptioner._object_crop_prompt_version(),
        "num_crops": len(records),
        "jsonl_path": str(jsonl_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe GPT crop descriptions on saved query_object_crop images.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="object_vpr_results",
        help="Root directory containing query_object_crop.jpg files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write probe outputs (default: <results_dir>/crop_prompt_probe_<timestamp>)",
    )
    parser.add_argument("--vlm_model", type=str, default="gpt-5-mini", help="OpenAI VLM model")
    parser.add_argument("--use_cache", type=str, default="true", help="Whether to cache crop prompt results")
    parser.add_argument("--force_refresh", type=str, default="false", help="Whether to bypass crop cache")
    parser.add_argument("--print_prompts", type=str, default="true", help="Whether to print full prompts for each crop")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of crops to probe")
    args = parser.parse_args()

    use_cache = str(args.use_cache).strip().lower() in {"1", "true", "t", "yes", "y"}
    force_refresh = str(args.force_refresh).strip().lower() in {"1", "true", "t", "yes", "y"}
    print_prompts = str(args.print_prompts).strip().lower() in {"1", "true", "t", "yes", "y"}
    output_dir = args.output_dir
    if not output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(args.results_dir) / f"crop_prompt_probe_{ts}")

    summary = run_probe(
        results_dir=args.results_dir,
        output_dir=output_dir,
        vlm_model=args.vlm_model,
        use_cache=use_cache,
        force_refresh=force_refresh,
        print_prompts=print_prompts,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
