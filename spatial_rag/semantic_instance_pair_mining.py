import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from spatial_rag.object_instance_pair_mining import CandidatePairRecord, export_candidate_artifacts


STUFF_GT_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "door frame",
    "window frame",
    "balustrade",
    "stairs",
    "railing",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_view_ids(value: Any) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for item in raw.split(","):
        view_id = str(item or "").strip()
        if view_id and view_id not in out:
            out.append(view_id)
    return out


def _canonical_view_id(value: Any) -> str:
    text = _string_value(value)
    if not text:
        return ""
    if text.startswith("view_"):
        return text
    try:
        numeric = int(text)
    except Exception:
        return text
    return f"view_{numeric:05d}"


def _row_view_id(row: Mapping[str, Any]) -> str:
    for key in ("view_id", "synthetic_view_id", "entry_id", "frame_id"):
        value = row.get(key)
        canonical = _canonical_view_id(value)
        if canonical:
            return canonical
    return ""


def _view_filter_keys(view_ids: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for value in view_ids:
        text = str(value or "").strip()
        if not text:
            continue
        out.add(text)
        canonical = _canonical_view_id(text)
        if canonical:
            out.add(canonical)
    return out


def _row_matches_view_filter(row: Mapping[str, Any], selected_keys: set[str]) -> bool:
    if not selected_keys:
        return True
    row_keys = {
        _string_value(row.get("view_id")),
        _string_value(row.get("synthetic_view_id")),
        _string_value(row.get("entry_id")),
        _string_value(row.get("frame_id")),
        _row_view_id(row),
    }
    row_keys.discard("")
    return bool(row_keys & selected_keys)


def _primary_object_label(row: Mapping[str, Any]) -> str:
    return _safe_text(
        row.get("final_label")
        or row.get("crop_vlm_label")
        or row.get("vlm_label")
        or row.get("label")
        or "unknown"
    ).lower()


def _gt_label(row: Mapping[str, Any]) -> str:
    return _safe_text(row.get("gt_label") or "unknown").lower()


def _text_tokens(*values: Any) -> set[str]:
    joined = " ".join(_safe_text(value).lower() for value in values if value is not None)
    return {token for token in re.findall(r"[a-z0-9]+", joined) if len(token) >= 3}


def _label_similarity(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
    tokens_a = _text_tokens(row_a.get("label"), row_a.get("final_label"), row_a.get("description"))
    tokens_b = _text_tokens(row_b.get("label"), row_b.get("final_label"), row_b.get("description"))
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return float(len(tokens_a & tokens_b) / len(union))


def _semantic_id(row: Mapping[str, Any]) -> Optional[int]:
    value = row.get("gt_semantic_id_top")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _estimated_planar_distance(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> Optional[float]:
    ax = _safe_float(row_a.get("estimated_global_x"))
    az = _safe_float(row_a.get("estimated_global_z"))
    bx = _safe_float(row_b.get("estimated_global_x"))
    bz = _safe_float(row_b.get("estimated_global_z"))
    if ax is None or az is None or bx is None or bz is None:
        return None
    return float(math.hypot(ax - bx, az - bz))


def _semantic_ratio(row: Mapping[str, Any]) -> float:
    return float(_safe_float(row.get("gt_label_pixel_ratio")) or 0.0)


def _is_valid_semantic_row(row: Mapping[str, Any], *, min_gt_label_ratio: float) -> bool:
    if str(row.get("gt_assignment_status") or "") != "ok":
        return False
    if _semantic_id(row) is None:
        return False
    if not _primary_object_label(row):
        return False
    return _semantic_ratio(row) >= float(min_gt_label_ratio)


def _normalized_pair_key(obj_a_id: int, obj_b_id: int) -> Tuple[int, int]:
    a = int(obj_a_id)
    b = int(obj_b_id)
    if a == b:
        raise ValueError("candidate pair cannot reference the same object twice")
    return (a, b) if a < b else (b, a)


def _positive_score(label_similarity: float, planar_distance_m: Optional[float], ratio_floor: float) -> float:
    score = 2.0 + float(label_similarity) + float(ratio_floor)
    if planar_distance_m is not None:
        score += max(0.0, 1.0 - min(float(planar_distance_m), 3.0) / 3.0)
    return float(score)


def _negative_score(label_similarity: float, planar_distance_m: Optional[float]) -> float:
    score = 1.0 + float(label_similarity)
    if planar_distance_m is not None:
        score += min(float(planar_distance_m), 6.0) / 6.0
    return float(score)


def mine_semantic_candidate_pairs(
    *,
    db_dir: str,
    semantic_meta_path: str,
    view_ids: Sequence[str] = (),
    include_same_view: bool = False,
    max_pairs_per_bucket: int = 50,
    min_gt_label_ratio: float = 0.35,
    same_semantic_max_distance_m: float = 1.5,
    stuff_same_semantic_max_distance_m: float = 0.75,
    negative_min_distance_m: float = 1.5,
    output_dir: Optional[str] = None,
) -> List[CandidatePairRecord]:
    rows = _load_jsonl(Path(semantic_meta_path))
    selected_keys = _view_filter_keys(view_ids)

    selected_rows = []
    for row in rows:
        enriched = dict(row)
        enriched["view_id"] = _row_view_id(enriched)
        if not _row_matches_view_filter(enriched, selected_keys):
            continue
        if not _is_valid_semantic_row(enriched, min_gt_label_ratio=float(min_gt_label_ratio)):
            continue
        selected_rows.append(enriched)

    selected_rows.sort(
        key=lambda row: (
            str(row.get("view_id") or ""),
            int(row.get("object_global_id", -1)),
        )
    )
    object_by_id = {int(row["object_global_id"]): row for row in selected_rows}

    raw_candidates: Dict[str, List[Tuple[float, CandidatePairRecord]]] = {
        "same_semantic_id": [],
        "same_label_different_semantic_id": [],
        "same_semantic_stuff_conflict": [],
    }
    seen_pairs: set[Tuple[int, int]] = set()

    def add_candidate(bucket: str, row_a: Mapping[str, Any], row_b: Mapping[str, Any], score: float, suggested_is_same_instance: bool, notes: str) -> None:
        key = _normalized_pair_key(int(row_a["object_global_id"]), int(row_b["object_global_id"]))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        raw_candidates[bucket].append(
            (
                float(score),
                CandidatePairRecord(
                    pair_id=f"semcand_{len(seen_pairs):06d}",
                    db_dir=str(Path(db_dir).name),
                    obj_a_id=int(key[0]),
                    obj_b_id=int(key[1]),
                    bucket=str(bucket),
                    heuristic_score=float(score),
                    suggested_is_same_instance=bool(suggested_is_same_instance),
                    obj_a_label=str(object_by_id[key[0]].get("label") or "unknown"),
                    obj_b_label=str(object_by_id[key[1]].get("label") or "unknown"),
                    obj_a_place_id=str(object_by_id[key[0]].get("view_id") or ""),
                    obj_b_place_id=str(object_by_id[key[1]].get("view_id") or ""),
                    notes=str(notes),
                ),
            )
        )

    for row_a, row_b in itertools.combinations(selected_rows, 2):
        view_a = str(row_a.get("view_id") or "")
        view_b = str(row_b.get("view_id") or "")
        if not include_same_view and view_a == view_b:
            continue

        semantic_a = _semantic_id(row_a)
        semantic_b = _semantic_id(row_b)
        if semantic_a is None or semantic_b is None:
            continue

        label_similarity = _label_similarity(row_a, row_b)
        same_raw_label = _primary_object_label(row_a) == _primary_object_label(row_b)
        same_gt_label = _gt_label(row_a) == _gt_label(row_b)
        same_semantic_id = semantic_a == semantic_b
        planar_distance_m = _estimated_planar_distance(row_a, row_b)
        stuff_like = _gt_label(row_a) in STUFF_GT_LABELS or _gt_label(row_b) in STUFF_GT_LABELS
        ratio_floor = min(_semantic_ratio(row_a), _semantic_ratio(row_b))

        if same_semantic_id:
            max_distance_m = float(stuff_same_semantic_max_distance_m if stuff_like else same_semantic_max_distance_m)
            if (same_raw_label or label_similarity >= 0.5 or same_gt_label) and (
                planar_distance_m is None or planar_distance_m <= max_distance_m
            ):
                add_candidate(
                    bucket="same_semantic_id",
                    row_a=row_a,
                    row_b=row_b,
                    score=_positive_score(label_similarity, planar_distance_m, ratio_floor),
                    suggested_is_same_instance=True,
                    notes=(
                        f"same gt_semantic_id_top={semantic_a}; views {view_a} vs {view_b}; "
                        f"raw_label_same={same_raw_label}; gt_label={_gt_label(row_a)}"
                    ),
                )
                continue
            if stuff_like and label_similarity < 0.5 and not same_raw_label:
                add_candidate(
                    bucket="same_semantic_stuff_conflict",
                    row_a=row_a,
                    row_b=row_b,
                    score=_negative_score(label_similarity, planar_distance_m),
                    suggested_is_same_instance=False,
                    notes=(
                        f"same stuff-like gt_semantic_id_top={semantic_a} but conflicting labels; "
                        f"views {view_a} vs {view_b}; gt_label={_gt_label(row_a)}"
                    ),
                )
                continue

        if not same_semantic_id and (same_raw_label or label_similarity >= 0.5):
            if planar_distance_m is not None and planar_distance_m >= float(negative_min_distance_m):
                add_candidate(
                    bucket="same_label_different_semantic_id",
                    row_a=row_a,
                    row_b=row_b,
                    score=_negative_score(label_similarity, planar_distance_m),
                    suggested_is_same_instance=False,
                    notes=(
                        f"same/similar object label but different gt_semantic_id_top "
                        f"({semantic_a} vs {semantic_b}); views {view_a} vs {view_b}"
                    ),
                )

    mined: List[CandidatePairRecord] = []
    for bucket, ranked in raw_candidates.items():
        sorted_ranked = sorted(ranked, key=lambda item: (-float(item[0]), item[1].pair_id))
        for _score, record in sorted_ranked[: max(0, int(max_pairs_per_bucket))]:
            mined.append(record)

    mined.sort(key=lambda item: (item.bucket, -float(item.heuristic_score), item.pair_id))
    if output_dir:
        export_candidate_artifacts(
            candidates=mined,
            db_dir=db_dir,
            output_dir=output_dir,
            object_by_id=object_by_id,
        )
    return mined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine candidate same-instance object pairs from semantic GT rows.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory containing images referenced by semantic rows")
    parser.add_argument("--semantic_meta_path", type=str, required=True, help="Path to semantic_gt_object_meta.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for pair manifests and copied images")
    parser.add_argument("--view_ids", type=str, default="", help="Optional comma-separated view ids, e.g. 'view_00000,view_00002' or '0,2'")
    parser.add_argument("--include_same_view", action="store_true", help="Allow candidate pairs from the same view")
    parser.add_argument("--max_pairs_per_bucket", type=int, default=50, help="Maximum number of candidates to keep per bucket")
    parser.add_argument("--min_gt_label_ratio", type=float, default=0.35, help="Minimum dominant semantic label ratio required to keep an object row")
    parser.add_argument("--same_semantic_max_distance_m", type=float, default=1.5, help="Max planar distance for positive same-semantic candidates")
    parser.add_argument("--stuff_same_semantic_max_distance_m", type=float, default=0.75, help="Stricter max planar distance when the GT semantic label is stuff-like")
    parser.add_argument("--negative_min_distance_m", type=float, default=1.5, help="Minimum planar distance for hard negative same-label pairs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidates = mine_semantic_candidate_pairs(
        db_dir=args.db_dir,
        semantic_meta_path=args.semantic_meta_path,
        view_ids=_parse_view_ids(args.view_ids),
        include_same_view=bool(args.include_same_view),
        max_pairs_per_bucket=args.max_pairs_per_bucket,
        min_gt_label_ratio=args.min_gt_label_ratio,
        same_semantic_max_distance_m=args.same_semantic_max_distance_m,
        stuff_same_semantic_max_distance_m=args.stuff_same_semantic_max_distance_m,
        negative_min_distance_m=args.negative_min_distance_m,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "num_candidates": len(candidates),
                "selected_view_ids": _parse_view_ids(args.view_ids),
                "buckets": sorted({record.bucket for record in candidates}),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()