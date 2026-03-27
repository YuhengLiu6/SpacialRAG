import argparse
import itertools
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from spatial_rag.graph_builder import build_graph_payload


TRICKY_KEYWORDS = ("reflection", "background", "doorway", "adjacent", "visible through", "mirror", "tv reflection")


@dataclass(frozen=True)
class CandidatePairRecord:
    pair_id: str
    db_dir: str
    obj_a_id: int
    obj_b_id: int
    bucket: str
    heuristic_score: float
    suggested_is_same_instance: bool
    obj_a_label: str
    obj_b_label: str
    obj_a_place_id: str
    obj_b_place_id: str
    notes: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _text_tokens(*values: Any) -> set:
    joined = " ".join(_safe_text(value).lower() for value in values if value is not None)
    return {token for token in re.findall(r"[a-z0-9]+", joined) if len(token) >= 3}


def _token_jaccard(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
    tokens_a = _text_tokens(row_a.get("label"), row_a.get("description"), row_a.get("long_form_open_description"))
    tokens_b = _text_tokens(row_b.get("label"), row_b.get("description"), row_b.get("long_form_open_description"))
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return float(len(tokens_a & tokens_b) / len(union))


def _projected_distance(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> Optional[float]:
    ax = row_a.get("projected_x")
    az = row_a.get("projected_z")
    bx = row_b.get("projected_x")
    bz = row_b.get("projected_z")
    if ax is None or az is None or bx is None or bz is None:
        return None
    dx = float(ax) - float(bx)
    dz = float(az) - float(bz)
    return float(math.hypot(dx, dz))


def _is_tricky(row: Mapping[str, Any]) -> bool:
    combined = f"{_safe_text(row.get('description'))} {_safe_text(row.get('long_form_open_description'))}".lower()
    return any(keyword in combined for keyword in TRICKY_KEYWORDS)


def _normalized_pair_key(obj_a_id: int, obj_b_id: int) -> Tuple[int, int]:
    a = int(obj_a_id)
    b = int(obj_b_id)
    if a == b:
        raise ValueError("candidate pair cannot reference the same object twice")
    return (a, b) if a < b else (b, a)


def _adjacent_place_map(direction_edges: Sequence[Mapping[str, Any]]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for edge in direction_edges:
        src = str(edge["source_place_id"])
        dst = str(edge["target_place_id"])
        out.setdefault(src, set()).add(dst)
    return out


def _pair_manifest(record: CandidatePairRecord, row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate": asdict(record),
        "object_a": dict(row_a),
        "object_b": dict(row_b),
        "generated_at": _now_iso(),
    }


def _candidate_score_same_place(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
    score = 1.5 + _token_jaccard(row_a, row_b)
    if str(row_a.get("view_id")) != str(row_b.get("view_id")):
        score += 0.25
    projected = _projected_distance(row_a, row_b)
    if projected is not None:
        score += max(0.0, 1.0 - min(projected, 4.0) / 4.0)
    return float(score)


def _candidate_score_adjacent(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
    score = 1.0 + _token_jaccard(row_a, row_b)
    projected = _projected_distance(row_a, row_b)
    if projected is not None:
        score += max(0.0, 1.0 - min(projected, 6.0) / 6.0)
    return float(score)


def _candidate_score_same_label_distant(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
    score = _token_jaccard(row_a, row_b)
    projected = _projected_distance(row_a, row_b)
    if projected is not None:
        score += max(0.0, 1.0 - min(projected, 10.0) / 10.0)
    return float(score)


def _candidate_score_diff_label_same_place(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
    score = 0.25
    if str(row_a.get("distance_bin")) == str(row_b.get("distance_bin")):
        score += 0.5
    if str(row_a.get("laterality")) == str(row_b.get("laterality")):
        score += 0.25
    if str(row_a.get("verticality")) == str(row_b.get("verticality")):
        score += 0.25
    return float(score)


def _candidate_score_tricky(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
    score = 0.5 + _token_jaccard(row_a, row_b)
    if _is_tricky(row_a) or _is_tricky(row_b):
        score += 1.0
    return float(score)


def _append_ranked_candidates(
    output: List[CandidatePairRecord],
    ranked: Sequence[Tuple[float, Dict[str, Any]]],
    max_count: int,
) -> None:
    for score, payload in ranked[: max(0, int(max_count))]:
        output.append(
            CandidatePairRecord(
                pair_id=str(payload["pair_id"]),
                db_dir=str(payload["db_dir"]),
                obj_a_id=int(payload["obj_a_id"]),
                obj_b_id=int(payload["obj_b_id"]),
                bucket=str(payload["bucket"]),
                heuristic_score=float(score),
                suggested_is_same_instance=bool(payload["suggested_is_same_instance"]),
                obj_a_label=str(payload["obj_a_label"]),
                obj_b_label=str(payload["obj_b_label"]),
                obj_a_place_id=str(payload["obj_a_place_id"]),
                obj_b_place_id=str(payload["obj_b_place_id"]),
                notes=str(payload.get("notes") or ""),
            )
        )


def mine_candidate_pairs(
    db_dir: str,
    max_pairs_per_bucket: int = 50,
    output_dir: Optional[str] = None,
) -> List[CandidatePairRecord]:
    payload = build_graph_payload(db_dir)
    raw_meta = _load_jsonl(Path(db_dir) / "object_meta.jsonl")
    raw_by_id = {int(row["object_global_id"]): dict(row) for row in raw_meta}
    objects = []
    for row in payload.get("objects", []):
        object_row = dict(row)
        object_row.update(raw_by_id.get(int(object_row["object_global_id"]), {}))
        objects.append(object_row)
    objects.sort(key=lambda row: int(row["object_global_id"]))
    object_by_id = {int(row["object_global_id"]): row for row in objects}
    place_to_objects: Dict[str, List[Dict[str, Any]]] = {}
    label_to_objects: Dict[str, List[Dict[str, Any]]] = {}
    for row in objects:
        place_to_objects.setdefault(str(row["place_id"]), []).append(row)
        label_to_objects.setdefault(str(row["label"]), []).append(row)
    for rows in place_to_objects.values():
        rows.sort(key=lambda row: (str(row.get("view_id") or ""), int(row["object_global_id"])))
    adjacency = _adjacent_place_map(payload.get("direction_edges", []))

    seen_pairs = set()
    raw_candidates: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {
        "same_label_same_place": [],
        "same_label_adjacent_place": [],
        "same_label_distant_place": [],
        "different_label_same_place": [],
        "reflection_background_doorway": [],
    }

    def add_candidate(
        *,
        bucket: str,
        row_a: Mapping[str, Any],
        row_b: Mapping[str, Any],
        score: float,
        suggested_is_same_instance: bool,
        notes: str = "",
    ) -> None:
        key = _normalized_pair_key(row_a["object_global_id"], row_b["object_global_id"])
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        raw_candidates[bucket].append(
            (
                float(score),
                {
                    "pair_id": f"cand_{len(seen_pairs):06d}",
                    "db_dir": str(Path(db_dir).name),
                    "obj_a_id": int(key[0]),
                    "obj_b_id": int(key[1]),
                    "bucket": bucket,
                    "suggested_is_same_instance": bool(suggested_is_same_instance),
                    "obj_a_label": str(object_by_id[key[0]].get("label") or "unknown"),
                    "obj_b_label": str(object_by_id[key[1]].get("label") or "unknown"),
                    "obj_a_place_id": str(object_by_id[key[0]].get("place_id") or ""),
                    "obj_b_place_id": str(object_by_id[key[1]].get("place_id") or ""),
                    "notes": notes,
                },
            )
        )

    for place_id, rows in place_to_objects.items():
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_label.setdefault(str(row["label"]), []).append(row)
        for label, label_rows in by_label.items():
            for row_a, row_b in itertools.combinations(label_rows, 2):
                add_candidate(
                    bucket="same_label_same_place",
                    row_a=row_a,
                    row_b=row_b,
                    score=_candidate_score_same_place(row_a, row_b),
                    suggested_is_same_instance=True,
                    notes=f"same label '{label}' in same place {place_id}",
                )
        for row_a, row_b in itertools.combinations(rows, 2):
            if str(row_a.get("label")) == str(row_b.get("label")):
                continue
            add_candidate(
                bucket="different_label_same_place",
                row_a=row_a,
                row_b=row_b,
                score=_candidate_score_diff_label_same_place(row_a, row_b),
                suggested_is_same_instance=False,
                notes=f"different labels in same place {place_id}",
            )

    for label, rows in label_to_objects.items():
        for row_a, row_b in itertools.combinations(rows, 2):
            place_a = str(row_a["place_id"])
            place_b = str(row_b["place_id"])
            if place_a == place_b:
                continue
            if place_b in adjacency.get(place_a, set()) or place_a in adjacency.get(place_b, set()):
                add_candidate(
                    bucket="same_label_adjacent_place",
                    row_a=row_a,
                    row_b=row_b,
                    score=_candidate_score_adjacent(row_a, row_b),
                    suggested_is_same_instance=True,
                    notes=f"same label '{label}' across adjacent places",
                )
            else:
                add_candidate(
                    bucket="same_label_distant_place",
                    row_a=row_a,
                    row_b=row_b,
                    score=_candidate_score_same_label_distant(row_a, row_b),
                    suggested_is_same_instance=False,
                    notes=f"same label '{label}' across non-adjacent places",
                )

    tricky_rows = [row for row in objects if _is_tricky(row)]
    for row_a in tricky_rows:
        for row_b in label_to_objects.get(str(row_a["label"]), []):
            if int(row_a["object_global_id"]) == int(row_b["object_global_id"]):
                continue
            add_candidate(
                bucket="reflection_background_doorway",
                row_a=row_a,
                row_b=row_b,
                score=_candidate_score_tricky(row_a, row_b),
                suggested_is_same_instance=False,
                notes="contains tricky reflection/background/doorway wording",
            )

    mined: List[CandidatePairRecord] = []
    for bucket, ranked in raw_candidates.items():
        sorted_ranked = sorted(ranked, key=lambda item: (-float(item[0]), item[1]["pair_id"]))
        _append_ranked_candidates(mined, sorted_ranked, max_count=max_pairs_per_bucket)

    mined.sort(key=lambda item: (item.bucket, -float(item.heuristic_score), item.pair_id))

    if output_dir:
        export_candidate_artifacts(
            candidates=mined,
            db_dir=db_dir,
            output_dir=output_dir,
            object_by_id=object_by_id,
        )

    return mined


def export_candidate_artifacts(
    candidates: Sequence[CandidatePairRecord],
    db_dir: str,
    output_dir: str,
    object_by_id: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pairs_root = root / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)

    if object_by_id is None:
        raw_meta = _load_jsonl(Path(db_dir) / "object_meta.jsonl")
        object_by_id = {int(row["object_global_id"]): dict(row) for row in raw_meta}

    copied_images = 0
    for record in candidates:
        row_a = dict(object_by_id[int(record.obj_a_id)])
        row_b = dict(object_by_id[int(record.obj_b_id)])
        pair_dir = pairs_root / record.pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        manifest = _pair_manifest(record, row_a=row_a, row_b=row_b)
        (pair_dir / "pair_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
        for prefix, row in (("a", row_a), ("b", row_b)):
            file_name = str(row.get("file_name") or "").strip()
            if not file_name:
                continue
            src = Path(file_name)
            if not src.is_absolute():
                src = Path(db_dir) / src
            if not src.exists():
                continue
            dst = pair_dir / f"{prefix}_{src.name}"
            shutil.copy2(src, dst)
            copied_images += 1

    with (root / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for record in candidates:
            handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    instructions = [
        "# Object Instance Pair Annotation Candidates",
        "",
        f"Generated at: `{_now_iso()}`",
        f"DB: `{db_dir}`",
        "",
        "Each folder in `pairs/` contains:",
        "- `pair_manifest.json`",
        "- source image for object A",
        "- source image for object B",
        "",
        "Create manual labels in `evaluation/object_instance_pairs.jsonl` using:",
        "",
        "```json",
        '{"pair_id":"cand_000001","db_dir":"spatial_db_split_k8s","obj_a_id":1,"obj_b_id":2,"is_same_instance":true,"split":"dev","notes":"same chair"}',
        "```",
        "",
        "Suggested labels in `candidates.jsonl` are heuristics only and must be verified by a human annotator.",
    ]
    (root / "README.md").write_text("\n".join(instructions) + "\n", encoding="utf-8")
    return {
        "output_dir": str(root),
        "num_candidates": len(candidates),
        "copied_images": int(copied_images),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine candidate same-instance object pairs for manual annotation.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory containing object_meta.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for pair manifests and copied images")
    parser.add_argument("--max_pairs_per_bucket", type=int, default=50, help="Maximum number of candidates to keep per heuristic bucket")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidates = mine_candidate_pairs(
        db_dir=args.db_dir,
        max_pairs_per_bucket=args.max_pairs_per_bucket,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "num_candidates": len(candidates),
                "buckets": sorted({record.bucket for record in candidates}),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
