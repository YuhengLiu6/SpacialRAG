import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return text.strip("_") or "unknown_object"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def add(self, item: int) -> None:
        value = int(item)
        self.parent.setdefault(value, value)

    def find(self, item: int) -> int:
        value = int(item)
        root = self.parent.setdefault(value, value)
        if root != value:
            root = self.find(root)
            self.parent[value] = root
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _group_positive_pairs(candidate_rows: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    uf = _UnionFind()
    positive_rows = [row for row in candidate_rows if bool(row.get("suggested_is_same_instance"))]
    for row in positive_rows:
        a = int(row["obj_a_id"])
        b = int(row["obj_b_id"])
        uf.add(a)
        uf.add(b)
        uf.union(a, b)
    groups: Dict[int, List[int]] = {}
    for obj_id in sorted(uf.parent):
        groups.setdefault(uf.find(obj_id), []).append(obj_id)
    return [sorted(group) for group in groups.values() if len(group) >= 2]


def export_pipeline_same_object_groups(
    db_dir: str,
    candidates_jsonl: str,
    output_dir: str,
) -> Dict[str, Any]:
    db_path = Path(db_dir)
    candidate_rows = _load_jsonl(Path(candidates_jsonl))
    object_rows = _load_jsonl(db_path / "object_meta.jsonl")
    object_by_id = {int(row["object_global_id"]): dict(row) for row in object_rows}
    groups = _group_positive_pairs(candidate_rows)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    copied_images = 0
    exported_groups = 0

    for group_index, obj_ids in enumerate(groups):
        rows = [object_by_id[obj_id] for obj_id in obj_ids if obj_id in object_by_id]
        if len(rows) < 2:
            continue
        label_counts = Counter(str(row.get("label") or "unknown") for row in rows)
        label = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        group_dir = root / f"{_safe_slug(label)}_{group_index:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "group_id": group_dir.name,
            "label": label,
            "object_ids": obj_ids,
            "num_objects": len(obj_ids),
            "members": rows,
        }
        (group_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

        seen_sources = set()
        for member_index, row in enumerate(rows):
            file_name = str(row.get("file_name") or "").strip()
            if not file_name:
                continue
            src = Path(file_name)
            if not src.is_absolute():
                src = db_path / src
            if not src.exists():
                continue
            if str(src) in seen_sources:
                continue
            seen_sources.add(str(src))
            dst = group_dir / f"{member_index:03d}_{src.name}"
            shutil.copy2(src, dst)
            copied_images += 1

        exported_groups += 1

    summary = {
        "output_dir": str(root),
        "num_groups": exported_groups,
        "num_positive_pairs": sum(1 for row in candidate_rows if bool(row.get("suggested_is_same_instance"))),
        "copied_images": copied_images,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pipeline-suggested same-object groups into image folders.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory containing object_meta.jsonl and images")
    parser.add_argument("--candidates_jsonl", type=str, required=True, help="Candidates JSONL produced by object_instance_pair_mining")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder where grouped images should be exported")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = export_pipeline_same_object_groups(
        db_dir=args.db_dir,
        candidates_jsonl=args.candidates_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
