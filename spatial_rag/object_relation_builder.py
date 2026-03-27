import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from spatial_rag.spatial_db_builder import (
    _build_object_object_relations,
    _build_view_object_relations,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_object_relations(db_dir: str) -> Dict[str, Any]:
    db_path = Path(db_dir)
    meta_path = db_path / "meta.jsonl"
    object_meta_path = db_path / "object_meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.jsonl: {meta_path}")
    if not object_meta_path.exists():
        raise FileNotFoundError(f"Missing object_meta.jsonl: {object_meta_path}")

    metadata_records = _load_jsonl(meta_path)
    object_metadata_records = _load_jsonl(object_meta_path)
    view_object_relations = _build_view_object_relations(
        metadata_records=metadata_records,
        object_metadata_records=object_metadata_records,
    )
    object_object_relations = _build_object_object_relations(
        metadata_records=metadata_records,
        object_metadata_records=object_metadata_records,
    )

    view_object_path = db_path / "view_object_relations.jsonl"
    object_object_path = db_path / "object_object_relations.jsonl"
    _write_jsonl(view_object_path, view_object_relations)
    _write_jsonl(object_object_path, object_object_relations)

    build_report_path = db_path / "build_report.json"
    if build_report_path.exists():
        report = json.loads(build_report_path.read_text(encoding="utf-8"))
        report["total_view_object_relations"] = len(view_object_relations)
        report["total_object_object_relations"] = len(object_object_relations)
        build_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "db_dir": str(db_path),
        "view_object_relations_path": str(view_object_path),
        "object_object_relations_path": str(object_object_path),
        "num_view_object_relations": len(view_object_relations),
        "num_object_object_relations": len(object_object_relations),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build view-object and object-object relation files from existing spatial DB metadata."
    )
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory containing meta.jsonl/object_meta.jsonl")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_object_relations(args.db_dir)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
