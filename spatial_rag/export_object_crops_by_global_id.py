from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from spatial_rag.spatial_db_builder import export_object_crops_by_global_id


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export one crop image per object, named by object_global_id, label, and occlusion_level.",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        required=True,
        help="Spatial DB directory containing object_meta.jsonl and source images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Destination directory for exported crops. Default: <db_dir>/object_crops_by_global_id",
    )
    args = parser.parse_args(argv)

    db_root = Path(args.db_dir).expanduser().resolve()
    object_meta_path = db_root / "object_meta.jsonl"
    if not object_meta_path.exists():
        raise FileNotFoundError(f"Missing object_meta.jsonl: {object_meta_path}")

    object_rows = _load_jsonl(object_meta_path)
    summary = export_object_crops_by_global_id(
        db_root=db_root,
        object_rows=object_rows,
        output_dir=None if args.output_dir is None else Path(args.output_dir).expanduser().resolve(),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
