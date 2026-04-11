import csv
import json

import cv2
import numpy as np

from spatial_rag.export_object_crops_by_global_id import main


def test_export_object_crops_cli_writes_default_output_dir(tmp_path, monkeypatch, capsys):
    db_root = tmp_path / "db"
    crop_path = db_root / "geometry" / "view_00000" / "objects" / "obj_000_crop.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(crop_path), np.full((12, 16, 3), 200, dtype=np.uint8))
    object_row = {
        "object_global_id": 3,
        "occlusion_level": "heavily occluded",
        "entry_id": 0,
        "frame_id": 0,
        "file_name": "images/frame.jpg",
        "label": "chair",
        "geometry_source": "mask_depth",
        "crop_path": str(crop_path),
    }
    (db_root / "object_meta.jsonl").write_text(json.dumps(object_row, ensure_ascii=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_object_crops_by_global_id.py",
            "--db_dir",
            str(db_root),
        ],
    )

    main()

    out = json.loads(capsys.readouterr().out)
    export_dir = db_root / "object_crops_by_global_id"
    assert out["exported_count"] == 1
    assert (export_dir / "3_chair_heavily_occluded.jpg").exists()
    manifest_rows = list(csv.DictReader((export_dir / "manifest.csv").open()))
    assert manifest_rows[0]["object_global_id"] == "3"
    assert manifest_rows[0]["occlusion_level"] == "heavily occluded"
