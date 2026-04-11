import csv
import json

from spatial_rag.export_object_occlusion_levels import main


def test_export_object_occlusion_levels_cli_writes_default_output_csv(tmp_path, monkeypatch, capsys):
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"object_global_id": 3, "occlusion_level": "slightly occluded"},
        {"object_global_id": 4},
    ]
    (db_root / "object_meta.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_object_occlusion_levels.py",
            "--db_dir",
            str(db_root),
        ],
    )

    main()

    out = json.loads(capsys.readouterr().out)
    csv_path = db_root / "object_occlusion_levels.csv"
    assert out["row_count"] == 2
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))

    assert exported == [
        {"object_global_id": "3", "occlusion_level": "slightly occluded"},
        {"object_global_id": "4", "occlusion_level": "uncertain"},
    ]
