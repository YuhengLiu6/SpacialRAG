import json

from spatial_rag.object_relation_builder import build_object_relations


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def test_build_object_relations_from_existing_metadata(tmp_path):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    _write_jsonl(
        db_dir / "meta.jsonl",
        [
            {"id": 0, "x": 0.0, "y": 0.0, "world_position": [0.0, 1.6, 0.0], "orientation": 0, "file_name": "images/view_00000.jpg"},
            {"id": 1, "x": 5.0, "y": 5.0, "world_position": [5.0, 1.6, 5.0], "orientation": 90, "file_name": "images/view_00001.jpg"},
        ],
    )
    _write_jsonl(
        db_dir / "object_meta.jsonl",
        [
            {"entry_id": 0, "object_global_id": 1, "label": "chair", "estimated_global_x": 0.0, "estimated_global_y": 0.8, "estimated_global_z": -1.0},
            {"entry_id": 0, "object_global_id": 2, "label": "table", "estimated_global_x": 1.0, "estimated_global_y": 1.2, "estimated_global_z": -1.0},
            {"entry_id": 1, "object_global_id": 3, "label": "lamp", "estimated_global_x": 6.0, "estimated_global_y": 1.6, "estimated_global_z": 5.0},
        ],
    )
    (db_dir / "build_report.json").write_text(json.dumps({"started_at": "t0"}, ensure_ascii=True), encoding="utf-8")

    result = build_object_relations(str(db_dir))

    assert result["num_view_object_relations"] == 3
    assert result["num_object_object_relations"] == 2
    assert (db_dir / "view_object_relations.jsonl").exists()
    assert (db_dir / "object_object_relations.jsonl").exists()
    view_rows = [json.loads(line) for line in (db_dir / "view_object_relations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    object_rows = [json.loads(line) for line in (db_dir / "object_object_relations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert view_rows[0]["vertical_direction"] == "below"
    assert "distance_3d_m" in view_rows[0]
    assert object_rows[0]["vertical_direction"] == "above"
    assert "distance_3d_m" in object_rows[0]
    updated_report = json.loads((db_dir / "build_report.json").read_text(encoding="utf-8"))
    assert updated_report["total_view_object_relations"] == 3
    assert updated_report["total_object_object_relations"] == 2
