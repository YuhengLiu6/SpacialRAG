import json

from spatial_rag.export_pipeline_same_object_groups import export_pipeline_same_object_groups


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def test_export_pipeline_same_object_groups_copies_images_into_group_dirs(tmp_path):
    db_dir = tmp_path / "db"
    image_dir = db_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "view_00000.jpg").write_bytes(b"a")
    (image_dir / "view_00001.jpg").write_bytes(b"b")
    (image_dir / "view_00002.jpg").write_bytes(b"c")

    _write_jsonl(
        db_dir / "object_meta.jsonl",
        [
            {"object_global_id": 10, "label": "chair", "file_name": "images/view_00000.jpg"},
            {"object_global_id": 20, "label": "chair", "file_name": "images/view_00001.jpg"},
            {"object_global_id": 30, "label": "lamp", "file_name": "images/view_00002.jpg"},
        ],
    )
    _write_jsonl(
        tmp_path / "candidates.jsonl",
        [
            {"obj_a_id": 10, "obj_b_id": 20, "suggested_is_same_instance": True},
            {"obj_a_id": 20, "obj_b_id": 30, "suggested_is_same_instance": False},
        ],
    )

    output_dir = tmp_path / "test_groups"
    summary = export_pipeline_same_object_groups(
        db_dir=str(db_dir),
        candidates_jsonl=str(tmp_path / "candidates.jsonl"),
        output_dir=str(output_dir),
    )

    assert summary["num_groups"] == 1
    group_dir = output_dir / "chair_000"
    assert (group_dir / "manifest.json").exists()
    assert (group_dir / "000_view_00000.jpg").exists()
    assert (group_dir / "001_view_00001.jpg").exists()
