import json

from spatial_rag.object_instance_pair_mining import export_candidate_artifacts, mine_candidate_pairs


def _meta_row(entry_id, x, z, orientation, room_function="resting", view_type="living room"):
    return {
        "id": entry_id,
        "x": float(x),
        "y": float(z),
        "world_position": [float(x), 0.0, float(z)],
        "orientation": int(orientation),
        "file_name": f"images/view_{entry_id:05d}.jpg",
        "parse_status": "ok",
        "frame_text_short": f"short {entry_id}",
        "frame_text_long": f"long {entry_id}",
        "room_function": room_function,
        "view_type": view_type,
    }


def _object_row(object_global_id, entry_id, label, description, orientation, distance):
    return {
        "object_global_id": object_global_id,
        "entry_id": entry_id,
        "file_name": f"images/view_{entry_id:05d}.jpg",
        "label": label,
        "description": description,
        "long_form_open_description": f"{description} long",
        "laterality": "center",
        "distance_bin": "middle",
        "verticality": "middle",
        "distance_from_camera_m": distance,
        "object_orientation_deg": orientation,
        "location_relative_to_other_objects": "",
        "parse_status": "ok",
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def _make_candidate_db(tmp_path):
    db_dir = tmp_path / "toy_db"
    image_dir = db_dir / "images"
    image_dir.mkdir(parents=True)
    for entry_id in range(5):
        (image_dir / f"view_{entry_id:05d}.jpg").write_bytes(f"img-{entry_id}".encode("utf-8"))

    meta_rows = [
        _meta_row(0, 0.0, 0.0, 0),
        _meta_row(1, 0.0, 0.0, 90),
        _meta_row(2, 0.0, 3.0, 0, room_function="bedroom", view_type="bedroom"),
        _meta_row(3, 0.0, 3.0, 90, room_function="bedroom", view_type="bedroom"),
        _meta_row(4, 3.0, 0.0, 0, room_function="hallway", view_type="hallway"),
    ]
    object_rows = [
        _object_row(10, 0, "chair", "red chair near table", 0.0, 1.0),
        _object_row(20, 1, "chair", "red chair side view", 45.0, 1.1),
        _object_row(30, 2, "chair", "chair in mirror reflection", 180.0, 2.5),
        _object_row(40, 0, "table", "small wooden table", 20.0, 1.2),
        _object_row(50, 2, "lamp", "floor lamp by bed", 200.0, 2.0),
        _object_row(60, 4, "chair", "blue chair in hallway", 10.0, 1.6),
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)
    (db_dir / "build_report.json").write_text(
        json.dumps({"scan_angles": [0, 90], "random_config": {"scan_angles": [0, 90]}}),
        encoding="utf-8",
    )
    return db_dir


def test_mine_candidate_pairs_returns_multiple_buckets_and_exports_images(tmp_path):
    db_dir = _make_candidate_db(tmp_path)
    output_dir = tmp_path / "candidates"

    candidates = mine_candidate_pairs(
        db_dir=str(db_dir),
        max_pairs_per_bucket=5,
        output_dir=str(output_dir),
    )

    buckets = {record.bucket for record in candidates}
    assert "same_label_same_place" in buckets
    assert "same_label_adjacent_place" in buckets
    assert "different_label_same_place" in buckets
    assert (output_dir / "candidates.jsonl").exists()
    sample_pair = output_dir / "pairs" / candidates[0].pair_id
    assert (sample_pair / "pair_manifest.json").exists()
    copied_images = list(sample_pair.glob("*.jpg"))
    assert copied_images


def test_export_candidate_artifacts_uses_object_meta_file_names_when_object_map_missing(tmp_path):
    db_dir = _make_candidate_db(tmp_path)
    output_dir = tmp_path / "exported"

    candidates = mine_candidate_pairs(db_dir=str(db_dir), max_pairs_per_bucket=1, output_dir=None)
    exported = export_candidate_artifacts(candidates[:1], db_dir=str(db_dir), output_dir=str(output_dir))

    pair_dir = output_dir / "pairs" / candidates[0].pair_id
    manifest = json.loads((pair_dir / "pair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["object_a"]["file_name"].startswith("images/")
    assert len(list(pair_dir.glob("*.jpg"))) >= 1
    assert exported["num_candidates"] == 1
