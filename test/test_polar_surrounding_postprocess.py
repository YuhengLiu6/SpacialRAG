import json

from spatial_rag.polar_surrounding_postprocess import (
    build_polar_surroundings,
    classify_local_semantic_relation,
    pair_distance_from_polar,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def test_pair_distance_from_polar_matches_cosine_law_example():
    distance = pair_distance_from_polar(1.0, 1.1, 10.0)
    assert abs(distance - 0.20838172514196715) < 1e-6


def test_classify_local_semantic_relation_handles_lateral_depth_and_height():
    assert classify_local_semantic_relation(10.0, 0.1, None) == "slightly right, slightly behind"
    assert classify_local_semantic_relation(-25.0, -1.0, None) == "left, in front"
    assert classify_local_semantic_relation(0.0, 0.0, 0.4) == "above"


def test_build_polar_surroundings_writes_new_files_and_preserves_order(tmp_path):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    meta_rows = [
        {
            "id": 0,
            "orientation": 0,
            "file_name": "images/view_00000.jpg",
        }
    ]
    object_rows = [
        {
            "entry_id": 0,
            "view_id": "view_00000",
            "object_global_id": 10,
            "label": "laptop",
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": 10.0,
            "relative_height_from_camera_m": 0.2,
            "estimated_global_x": 0.0,
            "estimated_global_y": 0.2,
            "estimated_global_z": 0.0,
            "surrounding_context": [],
            "location_relative_to_other_objects": "",
        },
        {
            "entry_id": 0,
            "view_id": "view_00000",
            "object_global_id": 11,
            "label": "mug",
            "distance_from_camera_m": 1.1,
            "relative_bearing_deg": 20.0,
            "relative_height_from_camera_m": 0.3,
            "estimated_global_x": 1.0,
            "estimated_global_y": 0.3,
            "estimated_global_z": 0.0,
            "surrounding_context": [],
            "location_relative_to_other_objects": "",
        },
        {
            "entry_id": 0,
            "view_id": "view_00000",
            "object_global_id": 12,
            "label": "plant",
            "distance_from_camera_m": 1.8,
            "relative_bearing_deg": -15.0,
            "relative_height_from_camera_m": -0.1,
            "estimated_global_x": 0.0,
            "estimated_global_y": -0.1,
            "estimated_global_z": -1.0,
            "surrounding_context": [],
            "location_relative_to_other_objects": "",
        },
        {
            "entry_id": 0,
            "view_id": "view_00000",
            "object_global_id": 13,
            "label": "other",
            "distance_from_camera_m": 0.8,
            "relative_bearing_deg": 0.0,
            "estimated_global_x": 5.0,
            "estimated_global_y": 0.0,
            "estimated_global_z": 5.0,
        },
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)
    original_contents = (db_dir / "object_meta.jsonl").read_text(encoding="utf-8")

    result = build_polar_surroundings(str(db_dir), max_neighbors=1)

    assert result["num_objects"] == 4
    assert result["num_pair_relations"] >= 2
    assert (db_dir / "object_polar_relations.jsonl").exists()
    assert (db_dir / "object_meta_with_polar_surroundings.jsonl").exists()
    assert (db_dir / "object_meta.jsonl").read_text(encoding="utf-8") == original_contents

    updated_rows = [
        json.loads(line)
        for line in (db_dir / "object_meta_with_polar_surroundings.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["object_global_id"] for row in updated_rows] == [10, 11, 12, 13]

    laptop = updated_rows[0]
    assert laptop["surrounding_source"] == "polar_postprocess_v1"
    assert len(laptop["surrounding_context"]) == 1
    assert laptop["surrounding_context"][0]["target_object_global_id"] == 11
    assert laptop["surrounding_context"][0]["allocentric_direction_8"] == "E"
    assert "mug@" in laptop["location_relative_to_other_objects"]
    assert "E" in laptop["location_relative_to_other_objects"]

    other = updated_rows[3]
    assert other["surrounding_context"] == []
    assert other["location_relative_to_other_objects"] == ""
