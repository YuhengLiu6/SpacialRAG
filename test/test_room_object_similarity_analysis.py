import json

import numpy as np

from spatial_rag.room_object_similarity_analysis import (
    analyze_rooms,
    compute_fused_similarity_matrix,
    compute_geometry_similarity_matrix,
    compute_similarity_matrix,
    group_semantic_rooms,
    sample_room_objects,
    select_simple_and_complex_rooms,
)


def _meta_row(entry_id, x, z, orientation, room_function, view_type):
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


def _object_row(object_global_id, entry_id, label, orientation, distance, desc=None):
    return {
        "object_global_id": int(object_global_id),
        "entry_id": int(entry_id),
        "view_id": f"view_{entry_id:05d}",
        "file_name": f"images/view_{entry_id:05d}.jpg",
        "label": label,
        "description": desc or f"{label} description",
        "long_form_open_description": f"{label} long description",
        "laterality": "center",
        "distance_bin": "middle",
        "verticality": "middle",
        "distance_from_camera_m": float(distance),
        "object_orientation_deg": float(orientation),
        "estimated_global_x": float(object_global_id) / 10.0,
        "estimated_global_y": 0.0 if int(object_global_id) % 2 == 0 else None,
        "estimated_global_z": -float(object_global_id) / 10.0,
        "location_relative_to_other_objects": "",
        "parse_status": "ok",
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def _make_room_db(tmp_path):
    db_dir = tmp_path / "room_db"
    image_dir = db_dir / "images"
    image_dir.mkdir(parents=True)
    for entry_id in range(8):
        (image_dir / f"view_{entry_id:05d}.jpg").write_bytes(f"img-{entry_id}".encode("utf-8"))

    meta_rows = [
        _meta_row(0, 0.0, 0.0, 0, "resting", "living room"),
        _meta_row(1, 0.0, 0.0, 90, "resting", "living room"),
        _meta_row(2, 2.0, 0.0, 0, "resting", "living room"),
        _meta_row(3, 2.0, 0.0, 90, "resting", "living room"),
        _meta_row(4, 0.0, 3.0, 0, "sleeping", "bedroom"),
        _meta_row(5, 0.0, 3.0, 90, "sleeping", "bedroom"),
        _meta_row(6, 2.0, 3.0, 0, "sleeping", "bedroom"),
        _meta_row(7, 2.0, 3.0, 90, "sleeping", "bedroom"),
    ]
    object_rows = [
        _object_row(10, 0, "chair", 0.0, 1.0, "blue chair near sofa"),
        _object_row(11, 1, "chair", 10.0, 1.1, "second blue chair"),
        _object_row(12, 2, "table", 20.0, 1.2, "small round table"),
        _object_row(13, 3, "table", 25.0, 1.3, "second small table"),
        _object_row(14, 2, "lamp", 40.0, 1.5, "floor lamp"),
        _object_row(20, 4, "bed", 180.0, 1.0, "king bed"),
        _object_row(21, 5, "dresser", 200.0, 1.2, "wood dresser"),
        _object_row(22, 6, "mirror", 225.0, 1.4, "wall mirror"),
        _object_row(23, 7, "plant", 240.0, 1.6, "small green plant"),
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)
    (db_dir / "build_report.json").write_text(
        json.dumps({"scan_angles": [0, 90], "random_config": {"scan_angles": [0, 90]}}),
        encoding="utf-8",
    )

    short_emb = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.02, 0.98],
            [-1.0, 0.0],
            [0.4, 0.8],
            [-0.5, 0.8],
            [-0.8, -0.2],
            [0.2, -0.9],
        ],
        dtype=np.float32,
    )
    long_emb = np.asarray(
        [
            [1.0, 0.0],
            [0.97, 0.03],
            [0.05, 0.95],
            [0.08, 0.92],
            [-0.9, 0.1],
            [0.35, 0.82],
            [-0.45, 0.85],
            [-0.85, -0.15],
            [0.25, -0.88],
        ],
        dtype=np.float32,
    )
    np.save(db_dir / "object_text_emb_short.npy", short_emb)
    np.save(db_dir / "object_text_emb_long.npy", long_emb)
    return db_dir


def test_group_semantic_rooms_and_auto_selection(tmp_path):
    db_dir = _make_room_db(tmp_path)
    with (db_dir / "object_meta.jsonl").open("r", encoding="utf-8") as handle:
        raw_object_rows = [json.loads(line) for line in handle if line.strip()]

    from spatial_rag.graph_builder import build_graph_payload

    payload = build_graph_payload(str(db_dir))
    rooms = group_semantic_rooms(payload, raw_object_rows, min_objects=3)
    simple_room, complex_room = select_simple_and_complex_rooms(rooms)

    assert len(rooms) == 2
    assert simple_room.view_type == "bedroom"
    assert simple_room.duplicate_objects == 0
    assert complex_room.view_type == "living room"
    assert complex_room.duplicate_objects == 2
    assert complex_room.repeated_labels == {"chair": 2, "table": 2}


def test_sample_room_objects_prefers_unique_for_simple_and_repeated_for_complex(tmp_path):
    db_dir = _make_room_db(tmp_path)
    with (db_dir / "object_meta.jsonl").open("r", encoding="utf-8") as handle:
        raw_object_rows = [json.loads(line) for line in handle if line.strip()]

    from spatial_rag.graph_builder import build_graph_payload

    payload = build_graph_payload(str(db_dir))
    rooms = group_semantic_rooms(payload, raw_object_rows, min_objects=3)
    simple_room, complex_room = select_simple_and_complex_rooms(rooms)
    objects_by_id = {
        int(row["object_global_id"]): row
        for row in raw_object_rows
    }

    simple_rows = sample_room_objects(simple_room, objects_by_id, room_kind="simple", min_objects=3, max_objects=4)
    complex_rows = sample_room_objects(complex_room, objects_by_id, room_kind="complex", min_objects=3, max_objects=4)

    assert len({row["label"] for row in simple_rows}) == len(simple_rows)
    complex_counts = {}
    for row in complex_rows:
        complex_counts[row["label"]] = complex_counts.get(row["label"], 0) + 1
    assert any(count >= 2 for count in complex_counts.values())


def test_compute_similarity_matrix_is_symmetric_with_unit_diagonal():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    matrix = compute_similarity_matrix(embeddings, [0, 1, 2])

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), np.ones(3))


def test_compute_geometry_similarity_matrix_supports_2d_and_3d():
    rows = [
        {
            "object_global_id": 1,
            "view_id": "view_00000",
            "estimated_global_x": 0.0,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
        },
        {
            "object_global_id": 2,
            "view_id": "view_00001",
            "estimated_global_x": 0.0,
            "estimated_global_y": 1.0,
            "estimated_global_z": 0.0,
        },
        {
            "object_global_id": 3,
            "view_id": "view_00002",
            "estimated_global_x": 3.0,
            "estimated_global_y": None,
            "estimated_global_z": 4.0,
        },
    ]

    matrix = compute_geometry_similarity_matrix(rows, sigma_2d=2.0, sigma_3d=2.5)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), np.ones(3))
    assert matrix[0, 1] > matrix[0, 2]
    assert np.isclose(matrix[0, 2], np.exp(-((5.0**2) / (2.0**2))), atol=1e-6)


def test_compute_fused_similarity_matrix_applies_same_view_soft_penalty_and_falls_back_to_text():
    text = np.asarray(
        [
            [1.0, 0.8, 0.4],
            [0.8, 1.0, 0.5],
            [0.4, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    geometry = np.asarray(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
        dtype=np.float32,
    )
    geometry_available = np.asarray(
        [
            [True, True, False],
            [True, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    rows = [
        {"object_global_id": 1, "view_id": "view_00000"},
        {"object_global_id": 2, "view_id": "view_00000"},
        {"object_global_id": 3, "view_id": "view_00001"},
    ]

    before, after = compute_fused_similarity_matrix(
        text_matrix=text,
        geometry_matrix=geometry,
        rows=rows,
        geometry_available=geometry_available,
        same_view_policy="soft_penalty",
        same_view_penalty=0.25,
        weight_text=0.7,
        weight_geo=0.3,
    )

    expected_same_view = (0.7 * 0.8) + (0.3 * 0.9)
    assert np.isclose(before[0, 1], expected_same_view, atol=1e-6)
    assert np.isclose(after[0, 1], expected_same_view * 0.25, atol=1e-6)
    assert np.isclose(before[0, 2], 0.4, atol=1e-6)
    assert np.isclose(after[0, 2], 0.4, atol=1e-6)
    assert np.allclose(np.diag(after), np.ones(3))


def test_analyze_rooms_writes_json_csv_and_heatmaps(tmp_path, monkeypatch):
    db_dir = _make_room_db(tmp_path)
    output_dir = tmp_path / "analysis_out"

    def _fake_graph_embed(context_by_obj_id, embedder=None):
        out = np.zeros((max(context_by_obj_id) + 1, 2), dtype=np.float32)
        out[10] = [1.0, 0.0]
        out[11] = [0.98, 0.02]
        out[12] = [0.0, 1.0]
        out[13] = [0.03, 0.97]
        out[14] = [-1.0, 0.0]
        out[20] = [0.35, 0.82]
        out[21] = [-0.42, 0.86]
        out[22] = [-0.88, -0.12]
        out[23] = [0.22, -0.9]
        return out

    monkeypatch.setattr("spatial_rag.room_object_similarity_analysis.embed_graph_contexts", _fake_graph_embed)

    report = analyze_rooms(
        db_dir=str(db_dir),
        output_dir=str(output_dir),
        min_objects=3,
        max_objects=4,
        fused_text_modes=("short", "long"),
        weight_text=0.7,
        weight_geo=0.3,
        same_view_policy="soft_penalty",
        same_view_penalty=0.25,
    )

    assert report["simple_room"]["view_type"] == "bedroom"
    assert report["complex_room"]["view_type"] == "living room"

    for folder in ("simple_room", "complex_room"):
        room_dir = output_dir / folder
        assert (room_dir / "objects.json").exists()
        assert (room_dir / "summary.json").exists()
        for mode in ("short", "long", "graph", "geometry", "short_geo", "long_geo"):
            assert (room_dir / f"{mode}_similarity.csv").exists()
            assert (room_dir / f"{mode}_similarity_heatmap.png").exists()
        assert (room_dir / "pair_breakdown_short_geo.json").exists()
        assert (room_dir / "pair_breakdown_long_geo.json").exists()

        summary = json.loads((room_dir / "summary.json").read_text(encoding="utf-8"))
        for mode in ("short", "long", "graph", "geometry", "short_geo", "long_geo"):
            matrix = np.asarray(summary["matrices"][mode], dtype=np.float32)
            assert matrix.shape[0] == matrix.shape[1]
            assert np.allclose(matrix, matrix.T)
            assert np.allclose(np.diag(matrix), np.ones(matrix.shape[0]))
        assert summary["fused_config"]["fused_text_modes"] == ["short", "long"]
        assert np.isclose(summary["fused_config"]["weight_text"], 0.7)
        assert np.isclose(summary["fused_config"]["weight_geo"], 0.3)
        assert summary["fused_config"]["same_view_policy"] == "soft_penalty"

        short_geo_pairs = json.loads((room_dir / "pair_breakdown_short_geo.json").read_text(encoding="utf-8"))
        assert short_geo_pairs
        assert {
            "obj_a_id",
            "obj_b_id",
            "label_a",
            "label_b",
            "same_label",
            "same_view",
            "text_mode",
            "text_similarity",
            "geometry_similarity",
            "fused_similarity_before_penalty",
            "fused_similarity_after_penalty",
            "distance_m_2d",
            "distance_m_3d",
            "used_3d_geometry",
        }.issubset(short_geo_pairs[0].keys())


def test_analyze_rooms_respects_single_fused_text_mode(tmp_path, monkeypatch):
    db_dir = _make_room_db(tmp_path)
    output_dir = tmp_path / "analysis_long_only"

    def _fake_graph_embed(context_by_obj_id, embedder=None):
        out = np.zeros((max(context_by_obj_id) + 1, 2), dtype=np.float32)
        for idx, obj_id in enumerate(sorted(context_by_obj_id)):
            out[obj_id] = np.asarray([1.0 - (0.01 * idx), 0.01 * idx], dtype=np.float32)
        return out

    monkeypatch.setattr("spatial_rag.room_object_similarity_analysis.embed_graph_contexts", _fake_graph_embed)

    analyze_rooms(
        db_dir=str(db_dir),
        output_dir=str(output_dir),
        min_objects=3,
        max_objects=4,
        fused_text_modes=("long",),
    )

    for folder in ("simple_room", "complex_room"):
        room_dir = output_dir / folder
        assert (room_dir / "long_geo_similarity.csv").exists()
        assert not (room_dir / "short_geo_similarity.csv").exists()
        assert (room_dir / "pair_breakdown_long_geo.json").exists()
        assert not (room_dir / "pair_breakdown_short_geo.json").exists()


def test_analyze_rooms_supports_selected_view_subset(tmp_path, monkeypatch):
    db_dir = _make_room_db(tmp_path)
    output_dir = tmp_path / "analysis_selected_views"

    def _fake_graph_embed(context_by_obj_id, embedder=None):
        out = np.zeros((max(context_by_obj_id) + 1, 2), dtype=np.float32)
        for idx, obj_id in enumerate(sorted(context_by_obj_id)):
            out[obj_id] = np.asarray([1.0 - (0.02 * idx), 0.02 * idx], dtype=np.float32)
        return out

    monkeypatch.setattr("spatial_rag.room_object_similarity_analysis.embed_graph_contexts", _fake_graph_embed)

    report = analyze_rooms(
        db_dir=str(db_dir),
        output_dir=str(output_dir),
        min_objects=3,
        max_objects=16,
        fused_text_modes=("short", "long"),
        view_ids=("view_00000", "view_00001", "view_00002", "view_00003"),
    )

    assert report["analysis_mode"] == "selected_views"
    assert report["selected_view_ids"] == ["view_00000", "view_00001", "view_00002", "view_00003"]
    assert "selected_views" in report
    assert "simple_room" not in report
    assert "complex_room" not in report

    room_dir = output_dir / "selected_views"
    assert (room_dir / "objects.json").exists()
    assert (room_dir / "summary.json").exists()
    for mode in ("short", "long", "graph", "geometry", "short_geo", "long_geo"):
        assert (room_dir / f"{mode}_similarity.csv").exists()
        assert (room_dir / f"{mode}_similarity_heatmap.png").exists()
    assert (room_dir / "pair_breakdown_short_geo.json").exists()
    assert (room_dir / "pair_breakdown_long_geo.json").exists()

    summary = json.loads((room_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["room_kind"] == "selected"
    assert summary["room_id"] == "selected_views"
    assert summary["num_selected_objects"] == 5
