import json

import cv2
import numpy as np
import pytest

from spatial_rag.object_instance_clustering import (
    apply_constraints,
    build_manual_anchor_camera_specs,
    compute_text_similarity,
    estimate_cluster_count_eigengap,
    plot_similarity_heatmap,
    run_object_instance_clustering,
    run_spectral_clustering,
    summarize_clusters,
    validate_manual_anchor_pose_results,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def _meta_row(entry_id, x, z, orientation, room_function="resting", view_type="living room"):
    return {
        "id": int(entry_id),
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


def _object_row(object_global_id, entry_id, label, description, long_description, orientation, distance):
    return {
        "object_global_id": int(object_global_id),
        "entry_id": int(entry_id),
        "file_name": f"images/view_{entry_id:05d}.jpg",
        "label": label,
        "description": description,
        "long_form_open_description": long_description,
        "text_input_for_clip_short": description,
        "text_input_for_clip_long": long_description,
        "object_text_short": description,
        "object_text_long": long_description,
        "laterality": "center",
        "distance_bin": "middle",
        "verticality": "middle",
        "distance_from_camera_m": float(distance),
        "object_orientation_deg": float(orientation),
        "estimated_global_x": float(object_global_id) / 10.0,
        "estimated_global_y": 0.5,
        "estimated_global_z": -float(object_global_id) / 10.0,
        "location_relative_to_other_objects": "",
        "parse_status": "ok",
    }


def _make_place_db(tmp_path):
    db_dir = tmp_path / "place_db"
    image_dir = db_dir / "images"
    image_dir.mkdir(parents=True)
    for entry_id in range(3):
        canvas = np.full((180, 240, 3), 235, dtype=np.uint8)
        cv2.putText(canvas, f"view {entry_id}", (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
        ok = cv2.imwrite(str(image_dir / f"view_{entry_id:05d}.jpg"), canvas)
        assert ok

    meta_rows = [
        _meta_row(0, 0.0, 0.0, 0),
        _meta_row(1, 0.0, 0.0, 90),
        _meta_row(2, 3.0, 0.0, 0, room_function="working", view_type="study"),
    ]
    object_rows = [
        _object_row(
            10,
            0,
            "chair",
            "blue chair",
            "blue upholstered chair by the window",
            0.0,
            1.0,
        ),
        _object_row(
            11,
            1,
            "chair",
            "blue chair second view",
            "blue upholstered chair near the same window",
            15.0,
            1.1,
        ),
        _object_row(
            12,
            0,
            "table",
            "round table",
            "small round wooden table",
            20.0,
            1.2,
        ),
        _object_row(
            20,
            2,
            "lamp",
            "tall lamp",
            "tall standing floor lamp",
            180.0,
            1.4,
        ),
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)
    (db_dir / "build_report.json").write_text(
        json.dumps({"scan_angles": [0, 90], "random_config": {"scan_angles": [0, 90]}}),
        encoding="utf-8",
    )
    object_emb_long = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    object_emb_short = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(db_dir / "object_text_emb_long.npy", object_emb_long)
    np.save(db_dir / "object_text_emb_short.npy", object_emb_short)
    return db_dir


def test_build_manual_anchor_camera_specs_returns_four_surrounding_views():
    specs = build_manual_anchor_camera_specs(
        {"anchor_id": "anchor_a", "center_x": 2.0, "center_z": 3.0},
        default_radius_m=1.5,
    )

    assert [(row["anchor_direction"], row["orientation_deg"]) for row in specs] == [
        ("north", 0),
        ("east", 90),
        ("south", 180),
        ("west", 270),
    ]
    assert specs[0]["requested_x"] == 2.0
    assert specs[0]["requested_z"] == 4.5
    assert specs[1]["requested_x"] == 3.5
    assert specs[1]["requested_z"] == 3.0
    assert specs[3]["requested_x"] == 0.5


def test_validate_manual_anchor_pose_results_accepts_valid_snaps():
    pose_results = [
        {
            "synthetic_view_id": "anchor_a_north",
            "requested_x": 0.0,
            "requested_z": 1.5,
            "actual_world_position": [0.05, 0.0, 1.52],
        },
        {
            "synthetic_view_id": "anchor_a_east",
            "requested_x": 1.5,
            "requested_z": 0.0,
            "actual_world_position": [1.55, 0.0, 0.03],
        },
        {
            "synthetic_view_id": "anchor_a_south",
            "requested_x": 0.0,
            "requested_z": -1.5,
            "actual_world_position": [0.02, 0.0, -1.45],
        },
        {
            "synthetic_view_id": "anchor_a_west",
            "requested_x": -1.5,
            "requested_z": 0.0,
            "actual_world_position": [-1.46, 0.0, -0.01],
        },
    ]

    validate_manual_anchor_pose_results(
        pose_results,
        max_snap_distance_m=0.75,
        min_pose_separation_m=0.1,
    )


def test_validate_manual_anchor_pose_results_rejects_oversnap():
    pose_results = [
        {
            "synthetic_view_id": "anchor_a_north",
            "requested_x": 0.0,
            "requested_z": 1.5,
            "actual_world_position": [2.0, 0.0, 1.5],
        },
        {
            "synthetic_view_id": "anchor_a_east",
            "requested_x": 1.5,
            "requested_z": 0.0,
            "actual_world_position": [1.5, 0.0, 0.0],
        },
        {
            "synthetic_view_id": "anchor_a_south",
            "requested_x": 0.0,
            "requested_z": -1.5,
            "actual_world_position": [0.0, 0.0, -1.5],
        },
        {
            "synthetic_view_id": "anchor_a_west",
            "requested_x": -1.5,
            "requested_z": 0.0,
            "actual_world_position": [-1.5, 0.0, 0.0],
        },
    ]

    with pytest.raises(ValueError, match="exceeded snap threshold"):
        validate_manual_anchor_pose_results(pose_results, max_snap_distance_m=0.75)


def test_validate_manual_anchor_pose_results_rejects_degenerate_collapse():
    pose_results = [
        {
            "synthetic_view_id": "anchor_a_north",
            "requested_x": 0.0,
            "requested_z": 1.5,
            "actual_world_position": [0.0, 0.0, 1.5],
        },
        {
            "synthetic_view_id": "anchor_a_east",
            "requested_x": 1.5,
            "requested_z": 0.0,
            "actual_world_position": [1.5, 0.0, 0.0],
        },
        {
            "synthetic_view_id": "anchor_a_south",
            "requested_x": 0.0,
            "requested_z": -1.5,
            "actual_world_position": [1.5, 0.0, 0.0],
        },
        {
            "synthetic_view_id": "anchor_a_west",
            "requested_x": -1.5,
            "requested_z": 0.0,
            "actual_world_position": [-1.5, 0.0, 0.0],
        },
    ]

    with pytest.raises(ValueError, match="collapsed"):
        validate_manual_anchor_pose_results(
            pose_results,
            max_snap_distance_m=3.0,
            min_pose_separation_m=0.1,
        )


def test_compute_text_similarity_is_symmetric_with_unit_diagonal():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    matrix = compute_text_similarity(embeddings)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), np.ones(3))


def test_apply_constraints_supports_soft_penalty_hard_block_and_none():
    rows = [
        {"view_id": "view_0"},
        {"view_id": "view_0"},
        {"view_id": "view_1"},
    ]
    similarity = np.asarray(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
        dtype=np.float32,
    )

    soft = apply_constraints(
        similarity,
        rows,
        same_view_policy="soft_penalty",
        same_view_penalty=0.25,
    )
    hard = apply_constraints(similarity, rows, same_view_policy="hard_block")
    none = apply_constraints(similarity, rows, same_view_policy="none")

    assert np.isclose(soft[0, 1], 0.2)
    assert np.isclose(soft[1, 0], 0.2)
    assert np.isclose(hard[0, 1], 0.0)
    assert np.isclose(hard[1, 0], 0.0)
    assert np.isclose(none[0, 1], 0.8)
    assert np.allclose(np.diag(soft), np.ones(3))


def test_estimate_cluster_count_eigengap_finds_two_blocks():
    affinity = np.asarray(
        [
            [1.0, 0.95, 0.10, 0.05],
            [0.95, 1.0, 0.10, 0.05],
            [0.10, 0.10, 1.0, 0.92],
            [0.05, 0.05, 0.92, 1.0],
        ],
        dtype=np.float32,
    )

    assert estimate_cluster_count_eigengap(affinity) == 2


def test_run_spectral_clustering_zero_affinity_falls_back_to_singletons():
    affinity = np.eye(3, dtype=np.float32)

    result = run_spectral_clustering(
        affinity,
        object_ids=[10, 11, 12],
        cluster_count_mode="fixed",
        n_clusters=2,
    )

    assert result["n_clusters"] == 3
    assert result["fallback_reason"] == "zero_affinity_graph"
    assert result["labels"].tolist() == [0, 1, 2]


def test_plot_similarity_heatmap_supports_axis_labels(tmp_path):
    matrix = np.asarray([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    output_path = tmp_path / "labeled_heatmap.png"

    plot_similarity_heatmap(
        matrix,
        output_path,
        title="Text Cosine Similarity",
        axis_labels=[
            "obj10|view_00000|very long chair label for testing",
            "obj11|view_00001|very long table label for testing",
        ],
        annotate_values=True,
    )

    image = cv2.imread(str(output_path))
    assert image is not None
    assert image.shape[0] >= 520
    assert image.shape[1] >= 520


def test_plot_similarity_heatmap_supports_log_display_transform(tmp_path):
    matrix = np.asarray(
        [
            [1.0, 0.82, 0.67],
            [0.82, 1.0, 0.61],
            [0.67, 0.61, 1.0],
        ],
        dtype=np.float32,
    )
    output_path = tmp_path / "log_heatmap.png"

    plot_similarity_heatmap(
        matrix,
        output_path,
        title="Text Cosine Similarity (log)",
        axis_labels=[
            "view_00019|table|obj195",
            "view_00024|table|obj266",
            "view_00058|table|obj663",
        ],
        annotate_values=True,
        vmin=0.0,
        vmax=1.0,
        display_transform="log",
    )

    image = cv2.imread(str(output_path))
    assert image is not None
    assert image.shape[0] >= 520
    assert image.shape[1] >= 520


def test_summarize_clusters_marks_same_view_collisions():
    rows = [
        {
            "object_global_id": 10,
            "observation_id": "obs_000010",
            "view_id": "view_0",
            "label": "chair",
            "description": "chair one",
            "long_form_open_description": "chair one long",
        },
        {
            "object_global_id": 11,
            "observation_id": "obs_000011",
            "view_id": "view_0",
            "label": "chair",
            "description": "chair two",
            "long_form_open_description": "chair two long",
        },
    ]
    similarity = np.asarray([[1.0, 0.9], [0.9, 1.0]], dtype=np.float32)

    summary = summarize_clusters(
        rows,
        similarity,
        [0, 0],
        group_id="place_00000",
        group_type="place",
    )

    assert summary["n_clusters"] == 1
    assert summary["clusters"][0]["same_view_collision"] is True
    assert summary["clusters"][0]["offending_view_ids"] == ["view_0"]


def test_run_object_instance_clustering_place_mode_writes_expected_artifacts(tmp_path):
    db_dir = _make_place_db(tmp_path)
    output_dir = tmp_path / "place_out"

    report = run_object_instance_clustering(
        output_dir=str(output_dir),
        group_mode="place",
        db_dir=str(db_dir),
        text_mode="long",
        cluster_count_mode="fixed",
        n_clusters=2,
        same_view_policy="soft_penalty",
    )

    assert report["group_mode"] == "place"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "instance_candidate_graph.json").exists()

    place_dir = output_dir / "place_00000"
    assert (place_dir / "objects.json").exists()
    assert (place_dir / "similarity_matrix.npy").exists()
    assert (place_dir / "affinity_matrix.npy").exists()
    assert (place_dir / "cluster_labels.json").exists()
    assert (place_dir / "cluster_summary.json").exists()
    assert (place_dir / "cluster_summary.md").exists()
    assert (place_dir / "similarity_heatmap.png").exists()
    assert (place_dir / "affinity_heatmap.png").exists()
    assert (place_dir / "view_annotations" / "manifest.json").exists()
    assert (place_dir / "view_annotations" / "annotated_views_grid.jpg").exists()
    assert (place_dir / "view_annotations" / "view_00000_annotated.jpg").exists()

    cluster_summary = json.loads((place_dir / "cluster_summary.json").read_text(encoding="utf-8"))
    assert cluster_summary["n_objects"] == 3
    assert cluster_summary["n_clusters"] == 2
    cluster_sizes = sorted(cluster["num_members"] for cluster in cluster_summary["clusters"])
    assert cluster_sizes == [1, 2]


def test_run_object_instance_clustering_manual_anchor_mode_writes_generated_views(monkeypatch, tmp_path):
    output_dir = tmp_path / "anchor_out"

    def _fake_collect_manual_anchor_groups(**kwargs):
        root = kwargs["output_dir"]
        group_dir = output_dir / "anchor_alpha" / "generated_views"
        group_dir.mkdir(parents=True, exist_ok=True)
        meta_rows = [
            {
                "id": 0,
                "anchor_id": "anchor_alpha",
                "synthetic_view_id": "anchor_anchor_alpha_north",
                "anchor_direction": "north",
                "file_name": "generated_views/anchor_anchor_alpha_north.jpg",
            },
            {
                "id": 1,
                "anchor_id": "anchor_alpha",
                "synthetic_view_id": "anchor_anchor_alpha_east",
                "anchor_direction": "east",
                "file_name": "generated_views/anchor_anchor_alpha_east.jpg",
            },
            {
                "id": 2,
                "anchor_id": "anchor_alpha",
                "synthetic_view_id": "anchor_anchor_alpha_south",
                "anchor_direction": "south",
                "file_name": "generated_views/anchor_anchor_alpha_south.jpg",
            },
            {
                "id": 3,
                "anchor_id": "anchor_alpha",
                "synthetic_view_id": "anchor_anchor_alpha_west",
                "anchor_direction": "west",
                "file_name": "generated_views/anchor_anchor_alpha_west.jpg",
            },
        ]
        object_rows = [
            {
                "object_global_id": 10,
                "observation_id": "obs_000010",
                "anchor_id": "anchor_alpha",
                "view_id": "anchor_anchor_alpha_north",
                "synthetic_view_id": "anchor_anchor_alpha_north",
                "entry_id": 0,
                "label": "chair",
                "description": "blue chair",
                "long_form_open_description": "blue upholstered chair",
            },
            {
                "object_global_id": 11,
                "observation_id": "obs_000011",
                "anchor_id": "anchor_alpha",
                "view_id": "anchor_anchor_alpha_east",
                "synthetic_view_id": "anchor_anchor_alpha_east",
                "entry_id": 1,
                "label": "chair",
                "description": "same blue chair",
                "long_form_open_description": "same blue upholstered chair",
            },
            {
                "object_global_id": 12,
                "observation_id": "obs_000012",
                "anchor_id": "anchor_alpha",
                "view_id": "anchor_anchor_alpha_south",
                "synthetic_view_id": "anchor_anchor_alpha_south",
                "entry_id": 2,
                "label": "table",
                "description": "wood table",
                "long_form_open_description": "small wooden table",
            },
        ]
        _write_jsonl(group_dir / "meta.jsonl", meta_rows)
        _write_jsonl(group_dir / "object_meta.jsonl", object_rows)
        for name in (
            "anchor_anchor_alpha_north.jpg",
            "anchor_anchor_alpha_east.jpg",
            "anchor_anchor_alpha_south.jpg",
            "anchor_anchor_alpha_west.jpg",
        ):
            canvas = np.full((180, 240, 3), 235, dtype=np.uint8)
            cv2.putText(canvas, name.replace(".jpg", ""), (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
            ok = cv2.imwrite(str(group_dir / name), canvas)
            assert ok
        assert root == str(output_dir)
        return [
            {
                "group_id": "anchor_alpha",
                "group_type": "manual_anchor_center",
                "objects": [
                        {
                            "object_global_id": 10,
                            "observation_id": "obs_000010",
                            "anchor_id": "anchor_alpha",
                            "view_id": "anchor_anchor_alpha_north",
                            "synthetic_view_id": "anchor_anchor_alpha_north",
                            "entry_id": 0,
                            "file_name": "generated_views/anchor_anchor_alpha_north.jpg",
                            "label": "chair",
                            "description": "blue chair",
                            "long_form_open_description": "blue upholstered chair",
                            "text_input_for_clip_long": "blue upholstered chair",
                            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                    },
                        {
                            "object_global_id": 11,
                            "observation_id": "obs_000011",
                            "anchor_id": "anchor_alpha",
                            "view_id": "anchor_anchor_alpha_east",
                            "synthetic_view_id": "anchor_anchor_alpha_east",
                            "entry_id": 1,
                            "file_name": "generated_views/anchor_anchor_alpha_east.jpg",
                            "label": "chair",
                            "description": "same blue chair",
                            "long_form_open_description": "same blue upholstered chair",
                            "text_input_for_clip_long": "same blue upholstered chair",
                            "embedding": np.asarray([0.99, 0.01], dtype=np.float32),
                    },
                        {
                            "object_global_id": 12,
                            "observation_id": "obs_000012",
                            "anchor_id": "anchor_alpha",
                            "view_id": "anchor_anchor_alpha_south",
                            "synthetic_view_id": "anchor_anchor_alpha_south",
                            "entry_id": 2,
                            "file_name": "generated_views/anchor_anchor_alpha_south.jpg",
                            "label": "table",
                            "description": "wood table",
                            "long_form_open_description": "small wooden table",
                            "text_input_for_clip_long": "small wooden table",
                            "embedding": np.asarray([0.0, 1.0], dtype=np.float32),
                    },
                ],
                "group_metadata": {
                    "anchor": {"anchor_id": "anchor_alpha", "center_x": 0.0, "center_z": 0.0},
                    "pose_results": [],
                },
            }
        ]

    monkeypatch.setattr(
        "spatial_rag.object_instance_clustering.collect_manual_anchor_groups",
        _fake_collect_manual_anchor_groups,
    )

    report = run_object_instance_clustering(
        output_dir=str(output_dir),
        group_mode="manual_anchor_center",
        anchors_jsonl=str(tmp_path / "anchors.jsonl"),
        text_mode="long",
        cluster_count_mode="fixed",
        n_clusters=2,
    )

    assert report["group_mode"] == "manual_anchor_center"
    anchor_dir = output_dir / "anchor_alpha"
    assert (anchor_dir / "generated_views" / "meta.jsonl").exists()
    assert (anchor_dir / "generated_views" / "object_meta.jsonl").exists()
    assert (anchor_dir / "generated_views" / "anchor_anchor_alpha_north.jpg").exists()
    assert (anchor_dir / "cluster_summary.json").exists()
    assert (anchor_dir / "cluster_labels.json").exists()
    assert (anchor_dir / "view_annotations" / "manifest.json").exists()
    assert (anchor_dir / "view_annotations" / "anchor_anchor_alpha_north_annotated.jpg").exists()

    objects_payload = json.loads((anchor_dir / "objects.json").read_text(encoding="utf-8"))
    assert objects_payload["objects"][0]["anchor_id"] == "anchor_alpha"
    assert objects_payload["objects"][0]["view_id"].startswith("anchor_anchor_alpha_")
