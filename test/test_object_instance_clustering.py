import json

import cv2
import numpy as np
import pytest

from spatial_rag.object_instance_clustering import (
    _attach_object_embeddings,
    _compute_heatmap_display_values,
    _resolve_object_text,
    _serialize_surrounding_context_for_embedding,
    apply_constraints,
    build_similarity_matrix_from_descriptions,
    build_manual_anchor_camera_specs,
    compute_text_similarity,
    deduplicate_multi_view_embeddings,
    estimate_cluster_count_eigengap,
    load_object_observations,
    plot_similarity_heatmap,
    run_refined_graph_visualization_pipeline,
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


class _KeywordEmbedder:
    def __init__(self):
        self.calls = []
        self.vocab = [
            "chair",
            "window",
            "cabinet",
            "lamp",
            "table",
            "left",
            "right",
            "above",
            "below",
            "none",
        ]

    def embed_text(self, text):
        text_value = str(text or "")
        self.calls.append(text_value)
        lowered = text_value.lower()
        features = [float(lowered.count(token)) for token in self.vocab]
        features.append(float(len(lowered.split())))
        features.append(float(sum(ord(ch) for ch in lowered) % 997))
        return np.asarray(features, dtype=np.float32)


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


def test_serialize_surrounding_context_for_embedding_uses_compact_fields_and_rounds_distance():
    serialized = _serialize_surrounding_context_for_embedding(
        [
            {
                "label": "picture frame",
                "relation_to_primary": "slightly right, above",
                "distance_from_primary_m": 1.24,
                "estimated_global_x": 10.0,
            },
            {
                "label": "cabinet",
                "relation_to_primary": "left",
                "distance_from_primary_m": 0.81,
                "attributes": ["wooden"],
            },
            {
                "label": "",
                "relation_to_primary": "below",
                "distance_from_primary_m": 0.55,
            },
        ]
    )

    assert serialized == "picture frame [slightly right, above, 1.2m]; cabinet [left, 0.8m]"


def test_resolve_object_text_long_neighbors_appends_neighbors_none_when_empty():
    row = {
        "text_input_for_clip_long": "blue upholstered chair by the window",
        "surrounding_context": [],
    }

    resolved = _resolve_object_text(row, text_mode="long_neighbors")

    assert resolved == "blue upholstered chair by the window | neighbors: none"


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


def test_apply_constraints_thresholds_on_original_similarity_before_penalty():
    rows = [
        {"view_id": "view_0"},
        {"view_id": "view_0"},
        {"view_id": "view_1"},
    ]
    similarity = np.asarray(
        [
            [1.0, 0.80, 0.55],
            [0.80, 1.0, 0.72],
            [0.55, 0.72, 1.0],
        ],
        dtype=np.float32,
    )

    affinity = apply_constraints(
        similarity,
        rows,
        same_view_policy="soft_penalty",
        same_view_penalty=0.25,
        min_similarity=0.60,
    )

    assert np.isclose(affinity[0, 1], 0.20)
    assert np.isclose(affinity[1, 0], 0.20)
    assert np.isclose(affinity[0, 2], 0.0)
    assert np.isclose(affinity[2, 0], 0.0)
    assert np.isclose(affinity[1, 2], 0.72)


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


def test_deduplicate_multi_view_embeddings_reorders_similarity_into_blocks():
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.02, 0.98],
        ],
        dtype=np.float32,
    )
    view_ids = ["view_0", "view_1", "view_0", "view_2", "view_1", "view_2"]

    result = deduplicate_multi_view_embeddings(
        embeddings,
        view_ids,
        object_ids=[10, 11, 20, 21, 30, 31],
        cluster_count_mode="fixed",
        n_clusters=3,
        same_view_penalty=0.25,
        similarity_threshold=0.60,
        random_state=0,
    )

    labels = result["cluster_labels"].tolist()
    assert labels == [0, 0, 1, 1, 2, 2]
    assert result["boundary_after_indices"] == [1, 3]
    reordered = result["reordered_similarity_matrix"]
    assert reordered.shape == (6, 6)
    assert np.min(np.diag(reordered)) >= 0.999
    assert float(np.mean(reordered[:2, :2])) > 0.99
    assert float(np.mean(reordered[2:4, 2:4])) > 0.99
    assert float(np.mean(reordered[4:, 4:])) > 0.99
    assert float(np.max(reordered[:2, 2:])) < 0.05


def test_run_refined_graph_visualization_pipeline_reorders_blocks_with_dbscan():
    rows = [
        {"object_global_id": 10, "embedding": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)},
        {"object_global_id": 11, "embedding": np.asarray([0.99, 0.01, 0.0], dtype=np.float32)},
        {"object_global_id": 20, "embedding": np.asarray([0.0, 1.0, 0.0], dtype=np.float32)},
        {"object_global_id": 21, "embedding": np.asarray([0.01, 0.99, 0.0], dtype=np.float32)},
        {"object_global_id": 30, "embedding": np.asarray([0.0, 0.0, 1.0], dtype=np.float32)},
        {"object_global_id": 31, "embedding": np.asarray([0.0, 0.02, 0.98], dtype=np.float32)},
    ]

    result = run_refined_graph_visualization_pipeline(
        rows,
        object_ids=[10, 11, 20, 21, 30, 31],
        knn_k=2,
        spectral_dim=3,
        dbscan_eps=0.1,
        dbscan_min_samples=2,
    )

    assert result["labels"].tolist() == [0, 0, 1, 1, 2, 2]
    assert result["boundary_after_indices"] == [1, 3]
    assert result["noise_count"] == 0
    assert result["num_clusters_excluding_noise"] == 3
    assert np.allclose(np.diag(result["visual_similarity_matrix"]), np.zeros(6))
    assert result["knn_affinity_matrix"].shape == (6, 6)


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


def test_compute_heatmap_display_values_supports_offdiag_extreme_with_dark_diagonal():
    matrix = np.asarray(
        [
            [1.0, 0.92, 0.10, 0.08],
            [0.92, 1.0, 0.12, 0.09],
            [0.10, 0.12, 1.0, 0.87],
            [0.08, 0.09, 0.87, 1.0],
        ],
        dtype=np.float32,
    )

    display = _compute_heatmap_display_values(
        matrix,
        display_transform="offdiag_extreme",
        suppress_diagonal_display=True,
    )

    assert display.shape == matrix.shape
    assert np.allclose(np.diag(display), np.zeros(4))
    assert float(display[0, 1]) > 0.95
    assert float(display[2, 3]) > 0.95
    assert float(display[0, 2]) < 0.1
    assert float(display[1, 3]) < 0.1


def test_compute_heatmap_display_values_supports_budgeted_diagonal_display():
    matrix = np.asarray(
        [
            [1.0, 0.92, 0.12, 0.10],
            [0.92, 1.0, 0.14, 0.12],
            [0.12, 0.14, 1.0, 0.88],
            [0.10, 0.12, 0.88, 1.0],
        ],
        dtype=np.float32,
    )

    display = _compute_heatmap_display_values(
        matrix,
        display_transform="offdiag_extreme",
        suppress_diagonal_display=False,
        diagonal_brightness_budget_ratio=0.18,
        diagonal_display_cap=0.82,
    )

    diag = np.diag(display)
    assert np.all(diag > 0.0)
    assert np.all(diag < 1.0)
    assert np.all(diag <= 0.82 + 1e-6)


def test_compute_heatmap_display_values_linear_preserves_similarity_proportions():
    matrix = np.asarray(
        [
            [1.0, 0.80, 0.35],
            [0.80, 1.0, 0.20],
            [0.35, 0.20, 1.0],
        ],
        dtype=np.float32,
    )

    display = _compute_heatmap_display_values(
        matrix,
        vmin=0.0,
        vmax=1.0,
        display_transform="linear",
        suppress_diagonal_display=False,
    )

    assert display.shape == matrix.shape
    assert np.allclose(np.diag(display), np.ones(3))
    assert np.isclose(float(display[0, 1]), 0.80)
    assert np.isclose(float(display[0, 2]), 0.35)
    assert np.isclose(float(display[1, 2]), 0.20)


def test_load_object_observations_short_and_long_keep_using_precomputed_embeddings(tmp_path):
    db_dir = _make_place_db(tmp_path)
    embedder = _KeywordEmbedder()

    observations_short = load_object_observations(
        db_dir=str(db_dir),
        text_mode="short",
        embedder=embedder,
    )
    observations_long = load_object_observations(
        db_dir=str(db_dir),
        text_mode="long",
        embedder=embedder,
    )

    assert embedder.calls == []
    assert np.allclose(observations_short[0]["embedding"], np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.allclose(observations_long[0]["embedding"], np.asarray([1.0, 0.0], dtype=np.float32))


def test_attach_object_embeddings_long_neighbors_bypasses_precomputed_embeddings(tmp_path):
    db_dir = _make_place_db(tmp_path)
    rows = [
        {
            "object_global_id": 10,
            "text_input_for_clip_long": "blue upholstered chair by the window",
            "surrounding_context": [
                {
                    "label": "cabinet",
                    "relation_to_primary": "left",
                    "distance_from_primary_m": 0.84,
                }
            ],
        }
    ]
    embedder = _KeywordEmbedder()

    attached = _attach_object_embeddings(
        rows,
        db_dir=str(db_dir),
        text_mode="long_neighbors",
        embedder=embedder,
    )

    assert len(embedder.calls) == 1
    assert "neighbors: cabinet [left, 0.8m]" in attached[0]["embedding_text"]
    assert attached[0]["embedding"] is not None


def test_long_neighbors_changes_similarity_when_surroundings_differ():
    rows = [
        {
            "object_global_id": 10,
            "view_id": "view_0",
            "entry_id": 0,
            "text_input_for_clip_long": "blue upholstered chair by the window",
            "surrounding_context": [
                {
                    "label": "cabinet",
                    "relation_to_primary": "left",
                    "distance_from_primary_m": 0.8,
                }
            ],
        },
        {
            "object_global_id": 11,
            "view_id": "view_1",
            "entry_id": 1,
            "text_input_for_clip_long": "blue upholstered chair by the window",
            "surrounding_context": [
                {
                    "label": "lamp",
                    "relation_to_primary": "right",
                    "distance_from_primary_m": 0.8,
                }
            ],
        },
    ]
    embedder = _KeywordEmbedder()

    long_rows = _attach_object_embeddings(rows, db_dir=None, text_mode="long", embedder=embedder)
    long_neighbors_rows = _attach_object_embeddings(rows, db_dir=None, text_mode="long_neighbors", embedder=embedder)
    similarity_long = build_similarity_matrix_from_descriptions(long_rows)
    similarity_long_neighbors = build_similarity_matrix_from_descriptions(long_neighbors_rows)

    assert np.isclose(float(similarity_long[0, 1]), 1.0)
    assert float(similarity_long_neighbors[0, 1]) < float(similarity_long[0, 1])


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
    assert (place_dir / "clustered_similarity_matrix.npy").exists()
    assert (place_dir / "clustered_similarity_matrix.csv").exists()
    assert (place_dir / "clustered_similarity_order.json").exists()
    assert (place_dir / "clustered_similarity_heatmap.png").exists()
    assert (place_dir / "clustered_similarity_heatmap_offdiag_only.png").exists()
    assert (place_dir / "refined_graph_cluster_labels.json").exists()
    assert (place_dir / "refined_graph_knn_affinity_matrix.npy").exists()
    assert (place_dir / "refined_graph_laplacian.npy").exists()
    assert (place_dir / "refined_graph_spectral_embedding.npy").exists()
    assert (place_dir / "refined_graph_reordered_similarity_matrix.npy").exists()
    assert (place_dir / "refined_graph_reordered_visual_similarity_matrix.npy").exists()
    assert (place_dir / "refined_graph_similarity_diag1_source_matrix.npy").exists()
    assert (place_dir / "refined_graph_clustered_similarity_heatmap.png").exists()
    assert (place_dir / "refined_graph_clustered_similarity_heatmap_offdiag_only.png").exists()
    assert (place_dir / "refined_graph_clustered_similarity_heatmap_diag1.png").exists()
    assert (place_dir / "view_annotations" / "manifest.json").exists()
    assert (place_dir / "view_annotations" / "annotated_views_grid.jpg").exists()
    assert (place_dir / "view_annotations" / "view_00000_annotated.jpg").exists()

    cluster_summary = json.loads((place_dir / "cluster_summary.json").read_text(encoding="utf-8"))
    assert cluster_summary["n_objects"] == 3
    assert cluster_summary["n_clusters"] == 2
    cluster_sizes = sorted(cluster["num_members"] for cluster in cluster_summary["clusters"])
    assert cluster_sizes == [1, 2]


def test_run_object_instance_clustering_place_mode_supports_entry_id_filter(tmp_path):
    db_dir = _make_place_db(tmp_path)
    output_dir = tmp_path / "place_out_subset"

    report = run_object_instance_clustering(
        output_dir=str(output_dir),
        group_mode="place",
        db_dir=str(db_dir),
        entry_ids=[0, 1],
        text_mode="long",
        cluster_count_mode="fixed",
        n_clusters=2,
        same_view_policy="soft_penalty",
    )

    assert report["selected_entry_ids"] == [0, 1]
    assert (output_dir / "summary.json").exists()

    place_dir = output_dir / "place_00000"
    cluster_summary = json.loads((place_dir / "cluster_summary.json").read_text(encoding="utf-8"))
    assert cluster_summary["n_objects"] == 3

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_entry_ids"] == [0, 1]


def test_run_object_instance_clustering_long_neighbors_runs_without_precomputed_embeddings(tmp_path):
    db_dir = _make_place_db(tmp_path)
    (db_dir / "object_text_emb_long.npy").unlink()
    output_dir = tmp_path / "place_out_long_neighbors"
    embedder = _KeywordEmbedder()

    report = run_object_instance_clustering(
        output_dir=str(output_dir),
        group_mode="place",
        db_dir=str(db_dir),
        text_mode="long_neighbors",
        cluster_count_mode="fixed",
        n_clusters=2,
        same_view_policy="soft_penalty",
        embedder=embedder,
    )

    assert report["text_mode"] == "long_neighbors"
    assert len(embedder.calls) == 4
    assert (output_dir / "place_00000" / "cluster_summary.json").exists()


def test_run_object_instance_clustering_selected_views_mode_combines_entries(tmp_path):
    db_dir = _make_place_db(tmp_path)
    output_dir = tmp_path / "selected_views_out"

    report = run_object_instance_clustering(
        output_dir=str(output_dir),
        group_mode="selected_views",
        db_dir=str(db_dir),
        entry_ids=[0, 1],
        text_mode="long",
        cluster_count_mode="fixed",
        n_clusters=2,
        same_view_policy="soft_penalty",
    )

    assert report["group_mode"] == "selected_views"
    assert report["selected_entry_ids"] == [0, 1]
    assert report["num_groups"] == 1

    group_dir = output_dir / "selected_views"
    assert (group_dir / "cluster_summary.json").exists()
    cluster_summary = json.loads((group_dir / "cluster_summary.json").read_text(encoding="utf-8"))
    assert cluster_summary["group_type"] == "selected_views"
    assert cluster_summary["n_objects"] == 3


def test_run_object_instance_clustering_rejects_entry_ids_for_manual_anchor_mode(tmp_path):
    with pytest.raises(ValueError, match="entry_ids is only supported when group_mode='place' or 'selected_views'"):
        run_object_instance_clustering(
            output_dir=str(tmp_path / "anchor_out"),
            group_mode="manual_anchor_center",
            entry_ids=[0, 1],
            anchors_jsonl=str(tmp_path / "anchors.jsonl"),
            scene_path="scene.glb",
        )


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
