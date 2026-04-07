import csv
import json
from pathlib import Path

import cv2
import numpy as np

import spatial_rag.sequential_spectral_experiment as sequential_spectral_experiment
from spatial_rag.sequential_spectral_experiment import (
    DEFAULT_CROSS_AFFINITY_MIN,
    DEFAULT_DISTANCE_GATE_DSQ0_SWEEP,
    DEFAULT_SPECTRAL_MAX_EXTRA_CLUSTERS,
    DEFAULT_VIEW_IDS,
    _build_cluster,
    _full_bipartite_affinity,
    _run_capped_sequential_spectral_clustering,
    apply_incremental_step,
    build_cross_affinity_matrix,
    load_sequence_objects,
    run_sequential_spectral_experiment,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def _make_sequence_db(tmp_path):
    db_dir = tmp_path / "seq_db"
    db_dir.mkdir(parents=True)
    image_dir = db_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = [
        {"id": 19, "orientation": 270, "file_name": "images/pose_00004_o270_000019.jpg"},
        {"id": 24, "orientation": 0, "file_name": "images/pose_00006_o000_000024.jpg"},
        {"id": 58, "orientation": 180, "file_name": "images/pose_00014_o180_000058.jpg"},
        {"id": 65, "orientation": 90, "file_name": "images/pose_00016_o090_000065.jpg"},
    ]
    object_rows = [
        {
            "entry_id": 19,
            "view_id": "view_00019",
            "object_global_id": 1,
            "label": "chair",
            "bbox_xyxy": [12.0, 16.0, 96.0, 120.0],
            "estimated_global_x": 0.0,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": 0.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "entry_id": 19,
            "view_id": "view_00019",
            "object_global_id": 2,
            "label": "table",
            "bbox_xyxy": [120.0, 24.0, 228.0, 132.0],
            "estimated_global_x": 5.0,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 2.0,
            "relative_bearing_deg": 12.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "entry_id": 24,
            "view_id": "view_00024",
            "object_global_id": 3,
            "label": "wooden seat",
            "bbox_xyxy": [18.0, 110.0, 98.0, 210.0],
            "estimated_global_x": 0.1,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.1,
            "distance_from_camera_m": 1.1,
            "relative_bearing_deg": 2.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "entry_id": 24,
            "view_id": "view_00024",
            "object_global_id": 4,
            "label": "plant",
            "bbox_xyxy": [170.0, 32.0, 250.0, 152.0],
            "estimated_global_x": 8.0,
            "estimated_global_y": 0.0,
            "estimated_global_z": 1.0,
            "distance_from_camera_m": 3.0,
            "relative_bearing_deg": 35.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "entry_id": 58,
            "view_id": "view_00058",
            "object_global_id": 5,
            "label": "table",
            "bbox_xyxy": [108.0, 72.0, 220.0, 164.0],
            "estimated_global_x": 5.1,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.2,
            "distance_from_camera_m": 2.1,
            "relative_bearing_deg": 10.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "entry_id": 65,
            "view_id": "view_00065",
            "object_global_id": 6,
            "label": "chair",
            "bbox_xyxy": [40.0, 54.0, 126.0, 180.0],
            "estimated_global_x": 0.2,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": -5.0,
            "relative_height_from_camera_m": 0.0,
        },
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)
    for index, row in enumerate(meta_rows):
        image_path = db_dir / row["file_name"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_rgb = np.full((240, 320, 3), 235 - index * 15, dtype=np.uint8)
        ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        assert ok
    emb = np.asarray(
        [
            [1.0, 0.0, 0.0],     # chair
            [0.0, 1.0, 0.0],     # table
            [0.98, 0.02, 0.0],   # wooden seat ~ chair
            [0.0, 0.0, 1.0],     # plant
            [0.02, 0.97, 0.01],  # table near table
            [0.96, 0.04, 0.0],   # chair near chair
        ],
        dtype=np.float32,
    )
    np.save(db_dir / "object_text_emb_long.npy", emb)
    return db_dir


def test_load_sequence_objects_respects_fixed_view_order(tmp_path):
    db_dir = _make_sequence_db(tmp_path)
    sequence = load_sequence_objects(str(db_dir))

    assert sequence["selected_view_ids"] == list(DEFAULT_VIEW_IDS)
    assert [view["view_id"] for view in sequence["views"]] == list(DEFAULT_VIEW_IDS)
    assert [len(view["objects"]) for view in sequence["views"]] == [2, 2, 1, 1]


def test_load_sequence_objects_accepts_manual_entry_ids_in_order(tmp_path):
    db_dir = _make_sequence_db(tmp_path)
    sequence = load_sequence_objects(str(db_dir), entry_ids=[24, 19, 65])

    assert sequence["selected_view_ids"] == ["view_00024", "view_00019", "view_00065"]
    assert [view["view_id"] for view in sequence["views"]] == ["view_00024", "view_00019", "view_00065"]


def test_store_selected_view_images_renders_overlay_with_object_ids(tmp_path):
    db_dir = _make_sequence_db(tmp_path)
    run_dir = tmp_path / "seq_run"
    views = load_sequence_objects(str(db_dir), view_ids=["view_00019"])["views"]

    stored = sequential_spectral_experiment._store_selected_view_images(run_dir, str(db_dir), views)

    assert len(stored) == 1
    overlay_path = Path(stored[0]["stored_detection_overlay_path"])
    assert stored[0]["detection_overlay_status"] == "rendered"
    assert stored[0]["source_detection_overlay_path"] is None
    assert overlay_path.exists()
    source_image = cv2.imread(str(db_dir / views[0]["file_name"]), cv2.IMREAD_COLOR)
    overlay_image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    assert source_image is not None
    assert overlay_image is not None
    assert source_image.shape == overlay_image.shape
    assert np.any(source_image != overlay_image)


def test_store_selected_view_images_skips_rows_without_bbox_or_object_id(tmp_path):
    db_dir = _make_sequence_db(tmp_path)
    run_dir = tmp_path / "seq_run_skip"
    view = load_sequence_objects(str(db_dir), view_ids=["view_00024"])["views"][0]
    view_with_invalid_rows = dict(view)
    view_with_invalid_rows["objects"] = list(view["objects"]) + [
        {"label": "ghost", "bbox_xyxy": [8.0, 8.0, 30.0, 30.0]},
        {"object_global_id": 99, "label": "ghost"},
    ]

    stored = sequential_spectral_experiment._store_selected_view_images(
        run_dir, str(db_dir), [view_with_invalid_rows]
    )

    overlay_path = Path(stored[0]["stored_detection_overlay_path"])
    assert stored[0]["detection_overlay_status"] == "rendered"
    assert overlay_path.exists()


def test_apply_incremental_step_appends_best_candidate_and_spawns_tail():
    memory_row = {
        "object_global_id": 1,
        "view_id": "view_00019",
        "label": "chair",
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "estimated_global_x": 0.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    current_rows = [
        {
            "object_global_id": 2,
            "view_id": "view_00024",
            "label": "wooden seat",
            "embedding": np.asarray([0.99, 0.01], dtype=np.float32),
            "estimated_global_x": 0.1,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.1,
            "distance_from_camera_m": 1.1,
            "relative_bearing_deg": 2.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "object_global_id": 3,
            "view_id": "view_00024",
            "label": "plant",
            "embedding": np.asarray([0.0, 1.0], dtype=np.float32),
            "estimated_global_x": 8.0,
            "estimated_global_y": 0.0,
            "estimated_global_z": 1.0,
            "distance_from_camera_m": 3.0,
            "relative_bearing_deg": 30.0,
            "relative_height_from_camera_m": 0.0,
        },
    ]
    memory_clusters = [_build_cluster(0, [memory_row])]
    cross_affinity = np.asarray([[0.9], [0.1]], dtype=np.float32)
    full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=0.35)
    result = apply_incremental_step(
        memory_clusters,
        current_rows,
        cross_affinity=cross_affinity,
        cross_details=[[], []],
        full_affinity=full_affinity,
        spectral_result={"labels": np.asarray([0, 0, 1], dtype=np.int32)},
        step_index=1,
        next_cluster_id=1,
    )

    assert result["num_appended"] == 1
    assert result["num_new_tail_clusters"] == 1
    cluster_ids = [cluster["cluster_id"] for cluster in result["memory_clusters"]]
    assert cluster_ids == [0, 1]
    merged = result["memory_clusters"][0]
    assert merged["member_object_ids"] == [1, 2]
    assert result["tail_spawn_cases"][0]["object_ids"] == [3]
    diagnostics_by_object = {
        item["object_global_id"]: item for item in result["assignment_diagnostics"]
    }
    assert diagnostics_by_object[2]["assignment_reason"] == "component_best_append"
    assert diagnostics_by_object[2]["similarity_detail_status"] == "assigned_match"
    assert diagnostics_by_object[2]["cluster_id_at_assignment"] == 0
    assert diagnostics_by_object[3]["assignment_reason"] == "current_only_component"
    assert diagnostics_by_object[3]["similarity_detail_status"] == "best_rejected_candidate"
    assert diagnostics_by_object[3]["cluster_id_at_assignment"] == 1
    assert diagnostics_by_object[3]["similarity_reference_cluster_id"] == 0


def test_apply_incremental_step_merges_memory_clusters_when_same_component():
    memory_a = {
        "object_global_id": 10,
        "view_id": "view_00019",
        "label": "chair",
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "estimated_global_x": 0.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    memory_b = {
        "object_global_id": 11,
        "view_id": "view_00024",
        "label": "wooden seat",
        "embedding": np.asarray([0.98, 0.02], dtype=np.float32),
        "estimated_global_x": 0.1,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.1,
        "relative_bearing_deg": 1.0,
        "relative_height_from_camera_m": 0.0,
    }
    current = {
        "object_global_id": 12,
        "view_id": "view_00058",
        "label": "chair",
        "embedding": np.asarray([0.99, 0.01], dtype=np.float32),
        "estimated_global_x": 0.05,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.05,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    memory_clusters = [_build_cluster(0, [memory_a]), _build_cluster(1, [memory_b])]
    cross_affinity = np.asarray([[0.9, 0.85]], dtype=np.float32)
    full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=0.35)
    result = apply_incremental_step(
        memory_clusters,
        [current],
        cross_affinity=cross_affinity,
        cross_details=[[]],
        full_affinity=full_affinity,
        spectral_result={"labels": np.asarray([0, 0, 0], dtype=np.int32)},
        step_index=2,
        next_cluster_id=2,
    )

    assert result["num_merged_clusters"] == 1
    assert result["num_appended"] == 1
    assert len(result["memory_clusters"]) == 1
    merged_cluster = result["memory_clusters"][0]
    assert merged_cluster["cluster_id"] == 0
    assert merged_cluster["member_object_ids"] == [10, 11, 12]


def test_full_bipartite_affinity_uses_lower_default_cross_affinity_threshold():
    cross_affinity = np.asarray([[0.30, 0.20]], dtype=np.float32)

    full_affinity = _full_bipartite_affinity(cross_affinity)

    assert np.isclose(DEFAULT_CROSS_AFFINITY_MIN, 0.25)
    assert np.isclose(full_affinity[2, 0], 0.30)
    assert np.isclose(full_affinity[2, 1], 0.0)


def test_run_capped_sequential_spectral_clustering_caps_eigengap_to_cc_plus_two(monkeypatch):
    affinity = np.eye(6, dtype=np.float32)
    affinity[0, 3] = affinity[3, 0] = 0.90
    affinity[1, 4] = affinity[4, 1] = 0.88
    affinity[2, 5] = affinity[5, 2] = 0.86
    monkeypatch.setattr(sequential_spectral_experiment, "estimate_cluster_count_eigengap", lambda *_args, **_kwargs: 6)

    result = _run_capped_sequential_spectral_clustering(affinity, object_ids=list(range(6)))

    assert result["cluster_count_mode"] == "eigengap_capped"
    assert result["connected_component_count"] == 3
    assert result["max_allowed_n_clusters"] == 3 + DEFAULT_SPECTRAL_MAX_EXTRA_CLUSTERS
    assert result["eigengap_n_clusters"] == 6
    assert result["requested_n_clusters"] == result["max_allowed_n_clusters"]
    assert result["n_clusters"] <= result["requested_n_clusters"]


def test_build_cross_affinity_matrix_uses_cosine_geo_gate_on_xz_distance():
    memory_clusters = [
        _build_cluster(
            0,
            [
                {
                    "object_global_id": 1,
                    "view_id": "view_00019",
                    "label": "chair",
                    "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                    "estimated_global_x": 0.0,
                    "estimated_global_y": 10.0,
                    "estimated_global_z": 0.0,
                    "distance_from_camera_m": 1.0,
                    "relative_bearing_deg": 0.0,
                    "relative_height_from_camera_m": 0.0,
                }
            ],
        )
    ]
    current_rows = [
        {
            "object_global_id": 2,
            "view_id": "view_00024",
            "label": "chair",
            "embedding": np.asarray([0.8, 0.6], dtype=np.float32),
            "estimated_global_x": 3.0,
            "estimated_global_y": -10.0,
            "estimated_global_z": 4.0,
            "distance_from_camera_m": 5.0,
            "relative_bearing_deg": 15.0,
            "relative_height_from_camera_m": 0.0,
        }
    ]

    matrix, details = build_cross_affinity_matrix(
        memory_clusters,
        current_rows,
        similarity_mode="cosine_geo_gate",
        distance_gate_dsq0=2.0,
    )

    expected_cosine = float(np.dot(current_rows[0]["embedding"], memory_clusters[0]["prototype_embedding"]))
    expected_dsq = 3.0**2 + 4.0**2
    expected_gate = float(np.exp(-(expected_dsq / (2.0 * 2.0))))
    assert np.isclose(matrix[0, 0], expected_cosine * expected_gate)
    detail = details[0][0]
    assert detail["similarity_mode"] == "cosine_geo_gate"
    assert np.isclose(detail["distance_gate"], expected_gate)
    assert detail["xz_distance_m"] == 5.0
    assert detail["xz_distance_sq_m2"] == expected_dsq
    assert detail["polar_similarity"] is None


def test_build_cross_affinity_matrix_legacy_mode_keeps_weighted_similarity():
    memory_clusters = [
        _build_cluster(
            0,
            [
                {
                    "object_global_id": 1,
                    "view_id": "view_00019",
                    "label": "chair",
                    "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                    "estimated_global_x": 0.0,
                    "estimated_global_y": 0.0,
                    "estimated_global_z": 0.0,
                    "distance_from_camera_m": 1.0,
                    "relative_bearing_deg": 0.0,
                    "relative_height_from_camera_m": 0.0,
                }
            ],
        )
    ]
    current_rows = [
        {
            "object_global_id": 2,
            "view_id": "view_00024",
            "label": "chair",
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
            "estimated_global_x": 0.0,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": 0.0,
            "relative_height_from_camera_m": 0.0,
        }
    ]

    matrix, details = build_cross_affinity_matrix(
        memory_clusters,
        current_rows,
        similarity_mode="legacy_weighted_fusion",
    )

    assert np.isclose(matrix[0, 0], 1.0)
    assert details[0][0]["similarity_mode"] == "legacy_weighted_fusion"
    assert details[0][0]["distance_gate"] is None


def _affinity_detail(score: float) -> dict:
    return {
        "combined_similarity": float(score),
        "text_similarity": float(score),
        "global_geo_similarity": None,
        "polar_similarity": None,
        "global_geo_distance_m": None,
        "used_3d_global_geo": False,
    }


def test_apply_incremental_step_blocks_same_view_bridge_merge():
    memory_a = {
        "object_global_id": 20,
        "view_id": "view_00019",
        "label": "cabinet",
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "estimated_global_x": 0.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    memory_b = {
        "object_global_id": 21,
        "view_id": "view_00019",
        "label": "cabinet",
        "embedding": np.asarray([0.98, 0.02], dtype=np.float32),
        "estimated_global_x": 0.5,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.1,
        "relative_bearing_deg": 6.0,
        "relative_height_from_camera_m": 0.0,
    }
    current = {
        "object_global_id": 22,
        "view_id": "view_00024",
        "label": "wide cabinet view",
        "embedding": np.asarray([0.99, 0.01], dtype=np.float32),
        "estimated_global_x": 0.25,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 3.0,
        "relative_height_from_camera_m": 0.0,
    }

    memory_clusters = [_build_cluster(0, [memory_a]), _build_cluster(1, [memory_b])]
    cross_affinity = np.asarray([[0.91, 0.87]], dtype=np.float32)
    full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=0.35)
    result = apply_incremental_step(
        memory_clusters,
        [current],
        cross_affinity=cross_affinity,
        cross_details=[[_affinity_detail(0.91), _affinity_detail(0.87)]],
        full_affinity=full_affinity,
        spectral_result={"labels": np.asarray([0, 0, 0], dtype=np.int32)},
        step_index=1,
        next_cluster_id=2,
    )

    assert result["num_merged_clusters"] == 0
    assert result["num_same_view_blocked_components"] == 1
    assert result["num_appended"] == 1
    assert result["num_new_tail_clusters"] == 0
    assert [cluster["cluster_id"] for cluster in result["memory_clusters"]] == [0, 1]
    assert result["memory_clusters"][0]["member_object_ids"] == [20, 22]
    assert result["memory_clusters"][1]["member_object_ids"] == [21]
    block_case = result["same_view_block_cases"][0]
    assert block_case["blocked_merge_cluster_ids"] == [0, 1]
    assert block_case["collision_pairs"][0]["shared_view_ids"] == ["view_00019"]


def test_apply_incremental_step_same_view_block_competitive_matching():
    memory_a = {
        "object_global_id": 30,
        "view_id": "view_00019",
        "label": "cabinet",
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "estimated_global_x": 0.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    memory_b = {
        "object_global_id": 31,
        "view_id": "view_00019",
        "label": "cabinet",
        "embedding": np.asarray([0.0, 1.0], dtype=np.float32),
        "estimated_global_x": 2.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 25.0,
        "relative_height_from_camera_m": 0.0,
    }
    current_rows = [
        {
            "object_global_id": 32,
            "view_id": "view_00024",
            "label": "cabinet left",
            "embedding": np.asarray([0.99, 0.01], dtype=np.float32),
            "estimated_global_x": 0.1,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": 2.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "object_global_id": 33,
            "view_id": "view_00024",
            "label": "cabinet right",
            "embedding": np.asarray([0.02, 0.98], dtype=np.float32),
            "estimated_global_x": 1.9,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": 22.0,
            "relative_height_from_camera_m": 0.0,
        },
    ]

    memory_clusters = [_build_cluster(0, [memory_a]), _build_cluster(1, [memory_b])]
    cross_affinity = np.asarray([[0.95, 0.82], [0.71, 0.93]], dtype=np.float32)
    full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=0.35)
    result = apply_incremental_step(
        memory_clusters,
        current_rows,
        cross_affinity=cross_affinity,
        cross_details=[
            [_affinity_detail(0.95), _affinity_detail(0.82)],
            [_affinity_detail(0.71), _affinity_detail(0.93)],
        ],
        full_affinity=full_affinity,
        spectral_result={"labels": np.asarray([0, 0, 0, 0], dtype=np.int32)},
        step_index=1,
        next_cluster_id=2,
    )

    assert result["num_merged_clusters"] == 0
    assert result["num_same_view_blocked_components"] == 1
    assert result["num_appended"] == 2
    assert result["num_new_tail_clusters"] == 0
    assert [cluster["member_object_ids"] for cluster in result["memory_clusters"]] == [[30, 32], [31, 33]]


def test_apply_incremental_step_same_view_block_records_rejected_tail_detail():
    memory_a = {
        "object_global_id": 40,
        "view_id": "view_00019",
        "label": "cabinet",
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "estimated_global_x": 0.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    memory_b = {
        "object_global_id": 41,
        "view_id": "view_00019",
        "label": "cabinet",
        "embedding": np.asarray([0.0, 1.0], dtype=np.float32),
        "estimated_global_x": 2.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 20.0,
        "relative_height_from_camera_m": 0.0,
    }
    current_rows = [
        {
            "object_global_id": 42,
            "view_id": "view_00024",
            "label": "cabinet left",
            "embedding": np.asarray([0.99, 0.01], dtype=np.float32),
            "estimated_global_x": 0.1,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": 1.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "object_global_id": 43,
            "view_id": "view_00024",
            "label": "cabinet center",
            "embedding": np.asarray([0.7, 0.3], dtype=np.float32),
            "estimated_global_x": 0.8,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": 10.0,
            "relative_height_from_camera_m": 0.0,
        },
        {
            "object_global_id": 44,
            "view_id": "view_00024",
            "label": "cabinet right",
            "embedding": np.asarray([0.01, 0.99], dtype=np.float32),
            "estimated_global_x": 1.9,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.0,
            "distance_from_camera_m": 1.0,
            "relative_bearing_deg": 22.0,
            "relative_height_from_camera_m": 0.0,
        },
    ]
    memory_clusters = [_build_cluster(0, [memory_a]), _build_cluster(1, [memory_b])]
    cross_affinity = np.asarray([[0.95, 0.30], [0.88, 0.55], [0.35, 0.93]], dtype=np.float32)
    full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=0.25)
    result = apply_incremental_step(
        memory_clusters,
        current_rows,
        cross_affinity=cross_affinity,
        cross_details=[
            [_affinity_detail(0.95), _affinity_detail(0.30)],
            [_affinity_detail(0.88), _affinity_detail(0.55)],
            [_affinity_detail(0.35), _affinity_detail(0.93)],
        ],
        full_affinity=full_affinity,
        spectral_result={"labels": np.asarray([0, 0, 0, 0, 0], dtype=np.int32)},
        step_index=1,
        next_cluster_id=2,
    )

    diagnostics_by_object = {
        item["object_global_id"]: item for item in result["assignment_diagnostics"]
    }
    assert diagnostics_by_object[43]["assignment_reason"] == "tail_after_same_view_hard_block_competition"
    assert diagnostics_by_object[43]["similarity_detail_status"] == "best_rejected_candidate"
    assert diagnostics_by_object[43]["similarity_reference_cluster_id"] == 0


def test_apply_incremental_step_reattaches_current_only_high_score_match():
    memory = {
        "object_global_id": 50,
        "view_id": "view_00019",
        "label": "chair",
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "estimated_global_x": 0.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    current = {
        "object_global_id": 51,
        "view_id": "view_00024",
        "label": "chair",
        "embedding": np.asarray([0.99, 0.01], dtype=np.float32),
        "estimated_global_x": 0.1,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 1.0,
        "relative_height_from_camera_m": 0.0,
    }
    memory_clusters = [_build_cluster(0, [memory])]
    cross_affinity = np.asarray([[0.82]], dtype=np.float32)
    full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=0.35)
    result = apply_incremental_step(
        memory_clusters,
        [current],
        cross_affinity=cross_affinity,
        cross_details=[[_affinity_detail(0.82)]],
        full_affinity=full_affinity,
        spectral_result={"labels": np.asarray([0, 1], dtype=np.int32)},
        step_index=1,
        next_cluster_id=1,
        current_only_reattach_min_affinity=0.75,
    )

    assert result["num_appended"] == 1
    assert result["num_current_only_reattached"] == 1
    assert result["num_new_tail_clusters"] == 0
    assert result["memory_clusters"][0]["member_object_ids"] == [50, 51]
    assert result["append_cases"][0]["reason"] == "current_only_high_score_reattach"


def test_apply_incremental_step_keeps_current_only_singleton_when_score_too_low():
    memory = {
        "object_global_id": 60,
        "view_id": "view_00019",
        "label": "chair",
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "estimated_global_x": 0.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    current = {
        "object_global_id": 61,
        "view_id": "view_00024",
        "label": "chair",
        "embedding": np.asarray([0.0, 1.0], dtype=np.float32),
        "estimated_global_x": 5.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 0.0,
        "distance_from_camera_m": 5.0,
        "relative_bearing_deg": 90.0,
        "relative_height_from_camera_m": 0.0,
    }
    memory_clusters = [_build_cluster(0, [memory])]
    cross_affinity = np.asarray([[0.72]], dtype=np.float32)
    full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=0.35)
    result = apply_incremental_step(
        memory_clusters,
        [current],
        cross_affinity=cross_affinity,
        cross_details=[[_affinity_detail(0.72)]],
        full_affinity=full_affinity,
        spectral_result={"labels": np.asarray([0, 1], dtype=np.int32)},
        step_index=1,
        next_cluster_id=1,
        current_only_reattach_min_affinity=0.75,
    )

    assert result["num_appended"] == 0
    assert result["num_current_only_reattached"] == 0
    assert result["num_new_tail_clusters"] == 1
    assert [cluster["member_object_ids"] for cluster in result["memory_clusters"]] == [[60], [61]]
    diagnostic = result["assignment_diagnostics"][0]
    assert diagnostic["object_global_id"] == 61
    assert diagnostic["assignment_reason"] == "current_only_component"
    assert diagnostic["similarity_detail_status"] == "best_rejected_candidate"
    assert diagnostic["similarity_reference_cluster_id"] == 0


def test_apply_incremental_step_marks_no_candidate_detail_when_no_memory_exists():
    current = {
        "object_global_id": 70,
        "view_id": "view_00024",
        "label": "chair",
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "estimated_global_x": 1.0,
        "estimated_global_y": 0.0,
        "estimated_global_z": 1.0,
        "distance_from_camera_m": 1.0,
        "relative_bearing_deg": 0.0,
        "relative_height_from_camera_m": 0.0,
    }
    cross_affinity = np.zeros((1, 0), dtype=np.float32)
    full_affinity = _full_bipartite_affinity(cross_affinity, min_cross_affinity=0.25)
    result = apply_incremental_step(
        [],
        [current],
        cross_affinity=cross_affinity,
        cross_details=[[]],
        full_affinity=full_affinity,
        spectral_result={"labels": np.asarray([0], dtype=np.int32)},
        step_index=1,
        next_cluster_id=0,
    )

    diagnostic = result["assignment_diagnostics"][0]
    assert diagnostic["assignment_reason"] == "current_only_component"
    assert diagnostic["similarity_detail_status"] == "no_candidate_detail"
    assert diagnostic["similarity_reference_cluster_id"] is None


def test_run_sequential_spectral_experiment_writes_artifacts(tmp_path):
    db_dir = _make_sequence_db(tmp_path)
    output_dir = tmp_path / "seq_out"

    report = run_sequential_spectral_experiment(str(db_dir), output_dir=str(output_dir))
    run_dir = output_dir / Path(report["output_dir"]).name

    assert report["selected_view_ids"] == list(DEFAULT_VIEW_IDS)
    assert run_dir.exists()
    assert (run_dir / "sequence_manifest.json").exists()
    assert (run_dir / "step_00_initial_registry.json").exists()
    assert (run_dir / "global_object_list_final.json").exists()
    assert (run_dir / "experiment_report.json").exists()
    assert (run_dir / "object_cluster_similarity_table.csv").exists()
    for step in (0, 1, 2, 3):
        assert (run_dir / f"step_{step:02d}_object_assignment_table.csv").exists()
    assert (run_dir / "cumulative_cluster_progression_manifest.json").exists()
    assert (run_dir / "cumulative_cluster_progression_overview.png").exists()
    assert (run_dir / "selected_view_images").exists()
    for view_id in DEFAULT_VIEW_IDS:
        assert (run_dir / "selected_view_images" / f"{view_id}.jpg").exists()
        assert (run_dir / "selected_view_images" / f"{view_id}_yolo_overlay.jpg").exists()
    for stage in (0, 1, 2, 3):
        assert (run_dir / f"cumulative_cluster_matrix_step_{stage:02d}.npy").exists()
        assert (run_dir / f"cumulative_cluster_matrix_step_{stage:02d}.csv").exists()
        assert (run_dir / f"cumulative_cluster_matrix_step_{stage:02d}.png").exists()
        assert (run_dir / f"cumulative_cluster_matrix_step_{stage:02d}.json").exists()
    for step in (1, 2, 3):
        assert (run_dir / f"step_{step:02d}_cross_affinity_matrix.npy").exists()
        assert (run_dir / f"step_{step:02d}_cross_affinity_matrix.csv").exists()
        assert (run_dir / f"cross_affinity_laplacian_step_{step:02d}.npy").exists()
        assert (run_dir / f"cross_affinity_laplacian_step_{step:02d}.csv").exists()
        assert (run_dir / f"cross_affinity_laplacian_step_{step:02d}.png").exists()
        assert (run_dir / f"step_{step:02d}_affinity_matrix.npy").exists()
        assert (run_dir / f"step_{step:02d}_affinity_matrix.csv").exists()
        assert (run_dir / f"spectral_block_heatmap_step_{step:02d}.png").exists()
        assert (run_dir / f"step_{step:02d}_cocluster_matrix.npy").exists()
        assert (run_dir / f"step_{step:02d}_cocluster_matrix.csv").exists()
        assert (run_dir / f"cocluster_laplacian_step_{step:02d}.npy").exists()
        assert (run_dir / f"cocluster_laplacian_step_{step:02d}.csv").exists()
        assert (run_dir / f"step_{step:02d}_cluster_update.json").exists()
        assert (run_dir / f"affinity_heatmap_step_{step:02d}.png").exists()
        assert (run_dir / f"cocluster_heatmap_step_{step:02d}.png").exists()
        assert (run_dir / f"cocluster_laplacian_step_{step:02d}.png").exists()
    step_report = json.loads((run_dir / "step_01_cluster_update.json").read_text(encoding="utf-8"))
    assert "spectral_summary" in step_report
    assert "spectral_result" not in step_report
    assert "num_connected_components_after_spectral" in step_report
    assert "cocluster_shape" in step_report
    assert step_report["clusters_after_step"]
    assert set(step_report["clusters_after_step"][0].keys()) == {
        "cluster_id",
        "members",
        "member_view_ids",
        "label_histogram",
    }
    assert isinstance(step_report["clusters_after_step"][0]["members"][0], str)
    assert "(" in step_report["clusters_after_step"][0]["members"][0]
    final_registry = json.loads((run_dir / "global_object_list_final.json").read_text(encoding="utf-8"))
    assert final_registry
    assert set(final_registry[0].keys()) == {
        "cluster_id",
        "members",
        "member_view_ids",
        "label_histogram",
    }
    assert isinstance(final_registry[0]["members"][0], str)
    assert "(" in final_registry[0]["members"][0]
    experiment_report = json.loads((run_dir / "experiment_report.json").read_text(encoding="utf-8"))
    assert experiment_report["object_cluster_similarity_table"].endswith(
        "object_cluster_similarity_table.csv"
    )
    assert len(experiment_report["object_cluster_similarity_tables_by_step"]) == 4
    assert experiment_report["object_cluster_similarity_tables_by_step"][0]["path"].endswith(
        "step_00_object_assignment_table.csv"
    )
    assert "step_summaries" in experiment_report
    assert "steps" not in experiment_report
    assert experiment_report["cumulative_cluster_progression_manifest"].endswith(
        "cumulative_cluster_progression_manifest.json"
    )
    progression_manifest = json.loads(
        (run_dir / "cumulative_cluster_progression_manifest.json").read_text(encoding="utf-8")
    )
    assert len(progression_manifest["steps"]) == 4
    assert progression_manifest["overview_path"].endswith("cumulative_cluster_progression_overview.png")
    assert "num_current_only_reattached" in experiment_report["step_summaries"][0]
    assert "num_same_view_blocked_components" in experiment_report["step_summaries"][0]
    assert "current_only_reattach_cases" in step_report
    assert "same_view_block_cases" in step_report
    assert "total_current_only_reattached" in experiment_report
    assert "total_same_view_blocked_components" in experiment_report
    first_progression_step = json.loads(
        (run_dir / "cumulative_cluster_matrix_step_00.json").read_text(encoding="utf-8")
    )
    assert first_progression_step["axis_labels"]
    assert first_progression_step["axis_labels"][0].startswith("obj")
    assert "|" in first_progression_step["axis_labels"][0]
    assert "final_clusters" in experiment_report
    assert experiment_report["step_summaries"]
    assert experiment_report["final_clusters"]
    assert "append_case_examples" in experiment_report
    assert "tail_spawn_case_examples" in experiment_report
    assert experiment_report["views"][0]["stored_detection_overlay_path"].endswith("_yolo_overlay.jpg")
    assert experiment_report["views"][0]["detection_overlay_status"] == "rendered"
    with (run_dir / "object_cluster_similarity_table.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    seed_rows = [row for row in rows if row["assignment_reason"] == "initial_seed"]
    assert len(seed_rows) == 2
    assert seed_rows[0]["similarity_detail_status"] == "initial_seed"
    assert seed_rows[0]["term1_cosine"] == ""
    matched_rows = [
        row for row in rows if row["similarity_detail_status"] == "assigned_match" and row["term1_cosine"]
    ]
    assert matched_rows
    sample = matched_rows[0]
    cosine = float(sample["term1_cosine"])
    gate = float(sample["distance_gate"])
    combined = float(sample["combined_similarity"])
    dsq = float(sample["dsq"])
    dsq0 = float(sample["distance_gate_dsq0"])
    exponent = float(sample["distance_gate_exponent"])
    assert np.isclose(combined, cosine * gate)
    assert np.isclose(exponent, -(dsq / (2.0 * dsq0)))
    with (run_dir / "step_00_object_assignment_table.csv").open("r", encoding="utf-8", newline="") as handle:
        step0_rows = list(csv.DictReader(handle))
    with (run_dir / "step_01_object_assignment_table.csv").open("r", encoding="utf-8", newline="") as handle:
        step1_rows = list(csv.DictReader(handle))
    with (run_dir / "step_02_object_assignment_table.csv").open("r", encoding="utf-8", newline="") as handle:
        step2_rows = list(csv.DictReader(handle))
    with (run_dir / "step_03_object_assignment_table.csv").open("r", encoding="utf-8", newline="") as handle:
        step3_rows = list(csv.DictReader(handle))
    assert [len(step0_rows), len(step1_rows), len(step2_rows), len(step3_rows)] == [2, 2, 1, 1]
    assert report["final_cluster_count"] >= 2


def test_run_sequential_spectral_experiment_supports_dsq0_sweep(tmp_path):
    db_dir = _make_sequence_db(tmp_path)
    output_dir = tmp_path / "seq_sweep"

    report = run_sequential_spectral_experiment(
        str(db_dir),
        output_dir=str(output_dir),
        distance_gate_dsq0_values=DEFAULT_DISTANCE_GATE_DSQ0_SWEEP[:2],
    )

    sweep_dir = output_dir / Path(report["output_dir"]).name
    assert report["distance_gate_dsq0_values"] == [0.5, 1.0]
    assert len(report["runs"]) == 2
    assert (sweep_dir / "sweep_summary.json").exists()
    for run in report["runs"]:
        assert Path(run["output_dir"]).exists()
        assert run["dsq0"] in {0.5, 1.0}
        assert Path(run["object_cluster_similarity_table"]).exists()
