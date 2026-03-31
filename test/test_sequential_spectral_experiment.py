import json
from pathlib import Path

import numpy as np

from spatial_rag.sequential_spectral_experiment import (
    DEFAULT_VIEW_IDS,
    _build_cluster,
    _full_bipartite_affinity,
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
    for row in meta_rows:
        image_path = db_dir / row["file_name"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")
        view_id = f"view_{int(row['id']):05d}"
        overlay_path = db_dir / "geometry" / view_id / "detection_overlay.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_bytes(b"fake-overlay")
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
    assert [len(view["objects"]) for view in sequence["views"]] == [2, 2, 1]


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
    assert experiment_report["views"][0]["detection_overlay_status"] == "copied"
    assert report["final_cluster_count"] >= 2
