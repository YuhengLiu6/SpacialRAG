import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import spatial_rag.reweight_sweep as reweight_sweep_module
from spatial_rag.occlusion_scoring import compute_reweighted_detection_score
from spatial_rag.spectral_threshold_ablation import _build_arg_parser, run_spectral_threshold_ablation
from spatial_rag.spatial_db_builder import _write_jsonl


def _fake_text_embedding(text: str) -> np.ndarray:
    chars = [ord(ch) for ch in str(text)]
    base = np.asarray(
        [
            float(sum(chars) % 17 + 1),
            float(len(chars) % 13 + 1),
            float((sum(chars[::2]) if chars else 0) % 19 + 1),
            float((sum(chars[1::2]) if chars else 0) % 23 + 1),
        ],
        dtype=np.float32,
    )
    base /= np.linalg.norm(base)
    return base.astype(np.float32)


class _FakeEmbedder:
    def embed_text(self, text):
        return _fake_text_embedding(text)


def _fake_save_faiss_index(arr: np.ndarray, path: Path) -> int:
    Path(path).write_bytes(b"fake-faiss")
    return int(arr.shape[0])


def _write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _meta_row(entry_id: int, x: float, z: float, orientation: int, text: str, object_count: int) -> dict:
    return {
        "id": int(entry_id),
        "frame_id": int(entry_id),
        "x": float(x),
        "y": 0.0,
        "z": float(z),
        "world_position": [float(x), 0.0, float(z)],
        "orientation": int(orientation),
        "file_name": f"images/pose_{entry_id:05d}.jpg",
        "parse_status": "ok",
        "text": text,
        "frame_text_short": text,
        "frame_text_long": f"{text} long",
        "text_input_for_clip_short": text,
        "text_input_for_clip_long": f"{text} long",
        "object_text_inputs_short": [text],
        "object_text_inputs_long": [f"{text} long"],
        "room_function": "living room",
        "view_type": "living room",
        "attribute": {"view_type": "living room", "scene_attributes": []},
        "object_count": int(object_count),
    }


def _object_row(
    object_global_id: int,
    entry_id: int,
    *,
    label: str,
    short_text: str,
    long_text: str,
    score: float,
    x: float,
    z: float,
    bearing: float,
    bbox_xyxy,
) -> dict:
    return {
        "object_global_id": int(object_global_id),
        "entry_id": int(entry_id),
        "frame_id": int(entry_id),
        "view_id": f"view_{entry_id:05d}",
        "file_name": f"images/pose_{entry_id:05d}.jpg",
        "object_local_id": f"det_{object_global_id:03d}",
        "label": label,
        "description": short_text,
        "long_form_open_description": long_text,
        "text_input_for_clip_short": short_text,
        "text_input_for_clip_long": long_text,
        "object_text_short": short_text,
        "object_text_long": long_text,
        "object_confidence": float(score),
        "detector_confidence": float(score),
        "occlusion_level": "fully visible",
        "occlusion_penalty_p_o": 0.0,
        "reweighted_detection_score_r": compute_reweighted_detection_score(float(score), "fully visible"),
        "geometry_source": "mask_depth",
        "estimated_global_x": float(x),
        "estimated_global_y": 0.0,
        "estimated_global_z": float(z),
        "distance_from_camera_m": float(max(abs(x), abs(z), 0.5)),
        "relative_bearing_deg": float(bearing),
        "relative_height_from_camera_m": 0.0,
        "orientation": 0,
        "object_orientation_deg": 0.0,
        "angle_bucket": "center",
        "laterality": "center",
        "distance_bin": "middle",
        "verticality": "middle",
        "location_relative_to_other_objects": "",
        "parse_status": "ok",
        "bbox_xyxy": list(bbox_xyxy),
        "bbox_xywh_norm": [0.1, 0.1, 0.2, 0.2],
        "crop_path": f"canonical_db/geometry/view_{entry_id:05d}/objects/obj_{object_global_id:03d}_crop.jpg",
        "mask_path": f"canonical_db/geometry/view_{entry_id:05d}/objects/obj_{object_global_id:03d}_mask.png",
        "mask_overlay_path": f"canonical_db/geometry/view_{entry_id:05d}/objects/obj_{object_global_id:03d}_mask_overlay.jpg",
        "depth_map_path": f"canonical_db/geometry/view_{entry_id:05d}/depth_map.npy",
    }


def _make_ablation_db(tmp_path: Path, *, filtered: bool) -> Path:
    db_dir = tmp_path / ("filtered_ablation_db" if filtered else "canonical_ablation_db")
    db_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("images", "geometry", "overview", "vlm_cache", "vlm_object_cache"):
        (db_dir / dirname).mkdir(parents=True, exist_ok=True)

    entry_ids = [19, 24, 58, 65]
    orientations = [270, 0, 180, 90]
    base_texts = ["chair table", "seat plant", "table", "chair"]
    meta_rows = []
    for index, (entry_id, orientation, text) in enumerate(zip(entry_ids, orientations, base_texts)):
        image_path = db_dir / "images" / f"pose_{entry_id:05d}.jpg"
        canvas = np.full((240, 320, 3), 232 - index * 18, dtype=np.uint8)
        cv2.putText(
            canvas,
            f"view {entry_id}",
            (36, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        ok = cv2.imwrite(str(image_path), canvas)
        assert ok
        meta_rows.append(_meta_row(entry_id, float(index), 0.0, orientation, text, 0))

    object_rows_all = [
        _object_row(1, 19, label="chair", short_text="chair low", long_text="blue chair near sofa", score=0.45, x=0.0, z=0.0, bearing=0.0, bbox_xyxy=[12.0, 16.0, 96.0, 120.0]),
        _object_row(2, 19, label="table", short_text="table high", long_text="round wooden table", score=0.90, x=5.0, z=0.0, bearing=12.0, bbox_xyxy=[120.0, 24.0, 228.0, 132.0]),
        _object_row(3, 24, label="wooden seat", short_text="wooden seat", long_text="wooden dining chair", score=0.80, x=0.1, z=0.1, bearing=2.0, bbox_xyxy=[18.0, 110.0, 98.0, 210.0]),
        _object_row(4, 24, label="plant", short_text="small plant", long_text="potted indoor plant", score=0.35, x=8.0, z=1.0, bearing=35.0, bbox_xyxy=[170.0, 32.0, 250.0, 152.0]),
        _object_row(5, 58, label="table", short_text="table second view", long_text="round wooden table second view", score=0.88, x=5.1, z=0.2, bearing=10.0, bbox_xyxy=[108.0, 72.0, 220.0, 164.0]),
        _object_row(6, 65, label="chair", short_text="chair second view", long_text="blue chair from another view", score=0.80, x=0.2, z=0.0, bearing=-5.0, bbox_xyxy=[40.0, 54.0, 126.0, 180.0]),
    ]
    if filtered:
        object_rows = [row for row in object_rows_all if float(row["reweighted_detection_score_r"]) >= 0.5]
    else:
        object_rows = list(object_rows_all)

    objects_by_entry = {}
    for row in object_rows:
        objects_by_entry.setdefault(int(row["entry_id"]), []).append(row)
    for meta_row in meta_rows:
        kept_rows = objects_by_entry.get(int(meta_row["id"]), [])
        meta_row["object_count"] = len(kept_rows)
        if kept_rows:
            short_texts = [str(row["object_text_short"]) for row in kept_rows]
            long_texts = [str(row["object_text_long"]) for row in kept_rows]
            meta_row["text"] = " | ".join(short_texts)
            meta_row["frame_text_short"] = " | ".join(short_texts)
            meta_row["frame_text_long"] = " | ".join(long_texts)
            meta_row["text_input_for_clip_short"] = meta_row["frame_text_short"]
            meta_row["text_input_for_clip_long"] = meta_row["frame_text_long"]
            meta_row["object_text_inputs_short"] = short_texts
            meta_row["object_text_inputs_long"] = long_texts
        else:
            meta_row["text"] = "unknown"
            meta_row["frame_text_short"] = "unknown"
            meta_row["frame_text_long"] = "unknown"
            meta_row["text_input_for_clip_short"] = "unknown"
            meta_row["text_input_for_clip_long"] = "unknown"
            meta_row["object_text_inputs_short"] = ["unknown"]
            meta_row["object_text_inputs_long"] = ["unknown"]

    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "metadata.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)

    image_emb = np.vstack([_fake_text_embedding(f"image_{entry_id}") for entry_id in entry_ids]).astype(np.float32)
    text_emb_short = np.vstack([_fake_text_embedding(row["text_input_for_clip_short"]) for row in meta_rows]).astype(np.float32)
    text_emb_long = np.vstack([_fake_text_embedding(row["text_input_for_clip_long"]) for row in meta_rows]).astype(np.float32)
    object_emb_short_all = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.96, 0.04, 0.0],
            [0.0, 0.0, 1.0],
            [0.02, 0.97, 0.01],
            [0.98, 0.02, 0.0],
        ],
        dtype=np.float32,
    )
    object_emb_long_all = object_emb_short_all.copy()
    if filtered:
        keep_indices = [index for index, row in enumerate(object_rows_all) if float(row["reweighted_detection_score_r"]) >= 0.5]
        object_emb_short = object_emb_short_all[keep_indices]
        object_emb_long = object_emb_long_all[keep_indices]
    else:
        object_emb_short = object_emb_short_all
        object_emb_long = object_emb_long_all

    np.save(db_dir / "image_emb.npy", image_emb)
    np.save(db_dir / "text_emb_short.npy", text_emb_short)
    np.save(db_dir / "text_emb_long.npy", text_emb_long)
    np.save(db_dir / "object_text_emb_short.npy", object_emb_short)
    np.save(db_dir / "object_text_emb_long.npy", object_emb_long)

    pre_threshold_rows = []
    for row in object_rows_all:
        pre_threshold_rows.append(
            {
                "entry_id": row["entry_id"],
                "frame_id": row["frame_id"],
                "file_name": row["file_name"],
                "object_local_id": row["object_local_id"],
                "object_route": "geometry",
                "label": row["label"],
                "bbox_xyxy": json.dumps(row["bbox_xyxy"]),
                "bbox_xywh_norm": json.dumps(row["bbox_xywh_norm"]),
                "object_confidence": row["object_confidence"],
                "detector_confidence": row["detector_confidence"],
                "occlusion_level": row["occlusion_level"],
                "occlusion_penalty_p_o": row["occlusion_penalty_p_o"],
                "reweighted_detection_score_r": row["reweighted_detection_score_r"],
                "r_threshold_used": 0.5 if filtered else "",
                "would_be_filtered_by_r_threshold": "False",
            }
        )
    _write_csv(
        db_dir / "object_r_scores_pre_threshold.csv",
        [
            "entry_id",
            "frame_id",
            "file_name",
            "object_local_id",
            "object_route",
            "label",
            "bbox_xyxy",
            "bbox_xywh_norm",
            "object_confidence",
            "detector_confidence",
            "occlusion_level",
            "occlusion_penalty_p_o",
            "reweighted_detection_score_r",
            "r_threshold_used",
            "would_be_filtered_by_r_threshold",
        ],
        pre_threshold_rows,
    )
    _write_csv(
        db_dir / "object_r_scores.csv",
        ["object_global_id", "reweighted_detection_score_r"],
        [
            {
                "object_global_id": row["object_global_id"],
                "reweighted_detection_score_r": row["reweighted_detection_score_r"],
            }
            for row in object_rows
        ],
    )

    raw_api_rows = []
    timing_rows = []
    for meta_row in meta_rows:
        entry_id = int(meta_row["id"])
        before = sum(1 for row in object_rows_all if int(row["entry_id"]) == entry_id)
        after = sum(1 for row in object_rows if int(row["entry_id"]) == entry_id)
        raw_api_rows.append(
            {
                "entry_id": entry_id,
                "file_name": meta_row["file_name"],
                "geometry_artifacts": {
                    "detection_overlay_path": f"canonical_ablation_db/geometry/view_{entry_id:05d}/detection_overlay.jpg",
                    "depth_map_path": f"canonical_ablation_db/geometry/view_{entry_id:05d}/depth_map.npy",
                },
                "timing": {
                    "geometry_objects_before_r_threshold": before,
                    "geometry_objects_after_r_threshold": after,
                    "geometry_objects_filtered_by_r_threshold": before - after,
                    "geometry_all_objects_filtered_by_r_threshold": bool(before > 0 and after == 0),
                },
            }
        )
        timing_rows.append(
            {
                "entry_id": entry_id,
                "frame_idx": entry_id,
                "file_name": meta_row["file_name"],
                "object_count": meta_row["object_count"],
                "geometry_objects_before_r_threshold": before,
                "geometry_objects_after_r_threshold": after,
                "geometry_objects_filtered_by_r_threshold": before - after,
                "geometry_all_objects_filtered_by_r_threshold": bool(before > 0 and after == 0),
            }
        )
    _write_jsonl(db_dir / "raw_api_responses.jsonl", raw_api_rows)
    _write_jsonl(db_dir / "per_image_timings.jsonl", timing_rows)

    overview = db_dir / "overview"
    overview_canvas = np.full((100, 140, 3), 220, dtype=np.uint8)
    assert cv2.imwrite(str(overview / "center_highest_view.jpg"), overview_canvas)
    assert cv2.imwrite(str(overview / "textured_floor_plan.jpg"), overview_canvas)

    build_report = {
        "output_dir": str(db_dir),
        "scene_path": "data/fake_scene.glb",
        "builder_variant": "angle_split",
        "tour_mode": "random",
        "scan_angles": [0, 90, 180, 270],
        "random_config": {"scan_angles": [0, 90, 180, 270]},
        "r_threshold": 0.5 if filtered else None,
        "r_threshold_enabled": bool(filtered),
        "object_config": {
            "r_threshold": 0.5 if filtered else None,
            "r_threshold_enabled": bool(filtered),
            "occlusion_reweight": {"w1": 1.0, "w2": 1.0, "b": 0.0},
        },
        "total_entries": len(meta_rows),
        "total_objects": len(object_rows),
        "object_r_scores_pre_threshold_count": len(object_rows_all),
        "object_r_scores_count": len(object_rows),
        "overview_outputs": {
            "center_highest_view": str(overview / "center_highest_view.jpg"),
            "textured_floor_plan": str(overview / "textured_floor_plan.jpg"),
        },
    }
    (db_dir / "build_report.json").write_text(json.dumps(build_report, indent=2), encoding="utf-8")
    return db_dir


def test_threshold_ablation_writes_native_pipeline_outputs(tmp_path, monkeypatch):
    db_dir = _make_ablation_db(tmp_path, filtered=False)
    monkeypatch.setattr(reweight_sweep_module, "_make_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(reweight_sweep_module, "_save_faiss_index", _fake_save_faiss_index)

    report = run_spectral_threshold_ablation(
        db_dir=str(db_dir),
        entry_ids=[19, 24, 58, 65],
        thresholds=[None, 0.5],
        output_dir=str(tmp_path / "ablation_runs"),
    )

    run_root = Path(report["output_dir"])
    assert (run_root / "summary.json").exists()
    assert (run_root / "threshold_results.csv").exists()
    assert (run_root / "threshold_results.md").exists()
    assert len(report["threshold_runs"]) == 2

    summary_by_token = {row["threshold_token"]: row for row in report["threshold_runs"]}
    for token in ("none", "0p5"):
        threshold_root = run_root / f"threshold_{token}"
        assert threshold_root.exists()
        assert (threshold_root / "db_variant").exists()
        assert (threshold_root / "object_instance").exists()
        assert (threshold_root / "sequential").exists()

        object_instance_root = Path(summary_by_token[token]["object_instance_native_output_root"])
        selected_views_dir = object_instance_root / "selected_views"
        assert (object_instance_root / "summary.json").exists()
        assert (selected_views_dir / "cluster_summary.json").exists()
        assert (selected_views_dir / "cluster_summary.md").exists()
        assert (selected_views_dir / "similarity_matrix.npy").exists()
        assert (selected_views_dir / "affinity_matrix.npy").exists()
        assert (selected_views_dir / "clustered_similarity_matrix.csv").exists()
        assert (selected_views_dir / "similarity_heatmap.png").exists()
        assert (selected_views_dir / "affinity_heatmap.png").exists()
        assert (selected_views_dir / "clustered_similarity_heatmap.png").exists()
        assert (selected_views_dir / "clustered_similarity_heatmap_offdiag_only.png").exists()
        assert (selected_views_dir / "refined_graph_clustered_similarity_heatmap.png").exists()

        sequential_root = Path(summary_by_token[token]["sequential_native_output_root"])
        assert (sequential_root / "experiment_report.json").exists()
        assert (sequential_root / "step_00_initial_registry.json").exists()
        assert (sequential_root / "step_01_cross_affinity_matrix.npy").exists()
        assert (sequential_root / "step_01_cross_affinity_matrix.csv").exists()
        assert (sequential_root / "step_01_affinity_matrix.npy").exists()
        assert (sequential_root / "step_01_affinity_matrix.csv").exists()
        assert (sequential_root / "step_01_cocluster_matrix.npy").exists()
        assert (sequential_root / "step_01_cocluster_matrix.csv").exists()
        assert (sequential_root / "affinity_heatmap_step_01.png").exists()
        assert (sequential_root / "spectral_block_heatmap_step_01.png").exists()
        assert (sequential_root / "cocluster_heatmap_step_01.png").exists()
        assert (sequential_root / "cumulative_cluster_progression_overview.png").exists()
        assert (sequential_root / "object_cluster_similarity_table.csv").exists()

    rows = list(csv.DictReader((run_root / "threshold_results.csv").open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert {row["threshold"] for row in rows} == {"none", "0.5"}
    for row in rows:
        assert row["db_variant_root"]
        assert row["object_instance_native_output_root"]
        assert row["sequential_native_output_root"]


def test_threshold_ablation_propagates_sequential_dbscan_overrides(tmp_path, monkeypatch):
    db_dir = _make_ablation_db(tmp_path, filtered=False)
    monkeypatch.setattr(reweight_sweep_module, "_make_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(reweight_sweep_module, "_save_faiss_index", _fake_save_faiss_index)

    report = run_spectral_threshold_ablation(
        db_dir=str(db_dir),
        entry_ids=[19, 24, 58, 65],
        thresholds=[None],
        output_dir=str(tmp_path / "ablation_runs_dbscan_override"),
        weight_text=0.8,
        weight_dinov3=0.6,
        enable_dinov3_scoring=True,
        enable_vlm_compress=True,
        enable_vlm_member_spatial=True,
        distance_gate_dsq0=2.5,
        dbscan_eps=0.1,
        dbscan_min_samples=1,
        enforce_same_view_uniqueness=False,
    )

    assert report["sequential_overrides"] == {
        "weight_text": 0.8,
        "weight_dinov3": 0.6,
        "enable_dinov3_scoring": True,
        "enable_vlm_compress": True,
        "enable_vlm_member_spatial": True,
        "distance_gate_dsq0": 2.5,
        "dbscan_eps": 0.1,
        "dbscan_min_samples": 1,
        "enforce_same_view_uniqueness": False,
    }

    threshold_run = report["threshold_runs"][0]
    seq_report = threshold_run["sequential_report"]
    assert np.isclose(seq_report["weight_text"], 0.8)
    assert np.isclose(seq_report["weight_dinov3"], 0.6)
    assert seq_report["enable_dinov3_scoring"] is True
    assert seq_report["enable_vlm_compress"] is True
    assert seq_report["enable_vlm_member_spatial"] is True
    assert np.isclose(seq_report["distance_gate_dsq0"], 2.5)
    assert np.isclose(seq_report["dbscan_eps"], 0.1)
    assert seq_report["dbscan_min_samples"] == 1
    assert seq_report["enforce_same_view_uniqueness"] is False

    rows = list(csv.DictReader((Path(report["output_dir"]) / "threshold_results.csv").open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["sequential_weight_text"] == "0.8"
    assert rows[0]["sequential_weight_dinov3"] == "0.6"
    assert rows[0]["sequential_enable_dinov3_scoring"] == "True"
    assert rows[0]["sequential_enable_vlm_compress"] == "True"
    assert rows[0]["sequential_enable_vlm_member_spatial"] == "True"
    assert rows[0]["sequential_distance_gate_dsq0"] == "2.5"
    assert rows[0]["sequential_dbscan_eps"] == "0.1"
    assert rows[0]["sequential_dbscan_min_samples"] == "1"
    assert rows[0]["sequential_enforce_same_view_uniqueness"] == "False"

    experiment_report = json.loads(
        (Path(threshold_run["sequential_native_output_root"]) / "experiment_report.json").read_text(encoding="utf-8")
    )
    assert np.isclose(experiment_report["weights"]["text"], 0.8)
    assert np.isclose(experiment_report["weights"]["dinov3"], 0.6)
    assert experiment_report["enable_dinov3_scoring"] is True
    assert experiment_report["enable_vlm_compress"] is True
    assert experiment_report["enable_vlm_member_spatial"] is True
    assert np.isclose(experiment_report["distance_gate_dsq0"], 2.5)
    assert np.isclose(experiment_report["dbscan_eps"], 0.1)
    assert experiment_report["dbscan_min_samples"] == 1
    assert experiment_report["enforce_same_view_uniqueness"] is False


def test_threshold_ablation_can_export_filtered_object_crops(tmp_path, monkeypatch):
    db_dir = _make_ablation_db(tmp_path, filtered=False)
    monkeypatch.setattr(reweight_sweep_module, "_make_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(reweight_sweep_module, "_save_faiss_index", _fake_save_faiss_index)

    report = run_spectral_threshold_ablation(
        db_dir=str(db_dir),
        entry_ids=[19, 24, 58, 65],
        thresholds=[0.5],
        output_dir=str(tmp_path / "ablation_runs_filtered_obj"),
        export_filtered_objects=True,
    )

    threshold_run = report["threshold_runs"][0]
    filtered_dir = Path(threshold_run["filtered_object_dir"])
    manifest_path = Path(threshold_run["filtered_manifest_path"])

    assert filtered_dir.exists()
    assert manifest_path.exists()
    assert threshold_run["filtered_object_count"] == 2

    manifest_rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    assert len(manifest_rows) == 2
    assert all(Path(row["export_path"]).exists() for row in manifest_rows)

    rows = list(csv.DictReader((Path(report["output_dir"]) / "threshold_results.csv").open("r", encoding="utf-8")))
    assert rows[0]["filtered_object_dir"]
    assert rows[0]["filtered_manifest_path"]
    assert rows[0]["filtered_object_count"] == "2"


def test_threshold_ablation_rejects_filtered_db_for_export(tmp_path, monkeypatch):
    db_dir = _make_ablation_db(tmp_path, filtered=True)
    monkeypatch.setattr(reweight_sweep_module, "_make_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(reweight_sweep_module, "_save_faiss_index", _fake_save_faiss_index)

    with pytest.raises(ValueError, match="already filtered DB"):
        run_spectral_threshold_ablation(
            db_dir=str(db_dir),
            entry_ids=[19, 24, 58, 65],
            thresholds=[None, 0.5],
            output_dir=str(tmp_path / "ablation_runs_filtered"),
        )


def test_threshold_ablation_arg_parser_accepts_dbscan_overrides():
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--db_dir",
            "/tmp/fake_db",
            "--entry_ids",
            "15,19,23,27",
            "--weight_text",
            "0.8",
            "--weight_dinov3",
            "0.6",
            "--enable_dinov3_scoring",
            "--enable_vlm_compress",
            "--enable_vlm_member_spatial",
            "--distance_gate_dsq0",
            "2.5",
            "--dbscan_eps",
            "0.1",
            "--dbscan_min_samples",
            "1",
            "--no-enforce_same_view_uniqueness",
        ]
    )

    assert np.isclose(args.weight_text, 0.8)
    assert np.isclose(args.weight_dinov3, 0.6)
    assert args.enable_dinov3_scoring is True
    assert args.enable_vlm_compress is True
    assert args.enable_vlm_member_spatial is True
    assert np.isclose(args.distance_gate_dsq0, 2.5)
    assert np.isclose(args.dbscan_eps, 0.1)
    assert args.dbscan_min_samples == 1
    assert args.enforce_same_view_uniqueness is False
