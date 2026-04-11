import csv
import json
from pathlib import Path

import pytest
import numpy as np

from spatial_rag.occlusion_scoring import compute_reweighted_detection_score
from spatial_rag.reweight_sweep import run_reweight_sweep
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


def _make_base_db(tmp_path: Path, *, filtered: bool) -> Path:
    db_dir = tmp_path / ("filtered_db" if filtered else "canonical_db")
    db_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("images", "geometry", "overview", "vlm_cache", "vlm_object_cache"):
        (db_dir / dirname).mkdir(parents=True, exist_ok=True)
    (db_dir / "images" / "pose_00000_o000_000000.jpg").write_bytes(b"img0")
    (db_dir / "images" / "pose_00001_o090_000001.jpg").write_bytes(b"img1")
    (db_dir / "images" / "pose_00002_o180_000002.jpg").write_bytes(b"img2")
    (db_dir / "overview" / "center_highest_view.jpg").write_bytes(b"overview")
    (db_dir / "overview" / "textured_floor_plan.jpg").write_bytes(b"overview")

    meta_rows = [
        {
            "id": 0,
            "frame_id": 0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "world_position": [0.0, 0.0, 0.0],
            "orientation": 0,
            "file_name": "images/pose_00000_o000_000000.jpg",
            "text": "chair low",
            "frame_text_short": "chair low",
            "frame_text_long": "chair low long",
            "text_input_for_clip_short": "chair low",
            "text_input_for_clip_long": "chair low long",
            "object_text_inputs_short": ["chair low"],
            "object_text_inputs_long": ["chair low long"],
            "attribute": {"view_type": "living room", "scene_attributes": []},
            "object_count": 1 if not filtered else 0,
        },
        {
            "id": 1,
            "frame_id": 1,
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "world_position": [1.0, 0.0, 0.0],
            "orientation": 90,
            "file_name": "images/pose_00001_o090_000001.jpg",
            "text": "table high",
            "frame_text_short": "table high",
            "frame_text_long": "table high long",
            "text_input_for_clip_short": "table high",
            "text_input_for_clip_long": "table high long",
            "object_text_inputs_short": ["table high"],
            "object_text_inputs_long": ["table high long"],
            "attribute": {"view_type": "living room", "scene_attributes": []},
            "object_count": 1,
        },
        {
            "id": 2,
            "frame_id": 2,
            "x": 2.0,
            "y": 0.0,
            "z": 0.0,
            "world_position": [2.0, 0.0, 0.0],
            "orientation": 180,
            "file_name": "images/pose_00002_o180_000002.jpg",
            "text": "lamp fallback",
            "frame_text_short": "lamp fallback",
            "frame_text_long": "lamp fallback long",
            "text_input_for_clip_short": "lamp fallback",
            "text_input_for_clip_long": "lamp fallback long",
            "object_text_inputs_short": ["lamp fallback"],
            "object_text_inputs_long": ["lamp fallback long"],
            "attribute": {"view_type": "bedroom", "scene_attributes": []},
            "object_count": 1,
        },
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "metadata.jsonl", meta_rows)

    all_object_rows = [
        {
            "object_global_id": 0,
            "entry_id": 0,
            "frame_id": 0,
            "file_name": "images/pose_00000_o000_000000.jpg",
            "object_local_id": "det_000",
            "label": "chair",
            "object_confidence": 0.2,
            "detector_confidence": 0.2,
            "occlusion_level": "fully visible",
            "occlusion_penalty_p_o": 0.0,
            "reweighted_detection_score_r": compute_reweighted_detection_score(0.2, "fully visible"),
            "geometry_source": "mask_depth",
            "description": "chair low",
            "object_text_short": "chair low",
            "object_text_long": "chair low long",
            "estimated_global_x": 0.5,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.5,
            "orientation": 0,
            "angle_bucket": "center",
            "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
            "bbox_xywh_norm": [0.1, 0.1, 0.1, 0.1],
            "crop_path": "canonical_db/geometry/view_00000/objects/obj_000_crop.jpg",
            "mask_path": "canonical_db/geometry/view_00000/objects/obj_000_mask.png",
            "mask_overlay_path": "canonical_db/geometry/view_00000/objects/obj_000_mask_overlay.jpg",
            "depth_map_path": "canonical_db/geometry/view_00000/depth_map.npy",
        },
        {
            "object_global_id": 1,
            "entry_id": 1,
            "frame_id": 1,
            "file_name": "images/pose_00001_o090_000001.jpg",
            "object_local_id": "det_001",
            "label": "table",
            "object_confidence": 0.9,
            "detector_confidence": 0.9,
            "occlusion_level": "fully visible",
            "occlusion_penalty_p_o": 0.0,
            "reweighted_detection_score_r": compute_reweighted_detection_score(0.9, "fully visible"),
            "geometry_source": "mask_depth",
            "description": "table high",
            "object_text_short": "table high",
            "object_text_long": "table high long",
            "estimated_global_x": 1.5,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.5,
            "orientation": 90,
            "angle_bucket": "center",
            "bbox_xyxy": [1.0, 1.0, 2.0, 2.0],
            "bbox_xywh_norm": [0.2, 0.2, 0.2, 0.2],
            "crop_path": "canonical_db/geometry/view_00001/objects/obj_001_crop.jpg",
            "mask_path": "canonical_db/geometry/view_00001/objects/obj_001_mask.png",
            "mask_overlay_path": "canonical_db/geometry/view_00001/objects/obj_001_mask_overlay.jpg",
            "depth_map_path": "canonical_db/geometry/view_00001/depth_map.npy",
        },
        {
            "object_global_id": 2,
            "entry_id": 2,
            "frame_id": 2,
            "file_name": "images/pose_00002_o180_000002.jpg",
            "object_local_id": "feat_002",
            "label": "lamp",
            "object_confidence": 1.0,
            "detector_confidence": None,
            "occlusion_level": "uncertain",
            "occlusion_penalty_p_o": 0.35,
            "reweighted_detection_score_r": compute_reweighted_detection_score(1.0, "uncertain"),
            "geometry_source": "vlm_fallback",
            "description": "lamp fallback",
            "object_text_short": "lamp fallback",
            "object_text_long": "lamp fallback long",
            "estimated_global_x": 2.5,
            "estimated_global_y": 0.0,
            "estimated_global_z": 0.5,
            "orientation": 180,
            "angle_bucket": "center",
            "bbox_xyxy": [],
            "bbox_xywh_norm": [],
            "crop_path": None,
            "mask_path": None,
            "mask_overlay_path": None,
            "depth_map_path": None,
        },
    ]

    if filtered:
        object_rows = all_object_rows[1:]
    else:
        object_rows = all_object_rows
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)

    image_emb = np.vstack([_fake_text_embedding(f"image_{idx}") for idx in range(3)]).astype(np.float32)
    text_emb_short = np.vstack([_fake_text_embedding(row["text_input_for_clip_short"]) for row in meta_rows]).astype(np.float32)
    text_emb_long = np.vstack([_fake_text_embedding(row["text_input_for_clip_long"]) for row in meta_rows]).astype(np.float32)
    object_emb_short_all = np.vstack([_fake_text_embedding(row["object_text_short"]) for row in all_object_rows]).astype(np.float32)
    object_emb_long_all = np.vstack([_fake_text_embedding(row["object_text_long"]) for row in all_object_rows]).astype(np.float32)
    if filtered:
        object_emb_short = object_emb_short_all[1:]
        object_emb_long = object_emb_long_all[1:]
    else:
        object_emb_short = object_emb_short_all
        object_emb_long = object_emb_long_all

    np.save(db_dir / "image_emb.npy", image_emb)
    np.save(db_dir / "text_emb_short.npy", text_emb_short)
    np.save(db_dir / "text_emb_long.npy", text_emb_long)
    np.save(db_dir / "object_text_emb_short.npy", object_emb_short)
    np.save(db_dir / "object_text_emb_long.npy", object_emb_long)

    pre_threshold_rows = []
    for row in all_object_rows:
        route = "vlm_fallback" if row["geometry_source"] == "vlm_fallback" else "geometry"
        pre_threshold_rows.append(
            {
                "entry_id": row["entry_id"],
                "frame_id": row["frame_id"],
                "file_name": row["file_name"],
                "object_local_id": row["object_local_id"],
                "object_route": route,
                "label": row["label"],
                "bbox_xyxy": json.dumps(row["bbox_xyxy"]),
                "bbox_xywh_norm": json.dumps(row["bbox_xywh_norm"]),
                "object_confidence": row["object_confidence"],
                "detector_confidence": row["detector_confidence"],
                "occlusion_level": row["occlusion_level"],
                "occlusion_penalty_p_o": row["occlusion_penalty_p_o"],
                "reweighted_detection_score_r": row["reweighted_detection_score_r"],
                "r_threshold_used": "" if not filtered else 0.5,
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

    raw_api_rows = [
        {
            "entry_id": idx,
            "file_name": meta_rows[idx]["file_name"],
            "geometry_artifacts": {
                "detection_overlay_path": f"canonical_db/geometry/view_{idx:05d}/detection_overlay.jpg",
                "depth_map_path": f"canonical_db/geometry/view_{idx:05d}/depth_map.npy",
            },
            "timing": {
                "geometry_objects_before_r_threshold": 1 if idx < 2 else 0,
                "geometry_objects_after_r_threshold": 1 if idx != 0 else 0,
                "geometry_objects_filtered_by_r_threshold": 0,
                "geometry_all_objects_filtered_by_r_threshold": False,
            },
        }
        for idx in range(3)
    ]
    _write_jsonl(db_dir / "raw_api_responses.jsonl", raw_api_rows)

    timing_rows = [
        {
            "entry_id": idx,
            "frame_idx": idx,
            "file_name": meta_rows[idx]["file_name"],
            "object_count": meta_rows[idx]["object_count"],
            "geometry_objects_before_r_threshold": 1 if idx < 2 else 0,
            "geometry_objects_after_r_threshold": 1 if idx != 0 else 0,
            "geometry_objects_filtered_by_r_threshold": 0,
            "geometry_all_objects_filtered_by_r_threshold": False,
        }
        for idx in range(3)
    ]
    _write_jsonl(db_dir / "per_image_timings.jsonl", timing_rows)

    build_report = {
        "output_dir": str(db_dir),
        "scene_path": "data/fake_scene.glb",
        "builder_variant": "standard",
        "tour_mode": "random",
        "r_threshold": None if not filtered else 0.5,
        "r_threshold_enabled": bool(filtered),
        "object_config": {
            "r_threshold": None if not filtered else 0.5,
            "r_threshold_enabled": bool(filtered),
            "occlusion_reweight": {"w1": 1.0, "w2": 1.0, "b": 0.0},
        },
        "total_entries": 3,
        "total_objects": len(object_rows),
        "object_r_scores_pre_threshold_count": 3,
        "object_r_scores_count": len(object_rows),
        "overview_outputs": {
            "center_highest_view": str(db_dir / "overview" / "center_highest_view.jpg"),
            "textured_floor_plan": str(db_dir / "overview" / "textured_floor_plan.jpg"),
        },
    }
    (db_dir / "build_report.json").write_text(json.dumps(build_report, indent=2), encoding="utf-8")
    return db_dir


def test_run_reweight_sweep_exports_variant_from_canonical_db(tmp_path, monkeypatch):
    db_dir = _make_base_db(tmp_path, filtered=False)
    monkeypatch.setattr("spatial_rag.reweight_sweep._make_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("spatial_rag.reweight_sweep._save_faiss_index", _fake_save_faiss_index)

    report = run_reweight_sweep(
        str(db_dir),
        w1_values=[1.0],
        w2_values=[1.0],
        b_values=[0.0],
        thresholds=[0.5],
        export_db_variants=True,
        output_dir=str(tmp_path / "sweeps"),
    )

    assert report["exportable"] is True
    assert report["analysis_only"] is False
    assert report["exported_config_count"] == 1

    config = report["runs"][0]
    config_dir = Path(config["exported_db_dir"])
    assert config_dir.exists()
    assert (config_dir / "images").is_symlink()
    assert (config_dir / "geometry").is_symlink()
    assert (config_dir / "overview").is_symlink()

    exported_meta = [json.loads(line) for line in (config_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    exported_objects = [json.loads(line) for line in (config_dir / "object_meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(exported_objects) == 2
    assert [row["object_global_id"] for row in exported_objects] == [1, 2]
    assert exported_meta[0]["object_count"] == 0
    assert exported_meta[0]["text"] == "unknown"
    assert exported_meta[1]["text"] == "table high"
    assert exported_meta[2]["text"] == "lamp fallback"

    exported_short = np.load(config_dir / "object_text_emb_short.npy")
    base_short = np.load(db_dir / "object_text_emb_short.npy")
    assert exported_short.shape == (2, 4)
    np.testing.assert_allclose(exported_short, base_short[[1, 2]])

    assert (config_dir / "object_index_short.faiss").exists()
    assert (config_dir / "object_index_long.faiss").exists()

    relation_rows = [json.loads(line) for line in (config_dir / "view_object_relations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row["object_global_id"] for row in relation_rows} == {1, 2}

    build_report = json.loads((config_dir / "build_report.json").read_text(encoding="utf-8"))
    assert build_report["total_objects"] == 2
    assert build_report["geometry_objects_filtered_by_r_threshold"] == 1
    assert build_report["frames_all_geometry_objects_filtered"] == 1

    assert (Path(report["output_dir"]) / "sweep_summary.json").exists()
    assert (Path(report["output_dir"]) / "sweep_results.csv").exists()


def test_run_reweight_sweep_filtered_db_is_analysis_only_and_blocks_export(tmp_path):
    db_dir = _make_base_db(tmp_path, filtered=True)

    report = run_reweight_sweep(
        str(db_dir),
        w1_values=[1.0],
        w2_values=[1.0],
        b_values=[0.0],
        thresholds=[0.5],
        export_db_variants=False,
        output_dir=str(tmp_path / "filtered_sweeps"),
    )

    assert report["exportable"] is False
    assert report["analysis_only"] is True
    assert report["runs"][0]["candidate_total_objects"] == 3
    assert report["runs"][0]["kept_total_objects"] == 2
    assert "warning" in report["runs"][0]

    with pytest.raises(ValueError, match="already filtered DB"):
        run_reweight_sweep(
            str(db_dir),
            w1_values=[1.0],
            w2_values=[1.0],
            b_values=[0.0],
            thresholds=[0.5],
            export_db_variants=True,
            output_dir=str(tmp_path / "filtered_export"),
        )
