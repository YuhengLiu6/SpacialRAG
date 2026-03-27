import json

import cv2
import numpy as np
import pytest

from spatial_rag.object_birdview_visualizer import (
    render_all_object_nodes_birdview,
    render_object_nodes_birdview,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def test_render_object_nodes_birdview_writes_png_and_sidecar(tmp_path):
    db_dir = tmp_path / "db"
    overview_dir = db_dir / "overview"
    overview_dir.mkdir(parents=True)
    floorplan = np.full((200, 300, 3), 240, dtype=np.uint8)
    assert cv2.imwrite(str(overview_dir / "textured_floor_plan.jpg"), floorplan)

    meta_rows = [
        {"id": 0, "x": 0.0, "y": 0.0, "world_position": [0.0, 1.6, 0.0], "orientation": 0, "file_name": "images/view_00000.jpg"},
        {"id": 1, "x": 2.0, "y": 1.0, "world_position": [2.0, 1.6, 1.0], "orientation": 90, "file_name": "images/view_00001.jpg"},
    ]
    object_rows = [
        {"entry_id": 0, "object_global_id": 3, "label": "chair", "estimated_global_x": 0.5, "estimated_global_y": 0.8, "estimated_global_z": -1.0},
        {"entry_id": 0, "object_global_id": 4, "label": "table", "estimated_global_x": 1.0, "estimated_global_y": 1.2, "estimated_global_z": -0.5},
    ]
    relation_rows = [
        {"entry_id": 0, "source_object_global_id": 3, "target_object_global_id": 4},
        {"entry_id": 0, "source_object_global_id": 4, "target_object_global_id": 3},
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)
    _write_jsonl(db_dir / "object_object_relations.jsonl", relation_rows)

    out_path = db_dir / "rendered.png"
    result = render_object_nodes_birdview(
        db_dir=str(db_dir),
        entry_id=0,
        output_path=str(out_path),
        draw_object_object_edges=True,
    )

    assert result == str(out_path)
    assert out_path.exists()
    sidecar = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["entry_id"] == 0
    assert len(sidecar["objects"]) == 2
    assert sidecar["camera"]["y"] == 1.6
    assert sidecar["objects"][0]["y"] is not None
    assert sidecar["camera"]["pixel"]


def test_render_object_nodes_birdview_supports_sampling(tmp_path):
    db_dir = tmp_path / "db"
    overview_dir = db_dir / "overview"
    images_dir = db_dir / "images"
    overview_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    floorplan = np.full((200, 300, 3), 240, dtype=np.uint8)
    view_img = np.full((50, 80, 3), 200, dtype=np.uint8)
    assert cv2.imwrite(str(overview_dir / "textured_floor_plan.jpg"), floorplan)
    assert cv2.imwrite(str(images_dir / "view_00000.jpg"), view_img)

    meta_rows = [
        {"id": 0, "x": 0.0, "y": 0.0, "world_position": [0.0, 1.6, 0.0], "orientation": 0, "file_name": "images/view_00000.jpg"},
    ]
    object_rows = [
        {"entry_id": 0, "object_global_id": 3, "label": "chair", "estimated_global_x": 0.5, "estimated_global_y": 0.8, "estimated_global_z": -1.0},
        {"entry_id": 0, "object_global_id": 4, "label": "table", "estimated_global_x": 1.0, "estimated_global_y": 1.0, "estimated_global_z": -0.5},
        {"entry_id": 0, "object_global_id": 5, "label": "lamp", "estimated_global_x": 1.5, "estimated_global_y": 1.8, "estimated_global_z": -0.25},
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)

    out_path = db_dir / "rendered_sampled.png"
    result = render_object_nodes_birdview(
        db_dir=str(db_dir),
        entry_id=0,
        output_path=str(out_path),
        sample_objects=2,
        sample_seed=11,
    )

    assert result == str(out_path)
    sidecar = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["entry_id"] == 0
    assert sidecar["sampled"] is True
    assert sidecar["num_total_objects"] == 3
    assert sidecar["num_objects"] == 2
    assert sidecar["sample_objects"] == 2
    assert sidecar["sample_seed"] == 11
    assert len(sidecar["objects"]) == 2
    assert sidecar["camera"]["y"] == 1.6
    assert sidecar["view_image_output_path"]


def test_render_all_object_nodes_birdview_writes_png_and_sidecar(tmp_path):
    db_dir = tmp_path / "db"
    overview_dir = db_dir / "overview"
    overview_dir.mkdir(parents=True)
    floorplan = np.full((200, 300, 3), 240, dtype=np.uint8)
    assert cv2.imwrite(str(overview_dir / "textured_floor_plan.jpg"), floorplan)

    meta_rows = [
        {"id": 0, "x": 0.0, "y": 0.0, "world_position": [0.0, 1.6, 0.0], "orientation": 0, "file_name": "images/view_00000.jpg"},
        {"id": 1, "x": 2.0, "y": 1.0, "world_position": [2.0, 1.6, 1.0], "orientation": 90, "file_name": "images/view_00001.jpg"},
    ]
    object_rows = [
        {"entry_id": 0, "object_global_id": 3, "label": "chair", "estimated_global_x": 0.5, "estimated_global_y": 0.8, "estimated_global_z": -1.0},
        {"entry_id": 1, "object_global_id": 4, "label": "table", "estimated_global_x": 2.5, "estimated_global_y": 1.2, "estimated_global_z": 1.0},
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)

    out_path = db_dir / "all_objects.png"
    result = render_all_object_nodes_birdview(
        db_dir=str(db_dir),
        output_path=str(out_path),
        draw_labels=True,
        draw_cameras=True,
    )

    assert result == str(out_path)
    assert out_path.exists()
    sidecar = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["mode"] == "all_objects"
    assert sidecar["num_objects"] == 2
    assert sidecar["objects"][0]["y"] is not None


def test_render_all_object_nodes_birdview_prefers_saved_projection(tmp_path):
    db_dir = tmp_path / "db"
    overview_dir = db_dir / "overview"
    overview_dir.mkdir(parents=True)
    floorplan = np.full((200, 300, 3), 240, dtype=np.uint8)
    assert cv2.imwrite(str(overview_dir / "textured_floor_plan.jpg"), floorplan)
    (overview_dir / "floor_plan_projection.json").write_text(
        json.dumps(
            {
                "view_min_x": -10.0,
                "view_max_x": 10.0,
                "view_min_z": -20.0,
                "view_max_z": 20.0,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    meta_rows = [{"id": 0, "x": 100.0, "y": 200.0, "world_position": [100.0, 1.6, 200.0], "orientation": 0, "file_name": "images/view_00000.jpg"}]
    object_rows = [
        {"entry_id": 0, "object_global_id": 3, "label": "chair", "estimated_global_x": 101.0, "estimated_global_y": 0.8, "estimated_global_z": 199.0}
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)

    out_path = db_dir / "all_objects_projection.png"
    render_all_object_nodes_birdview(
        db_dir=str(db_dir),
        output_path=str(out_path),
    )

    sidecar = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["projection"] == {
        "view_min_x": -10.0,
        "view_max_x": 10.0,
        "view_min_z": -20.0,
        "view_max_z": 20.0,
    }


def test_render_all_object_nodes_birdview_supports_sampling_and_auto_labels(tmp_path):
    db_dir = tmp_path / "db"
    overview_dir = db_dir / "overview"
    overview_dir.mkdir(parents=True)
    floorplan = np.full((200, 300, 3), 240, dtype=np.uint8)
    assert cv2.imwrite(str(overview_dir / "textured_floor_plan.jpg"), floorplan)

    meta_rows = [
        {"id": 0, "x": 0.0, "y": 0.0, "world_position": [0.0, 1.6, 0.0], "orientation": 0, "file_name": "images/view_00000.jpg"},
    ]
    object_rows = [
        {"entry_id": 0, "object_global_id": 3, "label": "chair", "estimated_global_x": 0.5, "estimated_global_y": 0.8, "estimated_global_z": -1.0},
        {"entry_id": 0, "object_global_id": 4, "label": "table", "estimated_global_x": 1.5, "estimated_global_y": 1.0, "estimated_global_z": -0.5},
        {"entry_id": 0, "object_global_id": 5, "label": "lamp", "estimated_global_x": 2.5, "estimated_global_y": 1.8, "estimated_global_z": 0.5},
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)

    out_path = db_dir / "sampled_objects.png"
    render_all_object_nodes_birdview(
        db_dir=str(db_dir),
        output_path=str(out_path),
        sample_objects=2,
        sample_seed=7,
    )

    sidecar = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["mode"] == "all_objects"
    assert sidecar["sampled"] is True
    assert sidecar["num_total_objects"] == 3
    assert sidecar["num_objects"] == 2
    assert sidecar["sample_objects"] == 2
    assert sidecar["sample_seed"] == 7
    assert sidecar["draw_labels"] is True
    assert len(sidecar["objects"]) == 2


def test_render_all_object_nodes_birdview_raises_for_missing_db(tmp_path):
    missing_db = tmp_path / "missing_db"
    out_path = tmp_path / "out.png"
    with pytest.raises(FileNotFoundError):
        render_all_object_nodes_birdview(
            db_dir=str(missing_db),
            output_path=str(out_path),
        )


def test_render_all_object_nodes_birdview_raises_for_empty_object_rows(tmp_path):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    _write_jsonl(db_dir / "meta.jsonl", [{"id": 0, "x": 0.0, "y": 0.0, "orientation": 0}])
    _write_jsonl(db_dir / "object_meta.jsonl", [])
    out_path = db_dir / "out.png"
    with pytest.raises(ValueError, match="No plottable objects"):
        render_all_object_nodes_birdview(
            db_dir=str(db_dir),
            output_path=str(out_path),
        )
