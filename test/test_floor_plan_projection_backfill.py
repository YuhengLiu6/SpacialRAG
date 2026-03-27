import json

import cv2
import numpy as np

from spatial_rag.floor_plan_projection_backfill import backfill_floor_plan_projection


class _FakeExplorer:
    def __init__(self, scene_path):
        self.scene_path = scene_path
        self._last_top_down_projection = None

    def render_true_floor_plan(self):
        self._last_top_down_projection = {
            "view_min_x": -10.0,
            "view_max_x": 5.0,
            "view_min_z": -3.0,
            "view_max_z": 7.0,
        }
        return None

    def close(self):
        return None


def test_backfill_floor_plan_projection_writes_projection_and_updates_report(tmp_path):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    (db_dir / "build_report.json").write_text(
        json.dumps({"scene_path": "/tmp/fake_scene.glb", "overview_outputs": {}}, ensure_ascii=True),
        encoding="utf-8",
    )

    result = backfill_floor_plan_projection(
        db_dir=str(db_dir),
        explorer_factory=lambda scene_path: _FakeExplorer(scene_path),
    )

    projection_path = db_dir / "overview" / "floor_plan_projection.json"
    assert result["status"] == "written"
    assert projection_path.exists()
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    assert projection["view_min_x"] == -10.0
    report = json.loads((db_dir / "build_report.json").read_text(encoding="utf-8"))
    assert report["overview_outputs"]["floor_plan_projection"] == str(projection_path)


def test_backfill_floor_plan_projection_skips_existing_when_not_overwriting(tmp_path):
    db_dir = tmp_path / "db"
    overview_dir = db_dir / "overview"
    overview_dir.mkdir(parents=True)
    projection_path = overview_dir / "floor_plan_projection.json"
    projection_path.write_text(
        json.dumps({"view_min_x": 0.0, "view_max_x": 1.0, "view_min_z": 0.0, "view_max_z": 1.0}, ensure_ascii=True),
        encoding="utf-8",
    )

    result = backfill_floor_plan_projection(
        db_dir=str(db_dir),
        scene_path="/tmp/fake_scene.glb",
        explorer_factory=lambda scene_path: _FakeExplorer(scene_path),
    )

    assert result["status"] == "exists"


def test_backfill_floor_plan_projection_falls_back_to_floorplan_bbox_calibration(tmp_path):
    db_dir = tmp_path / "db"
    overview_dir = db_dir / "overview"
    overview_dir.mkdir(parents=True)
    floorplan = np.full((100, 200, 3), 70, dtype=np.uint8)
    floorplan[20:80, 40:160] = 240
    assert cv2.imwrite(str(overview_dir / "textured_floor_plan.jpg"), floorplan)
    (db_dir / "build_report.json").write_text(
        json.dumps({"scene_path": "/tmp/missing_scene.glb", "overview_outputs": {}}, ensure_ascii=True),
        encoding="utf-8",
    )
    (db_dir / "meta.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": 0, "x": 0.0, "y": 0.0}, ensure_ascii=True),
                json.dumps({"id": 1, "x": 10.0, "y": 5.0}, ensure_ascii=True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _broken_factory(scene_path):
        raise RuntimeError("scene unavailable")

    result = backfill_floor_plan_projection(
        db_dir=str(db_dir),
        explorer_factory=_broken_factory,
        overwrite=True,
    )

    projection = json.loads((overview_dir / "floor_plan_projection.json").read_text(encoding="utf-8"))
    assert result["projection_source"] == "floorplan_bbox_calibration"
    assert projection["view_min_x"] < 0.0
    assert projection["view_max_x"] > 10.0
