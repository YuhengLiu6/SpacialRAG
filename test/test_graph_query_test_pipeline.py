import json

from spatial_rag.graph_query_test_pipeline import export_graph_query_artifacts, run_graph_query_test_pipeline


class _DummyDriver:
    pass


def test_run_graph_query_test_pipeline_reports_success(monkeypatch):
    dummy = _DummyDriver()

    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline._graph_counts",
        lambda driver, database=None: {"places": 3, "views": 12, "objects": 9, "object_classes": 2},
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline._choose_anchor_place",
        lambda driver, preferred_place_id=None, database=None: "place_00000",
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline.query_same_node",
        lambda driver, place_id, database=None: {
            "place": {"place_id": place_id},
            "views": [{"orientation_deg": 0, "file_name": "images/view_00000.jpg"}, {"orientation_deg": 90, "file_name": "images/view_00001.jpg"}],
            "objects": [{"label": "chair"}, {"label": "table"}],
        },
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline.query_place_objects",
        lambda driver, place_id, object_label=None, database=None: [{"label": "chair", "file_name": "images/view_00000.jpg"}],
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline._choose_object_label",
        lambda driver, preferred_object_label=None, database=None: "chair",
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline.query_places_for_object",
        lambda driver, object_label, database=None: [
            {"place_id": "place_00000", "file_name": "images/view_00000.jpg"},
            {"place_id": "place_00001", "file_name": "images/view_00004.jpg"},
        ],
    )

    def _neighbors(driver, place_id, direction, database=None):
        if direction in {"north", "east"}:
            return [{"place_id": f"{direction}_place"}]
        return []

    monkeypatch.setattr("spatial_rag.graph_query_test_pipeline.query_direction_neighbors", _neighbors)
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline._find_directional_object_case",
        lambda driver, place_id, database=None: {
            "direction": "north",
            "neighbor_place_id": "north_place",
            "object_label": "chair",
        },
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline.query_direction_objects",
        lambda driver, place_id, direction, object_label=None, database=None: [{"place_id": "north_place", "label": "chair", "file_name": "images/view_00004.jpg"}],
    )

    report = run_graph_query_test_pipeline(dummy)

    assert report["summary"]["ok"] is True
    assert report["summary"]["failed"] == 0
    assert report["anchor_place_id"] == "place_00000"
    assert report["object_label"] == "chair"
    names = {check["name"]: check for check in report["checks"]}
    assert names["global_object_lookup"]["status"] == "passed"
    assert names["north_neighbors"]["status"] == "passed"
    assert names["directional_object_lookup"]["status"] == "passed"
    assert names["same_node_views"]["details"]["views"][0]["file_name"] == "images/view_00000.jpg"


def test_run_graph_query_test_pipeline_skips_when_directional_case_missing(monkeypatch):
    dummy = _DummyDriver()

    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline._graph_counts",
        lambda driver, database=None: {"places": 1, "views": 4, "objects": 0, "object_classes": 0},
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline._choose_anchor_place",
        lambda driver, preferred_place_id=None, database=None: "place_00000",
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline.query_same_node",
        lambda driver, place_id, database=None: {
            "place": {"place_id": place_id},
            "views": [{"orientation_deg": 0}],
            "objects": [],
        },
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline.query_place_objects",
        lambda driver, place_id, object_label=None, database=None: [],
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline._choose_object_label",
        lambda driver, preferred_object_label=None, database=None: None,
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline.query_direction_neighbors",
        lambda driver, place_id, direction, database=None: [],
    )
    monkeypatch.setattr(
        "spatial_rag.graph_query_test_pipeline._find_directional_object_case",
        lambda driver, place_id, database=None: None,
    )

    report = run_graph_query_test_pipeline(dummy)

    names = {check["name"]: check for check in report["checks"]}
    assert names["global_object_lookup"]["status"] == "failed"
    assert names["north_neighbors"]["status"] == "skipped"
    assert names["east_neighbors"]["status"] == "skipped"
    assert names["directional_object_lookup"]["status"] == "skipped"


def test_export_graph_query_artifacts_writes_report_and_copies_images(tmp_path):
    db_dir = tmp_path / "db"
    image_dir = db_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "view_00000.jpg").write_bytes(b"fakejpg0")
    (image_dir / "view_00004.jpg").write_bytes(b"fakejpg4")

    report = {
        "checks": [
            {"name": "same_node_views", "details": {"views": [{"file_name": "images/view_00000.jpg"}]}},
            {"name": "global_object_lookup", "details": {"sample_hits": [{"file_name": "images/view_00004.jpg"}]}},
        ]
    }

    output_dir = tmp_path / "artifacts"
    exported = export_graph_query_artifacts(report=report, db_dir=str(db_dir), output_dir=str(output_dir))

    assert (output_dir / "report.json").exists()
    assert (output_dir / "views" / "images" / "view_00000.jpg").exists()
    assert (output_dir / "views" / "images" / "view_00004.jpg").exists()
    saved_report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert saved_report["artifacts"]["copied_images"]
    assert exported["artifacts"]["missing_images"] == []
