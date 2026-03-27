import json

import numpy as np
import pytest

from spatial_rag.graph_builder import (
    PlaceRecord,
    build_direction_edges,
    build_graph_payload,
    build_place_records,
    query_direction_neighbors,
    query_direction_objects,
    query_places_for_object,
    query_place_objects,
    query_same_node,
    load_graph_to_neo4j,
)


def _meta_row(entry_id, x, z, orientation, room_function="resting", view_type="living room"):
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


def _object_row(object_global_id, entry_id, label, orientation, distance):
    return {
        "object_global_id": object_global_id,
        "entry_id": entry_id,
        "label": label,
        "description": f"{label} description",
        "long_form_open_description": f"{label} long description",
        "laterality": "center",
        "distance_bin": "middle",
        "verticality": "middle",
        "distance_from_camera_m": distance,
        "object_orientation_deg": orientation,
        "estimated_global_x": 0.0 if object_global_id == 0 else 2.0,
        "estimated_global_y": 0.6 if object_global_id == 0 else 1.1,
        "estimated_global_z": -1.5 if object_global_id == 0 else -0.5,
        "location_relative_to_other_objects": "",
        "parse_status": "ok",
    }


def _write_jsonl(path, rows):
    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


class _FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return dict(self._data)


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def run(self, query, **params):
        self.calls.append((query, params))
        rows = self._responses.pop(0) if self._responses else []
        return [_FakeRecord(row) for row in rows]


class _FakeDriver:
    def __init__(self, responses):
        self._session = _FakeSession(responses)

    def session(self, database=None):
        return self._session


def test_build_place_records_groups_four_angles_into_one_place():
    meta_rows = [
        _meta_row(0, 0.0, 0.0, 0),
        _meta_row(1, 0.0, 0.0, 90),
        _meta_row(2, 0.0, 0.0, 180),
        _meta_row(3, 0.0, 0.0, 270),
        _meta_row(4, 2.0, 0.0, 0, room_function="dining", view_type="dining room"),
        _meta_row(5, 2.0, 0.0, 180, room_function="dining", view_type="dining room"),
    ]

    places = build_place_records(meta_rows, scan_angles=[0, 90, 180, 270])

    assert len(places) == 2
    assert places[0].source_entry_ids == [0, 1, 2, 3]
    assert places[0].scan_angles == [0, 90, 180, 270]
    assert places[1].source_entry_ids == [4, 5]
    assert places[1].room_function == "dining"
    assert places[1].view_type == "dining room"


def test_build_direction_edges_creates_cardinal_and_reverse_edges():
    places = [
        PlaceRecord(
            place_id="place_center",
            x=0.0,
            y=0.0,
            z=0.0,
            point={"x": 0.0, "y": 0.0, "crs": "cartesian"},
            room_function="resting",
            view_type="living room",
            source_entry_ids=[0, 1, 2, 3],
            scan_angles=[0, 90, 180, 270],
        ),
        PlaceRecord(
            place_id="place_north",
            x=0.0,
            y=0.0,
            z=2.0,
            point={"x": 0.0, "y": 2.0, "crs": "cartesian"},
            room_function="resting",
            view_type="living room",
            source_entry_ids=[4, 5, 6, 7],
            scan_angles=[0, 90, 180, 270],
        ),
        PlaceRecord(
            place_id="place_east",
            x=3.0,
            y=0.0,
            z=0.0,
            point={"x": 3.0, "y": 0.0, "crs": "cartesian"},
            room_function="resting",
            view_type="living room",
            source_entry_ids=[8, 9, 10, 11],
            scan_angles=[0, 90, 180, 270],
        ),
    ]

    edges = build_direction_edges(places, k_neighbors=4, same_axis_eps=0.25)
    edge_keys = {(edge.source_place_id, edge.relation_type, edge.target_place_id) for edge in edges}

    assert ("place_center", "NORTH_OF", "place_north") in edge_keys
    assert ("place_north", "SOUTH_OF", "place_center") in edge_keys
    assert ("place_center", "EAST_OF", "place_east") in edge_keys
    assert ("place_east", "WEST_OF", "place_center") in edge_keys


def test_build_graph_payload_creates_places_views_objects_and_direction_edges(tmp_path):
    meta_rows = [
        _meta_row(0, 0.0, 0.0, 0),
        _meta_row(1, 0.0, 0.0, 90),
        _meta_row(2, 0.0, 0.0, 180),
        _meta_row(3, 0.0, 0.0, 270),
        _meta_row(4, 2.0, 0.0, 0, room_function="dining", view_type="dining room"),
        _meta_row(5, 2.0, 0.0, 90, room_function="dining", view_type="dining room"),
        _meta_row(6, 2.0, 0.0, 180, room_function="dining", view_type="dining room"),
        _meta_row(7, 2.0, 0.0, 270, room_function="dining", view_type="dining room"),
    ]
    object_rows = [
        {
            **_object_row(0, 0, "chair", 0.0, 1.5),
            "estimated_global_x": 0.0,
            "estimated_global_z": -1.5,
        },
        {
            **_object_row(1, 0, "table", 0.0, 2.0),
            "estimated_global_x": 1.0,
            "estimated_global_z": -1.5,
        },
        {
            **_object_row(2, 4, "lamp", 90.0, 2.0),
            "estimated_global_x": 0.0,
            "estimated_global_z": 0.0,
        },
    ]
    build_report = {
        "scan_angles": [0, 90, 180, 270],
        "random_config": {"scan_angles": [0, 90, 180, 270]},
    }

    _write_jsonl(tmp_path / "meta.jsonl", meta_rows)
    _write_jsonl(tmp_path / "object_meta.jsonl", object_rows)
    (tmp_path / "build_report.json").write_text(json.dumps(build_report), encoding="utf-8")
    np.save(tmp_path / "image_emb.npy", np.ones((8, 4), dtype=np.float32))
    np.save(tmp_path / "text_emb_long.npy", np.full((8, 4), 2.0, dtype=np.float32))
    np.save(tmp_path / "object_text_emb_long.npy", np.full((3, 4), 3.0, dtype=np.float32))

    payload = build_graph_payload(str(tmp_path), k_neighbors=4, same_axis_eps=0.25)

    assert payload["summary"]["num_places"] == 2
    assert payload["summary"]["num_views"] == 8
    assert payload["summary"]["num_objects"] == 3
    assert any(edge["relation_type"] == "EAST_OF" for edge in payload["direction_edges"])
    assert any(edge["relation_type"] == "WEST_OF" for edge in payload["direction_edges"])
    assert payload["place_object_edges"][0]["projected_x"] == 0.0
    assert payload["place_object_edges"][0]["projected_z"] == -1.5
    assert payload["view_nodes"][0]["node_type"] == "View"
    assert payload["view_nodes"][0]["desc_emb"] == [2.0, 2.0, 2.0, 2.0]
    assert payload["object_nodes"][0]["node_type"] == "Object"
    assert payload["object_nodes"][0]["image_emb"] is None
    assert payload["object_nodes"][0]["y"] == 0.6
    assert "vertical_direction" in payload["view_object_edges"][0]
    assert "distance_3d_m" in payload["object_object_edges"][0]
    assert payload["summary"]["num_view_view_edges"] > 0
    assert payload["summary"]["num_view_object_edges"] == 3
    assert payload["summary"]["num_object_object_edges"] == 2


def test_load_graph_to_neo4j_writes_new_relation_types(monkeypatch):
    captured = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def run(self, query, **params):
            captured.append((query, params))
            return []

    class _Driver:
        def session(self, database=None):
            return _Session()

        def close(self):
            return None

    monkeypatch.setattr("spatial_rag.graph_builder._neo4j_driver", lambda uri, auth: _Driver())

    payload = {
        "places": [{"place_id": "place_00000", "x": 0.0, "y": 0.0, "z": 0.0, "point": {}, "room_function": "resting", "view_type": "living room", "source_entry_ids": [0], "scan_angles": [0]}],
        "views": [{"view_id": "view_00000", "place_id": "place_00000", "entry_id": 0, "orientation_deg": 0, "file_name": "images/view_00000.jpg", "parse_status": "ok", "frame_text_short": "short", "frame_text_long": "long"}],
        "objects": [{"obs_id": "obs_000000", "place_id": "place_00000", "view_id": "view_00000", "entry_id": 0, "object_global_id": 0, "object_class": "chair", "label": "chair", "description": "chair", "long_form_open_description": "chair", "laterality": "center", "distance_bin": "middle", "verticality": "middle", "distance_from_camera_m": 1.0, "object_orientation_deg": 0.0, "projected_x": 0.0, "projected_y": 0.8, "projected_z": -1.0, "location_relative_to_other_objects": "", "parse_status": "ok"}],
        "object_classes": [{"name": "chair"}],
        "place_object_edges": [{"place_id": "place_00000", "obs_id": "obs_000000", "view_id": "view_00000", "object_orientation_deg": 0.0, "distance_from_camera_m": 1.0, "projected_x": 0.0, "projected_z": -1.0}],
        "direction_edges": [],
        "view_nodes": [{"view_id": "view_00000", "node_type": "View", "x": 0.0, "y": 0.0, "z": 0.0, "desc_emb": [1.0], "image_emb": [2.0]}],
        "object_nodes": [{"obs_id": "obs_000000", "node_type": "Object", "x": 0.0, "y": 0.8, "z": -1.0, "desc_emb": [3.0], "image_emb": None}],
        "view_view_edges": [{"source_view_id": "view_00000", "target_view_id": "view_00000", "direction": "neighbor", "relation_type": "ViewView", "dx": 0.0, "dy": 0.0, "dz": 0.0, "distance_m": 0.0}],
        "view_object_edges": [{"view_id": "view_00000", "obs_id": "obs_000000", "direction": "in", "direction_frame": "view_aligned", "vertical_direction": "below", "relation_type": "ViewObject", "dx": 0.0, "dy": -0.8, "dz": -1.0, "distance_m": 1.0, "distance_3d_m": 1.2806248474865698}],
        "object_object_edges": [{"source_obs_id": "obs_000000", "target_obs_id": "obs_000000", "direction": "right", "direction_frame": "view_aligned", "vertical_direction": "above", "relation_type": "ObjectObject", "dx": 1.0, "dy": 0.5, "dz": 0.0, "distance_m": 1.0, "distance_3d_m": 1.118033988749895, "relation_source": "geometry_postprocess"}],
    }

    counts = load_graph_to_neo4j(payload=payload, uri="bolt://example", auth=("neo4j", "pw"))

    assert counts["view_view_edges"] == 1
    assert counts["view_object_edges"] == 1
    assert counts["object_object_edges"] == 1
    queries = "\n".join(query for query, _params in captured)
    assert "NEIGHBOR_VIEW" in queries
    assert "ABOVE" in queries
    assert "IN_VIEW" in queries
    assert "RIGHT_OF" in queries


def test_query_same_node_returns_views_and_objects():
    driver = _FakeDriver(
        responses=[
            [{"place_id": "place_00000", "scan_angles": [0, 90, 180, 270]}],
            [{"view_id": "view_00000", "entry_id": 0, "orientation_deg": 0, "file_name": "images/view_00000.jpg"}],
            [{"obs_id": "obs_000000", "label": "chair", "description": "chair description", "object_orientation_deg": 0.0}],
        ]
    )

    result = query_same_node(driver, "place_00000")

    assert result["place"]["place_id"] == "place_00000"
    assert result["views"][0]["view_id"] == "view_00000"
    assert result["objects"][0]["label"] == "chair"
    assert "HAS_VIEW" in driver._session.calls[1][0]


def test_query_direction_neighbors_uses_explicit_direction_relationship():
    driver = _FakeDriver(
        responses=[
            [{"place_id": "place_00001", "x": 0.0, "y": 0.0, "z": 2.0, "room_function": "resting", "view_type": "living room", "dx": 0.0, "dz": 2.0, "distance_m": 2.0}]
        ]
    )

    rows = query_direction_neighbors(driver, "place_00000", "north")

    assert rows[0]["place_id"] == "place_00001"
    assert "[r:NORTH_OF]" in driver._session.calls[0][0]


def test_query_place_objects_and_direction_objects_support_object_label_filter():
    driver = _FakeDriver(
        responses=[
            [{"obs_id": "obs_000000", "label": "chair", "description": "chair description", "view_id": "view_00000", "file_name": "images/view_00000.jpg", "object_orientation_deg": 0.0, "distance_from_camera_m": 1.5, "projected_x": 0.0, "projected_z": -1.5}],
            [{"place_id": "place_00001", "obs_id": "obs_000001", "label": "chair", "description": "chair north", "dx": 0.0, "dz": 2.0, "distance_m": 2.0, "view_id": "view_00004", "file_name": "images/view_00004.jpg", "object_orientation_deg": 90.0, "distance_from_camera_m": 1.0}],
        ]
    )

    place_rows = query_place_objects(driver, "place_00000", object_label="chair")
    direction_rows = query_direction_objects(driver, "place_00000", "north", object_label="chair")

    assert place_rows[0]["label"] == "chair"
    assert place_rows[0]["file_name"] == "images/view_00000.jpg"
    assert direction_rows[0]["place_id"] == "place_00001"
    assert direction_rows[0]["file_name"] == "images/view_00004.jpg"
    assert "WHERE o.label = $object_label" in driver._session.calls[0][0]
    assert "[d:NORTH_OF]" in driver._session.calls[1][0]


def test_query_places_for_object_returns_place_view_and_object_fields():
    driver = _FakeDriver(
        responses=[
            [{
                "place_id": "place_00000",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "room_function": "resting",
                "view_type": "living room",
                "view_id": "view_00000",
                "orientation_deg": 0,
                "file_name": "images/view_00000.jpg",
                "obs_id": "obs_000000",
                "label": "chair",
                "description": "wooden chair",
            }]
        ]
    )

    rows = query_places_for_object(driver, "chair")

    assert rows[0]["place_id"] == "place_00000"
    assert rows[0]["view_id"] == "view_00000"
    assert rows[0]["label"] == "chair"
    assert "toLower(o.label) = toLower($object_label)" in driver._session.calls[0][0]


def test_query_direction_neighbors_rejects_invalid_direction():
    driver = _FakeDriver(responses=[[]])

    with pytest.raises(ValueError):
        query_direction_neighbors(driver, "place_00000", "northeast")
