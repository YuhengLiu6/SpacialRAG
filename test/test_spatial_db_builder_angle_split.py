import json
from types import SimpleNamespace

import numpy as np

from spatial_rag.spatial_db_builder import (
    _build_view_attribute,
    _build_object_object_relations,
    _build_location_summary_from_surroundings,
    _build_view_object_relations,
    _classify_view_aligned_direction,
    _classify_vertical_direction,
    _compute_object_orientation,
    _fallback_relative_bearing_from_laterality,
    _format_object_text_long,
    _load_resume_state,
    _make_object_record,
    _project_global_xz,
    _response_has_length_finish_reason,
    _run_optional_polar_surrounding_postprocess,
    _should_reuse_existing_entry,
    _write_floor_plan_projection,
    build_spatial_database_angle_split,
)


def test_compute_object_orientation_offsets_left_center_right():
    assert _compute_object_orientation(90, "left", angle_split_enable=True, angle_step=30) == 60
    assert _compute_object_orientation(90, "center", angle_split_enable=True, angle_step=30) == 90
    assert _compute_object_orientation(90, "right", angle_split_enable=True, angle_step=30) == 120


def test_compute_object_orientation_falls_back_to_center_for_invalid_laterality():
    assert _compute_object_orientation(180, "unknown", angle_split_enable=True, angle_step=30) == 180


def test_project_global_xz_uses_camera_orientation_plus_relative_bearing():
    projected_x, projected_z = _project_global_xz(
        origin_x=0.0,
        origin_z=0.0,
        camera_orientation_deg=270.0,
        relative_bearing_deg=-30.0,
        distance_m=2.0,
    )

    assert round(projected_x, 3) == 1.732
    assert round(projected_z, 3) == -1.0


def test_fallback_relative_bearing_from_laterality_uses_fixed_bucket_offsets():
    assert _fallback_relative_bearing_from_laterality("left", angle_step=30) == -30.0
    assert _fallback_relative_bearing_from_laterality("center", angle_step=30) == 0.0
    assert _fallback_relative_bearing_from_laterality("right", angle_step=30) == 30.0


def test_build_location_summary_from_surroundings_formats_anchor_and_distance():
    summary = _build_location_summary_from_surroundings(
        [
            {
                "label": "table",
                "relation_to_primary": "right of chair",
                "distance_from_primary_m": 0.84,
                "estimated_global_x": 1.24,
                "estimated_global_z": -2.26,
            }
        ]
    )

    assert summary == "table relation=right of chair d=0.8m anchor=(1.0,-2.5)"


def test_classify_view_aligned_direction_uses_view_orientation():
    assert _classify_view_aligned_direction(dx=1.0, dz=0.0, view_orientation_deg=0.0) == "right"
    assert _classify_view_aligned_direction(dx=-1.0, dz=0.0, view_orientation_deg=0.0) == "left"
    assert _classify_view_aligned_direction(dx=0.0, dz=-1.0, view_orientation_deg=0.0) == "in front"
    assert _classify_view_aligned_direction(dx=0.0, dz=1.0, view_orientation_deg=0.0) == "behind"


def test_classify_vertical_direction_uses_height_delta():
    assert _classify_vertical_direction(0.3) == "above"
    assert _classify_vertical_direction(-0.3) == "below"
    assert _classify_vertical_direction(0.1) == "level"


def test_build_view_object_relations_uses_estimated_global_coordinates():
    rows = _build_view_object_relations(
        metadata_records=[{"id": 0, "x": 1.0, "y": 2.0, "world_position": [1.0, 1.6, 2.0], "orientation": 0}],
        object_metadata_records=[
            {
                "entry_id": 0,
                "object_global_id": 7,
                "label": "chair",
                "estimated_global_x": 2.0,
                "estimated_global_y": 0.8,
                "estimated_global_z": 4.0,
            }
        ],
    )

    assert rows == [
        {
            "entry_id": 0,
            "view_id": "view_00000",
            "object_global_id": 7,
            "obs_id": "obs_000007",
            "label": "chair",
            "view_x": 1.0,
            "view_y": 1.6,
            "view_z": 2.0,
            "object_x": 2.0,
            "object_y": 0.8,
            "object_z": 4.0,
            "dx": 1.0,
            "dy": -0.8,
            "dz": 2.0,
            "distance_m": 2.23606797749979,
            "distance_3d_m": 2.3748684174075834,
            "direction": "in",
            "direction_frame": "view_aligned",
            "vertical_direction": "below",
            "relation_type": "ViewObject",
        }
    ]


def test_build_object_object_relations_generates_ordered_pairs_with_direction():
    rows = _build_object_object_relations(
        metadata_records=[{"id": 0, "orientation": 0}],
        object_metadata_records=[
            {
                "entry_id": 0,
                "object_global_id": 1,
                "label": "chair",
                "estimated_global_x": 0.0,
                "estimated_global_y": 0.8,
                "estimated_global_z": 0.0,
            },
            {
                "entry_id": 0,
                "object_global_id": 2,
                "label": "table",
                "estimated_global_x": 1.0,
                "estimated_global_y": 1.2,
                "estimated_global_z": 0.0,
            },
        ],
    )

    assert len(rows) == 2
    assert rows[0]["source_obs_id"] == "obs_000001"
    assert rows[0]["target_obs_id"] == "obs_000002"
    assert rows[0]["direction"] == "right"
    assert rows[0]["vertical_direction"] == "above"
    assert rows[0]["dy"] == 0.3999999999999999
    assert rows[0]["distance_3d_m"] == 1.0770329614269007
    assert rows[0]["relation_type"] == "ObjectObject"
    assert rows[0]["relation_source"] == "geometry_postprocess"
    assert rows[1]["direction"] == "left"
    assert rows[1]["vertical_direction"] == "below"


def test_format_object_text_long_prefixes_angle_bucket_for_angle_split():
    assert _format_object_text_long("wooden chair", angle_bucket="left", builder_variant="angle_split") == (
        "left sector | wooden chair"
    )
    assert _format_object_text_long("wooden chair", angle_bucket="left", builder_variant="standard") == "wooden chair"


def test_make_object_record_keeps_core_fields_and_adds_geometry_metadata():
    scene = SimpleNamespace(
        view_type="living room",
        room_function="resting",
        style_hint="modern",
        clutter_level="low",
    )

    record = _make_object_record(
        object_global_id=0,
        frame_id=2,
        entry_id=4,
        file_name="images/sample.jpg",
        x=1.0,
        y=2.0,
        world_position=[1.0, 0.0, 2.0],
        orientation=90,
        parse_status="ok",
        builder_variant="angle_split",
        angle_split_enable=True,
        angle_step=30,
        scene_objects=scene,
        object_local_id="feat_000",
        label="chair",
        object_confidence=1.0,
        description="object: chair | attrs: wooden | anchor: x=1.0, z=2.0 | nearby: table@(1.5,2.5)",
        long_form_open_description="object: chair | attributes: wooden | camera_relation: distance=2.0, bearing=-30.0",
        attributes=["wooden"],
        laterality="left",
        distance_bin="middle",
        verticality="low",
        distance_from_camera_m=2.0,
        relative_height_from_camera_m=-0.8,
        relative_bearing_deg=-30.0,
        estimated_global_x=1.0,
        estimated_global_y=0.8,
        estimated_global_z=2.0,
        any_text="",
        location_relative_to_other_objects="table relation=right of chair d=0.8m anchor=(1.0,2.5)",
        surrounding_context=[
            {
                "label": "table",
                "attributes": ["round"],
                "distance_from_primary_m": 0.8,
                "distance_from_camera_m": 2.4,
                "relative_height_from_camera_m": -0.6,
                "relative_bearing_deg": 20.0,
                "estimated_global_x": 1.5,
                "estimated_global_y": 1.0,
                "estimated_global_z": 2.5,
                "relation_to_primary": "right of chair",
            }
        ],
        scene_attributes=["painted trim"],
        object_text_short="object: chair | attrs: wooden | anchor: x=1.0, z=2.0 | nearby: table@(1.5,2.5)",
        object_text_long="left sector | object: chair | attributes: wooden | camera_relation: distance=2.0, bearing=-30.0",
    )

    assert record["orientation"] == 90
    assert record["frame_orientation"] == 90
    assert record["object_orientation_deg"] == 60
    assert record["angle_bucket"] == "left"
    assert record["angle_split_step_deg"] == 30
    assert record["builder_variant"] == "angle_split"
    assert record["attributes"] == ["wooden"]
    assert record["relative_height_from_camera_m"] == -0.8
    assert record["relative_bearing_deg"] == -30.0
    assert record["estimated_global_x"] == 1.0
    assert record["estimated_global_y"] == 0.8
    assert record["estimated_global_z"] == 2.0
    assert record["surrounding_context"][0]["relative_height_from_camera_m"] == -0.6
    assert record["surrounding_context"][0]["estimated_global_y"] == 1.0
    assert record["surrounding_context"][0]["label"] == "table"
    assert record["scene_attributes"] == ["painted trim"]
    assert record["object_text_short"].startswith("object: chair")
    assert record["object_text_long"].startswith("left sector | object: chair")
    assert record["view_type"] == "living room"
    assert record["geometry_source"] == "vlm_fallback"


def test_build_view_attribute_collects_room_level_fields():
    scene = SimpleNamespace(
        view_type="living room",
        room_function="resting",
        style_hint="traditional",
        clutter_level="low",
        floor_pattern="carpet",
        lighting_ceiling="natural light source",
        wall_color="beige",
        scene_attributes=["painted trim", "sloped ceiling"],
        additional_notes="stairs on the right",
        image_summary="living room facing the back wall",
    )

    attribute = _build_view_attribute(scene_objects=scene)

    assert attribute == {
        "view_type": "living room",
        "room_function": "resting",
        "style_hint": "traditional",
        "clutter_level": "low",
        "floor_pattern": "carpet",
        "lighting_ceiling": "natural light source",
        "wall_color": "beige",
        "scene_attributes": ["painted trim", "sloped ceiling"],
        "additional_notes": "stairs on the right",
        "image_summary": "living room facing the back wall",
    }


def test_load_resume_state_backfills_attribute_from_raw_vlm_output(tmp_path):
    raw_vlm_output = {
        "view_type": "hallway",
        "room_function": "circulation",
        "style_hint": "traditional",
        "clutter_level": "low",
        "scene_attributes": ["textured painted wall", "framed nautical artwork"],
        "visual_feature": [],
        "floor_pattern": "carpet",
        "lighting_ceiling": "natural light source",
        "wall_color": "light blue-gray",
        "additional_notes": "entry corridor",
        "image_summary": "hallway facing a white door",
    }
    meta_row = {
        "id": 0,
        "x": 1.0,
        "y": 2.0,
        "world_position": [1.0, 1.6, 2.0],
        "orientation": 90,
        "file_name": "images/sample.jpg",
        "text": "short text",
        "frame_text_short": "short text",
        "frame_text_long": "long text",
        "parse_status": "ok",
        "parse_warnings": [],
        "raw_vlm_output": json.dumps(raw_vlm_output, ensure_ascii=True),
        "raw_api_source": "api",
        "text_input_for_clip_short": "short text",
        "text_input_for_clip_long": "long text",
        "object_text_inputs_short": ["obj short"],
        "object_text_inputs_long": ["obj long"],
        "builder_variant": "angle_split",
        "object_prompt_variant": "angle_split",
        "object_count": 1,
    }
    (tmp_path / "meta.jsonl").write_text(json.dumps(meta_row, ensure_ascii=True) + "\n", encoding="utf-8")
    np.save(tmp_path / "image_emb.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(tmp_path / "text_emb_short.npy", np.asarray([[0.0, 1.0]], dtype=np.float32))
    np.save(tmp_path / "text_emb_long.npy", np.asarray([[0.5, 0.5]], dtype=np.float32))

    state = _load_resume_state(tmp_path, emb_dim=2)

    assert state["metadata_records"][0]["attribute"] == {
        "view_type": "hallway",
        "room_function": "circulation",
        "style_hint": "traditional",
        "clutter_level": "low",
        "floor_pattern": "carpet",
        "lighting_ceiling": "natural light source",
        "wall_color": "light blue-gray",
        "scene_attributes": ["textured painted wall", "framed nautical artwork"],
        "additional_notes": "entry corridor",
        "image_summary": "hallway facing a white door",
    }


def test_response_has_length_finish_reason_matches_chat_completion_choice():
    assert _response_has_length_finish_reason({"choices": [{"finish_reason": "length"}]})
    assert not _response_has_length_finish_reason({"choices": [{"finish_reason": "stop"}]})


def test_should_reuse_existing_entry_rejects_length_truncated_cache():
    reusable = _should_reuse_existing_entry(
        existing_meta={"file_name": "images/sample.jpg"},
        existing_raw_api={"raw_api_response": {"choices": [{"finish_reason": "stop"}]}},
        existing_image_emb=[1.0, 0.0],
        existing_text_emb_short=[1.0, 0.0],
        existing_text_emb_long=[1.0, 0.0],
        existing_object_group=[({"label": "chair"}, [1.0, 0.0], [1.0, 0.0])],
        expected_file_name="images/sample.jpg",
    )
    not_reusable = _should_reuse_existing_entry(
        existing_meta={"file_name": "images/sample.jpg"},
        existing_raw_api={"raw_api_response": {"choices": [{"finish_reason": "length"}]}},
        existing_image_emb=[1.0, 0.0],
        existing_text_emb_short=[1.0, 0.0],
        existing_text_emb_long=[1.0, 0.0],
        existing_object_group=[({"label": "chair"}, [1.0, 0.0], [1.0, 0.0])],
        expected_file_name="images/sample.jpg",
    )

    assert reusable
    assert not not_reusable


def test_should_reuse_existing_entry_requires_geometry_fields_when_enabled():
    reusable = _should_reuse_existing_entry(
        existing_meta={"file_name": "images/sample.jpg"},
        existing_raw_api={"raw_api_response": {"choices": [{"finish_reason": "stop"}]}},
        existing_image_emb=[1.0, 0.0],
        existing_text_emb_short=[1.0, 0.0],
        existing_text_emb_long=[1.0, 0.0],
        existing_object_group=[({"label": "chair", "geometry_source": "mask_depth"}, [1.0, 0.0], [1.0, 0.0])],
        expected_file_name="images/sample.jpg",
        require_geometry_fields=True,
    )
    not_reusable = _should_reuse_existing_entry(
        existing_meta={"file_name": "images/sample.jpg"},
        existing_raw_api={"raw_api_response": {"choices": [{"finish_reason": "stop"}]}},
        existing_image_emb=[1.0, 0.0],
        existing_text_emb_short=[1.0, 0.0],
        existing_text_emb_long=[1.0, 0.0],
        existing_object_group=[({"label": "chair"}, [1.0, 0.0], [1.0, 0.0])],
        expected_file_name="images/sample.jpg",
        require_geometry_fields=True,
    )

    assert reusable
    assert not not_reusable


def test_write_floor_plan_projection_persists_expected_fields(tmp_path):
    out = tmp_path / "floor_plan_projection.json"
    saved = _write_floor_plan_projection(
        out,
        {
            "view_min_x": -1.0,
            "view_max_x": 2.0,
            "view_min_z": -3.0,
            "view_max_z": 4.0,
            "ignored": 99,
        },
    )

    assert saved == str(out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip().startswith("{")


def test_run_optional_polar_surrounding_postprocess_is_noop_when_disabled(tmp_path):
    summary = _run_optional_polar_surrounding_postprocess(tmp_path, enabled=False)

    assert summary == {"enabled": False, "ran": False, "ok": False}


def test_run_optional_polar_surrounding_postprocess_calls_builder_when_enabled(tmp_path, monkeypatch):
    calls = []

    def _fake_build(db_dir: str):
        calls.append(db_dir)
        return {"relation_output_path": str(tmp_path / "object_polar_relations.jsonl"), "num_pair_relations": 3}

    monkeypatch.setattr("spatial_rag.spatial_db_builder._get_polar_surroundings_builder", lambda: _fake_build)

    summary = _run_optional_polar_surrounding_postprocess(tmp_path, enabled=True)

    assert calls == [str(tmp_path)]
    assert summary["enabled"] is True
    assert summary["ran"] is True
    assert summary["ok"] is True
    assert summary["num_pair_relations"] == 3


def test_build_spatial_database_angle_split_forwards_polar_postprocess_flag(monkeypatch):
    captured = {}

    def _fake_core(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("spatial_rag.spatial_db_builder._build_spatial_database_core", _fake_core)

    report = build_spatial_database_angle_split(run_polar_surrounding_postprocess=True)

    assert report == {"ok": True}
    assert captured["run_polar_surrounding_postprocess"] is True
