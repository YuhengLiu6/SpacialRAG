import json

from spatial_rag.object_parser import parse_scene_objects


def test_parse_scene_objects_home_schema_success():
    raw = {
        "view_type": "living room",
        "room_function": "resting",
        "style_hint": "modern",
        "clutter_level": "low",
        "scene_attributes": ["painted trim on left wall"],
        "visual_feature": [
            {
                "type": "chair",
                "description": "wooden chair near door",
                "attributes": ["wooden", "brown"],
                "relative_position_laterality": "left",
                "relative_position_distance": "near",
                "relative_position_verticality": "low",
                "distance_from_camera_m": "1.4",
                "relative_height_from_camera_m": "-0.7",
                "relative_bearing_deg": "25",
                "support_relation": "freestanding",
                "any_text": "",
                "Long form open description": "A wooden chair with curved backrest.",
                "Location relative to other objects in the environment": "Left of the door.",
                "surrounding_context": [
                    {
                        "label": "door",
                        "attributes": ["gray"],
                        "distance_from_primary_m": 0.8,
                        "distance_from_camera_m": 1.1,
                        "relative_height_from_camera_m": -0.2,
                        "relative_bearing_deg": -5.0,
                        "relation_to_primary": "right of chair",
                    }
                ],
            }
        ],
        "floor_pattern": "wood",
        "lighting_ceiling": "recessed lights",
        "wall color": "off-white",
        "additional_notes": "none",
        "image_summary": "A living room with a single chair.",
    }

    result = parse_scene_objects(json.dumps(raw), image_context={"image_id": "frame_0001"})
    assert result.parse_status == "ok"
    assert result.scene_objects is not None
    assert result.scene_objects.view_type == "living room"
    assert result.scene_objects.room_function == "resting"
    assert result.scene_objects.style_hint == "modern"
    assert result.scene_objects.clutter_level == "low"
    assert result.scene_objects.wall_color == "off-white"
    assert result.scene_objects.scene_attributes == ["painted trim on left wall"]
    assert len(result.scene_objects.visual_feature) == 1

    feat = result.scene_objects.visual_feature[0]
    assert feat.type == "chair"
    assert feat.attributes == ["wooden", "brown"]
    assert feat.relative_position_laterality == "left"
    assert feat.relative_position_distance == "near"
    assert feat.relative_position_verticality == "low"
    assert feat.distance_from_camera_m == 1.4
    assert feat.relative_height_from_camera_m == -0.7
    assert feat.relative_bearing_deg == 25.0
    assert feat.support_relation == "freestanding"
    assert len(feat.surrounding_context) == 1
    assert feat.surrounding_context[0].label == "door"
    assert feat.surrounding_context[0].relative_height_from_camera_m == -0.2
    assert feat.surrounding_context[0].relative_bearing_deg == -5.0
    assert "wooden" in feat.long_form_open_description.lower()


def test_parse_scene_objects_migrates_background_features_to_scene_attributes():
    raw = {
        "view_type": "hallway",
        "room_function": "circulation",
        "style_hint": "traditional",
        "clutter_level": "low",
        "visual_feature": [
            {
                "type": "wall feature",
                "description": "painted crown molding",
                "attributes": ["cream", "ornate"],
                "relative_position_laterality": "left",
                "relative_position_distance": "middle",
                "relative_position_verticality": "high",
                "distance_from_camera_m": None,
                "relative_bearing_deg": None,
                "support_relation": "attached_to",
                "any_text": "",
                "long_form_open_description": "Decorative molding running along the top of the wall.",
                "location_relative_to_other_objects": "",
                "surrounding_context": [],
            },
            {
                "type": "floor pattern",
                "description": "wood plank floor transition",
                "attributes": ["wood", "striped"],
                "relative_position_laterality": "center",
                "relative_position_distance": "near",
                "relative_position_verticality": "low",
                "distance_from_camera_m": None,
                "relative_bearing_deg": None,
                "support_relation": "unknown",
                "any_text": "",
                "long_form_open_description": "Wood planks meeting a rug edge near the doorway.",
                "location_relative_to_other_objects": "",
                "surrounding_context": [],
            },
        ],
        "floor_pattern": "wood",
        "lighting_ceiling": "natural light source",
        "wall_color": "white",
        "additional_notes": "",
        "image_summary": "A hallway with decorative trim and wood flooring.",
    }

    result = parse_scene_objects(raw)

    assert result.parse_status == "ok"
    assert result.scene_objects is not None
    assert result.scene_objects.visual_feature == []
    assert any("wall detail:" in attr for attr in result.scene_objects.scene_attributes)
    assert any("floor detail:" in attr for attr in result.scene_objects.scene_attributes)
    assert any("background feature migrated" in warning for warning in result.warnings)


def test_parse_scene_objects_legacy_object_v1_conversion():
    raw = {
        "schema_version": "object_v1",
        "objects": [
            {
                "meta": {
                    "object_id": "obj_a",
                    "label": "chair",
                    "confidence": 0.95,
                    "attributes": ["wooden", "brown"],
                },
                "location": {
                    "reference_frame": "image",
                    "bbox_xywh_norm": [0.0, 0.2, 0.4, 0.5],
                },
                "orientation": {"facing": "LEFT", "yaw_deg": 89.3, "confidence": 0.8},
            }
        ],
    }

    result = parse_scene_objects(json.dumps(raw), image_context={"image_id": "frame_legacy"})
    assert result.parse_status == "ok"
    assert result.scene_objects is not None
    assert len(result.scene_objects.visual_feature) == 1

    feat = result.scene_objects.visual_feature[0]
    assert feat.type == "chair"
    assert feat.attributes == ["wooden", "brown"]
    assert feat.relative_position_laterality == "left"
    assert feat.relative_position_distance in {"near", "middle", "far"}
    assert feat.relative_bearing_deg is None
    assert feat.support_relation == "unknown"
    assert any("legacy object_v1" in w for w in result.warnings)


def test_parse_scene_objects_failure_for_non_json():
    result = parse_scene_objects("not a json")
    assert result.parse_status == "failed"
    assert result.scene_objects is None
    assert result.warnings


def test_parse_scene_objects_normalizes_clock_aliases():
    raw = {
        "view_type": "living room",
        "room_function": "resting",
        "style_hint": "traditional",
        "clutter_level": "low",
        "scene_attributes": [],
        "visual_feature": [
            {
                "type": "wall clock",
                "description": "round wall clock",
                "attributes": ["round", "roman numerals"],
                "relative_position_laterality": "right",
                "relative_position_distance": "middle",
                "relative_position_verticality": "high",
                "distance_from_camera_m": 2.0,
                "relative_height_from_camera_m": 0.5,
                "relative_bearing_deg": 20.0,
                "support_relation": "attached_to",
                "any_text": "",
                "long_form_open_description": "Large decorative wall clock.",
                "location_relative_to_other_objects": "",
                "surrounding_context": [],
            }
        ],
        "floor_pattern": "wood",
        "lighting_ceiling": "mixed lighting",
        "wall_color": "beige",
        "additional_notes": "",
        "image_summary": "A living room with a wall clock.",
    }

    result = parse_scene_objects(raw)

    assert result.parse_status == "ok"
    assert result.scene_objects is not None
    assert result.scene_objects.visual_feature[0].type == "clock"
