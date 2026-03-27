from spatial_rag.object_canonicalizer import (
    EMPTY_OBJECT_SENTINEL,
    UNKNOWN_TEXT_TOKEN,
    canonical_object_line,
    canonical_scene_text,
    canonicalize_scene_objects,
    collect_object_texts,
    compose_frame_text,
    select_object_text,
)
from spatial_rag.object_parser import parse_scene_objects


def test_canonicalize_is_deterministic_and_includes_anchor_fields():
    raw = {
        "view_type": "living room",
        "room_function": "resting",
        "style_hint": "modern",
        "clutter_level": "medium",
        "scene_attributes": ["painted trim"],
        "visual_feature": [
            {
                "feature_id": "feat_2",
                "type": "chair",
                "description": "chair near wall",
                "attributes": ["blue", "wooden"],
                "relative_position_laterality": "right",
                "relative_position_distance": "middle",
                "relative_position_verticality": "low",
                "distance_from_camera_m": 2.0,
                "relative_bearing_deg": -15.0,
                "estimated_global_x": 1.2,
                "estimated_global_z": -2.4,
                "support_relation": "freestanding",
                "any_text": "",
                "long_form_open_description": "A wooden chair.",
                "location_relative_to_other_objects": "Right side.",
                "surrounding_context": [],
            },
            {
                "feature_id": "feat_1",
                "type": "door",
                "description": "gray door",
                "attributes": ["gray", "metal handle"],
                "relative_position_laterality": "left",
                "relative_position_distance": "near",
                "relative_position_verticality": "middle",
                "distance_from_camera_m": 1.2,
                "relative_bearing_deg": 20.0,
                "estimated_global_x": 0.5,
                "estimated_global_z": -1.0,
                "support_relation": "attached_to",
                "any_text": "PULL",
                "long_form_open_description": "Door with metal handle.",
                "location_relative_to_other_objects": "Left of chair.",
                "surrounding_context": [
                    {
                        "label": "chair",
                        "attributes": ["blue"],
                        "distance_from_primary_m": 1.4,
                        "distance_from_camera_m": 2.0,
                        "relative_bearing_deg": -15.0,
                        "estimated_global_x": 1.2,
                        "estimated_global_z": -2.4,
                        "relation_to_primary": "right of door",
                    }
                ],
            },
        ],
        "floor_pattern": "wood",
        "lighting_ceiling": "recessed lights",
        "wall_color": "white",
        "additional_notes": "",
        "image_summary": "A living room with a door and a chair.",
    }
    parsed = parse_scene_objects(raw)
    assert parsed.scene_objects is not None

    lines_a = canonicalize_scene_objects(parsed.scene_objects, max_objects=24, object_text_mode="short")
    lines_b = canonicalize_scene_objects(parsed.scene_objects, max_objects=24, object_text_mode="short")
    text_a = canonical_scene_text(parsed.scene_objects, max_objects=24, object_text_mode="short")
    text_b = canonical_scene_text(parsed.scene_objects, max_objects=24, object_text_mode="short")

    lines_long_a = canonicalize_scene_objects(parsed.scene_objects, max_objects=24, object_text_mode="long")
    lines_long_b = canonicalize_scene_objects(parsed.scene_objects, max_objects=24, object_text_mode="long")
    text_long_a = canonical_scene_text(parsed.scene_objects, max_objects=24, object_text_mode="long")
    text_long_b = canonical_scene_text(parsed.scene_objects, max_objects=24, object_text_mode="long")

    assert lines_a == lines_b
    assert text_a == text_b
    assert lines_long_a == lines_long_b
    assert text_long_a == text_long_b
    assert lines_a[0].startswith("feature=feat_1;")
    assert "support=attached_to" in lines_a[0]
    assert "bearing=20.0" in lines_a[0]
    assert "gx=0.5" in lines_a[0]
    assert "ctx=chair@1.0,-2.5~1.4#right of door" in lines_a[0]
    assert "scene_attributes=painted trim" in text_a


def test_object_text_mode_long_generates_hierarchy_with_na_anchor_when_missing():
    raw = {
        "view_type": "living room",
        "room_function": "resting",
        "style_hint": "modern",
        "clutter_level": "low",
        "visual_feature": [
            {
                "feature_id": "feat_0",
                "type": "table",
                "description": "round wooden coffee table",
                "attributes": ["round", "wooden"],
                "relative_position_laterality": "center",
                "relative_position_distance": "near",
                "relative_position_verticality": "low",
                "distance_from_camera_m": 1.0,
                "relative_bearing_deg": None,
                "support_relation": "freestanding",
                "any_text": "",
                "long_form_open_description": "",
                "location_relative_to_other_objects": "In front of couch.",
                "surrounding_context": [],
            }
        ],
        "floor_pattern": "wood",
        "lighting_ceiling": "natural light source",
        "wall_color": "white",
        "additional_notes": "",
        "image_summary": "A simple living room.",
    }
    parsed = parse_scene_objects(raw)
    assert parsed.scene_objects is not None

    obj = parsed.scene_objects.visual_feature[0]
    line_short = canonical_object_line(obj, object_text_mode="short")
    line_long = canonical_object_line(obj, object_text_mode="long")
    assert "desc=object: table / attrs: round, wooden / anchor: x=na, z=na / nearby: none" in line_short
    assert "desc=object: table / attributes: round, wooden / camera_relation: distance=1.0, bearing=na" in line_long
    assert select_object_text(obj, mode="short").startswith("object: table | attrs: round, wooden")
    assert "global_anchor: x=na, z=na" in select_object_text(obj, mode="long")


def test_collect_and_compose_use_generated_short_long_hierarchy_text():
    raw = {
        "view_type": "living room",
        "room_function": "resting",
        "style_hint": "modern",
        "clutter_level": "medium",
        "scene_attributes": ["painted trim"],
        "visual_feature": [
            {
                "feature_id": "feat_2",
                "type": "chair",
                "description": "blue chair",
                "attributes": ["blue", "curved back"],
                "relative_position_laterality": "right",
                "relative_position_distance": "middle",
                "relative_position_verticality": "low",
                "distance_from_camera_m": 2.0,
                "relative_bearing_deg": -10.0,
                "estimated_global_x": 1.0,
                "estimated_global_z": -2.0,
                "support_relation": "freestanding",
                "any_text": "",
                "long_form_open_description": "A bright blue chair with curved back.",
                "location_relative_to_other_objects": "Right side.",
                "surrounding_context": [],
            },
            {
                "feature_id": "feat_1",
                "type": "door",
                "description": "gray door",
                "attributes": ["gray"],
                "relative_position_laterality": "left",
                "relative_position_distance": "near",
                "relative_position_verticality": "middle",
                "distance_from_camera_m": 1.2,
                "relative_bearing_deg": 15.0,
                "estimated_global_x": 0.5,
                "estimated_global_z": -1.0,
                "support_relation": "attached_to",
                "any_text": "PULL",
                "long_form_open_description": "A gray wooden door with metal handle.",
                "location_relative_to_other_objects": "Left side.",
                "surrounding_context": [],
            },
        ],
        "floor_pattern": "wood",
        "lighting_ceiling": "recessed lights",
        "wall_color": "white",
        "additional_notes": "",
        "image_summary": "",
    }
    parsed = parse_scene_objects(raw)
    assert parsed.scene_objects is not None

    short_texts = collect_object_texts(parsed.scene_objects, max_objects=24, mode="short")
    long_texts = collect_object_texts(parsed.scene_objects, max_objects=24, mode="long")

    assert short_texts == [
        "object: door | attrs: gray | anchor: x=0.5, z=-1.0 | nearby: none",
        "object: chair | attrs: blue, curved back | anchor: x=1.0, z=-2.0 | nearby: none",
    ]
    assert long_texts[0].startswith(
        "object: door | attributes: gray | camera_relation: distance=1.2, bearing=15.0"
    )
    assert long_texts[1].startswith(
        "object: chair | attributes: blue, curved back | camera_relation: distance=2.0, bearing=-10.0"
    )
    assert compose_frame_text(parsed.scene_objects, max_objects=24, mode="short").startswith(
        "object: door | attrs: gray | anchor: x=0.5, z=-1.0 | nearby: none | object: chair"
    )


def test_canonicalize_empty_uses_sentinel():
    raw = {
        "view_type": "unknown",
        "room_function": "unknown",
        "style_hint": "unknown",
        "clutter_level": "unknown",
        "visual_feature": [],
        "floor_pattern": "unknown",
        "lighting_ceiling": "unknown",
        "wall_color": "unknown",
        "additional_notes": "",
        "image_summary": "",
    }
    parsed = parse_scene_objects(raw)
    assert parsed.scene_objects is not None

    lines = canonicalize_scene_objects(parsed.scene_objects, max_objects=24)
    assert lines == [EMPTY_OBJECT_SENTINEL]
    assert collect_object_texts(parsed.scene_objects, max_objects=24, mode="short") == [UNKNOWN_TEXT_TOKEN]
    assert compose_frame_text(parsed.scene_objects, max_objects=24, mode="long") == UNKNOWN_TEXT_TOKEN
