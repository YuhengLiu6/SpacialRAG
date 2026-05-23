import json

from spatial_rag.config import OBJECT_SURROUNDING_MAX
from spatial_rag.household_taxonomy import COMMON_PRELIST_OBJECT_TYPES
from spatial_rag.vlm_captioner import VLMCaptioner


class _FakeChatCompletionResponse:
    def __init__(self, text: str):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": text})()})()]


class _FakeCompletionsClient:
    def __init__(self, text: str):
        self.text = str(text)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _FakeChatCompletionResponse(self.text)


class _FakeOpenAIClient:
    def __init__(self, text: str):
        self.completions = _FakeCompletionsClient(text)
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_object_crop_prompt_version_bumped_for_vlm_label_generation():
    assert VLMCaptioner._object_crop_prompt_version() == "object_crop_descriptor_builder_aligned_v8_prelist_label"


def test_object_crop_prompt_version_differs_when_occlusion_requested():
    assert VLMCaptioner._object_crop_prompt_version(include_occlusion=True) != VLMCaptioner._object_crop_prompt_version()


def test_batched_object_prompt_version_bumped_for_text_only_batch_schema():
    assert VLMCaptioner._batched_object_description_prompt_version() == "object_detection_batch_descriptor_textonly_v5_prelist_label"


def test_object_prompt_version_differs_between_standard_and_angle_split():
    assert VLMCaptioner._object_prompt_version("standard") != VLMCaptioner._object_prompt_version("angle_split")


def test_default_crop_description_schema_fields_exist():
    payload = VLMCaptioner._default_object_crop_description()
    assert payload["label"] == "unknown"
    assert payload["short_description"] == "unknown"
    assert payload["long_description"] == "unknown"
    assert payload["attributes"] == []
    assert payload["distance_from_camera_m"] is None


def test_crop_response_schema_contains_required_fields():
    schema = VLMCaptioner._object_crop_response_schema()
    required = schema["schema"]["required"]
    assert required == ["label", "short_description", "long_description", "attributes", "distance_from_camera_m"]
    props = schema["schema"]["properties"]
    assert "label" in props
    assert "short_description" in props
    assert "long_description" in props
    assert "attributes" in props
    assert "distance_from_camera_m" in props
    assert props["label"]["enum"] == [*COMMON_PRELIST_OBJECT_TYPES, "unknown"]


def test_crop_response_schema_includes_occlusion_when_requested():
    schema = VLMCaptioner._object_crop_response_schema(include_occlusion=True)
    required = schema["schema"]["required"]
    assert "occlusion_level" in required
    props = schema["schema"]["properties"]
    assert "occlusion_level" in props


def test_batched_detected_object_schema_is_text_only():
    schema = VLMCaptioner._batched_detected_objects_response_schema(max_objects=3)
    item_schema = schema["schema"]["properties"]["objects"]["items"]
    assert "occlusion_level" not in item_schema["required"]
    assert "occlusion_level" not in item_schema["properties"]


def test_object_cache_path_varies_by_prompt_variant_and_camera_context(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image")
    captioner = VLMCaptioner(use_cache=False, object_use_cache=False, cache_dir=str(tmp_path))

    standard = captioner._object_cache_path(str(image_path), prompt_variant="standard")
    angle_split = captioner._object_cache_path(str(image_path), prompt_variant="angle_split")
    with_pose = captioner._object_cache_path(
        str(image_path),
        prompt_variant="standard",
        camera_context={"camera_x": 1.0, "camera_y": 2.0, "camera_orientation_deg": 90.0},
    )

    assert standard != angle_split
    assert standard != with_pose


def test_object_cache_path_uses_structured_directory_layout(tmp_path):
    image_path = tmp_path / "images" / "pose_00000_o090_000001.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-image")
    captioner = VLMCaptioner(use_cache=False, object_use_cache=True, object_cache_dir=str(tmp_path / "vlm_object_cache"))

    cache_path = captioner._object_cache_path(
        str(image_path),
        prompt_variant="angle_split",
        camera_context={"camera_x": 1.0, "camera_y": 2.0, "camera_orientation_deg": 90.0},
    )

    rel_parts = cache_path.relative_to(captioner.object_cache_dir).parts
    assert rel_parts[0] == "scene_objects"
    assert rel_parts[1] == "gpt-4o-mini"
    assert rel_parts[2] == "home_prompt_angle_split_surrounding_anchor_height_hierarchy_v3"
    assert rel_parts[3] == "pose_00000"
    assert cache_path.name.startswith("pose_00000_o090_000001__")
    assert cache_path.name.endswith(".objects.json")


def test_cluster_text_compression_user_prompt_includes_member_spatial_relations():
    prompt = VLMCaptioner._cluster_text_compression_user_prompt(
        ["brown wooden chair", "side view of the same chair"],
        member_count=2,
        member_spatial_relations=[
            {
                "member_index": 1,
                "object_global_id": 10,
                "label": "chair",
                "direction_8": "W",
                "direction_text": "west",
                "distance_m": 1.0,
            },
            {
                "member_index": 2,
                "object_global_id": 11,
                "label": "chair",
                "direction_8": "E",
                "direction_text": "east",
                "distance_m": 1.0,
            },
        ],
    )

    assert "Member spatial relations:" in prompt
    assert "member 1, object_id=10, label=chair: west, 1.00m from cluster prototype on the global x-z plane" in prompt
    assert "member 2, object_id=11, label=chair: east, 1.00m from cluster prototype on the global x-z plane" in prompt


def test_cluster_text_compression_cache_path_varies_by_member_spatial_relations(tmp_path):
    captioner = VLMCaptioner(use_cache=False, object_use_cache=True, object_cache_dir=str(tmp_path / "vlm_object_cache"))

    cache_path_a = captioner._cluster_text_compression_cache_path(
        ["brown wooden chair"],
        member_spatial_relations=[
            {
                "member_index": 1,
                "object_global_id": 10,
                "label": "chair",
                "direction_8": "W",
                "direction_text": "west",
                "distance_m": 1.0,
            }
        ],
    )
    cache_path_b = captioner._cluster_text_compression_cache_path(
        ["brown wooden chair"],
        member_spatial_relations=[
            {
                "member_index": 1,
                "object_global_id": 10,
                "label": "chair",
                "direction_8": "E",
                "direction_text": "east",
                "distance_m": 1.0,
            }
        ],
    )

    assert cache_path_a != cache_path_b


def test_compress_cluster_member_texts_cache_payload_includes_member_spatial_relations(tmp_path):
    captioner = VLMCaptioner(use_cache=False, object_use_cache=True, object_cache_dir=str(tmp_path / "vlm_object_cache"))
    fake_client = _FakeOpenAIClient("compressed spatial chair")
    captioner.client = fake_client
    member_spatial_relations = [
        {
            "member_index": 1,
            "object_global_id": 10,
            "label": "chair",
            "direction_8": "W",
            "direction_text": "west",
            "distance_m": 1.0,
        }
    ]

    result = captioner.compress_cluster_member_texts(
        ["brown wooden chair", "side view of the same chair"],
        member_count=2,
        member_spatial_relations=member_spatial_relations,
    )

    cache_path = captioner._cluster_text_compression_cache_path(
        ["brown wooden chair", "side view of the same chair"],
        member_spatial_relations=member_spatial_relations,
    )
    cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))

    assert result == "compressed spatial chair"
    assert cache_path.exists()
    assert cached_payload["cluster_text_description"] == "compressed spatial chair"
    assert cached_payload["member_spatial_relations"] == member_spatial_relations
    user_prompt = fake_client.completions.calls[0]["messages"][1]["content"]
    assert "Member spatial relations:" in user_prompt


def test_selector_cache_legacy_file_is_promoted_to_structured_path(tmp_path):
    image_path = tmp_path / "images" / "pose_00000_o000_000000.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-image")
    cache_root = tmp_path / "vlm_object_cache"
    captioner = VLMCaptioner(use_cache=False, object_use_cache=True, object_cache_dir=str(cache_root))

    legacy_path = captioner._legacy_selector_cache_path(
        str(image_path),
        camera_context={"camera_x": 0.0, "camera_y": 0.0, "camera_orientation_deg": 0.0},
    )
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "payload": {"selected_object_types": ["chair"], "image_summary": "test summary"},
        "raw_api_response": {"choices": [{"finish_reason": "stop"}]},
    }
    legacy_path.write_text(json.dumps(legacy_payload, ensure_ascii=True), encoding="utf-8")

    result = captioner.select_object_types_with_meta(
        str(image_path),
        camera_context={"camera_x": 0.0, "camera_y": 0.0, "camera_orientation_deg": 0.0},
    )

    new_path = captioner._selector_cache_path(
        str(image_path),
        camera_context={"camera_x": 0.0, "camera_y": 0.0, "camera_orientation_deg": 0.0},
    )
    assert result["source"] == "cache"
    assert result["payload"]["selected_object_types"] == ["chair"]
    assert new_path.exists()
    assert json.loads(new_path.read_text(encoding="utf-8"))["payload"]["selected_object_types"] == ["chair"]


def test_object_crop_cache_path_groups_by_view_directory(tmp_path):
    crop_path = tmp_path / "geometry" / "view_00007" / "objects" / "obj_003_crop.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake-image")
    captioner = VLMCaptioner(use_cache=False, object_use_cache=True, object_cache_dir=str(tmp_path / "vlm_object_cache"))

    cache_path = captioner._object_crop_cache_path(str(crop_path))

    rel_parts = cache_path.relative_to(captioner.object_cache_dir).parts
    assert rel_parts[0] == "object_crops"
    assert rel_parts[3] == "view_00007"
    assert cache_path.name.startswith("obj_003_crop__")
    assert cache_path.name.endswith(".crop.json")


def test_object_crop_cache_path_differs_when_occlusion_requested(tmp_path):
    crop_path = tmp_path / "geometry" / "view_00007" / "objects" / "obj_003_crop.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake-image")
    captioner = VLMCaptioner(use_cache=False, object_use_cache=True, object_cache_dir=str(tmp_path / "vlm_object_cache"))

    default_cache_path = captioner._object_crop_cache_path(str(crop_path))
    occlusion_cache_path = captioner._object_crop_cache_path(str(crop_path), include_occlusion=True)

    assert default_cache_path != occlusion_cache_path


def test_angle_split_prompt_mentions_new_geometry_and_scene_attributes():
    prompt = VLMCaptioner._object_user_prompt(
        max_objects=8,
        prompt_variant="angle_split",
        camera_context={"camera_x": 1.5, "camera_y": -2.0, "camera_orientation_deg": 270.0},
    )

    assert "three discrete sectors" in prompt
    assert "left third" in prompt
    assert "relative_bearing_deg" in prompt
    assert "relative_height_from_camera_m" in prompt
    assert "scene_attributes" in prompt
    assert "surrounding_context" in prompt
    assert "orientation_deg=270.0" in prompt
    assert "horizontal FOV is 90.0 degrees" in prompt
    assert "left image edge are about -45 degrees" in prompt
    assert "right image edge are about +45 degrees" in prompt
    assert "negative means left of image center" in prompt
    assert "positive means right of image center" in prompt


def test_object_response_schema_contains_surrounding_and_scene_attributes():
    schema = VLMCaptioner._object_response_schema(max_objects=8)
    top_required = schema["schema"]["required"]
    assert "scene_attributes" in top_required
    props = schema["schema"]["properties"]
    assert "scene_attributes" in props
    feature_props = props["visual_feature"]["items"]["properties"]
    assert "attributes" in feature_props
    assert "relative_bearing_deg" in feature_props
    assert "relative_height_from_camera_m" in feature_props
    assert "surrounding_context" in feature_props
    assert feature_props["surrounding_context"]["maxItems"] == int(OBJECT_SURROUNDING_MAX)
    surrounding_props = feature_props["surrounding_context"]["items"]["properties"]
    assert "relative_height_from_camera_m" in surrounding_props
    assert "wall feature" not in feature_props["type"]["enum"]
    assert "floor pattern" not in feature_props["type"]["enum"]
    assert "clock" in feature_props["type"]["enum"]


def test_default_object_json_includes_scene_attributes():
    payload = json.loads(VLMCaptioner._default_object_json())
    assert payload["scene_attributes"] == []
    assert payload["visual_feature"] == []


def test_object_crop_user_prompt_includes_detector_context():
    prompt = VLMCaptioner._object_crop_user_prompt("chair", "0.716")

    assert '"chair"' in prompt
    assert "0.716" in prompt
    assert "Treat this detected class only as a hint" in prompt
    assert "Choose label from this same household pre-list" in prompt
    assert "If none of those labels fit the visible object, return unknown." in prompt


def test_batched_detected_objects_prompt_no_longer_requests_occlusion_level():
    system_prompt = VLMCaptioner._batched_detected_objects_system_prompt()
    user_prompt = VLMCaptioner._batched_detected_objects_user_prompt(
        [
            {
                "object_local_id": "det_000",
                "detector_label": "chair",
                "detector_confidence": 0.912,
                "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
                "bbox_xywh_norm": [0.1, 0.2, 0.3, 0.4],
            }
        ]
    )

    assert "occlusion_level" not in system_prompt
    assert "occlusion_level" not in user_prompt
    assert "Judge occlusion for the detector-localized object instance itself" not in user_prompt
    assert "Use the detector-provided class only as a localization hint" in system_prompt
    assert "Allowed label values:" in system_prompt
    assert "choose label from this same household pre-list" in user_prompt
    assert "Return exactly one JSON object per listed object_local_id" in user_prompt


def test_object_description_schema_uses_non_empty_strings():
    schema = VLMCaptioner._object_crop_response_schema()
    props = schema["schema"]["properties"]
    assert props["label"]["minLength"] == 1
    assert props["short_description"]["minLength"] == 1
    assert props["long_description"]["minLength"] == 1


def test_object_description_schema_constrains_label_to_prelist_and_unknown():
    schema = VLMCaptioner._batched_detected_objects_response_schema(max_objects=2)
    item_props = schema["schema"]["properties"]["objects"]["items"]["properties"]
    assert item_props["label"]["enum"] == [*COMMON_PRELIST_OBJECT_TYPES, "unknown"]


def test_normalize_object_description_forces_out_of_prelist_label_to_unknown():
    payload = VLMCaptioner._normalize_object_description_payload(
        {"label": "sofa", "short_description": "gray sofa", "long_description": "gray sofa"},
        default_payload=VLMCaptioner._default_object_crop_description(),
        label_hint="chair",
        include_occlusion=False,
    )

    assert payload["label"] == "unknown"


def test_batched_object_schema_requires_non_empty_object_local_id():
    schema = VLMCaptioner._batched_detected_objects_response_schema(max_objects=2)
    item_props = schema["schema"]["properties"]["objects"]["items"]["properties"]
    assert item_props["object_local_id"]["minLength"] == 1


def test_normalize_object_description_does_not_replace_unknown_label_with_hint():
    payload = VLMCaptioner._normalize_object_description_payload(
        {},
        default_payload=VLMCaptioner._default_object_crop_description(),
        label_hint="chair",
        include_occlusion=False,
    )

    assert payload["label"] == "unknown"


def test_response_has_length_finish_reason_detects_truncated_cache_payload():
    assert VLMCaptioner._response_has_length_finish_reason(
        {"choices": [{"finish_reason": "length"}]}
    )
    assert not VLMCaptioner._response_has_length_finish_reason(
        {"choices": [{"finish_reason": "stop"}]}
    )


def test_selector_defaults_and_schema_include_selected_object_types():
    payload = VLMCaptioner._default_selector_payload()
    schema = VLMCaptioner._selector_response_schema()

    assert payload["selected_object_types"] == []
    assert "selected_object_types" in schema["schema"]["required"]
    assert "clock" in schema["schema"]["properties"]["selected_object_types"]["items"]["enum"]


def test_selector_prompt_mentions_candidate_subset_behavior():
    prompt = VLMCaptioner._selector_user_prompt(
        camera_context={"camera_x": 0.0, "camera_y": 0.0, "camera_orientation_deg": 90.0}
    )

    assert "object category pre-selection only" in prompt
    assert "selected_object_types" not in prompt
    assert "Candidate object list:" in prompt
    assert "clock" in prompt
    assert "horizontal FOV is 90.0 degrees" in prompt
