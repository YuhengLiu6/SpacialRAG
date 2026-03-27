import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from spatial_rag.config import FOV, OBJECT_SURROUNDING_MAX
from spatial_rag.household_taxonomy import (
    COMMON_PRELIST_OBJECT_TYPES,
    canonicalize_household_object_label,
    household_label_enum_values,
    normalize_selector_subset,
    selector_candidate_list_text,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class VLMCaptioner:
    """Generate image summaries with OpenAI VLM and optional on-disk caching."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        use_cache: bool = True,
        cache_dir: str = "spatial_db/vlm_cache",
        object_use_cache: Optional[bool] = None,
        object_cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        default_caption: str = "A view of the room.",
        log_exceptions: bool = True,
    ):
        self.model_name = model_name
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir)
        self.object_use_cache = self.use_cache if object_use_cache is None else bool(object_use_cache)
        if object_cache_dir is None:
            self.object_cache_dir = self.cache_dir.parent / "vlm_object_cache"
        else:
            self.object_cache_dir = Path(object_cache_dir)
        self.default_caption = default_caption
        self.log_exceptions = bool(log_exceptions)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.object_use_cache:
            self.object_cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        if OpenAI and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def _md5_file(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _cache_path(self, image_path: str) -> Path:
        return self.cache_dir / f"{self._md5_file(image_path)}.txt"

    @staticmethod
    def _sanitize_cache_component(value: str) -> str:
        token = str(value or "").strip().lower()
        token = re.sub(r"[^a-z0-9._-]+", "_", token)
        token = re.sub(r"_+", "_", token).strip("._-")
        return token or "unknown"

    def _cache_group_token(self, image_path: str) -> str:
        path = Path(image_path)
        stem = self._sanitize_cache_component(path.stem)
        stem_parts = stem.split("_")
        if stem.startswith("pose_") and len(stem_parts) >= 2:
            return "_".join(stem_parts[:2])
        if path.parent.name == "objects" and path.parent.parent.name:
            return self._sanitize_cache_component(path.parent.parent.name)
        if path.parent.name and path.parent.name not in {"images", "objects"}:
            return self._sanitize_cache_component(path.parent.name)
        return "misc"

    def _structured_object_cache_path(
        self,
        *,
        image_path: str,
        cache_kind: str,
        prompt_version: str,
        cache_key: str,
        suffix: str,
    ) -> Path:
        file_stem = self._sanitize_cache_component(Path(image_path).stem)
        model_token = self._sanitize_cache_component(self.model_name)
        prompt_token = self._sanitize_cache_component(prompt_version)
        group_token = self._cache_group_token(image_path)
        file_name = f"{file_stem}__{cache_key}{suffix}"
        return self.object_cache_dir / cache_kind / model_token / prompt_token / group_token / file_name

    @staticmethod
    def _promote_legacy_cache_file(legacy_path: Path, target_path: Path) -> Path:
        if target_path.exists() or not legacy_path.exists():
            return target_path if target_path.exists() else legacy_path
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")
            return target_path
        except Exception:
            return legacy_path

    def _resolve_cache_path(self, target_path: Path, legacy_path: Path) -> Path:
        if target_path.exists():
            return target_path
        if legacy_path.exists():
            return self._promote_legacy_cache_file(legacy_path, target_path)
        return target_path

    def _legacy_object_cache_path(
        self,
        image_path: str,
        prompt_variant: str = "standard",
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        image_hash = self._md5_file(image_path)
        camera_token = self._camera_context_cache_token(camera_context)
        cache_key_src = (
            f"{image_hash}|{self.model_name}|"
            f"{self._object_prompt_version(prompt_variant=prompt_variant)}|{camera_token}"
        )
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()
        return self.object_cache_dir / f"{cache_key}.json"

    def _object_cache_path(
        self,
        image_path: str,
        prompt_variant: str = "standard",
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        prompt_version = self._object_prompt_version(prompt_variant=prompt_variant)
        image_hash = self._md5_file(image_path)
        camera_token = self._camera_context_cache_token(camera_context)
        cache_key_src = (
            f"{image_hash}|{self.model_name}|"
            f"{prompt_version}|{camera_token}"
        )
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()
        return self._structured_object_cache_path(
            image_path=image_path,
            cache_kind="scene_objects",
            prompt_version=prompt_version,
            cache_key=cache_key,
            suffix=".objects.json",
        )

    def _resolve_object_cache_path(
        self,
        image_path: str,
        prompt_variant: str = "standard",
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        target_path = self._object_cache_path(
            image_path,
            prompt_variant=prompt_variant,
            camera_context=camera_context,
        )
        legacy_path = self._legacy_object_cache_path(
            image_path,
            prompt_variant=prompt_variant,
            camera_context=camera_context,
        )
        return self._resolve_cache_path(target_path, legacy_path)

    def _legacy_object_crop_cache_path(self, image_path: str) -> Path:
        image_hash = self._md5_file(image_path)
        cache_key_src = f"{image_hash}|{self.model_name}|{self._object_crop_prompt_version()}"
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()
        return self.object_cache_dir / f"{cache_key}.json"

    def _object_crop_cache_path(self, image_path: str) -> Path:
        image_hash = self._md5_file(image_path)
        prompt_version = self._object_crop_prompt_version()
        cache_key_src = f"{image_hash}|{self.model_name}|{prompt_version}"
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()
        return self._structured_object_cache_path(
            image_path=image_path,
            cache_kind="object_crops",
            prompt_version=prompt_version,
            cache_key=cache_key,
            suffix=".crop.json",
        )

    def _resolve_object_crop_cache_path(self, image_path: str) -> Path:
        target_path = self._object_crop_cache_path(image_path)
        legacy_path = self._legacy_object_crop_cache_path(image_path)
        return self._resolve_cache_path(target_path, legacy_path)

    def _legacy_selector_cache_path(
        self,
        image_path: str,
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        image_hash = self._md5_file(image_path)
        camera_token = self._camera_context_cache_token(camera_context)
        cache_key_src = f"{image_hash}|{self.model_name}|{self._selector_prompt_version()}|{camera_token}"
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()
        return self.object_cache_dir / f"{cache_key}.selector.json"

    def _selector_cache_path(
        self,
        image_path: str,
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        image_hash = self._md5_file(image_path)
        camera_token = self._camera_context_cache_token(camera_context)
        prompt_version = self._selector_prompt_version()
        cache_key_src = f"{image_hash}|{self.model_name}|{prompt_version}|{camera_token}"
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()
        return self._structured_object_cache_path(
            image_path=image_path,
            cache_kind="selector",
            prompt_version=prompt_version,
            cache_key=cache_key,
            suffix=".selector.json",
        )

    def _resolve_selector_cache_path(
        self,
        image_path: str,
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        target_path = self._selector_cache_path(image_path, camera_context=camera_context)
        legacy_path = self._legacy_selector_cache_path(image_path, camera_context=camera_context)
        return self._resolve_cache_path(target_path, legacy_path)

    @staticmethod
    def _object_prompt_version(prompt_variant: str = "standard") -> str:
        variant = str(prompt_variant or "standard").strip().lower()
        if variant == "standard":
            return "home_prompt_surrounding_anchor_height_hierarchy_v3"
        if variant == "angle_split":
            return "home_prompt_angle_split_surrounding_anchor_height_hierarchy_v3"
        raise ValueError(f"Unsupported object prompt_variant: {prompt_variant}")

    @staticmethod
    def _selector_prompt_version() -> str:
        return "household_selector_scene_summary_v1"

    @staticmethod
    def _camera_context_cache_token(camera_context: Optional[Dict[str, Any]] = None) -> str:
        if not camera_context:
            return "camera:none"
        normalized = {
            "camera_x": float(camera_context.get("camera_x", 0.0)),
            "camera_z": float(camera_context.get("camera_z", 0.0)),
            "camera_orientation_deg": float(camera_context.get("camera_orientation_deg", 0.0)),
        }
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _camera_context_prompt_block(camera_context: Optional[Dict[str, Any]] = None) -> str:
        angle_convention = (
            f"Image geometry convention: horizontal FOV is {float(FOV):.1f} degrees. "
            "Straight ahead is 0 degrees. "
            "Objects near the left image edge are about -45 degrees. "
            "Objects near the right image edge are about +45 degrees. "
            "Negative bearing means left of image center. Positive bearing means right of image center. "
        )
        if not camera_context:
            return (
                "Camera global pose is unavailable for this request. "
                f"{angle_convention}"
                "Still estimate each object's distance_from_camera_m and relative_bearing_deg from the image alone. "
            )
        camera_x = float(camera_context.get("camera_x", 0.0))
        camera_z = float(camera_context.get("camera_z", 0.0))
        camera_orientation_deg = float(camera_context.get("camera_orientation_deg", 0.0))
        return (
            "Current camera global pose: "
            f"x={camera_x:.3f}, z={camera_z:.3f}, orientation_deg={camera_orientation_deg:.1f}. "
            "Use this pose only to reason about spatial consistency. "
            "Do not return absolute global coordinates. "
            "Return relative geometry only, and let the downstream program compute global coordinates. "
            f"{angle_convention}"
        )

    @staticmethod
    def _object_crop_prompt_version() -> str:
        return "object_crop_descriptor_builder_aligned_v5"

    @staticmethod
    def _object_crop_system_prompt() -> str:
        return (
            "You are a strict vision parser for object crops. "
            "Return JSON only, matching the schema exactly. "
            "Describe only the main visible object in the crop. "
            "Use the detector-provided class as the object category you must describe. "
            "Match the style of object descriptions used in a spatial database builder: "
            "the short description should read like a concise object instance description, "
            "and the long description should read like a detailed open-form object description. "
            "Do not return generic placeholders when any visible cue is available. "
            "If the object is partial, edge-cropped, occluded, blurred, dark, or tiny, explicitly say so in the descriptions."
        )

    @staticmethod
    def _object_crop_user_prompt(yolo_label_clean: str, yolo_conf_text: str) -> str:
        return (
            f'A detector has identified the object in this crop as "{yolo_label_clean or "unknown"}" '
            f"(confidence: {yolo_conf_text}). "
            "Treat this detected class as the object category to describe. "
            "Describe this specific object instance visible in the cropped image in the same style as the database builder's "
            "object fields: short_description should correspond to a short precise object description, and long_description "
            "should correspond to a detailed long-form open description. "
            "Ignore the wider room and focus on the object itself. "
            "Return a compact label for that detected category, a short description useful for retrieval, "
            "a richer long description, a list of notable visual attributes, and an approximate distance "
            "from the camera in meters when you can infer it. Mention the approximate distance directly "
            "in the short or long description when possible. "
            "Requirements: "
            "short_description must be 3 to 8 words and must include at least one visible cue beyond the class name, "
            "such as color, material, shape, state, position in crop, or partial visibility. "
            "Prefer noun phrases like 'dark wooden chair edge crop' or 'gold-framed wall picture' over generic labels. "
            "Do not use only the bare class name for short_description unless absolutely nothing except the class is visible. "
            "long_description must be a concrete sentence or phrase with all visible cues you can infer, and should not be "
            "\"unknown\" unless the crop is too poor to identify any visual property at all. "
            "Include visible properties that also help matching against database object text, such as color, material, texture, "
            "shape, condition, approximate size cues, and whether the object appears partial or cut off. "
            "If the crop is partial or clipped by the image border, explicitly include words like partial, cropped, edge, "
            "or cut off. If you are uncertain, describe what is visible rather than refusing. "
            "attributes should list concise visible cues, not generic words. "
            "Output JSON only."
        )

    @staticmethod
    def _object_system_prompt(prompt_variant: str = "standard") -> str:
        variant = str(prompt_variant or "standard").strip().lower()
        if variant not in {"standard", "angle_split"}:
            raise ValueError(f"Unsupported object prompt_variant: {prompt_variant}")
        return (
            "You are a strict vision parser for spatial retrieval. "
            "Return JSON only, matching the schema exactly. "
            "No markdown, no explanations, no extra keys."
        )

    @staticmethod
    def _object_user_prompt(
        max_objects: int,
        prompt_variant: str = "standard",
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        variant = str(prompt_variant or "standard").strip().lower()
        if variant not in {"standard", "angle_split"}:
            raise ValueError(f"Unsupported object prompt_variant: {prompt_variant}")

        angle_split_instructions = ""
        if variant == "angle_split":
            angle_split_instructions = (
                "For relative_position_laterality, divide the image horizontally into exactly three discrete sectors: "
                "left, center, and right. Assign every object to exactly one of these sectors based on where most of the "
                "object appears in the image. Do not use laterality vaguely or comparatively; treat it as a strict bucket. "
                "Use left for objects mainly in the left third, center for the middle third, and right for the right third. "
            )

        camera_context_block = VLMCaptioner._camera_context_prompt_block(camera_context)
        object_enum_text = " | ".join(household_label_enum_values(include_unknown=True, include_other=True))

        return (
            "I am going to show you a photograph taken from a particular node location and orientation on a building. "
            "Your job is to describe the overall image, and separately extract concrete visible objects from this image "
            "to provide detailed descriptions and spatial characteristics for downstream retrieval and deduplication. "
            "Do not output wall feature or floor pattern as standalone objects. Put those into scene_attributes or into a "
            "nearby concrete object's attributes when relevant.\n\n"
            f"{camera_context_block}\n\n"
            f"{angle_split_instructions}"
            "For every object, estimate its approximate distance from the camera in meters whenever possible and fill "
            "distance_from_camera_m with that estimate. Also estimate relative_bearing_deg in the range [-90, 90], where "
            "0 means the object is straight ahead, negative means left of image center, and positive means right of image center. "
            "Also estimate relative_height_from_camera_m, the object's vertical offset relative to the camera center in meters: "
            "negative means lower than the camera, positive means higher than the camera, and 0 means roughly level with the camera. "
            "Keep relative_position_distance broadly consistent with the meter estimate. "
            f"For each primary object, list up to {int(OBJECT_SURROUNDING_MAX)} visible surrounding objects in surrounding_context, sorted by increasing "
            "distance_from_primary_m. Each surrounding item must include label, attributes, distance_from_primary_m, "
            "distance_from_camera_m, relative_height_from_camera_m, relative_bearing_deg, and relation_to_primary. "
            "If a quantity cannot be inferred, return null instead of guessing. "
            "Do not return absolute global coordinates.\n\n"
            "When reporting information, follow the JSON format below to provide information on the overall image and for "
            "each concrete object with as much detail as possible. Make each value as specific as possible:\n"
            "{\n"
            '  "view_type": "<living room | bedroom | kitchen | bathroom | dining room | hallway | entryway | balcony | laundry room | staircase | study | utility room | unknown | other>",\n'
            '  "room_function": "<resting | cooking | dining | bathing | working | storage | circulation | mixed | unknown>",\n'
            '  "style_hint": "<modern | minimalist | traditional | rustic | industrial | scandinavian | eclectic | unknown | other>",\n'
            '  "clutter_level": "<low | medium | high | unknown>",\n'
            '  "scene_attributes": ["<background or scene-level cue>", "..."],\n'
            '  "visual_feature": [\n'
            "    {\n"
            f'      "type": "<{object_enum_text}>",\n'
            '      "description": "<short precise description for this object instance>",\n'
            '      "attributes": ["<visible attribute>", "..."],\n'
            '      "relative_position_laterality": "<left | center | right>",\n'
            '      "relative_position_distance": "<near | middle | far>",\n'
            '      "relative_position_verticality": "<low | middle | high>",\n'
            '      "distance_from_camera_m": "<number or null>",\n'
            '      "relative_height_from_camera_m": "<number or null>",\n'
            '      "relative_bearing_deg": "<number or null>",\n'
            '      "support_relation": "<on | under | inside | hanging_on | attached_to | freestanding | unknown>",\n'
            '      "any_text": "<legible text or empty string>",\n'
            '      "long_form_open_description": "<detailed description: color, material, texture, shape, condition, exact placement>",\n'
            '      "location_relative_to_other_objects": "<spatial relation to nearby objects>",\n'
            '      "surrounding_context": [\n'
            "        {\n"
            '          "label": "<surrounding object label>",\n'
            '          "attributes": ["<visible attribute>", "..."],\n'
            '          "distance_from_primary_m": "<number or null>",\n'
            '          "distance_from_camera_m": "<number or null>",\n'
            '          "relative_height_from_camera_m": "<number or null>",\n'
            '          "relative_bearing_deg": "<number or null>",\n'
            '          "relation_to_primary": "<short spatial relation>"\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ],\n"
            '  "floor_pattern": "<wood | laminate | tile | carpet | rug | stone | concrete | unknown | other>",\n'
            '  "lighting_ceiling": "<recessed lights | chandelier | pendant lights | ceiling lamp | ceiling fan light | natural light source | mixed lighting | unknown | other>",\n'
            '  "wall_color": "<predominant wall color>",\n'
            '  "additional_notes": "<extra location cues useful for matching>",\n'
            '  "image_summary": "<comprehensive scene summary with spatial layout and notable details>"\n'
            "}\n"
            "Fill in every JSON field above as completely as possible. If something is entirely out of frame or unidentifiable, "
            "you can use 'unknown' or null where allowed. "
            f"Return at most {max_objects} items in visual_feature. "
            "Output JSON only."
        )

    @staticmethod
    def _object_response_schema(max_objects: int) -> dict:
        object_enum_values = list(household_label_enum_values(include_unknown=True, include_other=True))
        return {
            "name": "scene_objects",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "view_type",
                    "room_function",
                    "style_hint",
                    "clutter_level",
                    "scene_attributes",
                    "visual_feature",
                    "floor_pattern",
                    "lighting_ceiling",
                    "wall_color",
                    "additional_notes",
                    "image_summary",
                ],
                "properties": {
                    "view_type": {
                        "type": "string",
                        "enum": [
                            "living room",
                            "bedroom",
                            "kitchen",
                            "bathroom",
                            "dining room",
                            "hallway",
                            "entryway",
                            "balcony",
                            "laundry room",
                            "staircase",
                            "study",
                            "utility room",
                            "unknown",
                            "other",
                        ],
                    },
                    "room_function": {
                        "type": "string",
                        "enum": [
                            "resting",
                            "cooking",
                            "dining",
                            "bathing",
                            "working",
                            "storage",
                            "circulation",
                            "mixed",
                            "unknown",
                        ],
                    },
                    "style_hint": {
                        "type": "string",
                        "enum": [
                            "modern",
                            "minimalist",
                            "traditional",
                            "rustic",
                            "industrial",
                            "scandinavian",
                            "eclectic",
                            "unknown",
                            "other",
                        ],
                    },
                    "clutter_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "unknown"],
                    },
                    "scene_attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 24,
                    },
                    "visual_feature": {
                        "type": "array",
                        "maxItems": int(max_objects),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "type",
                                "description",
                                "attributes",
                                "relative_position_laterality",
                                "relative_position_distance",
                                "relative_position_verticality",
                                "distance_from_camera_m",
                                "relative_height_from_camera_m",
                                "relative_bearing_deg",
                                "support_relation",
                                "any_text",
                                "long_form_open_description",
                                "location_relative_to_other_objects",
                                "surrounding_context",
                            ],
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": object_enum_values,
                                },
                                "description": {"type": "string"},
                                "attributes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 12,
                                },
                                "relative_position_laterality": {
                                    "type": "string",
                                    "enum": ["left", "right", "center"],
                                },
                                "relative_position_distance": {
                                    "type": "string",
                                    "enum": ["near", "middle", "far"],
                                },
                                "relative_position_verticality": {
                                    "type": "string",
                                    "enum": ["high", "middle", "low"],
                                },
                                "distance_from_camera_m": {"type": ["number", "null"]},
                                "relative_height_from_camera_m": {"type": ["number", "null"]},
                                "relative_bearing_deg": {"type": ["number", "null"]},
                                "support_relation": {
                                    "type": "string",
                                    "enum": ["on", "under", "inside", "hanging_on", "attached_to", "freestanding", "unknown"],
                                },
                                "any_text": {"type": "string"},
                                "long_form_open_description": {"type": "string"},
                                "location_relative_to_other_objects": {"type": "string"},
                                "surrounding_context": {
                                    "type": "array",
                                    "maxItems": int(OBJECT_SURROUNDING_MAX),
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": [
                                            "label",
                                            "attributes",
                                            "distance_from_primary_m",
                                            "distance_from_camera_m",
                                            "relative_height_from_camera_m",
                                            "relative_bearing_deg",
                                            "relation_to_primary",
                                        ],
                                        "properties": {
                                            "label": {"type": "string"},
                                            "attributes": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "maxItems": 8,
                                            },
                                            "distance_from_primary_m": {"type": ["number", "null"]},
                                            "distance_from_camera_m": {"type": ["number", "null"]},
                                            "relative_height_from_camera_m": {"type": ["number", "null"]},
                                            "relative_bearing_deg": {"type": ["number", "null"]},
                                            "relation_to_primary": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "floor_pattern": {
                        "type": "string",
                        "enum": [
                            "wood",
                            "laminate",
                            "tile",
                            "carpet",
                            "rug",
                            "stone",
                            "concrete",
                            "unknown",
                            "other",
                        ],
                    },
                    "lighting_ceiling": {
                        "type": "string",
                        "enum": [
                            "recessed lights",
                            "chandelier",
                            "pendant lights",
                            "ceiling lamp",
                            "ceiling fan light",
                            "natural light source",
                            "mixed lighting",
                            "unknown",
                            "other",
                        ],
                    },
                    "wall_color": {"type": "string"},
                    "additional_notes": {"type": "string"},
                    "image_summary": {"type": "string"},
                },
            },
        }

    @staticmethod
    def _default_object_json() -> str:
        return json.dumps(
            {
                "view_type": "unknown",
                "room_function": "unknown",
                "style_hint": "unknown",
                "clutter_level": "unknown",
                "scene_attributes": [],
                "visual_feature": [],
                "floor_pattern": "unknown",
                "lighting_ceiling": "unknown",
                "wall_color": "unknown",
                "additional_notes": "",
                "image_summary": "",
            },
            ensure_ascii=True,
        )

    @staticmethod
    def _default_selector_payload() -> Dict[str, Any]:
        return {
            "view_type": "unknown",
            "room_function": "unknown",
            "style_hint": "unknown",
            "clutter_level": "unknown",
            "scene_attributes": [],
            "floor_pattern": "unknown",
            "lighting_ceiling": "unknown",
            "wall_color": "unknown",
            "additional_notes": "",
            "image_summary": "",
            "selected_object_types": [],
        }

    @staticmethod
    def _selector_response_schema() -> dict:
        selector_enum = list(COMMON_PRELIST_OBJECT_TYPES)
        return {
            "name": "scene_selector",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "view_type",
                    "room_function",
                    "style_hint",
                    "clutter_level",
                    "scene_attributes",
                    "floor_pattern",
                    "lighting_ceiling",
                    "wall_color",
                    "additional_notes",
                    "image_summary",
                    "selected_object_types",
                ],
                "properties": {
                    "view_type": {
                        "type": "string",
                        "enum": [
                            "living room",
                            "bedroom",
                            "kitchen",
                            "bathroom",
                            "dining room",
                            "hallway",
                            "entryway",
                            "balcony",
                            "laundry room",
                            "staircase",
                            "study",
                            "utility room",
                            "unknown",
                            "other",
                        ],
                    },
                    "room_function": {
                        "type": "string",
                        "enum": [
                            "resting",
                            "cooking",
                            "dining",
                            "bathing",
                            "working",
                            "storage",
                            "circulation",
                            "mixed",
                            "unknown",
                        ],
                    },
                    "style_hint": {
                        "type": "string",
                        "enum": [
                            "modern",
                            "minimalist",
                            "traditional",
                            "rustic",
                            "industrial",
                            "scandinavian",
                            "eclectic",
                            "unknown",
                            "other",
                        ],
                    },
                    "clutter_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "unknown"],
                    },
                    "scene_attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 24,
                    },
                    "floor_pattern": {
                        "type": "string",
                        "enum": [
                            "wood",
                            "laminate",
                            "tile",
                            "carpet",
                            "rug",
                            "stone",
                            "concrete",
                            "unknown",
                            "other",
                        ],
                    },
                    "lighting_ceiling": {
                        "type": "string",
                        "enum": [
                            "recessed lights",
                            "chandelier",
                            "pendant lights",
                            "ceiling lamp",
                            "ceiling fan light",
                            "natural light source",
                            "mixed lighting",
                            "unknown",
                            "other",
                        ],
                    },
                    "wall_color": {"type": "string"},
                    "additional_notes": {"type": "string"},
                    "image_summary": {"type": "string"},
                    "selected_object_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": selector_enum},
                        "maxItems": min(32, len(selector_enum)),
                    },
                },
            },
        }

    @staticmethod
    def _selector_user_prompt(camera_context: Optional[Dict[str, Any]] = None) -> str:
        camera_context_block = VLMCaptioner._camera_context_prompt_block(camera_context)
        return (
            "I am going to show you a room image. "
            "Your job is to do scene summarization and object category pre-selection only. "
            "Do not enumerate object instances and do not estimate per-object geometry in this step. "
            "Use the candidate object list provided below as a household pre-list, and return only the subset that is clearly visible in the image. "
            "Prefer categories that are visually present as concrete objects, not inferred from context alone. "
            "Include an object type only if it is likely visible enough for a detector to localize. "
            "Exclude categories that are absent, ambiguous, or only suggested by the room type. "
            f"{camera_context_block}"
            f"Candidate object list: {selector_candidate_list_text(COMMON_PRELIST_OBJECT_TYPES)}. "
            "Return JSON only."
        )

    @staticmethod
    def _default_object_crop_description() -> Dict[str, Any]:
        return {
            "label": "unknown",
            "short_description": "unknown",
            "long_description": "unknown",
            "attributes": [],
            "distance_from_camera_m": None,
        }

    @staticmethod
    def _object_crop_response_schema() -> dict:
        return {
            "name": "object_crop_description",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "label",
                    "short_description",
                    "long_description",
                    "attributes",
                    "distance_from_camera_m",
                ],
                "properties": {
                    "label": {"type": "string"},
                    "short_description": {"type": "string"},
                    "long_description": {"type": "string"},
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                    "distance_from_camera_m": {"type": ["number", "null"]},
                },
            },
        }

    @staticmethod
    def _encode_image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _log_exception(self, where: str, image_path: str, exc: Exception) -> None:
        if not self.log_exceptions:
            return
        print(
            f"[VLMCaptioner][{where}] image={image_path} "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    @staticmethod
    def _log(message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[VLMCaptioner][{ts}] {message}", flush=True)

    @staticmethod
    def _serialize_api_response(response: Any) -> Optional[Dict[str, Any]]:
        if response is None:
            return None
        try:
            if hasattr(response, "model_dump"):
                dumped = response.model_dump(mode="json")
                if isinstance(dumped, dict):
                    return dumped
        except Exception:
            pass
        return None

    @staticmethod
    def _response_has_length_finish_reason(raw_api_response: Any) -> bool:
        if not isinstance(raw_api_response, dict):
            return False
        for choice in list(raw_api_response.get("choices") or []):
            if str(choice.get("finish_reason") or "").strip().lower() == "length":
                return True
        return False

    def caption_image(self, image_path: str) -> str:
        """Return a text summary for an image, using cache when enabled."""
        cache_path = self._cache_path(image_path) if self.use_cache else None

        if cache_path and cache_path.exists():
            self._log(f"caption cache_hit image={image_path} cache={cache_path}")
            return cache_path.read_text(encoding="utf-8").strip()

        if self.client is None:
            self._log("caption client_unavailable -> return default caption")
            return self.default_caption

        try:
            self._log(f"caption request_start model={self.model_name} image={image_path}")
            t0 = time.perf_counter()
            encoded_string = self._encode_image_to_base64(image_path)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an image understanding assistant. "
                            "Write a compact factual summary of this scene for spatial retrieval."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Summarize the visible scene in 2-4 sentences, "
                                    "including key objects and rough layout cues."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_string}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                max_completion_tokens=4000,
            )
            caption = (response.choices[0].message.content or "").strip() or self.default_caption
            self._log(
                f"caption request_done chars={len(caption)} "
                f"elapsed_sec={time.perf_counter() - t0:.2f}"
            )

            if cache_path:
                cache_path.write_text(caption, encoding="utf-8")
                self._log(f"caption cache_write cache={cache_path}")

            return caption
        except Exception as exc:
            self._log_exception("caption_image", image_path, exc)
            self._log("caption request_failed -> return default caption")
            return self.default_caption

    def select_object_types_with_meta(
        self,
        image_path: str,
        force_refresh: bool = False,
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cache_path = (
            self._resolve_selector_cache_path(image_path, camera_context=camera_context)
            if self.object_use_cache
            else None
        )
        default_payload = self._default_selector_payload()

        if cache_path and cache_path.exists() and not force_refresh:
            self._log(f"selector cache_hit image={image_path} cache={cache_path}")
            try:
                loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                loaded = None
            if isinstance(loaded, dict):
                payload = loaded.get("payload", loaded)
                if isinstance(payload, dict):
                    normalized_payload = dict(default_payload)
                    normalized_payload.update(payload)
                    normalized_payload["selected_object_types"] = normalize_selector_subset(
                        payload.get("selected_object_types") or []
                    )
                    return {
                        "payload": normalized_payload,
                        "raw_json": json.dumps(normalized_payload, ensure_ascii=True),
                        "raw_api_response": loaded.get("raw_api_response"),
                        "source": "cache",
                    }

        if self.client is None:
            self._log("selector client_unavailable -> return default payload")
            return {
                "payload": dict(default_payload),
                "raw_json": json.dumps(default_payload, ensure_ascii=True),
                "raw_api_response": None,
                "source": "default-no-client",
            }

        try:
            self._log(
                f"selector request_start model={self.model_name} image={image_path} "
                f"force_refresh={bool(force_refresh)} "
                f"camera_context={self._camera_context_cache_token(camera_context)}"
            )
            t0 = time.perf_counter()
            encoded_string = self._encode_image_to_base64(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={
                    "type": "json_schema",
                    "json_schema": self._selector_response_schema(),
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict vision parser for scene summarization and household category selection. "
                            "Return JSON only, matching the schema exactly. "
                            "Use only categories from the provided candidate list."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._selector_user_prompt(camera_context=camera_context)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_string}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                max_completion_tokens=6000,
            )
            raw_json = (response.choices[0].message.content or "").strip() or json.dumps(default_payload, ensure_ascii=True)
            raw_api_response = self._serialize_api_response(response)
            payload = json.loads(raw_json)
            if not isinstance(payload, dict):
                payload = dict(default_payload)
            normalized_payload = dict(default_payload)
            normalized_payload.update(payload)
            normalized_payload["selected_object_types"] = normalize_selector_subset(
                payload.get("selected_object_types") or []
            )
            raw_json = json.dumps(normalized_payload, ensure_ascii=True)
            self._log(
                f"selector request_done chars={len(raw_json)} "
                f"elapsed_sec={time.perf_counter() - t0:.2f}"
            )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_payload = {
                    "payload": normalized_payload,
                    "raw_api_response": raw_api_response,
                    "cached_at_unix_s": int(time.time()),
                    "model": self.model_name,
                }
                cache_path.write_text(json.dumps(cache_payload, ensure_ascii=True), encoding="utf-8")
                self._log(f"selector cache_write cache={cache_path}")
            return {
                "payload": normalized_payload,
                "raw_json": raw_json,
                "raw_api_response": raw_api_response,
                "source": "api",
            }
        except Exception as exc:
            self._log_exception("select_object_types", image_path, exc)
            self._log("selector request_failed -> return default payload")
            return {
                "payload": dict(default_payload),
                "raw_json": json.dumps(default_payload, ensure_ascii=True),
                "raw_api_response": None,
                "source": "default-exception",
            }

    def extract_objects_with_meta(
        self,
        image_path: str,
        max_objects: int = 24,
        force_refresh: bool = False,
        prompt_variant: str = "standard",
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return object JSON plus optional raw API response metadata."""
        selected_prompt_variant = str(prompt_variant or "standard").strip().lower()
        cache_path = (
            self._resolve_object_cache_path(
                image_path,
                prompt_variant=selected_prompt_variant,
                camera_context=camera_context,
            )
            if self.object_use_cache
            else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            self._log(f"object cache_hit image={image_path} cache={cache_path}")
            raw_cache = cache_path.read_text(encoding="utf-8")
            cache_text = raw_cache.strip()
            if not cache_text:
                return {
                    "raw_json": self._default_object_json(),
                    "raw_api_response": None,
                    "source": "cache-empty",
                }
            try:
                loaded = json.loads(cache_text)
                if isinstance(loaded, dict) and "raw_json" in loaded:
                    raw_json = str(loaded.get("raw_json") or "").strip() or self._default_object_json()
                    raw_api_response = loaded.get("raw_api_response")
                    if not isinstance(raw_api_response, dict):
                        raw_api_response = None
                    if self._response_has_length_finish_reason(raw_api_response):
                        self._log(f"object cache_skip_length image={image_path} cache={cache_path}")
                    else:
                        return {
                            "raw_json": raw_json,
                            "raw_api_response": raw_api_response,
                            "source": "cache-envelope",
                        }
            except Exception:
                pass
            legacy_loaded = None
            try:
                legacy_loaded = json.loads(cache_text)
            except Exception:
                legacy_loaded = None
            if self._response_has_length_finish_reason(legacy_loaded):
                self._log(f"object cache_skip_length_legacy image={image_path} cache={cache_path}")
            else:
                return {
                    "raw_json": cache_text,
                    "raw_api_response": None,
                    "source": "cache-legacy",
                }

        default_json = self._default_object_json()
        if self.client is None:
            self._log("object client_unavailable -> return default json")
            return {
                "raw_json": default_json,
                "raw_api_response": None,
                "source": "default-no-client",
            }

        max_objects = max(1, int(max_objects))
        try:
            self._log(
                f"object request_start model={self.model_name} image={image_path} "
                f"max_objects={max_objects} force_refresh={bool(force_refresh)} "
                f"prompt_variant={selected_prompt_variant} "
                f"camera_context={self._camera_context_cache_token(camera_context)}"
            )
            t0 = time.perf_counter()
            encoded_string = self._encode_image_to_base64(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={
                    "type": "json_schema",
                    "json_schema": self._object_response_schema(max_objects=max_objects),
                },
                messages=[
                    {
                        "role": "system",
                        "content": self._object_system_prompt(prompt_variant=selected_prompt_variant),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._object_user_prompt(
                                    max_objects=max_objects,
                                    prompt_variant=selected_prompt_variant,
                                    camera_context=camera_context,
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_string}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                max_completion_tokens=8000,
            )
            raw_json = (response.choices[0].message.content or "").strip()
            if not raw_json:
                raw_json = default_json
            raw_api_response = self._serialize_api_response(response)
            self._log(
                f"object request_done chars={len(raw_json)} "
                f"elapsed_sec={time.perf_counter() - t0:.2f}"
            )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "raw_json": raw_json,
                    "raw_api_response": raw_api_response,
                    "cached_at_unix_s": int(time.time()),
                    "model": self.model_name,
                    "prompt_variant": selected_prompt_variant,
                    "camera_context": camera_context,
                }
                cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
                self._log(f"object cache_write cache={cache_path}")
            return {
                "raw_json": raw_json,
                "raw_api_response": raw_api_response,
                "source": "api",
            }
        except Exception as exc:
            self._log_exception("extract_objects", image_path, exc)
            self._log("object request_failed -> return default json")
            return {
                "raw_json": default_json,
                "raw_api_response": None,
                "source": "default-exception",
            }

    def extract_objects(
        self,
        image_path: str,
        max_objects: int = 24,
        force_refresh: bool = False,
        prompt_variant: str = "standard",
        camera_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compatibility helper returning only raw JSON text."""
        result = self.extract_objects_with_meta(
            image_path=image_path,
            max_objects=max_objects,
            force_refresh=force_refresh,
            prompt_variant=prompt_variant,
            camera_context=camera_context,
        )
        return str(result.get("raw_json") or self._default_object_json())

    def describe_object_crop_with_meta(
        self,
        image_path: str,
        force_refresh: bool = False,
        yolo_label: Optional[str] = None,
        yolo_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Describe the primary object shown in a crop image."""
        cache_path = self._resolve_object_crop_cache_path(image_path) if self.object_use_cache else None
        default_payload = self._default_object_crop_description()
        yolo_label_clean = str(yolo_label or "").strip()
        yolo_conf_text = (
            f"{float(yolo_confidence):.3f}"
            if yolo_confidence is not None
            else "unknown"
        )

        if cache_path and cache_path.exists() and not force_refresh:
            self._log(f"object_crop cache_hit image={image_path} cache={cache_path}")
            raw_cache = cache_path.read_text(encoding="utf-8")
            cache_text = raw_cache.strip()
            if not cache_text:
                return {
                    **default_payload,
                    "raw_response": None,
                    "source": "cache-empty",
                }
            try:
                loaded = json.loads(cache_text)
            except Exception:
                loaded = None

            if isinstance(loaded, dict):
                payload = loaded.get("payload", loaded)
                if isinstance(payload, dict):
                    return {
                        "label": str(payload.get("label") or default_payload["label"]).strip() or "unknown",
                        "short_description": str(
                            payload.get("short_description") or default_payload["short_description"]
                        ).strip()
                        or "unknown",
                        "long_description": str(
                            payload.get("long_description") or default_payload["long_description"]
                        ).strip()
                        or "unknown",
                        "attributes": [
                            str(v).strip()
                            for v in list(payload.get("attributes") or [])
                            if str(v).strip()
                        ],
                        "distance_from_camera_m": payload.get("distance_from_camera_m"),
                        "raw_response": loaded.get("raw_response"),
                        "source": "cache",
                    }

            return {
                **default_payload,
                "raw_response": None,
                "source": "cache-invalid",
            }

        if self.client is None:
            self._log("object_crop client_unavailable -> return default payload")
            return {
                **default_payload,
                "raw_response": None,
                "source": "default-no-client",
            }

        try:
            self._log(
                f"object_crop request_start model={self.model_name} image={image_path} "
                f"force_refresh={bool(force_refresh)} yolo_label={yolo_label_clean or 'none'}"
            )
            t0 = time.perf_counter()
            encoded_string = self._encode_image_to_base64(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={
                    "type": "json_schema",
                    "json_schema": self._object_crop_response_schema(),
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict vision parser for object crops. "
                            "Return JSON only, matching the schema exactly. "
                            "Describe only the main visible object in the crop. "
                            "Use the detector-provided class as the object category you must describe. "
                            "Match the style of object descriptions used in a spatial database builder: "
                            "the short description should read like a concise object instance description, "
                            "and the long description should read like a detailed open-form object description. "
                            "Do not return generic placeholders when any visible cue is available. "
                            "If the object is partial, edge-cropped, occluded, blurred, dark, or tiny, explicitly say so in the descriptions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f'A detector has identified the object in this crop as "{yolo_label_clean or "unknown"}" '
                                    f"(confidence: {yolo_conf_text}). "
                                    "Treat this detected class as the object category to describe. "
                                    "Describe this specific object instance visible in the cropped image in the same style as the database builder's "
                                    "object fields: short_description should correspond to a short precise object description, and long_description "
                                    "should correspond to a detailed long-form open description. "
                                    "Ignore the wider room and focus on the object itself. "
                                    "Return a compact label for that detected category, a short description useful for retrieval, "
                                    "a richer long description, a list of notable visual attributes, and an approximate distance "
                                    "from the camera in meters when you can infer it. Mention the approximate distance directly "
                                    "in the short or long description when possible. "
                                    "Requirements: "
                                    "short_description must be 3 to 8 words and must include at least one visible cue beyond the class name, "
                                    "such as color, material, shape, state, position in crop, or partial visibility. "
                                    "Prefer noun phrases like 'dark wooden chair edge crop' or 'gold-framed wall picture' over generic labels. "
                                    "Do not use only the bare class name for short_description unless absolutely nothing except the class is visible. "
                                    "long_description must be a concrete sentence or phrase with all visible cues you can infer, and should not be "
                                    "\"unknown\" unless the crop is too poor to identify any visual property at all. "
                                    "Include visible properties that also help matching against database object text, such as color, material, texture, "
                                    "shape, condition, approximate size cues, and whether the object appears partial or cut off. "
                                    "If the crop is partial or clipped by the image border, explicitly include words like partial, cropped, edge, "
                                    "or cut off. If you are uncertain, describe what is visible rather than refusing. "
                                    "attributes should list concise visible cues, not generic words. "
                                    "Output JSON only."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_string}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                max_completion_tokens=4000,
            )
            raw_content = (response.choices[0].message.content or "").strip()
            payload = json.loads(raw_content) if raw_content else dict(default_payload)
            if not isinstance(payload, dict):
                payload = dict(default_payload)

            result_label = str(payload.get("label") or default_payload["label"]).strip() or "unknown"
            result_short = str(
                payload.get("short_description") or default_payload["short_description"]
            ).strip() or "unknown"
            result_long = str(
                payload.get("long_description") or default_payload["long_description"]
            ).strip()
            if (not result_label or result_label.lower() == "unknown") and yolo_label_clean:
                result_label = yolo_label_clean
            if (not result_short or result_short.lower() == "unknown") and yolo_label_clean:
                result_short = yolo_label_clean
            if not result_long:
                result_long = result_short

            result = {
                "label": result_label,
                "short_description": result_short,
                "long_description": result_long,
                "attributes": [
                    str(v).strip()
                    for v in list(payload.get("attributes") or [])
                    if str(v).strip()
                ],
                "distance_from_camera_m": payload.get("distance_from_camera_m"),
                "raw_response": self._serialize_api_response(response),
                "source": "api",
                "yolo_label_hint": yolo_label_clean or None,
                "yolo_confidence_hint": None if yolo_confidence is None else float(yolo_confidence),
            }
            self._log(
                f"object_crop request_done chars={len(raw_content)} "
                f"elapsed_sec={time.perf_counter() - t0:.2f}"
            )

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_payload = {
                    "payload": {
                        "label": result["label"],
                        "short_description": result["short_description"],
                        "long_description": result["long_description"],
                        "attributes": result["attributes"],
                        "distance_from_camera_m": result["distance_from_camera_m"],
                        "yolo_label_hint": result["yolo_label_hint"],
                        "yolo_confidence_hint": result["yolo_confidence_hint"],
                    },
                    "raw_response": result["raw_response"],
                    "cached_at_unix_s": int(time.time()),
                    "model": self.model_name,
                }
                cache_path.write_text(json.dumps(cache_payload, ensure_ascii=True), encoding="utf-8")
                self._log(f"object_crop cache_write cache={cache_path}")

            return result
        except Exception as exc:
            self._log_exception("describe_object_crop_with_meta", image_path, exc)
            self._log("object_crop request_failed -> return default payload")
            return {
                **default_payload,
                "raw_response": None,
                "source": "default-exception",
                "yolo_label_hint": yolo_label_clean or None,
                "yolo_confidence_hint": None if yolo_confidence is None else float(yolo_confidence),
            }
