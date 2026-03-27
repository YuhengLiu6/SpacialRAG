from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from spatial_rag.household_taxonomy import household_label_enum_values


ViewType = Literal[
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
]
RoomFunction = Literal["resting", "cooking", "dining", "bathing", "working", "storage", "circulation", "mixed", "unknown"]
StyleHint = Literal["modern", "minimalist", "traditional", "rustic", "industrial", "scandinavian", "eclectic", "unknown", "other"]
ClutterLevel = Literal["low", "medium", "high", "unknown"]
VisualFeatureType = Literal.__getitem__(household_label_enum_values(include_unknown=True, include_other=True))
Laterality = Literal["left", "right", "center"]
DistanceBin = Literal["near", "middle", "far"]
Verticality = Literal["high", "middle", "low"]
SupportRelation = Literal["on", "under", "inside", "hanging_on", "attached_to", "freestanding", "unknown"]
FloorPattern = Literal["wood", "laminate", "tile", "carpet", "rug", "stone", "concrete", "unknown", "other"]
LightingCeiling = Literal["recessed lights", "chandelier", "pendant lights", "ceiling lamp", "ceiling fan light", "natural light source", "mixed lighting", "unknown", "other"]


class SurroundingObject(BaseModel):
    label: str
    attributes: List[str] = Field(default_factory=list)
    distance_from_primary_m: Optional[float] = None
    distance_from_camera_m: Optional[float] = None
    relative_height_from_camera_m: Optional[float] = None
    relative_bearing_deg: Optional[float] = None
    estimated_global_x: Optional[float] = None
    estimated_global_y: Optional[float] = None
    estimated_global_z: Optional[float] = None
    relation_to_primary: str = ""


class VisualFeature(BaseModel):
    feature_id: str
    type: VisualFeatureType
    description: str
    attributes: List[str] = Field(default_factory=list)
    relative_position_laterality: Laterality
    relative_position_distance: DistanceBin
    relative_position_verticality: Verticality
    distance_from_camera_m: Optional[float] = None
    relative_height_from_camera_m: Optional[float] = None
    relative_bearing_deg: Optional[float] = None
    estimated_global_x: Optional[float] = None
    estimated_global_y: Optional[float] = None
    estimated_global_z: Optional[float] = None
    support_relation: SupportRelation = "unknown"
    any_text: str = ""
    long_form_open_description: str = ""
    location_relative_to_other_objects: str = ""
    surrounding_context: List[SurroundingObject] = Field(default_factory=list)


class SceneObjects(BaseModel):
    image_id: Optional[str] = None
    view_type: ViewType = "unknown"
    room_function: RoomFunction = "unknown"
    style_hint: StyleHint = "unknown"
    clutter_level: ClutterLevel = "unknown"
    scene_attributes: List[str] = Field(default_factory=list)
    visual_feature: List[VisualFeature] = Field(default_factory=list)
    floor_pattern: FloorPattern = "unknown"
    lighting_ceiling: LightingCeiling = "unknown"
    wall_color: str = "unknown"
    additional_notes: str = ""
    image_summary: str = ""
    raw_vlm_text: Optional[str] = None
