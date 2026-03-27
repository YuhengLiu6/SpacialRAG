from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


HOUSEHOLD_OBJECT_TYPES: Tuple[str, ...] = (
    "door",
    "window",
    "blinds",
    "mirror",
    "clock",
    "stairs",
    "railing",
    "art",
    "picture frame",
    "plant",
    "vase",
    "candle",
    "chair",
    "stool",
    "bench",
    "couch",
    "ottoman",
    "bed",
    "crib",
    "nightstand",
    "dresser",
    "table",
    "coffee table",
    "side table",
    "dining table",
    "desk",
    "cabinet",
    "wardrobe",
    "shelf",
    "bookcase",
    "tv",
    "monitor",
    "speaker",
    "lamp",
    "floor lamp",
    "table lamp",
    "ceiling light",
    "fan",
    "fireplace",
    "refrigerator",
    "microwave",
    "oven",
    "stove",
    "dishwasher",
    "sink",
    "faucet",
    "toilet",
    "shower",
    "bathtub",
    "washing machine",
    "dryer",
    "towel",
    "soap dispenser",
    "rug",
    "curtain",
    "pillow",
    "blanket",
    "basket",
    "bin",
    "trash can",
    "book",
    "laptop",
    "keyboard",
    "mouse",
    "phone",
    "remote",
    "toy",
    "pet item",
    "food bowl",
    "bottle",
    "cup",
    "plate",
    "bowl",
    "utensil",
    "kettle",
    "toaster",
    "blender",
    "coffee maker",
    "vacuum",
    "suitcase",
    "shoes",
    "clothes",
    "hanger",
    "bag",
    "wall switch",
    "outlet",
    "unknown",
    "other",
)

HOUSEHOLD_OBJECT_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "alarm clock": "clock",
    "arm chair": "chair",
    "armchair": "chair",
    "bar stool": "stool",
    "barstool": "stool",
    "basket bin": "basket",
    "bath tub": "bathtub",
    "books": "book",
    "bookshelf": "bookcase",
    "ceiling fan": "fan",
    "ceiling fan light": "ceiling light",
    "ceiling lamp": "ceiling light",
    "cell phone": "phone",
    "coffee table": "coffee table",
    "counter": "table",
    "countertop": "table",
    "couch": "couch",
    "couches": "couch",
    "cupboard": "cabinet",
    "curtains": "curtain",
    "decor": "art",
    "decorative clock": "clock",
    "desk chair": "chair",
    "dining chair": "chair",
    "dining table": "dining table",
    "dish washer": "dishwasher",
    "dishwasher": "dishwasher",
    "door frame": "door",
    "doors": "door",
    "end table": "side table",
    "fan light": "ceiling light",
    "floor lamp": "floor lamp",
    "footstool": "ottoman",
    "framed art": "art",
    "framed picture": "picture frame",
    "fridge": "refrigerator",
    "garbage can": "trash can",
    "hanging lamp": "lamp",
    "kitchen island": "table",
    "light": "lamp",
    "light fixture": "ceiling light",
    "microwave oven": "microwave",
    "office chair": "chair",
    "ottomans": "ottoman",
    "painting": "art",
    "pendant light": "lamp",
    "pendant lights": "lamp",
    "pet bowl": "food bowl",
    "phone charger": "phone",
    "picture": "picture frame",
    "pictures": "picture frame",
    "photo": "picture frame",
    "photo frame": "picture frame",
    "portrait": "art",
    "power outlet": "outlet",
    "recliner": "chair",
    "remote control": "remote",
    "roman clock": "clock",
    "sconce": "lamp",
    "settee": "couch",
    "shelves": "shelf",
    "shoe": "shoes",
    "sofa": "couch",
    "stairs railing": "railing",
    "stair railing": "railing",
    "switch": "wall switch",
    "table lamp": "table lamp",
    "television": "tv",
    "tv stand": "cabinet",
    "wall art": "art",
    "wall clock": "clock",
    "wall outlet": "outlet",
    "wall picture": "picture frame",
    "wall switch": "wall switch",
    "window blinds": "blinds",
    "windows": "window",
    "wine bottle": "bottle",
    "wood stool": "stool",
}

_CANONICAL_SET = set(HOUSEHOLD_OBJECT_TYPES)

COMMON_PRELIST_OBJECT_TYPES: Tuple[str, ...] = tuple(
    label for label in HOUSEHOLD_OBJECT_TYPES if label not in {"unknown", "other"}
)


def normalize_taxonomy_token(value: str) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def canonicalize_household_object_label(value: str, default: str = "other") -> str:
    token = normalize_taxonomy_token(value)
    if not token:
        return default
    token = HOUSEHOLD_OBJECT_ALIAS_TO_CANONICAL.get(token, token)
    if token in _CANONICAL_SET:
        return token
    if token.endswith("s") and token[:-1] in _CANONICAL_SET:
        return token[:-1]
    if token.endswith("es") and token[:-2] in _CANONICAL_SET:
        return token[:-2]
    return default


def household_label_enum_values(include_unknown: bool = True, include_other: bool = True) -> Tuple[str, ...]:
    out: List[str] = []
    for label in HOUSEHOLD_OBJECT_TYPES:
        if label == "unknown" and not include_unknown:
            continue
        if label == "other" and not include_other:
            continue
        out.append(label)
    return tuple(out)


def selector_candidate_list_text(labels: Sequence[str] | None = None) -> str:
    values = list(labels or COMMON_PRELIST_OBJECT_TYPES)
    return ", ".join(str(label).strip() for label in values if str(label).strip())


def normalize_selector_subset(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in values:
        canonical = canonicalize_household_object_label(str(item or ""), default="")
        if not canonical or canonical in {"unknown", "other"}:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out
