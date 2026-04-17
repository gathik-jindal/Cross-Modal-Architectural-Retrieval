# constants.py

# ---------------------------------------------------------------------------
# Semantic ID → human-readable label map (from FloorPlanCAD annotations)
# ---------------------------------------------------------------------------
SEMANTIC_ID_TO_LABEL = {
    0:  "background",
    1:  "wall",
    2:  "railing",
    3:  "window",
    4:  "sliding_door",
    5:  "double_door",
    6:  "single_door",
    7:  "opening_symbol",
    8:  "stair",
    9:  "blind_window",
    10: "bay_window",
    11: "column",
    12: "beam",
    13: "gas_stove",
    14: "bed",
    15: "sofa",
    16: "chair",
    17: "table",
    18: "wardrobe",
    19: "washing_machine",
    20: "refrigerator",
    21: "TV_cabinet",
    22: "half_height_cabinet",
    23: "bath_tub",
    24: "toilet",
    25: "sink",
    26: "squat_toilet",
    27: "urinal",
    28: "elevator",
    29: "escalator",
    30: "parking",
}

# Layer name → rough semantic label (fallback when no semantic-id attribute)
LAYER_LABEL_MAP = {
    "WALL":        "wall",
    "WINDOW":      "window",
    "COLUMN":      "column",
    "DOOR":        "single_door",
    "STAIR":       "stair",
    "AXIS":        "axis",           # annotation / structural axis
    "DOTE":        "axis",
    "DIM_LEAD":    "dimension",
    "PUB_DIM":     "dimension",
    "PUB_HATCH":   "hatch",
    "PUB_TEXT":    "room_text",
    "WINDOW_TEXT": "window_text",
    "地饰":         "floor_decoration",
    "家具":         "furniture",
    "装施家具":     "installed_furniture",
}

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
