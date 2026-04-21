import argparse
import json
import re
from pathlib import Path


INVENTORY_TEXT_EXCLUDE = {
    "wall",
    "window",
    "opening_symbol",
    "dimension",
    "single_door",
    "axis",
    "floor_decoration",
    "hatch",
    "installed_furniture",
    "column",
    "blind_window",
    "dote",
    "curtwall",
}

IRREGULAR_PLURALS = {
    "axis": "axes",
}

# Remap raw FloorPlanCAD labels to human-readable equivalents.
# "bed" → "bedroom" so the query says "160 bedrooms" not "160 beds".
LABEL_REMAP = {
    "bed": "bedroom",
}

CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_CAD_GARBAGE_RE = re.compile(
    r"""
    \$               # dollar CAD separators
    | \#             # hash CAD refs
    | \b\d{4,}\s*x\s+ # 18014 x patterns
    | \blf\$         # lf$ CAD prefix
    | \bp\$          # p$ CAD prefix
    | ^\d+\s*x\s+\w  # starts with dimension x word
    """,
    re.VERBOSE | re.IGNORECASE,
)


def contains_cjk(text):
    return bool(CJK_PATTERN.search(str(text)))


def sanitize_label_text(text):
    cleaned = str(text).strip().lower()
    if "$" in cleaned:
        cleaned = cleaned.split("$")[-1].strip()

    cleaned = re.sub(r"^(?:[a-z]-)+", "", cleaned)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if _CAD_GARBAGE_RE.search(cleaned):
        return ""
    return cleaned


def load_label_translation_map(map_path):
    if not map_path:
        return {}

    path = Path(map_path)
    if not path.exists():
        print(f"[WARN] Translation map not found: {path}. Continuing without it.")
        return {}

    mapping = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        source, target = line.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            continue
        mapping[source] = target

    print(f"Loaded {len(mapping)} label translations from {path}")
    return mapping


def special_count_phrase(label, count):
    if label == "furniture":
        suffix = "item" if count == 1 else "items"
        return f"{count} furniture {suffix}"

    if label == "installed furniture":
        suffix = "item" if count == 1 else "items"
        return f"{count} installed furniture {suffix}"

    if label.startswith("text "):
        base = label.replace("text ", "", 1)
        suffix = "annotation" if count == 1 else "annotations"
        return f"{count} {base} {suffix}"

    return None


def humanize_label(label, translation_map):
    raw = str(label)
    translated = translation_map.get(raw, raw)
    label = sanitize_label_text(translated)
    if not label:
        return ""
    return LABEL_REMAP.get(label, label)


def pluralize(noun, count):
    if count == 1:
        return noun

    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]

    lower = noun.lower()
    if lower.endswith("y") and len(noun) > 1 and noun[-2].lower() not in "aeiou":
        return noun[:-1] + "ies"
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def join_english(parts):
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def format_inventory(inventory, translation_map):
    filtered = []

    for label, count in sorted(inventory.items(), key=lambda item: item[0]):
        if "text" in str(label).lower():
            continue
        if label in INVENTORY_TEXT_EXCLUDE:
            continue
        if not isinstance(count, int) or count <= 0:
            continue

        readable = humanize_label(label, translation_map)
        if not readable:
            continue
        # Skip unresolved CJK labels to keep BERT-facing text fully ASCII/English.
        if contains_cjk(readable):
            continue

        custom = special_count_phrase(readable, count)
        if custom is not None:
            filtered.append((count, str(label), custom))
            continue

        # Use real, uncapped count — e.g. "160 bedrooms" not "4 bedrooms"
        filtered.append((count, str(label), f"{count} {pluralize(readable, count)}"))

    # Keep only the top 4 items by frequency to avoid run-on sentences.
    top_items = sorted(filtered, key=lambda item: (-item[0], item[1]))[:4]
    phrases = [item[2] for item in top_items]

    if not phrases:
        return "key interior elements"
    return join_english(phrases)


def clean_adjacency_text(adjacency, translation_map):
    return humanize_label(adjacency, translation_map)


def build_query_text(density, inventory, adjacencies, translation_map):
    inventory_text = format_inventory(inventory, translation_map)
    query = f"This is a {density} layout featuring {inventory_text}."

    cleaned_adjacencies = []
    for item in adjacencies[:2]:
        translated = clean_adjacency_text(item, translation_map)
        if not translated:
            continue
        if contains_cjk(translated):
            continue
        cleaned_adjacencies.append(translated)

    if len(cleaned_adjacencies) == 1:
        query += f" Spatially, the {cleaned_adjacencies[0]}."
    elif len(cleaned_adjacencies) == 2:
        query += (
            f" Spatially, the {cleaned_adjacencies[0]}, "
            f"and the {cleaned_adjacencies[1]}."
        )

    return query


def scale_bucket(inventory):
    """
    Bucket the real bedroom count into 4 scale tiers.
    Used by BalancedCategoryBatchSampler in alignment_trainer.py to ensure
    each training batch contains examples from every scale tier.

    0 = commercial / no bedrooms
    1 = small residential   (1–50 bedrooms)
    2 = medium complex      (51–150 bedrooms)
    3 = large complex       (151+ bedrooms)
    """
    bed_count = inventory.get("bed", 0)
    if not isinstance(bed_count, int) or bed_count <= 0:
        return 0
    if bed_count <= 50:
        return 1
    if bed_count <= 150:
        return 2
    return 3


def generate_pairs(attributes_path, output_path, label_map_path):
    attributes = json.loads(Path(attributes_path).read_text(encoding="utf-8"))
    print(f"Loaded {len(attributes)} attribute records from {attributes_path}")
    translation_map = load_label_translation_map(label_map_path)

    pairs = []
    bucket_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for idx, record in enumerate(attributes, start=1):
        inventory = record.get("inventory", {})
        adjacencies = record.get("adjacencies", [])
        density = record.get("density", "moderate")
        source_json = record.get("source_json", "")

        query = build_query_text(
            density=density,
            inventory=inventory,
            adjacencies=adjacencies,
            translation_map=translation_map,
        )
        bucket = scale_bucket(inventory)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        pair = {
            "query": query,
            "graph_path": source_json,
            "floor_plan_id": record.get("floor_plan_id"),
            # bedroom_count now holds a scale bucket (0-3), not a capped raw count.
            # Real bed count is already embedded in the query text itself.
            "bedroom_count": bucket,
            "template_type": "dynamic",
        }
        pairs.append(pair)

        if idx % 200 == 0 or idx == len(attributes):
            print(f"Generated {idx}/{len(attributes)} queries")

    Path(output_path).write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Saved {len(pairs)} pairs to {output_path}")
    print(f"Scale bucket distribution: {bucket_counts}")
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deterministic dynamic text queries from plan attributes."
    )
    parser.add_argument(
        "--attributes-path",
        default="plan_attributes.json",
        help="Path to plan_attributes.json",
    )
    parser.add_argument(
        "--output-path",
        default="pairs.json",
        help="Path to write pairs JSON",
    )
    parser.add_argument(
        "--label-map-path",
        default="label_translation_map.txt",
        help="Path to text file with source_label=english_translation entries",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_pairs(
        attributes_path=args.attributes_path,
        output_path=args.output_path,
        label_map_path=args.label_map_path,
    )