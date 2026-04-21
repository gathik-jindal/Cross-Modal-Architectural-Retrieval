import argparse
import json
import re
from collections import Counter
from pathlib import Path


INVALID_LABELS = {"", "0", "-1"}
ADJACENCY_IGNORE_LABELS = {"wall", "window", "dimension", "room_text"}

_CAD_GARBAGE_RE = re.compile(
    r"""
    \$               # dollar signs used in CAD layer names
    | \#             # hash references
    | \b\d{4,}x\b   # patterns like 18014x
    | \blf\$         # explicit CAD lf$ prefix
    | \bp\$          # explicit CAD p$ prefix
    | ^\d+x\w        # starts like 123xword
    """,
    re.VERBOSE | re.IGNORECASE,
)


def sanitize_semantic_label(label):
    text = str(label).strip().lower()

    # Keep only semantic tail after CAD namespace separators.
    if "$" in text:
        text = text.split("$")[-1].strip()

    # Remove chained one-letter discipline prefixes such as a-, p-, m-.
    text = re.sub(r"^(?:[a-z]-)+", "", text)

    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        return None
    if _CAD_GARBAGE_RE.search(text):
        return None

    return text


def normalize_label(raw_label):
    if raw_label is None:
        return None
    label = str(raw_label).strip()
    if label in INVALID_LABELS:
        return None
    cleaned = sanitize_semantic_label(label)
    if cleaned in INVALID_LABELS:
        return None
    return cleaned


def resolve_floor_plan_id(contract_path, data):
    metadata = data.get("metadata", {})
    metadata_filename = metadata.get("filename")
    if isinstance(metadata_filename, str) and metadata_filename.strip():
        return Path(metadata_filename).stem
    return contract_path.name.replace("_contract.json", "")


def extract_inventory(nodes):
    counts = Counter()
    for node in nodes:
        label = normalize_label(node.get("semantic_label"))
        if label is None:
            continue
        counts[label] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def build_id_to_label(nodes):
    mapping = {}
    for node in nodes:
        node_id = node.get("id")
        label = normalize_label(node.get("semantic_label"))
        if not node_id or label is None:
            continue
        mapping[str(node_id)] = label
    return mapping


def extract_adjacencies(edges, id_to_label):
    unique_pairs = set()

    for edge in edges:
        source_id = edge.get("source")
        target_id = edge.get("target")
        if not source_id or not target_id:
            continue

        source_label = id_to_label.get(str(source_id))
        target_label = id_to_label.get(str(target_id))
        if source_label is None or target_label is None:
            continue
        if source_label == target_label:
            continue
        if source_label in ADJACENCY_IGNORE_LABELS or target_label in ADJACENCY_IGNORE_LABELS:
            continue

        unique_pairs.add(tuple(sorted((source_label, target_label))))

    relationships = []
    for first, second in sorted(unique_pairs):
        relationships.append(f"{first} is adjacent to the {second}")
    return relationships


def density_bucket(total_nodes):
    if total_nodes < 300:
        return "compact"
    if total_nodes >= 700:
        return "large"
    return "moderate"


def extract_attributes_for_file(contract_path):
    data = json.loads(contract_path.read_text(encoding="utf-8"))

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    inventory = extract_inventory(nodes)
    id_to_label = build_id_to_label(nodes)
    adjacencies = extract_adjacencies(edges, id_to_label)

    return {
        "floor_plan_id": resolve_floor_plan_id(contract_path, data),
        "source_json": str(contract_path),
        "inventory": inventory,
        "adjacencies": adjacencies,
        "density": density_bucket(len(nodes)),
    }


def build_plan_attributes(train_dir, output_path):
    train_path = Path(train_dir)
    contract_paths = sorted(train_path.glob("*_contract.json"))

    print(f"Scanning {train_path} for contract files...")
    print(f"Found {len(contract_paths)} files")

    results = []
    for idx, contract_path in enumerate(contract_paths, start=1):
        try:
            record = extract_attributes_for_file(contract_path)
            results.append(record)
        except Exception as exc:
            print(f"[WARN] Skipping {contract_path}: {exc}")
            continue

        if idx % 100 == 0 or idx == len(contract_paths):
            print(f"Processed {idx}/{len(contract_paths)} files")

    output = Path(output_path)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved {len(results)} records to {output}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract deterministic dynamic attributes from floor plan contract JSON files."
    )
    parser.add_argument(
        "--train-dir",
        default="../train",
        help="Directory containing *_contract.json files",
    )
    parser.add_argument(
        "--output-path",
        default="plan_attributes.json",
        help="Path to write extracted plan attributes JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_plan_attributes(train_dir=args.train_dir, output_path=args.output_path)
