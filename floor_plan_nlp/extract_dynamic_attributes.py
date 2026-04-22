import argparse
import json
import re
from collections import Counter
from pathlib import Path


INVALID_LABELS = {"", "0", "-1"}
ADJACENCY_IGNORE_LABELS = {"wall", "window", "dimension", "room_text"}

# Blocks raw CAD layer/dimension strings that BERT cannot understand.
# Does NOT rename or transform valid labels — that would break graph alignment.
_CAD_GARBAGE_RE = re.compile(
    r"""
    \$              # dollar signs (CAD layer code separator)
    | \#            # hash (CAD reference marker)
    | \d{4,}\s*x\s  # "18014 x" style dimension strings
    | ^lf           # starts with CAD "lf" prefix
    | ^p\d          # starts with CAD parameter "p0", "p1" etc
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MAX_LABEL_CHARS = 40  # garbage labels are usually very long


def is_garbage_label(label: str) -> bool:
    if len(label) > _MAX_LABEL_CHARS:
        return True
    if _CAD_GARBAGE_RE.search(label):
        return True
    return False


def normalize_label(raw_label):
    if raw_label is None:
        return None
    label = str(raw_label).strip()
    if label in INVALID_LABELS:
        return None
    if is_garbage_label(label):
        return None
    # Return the label as-is. Do NOT rename or sanitize —
    # the graph encoder was trained on these exact label strings.
    return label


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
    skipped_garbage = 0
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
    parser.add_argument("--train-dir", default="../train")
    parser.add_argument("--output-path", default="plan_attributes.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_plan_attributes(train_dir=args.train_dir, output_path=args.output_path)