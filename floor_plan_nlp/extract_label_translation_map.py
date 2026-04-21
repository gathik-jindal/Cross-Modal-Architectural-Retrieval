import argparse
import json
import re
from pathlib import Path


# Covers CJK Unified Ideographs and common extension blocks used in labels.
CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def contains_cjk(text):
    return bool(CJK_PATTERN.search(text))


def collect_labels_from_file(contract_path):
    data = json.loads(contract_path.read_text(encoding="utf-8"))
    found = set()

    metadata = data.get("metadata", {})
    layer_counts = metadata.get("layer_counts", {})
    if isinstance(layer_counts, dict):
        for key in layer_counts.keys():
            key_str = str(key).strip()
            if key_str and contains_cjk(key_str):
                found.add(key_str)

    for node in data.get("nodes", []):
        for field_name in ("semantic_label", "layer"):
            value = node.get(field_name)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str and contains_cjk(value_str):
                found.add(value_str)

    return found


def build_translation_map(input_dir, output_path):
    root = Path(input_dir)
    contract_paths = sorted(root.glob("*_contract.json"))

    print(f"Scanning {root} for contract files...")
    print(f"Found {len(contract_paths)} files")

    labels = set()
    for idx, contract_path in enumerate(contract_paths, start=1):
        try:
            labels.update(collect_labels_from_file(contract_path))
        except Exception as exc:
            print(f"[WARN] Skipping {contract_path}: {exc}")
            continue

        if idx % 200 == 0 or idx == len(contract_paths):
            print(f"Processed {idx}/{len(contract_paths)} files")

    output_lines = [
        "# Label translation map",
        "# Format: source_label=english_translation",
        "# Fill the right side for each label. Keep one mapping per line.",
        "",
    ]

    for label in sorted(labels):
        output_lines.append(f"{label}=")

    output = Path(output_path)
    output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(f"Saved {len(labels)} labels to {output}")
    return labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CJK labels from contract JSONs into a translation map template."
    )
    parser.add_argument(
        "--input-dir",
        default="../test",
        help="Directory containing *_contract.json files",
    )
    parser.add_argument(
        "--output-path",
        default="label_translation_map.txt",
        help="Output text file for label translation mappings",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_translation_map(input_dir=args.input_dir, output_path=args.output_path)
