import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Sampler

try:
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - import guard for first-time setup
    raise ImportError(
        "torch-geometric is required for Person 2 graph pipeline. "
        "Install with: pip install torch-geometric"
    ) from exc


RELATION_TO_ID = {
    "adjacent": 0,
    "same_layer_adjacent": 1,
    "wall_window": 2,
}


@dataclass
class GraphRecord:
    data: Data
    floor_plan_id: str
    source_json: str
    num_nodes: int


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _collect_category_maps(json_paths: Sequence[Path], skipped_files: List[str]) -> Dict[str, Dict[str, int]]:
    layer_vocab = {"UNK": 0}
    geo_vocab = {"other": 0}
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped_files.append(path.name)
            continue
        for node in payload.get("nodes", []):
            layer_name = str(node.get("layer", "UNK"))
            if layer_name not in layer_vocab:
                layer_vocab[layer_name] = len(layer_vocab)
            geo_name = str(node.get("geometry_type", "other"))
            if geo_name not in geo_vocab:
                geo_vocab[geo_name] = len(geo_vocab)
    return {"layer_vocab": layer_vocab, "geo_vocab": geo_vocab}


def _one_hot(index: int, width: int) -> List[float]:
    vec = [0.0] * width
    if 0 <= index < width:
        vec[index] = 1.0
    return vec


def contract_json_to_pyg(path: Path, category_maps: Dict[str, Dict[str, int]]) -> GraphRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])

    id_to_idx = {node["id"]: idx for idx, node in enumerate(nodes)}
    layer_vocab = category_maps["layer_vocab"]
    geo_vocab = category_maps["geo_vocab"]

    x_rows: List[List[float]] = []
    node_semantic_id: List[int] = []
    for node in nodes:
        feats = node.get("features", {})
        center = feats.get("center", [0.0, 0.0])
        bbox = feats.get("bbox", [0.0, 0.0, 0.0, 0.0])
        geo_type_oh = feats.get("geometry_type_onehot")
        if not isinstance(geo_type_oh, list):
            geo_type_oh = []

        layer_name = str(node.get("layer", "UNK"))
        layer_id = layer_vocab.get(layer_name, 0)
        geo_name = str(node.get("geometry_type", "other"))
        geo_id = geo_vocab.get(geo_name, 0)
        instance_flag = 1.0 if int(node.get("instance_id", -1)) != -1 else 0.0
        semantic_id = int(node.get("semantic_id", -1))

        numeric_features = [
            _safe_float(feats.get("length", 0.0)),
            _safe_float(center[0] if len(center) > 0 else 0.0),
            _safe_float(center[1] if len(center) > 1 else 0.0),
            _safe_float(bbox[0] if len(bbox) > 0 else 0.0),
            _safe_float(bbox[1] if len(bbox) > 1 else 0.0),
            _safe_float(bbox[2] if len(bbox) > 2 else 0.0),
            _safe_float(bbox[3] if len(bbox) > 3 else 0.0),
            instance_flag,
            float(semantic_id),
        ]
        row = numeric_features + [float(v) for v in geo_type_oh]
        row += _one_hot(layer_id, width=len(layer_vocab))
        row += _one_hot(geo_id, width=len(geo_vocab))
        x_rows.append(row)
        node_semantic_id.append(semantic_id)

    # Keep feature matrix valid for empty/degenerate examples.
    if not x_rows:
        x_rows = [[0.0] * (9 + 7 + len(layer_vocab) + len(geo_vocab))]

    edge_index_rows = [[], []]
    edge_attr_rows = []
    for edge in edges:
        src = id_to_idx.get(edge.get("source"))
        dst = id_to_idx.get(edge.get("target"))
        if src is None or dst is None:
            continue
        relation_name = str(edge.get("relation", "adjacent"))
        relation_id = RELATION_TO_ID.get(relation_name, RELATION_TO_ID["adjacent"])
        distance = _safe_float(edge.get("distance", 0.0))

        # Add bidirectional edges for message passing stability.
        edge_index_rows[0].append(src)
        edge_index_rows[1].append(dst)
        edge_attr_rows.append([distance, float(relation_id)])
        edge_index_rows[0].append(dst)
        edge_index_rows[1].append(src)
        edge_attr_rows.append([distance, float(relation_id)])

    if not edge_attr_rows:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_rows, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32)

    x = torch.tensor(x_rows, dtype=torch.float32)
    semantic_ids = torch.tensor(node_semantic_id if node_semantic_id else [-1], dtype=torch.long)
    floor_plan_id = payload.get("metadata", {}).get("filename", path.stem.replace("_contract", ""))
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        semantic_ids=semantic_ids,
        floor_plan_id=floor_plan_id,
        source_json=str(path),
    )
    return GraphRecord(
        data=data,
        floor_plan_id=floor_plan_id,
        source_json=str(path),
        num_nodes=int(x.shape[0]),
    )


def build_cache(
    train_dir: str,
    cache_path: str,
    stats_path: str,
) -> None:
    train_root = Path(train_dir)
    json_paths = sorted(train_root.glob("*_contract.json"))
    if not json_paths:
        raise FileNotFoundError(f"No *_contract.json files found under {train_dir}")

    skipped_files: List[str] = []
    maps = _collect_category_maps(json_paths, skipped_files)
    records: List[GraphRecord] = []
    for path in json_paths:
        if path.name in skipped_files:
            continue
        try:
            records.append(contract_json_to_pyg(path, maps))
        except json.JSONDecodeError:
            skipped_files.append(path.name)
            continue

    if not records:
        raise RuntimeError("No valid JSON contracts available after filtering malformed files.")
    cache_payload = {
        "records": records,
        "category_maps": maps,
    }
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_payload, cache_path)

    node_counts = [record.num_nodes for record in records]
    stats = {
        "num_graphs": len(records),
        "node_count_min": min(node_counts),
        "node_count_max": max(node_counts),
        "node_count_median": sorted(node_counts)[len(node_counts) // 2],
        "relation_to_id": RELATION_TO_ID,
        "layer_vocab_size": len(maps["layer_vocab"]),
        "geo_vocab_size": len(maps["geo_vocab"]),
        "feature_dim": int(records[0].data.x.shape[1]),
        "skipped_corrupt_files": skipped_files,
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_cache(cache_path: str):
    payload = torch.load(cache_path, map_location="cpu")
    return payload["records"], payload["category_maps"]


def create_or_load_split(
    records: Sequence[GraphRecord],
    split_path: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    split_file = Path(split_path)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    id_to_idx = {rec.floor_plan_id: idx for idx, rec in enumerate(records)}

    if split_file.exists():
        split_payload = json.loads(split_file.read_text(encoding="utf-8"))
        return {
            "train": [id_to_idx[pid] for pid in split_payload["train"] if pid in id_to_idx],
            "val": [id_to_idx[pid] for pid in split_payload["val"] if pid in id_to_idx],
            "test": [id_to_idx[pid] for pid in split_payload["test"] if pid in id_to_idx],
        }

    plan_ids = [record.floor_plan_id for record in records]
    rng = random.Random(seed)
    rng.shuffle(plan_ids)
    n_total = len(plan_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    split_payload = {
        "seed": seed,
        "train": plan_ids[:n_train],
        "val": plan_ids[n_train:n_train + n_val],
        "test": plan_ids[n_train + n_val:],
    }
    split_file.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    return {
        "train": [id_to_idx[pid] for pid in split_payload["train"]],
        "val": [id_to_idx[pid] for pid in split_payload["val"]],
        "test": [id_to_idx[pid] for pid in split_payload["test"]],
    }


class BucketBatchSampler(Sampler[List[int]]):
    """
    Groups similarly-sized graphs into the same batch to reduce padding/variance.
    """

    def __init__(self, records: Sequence[GraphRecord], indices: Sequence[int], batch_size: int, shuffle: bool):
        self.records = records
        self.indices = list(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterable[List[int]]:
        ordered = sorted(self.indices, key=lambda idx: self.records[idx].num_nodes)
        buckets = [
            ordered[start:start + self.batch_size]
            for start in range(0, len(ordered), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(buckets)
        for batch in buckets:
            yield batch

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

