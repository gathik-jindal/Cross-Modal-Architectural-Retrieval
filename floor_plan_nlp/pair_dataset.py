"""
pair_dataset.py
PairedDataset: wraps pairs.json and yields (query_string, PyG_Data_object) items.
The collate_fn handles heterogeneous batching (strings + graphs).
"""
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Sampler
import random
from collections import defaultdict

try:
    from torch_geometric.data import Data, Batch
except ImportError:
    raise ImportError("pip install torch-geometric")

from graph_dataset import contract_json_to_pyg, load_cache


class PairedDataset(Dataset):
    """
    Each item: {"query": str, "graph": PyG Data object, "bedroom_count": int}

    The graph is loaded lazily from the contract JSON path stored in pairs.json.
    If a graph_cache.pt is available from Person 2, loading is faster.
    """

    def __init__(
        self,
        pairs_path: str = "pairs.json",
        cache_path: str = "artifacts/cache/graph_cache.pt",
    ):
        self.pairs_path = Path(pairs_path)
        self.base_dir = self.pairs_path.parent
        self.pairs = json.loads(self.pairs_path.read_text(encoding="utf-8"))

        # Try to load Person 2's pre-built graph cache for speed
        self.graph_cache: Dict[str, Data] = {}
        self.category_maps = None

        if Path(cache_path).exists():
            print(f"Loading graph cache from {cache_path}...")
            records, category_maps = load_cache(cache_path)
            self.category_maps = category_maps
            for rec in records:
                key = rec.source_json
                if rec.data is not None:
                    self.graph_cache[key] = rec.data
                elif rec.graph_path:
                    self.graph_cache[key] = rec.graph_path
            # Also index by basename for robust matching
            self._basename_cache = {
                Path(k).name: v for k, v in self.graph_cache.items()
            }
            print(f"  Cached {len(self.graph_cache)} graphs")
        else:
            self._basename_cache = {}
            print("No graph cache found. Graphs will be parsed from JSON on the fly.")

    def __len__(self) -> int:
        return len(self.pairs)

    def _resolve_graph_path(self, graph_path: str) -> Path:
        """
        Resolve graph_path which may be relative.
        Handles paths like 'train/0000-0002_contract.json'.
        """
        p = Path(graph_path)
        # 1. Absolute or already accessible from CWD
        if p.exists():
            return p
        # 2. Relative to pairs.json directory
        rel = (self.base_dir / p).resolve()
        if rel.exists():
            return rel
        # 3. Relative to CWD explicitly
        cwd_rel = (Path.cwd() / p).resolve()
        if cwd_rel.exists():
            return cwd_rel
        # Return best guess (will raise FileNotFoundError downstream with a clear path)
        return p

    def _load_graph(self, graph_path: str) -> Data:
        """Load a graph from the cache or parse it from JSON."""
        # 1. Try exact key match
        cached = self.graph_cache.get(graph_path)

        # 2. Try basename match (handles abs vs rel path mismatch between cache and pairs.json)
        if cached is None:
            basename = Path(graph_path).name
            cached = self._basename_cache.get(basename)

        if cached is not None:
            if isinstance(cached, (str, Path)):
                cached_path = Path(cached)
                if cached_path.exists():
                    return torch.load(cached_path, map_location="cpu", weights_only=False)
            else:
                return cached

        # 3. Parse from JSON on the fly
        resolved = self._resolve_graph_path(graph_path)
        if self.category_maps is not None:
            record = contract_json_to_pyg(resolved, self.category_maps)
        else:
            record = contract_json_to_pyg(
                resolved,
                {"layer_vocab": {"UNK": 0}, "geo_vocab": {"other": 0}},
            )
        return record.data

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        query = pair["query"]
        graph_path = pair["graph_path"]
        bedroom_count = pair.get("bedroom_count", -1)

        graph = self._load_graph(graph_path)
        return {"query": query, "graph": graph, "bedroom_count": bedroom_count}


def paired_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate: strings stay as list, graphs get PyG batched.
    """
    queries = [item["query"] for item in batch]
    graphs = Batch.from_data_list([item["graph"] for item in batch])
    bedroom_counts = torch.tensor(
        [item["bedroom_count"] for item in batch], dtype=torch.long
    )
    return {
        "queries": queries,
        "graphs": graphs,
        "bedroom_counts": bedroom_counts,
    }


class BalancedCategoryBatchSampler(Sampler):
    """
    Produces batches where each bedroom category (0-4) is equally represented.
    Forces the model to learn hard within-category distinctions, not just
    easy commercial vs residential separation.
    """

    def __init__(self, pairs: list, batch_size: int, drop_last: bool = True):
        self.pairs = pairs
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.by_category = defaultdict(list)
        for idx, pair in enumerate(pairs):
            self.by_category[pair.get("bedroom_count", -1)].append(idx)

        self.categories = sorted(self.by_category.keys())
        self.n_categories = len(self.categories)
        self.per_cat = max(1, batch_size // self.n_categories)
        self.effective_batch = self.per_cat * self.n_categories

    def __iter__(self):
        shuffled = {cat: random.sample(idxs, len(idxs))
                    for cat, idxs in self.by_category.items()}
        min_batches = min(len(idxs) // self.per_cat
                          for idxs in shuffled.values())
        for b in range(min_batches):
            batch = []
            for cat in self.categories:
                start = b * self.per_cat
                batch.extend(shuffled[cat][start:start + self.per_cat])
            random.shuffle(batch)
            yield batch

    def __len__(self):
        min_cat_size = min(len(v) for v in self.by_category.values())
        return min_cat_size // self.per_cat
