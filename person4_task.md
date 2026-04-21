# Full Context Prompt for Person 4 (The Bridge Builder)
# Project: Bridging Text and Geometry — Learning Representations of Architectural Floor Plans

---

## WHO YOU ARE AND WHAT THIS PROJECT IS

You are implementing **Person 4** of a 4-person NLP project. The project goal is to build a cross-modal retrieval system that allows a user to type a natural language description of a floor plan (e.g. "a 2 bedroom apartment with an open kitchen and attached bathrooms") and retrieve the most matching real architectural floor plan drawings from the FloorPlanCAD dataset.

The core challenge is a **modality mismatch**: text is linguistic and abstract, floor plans are geometric and structural. Your job is to bridge them.

The system works by converting both modalities into vectors of 256 numbers (embeddings) in a shared semantic space, so that text about a 2-bedroom flat and an actual 2-bedroom floor plan drawing end up near each other in that space. Similarity is then measured by cosine distance (dot product of L2-normalised vectors).

---

## THE DATASET

The project uses the **FloorPlanCAD dataset** — over 15,000 real-world CAD floor plan drawings from residential buildings, schools, hospitals, shopping malls and office buildings. The dataset is stored as SVG (vector graphics) and also as parsed JSON contract files.

Key facts about FloorPlanCAD:
- 35 annotated object categories: single door, double door, window, bay window, stair, gas stove, refrigerator, washing machine, sofa, bed, chair, table, wardrobe, sink, bath, toilet, elevator, escalator, wall, parking, and more
- Graphs are built where **nodes = graphic entities** (line segments, arcs, curves) and **edges = adjacency relationships** between entities
- The `bed` semantic class is heavily represented (~1480 × 10^4 instances) — the most common furniture class, which is crucial for bedroom-count estimation
- Files are stored under `train/` as `*_contract.json` files, one per floor plan

---

## WHAT THE OTHER THREE PERSONS HAVE DONE

### Person 1 — SVG Parsing and Feature Extraction (COMPLETE)

Person 1 parsed the raw SVG files from FloorPlanCAD and converted them into structured JSON contract files. Each contract file has this schema:

```json
{
  "metadata": { "filename": "A001" },
  "nodes": [
    {
      "id": 0,
      "semantic_id": 8,
      "layer": "FURN",
      "geometry_type": "line",
      "geometry_type_onehot": [1, 0, 0, 0, 0, 0, 0],
      "instance_id": 42,
      "features": {
        "length": 1.23,
        "center": [0.45, 0.67],
        "bbox": [0.3, 0.5, 0.6, 0.8]
      }
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "relation": "adjacent",
      "distance": 0.05
    }
  ]
}
```

Edge `relation` values are: `"adjacent"`, `"same_layer_adjacent"`, `"wall_window"`.

`semantic_id` maps to FloorPlanCAD's 35 classes (0–34). Key mappings relevant to you:
- semantic_id 8 = `bed` (the primary signal for bedroom count estimation)
- semantic_id 28 = `wall`
- semantic_id 29 = `parking`

These contract files live at `train/*_contract.json`.

---

### Person 2 — Graph Construction and Geometric Encoding (COMPLETE)

Person 2 built a Graph Neural Network (GNN) that takes a floor plan's contract JSON, builds a PyTorch Geometric (PyG) graph from it, and encodes it into a single 256-dimensional L2-normalised vector.

**Architecture — `GraphPlanEncoder` (in `graph_model.py`):**
- 3 layers of SAGEConv (GraphSAGE convolution)
- LayerNorm after each layer
- Global mean pooling + global max pooling concatenated (→ 2×hidden_dim)
- MLP readout: Linear(2×hidden_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → 256)
- Final L2 normalisation — output is a unit vector

Node features fed into the GNN:
- `features.length` (1 float)
- `features.center` (2 floats)
- `features.bbox` (4 floats)
- `instance_flag` (1 float, 1.0 if instance_id != -1)
- `semantic_id` (1 raw float)
- `geometry_type_onehot` (variable length, from JSON)
- `layer` one-hot encoded (width = number of unique layers seen across dataset)
- `geometry_type` one-hot encoded (width = number of unique geometry types)

Total node feature dimension (`in_dim`) is variable but fixed once the vocabulary is built from the training set. It is stored in the checkpoint.

Edge features: `[distance, relation_id]` where relation_id ∈ {0, 1, 2} for the three relation types.

**Person 2 was trained as a pretext task**: semantic symbol classification across 37 classes (0 = unknown, 1–36 = semantic_id + 1). This forced the encoder to learn geometrically meaningful representations before your alignment training.

**What Person 2 gives you — three files:**

```
artifacts/handoff/
    embeddings.npy          # numpy array, shape [N, 256], float32, L2-normalised
    embedding_index.json    # list of {"row": int, "floor_plan_id": str, "source_json": str}
    embeddings.pt           # same as .npy but PyTorch tensor

artifacts/runs/graph_baseline/
    best_checkpoint.pt      # contains: model_state_dict, config dict, in_dim int
```

`embedding_index.json` example:
```json
[
  {"row": 0, "floor_plan_id": "A001", "source_json": "train/A001_contract.json"},
  {"row": 1, "floor_plan_id": "A002", "source_json": "train/A002_contract.json"},
  ...
]
```

Person 2 also gave you `retrieval_index.py` which contains the complete `PlanRetrievalIndex` class. **Do not rewrite this.** It already handles loading `embeddings.npy` and running cosine search via `index.search(query_vec, top_k=10)`.

**Also available from Person 2:**
- `graph_dataset.py` — contains `load_cache()`, `GraphRecord` dataclass, `contract_json_to_pyg()` function, `BucketBatchSampler`
- `graph_model.py` — contains `GraphPlanEncoder` class
- `train_graph_encoder.py` — Person 2's training script (for reference; you don't run this)
- `export_plan_embeddings.py` — Person 2's export script (you WILL re-run this after your alignment training)

---

### Person 3 — Natural Language Pipeline (COMPLETE)

Person 3 built a text encoder that converts a natural language floor plan description into a 256-dimensional L2-normalised vector.

**Architecture — `TextEncoder` (in `text_encoder.py`):**
- BERT tokenizer (`bert-base-uncased`)
- BERT model (frozen — weights do NOT update during your training initially)
- Masked mean pooling over token dimension (not just [CLS] token)
- Projection MLP: Linear(768 → 512) → ReLU → Dropout(0.1) → Linear(512 → 256)
- Final L2 normalisation — output is a unit vector

Usage:
```python
from text_encoder import TextEncoder, preprocess_query
encoder = TextEncoder(output_dim=256, freeze_bert=True)
vec = encoder(["a 2 bedroom apartment with open kitchen"])  # → tensor [1, 256]
```

`preprocess_query(text)` handles Indian real-estate abbreviations: BHK → bedroom hall kitchen, sqft → square feet, 2br → 2 bedroom, etc. Always call this before encoding user queries at inference time.

**What Person 3 gives you — two files:**

```
text_queries.json     # 2000 synthetic floor plan descriptions with metadata
text_encoder.py       # TextEncoder class, QueryDataset, preprocess_query
```

**`text_queries.json` structure** (2000 entries, each looks like):
```json
{
  "floor_plan_id": "fp_00042",
  "query": "This is a 2 bedroom and 1 bathroom residential unit. The carpet area is approximately 750 square feet. It includes 2 balcony spaces for outdoor usage. The overall layout follows a compact design. The living room is centrally positioned and connects to key spaces. Bedrooms are arranged to maintain privacy and ventilation. The kitchen is efficiently planned for daily use.",
  "template_type": "residential_basic",
  "layout_summary": {
    "bedrooms": 2,
    "bathrooms": 1,
    "has_parking": false
  },
  "area_details": {
    "carpet_area_sqft": 750,
    "built_up_area_sqft": 900
  },
  "balcony": { "present": true, "count": 2, "area_each": 45 },
  "room_details": [...],
  "constraints": {
    "natural_light": "high",
    "ventilation": "good",
    "privacy_level": "medium"
  },
  "spatial_characteristics": {
    "layout_type": "compact",
    "circulation": "efficient"
  }
}
```

**CRITICAL WARNING:** The `floor_plan_id` values in `text_queries.json` (e.g. `fp_00042`) are **completely synthetic and fictional**. They do NOT correspond to any real floor plan file in `train/`. They are just internal labels Person 3 used to organise the data. There is no `fp_00042_contract.json` file. You must NOT attempt to look up these IDs in Person 2's index.

The 6 template types in the dataset are:
- `residential_basic` — standard description with bedroom/bathroom/area info
- `spatial_relationship` — adds phrases like "the bedroom is adjacent to the bathroom"
- `feature_focused` — highlights a specific item like "the design highlights a wardrobe"
- `commercial` — prefixed with "A commercial layout"
- `multi_constraint` — adds circulation and opening types (stair, elevator, sliding door)
- `negative_constraint` — adds "the design avoids including a [item]"

Also available from Person 3:
- `evaluate_encoder.py` — Person 3's sanity checks (text-to-text retrieval proxy, template sensitivity)
- `generate_queries.py` — the script that generated `text_queries.json` (for reference)

---

## YOUR ROLE — PERSON 4: ALIGNMENT, RETRIEVAL AND SYSTEM INTEGRATION

You are the bridge. You receive frozen, untrained-with-respect-to-each-other encoders from Person 2 and Person 3, and you train them jointly so they produce compatible embeddings. Then you build the inference pipeline.

Your four deliverables are:
1. `pair_builder.py` — build (text, geometry) training pairs
2. `pair_dataset.py` + `alignment_trainer.py` — contrastive training
3. `evaluate_retrieval.py` — Recall@k metrics
4. `inference.py` — end-to-end user-facing pipeline

---

## THE PAIRING PROBLEM — HOW TO BUILD YOUR TRAINING DATASET

Since Person 3's `floor_plan_id`s are fake, you cannot directly pair text queries to floor plans by ID. You must use **weak supervision via bedroom count matching**.

**Strategy:**

From the text side, every entry in `text_queries.json` has `layout_summary.bedrooms` ∈ {1, 2, 3, 4}.

From the geometry side, every `*_contract.json` in `train/` contains nodes with `semantic_id` values. In FloorPlanCAD, `semantic_id = 8` corresponds to the `bed` class. Counting nodes with `semantic_id == 8` gives an estimate of how many beds (and therefore bedrooms) a floor plan has.

**Bedroom estimation logic:**
```python
def estimate_bedrooms(contract_json_path):
    data = json.load(open(contract_json_path))
    bed_nodes = [n for n in data["nodes"] if n.get("semantic_id") == 8]
    bed_count = len(bed_nodes)
    if bed_count == 0:
        return None   # commercial / parking / unknown — skip or treat as 0
    elif bed_count <= 1:
        return 1
    elif bed_count <= 3:
        return 2
    elif bed_count <= 5:
        return 3
    else:
        return 4
```

Note: bed nodes in FloorPlanCAD represent individual bed symbols, not rooms. A 2-bedroom flat typically has 2–3 bed symbols (one double bed per room, sometimes a single). Adjust thresholds based on what you observe in the data.

**Pairing logic:**
```python
# Group text queries by bedroom count
text_by_beds = defaultdict(list)
for item in text_queries:
    n = item["layout_summary"]["bedrooms"]
    text_by_beds[n].append(item["query"])

# Group floor plan paths by estimated bedroom count
geo_by_beds = defaultdict(list)
for path in glob("train/*_contract.json"):
    n = estimate_bedrooms(path)
    if n is not None:
        geo_by_beds[n].append(path)

# A valid pair: any text with N bedrooms + any floor plan with N bedrooms
# Sample pairs with balanced bedroom distribution
pairs = []
for n_beds in [1, 2, 3, 4]:
    texts = text_by_beds[n_beds]
    geos  = geo_by_beds[n_beds]
    # sample min(len(texts), len(geos), 500) pairs per category
    for t, g in zip(random.sample(texts, k), random.sample(geos, k)):
        pairs.append({"query": t, "graph_path": g, "bedroom_count": n_beds})
```

This gives you roughly 1500–2000 weakly supervised pairs. You do NOT need perfect ground truth labels. The model learns from the fact that the 31 negatives in each batch of 32 are genuinely different (they have different bedroom counts and layout types).

**Also include commercial floor plans as a 5th category:**
Commercial floor plans in FloorPlanCAD have very few or zero bed nodes. Text queries with `template_type == "commercial"` map to these. Pair them separately:
```python
commercial_texts = [q for q in text_queries if q["template_type"] == "commercial"]
commercial_geos  = [p for p in all_paths if estimate_bedrooms(p) is None or estimate_bedrooms(p) == 0]
```

---

## HOW CONTRASTIVE TRAINING WORKS — THE CORE CONCEPT

You will train using **InfoNCE loss** (also called NT-Xent, used in CLIP and SimCLR).

**The idea in plain terms:**

You take a batch of B pairs. Each pair is `(text_i, floor_plan_i)` where both are "about" the same type of space (e.g. a 2-bedroom unit).

You encode all B texts → `T` matrix of shape `[B, 256]`
You encode all B floor plans → `G` matrix of shape `[B, 256]`

Both are L2-normalised, so their dot product equals cosine similarity.

You compute a `[B, B]` similarity matrix:
```python
sim = T @ G.T / temperature   # shape [B, B]
# sim[i][j] = similarity of text i with floor plan j
# Correct matches are on the diagonal: (0,0), (1,1), (2,2), ...
```

The labels for the loss are `[0, 1, 2, ..., B-1]` — the diagonal.

```python
labels = torch.arange(B)
loss = (
    F.cross_entropy(sim, labels) +        # each text should rank its paired floor plan highest
    F.cross_entropy(sim.T, labels)        # each floor plan should rank its paired text highest
) / 2
```

Cross entropy here treats each row of `sim` as logits for a B-class classification problem. The model is penalised when the correct pair (diagonal) does not have the highest similarity score.

**Temperature τ:** Controls how sharp the distribution is.
- τ = 0.07 is the standard starting value (used in CLIP)
- Lower τ → sharper distribution → harder training signal
- Higher τ → softer → easier but weaker learning
- Start with 0.07. If training is unstable (loss explodes), increase to 0.1.

**What gets updated:**
- `TextEncoder.projection` — the 768→512→256 MLP head. BERT weights stay frozen initially.
- `GraphPlanEncoder` — ALL weights update (convolutional layers, norms, readout MLP).

BERT's transformer layers are expensive and already encode strong semantic knowledge. Keeping them frozen for the first 5 epochs is efficient. After epoch 5, you may optionally unfreeze the last 2 transformer layers with a very small learning rate (1e-5) for fine-grained alignment.

---

## FILE 1 — pair_builder.py

```python
"""
pair_builder.py
Builds weakly supervised (text_query, floor_plan_graph_path) pairs
using bedroom count as the matching signal.
Output: pairs.json — list of {"query": str, "graph_path": str, "bedroom_count": int}
"""
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

BED_SEMANTIC_ID = 8   # FloorPlanCAD semantic class for "bed"

def estimate_bedrooms(contract_path: str) -> int | None:
    """
    Returns estimated bedroom count from a contract JSON.
    Returns None for commercial/parking layouts with no beds.
    """
    data = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    bed_nodes = [n for n in data.get("nodes", []) if n.get("semantic_id") == BED_SEMANTIC_ID]
    count = len(bed_nodes)
    if count == 0:
        return None
    elif count <= 1:
        return 1
    elif count <= 3:
        return 2
    elif count <= 5:
        return 3
    else:
        return 4


def build_pairs(
    text_queries_path: str = "text_queries.json",
    train_dir: str = "train",
    output_path: str = "pairs.json",
    pairs_per_category: int = 500,
):
    text_data = json.loads(Path(text_queries_path).read_text(encoding="utf-8"))
    contract_paths = list(Path(train_dir).glob("*_contract.json"))
    print(f"Found {len(contract_paths)} contract files")

    # Group text queries by bedroom count
    text_by_beds = defaultdict(list)
    for item in text_data:
        n = item["layout_summary"]["bedrooms"]
        text_by_beds[n].append(item["query"])

    # Also group commercial
    commercial_texts = [item["query"] for item in text_data if item["template_type"] == "commercial"]

    # Group floor plans by estimated bedroom count
    geo_by_beds = defaultdict(list)
    commercial_geos = []
    print("Estimating bedroom counts for all floor plans...")
    for i, path in enumerate(contract_paths):
        if i % 200 == 0:
            print(f"  {i}/{len(contract_paths)}")
        try:
            n = estimate_bedrooms(str(path))
            if n is None:
                commercial_geos.append(str(path))
            else:
                geo_by_beds[n].append(str(path))
        except Exception:
            continue

    pairs = []

    # Residential pairs (1–4 bedrooms)
    for n_beds in [1, 2, 3, 4]:
        texts = text_by_beds[n_beds]
        geos  = geo_by_beds[n_beds]
        if not texts or not geos:
            print(f"  WARNING: no data for {n_beds} bedrooms (texts={len(texts)}, geos={len(geos)})")
            continue
        k = min(len(texts), len(geos), pairs_per_category)
        sampled_texts = random.sample(texts, k)
        sampled_geos  = random.sample(geos, k)
        for t, g in zip(sampled_texts, sampled_geos):
            pairs.append({"query": t, "graph_path": g, "bedroom_count": n_beds})
        print(f"  {n_beds} bedrooms: {k} pairs (from {len(texts)} texts, {len(geos)} geos)")

    # Commercial pairs
    k_commercial = min(len(commercial_texts), len(commercial_geos), pairs_per_category)
    if k_commercial > 0:
        for t, g in zip(random.sample(commercial_texts, k_commercial),
                        random.sample(commercial_geos, k_commercial)):
            pairs.append({"query": t, "graph_path": g, "bedroom_count": 0})
        print(f"  commercial: {k_commercial} pairs")

    random.shuffle(pairs)
    Path(output_path).write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"\nTotal pairs: {len(pairs)} → {output_path}")
    return pairs


if __name__ == "__main__":
    build_pairs()
```

---

## FILE 2 — pair_dataset.py

```python
"""
pair_dataset.py
PairedDataset: wraps pairs.json and yields (query_string, PyG_Data_object) items.
The collate_fn handles heterogeneous batching (strings + graphs).
"""
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset

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
        self.pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))

        # Try to load Person 2's pre-built graph cache for speed
        # If cache exists, build a lookup from source_json → graph Data
        self.graph_cache: Dict[str, Data] = {}
        if Path(cache_path).exists():
            print(f"Loading graph cache from {cache_path}...")
            records, category_maps = load_cache(cache_path)
            self.category_maps = category_maps
            for rec in records:
                if rec.data is not None:
                    self.graph_cache[rec.source_json] = rec.data
                elif rec.graph_path:
                    # lazy — load on first access
                    self.graph_cache[rec.source_json] = rec.graph_path  # store path string
            print(f"  Cached {len(self.graph_cache)} graphs")
        else:
            # No cache — will parse JSONs on the fly (slower)
            print("No graph cache found. Graphs will be parsed from JSON on the fly.")
            self.category_maps = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        query = pair["query"]
        graph_path = pair["graph_path"]
        bedroom_count = pair.get("bedroom_count", -1)

        # Load graph
        cached = self.graph_cache.get(graph_path)
        if cached is None:
            # Not in cache — parse from JSON
            if self.category_maps is not None:
                record = contract_json_to_pyg(Path(graph_path), self.category_maps)
                graph = record.data
            else:
                # No category maps — this will have variable feature dim, OK for testing
                record = contract_json_to_pyg(Path(graph_path), {"layer_vocab": {"UNK": 0}, "geo_vocab": {"other": 0}})
                graph = record.data
        elif isinstance(cached, str):
            # It's a path string — load the .pt file
            graph = torch.load(cached, map_location="cpu", weights_only=False)
        else:
            graph = cached

        return {"query": query, "graph": graph, "bedroom_count": bedroom_count}


def paired_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate: strings stay as list, graphs get PyG batched.
    """
    queries = [item["query"] for item in batch]
    graphs  = Batch.from_data_list([item["graph"] for item in batch])
    bedroom_counts = torch.tensor([item["bedroom_count"] for item in batch], dtype=torch.long)
    return {
        "queries": queries,             # list of B strings
        "graphs": graphs,               # PyG Batch object
        "bedroom_counts": bedroom_counts,
    }
```

---

## FILE 3 — alignment_trainer.py

```python
"""
alignment_trainer.py
Contrastive alignment training (InfoNCE / NT-Xent loss).
Trains TextEncoder.projection and GraphPlanEncoder jointly to share a 256-dim embedding space.

Run:
    python alignment_trainer.py --pairs pairs.json --epochs 10 --batch-size 32
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from text_encoder import TextEncoder, preprocess_query
from graph_model import GraphPlanEncoder
from graph_dataset import load_cache
from pair_dataset import PairedDataset, paired_collate_fn


# ── Loss ─────────────────────────────────────────────────────────────────────

def infonce_loss(text_vecs: torch.Tensor, geo_vecs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Symmetric InfoNCE loss.
    text_vecs: [B, 256]  L2-normalised
    geo_vecs:  [B, 256]  L2-normalised
    Returns: scalar loss
    """
    B = text_vecs.shape[0]
    labels = torch.arange(B, device=text_vecs.device)

    # [B, B] cosine similarity matrix
    sim = (text_vecs @ geo_vecs.T) / temperature

    # Symmetric: both text→geo and geo→text directions
    loss_t2g = F.cross_entropy(sim,   labels)   # each text finds its floor plan
    loss_g2t = F.cross_entropy(sim.T, labels)   # each floor plan finds its text
    return (loss_t2g + loss_g2t) / 2


# ── Device ───────────────────────────────────────────────────────────────────

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Main training function ────────────────────────────────────────────────────

def train(args):
    device = pick_device()
    print(f"Device: {device}")

    # ── Load models ───────────────────────────────────────────────────────────

    # Person 3's text encoder
    text_encoder = TextEncoder(output_dim=256, freeze_bert=True).to(device)
    print(f"TextEncoder loaded. Trainable params (projection only): "
          f"{sum(p.numel() for p in text_encoder.parameters() if p.requires_grad):,}")

    # Person 2's graph encoder
    checkpoint = torch.load(args.graph_checkpoint, map_location=device, weights_only=False)
    graph_encoder = GraphPlanEncoder(
        in_dim=checkpoint["in_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        out_dim=256,
        dropout=checkpoint["config"]["dropout"],
        conv_type=checkpoint["config"]["conv_type"],
    ).to(device)
    graph_encoder.load_state_dict(checkpoint["model_state_dict"])
    print(f"GraphPlanEncoder loaded. Trainable params: "
          f"{sum(p.numel() for p in graph_encoder.parameters() if p.requires_grad):,}")

    # ── Dataset and DataLoader ────────────────────────────────────────────────

    dataset = PairedDataset(pairs_path=args.pairs, cache_path=args.cache)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=paired_collate_fn,
        drop_last=True,      # InfoNCE needs full batches for in-batch negatives
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=paired_collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
    )
    print(f"Train pairs: {n_train}, Val pairs: {n_val}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Separate learning rates: small for graph encoder (it's already trained),
    # larger for text projection (random init, needs to move more)
    optimizer = torch.optim.AdamW([
        {"params": text_encoder.projection.parameters(), "lr": args.lr_text},
        {"params": graph_encoder.parameters(),           "lr": args.lr_graph},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    temperature = args.temperature

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── Optional: unfreeze BERT last 2 layers after epoch 5 ──────────────
        if epoch == args.unfreeze_bert_epoch and args.unfreeze_bert_epoch > 0:
            print(f"Epoch {epoch}: Unfreezing BERT last 2 transformer layers...")
            bert_layers = list(text_encoder.bert.encoder.layer)
            for layer in bert_layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
            optimizer.add_param_group({
                "params": [p for l in bert_layers[-2:] for p in l.parameters()],
                "lr": 1e-5,
            })

        # ── Train epoch ───────────────────────────────────────────────────────
        text_encoder.train()
        graph_encoder.train()
        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            graphs  = batch["graphs"].to(device)
            queries = batch["queries"]          # list of strings, stays on CPU

            # Encode
            text_vecs  = text_encoder(queries)            # [B, 256]
            geo_vecs   = graph_encoder(graphs)            # [B, 256]

            # Loss
            loss = infonce_loss(text_vecs, geo_vecs, temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(text_encoder.parameters()) + list(graph_encoder.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            total_loss   += loss.item()
            total_batches += 1

            if batch_idx % 20 == 0:
                print(f"  [{epoch}/{args.epochs}] batch {batch_idx}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        train_loss = total_loss / max(total_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        text_encoder.eval()
        graph_encoder.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch["queries"]) < 2:
                    continue  # InfoNCE needs at least 2 pairs
                graphs  = batch["graphs"].to(device)
                queries = batch["queries"]
                text_vecs = text_encoder(queries)
                geo_vecs  = graph_encoder(graphs)
                loss = infonce_loss(text_vecs, geo_vecs, temperature)
                val_loss    += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{args.epochs} | train={train_loss:.4f} val={val_loss:.4f} "
              f"time={epoch_time:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "text_encoder_projection": text_encoder.projection.state_dict(),
                "text_encoder_full": text_encoder.state_dict(),
                "graph_encoder": graph_encoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "temperature": temperature,
            }, out_dir / "best_alignment_checkpoint.pt")
            print(f"  Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save training history
    (out_dir / "alignment_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Next step: re-export Person 2 embeddings using updated graph encoder.")
    print(f"  python export_plan_embeddings.py --checkpoint-path {out_dir}/best_alignment_checkpoint.pt [MODIFIED]")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",               default="pairs.json")
    parser.add_argument("--graph-checkpoint",    default="artifacts/runs/graph_baseline/best_checkpoint.pt")
    parser.add_argument("--cache",               default="artifacts/cache/graph_cache.pt")
    parser.add_argument("--out-dir",             default="artifacts/runs/alignment")
    parser.add_argument("--epochs",              type=int,   default=10)
    parser.add_argument("--batch-size",          type=int,   default=32)
    parser.add_argument("--lr-text",             type=float, default=1e-3)
    parser.add_argument("--lr-graph",            type=float, default=1e-4)
    parser.add_argument("--temperature",         type=float, default=0.07)
    parser.add_argument("--unfreeze-bert-epoch", type=int,   default=6)
    parser.add_argument("--num-workers",         type=int,   default=4)
    args = parser.parse_args()
    train(args)
```

---

## FILE 4 — evaluate_retrieval.py

```python
"""
evaluate_retrieval.py
Measures Recall@k and MRR on held-out text queries.
Uses Person 2's PlanRetrievalIndex and the aligned TextEncoder.

Run:
    python evaluate_retrieval.py --checkpoint artifacts/runs/alignment/best_alignment_checkpoint.pt
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from text_encoder import TextEncoder, preprocess_query
from retrieval_index import PlanRetrievalIndex
from graph_dataset import load_cache


def estimate_bedrooms_from_index(embedding_index_path: str, train_dir: str) -> dict:
    """
    Build a mapping: floor_plan_id → estimated bedroom count
    by reading each contract JSON and counting bed nodes (semantic_id==8).
    """
    BED_SEMANTIC_ID = 8
    index_rows = json.loads(Path(embedding_index_path).read_text(encoding="utf-8"))
    id_to_beds = {}
    for row in index_rows:
        fid  = row["floor_plan_id"]
        path = Path(row["source_json"])
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            beds = sum(1 for n in data.get("nodes", []) if n.get("semantic_id") == BED_SEMANTIC_ID)
            if beds == 0:
                id_to_beds[fid] = 0   # commercial
            elif beds <= 1:
                id_to_beds[fid] = 1
            elif beds <= 3:
                id_to_beds[fid] = 2
            elif beds <= 5:
                id_to_beds[fid] = 3
            else:
                id_to_beds[fid] = 4
        except Exception:
            id_to_beds[fid] = -1
    return id_to_beds


def evaluate(args):
    device = torch.device("cpu")   # eval can run on CPU

    # Load text encoder
    text_encoder = TextEncoder(output_dim=256, freeze_bert=True)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "text_encoder_full" in ckpt:
            text_encoder.load_state_dict(ckpt["text_encoder_full"])
            print(f"Loaded aligned text encoder from {args.checkpoint}")
        elif "text_encoder_projection" in ckpt:
            text_encoder.projection.load_state_dict(ckpt["text_encoder_projection"])
            print(f"Loaded aligned projection from {args.checkpoint}")
    text_encoder.eval()

    # Load retrieval index
    index = PlanRetrievalIndex(
        embeddings_path=args.embeddings,
        index_path=args.embedding_index,
    )
    print(f"Retrieval index: {index.embeddings.shape[0]} floor plans, dim={index.dim}")

    # Build floor_plan_id → bedroom count mapping
    id_to_beds = estimate_bedrooms_from_index(args.embedding_index, args.train_dir)

    # Load test text queries (use last 200 of text_queries.json as held-out)
    all_queries = json.loads(Path(args.text_queries).read_text(encoding="utf-8"))
    test_queries = all_queries[-200:]   # held-out set

    # Evaluate
    hits_at = defaultdict(int)   # hits_at[k] = number of queries where correct bedroom in top-k
    reciprocal_ranks = []
    ks = [1, 5, 10]

    for item in test_queries:
        query_text  = preprocess_query(item["query"])
        query_beds  = item["layout_summary"]["bedrooms"]

        with torch.no_grad():
            q_vec = text_encoder([query_text])[0].numpy()   # [256]

        results = index.search(q_vec, top_k=max(ks))
        retrieved_ids = [r["floor_plan_id"] for r in results]

        # Check bedroom match at each k
        first_hit_rank = None
        for rank, fid in enumerate(retrieved_ids, start=1):
            retrieved_beds = id_to_beds.get(fid, -1)
            if retrieved_beds == query_beds:
                if first_hit_rank is None:
                    first_hit_rank = rank
            for k in ks:
                if rank == k:
                    break
        for k in ks:
            matched = any(id_to_beds.get(fid, -1) == query_beds
                          for fid in retrieved_ids[:k])
            if matched:
                hits_at[k] += 1

        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    n = len(test_queries)
    print(f"\n{'='*50}")
    print(f"Evaluation on {n} held-out queries")
    for k in ks:
        recall = hits_at[k] / n
        print(f"  Recall@{k:2d}: {recall:.3f}  ({hits_at[k]}/{n})")
    print(f"  MRR:      {sum(reciprocal_ranks)/n:.3f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      default="")
    parser.add_argument("--embeddings",      default="artifacts/handoff/embeddings.npy")
    parser.add_argument("--embedding-index", default="artifacts/handoff/embedding_index.json")
    parser.add_argument("--text-queries",    default="text_queries.json")
    parser.add_argument("--train-dir",       default="train")
    args = parser.parse_args()
    evaluate(args)
```

---

## FILE 5 — re-export after alignment

After alignment training, the graph encoder weights have changed. You must regenerate `embeddings.npy` using the updated encoder. Person 2's `export_plan_embeddings.py` accepts `--checkpoint-path`. You need to modify it slightly to load the graph encoder from your alignment checkpoint instead of Person 2's pretext checkpoint.

Add this to `export_plan_embeddings.py` or write a wrapper:

```python
# At the top of export_plan_embeddings.py, modify the checkpoint loading to:
checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

# Check if it's an alignment checkpoint (has "graph_encoder" key) 
# or Person 2's original checkpoint (has "model_state_dict" key)
if "graph_encoder" in checkpoint:
    # Alignment checkpoint
    model = GraphPlanEncoder(
        in_dim=checkpoint["in_dim"],          # need to store this in your checkpoint too
        hidden_dim=128,                        # or load from config
        out_dim=256,
        dropout=0.2,
        conv_type="sage",
    ).to(device)
    model.load_state_dict(checkpoint["graph_encoder"])
else:
    # Original Person 2 checkpoint
    model = GraphPlanEncoder(
        in_dim=checkpoint["in_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        out_dim=checkpoint["config"]["out_dim"],
        dropout=checkpoint["config"]["dropout"],
        conv_type=checkpoint["config"]["conv_type"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
```

**Store `in_dim` in your alignment checkpoint!** Add it when saving:
```python
torch.save({
    ...
    "in_dim": graph_encoder.conv1.in_channels,   # add this line
    ...
}, out_dir / "best_alignment_checkpoint.pt")
```

Then run:
```bash
python export_plan_embeddings.py \
    --checkpoint-path artifacts/runs/alignment/best_alignment_checkpoint.pt \
    --out-dir artifacts/handoff_aligned
```

This produces `artifacts/handoff_aligned/embeddings.npy` — the aligned embedding matrix that your inference pipeline will use.

---

## FILE 6 — inference.py

```python
"""
inference.py
End-to-end inference: user types a text query, gets back top-k floor plans.

Usage:
    python inference.py --query "2 bedroom flat with balcony and modern kitchen"

Or import and call:
    from inference import retrieve
    results = retrieve("a 3 bedroom house with attached bathrooms")
"""
import argparse
import json
import torch
from pathlib import Path

from text_encoder import TextEncoder, preprocess_query
from retrieval_index import PlanRetrievalIndex


_text_encoder = None
_retrieval_index = None

DEFAULT_TEXT_CHECKPOINT   = "artifacts/runs/alignment/best_alignment_checkpoint.pt"
DEFAULT_EMBEDDINGS        = "artifacts/handoff_aligned/embeddings.npy"
DEFAULT_EMBEDDING_INDEX   = "artifacts/handoff_aligned/embedding_index.json"


def _load_models(text_checkpoint: str, embeddings: str, embedding_index: str):
    global _text_encoder, _retrieval_index

    if _text_encoder is None:
        _text_encoder = TextEncoder(output_dim=256, freeze_bert=True)
        if Path(text_checkpoint).exists():
            ckpt = torch.load(text_checkpoint, map_location="cpu", weights_only=False)
            if "text_encoder_full" in ckpt:
                _text_encoder.load_state_dict(ckpt["text_encoder_full"])
            elif "text_encoder_projection" in ckpt:
                _text_encoder.projection.load_state_dict(ckpt["text_encoder_projection"])
            print(f"Loaded text encoder from {text_checkpoint}")
        _text_encoder.eval()

    if _retrieval_index is None:
        _retrieval_index = PlanRetrievalIndex(
            embeddings_path=embeddings,
            index_path=embedding_index,
        )
        print(f"Loaded retrieval index: {_retrieval_index.embeddings.shape[0]} floor plans")


def retrieve(
    query: str,
    top_k: int = 5,
    text_checkpoint: str = DEFAULT_TEXT_CHECKPOINT,
    embeddings: str = DEFAULT_EMBEDDINGS,
    embedding_index: str = DEFAULT_EMBEDDING_INDEX,
):
    """
    Main function. Takes a user's natural language query,
    returns top_k floor plan results sorted by cosine similarity.

    Returns: list of dicts, each with keys:
        rank, floor_plan_id, source_json, score
    """
    _load_models(text_checkpoint, embeddings, embedding_index)

    # Preprocess and encode
    cleaned = preprocess_query(query)
    with torch.no_grad():
        q_vec = _text_encoder([cleaned])[0].numpy()   # [256]

    # Search
    results = _retrieval_index.search(q_vec, top_k=top_k, normalize_query=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",     required=True, type=str)
    parser.add_argument("--top-k",    type=int, default=5)
    parser.add_argument("--checkpoint", default=DEFAULT_TEXT_CHECKPOINT)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--index",      default=DEFAULT_EMBEDDING_INDEX)
    args = parser.parse_args()

    results = retrieve(
        query=args.query,
        top_k=args.top_k,
        text_checkpoint=args.checkpoint,
        embeddings=args.embeddings,
        embedding_index=args.index,
    )

    print(f"\nQuery: {args.query}\n")
    for r in results:
        print(f"  Rank {r['rank']}: {r['floor_plan_id']}  score={r['score']:.4f}")
        print(f"           {r['source_json']}")
```

---

## THE COMPLETE EXECUTION ORDER

Run these steps in order:

```bash
# Step 1: Build pairs
python pair_builder.py
# Output: pairs.json (~2000 pairs)

# Step 2: Baseline evaluation (before training — should be ~25% Recall@5)
python evaluate_retrieval.py
# No checkpoint needed; uses random-init projection

# Step 3: Train alignment
python alignment_trainer.py \
    --pairs pairs.json \
    --epochs 10 \
    --batch-size 32 \
    --temperature 0.07
# Output: artifacts/runs/alignment/best_alignment_checkpoint.pt

# Step 4: Post-training evaluation (should be 50–65%+ Recall@5)
python evaluate_retrieval.py \
    --checkpoint artifacts/runs/alignment/best_alignment_checkpoint.pt

# Step 5: Re-export aligned embeddings
python export_plan_embeddings.py \
    --checkpoint-path artifacts/runs/alignment/best_alignment_checkpoint.pt \
    --out-dir artifacts/handoff_aligned

# Step 6: Final evaluation with aligned embeddings
python evaluate_retrieval.py \
    --checkpoint artifacts/runs/alignment/best_alignment_checkpoint.pt \
    --embeddings artifacts/handoff_aligned/embeddings.npy \
    --embedding-index artifacts/handoff_aligned/embedding_index.json

# Step 7: Test inference
python inference.py --query "2 bedroom apartment with balcony and attached bathroom"
python inference.py --query "commercial office layout with open workspace"
python inference.py --query "3 bhk flat with modular kitchen"   # tests preprocessing
```

---

## EXPECTED METRICS AND WHAT TO DO IF THEY'RE WRONG

| Stage | Expected Recall@5 | If lower |
|---|---|---|
| Before training (random projection) | ~20–30% | Normal — this is baseline |
| After 5 epochs | ~45–55% | Check temperature, check pair balance |
| After 10 epochs | ~55–70% | Check if BERT unfreezing helped |
| After BERT unfreezing (epoch 6+) | +5–10% | If not, keep BERT frozen |

**If val loss is not decreasing:**
- Lower learning rate for text projection (`--lr-text 3e-4`)
- Increase batch size if memory allows (`--batch-size 64`) — more in-batch negatives = stronger signal
- Check that `drop_last=True` in DataLoader — InfoNCE needs uniform batch sizes

**If Recall@5 is below 30% even after training:**
- Your pairs may be too noisy. Check `estimate_bedrooms` is returning sensible counts — print a distribution
- Your batch may lack diversity. Ensure each batch has a mix of bedroom counts, not all 2-bedroom pairs

**If training loss goes to 0 immediately:**
- Temperature is too low. Increase to 0.1 or 0.2
- Batch size is too small (fewer negatives)

---

## DEPENDENCIES REQUIRED

```
torch >= 2.0
torch-geometric
transformers
numpy
```

Install:
```bash
pip install torch torchvision
pip install torch-geometric
pip install transformers
pip install numpy
```

---

## IMPORTANT DESIGN DECISIONS TO KEEP IN MIND

1. **Do not modify Person 2's `retrieval_index.py`** — use it as-is. It is correct and complete.

2. **Do not modify Person 3's `TextEncoder` class** — only update its weights through the checkpoint save/load mechanism.

3. **Always call `preprocess_query()` at inference time** — it handles BHK, sqft, 2br abbreviations that Indian users commonly type.

4. **L2 normalisation is already done inside both encoders** — do not normalise again yourself. Person 2 and Person 3 both end with `F.normalize(projected, p=2, dim=-1)`. The similarity is just a dot product.

5. **`drop_last=True` is essential** — InfoNCE loss assumes all rows in the `[B, B]` matrix are valid. A batch of size 1 would have no negatives and produce NaN.

6. **After alignment training, you MUST re-run the export** — the graph encoder weights change during your training. The old `embeddings.npy` from Person 2 is now stale and will produce wrong retrieval results.

7. **The `in_dim` of the graph encoder is variable** — it depends on the vocabulary sizes built by Person 2's `build_cache()` function. Always read it from the checkpoint (`checkpoint["in_dim"]`), never hardcode it.

---

## SUMMARY: WHAT YOU RECEIVE, WHAT YOU DO, WHAT YOU PRODUCE

**You receive from Person 2:**
- `embeddings.npy` [N, 256] — pre-alignment geometric embeddings
- `embedding_index.json` — floor_plan_id ↔ row mapping
- `best_checkpoint.pt` — trained GraphPlanEncoder weights
- `retrieval_index.py` — ready-to-use cosine search class
- `graph_dataset.py`, `graph_model.py` — model and data utilities

**You receive from Person 3:**
- `text_queries.json` — 2000 synthetic queries with bedroom metadata
- `text_encoder.py` — TextEncoder class (BERT + projection → 256-dim unit vectors)

**You build:**
- `pairs.json` — ~2000 (text, floor plan path) pairs via bedroom-count matching
- Training infrastructure — PairedDataset, InfoNCE loss, joint training loop
- `best_alignment_checkpoint.pt` — aligned encoder weights
- `artifacts/handoff_aligned/embeddings.npy` — re-exported aligned geometric embeddings
- `inference.py` — end-to-end pipeline: string → top-k floor plans

**The final system behaviour:**
A user types "I want a 2 bedroom flat with an open kitchen." Your `inference.py` encodes this to a 256-dim vector using the aligned TextEncoder, does a dot-product search against `embeddings.npy`, and returns the top-5 most similar floor plan files from the FloorPlanCAD dataset, ranked by cosine similarity score.
