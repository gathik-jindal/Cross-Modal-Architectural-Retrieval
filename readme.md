# Cross-Modal Architectural Retrieval

Cross-modal retrieval pipeline for FloorPlanCAD-style floor plans, combining:

- a graph encoder (GNN) over contract JSONs, and
- a text encoder for natural-language floor plan queries.

The objective is to place both modalities in a shared embedding space for retrieval.

## Overview

This repository contains two main branches:

1. **Graph branch (plan geometry/structure)**
   - Parses/consumes floor plan contract JSON files
   - Trains a graph encoder
   - Exports normalized plan embeddings

2. **Text branch (user/query language)**
   - Encodes query text into vectors
   - Supports retrieval against exported plan embeddings

## Core Files

### Graph pipeline

- `floor_plan_nlp/graph_dataset.py`  
  Converts `*_contract.json` files into PyTorch Geometric graphs (`x`, `edge_index`, `edge_attr`), builds cache, and creates train/val/test splits.

- `floor_plan_nlp/graph_model.py`  
  Defines `GraphPlanEncoder` (GraphSAGE/GCN, pooling, projection, L2 normalization).

- `floor_plan_nlp/train_graph_encoder.py`  
  End-to-end training loop, validation, early stopping, and checkpoint/report saving.

- `floor_plan_nlp/export_plan_embeddings.py`  
  Loads best checkpoint, converts graphs into embeddings, exports `.npy/.pt`, and writes ID-row index mapping.

- `floor_plan_nlp/retrieval_index.py`  
  Lightweight cosine retrieval index over exported embeddings.

### Upstream contract generation and schema

- `src/svg_parser.py`  
  SVG to contract JSON parser.

- `src/constants.py`  
  Semantic ID and layer mapping constants used during contract creation.

- `schema.md`  
  Contract schema (`metadata`, `nodes`, `edges`, `symbols`).

## Expected Data Layout

Current training scripts expect contract JSON files under:

```text
train/*_contract.json
```

If you start from raw SVGs, generate contracts first using the parser pipeline in `src/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Graph Encoder

Run from repository root:

```bash
python floor_plan_nlp/train_graph_encoder.py \
  --train-dir train \
  --cache-path floor_plan_nlp/artifacts/cache/graph_cache.pt \
  --cache-stats-path floor_plan_nlp/artifacts/cache/cache_stats.json \
  --split-path floor_plan_nlp/artifacts/splits/train_val_test_split.json \
  --run-dir floor_plan_nlp/artifacts/runs/graph_baseline
```

Primary outputs:

- `floor_plan_nlp/artifacts/runs/graph_baseline/best_checkpoint.pt`
- `floor_plan_nlp/artifacts/runs/graph_baseline/train_report.json`

## Export Plan Embeddings

```bash
python floor_plan_nlp/export_plan_embeddings.py \
  --cache-path floor_plan_nlp/artifacts/cache/graph_cache.pt \
  --checkpoint-path floor_plan_nlp/artifacts/runs/graph_baseline/best_checkpoint.pt \
  --out-dir floor_plan_nlp/artifacts/handoff
```

Generated artifacts:

- `floor_plan_nlp/artifacts/handoff/embeddings.npy`
- `floor_plan_nlp/artifacts/handoff/embeddings.pt`
- `floor_plan_nlp/artifacts/handoff/embedding_index.json`
- `floor_plan_nlp/artifacts/handoff/handoff_summary.json`

## Retrieval

Example self-retrieval/smoke retrieval:

```bash
python floor_plan_nlp/retrieval_index.py \
  --embeddings-path floor_plan_nlp/artifacts/handoff/embeddings.npy \
  --index-path floor_plan_nlp/artifacts/handoff/embedding_index.json \
  --query-floor-plan-id 0000-0002.svg \
  --top-k 10
```

## Training Snapshot (Current Baseline)

From `floor_plan_nlp/artifacts/runs/graph_baseline/train_report.json`:

- Graphs: `4265`
- Split sizes: train `3412`, val `426`, test `427`
- Best epoch: `10`
- Best val loss: `0.9961`
- Test accuracy: `0.6885`

## Notes on Query Quality

For stable cross-modal performance, text queries should be generated from actual dataset facts (contract JSON semantics and structure), not random synthetic combinations. Misaligned query generation can significantly degrade retrieval quality even when the graph encoder is trained correctly.

## Dataset Reference

- FloorPlanCAD: <https://floorplancad.github.io/>
