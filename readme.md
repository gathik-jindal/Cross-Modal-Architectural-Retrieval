# Cross-Modal Architectural Retrieval

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph%20Learning-3C2179)](https://pyg.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E)](https://huggingface.co/docs/transformers)
[![OpenAI CLIP](https://img.shields.io/badge/OpenAI-CLIP-111111)](https://github.com/openai/CLIP)
[![Dataset](https://img.shields.io/badge/Dataset-FloorPlanCAD-2E7D32)](https://floorplancad.github.io/)

Natural-language retrieval for architectural floor plans. The system turns FloorPlanCAD-style SVG drawings into graph contracts, learns plan embeddings with a graph neural network, aligns those embeddings with natural-language queries, and retrieves the most relevant floor plans using cosine similarity.

## Results

These are the current checked-in experiment results from `floor_plan_nlp/artifacts/`.

### Graph Encoder Baseline

| Metric | Value |
|---|---:|
| Graphs processed | 4,265 |
| Train / Val / Test split | 3,412 / 426 / 427 |
| Best epoch | 10 |
| Best validation loss | 0.9961 |
| Validation accuracy | 72.30% |
| Test accuracy | 68.85% |
| Training wall time | 7m 04s |
| Device | CUDA with mixed precision |

### Text-to-Plan Retrieval

Exact floor-plan ID retrieval over a 6,000-plan index using 200 held-out text queries.

| Metric | Model | Random Baseline | Lift |
|---|---:|---:|---:|
| Recall@1 | 2.00% | 0.0167% | 120.0x |
| Recall@5 | 4.50% | 0.0833% | 54.0x |
| Recall@10 | 7.00% | 0.1667% | 42.0x |
| MRR | 3.14% | - | - |

Per-scale Recall@5:

| Scale bucket | Recall@5 |
|---|---:|
| Commercial / no bedrooms | 0.62% |
| Small residential | 25.00% |
| Medium complex | 41.67% |
| Large complex | 0.00% |

The graph encoder is already learning useful structural representations. The cross-modal retrieval stage also beats random by a wide margin, but the bucket breakdown shows the model is much stronger on small and medium residential plans than on commercial or very large layouts.

## Index

- [Results](#results)
- [What This Project Does](#what-this-project-does)
- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Core Components](#core-components)
- [Pipeline](#pipeline)
- [Setup](#setup)
- [Expected Data Layout](#expected-data-layout)
- [Generate Contracts](#generate-contracts)
- [Train Graph Encoder](#train-graph-encoder)
- [Export Plan Embeddings](#export-plan-embeddings)
- [Train Text-Graph Alignment](#train-text-graph-alignment)
- [Evaluate Retrieval](#evaluate-retrieval)
- [Run Inference](#run-inference)
- [CLIP Baseline](#clip-baseline)
- [Notes](#notes)
- [Dataset Reference](#dataset-reference)

## What This Project Does

Architectural retrieval is difficult because the two modalities are very different:

- A floor plan is geometric and symbolic: walls, doors, windows, fixtures, room-like objects, topology, and spatial adjacency.
- A user query is linguistic: "compact residential plan with bedrooms, doors, windows, and connected living spaces."

This repository bridges those two representations. It parses SVG drawings into structured graph contracts, trains a graph encoder to represent each plan as a dense vector, trains a text encoder to map natural-language descriptions into the same vector space, and then performs nearest-neighbor retrieval.

At a high level:

```text
SVG / PNG floor plans
        |
        v
SVG parser -> contract JSON -> PyTorch Geometric graph
        |                              |
        |                              v
        |                      GraphPlanEncoder
        |                              |
        v                              v
Generated text pairs          Plan embedding space
        |                              ^
        v                              |
Transformer text encoder -> contrastive alignment
        |
        v
Text query -> cosine retrieval -> ranked floor plans
```

## Highlights

- Converts raw SVG floor plans into graph-ready contract JSONs.
- Extracts semantic labels, normalized geometry, path features, symbols, and proximity edges.
- Uses PyTorch Geometric for graph representation and GraphSAGE/GCN encoders.
- Uses a Transformer text encoder with contrastive alignment for cross-modal retrieval.
- Exports reusable `.npy` and `.pt` embedding indexes for fast top-k search.
- Includes a CLIP image-text baseline for comparison against visual retrieval.
- Keeps experiment artifacts, split files, handoff notes, schema docs, and reports in predictable locations.

## Repository Layout

```text
.
+-- floor_plan_nlp/         # Main Python package
|   +-- artifacts/          # Model outputs, graph caches, exported embeddings, reports
|   +-- *.json              # Query pairs, extracted attributes, eval splits, baseline results
+-- contracts/              # Generated contract JSONs for train/test splits
+-- data/                   # Raw FloorPlanCAD SVG/PNG data and source archives
+-- docs/                   # Schema, mapping notes, handoffs, reference PDFs
+-- artifacts/              # Repo-level generated caches
+-- requirements.txt
```

The project uses one code package: `floor_plan_nlp/`.

## Core Components

| Component | Files | Role |
|---|---|---|
| SVG parsing | `svg_parser.py`, `geometry.py`, `constants.py` | Converts SVG geometry and metadata into contract JSONs |
| Batch processing | `batch_runner.py` | Generates train/test contract folders from raw SVGs |
| Graph dataset | `graph_dataset.py` | Converts contract JSONs into PyTorch Geometric `Data` objects |
| Graph encoder | `graph_model.py`, `train_graph_encoder.py` | Learns structural floor-plan embeddings |
| Text encoder | `text_encoder.py` | Converts natural-language queries into dense vectors |
| Pair generation | `extract_dynamic_attributes.py`, `generate_dynamic_queries.py` | Builds text-plan supervision from plan attributes |
| Alignment | `alignment_trainer.py`, `pair_dataset.py` | Trains text and graph embeddings into a shared space |
| Retrieval | `retrieval_index.py`, `inference.py`, `evaluate_retrieval.py` | Searches and evaluates the embedding index |
| CLIP baseline | `clip_baseline.py` | Tests image-text retrieval over rendered plan PNGs |

## Pipeline

| Stage | Entry Point | Output |
|---|---|---|
| Contract generation | `python -m floor_plan_nlp.batch_runner` | `contracts/train`, `contracts/test` |
| Graph cache build | `python -m floor_plan_nlp.train_graph_encoder --rebuild-cache` | `floor_plan_nlp/artifacts/cache` |
| Graph encoder training | `python -m floor_plan_nlp.train_graph_encoder` | `best_checkpoint.pt`, `train_report.json` |
| Plan embedding export | `python -m floor_plan_nlp.export_plan_embeddings` | `embeddings.npy`, `embedding_index.json` |
| Text-graph alignment | `python -m floor_plan_nlp.alignment_trainer` | `best_alignment_checkpoint.pt` |
| Retrieval evaluation | `python -m floor_plan_nlp.evaluate_retrieval` | Recall@k, MRR, eval JSON |
| User inference | `python -m floor_plan_nlp.inference` | Ranked floor-plan matches |
| CLIP baseline | `python -m floor_plan_nlp.clip_baseline` | Image-text retrieval metrics |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric wheels depend on your PyTorch/CUDA combination. If the generic install fails, use the official PyG install selector for your environment.

## Expected Data Layout

```text
data/
+-- train/                  # Raw train SVG/PNG files
+-- test/                   # Raw test SVG/PNG files

contracts/
+-- train/                  # *_contract.json files
+-- test/                   # *_contract.json files
```

Large raw data, generated contracts, checkpoints, `.pt`, and `.npy` files are intentionally excluded from Git. Contract fields are documented in [docs/schema.md](docs/schema.md).

## Generate Contracts

```bash
python -m floor_plan_nlp.batch_runner
```

This reads from `data/train` and `data/test`, then writes contract JSONs to:

```text
contracts/train/
contracts/test/
```

Each contract contains:

- `metadata`: source filename, viewBox, layer counts, and summary stats.
- `nodes`: geometry entities with semantic labels, normalized centers, bounding boxes, lengths, and raw attributes.
- `edges`: proximity relationships between entities.
- `symbols`: grouped semantic instances such as doors, windows, furniture, or room-like elements.

## Train Graph Encoder

```bash
python -m floor_plan_nlp.train_graph_encoder --rebuild-cache
```

Useful options:

```bash
python -m floor_plan_nlp.train_graph_encoder ^
  --epochs 10 ^
  --batch-size 8 ^
  --max-nodes-per-batch 3500 ^
  --conv-type sage
```

Main outputs:

```text
floor_plan_nlp/artifacts/cache/graph_cache_train_test.pt
floor_plan_nlp/artifacts/cache/cache_stats.json
floor_plan_nlp/artifacts/runs/graph_baseline/best_checkpoint.pt
floor_plan_nlp/artifacts/runs/graph_baseline/train_report.json
```

## Export Plan Embeddings

```bash
python -m floor_plan_nlp.export_plan_embeddings
```

Main outputs:

```text
floor_plan_nlp/artifacts/handoff/embeddings.npy
floor_plan_nlp/artifacts/handoff/embeddings.pt
floor_plan_nlp/artifacts/handoff/embedding_index.json
floor_plan_nlp/artifacts/handoff/handoff_summary.json
```

## Train Text-Graph Alignment

```bash
python -m floor_plan_nlp.alignment_trainer --epochs 10 --batch-size 16 --skip-oom-batches
```

Then export aligned graph embeddings:

```bash
python -m floor_plan_nlp.export_plan_embeddings ^
  --checkpoint-path floor_plan_nlp/artifacts/runs/alignment/best_alignment_checkpoint.pt ^
  --out-dir floor_plan_nlp/artifacts/handoff_aligned
```

Alignment uses a CLIP-style contrastive objective: matching text-plan pairs are pulled together in embedding space while non-matching pairs in the batch are pushed apart.

## Evaluate Retrieval

```bash
python -m floor_plan_nlp.evaluate_retrieval ^
  --checkpoint floor_plan_nlp/artifacts/runs/alignment/best_alignment_checkpoint.pt ^
  --out-json floor_plan_nlp/artifacts/eval_results.json
```

The evaluator uses exact floor-plan ID match as ground truth and reports Recall@1, Recall@5, Recall@10, MRR, random baselines, and per-scale bucket performance.

## Run Inference

```bash
python -m floor_plan_nlp.inference ^
  --query "a compact residential floor plan with bedrooms, doors, windows, and connected living spaces" ^
  --top-k 5
```

The command returns ranked floor-plan IDs, source contract paths, and cosine similarity scores.

## CLIP Baseline

Zero-shot:

```bash
python -m floor_plan_nlp.clip_baseline --mode zero-shot --cache artifacts/cache/clip
```

Fine-tune and compare:

```bash
python -m floor_plan_nlp.clip_baseline --mode both --epochs 5 --batch 64 --checkpoint artifacts/cache/finetuned_clip.pt
```

The CLIP baseline evaluates direct image-text matching over plan PNGs. It is useful as a visual retrieval comparison against the structured graph pipeline.

## Notes

- Query quality matters. Strong retrieval depends on text pairs generated from real plan attributes rather than random synthetic descriptions.
- Use `python -m floor_plan_nlp.<module>` from the repository root for the most reliable imports.
- Keep raw data and generated artifacts out of Git unless there is a specific release reason to include them.
- Reference material and handoff notes live in `docs/`.

## Dataset Reference

FloorPlanCAD: <https://floorplancad.github.io/>
