# Person 2 -> Person 3 Handoff Contract

## Status

Person 2 graph encoder training and export are complete. Artifacts are ready for Person 3.

## Output Artifacts For Person 3

- `artifacts/handoff/embeddings.npy`: `[N, 256]` float32 matrix, L2-normalized row-wise.
- `artifacts/handoff/embeddings.pt`: torch tensor copy of the same matrix.
- `artifacts/handoff/embedding_index.json`: row index to floor-plan metadata mapping.
- `artifacts/handoff/handoff_summary.json`: metadata summary.

## Embedding Dimension

- Geometric embedding size is fixed to `256`.
- Person 3 text encoder must output `256` dimensions for direct alignment.

## Similarity Contract

- Use cosine similarity for retrieval.
- Since vectors are L2-normalized, `dot(a, b)` is cosine similarity.

## Person 3 Training Contract

- Text embeddings must be L2-normalized before similarity scoring.
- Use the shared split file from Person 2 for all metrics.
- Align text embeddings against graph embeddings from `artifacts/handoff/embeddings.npy`.
- Keep inference output dimension exactly `256`.

## Split Contract

- Deterministic split file: `artifacts/splits/train_val_test_split.json`.
- Reuse this exact split for train/val/test consistency across team members.

## Suggested Person 3 Integration Steps

1. Load graph embeddings from `artifacts/handoff/embeddings.npy`.
2. Load row mapping from `artifacts/handoff/embedding_index.json`.
3. Join mapping with your query/annotation source to build text-graph pairs.
4. Train text encoder to produce normalized `256`-d vectors.
5. Evaluate retrieval with cosine similarity (dot product).

## Reproducible Person 2 Pipeline Commands

Run these from `floor_plan_nlp/`:

```bash
python3 -m pip install -r requirements.txt
python3 train_graph_encoder.py --train-dir ../train --epochs 10 --batch-size 8 --conv-type sage --num-workers 2 --log-every 5
python3 export_plan_embeddings.py --cache-path artifacts/cache/graph_cache.pt --checkpoint-path artifacts/runs/graph_baseline/best_checkpoint.pt --num-workers 2 --log-every 10
```

## Artifact Integrity Checks

- Embedding count `N` must match number of rows in `embedding_index.json`.
- Embedding dimension must remain `256`.
- Do not reorder rows unless index mapping is regenerated.


