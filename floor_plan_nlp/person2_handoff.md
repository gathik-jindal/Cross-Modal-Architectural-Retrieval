# Person 2 -> Person 4 Handoff Contract

## Output Artifacts

- `artifacts/handoff/embeddings.npy`: `[N, 256]` float32 matrix, L2-normalized row-wise.
- `artifacts/handoff/embeddings.pt`: Torch copy of the same embedding matrix.
- `artifacts/handoff/embedding_index.json`: row index to source floor-plan JSON mapping.
- `artifacts/handoff/handoff_summary.json`: metadata summary for reproducibility.

## Similarity

- Use cosine similarity. Since vectors are L2-normalized, `dot(a, b)` is cosine similarity.

## Embedding Dimension

- Geometric embedding size is fixed to `256`.
- Person 3 text encoder must output `256` dimensions for direct alignment.

## Reproducible Pipeline Commands

Run these from `floor_plan_nlp/`:

```bash
python3 -m pip install -r requirements.txt
python3 train_graph_encoder.py --train-dir ../train --epochs 20 --batch-size 8 --conv-type sage
python3 export_plan_embeddings.py --cache-path artifacts/cache/graph_cache.pt --checkpoint-path artifacts/runs/graph_baseline/best_checkpoint.pt
```

## Split Contract

- Deterministic split file is saved at `artifacts/splits/train_val_test_split.json`.
- Team members should reuse this exact split for metric consistency.

