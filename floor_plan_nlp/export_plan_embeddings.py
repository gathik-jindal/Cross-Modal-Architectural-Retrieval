import argparse
import json
from pathlib import Path

import numpy as np
import torch

from graph_dataset import load_cache
from graph_model import GraphPlanEncoder

try:
    from torch_geometric.loader import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for Person 2 export. "
        "Install with: pip install torch-geometric"
    ) from exc


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Export plan embeddings for Person 4")
    parser.add_argument("--cache-path", default="artifacts/cache/graph_cache.pt")
    parser.add_argument("--checkpoint-path", default="artifacts/runs/graph_baseline/best_checkpoint.pt")
    parser.add_argument("--out-dir", default="artifacts/handoff")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conv-type", choices=["sage", "gcn"], default="sage")
    args = parser.parse_args()

    records, _ = load_cache(args.cache_path)
    graphs = [record.data for record in records]
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = pick_device()
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = GraphPlanEncoder(
        in_dim=checkpoint["in_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        out_dim=checkpoint["config"]["out_dim"],
        dropout=checkpoint["config"]["dropout"],
        conv_type=checkpoint["config"]["conv_type"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_embeddings = []
    ordered_ids = []
    ordered_paths = []
    with torch.no_grad():
        ptr = 0
        for batch in loader:
            batch = batch.to(device)
            embeddings = model(batch).cpu().numpy()
            all_embeddings.append(embeddings)
            batch_size = embeddings.shape[0]
            for idx in range(ptr, ptr + batch_size):
                ordered_ids.append(records[idx].floor_plan_id)
                ordered_paths.append(records[idx].source_json)
            ptr += batch_size

    emb_matrix = np.concatenate(all_embeddings, axis=0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "embeddings.npy"
    np.save(npy_path, emb_matrix)
    pt_path = out_dir / "embeddings.pt"
    torch.save(torch.tensor(emb_matrix), pt_path)

    index_rows = [
        {"row": idx, "floor_plan_id": floor_plan_id, "source_json": source_json}
        for idx, (floor_plan_id, source_json) in enumerate(zip(ordered_ids, ordered_paths))
    ]
    index_path = out_dir / "embedding_index.json"
    index_path.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")

    summary_path = out_dir / "handoff_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "num_embeddings": int(emb_matrix.shape[0]),
                "embedding_dim": int(emb_matrix.shape[1]),
                "normalization": "L2",
                "similarity": "cosine (dot product on normalized vectors)",
                "files": {
                    "embeddings_npy": str(npy_path),
                    "embeddings_pt": str(pt_path),
                    "embedding_index": str(index_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved embeddings to {npy_path} and {pt_path}")
    print(f"Saved index map to {index_path}")


if __name__ == "__main__":
    main()

