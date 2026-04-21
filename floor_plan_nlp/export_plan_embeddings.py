import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from graph_dataset import load_cache, contract_json_to_pyg
from graph_model import GraphPlanEncoder

try:
    from torch_geometric.loader import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for Person 2 export. "
        "Install with: pip install torch-geometric"
    ) from exc


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class CachedGraphDataset(Dataset):
    def __init__(self, records, category_maps, base_dir: Path):
        self.records = records
        self.category_maps = category_maps
        self.base_dir = base_dir

    def _resolve_path(self, p: str | None) -> Path | None:
        if not p:
            return None
        path = Path(p)
        if path.exists():
            return path
        resolved = (self.base_dir / path).resolve()
        if resolved.exists():
            return resolved
        return path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if record.data is not None:
            return record.data
        if record.graph_path:
            graph_path = self._resolve_path(record.graph_path)
            if graph_path and graph_path.exists():
                return torch.load(graph_path, map_location="cpu", weights_only=False)
        if record.source_json:
            source_json = self._resolve_path(record.source_json)
            rec = contract_json_to_pyg(source_json, self.category_maps)
            return rec.data
        raise RuntimeError("Record has neither in-memory data nor graph_path")


def main():
    parser = argparse.ArgumentParser(description="Export plan embeddings for Person 4")
    parser.add_argument("--cache-path", default="artifacts/cache/graph_cache.pt")
    parser.add_argument("--checkpoint-path", default="artifacts/runs/graph_baseline/best_checkpoint.pt")
    parser.add_argument("--out-dir", default="artifacts/handoff")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conv-type", choices=["sage", "gcn"], default="sage")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    device = pick_device()
    use_cuda = device.type == "cuda"
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"Export device: {device} ({gpu_name}) | workers={args.num_workers}")
    else:
        print(f"Export device: {device} | workers={args.num_workers}")

    records, category_maps = load_cache(args.cache_path)
    base_dir = Path(args.cache_path).resolve().parents[2]
    graphs = CachedGraphDataset(records, category_maps, base_dir=base_dir)
    loader = DataLoader(
        graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    # Support both Person 2's original checkpoint and Person 4's alignment checkpoint
    if "graph_encoder" in checkpoint:
        # Person 4's alignment checkpoint
        print("Detected alignment checkpoint (Person 4)")
        model = GraphPlanEncoder(
            in_dim=checkpoint["in_dim"],
            hidden_dim=checkpoint["config"]["hidden_dim"],
            out_dim=256,
            dropout=checkpoint["config"]["dropout"],
            conv_type=checkpoint["config"]["conv_type"],
        ).to(device)
        model.load_state_dict(checkpoint["graph_encoder"])
    else:
        # Person 2's original checkpoint
        print("Detected original graph checkpoint (Person 2)")
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
    num_batches = len(loader)
    export_start = time.time()
    print(f"Export started | total_graphs={len(graphs)} total_batches={num_batches}")
    with torch.no_grad():
        ptr = 0
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=use_cuda)
            embeddings = model(batch).cpu().numpy()
            all_embeddings.append(embeddings)
            batch_size = embeddings.shape[0]
            for idx in range(ptr, ptr + batch_size):
                ordered_ids.append(records[idx].floor_plan_id)
                ordered_paths.append(records[idx].source_json)
            ptr += batch_size
            if (
                batch_idx == 1
                or batch_idx == num_batches
                or (args.log_every > 0 and batch_idx % args.log_every == 0)
            ):
                elapsed = time.time() - export_start
                batch_time = elapsed / max(batch_idx, 1)
                eta = batch_time * max(num_batches - batch_idx, 0)
                print(
                    f"  export {batch_idx:04d}/{num_batches:04d} rows={ptr} "
                    f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                    flush=True,
                )

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

