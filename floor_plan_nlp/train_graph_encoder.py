import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from graph_dataset import BucketBatchSampler, build_cache, create_or_load_split, load_cache
from graph_model import GraphPlanEncoder

try:
    from torch_geometric.loader import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for Person 2 training. "
        "Install with: pip install torch-geometric"
    ) from exc


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def graph_label_from_semantics(semantic_ids: torch.Tensor, unknown_value: int = 0) -> int:
    valid = semantic_ids[semantic_ids >= 0]
    if valid.numel() == 0:
        return unknown_value
    values, counts = valid.unique(return_counts=True)
    return int(values[counts.argmax()].item()) + 1  # +1 to keep 0 for unknown


def attach_graph_labels(records):
    for record in records:
        label = graph_label_from_semantics(record.data.semantic_ids)
        record.data.y = torch.tensor([label], dtype=torch.long)


def make_loader(records, subset_indices, batch_size, shuffle):
    dataset = [records[idx].data for idx in subset_indices]
    sampler = BucketBatchSampler(records, subset_indices, batch_size=batch_size, shuffle=shuffle)
    # Pin memory is not useful on MPS and can hurt throughput.
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * int(batch.num_graphs)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch.y).sum().item())
            total += int(batch.num_graphs)
    return total_loss / max(total, 1), correct / max(total, 1)


def mps_memory_bytes():
    if not torch.backends.mps.is_available():
        return None
    try:
        return int(torch.mps.current_allocated_memory())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train Person 2 GNN baseline")
    parser.add_argument("--train-dir", default="../train")
    parser.add_argument("--cache-path", default="artifacts/cache/graph_cache.pt")
    parser.add_argument("--cache-stats-path", default="artifacts/cache/cache_stats.json")
    parser.add_argument("--split-path", default="artifacts/splits/train_val_test_split.json")
    parser.add_argument("--run-dir", default="artifacts/runs/graph_baseline")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--conv-type", choices=["sage", "gcn"], default="sage")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        print("Building PyG cache...")
        build_cache(args.train_dir, args.cache_path, args.cache_stats_path)
    records, _ = load_cache(args.cache_path)
    if args.max_graphs > 0:
        records = records[:args.max_graphs]
    attach_graph_labels(records)

    split = create_or_load_split(records, args.split_path, seed=args.seed)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    if args.max_graphs > 0:
        keep = set(range(len(records)))
        train_idx = [i for i in train_idx if i in keep]
        val_idx = [i for i in val_idx if i in keep]
        test_idx = [i for i in test_idx if i in keep]

    train_loader = make_loader(records, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(records, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(records, test_idx, args.batch_size, shuffle=False)

    device = pick_device()
    in_dim = records[0].data.x.shape[1]
    model = GraphPlanEncoder(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
        conv_type=args.conv_type,
    ).to(device)
    num_classes = 37  # 0 unknown + semantic IDs 0..35
    classifier = nn.Linear(args.out_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[dict] = []
    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        classifier.train()
        epoch_start = time.time()
        running_loss = 0.0
        total_graphs = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            embedding = model(batch)
            logits = classifier(embedding)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * int(batch.num_graphs)
            total_graphs += int(batch.num_graphs)

        train_loss = running_loss / max(total_graphs, 1)
        val_loss, val_acc = evaluate(
            model=lambda b: classifier(model(b)),
            loader=val_loader,
            device=device,
            criterion=criterion,
        )
        epoch_time = time.time() - epoch_start
        samples_per_sec = total_graphs / max(epoch_time, 1e-6)
        mem = mps_memory_bytes()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "epoch_seconds": round(epoch_time, 4),
            "samples_per_sec": round(samples_per_sec, 4),
            "mps_memory_bytes": mem,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train={train_loss:.4f} "
            f"val={val_loss:.4f} val_acc={val_acc:.3f} "
            f"time={epoch_time:.1f}s sps={samples_per_sec:.1f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "config": vars(args),
                    "in_dim": int(in_dim),
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # Test metrics using best checkpoint
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    test_loss, test_acc = evaluate(
        model=lambda b: classifier(model(b)),
        loader=test_loader,
        device=device,
        criterion=criterion,
    )
    total_seconds = time.time() - total_start

    epoch1 = history[0]["epoch_seconds"] if history else None
    projected_20 = (epoch1 * 20.0) if epoch1 is not None else None
    report = {
        "device": str(device),
        "num_graphs": len(records),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "wall_seconds": total_seconds,
        "epoch1_seconds": epoch1,
        "projected_20_epoch_seconds": projected_20,
        "history": history,
        "tuning_recommendation": (
            "If epoch time > 180s on MPS, lower batch size to 4. "
            "If epoch time < 45s and stable memory, try batch size 12."
        ),
    }
    (run_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved run report -> {run_dir / 'train_report.json'}")
    print(f"Saved best checkpoint -> {run_dir / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    main()

