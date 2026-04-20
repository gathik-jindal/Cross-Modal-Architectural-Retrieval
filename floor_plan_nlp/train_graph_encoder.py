import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import Dataset

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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def graph_label_from_semantics(semantic_ids: torch.Tensor, unknown_value: int = 0) -> int:
    valid = semantic_ids[semantic_ids >= 0]
    if valid.numel() == 0:
        return unknown_value
    values, counts = valid.unique(return_counts=True)
    return int(values[counts.argmax()].item()) + 1  # +1 to keep 0 for unknown


class CachedGraphDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if record.data is not None:
            data = record.data
        elif record.graph_path:
            data = torch.load(record.graph_path, map_location="cpu", weights_only=False)
        else:
            raise RuntimeError("Record has neither in-memory data nor graph_path")

        label = record.graph_label
        if label is None:
            label = graph_label_from_semantics(data.semantic_ids)
        data.y = torch.tensor([int(label)], dtype=torch.long)
        return data


def make_loader(records, subset_indices, batch_size, shuffle, num_workers, use_cuda):
    dataset = CachedGraphDataset(records)
    sampler = BucketBatchSampler(records, subset_indices, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )


def evaluate(encoder, classifier, loader, device, criterion, use_cuda):
    encoder.eval()
    classifier.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=use_cuda)
            with amp.autocast(device_type="cuda", enabled=use_cuda):
                logits = classifier(encoder(batch))
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


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def main():
    parser = argparse.ArgumentParser(description="Train Person 2 GNN baseline")
    parser.add_argument("--train-dir", default="../train")
    parser.add_argument("--cache-path", default="artifacts/cache/graph_cache.pt")
    parser.add_argument("--cache-stats-path", default="artifacts/cache/cache_stats.json")
    parser.add_argument("--split-path", default="artifacts/splits/train_val_test_split.json")
    parser.add_argument("--run-dir", default="artifacts/runs/graph_baseline")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--conv-type", choices=["sage", "gcn"], default="sage")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
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

    split = create_or_load_split(records, args.split_path, seed=args.seed)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    if args.max_graphs > 0:
        keep = set(range(len(records)))
        train_idx = [i for i in train_idx if i in keep]
        val_idx = [i for i in val_idx if i in keep]
        test_idx = [i for i in test_idx if i in keep]

    device = pick_device()
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"Device: {device} ({gpu_name}) | AMP=on | workers={args.num_workers}")
    else:
        print(f"Device: {device} | AMP=off | workers={args.num_workers}")

    train_loader = make_loader(
        records,
        train_idx,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )
    val_loader = make_loader(
        records,
        val_idx,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )
    test_loader = make_loader(
        records,
        test_idx,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )

    first_data = records[0].data if records[0].data is not None else torch.load(
        records[0].graph_path,
        map_location="cpu",
        weights_only=False,
    )
    in_dim = first_data.x.shape[1]
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
    scaler = amp.GradScaler("cuda", enabled=use_cuda)

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
        num_batches = len(train_loader)
        print(f"Epoch {epoch:02d}/{args.epochs} started | train_batches={num_batches}")

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=use_cuda):
                embedding = model(batch)
                logits = classifier(embedding)
                loss = criterion(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * int(batch.num_graphs)
            total_graphs += int(batch.num_graphs)
            if (
                batch_idx == 1
                or batch_idx == num_batches
                or (args.log_every > 0 and batch_idx % args.log_every == 0)
            ):
                avg_loss = running_loss / max(total_graphs, 1)
                elapsed = time.time() - epoch_start
                batch_time = elapsed / max(batch_idx, 1)
                eta = batch_time * max(num_batches - batch_idx, 0)
                print(
                    f"  train {batch_idx:04d}/{num_batches:04d} "
                    f"avg_loss={avg_loss:.4f} "
                    f"elapsed={format_seconds(elapsed)} "
                    f"eta={format_seconds(eta)}",
                    flush=True,
                )

        train_loss = running_loss / max(total_graphs, 1)
        val_loss, val_acc = evaluate(
            encoder=model,
            classifier=classifier,
            loader=val_loader,
            device=device,
            criterion=criterion,
            use_cuda=use_cuda,
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
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    test_loss, test_acc = evaluate(
        encoder=model,
        classifier=classifier,
        loader=test_loader,
        device=device,
        criterion=criterion,
        use_cuda=use_cuda,
    )
    total_seconds = time.time() - total_start

    epoch1 = history[0]["epoch_seconds"] if history else None
    projected_20 = (epoch1 * 20.0) if epoch1 is not None else None
    report = {
        "device": str(device),
        "mixed_precision": bool(use_cuda),
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

