import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import Dataset, Sampler

from graph_dataset import (
    BucketBatchSampler,
    build_cache_from_dirs,
    graph_label_from_semantics,
    load_cache,
)
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
        label = max(0, int(label))
        data.y = torch.tensor([label], dtype=torch.long)
        return data


class NodeBudgetBatchSampler(Sampler[List[int]]):
    """
    Batches graphs by a node budget to avoid CUDA OOM spikes from large graphs.
    """

    def __init__(
        self,
        records,
        indices,
        max_graphs_per_batch,
        max_nodes_per_batch,
        shuffle,
    ):
        self.records = records
        self.indices = list(indices)
        self.max_graphs_per_batch = max(1, int(max_graphs_per_batch))
        self.max_nodes_per_batch = max(1, int(max_nodes_per_batch))
        self.shuffle = bool(shuffle)

    def _build_batches(self, order: List[int]) -> List[List[int]]:
        batches: List[List[int]] = []
        current: List[int] = []
        current_nodes = 0
        for idx in order:
            nodes = max(1, int(self.records[idx].num_nodes))
            # Always keep single oversized graphs in their own batch.
            if nodes >= self.max_nodes_per_batch:
                if current:
                    batches.append(current)
                    current = []
                    current_nodes = 0
                batches.append([idx])
                continue

            would_exceed_graphs = len(current) >= self.max_graphs_per_batch
            would_exceed_nodes = (current_nodes + nodes) > self.max_nodes_per_batch
            if current and (would_exceed_graphs or would_exceed_nodes):
                batches.append(current)
                current = [idx]
                current_nodes = nodes
            else:
                current.append(idx)
                current_nodes += nodes

        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterable[List[int]]:
        ordered = sorted(self.indices, key=lambda idx: self.records[idx].num_nodes)
        if not ordered:
            return

        batches = self._build_batches(ordered)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if not self.indices:
            return 0
        ordered = sorted(self.indices, key=lambda idx: self.records[idx].num_nodes)
        return len(self._build_batches(ordered))


def make_loader(
    records,
    subset_indices,
    batch_size,
    max_nodes_per_batch,
    shuffle,
    num_workers,
    use_cuda,
):
    dataset = CachedGraphDataset(records)
    if max_nodes_per_batch > 0:
        sampler = NodeBudgetBatchSampler(
            records,
            subset_indices,
            max_graphs_per_batch=batch_size,
            max_nodes_per_batch=max_nodes_per_batch,
            shuffle=shuffle,
        )
    else:
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
            try:
                batch = batch.to(device, non_blocking=use_cuda)
                with amp.autocast(device_type="cuda", enabled=use_cuda):
                    logits = classifier(encoder(batch))
                    loss = criterion(logits, batch.y)
                total_loss += loss.item() * int(batch.num_graphs)
                preds = logits.argmax(dim=-1)
                correct += int((preds == batch.y).sum().item())
                total += int(batch.num_graphs)
            except torch.OutOfMemoryError as oom:
                if use_cuda:
                    torch.cuda.empty_cache()
                print(
                    f"  OOM during eval. Skipping batch. Details: {oom}",
                    flush=True,
                )
                continue
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
    parser.add_argument("--test-dir", default="../test")
    parser.add_argument("--cache-path", default="artifacts/cache/graph_cache_train_test.pt")
    parser.add_argument("--cache-stats-path", default="artifacts/cache/cache_stats.json")
    parser.add_argument("--run-dir", default="artifacts/runs/graph_baseline")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-nodes-per-batch",
        type=int,
        default=3500,
        help=(
            "Node budget per mini-batch. "
            "Set <=0 to disable and fall back to fixed graph-count batching."
        ),
    )
    parser.add_argument("--max-graphs-in-memory", type=int, default=200)
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
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--val-from-train-ratio",
        type=float,
        default=0.2,
        help=(
            "Fraction of records from --train-dir used as validation set; "
            "remainder is used for training"
        ),
    )
    parser.add_argument(
        "--val-from-test-ratio",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for deterministic train/val split inside --train-dir",
    )
    args = parser.parse_args()

    if args.val_from_test_ratio is not None:
        print(
            "--val-from-test-ratio is deprecated. "
            "Use --val-from-train-ratio instead.",
            flush=True,
        )
        args.val_from_train_ratio = float(args.val_from_test_ratio)

    if not 0.0 < args.val_from_train_ratio < 1.0:
        raise ValueError("--val-from-train-ratio must be in (0, 1)")

    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_path)
    if args.rebuild_cache and cache_path.exists():
        cache_path.unlink()

    if args.rebuild_cache or not cache_path.exists():
        print("Building combined PyG cache from train+test...")
        build_cache_from_dirs(
            input_dirs=[args.train_dir, args.test_dir],
            cache_path=args.cache_path,
            stats_path=args.cache_stats_path,
        )

    records, _ = load_cache(args.cache_path)
    if args.max_graphs > 0:
        records = records[:args.max_graphs]

    train_root = Path(args.train_dir).resolve()
    test_root = Path(args.test_dir).resolve()
    train_idx = []
    test_idx = []
    for idx, record in enumerate(records):
        source = Path(record.source_json)
        try:
            resolved = source.resolve()
        except FileNotFoundError:
            resolved = source

        if train_root in resolved.parents:
            train_idx.append(idx)
        elif test_root in resolved.parents:
            test_idx.append(idx)

    if not train_idx:
        raise RuntimeError("No train records found in cache. Check --train-dir and cache content.")
    if not test_idx:
        raise RuntimeError("No test records found in cache. Check --test-dir and cache content.")

    rng = random.Random(args.split_seed)
    shuffled_train = list(train_idx)
    rng.shuffle(shuffled_train)
    val_size = int(len(shuffled_train) * args.val_from_train_ratio)
    val_size = max(1, min(val_size, len(shuffled_train) - 1))

    val_idx = shuffled_train[:val_size]
    train_idx = shuffled_train[val_size:]

    print(
        f"Split train-dir records into train/val: {len(train_idx)}/{len(val_idx)} "
        f"(val_ratio={args.val_from_train_ratio}, seed={args.split_seed})",
        flush=True,
    )
    print(
        f"Using test-dir records only for test: {len(test_idx)}",
        flush=True,
    )

    effective_batch_size = min(args.batch_size, args.max_graphs_in_memory)
    if effective_batch_size != args.batch_size:
        print(
            f"Capping batch size from {args.batch_size} to {effective_batch_size} "
            f"to respect max-graphs-in-memory={args.max_graphs_in_memory}",
            flush=True,
        )

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
        effective_batch_size,
        args.max_nodes_per_batch,
        shuffle=True,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )
    val_loader = make_loader(
        records,
        val_idx,
        effective_batch_size,
        args.max_nodes_per_batch,
        shuffle=False,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )
    test_loader = make_loader(
        records,
        test_idx,
        effective_batch_size,
        args.max_nodes_per_batch,
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
        num_layers=args.num_layers,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    ).to(device)

    observed_labels = [
        int(record.graph_label)
        for record in records
        if hasattr(record, "graph_label") and record.graph_label is not None
    ]
    max_label = max(observed_labels) if observed_labels else 0
    num_classes = max(2, max_label + 1)

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
            try:
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
            except torch.OutOfMemoryError as oom:
                if use_cuda:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                print(
                    f"  OOM at train batch {batch_idx}/{num_batches}. "
                    f"Skipping batch and continuing. Details: {oom}",
                    flush=True,
                )
                continue
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
        "train_dir": str(Path(args.train_dir).resolve()),
        "test_dir": str(Path(args.test_dir).resolve()),
        "val_from_train_ratio": float(args.val_from_train_ratio),
        "split_seed": int(args.split_seed),
        "effective_batch_size": int(effective_batch_size),
        "max_nodes_per_batch": int(args.max_nodes_per_batch),
        "max_graphs_in_memory": int(args.max_graphs_in_memory),
        "num_classes": int(num_classes),
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

