"""
alignment_trainer.py
Contrastive alignment training (InfoNCE / NT-Xent loss).
Trains TextEncoder.projection and GraphPlanEncoder jointly to share a 256-dim
embedding space.

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

from text_encoder import TextEncoder
from graph_model import GraphPlanEncoder
from pair_dataset import PairedDataset, paired_collate_fn


# ── Loss ─────────────────────────────────────────────────────────────────────


def infonce_loss(
    text_vecs: torch.Tensor, geo_vecs: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (NT-Xent, as used in CLIP).
    text_vecs: [B, 256]  L2-normalised
    geo_vecs:  [B, 256]  L2-normalised
    Returns: scalar loss
    """
    B = text_vecs.shape[0]
    labels = torch.arange(B, device=text_vecs.device)

    # [B, B] cosine similarity matrix with learnable scale (CLIP-style)
    sim = (text_vecs @ geo_vecs.T) * logit_scale.exp().clamp(max=100.0)

    # Symmetric: both text→geo and geo→text directions
    loss_t2g = F.cross_entropy(sim, labels)
    loss_g2t = F.cross_entropy(sim.T, labels)
    return (loss_t2g + loss_g2t) / 2


# ── Device ───────────────────────────────────────────────────────────────────


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Main training function ────────────────────────────────────────────────────


def train(args):
    device = pick_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        # Slightly improves throughput/memory behavior on RTX cards.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Load models ───────────────────────────────────────────────────────

    # Person 3's text encoder
    text_encoder = TextEncoder(output_dim=256, freeze_bert=True).to(device)
    print(
        f"TextEncoder loaded. Trainable params (projection only): "
        f"{sum(p.numel() for p in text_encoder.parameters() if p.requires_grad):,}"
    )

    # Person 2's graph encoder
    checkpoint = torch.load(
        args.graph_checkpoint, map_location=device, weights_only=False
    )
    in_dim = checkpoint["in_dim"]
    config = checkpoint["config"]
    graph_encoder = GraphPlanEncoder(
        in_dim=in_dim,
        hidden_dim=config["hidden_dim"],
        out_dim=256,
        dropout=config["dropout"],
        conv_type=config["conv_type"],
    ).to(device)
    graph_encoder.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"GraphPlanEncoder loaded (in_dim={in_dim}). Trainable params: "
        f"{sum(p.numel() for p in graph_encoder.parameters() if p.requires_grad):,}"
    )

    # ── Dataset and DataLoader ────────────────────────────────────────────

    dataset = PairedDataset(pairs_path=args.pairs, cache_path=args.cache)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=paired_collate_fn,
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

    logit_scale = torch.nn.Parameter(
        torch.log(torch.tensor(1.0 / args.temperature, device=device))
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": text_encoder.projection.parameters(), "lr": args.lr_text},
            {"params": graph_encoder.parameters(), "lr": args.lr_graph},
            {"params": [logit_scale], "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    start_epoch = 1
    unfreeze_applied = False
    grad_accum_steps = max(1, args.grad_accum_steps)

    # Optional resume: load full training state from a last-checkpoint file.
    if args.resume:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume requested but checkpoint not found: {resume_path}"
            )

        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        if "text_encoder_full" in resume_ckpt:
            text_encoder.load_state_dict(resume_ckpt["text_encoder_full"])
        elif "text_encoder_projection" in resume_ckpt:
            text_encoder.projection.load_state_dict(resume_ckpt["text_encoder_projection"])

        if "graph_encoder" in resume_ckpt:
            graph_encoder.load_state_dict(resume_ckpt["graph_encoder"])

        if "logit_scale" in resume_ckpt:
            logit_scale.data.copy_(torch.tensor(resume_ckpt["logit_scale"], device=device))

        # Restore full optimizer/scheduler/scaler state when available.
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in resume_ckpt and device.type == "cuda":
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_val_loss = float(
            resume_ckpt.get("best_val_loss", resume_ckpt.get("val_loss", float("inf")))
        )
        history = list(resume_ckpt.get("history", []))
        unfreeze_applied = bool(resume_ckpt.get("unfreeze_applied", False))

        print(
            f"Resumed training from {resume_path} | "
            f"start_epoch={start_epoch} best_val={best_val_loss:.4f}"
        )

    if start_epoch > args.epochs:
        print(
            f"Checkpoint epoch already >= target epochs "
            f"({start_epoch - 1} >= {args.epochs}). Nothing to train."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # ── Optional: unfreeze BERT last 2 layers ────────────────────────
        if (
            (not unfreeze_applied)
            and epoch == args.unfreeze_bert_epoch
            and args.unfreeze_bert_epoch > 0
        ):
            print(f"Epoch {epoch}: Unfreezing BERT last 2 transformer layers...")
            bert_layers = list(text_encoder.bert.encoder.layer)
            for layer in bert_layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
            optimizer.add_param_group(
                {
                    "params": [
                        p for l in bert_layers[-2:] for p in l.parameters()
                    ],
                    "lr": 1e-5,
                }
            )
            unfreeze_applied = True

        # ── Train epoch ───────────────────────────────────────────────────
        text_encoder.train()
        graph_encoder.train()
        total_loss = 0.0
        total_batches = 0
        oom_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            try:
                graphs = batch["graphs"].to(device)
                queries = batch["queries"]  # list of strings

                # Encode with Mixed Precision (AMP) to halve memory mapping
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    text_vecs = text_encoder(queries)  # [B, 256]
                    geo_vecs = graph_encoder(graphs)  # [B, 256]

                    # Loss
                    loss = infonce_loss(text_vecs, geo_vecs, logit_scale)

                # Gradient accumulation for memory safety while preserving effective batch.
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()

                should_step = (
                    (batch_idx % grad_accum_steps == 0) or (batch_idx == len(train_loader))
                )
                if should_step:
                    # Unscale before clipping max norm
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(text_encoder.parameters()) + list(graph_encoder.parameters()),
                        max_norm=1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()
                total_batches += 1

                if batch_idx % 10 == 0 or batch_idx == 1:
                    print(
                        f"  [{epoch}/{args.epochs}] batch {batch_idx}/{len(train_loader)} "
                        f"loss={loss.item():.4f}"
                    )
            except torch.OutOfMemoryError:
                oom_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if not args.skip_oom_batches:
                    raise
                print(
                    f"  [OOM] skipped train batch {batch_idx}/{len(train_loader)}; "
                    f"continuing..."
                )
                continue

        train_loss = total_loss / max(total_batches, 1)

        # ── Validation ────────────────────────────────────────────────────
        text_encoder.eval()
        graph_encoder.eval()
        val_loss = 0.0
        val_batches = 0
        val_oom_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch["queries"]) < 2:
                    continue  # InfoNCE needs at least 2 pairs
                try:
                    graphs = batch["graphs"].to(device)
                    queries = batch["queries"]
                    text_vecs = text_encoder(queries)
                    geo_vecs = graph_encoder(graphs)
                    loss = infonce_loss(text_vecs, geo_vecs, logit_scale)
                    val_loss += loss.item()
                    val_batches += 1
                except torch.OutOfMemoryError:
                    val_oom_batches += 1
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if not args.skip_oom_batches:
                        raise
                    continue

        val_loss = val_loss / max(val_batches, 1)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        temperature = 1.0 / logit_scale.exp().clamp(max=100.0).item()
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"temp={temperature:.4f} time={epoch_time:.1f}s"
        )
        if oom_batches or val_oom_batches:
            print(
                f"  OOM summary: train_skipped={oom_batches}, val_skipped={val_oom_batches}"
            )

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        # Save resumable checkpoint every epoch.
        torch.save(
            {
                "text_encoder_projection": text_encoder.projection.state_dict(),
                "text_encoder_full": text_encoder.state_dict(),
                "graph_encoder": graph_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "history": history,
                "unfreeze_applied": unfreeze_applied,
                "logit_scale": logit_scale.item(),
                "temperature": temperature,
                "in_dim": in_dim,
                "config": config,
            },
            out_dir / "last_alignment_checkpoint.pt",
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "text_encoder_projection": text_encoder.projection.state_dict(),
                    "text_encoder_full": text_encoder.state_dict(),
                    "graph_encoder": graph_encoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "logit_scale": logit_scale.item(),
                    "temperature": temperature,
                    # Store graph encoder config so we can re-instantiate for export
                    "in_dim": in_dim,
                    "config": config,
                },
                out_dir / "best_alignment_checkpoint.pt",
            )
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save training history
    (out_dir / "alignment_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print("Next steps:")
    print("  1. python evaluate_retrieval.py --checkpoint artifacts/runs/alignment/best_alignment_checkpoint.pt")
    print("  2. python export_plan_embeddings.py --checkpoint-path artifacts/runs/alignment/best_alignment_checkpoint.pt --out-dir artifacts/handoff_aligned")
    print("Resume support:")
    print("  python alignment_trainer.py --resume --resume-checkpoint artifacts/runs/alignment/last_alignment_checkpoint.pt [same args]")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive alignment training")
    parser.add_argument("--pairs", default="pairs.json")
    parser.add_argument(
        "--graph-checkpoint",
        default="artifacts/runs/graph_baseline/best_checkpoint.pt",
    )
    parser.add_argument("--cache", default="artifacts/cache/graph_cache.pt")
    parser.add_argument("--out-dir", default="artifacts/runs/alignment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-text", type=float, default=1e-3)
    parser.add_argument("--lr-graph", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--unfreeze-bert-epoch", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--skip-oom-batches", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        default="artifacts/runs/alignment/last_alignment_checkpoint.pt",
    )
    args = parser.parse_args()
    train(args)
