"""
clip_baseline.py  —  Zero-shot & Fine-tuned CLIP retrieval baseline
====================================================================
Layout
------
  project/
    floor_plan_nlp/
      train_pairs.json
      test_pairs.json
    src/
      clip_baseline.py   ← this file
    data/
      train/
        0000-0002.png  ...
      test/
        XXXX-XXXX.png  ...

Usage
-----
  # Zero-shot CLIP evaluated on test (default)
  python clip_baseline.py --mode zero-shot

  # Fine-tune on train, then evaluate on test
  python clip_baseline.py --mode fine-tune

  # Run both modes and print a side-by-side comparison
  python clip_baseline.py --mode both

  # Fine-tune options
  python clip_baseline.py --mode fine-tune --epochs 10 --batch 64 --lr 5e-6

  # Cache zero-shot embeddings (skips re-encoding on later runs)
  python clip_baseline.py --mode zero-shot --cache embeddings

  # Save / load the fine-tuned checkpoint
  python clip_baseline.py --mode fine-tune --checkpoint finetuned_clip.pt

  # Fast sanity-check on a small subset
  python clip_baseline.py --mode both --subset 200
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SRC_DIR = Path(__file__).parent

IMAGE_DIR_TRAIN = SRC_DIR.parent / "data" / "train"
IMAGE_DIR_TEST  = SRC_DIR.parent / "data" / "test"

TRAIN_FILE = SRC_DIR.parent / "floor_plan_nlp" / "train_pairs.json"
TEST_FILE  = SRC_DIR.parent / "floor_plan_nlp" / "test_pairs.json"


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_pairs(
    pairs_path: Path,
    image_dir: Path,
    subset: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load a pairs JSON file and attach resolved image paths.
    Entries whose PNG is missing on disk are silently skipped.
    """
    with open(pairs_path) as f:
        pairs = json.load(f)

    valid, missing = [], 0
    for p in pairs:
        img_path = image_dir / f"{p['floor_plan_id']}.png"
        if img_path.exists():
            p["image_path"] = img_path
            valid.append(p)
        else:
            missing += 1

    print(
        f"[pairs]  {pairs_path.name}: "
        f"{len(valid)} valid pairs  ({missing} missing PNGs skipped)"
    )

    if subset and subset < len(valid):
        random.seed(seed)
        valid = random.sample(valid, subset)
        print(f"[pairs]  using random subset of {subset}")

    return valid


class FloorPlanDataset(Dataset):
    """
    Returns (preprocessed_image_tensor, tokenized_text_tensor) per item.
    The CLIP preprocess transform and tokenizer are applied here so the
    DataLoader can collate them into batches without extra work.
    """
    def __init__(self, pairs: list[dict], preprocess):
        self.pairs      = pairs
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p     = self.pairs[idx]
        image = self.preprocess(Image.open(p["image_path"]).convert("RGB"))
        text  = clip.tokenize([p["query"]], truncate=True).squeeze(0)  # (77,)
        return image, text


# ──────────────────────────────────────────────
# Encoding  (inference-only, no grad)
# ──────────────────────────────────────────────

def encode_images(
    pairs: list[dict],
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Return L2-normalised image embeddings, shape (N, D)."""
    all_feats = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="  images"):
        batch = pairs[i : i + batch_size]
        imgs  = torch.stack([
            preprocess(Image.open(p["image_path"]).convert("RGB"))
            for p in batch
        ]).to(device)
        with torch.no_grad():
            feats = model.encode_image(imgs).float()
        all_feats.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(all_feats, dim=0)


def encode_texts(
    pairs: list[dict],
    model,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Return L2-normalised text embeddings, shape (N, D)."""
    all_feats = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="  texts "):
        batch  = pairs[i : i + batch_size]
        tokens = clip.tokenize(
            [p["query"] for p in batch], truncate=True
        ).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens).float()
        all_feats.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(all_feats, dim=0)


def get_embeddings(
    pairs: list[dict],
    model,
    preprocess,
    device: str,
    batch_size: int,
    cache_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (image_embs, text_embs), loading from cache if available."""
    if cache_path and cache_path.exists():
        print(f"[cache]  loading from {cache_path}")
        saved      = torch.load(cache_path, map_location="cpu")
        image_embs = saved["image_embs"]
        text_embs  = saved["text_embs"]
        N = len(pairs)
        if image_embs.shape[0] != N or text_embs.shape[0] != N:
            raise ValueError(
                f"Cache has {image_embs.shape[0]} entries but pairs has {N}. "
                "Delete the cache and re-run."
            )
        return image_embs, text_embs

    t0         = time.time()
    image_embs = encode_images(pairs, model, preprocess, device, batch_size)
    text_embs  = encode_texts(pairs, model, device)
    print(f"[timing] encoding took {time.time() - t0:.1f}s")

    if cache_path:
        torch.save({"image_embs": image_embs, "text_embs": text_embs}, cache_path)
        print(f"[cache]  saved to {cache_path}")

    return image_embs, text_embs


# ──────────────────────────────────────────────
# Fine-tuning
# ──────────────────────────────────────────────

def fine_tune_clip(
    model,
    train_pairs: list[dict],
    preprocess,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_path: Path | None,
) -> None:
    """
    Fine-tune CLIP on (image, text) pairs using symmetric contrastive loss
    (InfoNCE / CLIP loss).

    The model is modified in-place. If checkpoint_path is provided and the
    file already exists, training is skipped and the checkpoint is loaded instead.
    """
    # ── Load checkpoint if it already exists ──────────────────────────
    if checkpoint_path and checkpoint_path.exists():
        print(f"[finetune]  checkpoint found — loading from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        return

    # CLIP loads weights in FP16 by default; GradScaler requires FP32 parameters.
    # Cast to FP32 before setting up the optimizer so gradients are always FP32.
    model.float()

    print(f"\n[finetune]  training for {epochs} epoch(s) on {len(train_pairs)} pairs")
    print(f"[finetune]  lr={lr}  batch={batch_size}")

    dataset    = FloorPlanDataset(train_pairs, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Very small LR is critical — large LR causes catastrophic forgetting
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Mixed-precision scaler (safe no-op on CPU)
    use_amp = device == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()

    for epoch in range(epochs):
        total_loss  = 0.0
        t0          = time.time()

        for images, texts in tqdm(dataloader, desc=f"  epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            texts  = texts.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                # model() returns (logits_per_image, logits_per_text)
                # logits are already scaled by CLIP's learned temperature
                logits_per_image, logits_per_text = model(images, texts)

                # Ground truth: item i in the batch matches item i (diagonal)
                labels = torch.arange(images.shape[0], device=device)

                loss = (
                    loss_img(logits_per_image, labels) +
                    loss_txt(logits_per_text,  labels)
                ) / 2

            scaler.scale(loss).backward()

            # Clip gradients to avoid exploding gradients during fine-tuning
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        elapsed  = time.time() - t0
        print(f"  epoch {epoch + 1}/{epochs}  |  loss: {avg_loss:.4f}  |  {elapsed:.1f}s")

    model.eval()

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[finetune]  checkpoint saved to {checkpoint_path}")


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def retrieval_metrics(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """
    Text-to-image (or image-to-text) retrieval metrics.
    Ground truth: query i matches gallery item i (diagonal).
    """
    N    = query_embs.shape[0]
    sim  = query_embs @ gallery_embs.T
    ranks = sim.argsort(dim=1, descending=True)

    correct         = torch.arange(N).unsqueeze(1)
    rank_of_correct = (ranks == correct).nonzero(as_tuple=False)[:, 1]

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = round((rank_of_correct < k).float().mean().item() * 100, 2)
    metrics["MRR"]        = round((1.0 / (rank_of_correct.float() + 1)).mean().item() * 100, 2)
    metrics["MedianRank"] = round(rank_of_correct.float().median().item() + 1, 1)

    return metrics


def print_metrics(label: str, t2i: dict, i2t: dict, N: int) -> None:
    print(f"\n{'═' * 54}")
    print(f"  {label}  |  N = {N}")
    print(f"{'═' * 54}")
    for direction, m in [("Text → Image", t2i), ("Image → Text", i2t)]:
        print(f"\n  {direction}")
        for key, val in m.items():
            unit = "%" if key != "MedianRank" else ""
            print(f"    {key:<14} {val}{unit}")
    print(f"\n  — Random baseline —")
    print(f"    R@1            {100/N:.3f}%  (1/{N})")
    print(f"    MRR            {100 * sum(1/i for i in range(1,N+1))/N:.3f}%")
    print(f"{'═' * 54}")


def print_comparison(zero_shot: dict, fine_tuned: dict) -> None:
    """Side-by-side comparison of zero-shot vs fine-tuned metrics."""
    print(f"\n{'═' * 62}")
    print(f"  Comparison: Zero-shot vs Fine-tuned  (Text → Image)")
    print(f"{'═' * 62}")
    print(f"  {'Metric':<14} {'Zero-shot':>12} {'Fine-tuned':>12} {'Δ':>8}")
    print(f"  {'─'*14} {'─'*12} {'─'*12} {'─'*8}")
    for key in zero_shot:
        zs  = zero_shot[key]
        ft  = fine_tuned[key]
        delta = ft - zs
        unit  = "%" if key != "MedianRank" else ""
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {key:<14} {str(zs)+unit:>12} {str(ft)+unit:>12} {arrow}{abs(delta):.2f}{unit:>4}")
    print(f"{'═' * 62}\n")


# ──────────────────────────────────────────────
# Qualitative inspection
# ──────────────────────────────────────────────

def show_qualitative(
    pairs: list[dict],
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    n_examples: int,
    k: int = 5,
) -> None:
    random.seed(0)
    indices = random.sample(range(len(pairs)), min(n_examples, len(pairs)))
    print(f"\n[qualitative]  {len(indices)} random examples  (top-{k})")

    for query_idx in indices:
        q_vec       = text_embs[query_idx].unsqueeze(0)
        sims        = (q_vec @ image_embs.T).squeeze(0)
        top_indices = sims.argsort(descending=True)[:k].tolist()

        query      = pairs[query_idx]["query"]
        correct_id = pairs[query_idx]["floor_plan_id"]

        print(f"\n{'─' * 70}")
        print(f"Query [{query_idx}]: {query[:120]}{'...' if len(query) > 120 else ''}")
        print(f"Correct ID: {correct_id}")
        print(f"Top-{k} retrieved:")
        for rank, idx in enumerate(top_indices):
            marker = "✓" if idx == query_idx else " "
            print(f"  {rank+1}. [{marker}] {pairs[idx]['floor_plan_id']}  sim={sims[idx].item():.4f}")

    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP retrieval baseline")
    parser.add_argument("--mode",        default="zero-shot",
                        choices=["zero-shot", "fine-tune", "both"],
                        help="Evaluation mode (default: zero-shot)")
    parser.add_argument("--model",       default="ViT-B/32",
                        help="CLIP backbone (default: ViT-B/32)")
    parser.add_argument("--subset",      type=int, default=None,
                        help="Random subset size per split (for quick tests)")
    parser.add_argument("--batch",       type=int, default=64,
                        help="Batch size for encoding and fine-tuning (default: 64)")
    parser.add_argument("--epochs",      type=int, default=5,
                        help="Fine-tuning epochs (default: 5)")
    parser.add_argument("--lr",          type=float, default=5e-6,
                        help="Fine-tuning learning rate (default: 5e-6)")
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Path to save/load fine-tuned weights (.pt)")
    parser.add_argument("--cache",       type=str, default=None,
                        help="Base name for zero-shot embedding cache "
                             "(e.g. 'emb' → 'emb_zeroshot_test.pt')")
    parser.add_argument("--qualitative", type=int, default=5,
                        help="Qualitative examples to print (0 = skip)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[model]  loading CLIP {args.model} …")
    model, preprocess = clip.load(args.model, device=device)

    # ── Load test pairs (always needed for evaluation) ─────────────────
    test_pairs  = load_pairs(TEST_FILE,  IMAGE_DIR_TEST,  subset=args.subset)
    train_pairs = load_pairs(TRAIN_FILE, IMAGE_DIR_TRAIN, subset=args.subset) \
                  if args.mode in ("fine-tune", "both") else []

    zs_t2i = None  # populated in zero-shot block, reused in comparison

    # ══════════════════════════════════════════
    # ZERO-SHOT evaluation
    # ══════════════════════════════════════════
    if args.mode in ("zero-shot", "both"):
        print("\n[zero-shot]  evaluating on test …")
        model.eval()

        cache_path = Path(f"{args.cache}_zeroshot_test.pt") if args.cache else None
        image_embs, text_embs = get_embeddings(
            test_pairs, model, preprocess, device, args.batch, cache_path
        )

        zs_t2i = retrieval_metrics(text_embs, image_embs)
        zs_i2t = retrieval_metrics(image_embs, text_embs)
        print_metrics("CLIP zero-shot — test", zs_t2i, zs_i2t, len(test_pairs))

        if args.qualitative > 0:
            show_qualitative(test_pairs, text_embs, image_embs, args.qualitative)

    # ══════════════════════════════════════════
    # FINE-TUNED evaluation
    # ══════════════════════════════════════════
    if args.mode in ("fine-tune", "both"):
        # Reload a fresh model so zero-shot weights are not affected
        # (important when mode == "both")
        print("\n[fine-tune]  reloading base model …")
        ft_model, _ = clip.load(args.model, device=device)

        checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

        fine_tune_clip(
            model         = ft_model,
            train_pairs   = train_pairs,
            preprocess    = preprocess,
            device        = device,
            epochs        = args.epochs,
            batch_size    = args.batch,
            lr            = args.lr,
            checkpoint_path = checkpoint_path,
        )

        print("\n[fine-tune]  evaluating fine-tuned model on test …")
        ft_model.eval()

        ft_image_embs, ft_text_embs = get_embeddings(
            test_pairs, ft_model, preprocess, device, args.batch, cache_path=None
        )

        ft_t2i = retrieval_metrics(ft_text_embs,  ft_image_embs)
        ft_i2t = retrieval_metrics(ft_image_embs, ft_text_embs)
        print_metrics("CLIP fine-tuned — test", ft_t2i, ft_i2t, len(test_pairs))

        if args.qualitative > 0:
            show_qualitative(test_pairs, ft_text_embs, ft_image_embs, args.qualitative)

        # Side-by-side comparison (only when both modes ran)
        if args.mode == "both" and zs_t2i is not None:
            print_comparison(zs_t2i, ft_t2i)


if __name__ == "__main__":
    main()