"""
clip_baseline.py  —  Zero-shot CLIP retrieval baseline
=======================================================
Layout
------
  project/
    src/
      clip_baseline.py   ← this file
      pairs.json
    data/
      train/
        0000-0002.png
        0000-0003.png
        ...

Usage
-----
  # Full evaluation (encodes everything, caches embeddings)
  python clip_baseline.py

  # Use a pre-existing cache (skip re-encoding)
  python clip_baseline.py --cache embeddings_cache.pt

  # Evaluate on a random subset (fast sanity-check)
  python clip_baseline.py --subset 200

  # Change CLIP backbone
  python clip_baseline.py --model ViT-L/14
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
# Paths  (all relative to this script's location)
# ──────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
PAIRS_JSON = SRC_DIR / "pairs.json"
IMAGE_DIR = SRC_DIR.parent / "data" / "train"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_pairs(pairs_path: Path, subset: int | None = None) -> list[dict]:
    """Load pairs.json and optionally take a random subset."""
    with open(pairs_path) as f:
        pairs = json.load(f)

    # Filter to entries whose PNG actually exists on disk
    valid = []
    missing = 0
    for p in pairs:
        img_path = IMAGE_DIR / f"{p['floor_plan_id']}.png"
        if img_path.exists():
            p["image_path"] = img_path
            valid.append(p)
        else:
            missing += 1

    print(
        f"[pairs]  loaded {len(valid)} valid pairs  ({missing} missing PNGs skipped)")

    if subset and subset < len(valid):
        import random
        random.seed(42)
        valid = random.sample(valid, subset)
        print(f"[pairs]  using random subset of {subset}")

    return valid


def encode_images(
    pairs: list[dict],
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Return L2-normalised image embeddings, shape (N, D)."""
    all_feats = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Encoding images"):
        batch = pairs[i: i + batch_size]
        imgs = torch.stack([
            preprocess(Image.open(p["image_path"]).convert("RGB"))
            for p in batch
        ]).to(device)
        with torch.no_grad():
            feats = model.encode_image(imgs).float()
        all_feats.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(all_feats, dim=0)   # (N, D)


def encode_texts(
    pairs: list[dict],
    model,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Return L2-normalised text embeddings, shape (N, D)."""
    all_feats = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Encoding texts "):
        batch = pairs[i: i + batch_size]
        tokens = clip.tokenize(
            [p["query"] for p in batch], truncate=True
        ).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens).float()
        all_feats.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(all_feats, dim=0)   # (N, D)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_retrieval_metrics(
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """
    Text-to-image retrieval metrics.
    Ground truth: query i should retrieve image i (diagonal).

    Returns
    -------
    dict with Recall@k for each k and MRR.
    """
    N = text_embs.shape[0]

    # Cosine similarity matrix  (N_text × N_images)
    # Both are already L2-normalised, so dot product == cosine similarity.
    sim = text_embs @ image_embs.T           # (N, N)
    # (N, N)  — col idx sorted by sim
    ranks = sim.argsort(dim=1, descending=True)

    # For each query i, find the rank of the correct image i
    correct = torch.arange(N).unsqueeze(1)   # (N, 1)
    rank_of_correct = (ranks == correct).nonzero(
        as_tuple=False)[:, 1]  # (N,)  0-indexed

    metrics: dict[str, float] = {}
    for k in ks:
        recall_at_k = (rank_of_correct < k).float().mean().item()
        metrics[f"R@{k}"] = round(recall_at_k * 100, 2)

    mrr = (1.0 / (rank_of_correct.float() + 1)).mean().item()
    metrics["MRR"] = round(mrr * 100, 2)

    median_rank = rank_of_correct.float().median().item() + 1   # 1-indexed
    metrics["MedianRank"] = round(median_rank, 1)

    return metrics


def compute_image_to_text_metrics(
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Image-to-text retrieval (transpose of the above)."""
    return compute_retrieval_metrics(image_embs, text_embs, ks)


# ──────────────────────────────────────────────
# Qualitative: top-k text queries for a given image index
# ──────────────────────────────────────────────

def show_topk_text(
    query_idx: int,
    pairs: list[dict],
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    k: int = 5,
) -> None:
    """Print the top-k retrieved images for a single text query."""
    q_vec = text_embs[query_idx].unsqueeze(0)   # (1, D)
    sims = (q_vec @ image_embs.T).squeeze(0)   # (N,)
    top_indices = sims.argsort(descending=True)[:k].tolist()

    query = pairs[query_idx]["query"]
    correct_id = pairs[query_idx]["floor_plan_id"]
    print(f"\n{'─'*70}")
    print(f"Query [{query_idx}]: {query[:120]}...")
    print(f"Correct floor-plan ID: {correct_id}")
    print(f"Top-{k} retrieved:")
    for rank, idx in enumerate(top_indices):
        marker = "✓" if idx == query_idx else " "
        score = sims[idx].item()
        print(
            f"  {rank+1}. [{marker}] {pairs[idx]['floor_plan_id']}  sim={score:.4f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP zero-shot retrieval baseline")
    parser.add_argument("--model",   default="ViT-B/32",
                        help="CLIP backbone (default: ViT-B/32)")
    parser.add_argument("--subset",  type=int, default=None,
                        help="Evaluate on a random subset of N pairs")
    parser.add_argument("--cache",   type=str, default=None,
                        help="Path to a cached .pt file (saves / loads embeddings)")
    parser.add_argument("--batch",   type=int, default=64,
                        help="Image encoding batch size")
    parser.add_argument("--qualitative", type=int, default=5,
                        help="Number of qualitative examples to print (0 to skip)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # ── Load pairs ──────────────────────────────
    pairs = load_pairs(PAIRS_JSON, subset=args.subset)
    N = len(pairs)

    # ── Load / compute embeddings ────────────────
    cache_path = Path(args.cache) if args.cache else None

    if cache_path and cache_path.exists():
        print(f"[cache]  loading embeddings from {cache_path}")
        saved = torch.load(cache_path, map_location="cpu")
        image_embs = saved["image_embs"]
        text_embs = saved["text_embs"]
        assert image_embs.shape[0] == N == text_embs.shape[0], (
            "Cache size mismatch — delete the cache and re-run."
        )
    else:
        print(f"[model]  loading CLIP {args.model} …")
        model, preprocess = clip.load(args.model, device=device)
        model.eval()

        t0 = time.time()
        image_embs = encode_images(
            pairs, model, preprocess, device, batch_size=args.batch)
        text_embs = encode_texts(pairs, model, device)
        print(f"[timing] encoding took {time.time() - t0:.1f}s")

        if cache_path:
            torch.save({"image_embs": image_embs,
                       "text_embs": text_embs}, cache_path)
            print(f"[cache]  saved to {cache_path}")

    # ── Metrics ─────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  CLIP zero-shot retrieval  |  N={N}")
    print(f"{'═'*50}")

    t2i = compute_retrieval_metrics(text_embs, image_embs)
    print("\n  Text → Image retrieval")
    for k, v in t2i.items():
        unit = "%" if k != "MedianRank" else ""
        print(f"    {k:<12} {v}{unit}")

    i2t = compute_image_to_text_metrics(text_embs, image_embs)
    print("\n  Image → Text retrieval")
    for k, v in i2t.items():
        unit = "%" if k != "MedianRank" else ""
        print(f"    {k:<12} {v}{unit}")

    print(f"\n{'═'*50}")
    print("  Interpretation guide:")
    print("    Random R@1  ≈ {:.2f}%  (1/{})".format(100 / N, N))
    print("    Random MRR  ≈ {:.2f}%".format(
        100 * sum(1/i for i in range(1, N+1)) / N))
    print(f"{'═'*50}\n")

    # ── Qualitative examples ─────────────────────
    if args.qualitative > 0:
        import random
        random.seed(0)
        sample_indices = random.sample(range(N), min(args.qualitative, N))
        print(f"[qualitative]  {len(sample_indices)} random query examples")
        for idx in sample_indices:
            show_topk_text(idx, pairs, text_embs, image_embs, k=5)
        print()


if __name__ == "__main__":
    main()
