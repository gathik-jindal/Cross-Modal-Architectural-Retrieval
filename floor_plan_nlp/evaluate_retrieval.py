"""
evaluate_retrieval.py
Measures Recall@k and MRR using EXACT floor_plan_id match as ground truth.

Since every query in pairs.json was generated directly from its paired floor plan,
the correct retrieval for query i is the floor plan with id pairs[i]["floor_plan_id"].
This is cleaner and more honest than bedroom-count matching, which was a proxy.

Run (baseline, no training):
    python evaluate_retrieval.py

Run (after alignment training):
    python evaluate_retrieval.py \
        --checkpoint artifacts/runs/alignment/best_alignment_checkpoint.pt \
        --embeddings artifacts/handoff_aligned/embeddings.npy \
        --embedding-index artifacts/handoff_aligned/embedding_index.json \
        --out-json artifacts/eval_results.json
"""
import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from text_encoder import TextEncoder, preprocess_query
from retrieval_index import PlanRetrievalIndex


def _norm_id(fid: str) -> str:
    """Strip .svg suffix for consistent ID comparison."""
    return fid.removesuffix(".svg")


def _get_scale_bucket(item, default=0):
    """Read the canonical scale bucket with support for legacy pairs.json."""
    return item.get("scale_bucket", item.get("bedroom_count", default))


def create_eval_split(pairs, split_path, n_test=200, seed=99):
    """Stratified split by scale bucket."""
    rnd = random.Random(seed)
    strata = defaultdict(list)
    for i, p in enumerate(pairs):
        strata[_get_scale_bucket(p, 0)].append(i)

    test_indices = set()
    per_stratum = max(1, n_test // max(len(strata), 1))
    for key, indices in strata.items():
        test_indices.update(rnd.sample(indices, min(per_stratum, len(indices))))

    all_indices = list(range(len(pairs)))
    rnd.shuffle(all_indices)
    for i in all_indices:
        if len(test_indices) >= n_test:
            break
        test_indices.add(i)

    test_indices = sorted(test_indices)[:n_test]
    train_indices = [i for i in range(len(pairs)) if i not in set(test_indices)]
    split = {"train": train_indices, "test": test_indices, "seed": seed}
    Path(split_path).write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"  Created eval split: {len(train_indices)} train, {len(test_indices)} test")
    return split


def compute_random_baseline(n_index, ks):
    """Random baseline for exact-match: P(hit@k) = k / n_index."""
    return {k: min(k / n_index, 1.0) for k in ks}


def print_qualitative_examples(text_encoder, index, index_id_set, test_pairs, n_examples=5):
    BUCKET_LABELS = {0: "commercial", 1: "small", 2: "medium", 3: "large"}
    print(f"\n{'='*65}")
    print("  Qualitative retrieval examples (exact match eval)")
    print(f"{'='*65}")
    for i, item in enumerate(test_pairs[:n_examples]):
        query = preprocess_query(item["query"])
        correct_id = _norm_id(item["floor_plan_id"])
        bucket = BUCKET_LABELS.get(_get_scale_bucket(item, 0), "?")
        short_query = item["query"][:80] + "..."

        with torch.no_grad():
            q_vec = text_encoder([query])[0].numpy()

        results = index.search(q_vec, top_k=5)
        print(f"\n  Query {i+1} [{bucket}]: \"{short_query}\"")
        print(f"  Correct: {correct_id}")
        for r in results:
            rid = _norm_id(r["floor_plan_id"])
            mark = "✓" if rid == correct_id else "✗"
            print(f"    {mark} Rank {r['rank']}: {rid}  score={r['score']:.4f}")


def evaluate(args):
    BUCKET_LABELS = {0: "commercial", 1: "small (1-50 beds)", 2: "medium (51-150)", 3: "large (151+)"}

    # ── Load text encoder ─────────────────────────────────────────────────
    text_encoder = TextEncoder(output_dim=256, freeze_bert=True)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "text_encoder_full" in ckpt:
            text_encoder.load_state_dict(ckpt["text_encoder_full"])
            print(f"Loaded aligned text encoder from {args.checkpoint}")
        elif "text_encoder_projection" in ckpt:
            text_encoder.projection.load_state_dict(ckpt["text_encoder_projection"])
            print(f"Loaded aligned projection from {args.checkpoint}")
    else:
        print("No checkpoint — evaluating with random-init projection (baseline)")
    text_encoder.eval()

    # ── Load retrieval index ──────────────────────────────────────────────
    index = PlanRetrievalIndex(embeddings_path=args.embeddings, index_path=args.embedding_index)
    n_index = index.embeddings.shape[0]
    print(f"Retrieval index: {n_index} floor plans, dim={index.dim}")
    index_id_set = {_norm_id(row["floor_plan_id"]) for row in index.index_rows}

    # ── Load pairs.json ───────────────────────────────────────────────────
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    print(f"Loaded {len(pairs)} pairs from {args.pairs}")

    covered = sum(1 for p in pairs if _norm_id(p["floor_plan_id"]) in index_id_set)
    print(f"  Pairs with floor_plan_id in index: {covered}/{len(pairs)}")

    bucket_dist = defaultdict(int)
    for p in pairs:
        bucket_dist[_get_scale_bucket(p, 0)] += 1
    print("  Scale bucket distribution:")
    for b in sorted(bucket_dist):
        print(f"    bucket {b} ({BUCKET_LABELS.get(b, '?')}): {bucket_dist[b]}")

    # ── Eval split ────────────────────────────────────────────────────────
    split_path = Path(args.eval_split)
    if args.create_split or not split_path.exists():
        split = create_eval_split(pairs, str(split_path), args.n_test, args.split_seed)
    else:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        print(f"  Loaded existing eval split from {split_path}")

    test_pairs = [pairs[i] for i in split.get("test", [])]
    test_pairs = [p for p in test_pairs if _norm_id(p["floor_plan_id"]) in index_id_set]
    n_test = len(test_pairs)
    print(f"Using {n_test} evaluable held-out pairs\n")

    if n_test == 0:
        print("ERROR: No test pairs found in the embedding index.")
        print("       Regenerate pairs.json then rerun with --create-split.")
        return {}

    # ── Evaluate ──────────────────────────────────────────────────────────
    ks = [1, 5, 10]
    hits_at = defaultdict(int)
    reciprocal_ranks = []
    per_bucket_hits = defaultdict(lambda: defaultdict(int))
    per_bucket_total = defaultdict(int)

    for i, item in enumerate(test_pairs):
        query_text = preprocess_query(item["query"])
        correct_id = _norm_id(item["floor_plan_id"])
        bucket = _get_scale_bucket(item, 0)

        with torch.no_grad():
            q_vec = text_encoder([query_text])[0].numpy()

        results = index.search(q_vec, top_k=max(ks), normalize_query=True)
        retrieved_ids = [_norm_id(r["floor_plan_id"]) for r in results]

        first_hit_rank = next(
            (rank for rank, fid in enumerate(retrieved_ids, 1) if fid == correct_id),
            None
        )

        for k in ks:
            if correct_id in retrieved_ids[:k]:
                hits_at[k] += 1
                per_bucket_hits[bucket][k] += 1

        per_bucket_total[bucket] += 1
        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

        if i < 3:
            scores = [r["score"] for r in results[:5]]
            hit_str = f"rank {first_hit_rank}" if first_hit_rank else "NOT in top-10"
            print(f"  Query {i+1}: correct={correct_id} | found at {hit_str}")
            print(f"    Top-5 scores: {[f'{s:.4f}' for s in scores]}")

    n = len(test_pairs)
    random_baselines = compute_random_baseline(n_index, ks)
    mrr = sum(reciprocal_ranks) / max(n, 1)

    print(f"\n{'='*60}")
    print(f"  Evaluation: exact floor_plan_id match  (n={n})")
    print(f"  Index size: {n_index} floor plans")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Model':>10} {'Random':>10} {'Lift':>10}")
    print(f"{'='*60}")
    for k in ks:
        model_score = hits_at[k] / n
        rand_score = random_baselines[k]
        lift = model_score - rand_score
        sign = '+' if lift >= 0 else ''
        print(f"  Recall@{k:<5} {model_score:>10.3f} {rand_score:>10.3f} {sign}{lift:>9.3f}")
    print(f"  {'MRR':<12} {mrr:>10.3f}")
    print(f"{'='*60}")
    print(f"\n  Random baseline = k/{n_index}. Anything above that is real signal.")

    print(f"\n  Per-scale-bucket Recall@5:")
    for b in sorted(per_bucket_total.keys()):
        total = per_bucket_total[b]
        hits = per_bucket_hits[b].get(5, 0)
        print(f"    {BUCKET_LABELS.get(b, f'bucket {b}')}: {hits}/{total} = {hits/total:.3f}")

    results_dict = {
        "n_queries": n,
        "n_index": n_index,
        "evaluation": "exact_floor_plan_id_match",
        "checkpoint": args.checkpoint or "none (baseline)",
        "metrics": {f"recall@{k}": hits_at[k] / n for k in ks},
        "mrr": mrr,
        "random_baseline": {f"recall@{k}": random_baselines[k] for k in ks},
        "per_bucket_recall5": {
            str(b): per_bucket_hits[b].get(5, 0) / per_bucket_total[b]
            for b in sorted(per_bucket_total.keys())
        },
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
        print(f"\n  Results saved to {out_path}")

    print_qualitative_examples(text_encoder, index, index_id_set, test_pairs)
    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cross-modal retrieval (exact match)")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pairs", default="pairs.json")
    parser.add_argument("--embeddings", default="artifacts/handoff_aligned/embeddings.npy")
    parser.add_argument("--embedding-index", default="artifacts/handoff_aligned/embedding_index.json")
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--eval-split", default="eval_split.json")
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=99)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    evaluate(args)