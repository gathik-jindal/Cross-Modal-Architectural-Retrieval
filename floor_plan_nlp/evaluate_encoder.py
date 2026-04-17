import json
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from text_encoder import TextEncoder, QueryDataset, preprocess_query


# ── 1. Retrieval Simulation ───────────────────────────────────────────────────

def simulate_retrieval(encoder: TextEncoder, dataset: QueryDataset,
                       n_queries: int = 50, top_k: int = 5):
    """
    Simulates what Person 4's retrieval system will do.

    Since we don't have real geometric vectors yet, we use text-to-text
    retrieval as a proxy — if a query about "2 bedrooms" retrieves other
    "2 bedroom" entries in top-k, our encoder is working correctly.

    This is a standard trick in cross-modal retrieval papers when one
    modality isn't ready yet.
    """
    encoder.eval()

    # Encode a small pool to use as the "database"
    pool_size = min(200, len(dataset))
    pool_queries = [dataset[i]["query"] for i in range(pool_size)]
    pool_ids     = [dataset[i]["floor_plan_id"] for i in range(pool_size)]

    print(f"Encoding pool of {pool_size} floor plans...")
    with torch.no_grad():
        pool_embeddings = encoder(pool_queries)   # [pool_size, 256]

    # Load full metadata for bedroom count matching
    raw_data = json.loads(Path("text_queries.json").read_text())
    id_to_meta = {d["floor_plan_id"]: d for d in raw_data}

    # Run retrieval for n_queries test queries
    print(f"Running retrieval for {n_queries} test queries...\n")

    correct_bedroom_match = 0
    total = 0

    for i in range(n_queries):
        item = dataset[pool_size + i]   # use queries outside the pool
        query_text = item["query"]
        query_id   = item["floor_plan_id"]

        with torch.no_grad():
            q_vec = encoder([query_text])             # [1, 256]

        # Cosine similarity against entire pool
        sims = (q_vec @ pool_embeddings.T).squeeze()  # [pool_size]
        top_k_idx = sims.topk(top_k).indices.tolist()

        # Check: do retrieved results share the same bedroom count?
        query_beds = id_to_meta.get(query_id, {}).get(
            "layout_summary", {}).get("bedrooms", -1)

        retrieved_ids = [pool_ids[j] for j in top_k_idx]
        retrieved_beds = [
            id_to_meta.get(rid, {}).get(
                "layout_summary", {}).get("bedrooms", -1)
            for rid in retrieved_ids
        ]

        match = sum(1 for b in retrieved_beds if b == query_beds)
        correct_bedroom_match += match
        total += top_k

        if i < 5:   # print first 5 for inspection
            print(f"Query [{query_id}] ({query_beds} beds):")
            print(f"  {query_text[:80]}...")
            print(f"  Top-{top_k} retrieved bedroom counts: {retrieved_beds}")
            sim_scores = [round(sims[j].item(), 3) for j in top_k_idx]
            print(f"  Similarity scores: {sim_scores}\n")

    bedroom_accuracy = correct_bedroom_match / total
    print(f"Bedroom-count match in top-{top_k}: "
          f"{correct_bedroom_match}/{total} = {bedroom_accuracy:.1%}")
    print()
    if bedroom_accuracy > 0.5:
        print("[GOOD] Encoder is clustering by bedroom count before training.")
    else:
        print("[OK]   Low accuracy expected before alignment training.")
        print("       This number should rise significantly after Person 4")
        print("       runs contrastive fine-tuning.")

    return bedroom_accuracy


# ── 2. Template Sensitivity Check ────────────────────────────────────────────

def check_template_sensitivity(encoder: TextEncoder):
    """
    Verifies the encoder behaves sensibly across your 6 template types.
    Groups of 3 queries that should be similar are compared against
    a clearly different query.
    """
    print("\n── Template Sensitivity ─────────────────────────────────")

    groups = {
        "bedroom_count": [
            "a 1 bedroom apartment with a bathroom",
            "single bedroom unit with attached toilet",
            "studio apartment with one sleeping room",
        ],
        "commercial": [
            "a commercial office layout with open workspace",
            "office floor plan with meeting rooms",
            "commercial building with cubicle arrangement",
        ],
        "parking": [
            "underground parking lot with 50 spaces",
            "basement parking area for vehicles",
            "multi-level parking structure below the building",
        ]
    }

    outlier = "a children's bedroom with a wardrobe and bay window"

    encoder.eval()
    with torch.no_grad():
        outlier_vec = encoder([outlier])

        for group_name, queries in groups.items():
            vecs = encoder(queries)   # [3, 256]

            # intra-group similarity (should be high)
            intra = (vecs @ vecs.T)
            intra_mean = (intra.sum() - intra.trace()) / 6   # off-diagonal mean

            # cross-group similarity (should be lower)
            cross = (vecs @ outlier_vec.T).mean().item()

            print(f"{group_name:<20} "
                  f"intra-sim: {intra_mean:.3f}   "
                  f"vs outlier: {cross:.3f}   "
                  f"{'[OK]' if intra_mean > cross else '[WEAK]'}")

    print()


# ── 3. Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading encoder and dataset...")
    encoder = TextEncoder(output_dim=256, freeze_bert=True)
    dataset = QueryDataset("text_queries.json")
    print(f"Dataset size: {len(dataset)} queries\n")

    simulate_retrieval(encoder, dataset, n_queries=50, top_k=5)
    check_template_sensitivity(encoder)

    print("Step 4 complete.")
    print("Share text_encoder.py + mock_text_embeddings.pt with Person 4.")