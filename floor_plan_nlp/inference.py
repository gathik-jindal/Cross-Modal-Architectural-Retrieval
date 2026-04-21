"""
inference.py
End-to-end inference: user types a text query, gets back top-k floor plans.

Usage:
    python inference.py --query "2 bedroom flat with balcony and modern kitchen"

Or import and call:
    from inference import retrieve
    results = retrieve("a 3 bedroom house with attached bathrooms")
"""
import argparse
import json
import torch
from pathlib import Path

from text_encoder import TextEncoder, preprocess_query
from retrieval_index import PlanRetrievalIndex


_text_encoder = None
_retrieval_index = None

DEFAULT_TEXT_CHECKPOINT = "artifacts/runs/alignment/best_alignment_checkpoint.pt"
DEFAULT_EMBEDDINGS = "artifacts/handoff_aligned/embeddings.npy"
DEFAULT_EMBEDDING_INDEX = "artifacts/handoff_aligned/embedding_index.json"

# Fallback to pre-alignment embeddings if aligned ones don't exist
FALLBACK_EMBEDDINGS = "artifacts/handoff/embeddings.npy"
FALLBACK_EMBEDDING_INDEX = "artifacts/handoff/embedding_index.json"


def _load_models(text_checkpoint: str, embeddings: str, embedding_index: str):
    global _text_encoder, _retrieval_index

    if _text_encoder is None:
        _text_encoder = TextEncoder(output_dim=256, freeze_bert=True)
        if Path(text_checkpoint).exists():
            ckpt = torch.load(text_checkpoint, map_location="cpu", weights_only=False)
            if "text_encoder_full" in ckpt:
                _text_encoder.load_state_dict(ckpt["text_encoder_full"])
            elif "text_encoder_projection" in ckpt:
                _text_encoder.projection.load_state_dict(
                    ckpt["text_encoder_projection"]
                )
            print(f"Loaded text encoder from {text_checkpoint}")
        else:
            print(f"Warning: checkpoint not found at {text_checkpoint}")
            print("Using untrained text encoder (results will be meaningless)")
        _text_encoder.eval()

    if _retrieval_index is None:
        # Try aligned embeddings first, fall back to pre-alignment
        emb_path = embeddings
        idx_path = embedding_index
        if not Path(emb_path).exists():
            print(f"Aligned embeddings not found at {emb_path}")
            emb_path = FALLBACK_EMBEDDINGS
            idx_path = FALLBACK_EMBEDDING_INDEX
            print(f"Falling back to pre-alignment: {emb_path}")

        _retrieval_index = PlanRetrievalIndex(
            embeddings_path=emb_path,
            index_path=idx_path,
        )
        print(
            f"Loaded retrieval index: {_retrieval_index.embeddings.shape[0]} floor plans"
        )


def retrieve(
    query: str,
    top_k: int = 5,
    text_checkpoint: str = DEFAULT_TEXT_CHECKPOINT,
    embeddings: str = DEFAULT_EMBEDDINGS,
    embedding_index: str = DEFAULT_EMBEDDING_INDEX,
):
    """
    Main function. Takes a user's natural language query,
    returns top_k floor plan results sorted by cosine similarity.

    Returns: list of dicts, each with keys:
        rank, floor_plan_id, source_json, score
    """
    _load_models(text_checkpoint, embeddings, embedding_index)

    # Preprocess and encode
    cleaned = preprocess_query(query)
    with torch.no_grad():
        q_vec = _text_encoder([cleaned])[0].numpy()  # [256]

    # Search
    results = _retrieval_index.search(q_vec, top_k=top_k, normalize_query=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-modal floor plan retrieval"
    )
    parser.add_argument("--query", required=True, type=str, help="Natural language query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--checkpoint", default=DEFAULT_TEXT_CHECKPOINT)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--index", default=DEFAULT_EMBEDDING_INDEX)
    args = parser.parse_args()

    results = retrieve(
        query=args.query,
        top_k=args.top_k,
        text_checkpoint=args.checkpoint,
        embeddings=args.embeddings,
        embedding_index=args.index,
    )

    print(f"\nQuery: {args.query}")
    print(f"Preprocessed: {preprocess_query(args.query)}\n")
    for r in results:
        print(f"  Rank {r['rank']}: {r['floor_plan_id']}  score={r['score']:.4f}")
        print(f"           {r['source_json']}")
