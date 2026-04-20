import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class PlanRetrievalIndex:
    """
    Lightweight retrieval index for Person 4.

    Uses precomputed, L2-normalized plan embeddings from Person 2.
    Cosine similarity is computed as a dot product.
    """

    def __init__(self, embeddings_path: str, index_path: str):
        self.embeddings_path = Path(embeddings_path)
        self.index_path = Path(index_path)

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        self.embeddings = np.load(self.embeddings_path).astype(np.float32)
        self.index_rows = json.loads(self.index_path.read_text(encoding="utf-8"))

        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape {self.embeddings.shape}")
        if len(self.index_rows) != self.embeddings.shape[0]:
            raise ValueError(
                "Row count mismatch: "
                f"embeddings has {self.embeddings.shape[0]} rows but index has {len(self.index_rows)} rows"
            )

        self.dim = int(self.embeddings.shape[1])
        self.id_to_row = {
            str(item["floor_plan_id"]): int(item["row"])
            for item in self.index_rows
        }

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            raise ValueError("Query embedding norm is zero; cannot normalize")
        return vec / norm

    def search(self, query_embedding: np.ndarray, top_k: int = 10, normalize_query: bool = True) -> List[Dict]:
        if query_embedding.ndim != 1:
            raise ValueError(f"Expected query embedding shape [D], got {query_embedding.shape}")
        if query_embedding.shape[0] != self.dim:
            raise ValueError(f"Query dim mismatch: expected {self.dim}, got {query_embedding.shape[0]}")

        q = query_embedding.astype(np.float32)
        if normalize_query:
            q = self._l2_normalize(q)

        scores = self.embeddings @ q
        top_k = min(int(top_k), int(scores.shape[0]))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for rank, row_idx in enumerate(top_idx, start=1):
            meta = self.index_rows[int(row_idx)]
            results.append(
                {
                    "rank": int(rank),
                    "row": int(row_idx),
                    "floor_plan_id": meta["floor_plan_id"],
                    "source_json": meta["source_json"],
                    "score": float(scores[row_idx]),
                }
            )
        return results

    def query_from_floor_plan_id(self, floor_plan_id: str) -> np.ndarray:
        key = str(floor_plan_id)
        row = self.id_to_row.get(key)
        if row is None and not key.endswith(".svg"):
            row = self.id_to_row.get(f"{key}.svg")
        if row is None:
            raise KeyError(f"floor_plan_id not found in index: {floor_plan_id}")
        return self.embeddings[row]


def _load_query_vector(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"2D query vector must have shape [1, D], got {arr.shape}")
        arr = arr[0]
    return arr.astype(np.float32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Person 4 retrieval index using Person 2 embeddings")
    parser.add_argument("--embeddings-path", default="artifacts/handoff/embeddings.npy")
    parser.add_argument("--index-path", default="artifacts/handoff/embedding_index.json")
    parser.add_argument("--top-k", type=int, default=10)

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query-vector-path",
        help="Path to .npy query embedding. Shape [256] or [1,256].",
    )
    query_group.add_argument(
        "--query-floor-plan-id",
        help="Use an existing plan embedding as query for self-retrieval sanity check.",
    )
    query_group.add_argument(
        "--random-query",
        action="store_true",
        help="Use a random vector query for smoke testing pipeline wiring.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize-query", action="store_true")
    parser.add_argument("--out-json", default="")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    index = PlanRetrievalIndex(args.embeddings_path, args.index_path)

    if args.query_vector_path:
        query = _load_query_vector(args.query_vector_path)
        query_mode = f"query-vector:{args.query_vector_path}"
    elif args.query_floor_plan_id:
        query = index.query_from_floor_plan_id(args.query_floor_plan_id)
        query_mode = f"query-floor-plan-id:{args.query_floor_plan_id}"
    else:
        rng = np.random.default_rng(args.seed)
        query = rng.normal(size=(index.dim,)).astype(np.float32)
        query_mode = f"random-query:seed={args.seed}"

    results = index.search(
        query_embedding=query,
        top_k=args.top_k,
        normalize_query=(not args.no_normalize_query),
    )

    payload = {
        "query_mode": query_mode,
        "top_k": int(args.top_k),
        "embedding_dim": int(index.dim),
        "results": results,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved retrieval results -> {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
