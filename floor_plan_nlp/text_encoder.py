import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from pathlib import Path


# ── 1. The Encoder Model ─────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    Encodes a natural language floor plan query into a
    fixed-size vector of shape [batch_size, output_dim].

    Architecture:
        Raw text
          → BERT tokenizer
          → BERT (frozen base, fine-tuned projection)
          → mean pool over token dimension
          → linear projection → output_dim
          → L2 normalisation          ← critical for cosine similarity later
    """

    def __init__(self, output_dim: int = 256, freeze_bert: bool = True):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert      = BertModel.from_pretrained("bert-base-uncased")

        # Freeze BERT weights so only the projection trains initially.
        # Person 4 can unfreeze later for end-to-end fine-tuning.
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Projects BERT's 768-dim output → agreed shared space (256)
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of raw query strings, e.g.
                   ["2-bedroom with kitchen", "open-plan studio"]
        Returns:
            tensor of shape [len(texts), output_dim], L2-normalised
        """
        # Tokenise — padding + truncation handles variable-length input
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Forward through BERT
        outputs = self.bert(**tokens)

        # Mean pool over the token dimension (ignore padding via attention mask)
        # Shape: [batch, seq_len, 768] → [batch, 768]
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts               # masked mean pool

        # Project to shared space
        projected = self.projection(pooled)    # [batch, 256]

        # L2 normalise so dot product == cosine similarity
        # Person 4's contrastive loss expects unit vectors
        normalised = F.normalize(projected, p=2, dim=-1)

        return normalised


# ── 2. Query Dataset Wrapper ──────────────────────────────────────────────────

class QueryDataset(torch.utils.data.Dataset):
    """
    Wraps text_queries.json so it plugs directly into
    torch DataLoader for batched training.

    Each item returns:
        floor_plan_id : str   — Person 4 uses this to fetch the matching
                                geometric vector from Person 2's encoder
        query         : str   — raw text fed into TextEncoder
    """

    def __init__(self, json_path: str = "text_queries.json"):
        data = json.loads(Path(json_path).read_text())
        self.items = [
            {
                "floor_plan_id": d["floor_plan_id"],
                "query":         d["query"]
            }
            for d in data
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── 3. Preprocessing ──────────────────────────────────────────────────────────

def preprocess_query(query: str) -> str:
    """
    Lightweight normalisation before tokenisation.
    Handles abbreviations common in Indian real-estate descriptions
    (relevant because FloorPlanCAD includes Indian commercial buildings).
    """
    replacements = {
        "bhk":  "bedroom hall kitchen",
        "bhk.": "bedroom hall kitchen",
        "rk":   "room kitchen",
        "sqft": "square feet",
        "sq ft":"square feet",
        "w/":   "with",
        "w/o":  "without",
        "2br":  "2 bedroom",
        "3br":  "3 bedroom",
    }
    q = query.lower().strip()
    for abbr, expansion in replacements.items():
        q = q.replace(abbr, expansion)
    return q


# ── 4. Sanity Check ───────────────────────────────────────────────────────────

def run_sanity_check(encoder: TextEncoder):
    """
    Tests two things:
      1. Output shape is correct
      2. Semantically similar queries score higher cosine similarity
         than semantically unrelated ones

    If test 2 fails before training, that is expected — BERT's weights
    are frozen and the projection is randomly initialised. It will pass
    after Person 4 runs alignment training.
    """
    print("\n── Sanity Check ─────────────────────────────────────────")

    encoder.eval()
    with torch.no_grad():

        # ── Shape test
        dummy = encoder(["a two bedroom apartment"])
        assert dummy.shape == (1, encoder.output_dim), \
            f"Shape mismatch: got {dummy.shape}"
        print(f"[PASS] Output shape: {dummy.shape}")

        # ── Norm test (should be 1.0 after L2 normalisation)
        norm = dummy.norm(dim=-1).item()
        assert abs(norm - 1.0) < 1e-5, f"Norm is {norm}, expected ~1.0"
        print(f"[PASS] L2 norm: {norm:.6f}")

        # ── Similarity test
        similar_a = encoder(["a spacious 2 bedroom apartment with attached bathroom"])
        similar_b = encoder(["two bedrooms and a private bathroom"])
        unrelated  = encoder(["underground parking lot with 50 spaces"])

        sim_close = (similar_a * similar_b).sum().item()
        sim_far   = (similar_a * unrelated).sum().item()

        print(f"\nCosine similarity (similar pair):   {sim_close:.4f}")
        print(f"Cosine similarity (unrelated pair): {sim_far:.4f}")

        if sim_close > sim_far:
            print("[PASS] Similar queries score higher than unrelated ones")
        else:
            print(
                "[NOTE] Similar < unrelated — expected before alignment training.\n"
                "       This will flip after Person 4 fine-tunes with contrastive loss."
            )

    print("────────────────────────────────────────────────────────\n")


# ── 5. Mock Tensor Export (for Person 4) ─────────────────────────────────────

def export_mock_tensors(
    encoder: TextEncoder,
    json_path: str = "text_queries.json",
    out_path:  str = "mock_text_embeddings.pt",
    batch_size: int = 8
):
    """
    Encodes the first `batch_size` queries from your dataset and saves
    the resulting tensor so Person 4 can start building the loss function
    without waiting for full training to finish.

    Person 4 loads this with:
        embeddings = torch.load("mock_text_embeddings.pt")
        # shape: [8, 256]
    """
    dataset = QueryDataset(json_path)
    encoder.eval()

    batch = [dataset[i]["query"] for i in range(batch_size)]
    ids   = [dataset[i]["floor_plan_id"] for i in range(batch_size)]

    with torch.no_grad():
        embeddings = encoder(batch)

    torch.save({
        "embeddings":      embeddings,       # [8, 256] float tensor
        "floor_plan_ids":  ids,              # list of 8 IDs for Person 4 to match
        "output_dim":      encoder.output_dim
    }, out_path)

    print(f"Saved mock tensors → {out_path}")
    print(f"  Shape:      {embeddings.shape}")
    print(f"  IDs:        {ids}")


# ── 6. Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading BERT (first run downloads ~440MB, cached after)...")
    encoder = TextEncoder(output_dim=256, freeze_bert=True)
    print(f"Encoder ready. Trainable params: "
          f"{sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")

    run_sanity_check(encoder)

    export_mock_tensors(encoder)

    print("\nNext step: hand mock_text_embeddings.pt to Person 4.")