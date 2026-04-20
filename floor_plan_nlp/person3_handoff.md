# Person 3 → Person 4 Handoff Document
**Natural Language Pipeline — Handoff to Alignment & Retrieval**

**Author:** Akshat Betala (IMT2023019) — Person 3  
**Handoff to:** Person 4 (The Bridge Builder)  
**Project:** Bridging Text and Geometry — Cross-Modal Architectural Floor Plan Retrieval

---

## 1. Summary

Person 3's natural language pipeline is complete. The text encoder takes any raw
natural language query from a client and converts it into a **256-dimensional
L2-normalised vector** that lives in the same embedding space as Person 2's geometric
vectors. Person 4's job is to align these two spaces via contrastive training and
build the final retrieval system on top.

---

## 2. Deliverables

| File | Description | Who uses it |
|------|-------------|-------------|
| `text_encoder.py` | The `TextEncoder` model class + `preprocess_query()` | Person 4 imports this for training and inference |
| `text_queries.json` | 2000 synthetic NL queries with rich metadata | Person 4 uses as the text side of training pairs |
| `mock_text_embeddings.pt` | Pre-encoded tensor `[8, 256]` from the untrained encoder | Person 4 uses to build and test the loss function immediately |(this one is Garv ka embeddings not mine)
| `baseline_retrieval_results.json` | Pre-training retrieval scores against Person 2's embeddings | Person 4 uses to verify improvement after contrastive training |
| `query_demo.py` | Interactive CLI demo — type a query, get top-5 floor plans | Integration testing and final demo |

---

## 3. Architecture

```
Raw client query (string)
        │
        ▼
preprocess_query()          — normalises abbreviations (bhk, sqft, w/o etc.)
        │
        ▼
BertTokenizer               — tokenises to max 128 tokens with padding/truncation
        │
        ▼
BertModel (frozen)          — bert-base-uncased, weights frozen during initial training
        │
        ▼
Masked mean pooling         — pools over non-padding tokens → [batch, 768]
        │
        ▼
Projection head             — Linear(768→512) → ReLU → Dropout(0.1) → Linear(512→256)
        │
        ▼
L2 normalisation            — F.normalize(p=2, dim=-1)
        │
        ▼
Output: [batch_size, 256]   — unit vector, dot product = cosine similarity
```

---

## 4. Integration — How to Import and Use

```python
from text_encoder import TextEncoder, preprocess_query

# Load encoder
encoder = TextEncoder(output_dim=256, freeze_bert=True)
encoder.eval()

# Encode a single client query
query = "I want a 2 bedroom apartment with an attached bathroom and balcony"
with torch.no_grad():
    vec = encoder([query])    # torch.Tensor, shape [1, 256]

# vec is ready for dot product against Person 2's geo_tensor [4265, 256]
sims = (vec @ geo_tensor.T).squeeze()   # [4265] similarity scores
top5 = sims.topk(5).indices.tolist()
```

---

## 5. Embedding Contract

These are the guaranteed properties of every vector the text encoder outputs.
Person 4's loss function and retrieval logic can depend on all of these.

| Property | Value |
|----------|-------|
| Dimension | 256 |
| Dtype | `torch.float32` |
| Normalisation | L2-normalised (unit norm) |
| Similarity metric | Dot product = cosine similarity |
| Batch input | List of strings `List[str]` |
| Output shape | `[len(input_list), 256]` |
| Matches Person 2 format | Yes — same dim, same normalisation, same metric |

---

## 6. The Dataset — `text_queries.json`

2000 synthetic queries generated from FloorPlanCAD's 35 category labels.
Each entry has the following structure:

```json
{
  "floor_plan_id": "fp_01313",
  "query": "This is a 2 bedroom and 1 bathroom residential unit. The carpet
            area is approximately 1084 square feet. The living room is
            centrally positioned. The bedroom is adjacent to the bathroom.",
  "template_type": "spatial_relationship",
  "layout_summary": {
    "bedrooms": 2,
    "bathrooms": 1,
    "has_parking": false
  },
  "area_details": {
    "carpet_area_sqft": 1084,
    "built_up_area_sqft": 1246
  },
  "balcony": { "present": true, "count": 1, "area_each": 45 },
  "room_details": [...],
  "constraints": {
    "natural_light": "high",
    "ventilation": "good",
    "privacy_level": "medium"
  },
  "spatial_characteristics": {
    "layout_type": "open-plan",
    "circulation": "efficient"
  }
}
```

**6 template types** in the dataset:

| Template | Purpose |
|----------|---------|
| `residential_basic` | Standard bedroom/bathroom queries |
| `spatial_relationship` | Queries with adjacency constraints |
| `feature_focused` | Single-feature queries (kitchen, wardrobe etc.) |
| `commercial` | Non-residential layouts |
| `multi_constraint` | Multiple constraints combined |
| `negative_constraint` | Queries with "without" / avoidance clauses |

**Recommended use for Person 4:** Use `layout_summary.bedrooms` as a ground truth
label for evaluation metrics. Sample balanced batches across `template_type` to
prevent the model overfitting to the most common template.

---

## 7. Pre-Training Baseline (Record for Comparison)

These numbers were recorded by running the untrained encoder against
Person 2's 4265 geometric embeddings. Use them to prove improvement
after contrastive training.

| Metric | Pre-training value |
|--------|--------------------|
| Similarity score range | 0.0822 – 0.1314 |
| Score spread | 0.0492 |
| Mean rank-1 vs rank-5 gap | 0.0001 |
| Geometric embeddings indexed | 4265 |

**What to expect after training:** spread > 0.3, rank gap > 0.05, and
semantically similar queries clustering visibly away from unrelated ones.
Full results are in `baseline_retrieval_results.json`.

---

## 8. Training Instructions for Person 4

### Step 1 — Initial training (projection only)
Start with `freeze_bert=True`. Only the projection head trains.
This prevents random projection gradients from corrupting BERT's
pretrained weights in the first batches.

```python
encoder = TextEncoder(output_dim=256, freeze_bert=True)
# train for ~5 epochs with contrastive loss
```

### Step 2 — Fine-tuning (full model)
Once the projection has stabilised, unfreeze BERT for end-to-end fine-tuning:

```python
encoder.bert.requires_grad_(True)
# reduce learning rate significantly, e.g. 1e-5
# train for ~10 more epochs
```

### Step 3 — Save and return checkpoint
Save the aligned encoder and return it to Person 3 for the final demo:

```python
torch.save(encoder.state_dict(), "aligned_text_encoder.pt")
```

### Recommended loss
InfoNCE / NT-Xent contrastive loss. The text and geometric vectors are
already L2-normalised so temperature-scaled dot product works directly:

```python
# logits[i][j] = similarity between text query i and floor plan j
logits = (text_vecs @ geo_vecs.T) / temperature
labels = torch.arange(len(text_vecs))   # diagonal is the positive pair
loss   = F.cross_entropy(logits, labels)
```

---

## 9. Loading the Trained Encoder Back (Person 3 Final Step)

Once Person 4 returns `aligned_text_encoder.pt`, Person 3 loads it
into the demo with one change:

```python
# In query_demo.py, replace:
encoder = TextEncoder(output_dim=256, freeze_bert=True)

# With:
encoder = TextEncoder(output_dim=256, freeze_bert=False)
encoder.load_state_dict(torch.load("aligned_text_encoder.pt"))
encoder.eval()
```

The interactive demo (`query_demo.py`) then immediately works with
semantically meaningful retrieval — no other changes needed.

---

## 10. Verified Checks

- [x] Output shape `[batch_size, 256]` confirmed
- [x] L2 norm = 1.000000 confirmed
- [x] Dimension matches Person 2's geometric embeddings (256)
- [x] Normalisation matches Person 2's contract (L2, dot product = cosine)
- [x] Successfully loaded and queried against 4265 real geometric embeddings
- [x] Mock tensors exported and ready for Person 4's loss function
- [x] Baseline retrieval results recorded for post-training comparison
- [x] Interactive query demo working end-to-end

---

*Person 3 work complete. All files committed to `floor_plan_nlp/`.*