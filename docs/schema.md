# FloorPlanCAD Graph Contract Schema

**Purpose:**
These JSON files represent a fully parsed, structurally decomposed, and graph-ready version of raw SVG floor plans. The parser has already handled the heavy lifting of path mathematics, coordinate normalization, and proximity-based edge construction.

---

## High-Level Structure

Every JSON file contains four root-level keys:

```json
{
  "metadata": { ... },
  "nodes":    [ ... ],
  "edges":    [ ... ],
  "symbols":  [ ... ]
}
```

---

## 1. `metadata` (Document Context)

Provides global configuration and summary statistics for the specific floor plan.

- **`filename`** _(string)_: Original SVG filename
- **`viewbox`** _(array)_: `[min_x, min_y, width, height]` — original SVG coordinate boundaries
- **`epsilon`** _(float)_: Proximity threshold used to generate graph edges
- **`total_nodes`, `total_edges`, `total_symbols`** _(int)_: Quick counts for validation
- **`layer_counts`** _(object)_: Counts per layer

  ```json
  {
    "WALL": 101,
    "WINDOW": 67
  }
  ```

---

## 2. `nodes` (The Entities)

This array contains every geometric primitive extracted from the SVG.
**For downstream ML tasks, this is your raw node feature set.**

```json
{
  "id": "n00001",
  "layer": "WALL",
  "semantic_id": 1,
  "instance_id": -1,
  "semantic_label": "wall",
  "geometry_type": "line",
  "segments": [ ... ],
  "endpoints": { ... },
  "features": { ... },
  "raw_attributes": { ... }
}
```

### Key Fields for Downstream Logic

- **`id`**: Unique identifier for graph targeting
- **`layer`**: Standardized layer name (e.g., `WALL`, `INSTALLED_FURNITURE`)
- **`semantic_id`**: Class ID (0–30)
  - `-1` = unannotated

- **`instance_id`**: Object grouping
  - `-1` = "stuff" (walls, floors)
  - `>= 0` = "thing" (distinct objects like a bed)

- **`endpoints`**:

  ```json
  {
    "start": [x, y],
    "end": [x, y]
  }
  ```

- **`features`** (**CRITICAL**): Precomputed normalized features `[0, 1]`
  - `length` _(float)_
  - `center` _(array)_ → `[cx, cy]`
  - `bbox` _(array)_ → `[x0, y0, x1, y1]`
  - `geometry_type_onehot` _(array)_ →
    `[line, arc, curve, circle, ellipse, text, other]`

---

## 3. `edges` (The Graph Topology)

Defines relationships between nodes.
Edges are created when endpoint distance ≤ `epsilon`.

```json
{
  "source": "n00001",
  "target": "n00005",
  "relation": "adjacent",
  "distance": 0.0125
}
```

### Fields

- **`source` / `target`**: Node IDs (undirected connection)
- **`relation`**:
  - `"adjacent"` → general proximity
  - `"same_layer_adjacent"` → same layer
  - `"wall_window"` → specific structural relationship

- **`distance`**: Normalized distance between closest endpoints

---

## 4. `symbols` (Object Instances)

Groups nodes into complete objects using shared `semantic_id` and `instance_id`.

```json
{
  "symbol_id": "sym_14_2",
  "semantic_id": 14,
  "instance_id": 2,
  "semantic_label": "bed",
  "node_ids": ["n00102", "n00103", "n00104", "n00105"],
  "node_count": 4
}
```

### Why This Matters

Instead of inferring which primitives form an object, this provides **ground-truth subgraphs**.

- Useful for:
  - Instance segmentation
  - Graph classification
  - Object reconstruction tasks

- **`node_ids`** = exact components of each object

---

## Summary

| Section  | Purpose                       |
| -------- | ----------------------------- |
| metadata | Global context + stats        |
| nodes    | Feature-rich graph vertices   |
| edges    | Spatial relationships         |
| symbols  | Ground-truth object groupings |

---

This schema is designed to plug directly into machine learning pipelines, especially graph-based models such as GNNs.
