# Division of Work

## Overview

To effectively tackle the challenges of this project, we will divide the work into four distinct roles, each focusing on a critical aspect of the system. This division allows us to leverage individual expertise while ensuring that all components of the project are cohesively integrated.

### **Person 1: Data Parsing & Feature Extraction (The SVG Specialist)**

This person will focus on handling the raw dataset and extracting the foundational geometric data.

- **Dataset Management:** Acquire and manage the Floor PlanCAD dataset, focusing specifically on the SVG format files.
- **SVG Parsing:** Develop scripts to parse the XML-based vector representations of the SVG files to extract meaningful structural information.
- **Entity Extraction:** Identify and extract geometric primitives (lines, rectangles, paths), semantic annotations, and high-level entities like rooms and objects from the raw files.

### **Person 2: Graph Construction & Geometric Encoding (The Spatial Architect)**

This person will take the extracted entities from Person 1 and build the mathematical and structural representations needed for the machine learning model.

- **Graph Construction:** Transform the extracted geometric information into a structured, graph-like format.
- **Node & Edge Mapping:** Ensure spatial entities (e.g., rooms, fixtures) are mapped as nodes , and spatial relationships (e.g., adjacent-to, connected-by-door) are mapped as edges.
- **Geometric Encoder:** Build the model architecture to convert this graph-like structured representation into a vector form. This must be done carefully to preserve spatial connectivity and geometric information without losing semantics.

### **Person 3: Natural Language Pipeline (The Text Specialist)**

This person will handle the other side of the modality mismatch by dealing entirely with the text inputs.

- **Query Analysis:** Analyze how customers express their requirements in natural language (e.g., abstract, qualitative descriptions like "a spacious two-bedroom house").
- **Transformer Integration:** Implement and configure a pre-trained Transformer model to handle language encoding.
- **Semantic Intent Modeling:** Ensure the language encoder successfully captures vector representations of semantic intents, such as room count, spatial relationships, and layout preferences.

### **Person 4: Alignment, Retrieval & System Integration (The Bridge Builder)**

This person is responsible for connecting the geometric and text models, creating the core "bridge" described in the proposal, and building the final inference pipeline.

- **Shared Embedding Space:** Develop the dual-encoding strategy that maps both the geometric vectors (from Person 2) and the linguistic vectors (from Person 3) into a single, unified semantic space.
- **Similarity Matching:** Build the inference mechanism where an encoded customer query is compared against the encoded floor plans using a similarity comparison algorithm.
- **Retrieval System:** Finalize the pipeline so that the most relevant architectural layouts are successfully retrieved and presented to the user based on their text query.

## Pipelining of work

Here is exactly what needs to happen for Persons 2, 3, and 4 to start coding on day one:

### 1. The Handoff from Person 1 to Person 2 (The Geometry Contract)

Person 2 (Spatial Architect) does not need actual SVG files parsed. They just need to know how Person 1 (SVG Specialist) will hand them the data.

- **The "Something More":** They need to define a strict JSON schema or Python dictionary structure. For example, they must agree that the output will be a list of nodes (e.g., `{"id": 1, "type": "bedroom", "coordinates": [...]}`) and a list of edges (e.g., `{"source": 1, "target": 2, "relation": "adjacent"}`).
- **How to start:** Person 1 and 2 write a dummy JSON file by hand that represents a fake, simple floor plan. Person 2 uses this fake file to build their graph-encoding model while Person 1 figures out how to extract real data into that exact format.

#### Final Person 2 Contract (Locked for implementation)

- **Input file format:** one `*_contract.json` per plan from `train/`.
- **Input keys consumed by Person 2:**
  - `nodes`: list of node dictionaries with geometry and semantic fields.
  - `edges`: list of edge dictionaries with `source`, `target`, `relation`, `distance`.
  - `metadata.filename`: stable identifier for indexing.
- **Node feature schema (v1):**
  - **Numeric:** `features.length`, `features.center` (2), `features.bbox` (4).
  - **Categorical encoded as ids/one-hot:** `semantic_id`, `layer`, `geometry_type` / `geometry_type_onehot`.
  - **Auxiliary:** `instance_id` flag (`instance_id != -1`).
- **Edge feature schema (v1):**
  - Topology: directed `edge_index` from node id mapping.
  - Edge attributes: `distance` and relation id (`adjacent`, `same_layer_adjacent`, `wall_window`).
- **Output from Person 2:** one graph embedding vector per floor plan, shape `[256]`, L2-normalized for cosine retrieval.
- **Artifact contract for Person 4:**
  - `embeddings.npy` (shape `[N, 256]`)
  - `embedding_index.json` mapping row -> source `*_contract.json`
  - optional `embeddings.pt` mirror for PyTorch workflows

### 2. Person 3 (The Text Specialist) is Already Independent

Person 3 actually doesn't have to wait for anyone. The natural language pipeline is entirely separated from the geometric parsing.

- **The "Something More":** Person 3 just needs to agree with Person 4 on the **dimensionality** of the final vector space. If Person 3 is using a BERT model that outputs a 768-dimensional vector, Person 4 needs to know that.
- **How to start:** Person 3 can immediately start writing scripts to process text queries (like "two-bedroom house") and turn them into embeddings.

### 3. The Handoff to Person 4 (The Alignment Contract)

Person 4 (The Bridge Builder) needs to build a system that measures similarity between the math coming from Person 2 and the math coming from Person 3.

- **The "Something More":** Person 4 needs mock tensors (matrices). They need to agree on the exact tensor dimensions (e.g., both the text model and the graph model will output vectors of size `[1, 256]`).
- **How to start:** Person 4 can generate arrays of completely random numbers using `numpy` or `torch` shaped exactly like the expected outputs (`[batch_size, 256]`). Using this random "garbage" data, Person 4 can build the entire contrastive loss function, the similarity matching algorithm, and the final retrieval interface.

### Summary of the Setup

If your team spends the first meeting agreeing on these three things, everyone can work at the same time:

1. The exact JSON structure of the parsed floor plan.
2. The embedding dimension size for the geometric encoder.
3. The embedding dimension size for the text encoder (these must match!).

Would you like to draft out what those specific JSON schemas and tensor dimensions should look like so you can hand them to your team?
