"""
svg_parser.py  -  Improved FloorPlanCAD SVG Parser
====================================================
Improvements over v1
--------------------
1. Layer-aware traversal  : uses inkscape layer labels (WALL, WINDOW, 家具…)
                            as the primary semantic context instead of HTML tag type.
2. Semantic / instance IDs: reads ``semantic-id`` and ``instance-id`` attributes
                            that the FloorPlanCAD dataset embeds directly on
                            annotated elements.
3. Path decomposition     : parses SVG path commands (M/L/A/C/Z) to extract
                            the true start-point and end-point of every sub-path
                            segment, and approximates arc length geometrically.
4. Entity-level features  : records length, normalised centre (x_c, y_c),
                            and a geometry-type label for every node so that
                            Person 2 can build spatial feature vectors immediately.
5. Instance grouping      : elements that share the same (semantic-id, instance-id)
                            pair are bundled into a ``symbol`` entry, giving Person 2
                            ready-made object instances (bed, sink, door …).
6. Proximity-based edges  : edges are created when the closest endpoints of two
                            entities are within a configurable threshold ε
                            (matching the PanCADNet paper's construction).
7. Normalised co-ordinates: all coordinates are normalised to [0, 1] relative to
                            the SVG viewBox so downstream models see scale-invariant input.
8. Circles / Ellipses     : handled as first-class geometry types.
9. Summary statistics     : metadata section includes per-layer entity counts and
                            a list of distinct semantic/instance symbols detected.
"""

import xml.etree.ElementTree as ET
import json
import math
import os
import sys
from collections import defaultdict

# Internal Imports
from constants import SEMANTIC_ID_TO_LABEL, LAYER_LABEL_MAP, LAYER_TRANSLATIONS, SVG_NS, INKSCAPE_NS
from geometry import parse_path_geometry, segments_bbox, total_length, segment_length

# ===========================================================================
# Main parser
# ===========================================================================


def parse_svg_to_contract(svg_file_path: str,
                          output_dir: str = "data",
                          epsilon: float = 0.5,
                          max_edges_per_node: int = 3):
    """
    Parse a FloorPlanCAD SVG into a node/edge contract JSON.

    Parameters
    ----------
    svg_file_path      : path to the .svg file
    output_dir         : directory where the JSON will be saved
    epsilon            : proximity threshold (in SVG coordinate units) for edge creation
    max_edges_per_node : maximum number of edges allowed per node (sparsity control)

    Output contract schema
    ----------------------
    {
      "metadata": {
        "filename": str,
        "viewbox": [min_x, min_y, width, height],
        "total_nodes": int,
        "total_edges": int,
        "layer_counts": {layer_name: count, ...},
        "symbols": [{semantic_id, instance_id, label, node_ids}, ...]
      },
      "nodes": [
        {
          "id": str,               # unique node ID
          "layer": str,            # source inkscape layer name
          "semantic_id": int,      # -1 if unannotated
          "instance_id": int,      # -1 for stuff (wall), unique int for things
          "semantic_label": str,   # human-readable label
          "geometry_type": str,    # "line" | "arc" | "curve" | "circle" | "ellipse" | "text"
          "segments": [...],       # parsed path segments (paths only)
          "endpoints": {           # first/last point of entity (for edge construction)
              "start": [x, y],
              "end":   [x, y]
          },
          "features": {
              "length": float,           # total geometric length
              "center": [cx, cy],        # normalised centre in [0,1]
              "bbox": [x0,y0,x1,y1],    # normalised bounding box
              "geometry_type_onehot": [int, int, int, int, int]  # line/arc/curve/circle/ellipse
          },
          "raw_attributes": {...}  # original SVG attributes
        },
        ...
      ],
      "edges": [
        {
          "source": str,
          "target": str,
          "relation": str,         # "adjacent" | "parallel_wall_window"
          "distance": float
        },
        ...
      ],
      "symbols": [
        {
          "symbol_id": str,
          "semantic_id": int,
          "instance_id": int,
          "semantic_label": str,
          "node_ids": [str, ...]
        },
        ...
      ]
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        tree = ET.parse(svg_file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Could not parse {svg_file_path}: {e}")
        return None

    # Resolve namespace
    ns = ""
    if "}" in root.tag:
        ns = root.tag.split("}")[0] + "}"

    # Parse viewBox for normalisation
    viewbox = [0.0, 0.0, 100.0, 100.0]
    if "viewBox" in root.attrib:
        parts = root.attrib["viewBox"].split()
        if len(parts) == 4:
            viewbox = [float(p) for p in parts]
    vb_x, vb_y, vb_w, vb_h = viewbox

    def norm_x(x): return (x - vb_x) / vb_w if vb_w else x
    def norm_y(y): return (y - vb_y) / vb_h if vb_h else y

    # Geometry type → one-hot index
    GEO_TYPES = ["line", "arc", "curve", "circle", "ellipse", "text", "other"]

    def geo_onehot(geo_type):
        idx = GEO_TYPES.index(
            geo_type) if geo_type in GEO_TYPES else GEO_TYPES.index("other")
        oh = [0] * len(GEO_TYPES)
        oh[idx] = 1
        return oh

    nodes = []
    node_id_counter = [0]

    def next_id():
        node_id_counter[0] += 1
        return f"n{node_id_counter[0]:05d}"

    layer_counts = defaultdict(int)

    # -----------------------------------------------------------------------
    # Iterate over layers (top-level <g> with inkscape:label)
    # -----------------------------------------------------------------------
    for group in root.findall(f".//{ns}g"):
        layer_label_attrib = (
            group.attrib.get(f"{{{INKSCAPE_NS}}}label") or
            group.attrib.get("inkscape:label", "")
        )
        layer_id = group.attrib.get("id", "")

        raw_layer_name = layer_label_attrib or layer_id

        # Only process top-level named layers
        if not raw_layer_name:
            continue

        # Translate Chinese layer names to English if applicable
        layer_name = LAYER_TRANSLATIONS.get(raw_layer_name, raw_layer_name)
        fallback_semantic_label = LAYER_LABEL_MAP.get(
            layer_name, layer_name.lower())

        # --- PATH elements --------------------------------------------------
        for path_elem in group.findall(f"{ns}path"):
            d = path_elem.attrib.get("d", "")
            if not d:
                continue

            sem_id = int(path_elem.attrib.get("semantic-id",  -1))
            inst_id = int(path_elem.attrib.get("instance-id", -1))
            sem_label = SEMANTIC_ID_TO_LABEL.get(
                sem_id, fallback_semantic_label)

            segments = parse_path_geometry(d)
            if not segments:
                continue

            # Dominant geometry type for this path
            type_counts = defaultdict(int)
            for seg in segments:
                type_counts[seg["type"]] += 1
            dom_type = max(type_counts, key=type_counts.get)

            total_len = total_length(segments)
            bbox = segments_bbox(segments)

            if bbox:
                x0, y0, x1, y1 = bbox
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                norm_bbox = [round(norm_x(x0), 6), round(norm_y(y0), 6),
                             round(norm_x(x1), 6), round(norm_y(y1), 6)]
                norm_center = [round(norm_x(cx), 6), round(norm_y(cy), 6)]
            else:
                norm_bbox = [0, 0, 0, 0]
                norm_center = [0, 0]

            # Endpoints for proximity edge construction
            start_pt = list(segments[0]["start"])
            end_pt = list(segments[-1]["end"])

            raw_attrs = {k: v for k, v in path_elem.attrib.items()
                         if k not in ("d", "semantic-id", "instance-id")}

            node = {
                "id":              next_id(),
                "layer":           layer_name,
                "semantic_id":     sem_id,
                "instance_id":     inst_id,
                "semantic_label":  sem_label,
                "geometry_type":   dom_type,
                "segments":        segments,
                "endpoints": {
                    "start": [round(start_pt[0], 6), round(start_pt[1], 6)],
                    "end":   [round(end_pt[0],   6), round(end_pt[1],   6)],
                },
                "features": {
                    "length":               round(total_len, 6),
                    "center":               norm_center,
                    "bbox":                 norm_bbox,
                    "geometry_type_onehot": geo_onehot(dom_type),
                },
                "raw_attributes": raw_attrs,
            }
            nodes.append(node)
            layer_counts[layer_name] += 1

        # --- CIRCLE elements ------------------------------------------------
        for circ in group.findall(f"{ns}circle"):
            try:
                cx = float(circ.attrib.get("cx", 0))
                cy = float(circ.attrib.get("cy", 0))
                r = float(circ.attrib.get("r",  0))
            except ValueError:
                continue

            sem_id = int(circ.attrib.get("semantic-id",  -1))
            inst_id = int(circ.attrib.get("instance-id", -1))
            sem_label = SEMANTIC_ID_TO_LABEL.get(
                sem_id, fallback_semantic_label)

            circumference = 2 * math.pi * r
            node = {
                "id":             next_id(),
                "layer":          layer_name,
                "semantic_id":    sem_id,
                "instance_id":    inst_id,
                "semantic_label": sem_label,
                "geometry_type":  "circle",
                "segments":       [],
                "endpoints": {
                    "start": [round(norm_x(cx - r), 6), round(norm_y(cy), 6)],
                    "end":   [round(norm_x(cx + r), 6), round(norm_y(cy), 6)],
                },
                "features": {
                    "length":               round(circumference, 6),
                    "center":               [round(norm_x(cx), 6), round(norm_y(cy), 6)],
                    "bbox":                 [round(norm_x(cx - r), 6), round(norm_y(cy - r), 6),
                                             round(norm_x(cx + r), 6), round(norm_y(cy + r), 6)],
                    "geometry_type_onehot": geo_onehot("circle"),
                    "radius":               round(r, 6),
                },
                "raw_attributes": {k: v for k, v in circ.attrib.items()
                                   if k not in ("semantic-id", "instance-id")},
            }
            nodes.append(node)
            layer_counts[layer_name] += 1

        # --- ELLIPSE elements -----------------------------------------------
        for ell in group.findall(f"{ns}ellipse"):
            try:
                cx = float(ell.attrib.get("cx", 0))
                cy = float(ell.attrib.get("cy", 0))
                rx = float(ell.attrib.get("rx", 0))
                ry = float(ell.attrib.get("ry", 0))
            except ValueError:
                continue

            sem_id = int(ell.attrib.get("semantic-id",  -1))
            inst_id = int(ell.attrib.get("instance-id", -1))
            sem_label = SEMANTIC_ID_TO_LABEL.get(
                sem_id, fallback_semantic_label)

            # Ramanujan approximation for ellipse perimeter
            h = ((rx - ry) / (rx + ry)) ** 2 if (rx + ry) > 0 else 0
            perimeter = math.pi * (rx + ry) * \
                (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

            node = {
                "id":             next_id(),
                "layer":          layer_name,
                "semantic_id":    sem_id,
                "instance_id":    inst_id,
                "semantic_label": sem_label,
                "geometry_type":  "ellipse",
                "segments":       [],
                "endpoints": {
                    "start": [round(norm_x(cx - rx), 6), round(norm_y(cy), 6)],
                    "end":   [round(norm_x(cx + rx), 6), round(norm_y(cy), 6)],
                },
                "features": {
                    "length":               round(perimeter, 6),
                    "center":               [round(norm_x(cx), 6), round(norm_y(cy), 6)],
                    "bbox":                 [round(norm_x(cx - rx), 6), round(norm_y(cy - ry), 6),
                                             round(norm_x(cx + rx), 6), round(norm_y(cy + ry), 6)],
                    "geometry_type_onehot": geo_onehot("ellipse"),
                    "rx": round(rx, 6), "ry": round(ry, 6),
                },
                "raw_attributes": {k: v for k, v in ell.attrib.items()
                                   if k not in ("semantic-id", "instance-id")},
            }
            nodes.append(node)
            layer_counts[layer_name] += 1

        # --- TEXT elements --------------------------------------------------
        for text_elem in group.findall(f"{ns}text"):
            content = "".join(text_elem.itertext()).strip()
            if not content:
                continue
            try:
                tx = float(text_elem.attrib.get("x", 0))
                ty = float(text_elem.attrib.get("y", 0))
            except ValueError:
                tx, ty = 0.0, 0.0

            node = {
                "id":             next_id(),
                "layer":          layer_name,
                "semantic_id": -1,
                "instance_id": -1,
                "semantic_label": "text_" + fallback_semantic_label,
                "geometry_type":  "text",
                "segments":       [],
                "content":        content,
                "endpoints": {
                    "start": [round(norm_x(tx), 6), round(norm_y(ty), 6)],
                    "end":   [round(norm_x(tx), 6), round(norm_y(ty), 6)],
                },
                "features": {
                    "length":               0.0,
                    "center":               [round(norm_x(tx), 6), round(norm_y(ty), 6)],
                    "bbox":                 [round(norm_x(tx), 6), round(norm_y(ty), 6),
                                             round(norm_x(tx), 6), round(norm_y(ty), 6)],
                    "geometry_type_onehot": geo_onehot("text"),
                },
                "raw_attributes": {k: v for k, v in text_elem.attrib.items()},
            }
            nodes.append(node)
            layer_counts[layer_name] += 1

    # -----------------------------------------------------------------------
    # Proximity-based edge construction
    # Matches PanCADNet: edge iff min endpoint distance < epsilon
    # -----------------------------------------------------------------------
    def endpoint_distance(n1, n2):
        """Minimum distance between any pair of endpoints of two nodes."""
        pts1 = [n1["endpoints"]["start"], n1["endpoints"]["end"]]
        pts2 = [n2["endpoints"]["start"], n2["endpoints"]["end"]]
        min_d = float("inf")
        for p in pts1:
            for q in pts2:
                # Imported segment_length handles the math
                d = segment_length(p[0], p[1], q[0], q[1])
                if d < min_d:
                    min_d = d
        return min_d

    # Track edge count per node to enforce max_edges_per_node
    edge_count = defaultdict(int)
    edges = []

    # Only compare annotated structural nodes (skip dimension/text layers for edges)
    SKIP_LAYERS_FOR_EDGES = {"DIM_LEAD", "PUB_DIM", "AXIS", "layerAXIS",
                             "WINDOW_TEXT", "PUB_TEXT", "PUB_HATCH"}
    structural_nodes = [n for n in nodes
                        if n["layer"] not in SKIP_LAYERS_FOR_EDGES
                        and n["geometry_type"] != "text"]

    # Normalised epsilon: epsilon is in raw SVG coords; normalise for comparison
    eps_norm = epsilon / max(vb_w, vb_h) if max(vb_w, vb_h) > 0 else epsilon

    for i in range(len(structural_nodes)):
        for j in range(i + 1, len(structural_nodes)):
            if (edge_count[structural_nodes[i]["id"]] >= max_edges_per_node or
                    edge_count[structural_nodes[j]["id"]] >= max_edges_per_node):
                continue
            dist = endpoint_distance(structural_nodes[i], structural_nodes[j])
            if dist <= eps_norm:
                # Determine relation type
                li = structural_nodes[i]["layer"]
                lj = structural_nodes[j]["layer"]
                if ("WALL" in li and "WINDOW" in lj) or ("WINDOW" in li and "WALL" in lj):
                    relation = "wall_window"
                elif li == lj:
                    relation = "same_layer_adjacent"
                else:
                    relation = "adjacent"

                edges.append({
                    "source":   structural_nodes[i]["id"],
                    "target":   structural_nodes[j]["id"],
                    "relation": relation,
                    "distance": round(dist, 6),
                })
                edge_count[structural_nodes[i]["id"]] += 1
                edge_count[structural_nodes[j]["id"]] += 1

    # -----------------------------------------------------------------------
    # Instance / symbol grouping
    # Group nodes that share the same (semantic_id, instance_id) into symbols
    # (only for annotated things, not unannotated or stuff with inst_id == -1)
    # -----------------------------------------------------------------------
    symbol_map = defaultdict(list)
    for node in nodes:
        s_id = node["semantic_id"]
        i_id = node["instance_id"]
        if s_id != -1 and i_id != -1:
            symbol_map[(s_id, i_id)].append(node["id"])

    symbols = []
    for (s_id, i_id), node_ids in sorted(symbol_map.items()):
        symbols.append({
            "symbol_id":      f"sym_{s_id}_{i_id}",
            "semantic_id":    s_id,
            "instance_id":    i_id,
            "semantic_label": SEMANTIC_ID_TO_LABEL.get(s_id, f"class_{s_id}"),
            "node_ids":       node_ids,
            "node_count":     len(node_ids),
        })

    # -----------------------------------------------------------------------
    # Assemble contract
    # -----------------------------------------------------------------------
    filename = os.path.basename(svg_file_path)
    contract = {
        "metadata": {
            "filename":      filename,
            "viewbox":       viewbox,
            "epsilon":       epsilon,
            "total_nodes":   len(nodes),
            "total_edges":   len(edges),
            "total_symbols": len(symbols),
            "layer_counts":  dict(layer_counts),
        },
        "nodes":   nodes,
        "edges":   edges,
        "symbols": symbols,
    }

    # Save
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}_contract.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)

    # Print summary
    if (False):
        print(f"\n{'='*55}")
        print(f"  Contract saved → {output_path}")
        print(f"{'='*55}")
        print(f"  Nodes  : {len(nodes)}")
        print(f"  Edges  : {len(edges)}")
        print(f"  Symbols: {len(symbols)}")
        print(f"\n  Layer breakdown:")
        for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1]):
            print(f"    {layer:<25} {count:>4} entities")
        if symbols:
            print(f"\n  Detected symbol types:")
            by_label = defaultdict(int)
            for sym in symbols:
                by_label[sym["semantic_label"]] += 1
            for label, cnt in sorted(by_label.items(), key=lambda x: -x[1]):
                print(f"    {label:<25} {cnt:>4} instances")
        print(f"{'='*55}\n")

    return contract


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/train/0000-0003.svg"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/contracts"

    contract = parse_svg_to_contract(
        svg_file_path=target,
        output_dir=out_dir,
        epsilon=0.5,          # SVG units; ~5mm at typical scale
        max_edges_per_node=3  # matching PanCADNet K=3
    )
