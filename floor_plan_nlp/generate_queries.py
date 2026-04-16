import json
import random
from pathlib import Path

random.seed(42)

# ── 1. Vocabulary ────────────────────────────────────────────────────────────

BEDROOM_TYPES   = ["bedroom", "master bedroom", "guest bedroom", "children's bedroom"]
BATHROOM_ITEMS  = ["toilet", "sink", "bath tub", "squat toilet", "urinal", "bath"]
KITCHEN_ITEMS   = ["gas stove", "refrigerator", "washing machine", "sink"]
LIVING_ITEMS    = ["sofa", "TV cabinet", "table", "chair"]
STORAGE_ITEMS   = ["wardrobe", "high cabinet", "half-height cabinet", "bedside cupboard"]
CIRCULATION     = ["stair", "elevator", "escalator"]
OPENINGS        = ["single door", "double door", "sliding door", "window", "bay window"]

SIZE_ADJ        = ["spacious", "compact", "large", "small", "open-plan", "cozy"]
LAYOUT_ADJ      = ["well-lit", "modern", "traditional", "minimalist"]
CONNECTIVITY    = ["adjacent to", "connected to", "next to", "overlooking"]

# ── 2. Phrase Variations (FIX 1) ─────────────────────────────────────────────

LIVING_ROOM_PHRASES = [
    "The living room is centrally positioned and connects to key spaces.",
    "A large central living area ties the layout together.",
    "The drawing room opens directly onto the dining space.",
    "The main living area receives ample natural light."
]

BEDROOM_PHRASES = [
    "Bedrooms are arranged to maintain privacy and ventilation.",
    "The sleeping quarters are tucked away from the main circulation.",
    "Bedrooms face outward for natural light and cross ventilation.",
    "Private rooms are separated from common areas by a corridor."
]

KITCHEN_PHRASES = [
    "The kitchen is efficiently planned for daily use.",
    "A compact modular kitchen sits adjacent to the dining area.",
    "The kitchen is designed for maximum workflow efficiency.",
    "An open kitchen flows into the dining and living zones."
]

# ── 3. Helper Generators ─────────────────────────────────────────────────────

# FIX 3: Area depends on number of bedrooms
def generate_area(n_beds: int):
    base = 400 + (n_beds - 1) * 300
    carpet = random.randint(base, base + 400)
    built_up = int(carpet * random.uniform(1.1, 1.3))
    return carpet, built_up


def generate_balcony():
    has_balcony = random.random() > 0.4
    if not has_balcony:
        return {"present": False}
    return {
        "present": True,
        "count": random.randint(1, 3),
        "area_each": random.randint(20, 80)
    }


def generate_room_details(n_beds, n_baths):
    rooms = []

    for _ in range(n_beds):
        rooms.append({
            "type": "bedroom",
            "area": random.randint(100, 250),
            "attached_bath": random.random() > 0.5
        })

    for _ in range(n_baths):
        rooms.append({
            "type": "bathroom",
            "area": random.randint(40, 90)
        })

    rooms.append({
        "type": "kitchen",
        "area": random.randint(60, 150)
    })

    rooms.append({
        "type": "living_room",
        "area": random.randint(120, 300)
    })

    return rooms


# FIX 1: Use varied phrases
def generate_description(n_beds, n_baths, carpet, balcony, layout_type):
    desc = []
    desc.append(f"This is a {n_beds} bedroom and {n_baths} bathroom residential unit.")
    desc.append(f"The carpet area is approximately {carpet} square feet.")

    if balcony["present"]:
        desc.append(f"It includes {balcony['count']} balcony spaces for outdoor usage.")
    else:
        desc.append("The layout does not include a balcony.")

    desc.append(f"The overall layout follows a {layout_type} design.")
    desc.append(random.choice(LIVING_ROOM_PHRASES))
    desc.append(random.choice(BEDROOM_PHRASES))
    desc.append(random.choice(KITCHEN_PHRASES))

    return " ".join(desc)

# ── 4. Main Generator ────────────────────────────────────────────────────────

def generate_query_dataset(
    n_samples: int = 2000,
    output_path: str = "text_queries.json"
):
    dataset = []
    floor_plan_counter = 0

    for _ in range(n_samples):
        template_type = random.choice([
            "residential_basic",
            "spatial_relationship",
            "feature_focused",
            "commercial",
            "multi_constraint",
            "negative_constraint"
        ])

        n_beds = random.randint(1, 4)
        n_baths = random.randint(1, 3)
        has_parking = random.random() > 0.5

        # FIX 3 applied here
        carpet, built_up = generate_area(n_beds)

        balcony = generate_balcony()
        rooms = generate_room_details(n_beds, n_baths)

        layout_type = random.choice(["compact", "open-plan", "segregated"])

        description = generate_description(
            n_beds, n_baths, carpet, balcony, layout_type
        )

        # ── Template-specific enrichments ─────────────────────────────

        if template_type == "spatial_relationship":
            a = random.choice(["bedroom", "kitchen", "living room"])
            b = random.choice(["bathroom", "balcony", "dining area"])
            rel = random.choice(CONNECTIVITY)
            description += f" The {a} is {rel} the {b}."

        elif template_type == "feature_focused":
            feature = random.choice(KITCHEN_ITEMS + LIVING_ITEMS + STORAGE_ITEMS)
            description += f" The design highlights a {feature}."

        elif template_type == "commercial":
            description = f"A commercial layout. " + description

        # FIX 2: Proper multi_constraint handling
        elif template_type == "multi_constraint":
            circ = random.choice(CIRCULATION)
            opening = random.choice(OPENINGS)
            description += (
                f" The unit includes {circ} access and features "
                f"{opening}s throughout for light and ventilation."
            )

        elif template_type == "negative_constraint":
            unwanted = random.choice(KITCHEN_ITEMS + STORAGE_ITEMS)
            description += f" The design avoids including a {unwanted}."

        # ── Final JSON ────────────────────────────────────────────────

        dataset.append({
            "floor_plan_id": f"fp_{floor_plan_counter:05d}",
            "query": description,
            "template_type": template_type,

            "layout_summary": {
                "bedrooms": n_beds,
                "bathrooms": n_baths,
                "has_parking": has_parking
            },

            "area_details": {
                "carpet_area_sqft": carpet,
                "built_up_area_sqft": built_up
            },

            "balcony": balcony,

            "room_details": rooms,

            "constraints": {
                "natural_light": random.choice(["high", "medium", "low"]),
                "ventilation": random.choice(["good", "average"]),
                "privacy_level": random.choice(["high", "medium"])
            },

            "spatial_characteristics": {
                "layout_type": layout_type,
                "circulation": random.choice(["efficient", "moderate"])
            }
        })

        floor_plan_counter += 1

    random.shuffle(dataset)

    Path(output_path).write_text(json.dumps(dataset, indent=2))
    print(f"Generated {len(dataset)} queries → {output_path}")

    return dataset


# ── 5. Stats ────────────────────────────────────────────────────────────────

def print_stats(dataset):
    from collections import Counter

    types = Counter(d["template_type"] for d in dataset)

    print("\nTemplate distribution:")
    for t, count in types.most_common():
        print(f"{t:<25} {count:>5} ({100*count/len(dataset):.1f}%)")

    print("\nSample queries:\n")
    seen = set()
    for d in dataset:
        if d["template_type"] not in seen:
            print(f"[{d['template_type']}] {d['query']}\n")
            seen.add(d["template_type"])
        if len(seen) == 6:
            break


# ── 6. Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset = generate_query_dataset(n_samples=2000)
    print_stats(dataset)