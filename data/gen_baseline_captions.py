"""Generate a reproducible baseline set of single-object 3D captions.

Produces N (default 500) distinct, natural-language captions describing ONE 3D
object each (≤2 sentences), spanning many categories / colors / materials /
styles — for use as a fixed prompt set when scoring CLIP text-image alignment of
generated 3DGS objects (see jit/eval_clip_alignment.py --prompts_file).

Design notes:
  * Single object per caption (the model generates one object) — no multi-object
    scenes. Attributes/parts are fine.
  * Style mirrors Cap3D-ish phrasing the model was conditioned on ("A red wooden
    toy car.", "A 3D model of a ceramic vase.").
  * Deterministic given --seed, so the baseline is reproducible/regenerable.

Usage:
    python data/gen_baseline_captions.py --n 500 --seed 0 \
        --out data/baseline_captions_500.json
"""
from __future__ import annotations

import argparse
import json
import random

# --- vocabulary -----------------------------------------------------------------------
OBJECTS = [
    # animals
    "cat", "dog", "horse", "elephant", "rabbit", "fox", "owl", "turtle", "dolphin",
    "lion", "tiger", "bear", "penguin", "frog", "deer", "wolf", "panda", "koala",
    "giraffe", "dragon", "dinosaur", "snake", "eagle", "parrot", "goldfish", "crab",
    "butterfly", "hedgehog", "squirrel", "octopus",
    # vehicles
    "car", "sports car", "pickup truck", "bus", "motorcycle", "bicycle", "airplane",
    "helicopter", "sailboat", "ship", "submarine", "rocket", "steam train", "tractor",
    "spaceship", "scooter", "hot air balloon",
    # furniture
    "chair", "armchair", "sofa", "coffee table", "desk", "bookshelf", "bed", "stool",
    "cabinet", "wardrobe", "bench", "rocking chair",
    # household / decor
    "table lamp", "vase", "teapot", "coffee mug", "wine bottle", "plate", "bowl",
    "wall clock", "mirror", "wicker basket", "candle", "picture frame", "kettle",
    "chandelier", "fountain", "treasure chest", "hourglass", "lantern",
    # food
    "apple", "banana", "birthday cake", "cupcake", "donut", "hamburger", "pizza slice",
    "ice cream cone", "strawberry", "pumpkin", "mushroom", "loaf of bread", "sushi roll",
    # plants
    "potted plant", "cactus", "bonsai tree", "sunflower", "rose", "fern", "palm tree",
    "tulip",
    # tools / instruments
    "hammer", "wrench", "axe", "guitar", "violin", "grand piano", "drum", "trumpet",
    "flute",
    # electronics
    "camera", "laptop", "smartphone", "headphones", "television", "game controller",
    "robot vacuum",
    # architecture
    "house", "castle", "watchtower", "cottage", "lighthouse", "windmill", "tent",
    "wooden bridge",
    # weapons / fantasy
    "sword", "shield", "battle axe", "longbow", "magic staff", "dagger", "war hammer",
    # toys / misc
    "teddy bear", "rag doll", "toy robot", "spinning top", "kite", "building blocks",
    "rubber duck", "gemstone", "crystal cluster", "stone statue",
    # accessories
    "top hat", "cowboy boot", "backpack", "crown", "knight's helmet", "wristwatch",
    "pair of glasses", "umbrella",
]

CHARACTERS = [
    "knight", "wizard", "robot", "astronaut", "soldier", "ninja", "pirate",
    "superhero", "alien", "fairy", "dwarf", "elf", "viking", "samurai", "goblin",
    "princess", "cowboy", "garden gnome", "mermaid", "skeleton warrior", "monk",
    "forest ranger", "witch", "cartoon mouse", "robot dog",
]

COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black",
    "gray", "brown", "golden", "silver", "teal", "turquoise", "maroon", "navy blue",
    "beige", "ivory", "crimson", "emerald green", "bronze", "copper", "lavender",
    "mint green", "charcoal",
]

MATERIALS = [
    "wooden", "metal", "plastic", "ceramic", "glass", "stone", "marble", "leather",
    "fabric", "bronze", "rusty", "polished steel", "frosted glass", "woven straw",
    "carved jade", "weathered iron", "porcelain", "concrete",
]

STYLES = [
    "cartoon", "realistic", "low-poly", "stylized", "voxel-art", "antique", "vintage",
    "futuristic", "sci-fi", "fantasy", "minimalist", "ornate", "steampunk", "cute",
    "hand-painted", "geometric", "retro", "whimsical",
]

SIZES = ["small", "large", "tall", "tiny", "miniature", "chunky", "slender", "compact"]

WITH_CLAUSES = [
    "with intricate patterns", "with a glossy finish", "with rounded edges",
    "with delicate engravings", "with a weathered texture", "with vibrant colors",
    "with a curved handle", "with decorative trim", "with glowing accents",
    "with a matte surface", "with golden details", "with a worn, rustic look",
    "with smooth, flowing curves", "with sharp angular facets",
]

DETAILS = [
    "It has a smooth, glossy surface.", "The design is detailed and ornate.",
    "It features intricate carvings along the edges.",
    "The texture looks worn and weathered.", "It is brightly colored and cartoonish.",
    "The shape is simple and geometric.", "It rests on a small circular base.",
    "Fine details cover its entire surface.", "The finish is matte and slightly rough.",
    "It has a sleek, modern silhouette.", "The colors are soft and pastel.",
    "It looks hand-crafted and one-of-a-kind.", "The proportions are exaggerated and playful.",
    "It has a symmetrical, balanced form.", "Subtle highlights catch the light.",
]

CLOTHING = [
    "a flowing red cape", "a golden helmet", "a blue cloak", "ornate plate armor",
    "a pointed wizard hat", "a worn leather jacket", "a feathered cap",
    "a long hooded robe", "a tribal mask", "a steel breastplate", "a wide-brimmed hat",
    "a backpack and goggles",
]


def _art(word: str, cap: bool = True) -> str:
    a = "an" if word[:1].lower() in "aeiou" else "a"
    return a.capitalize() if cap else a


def _cap_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def generate_one(rng: random.Random) -> str:
    """Build one caption. Templates weighted toward short, Cap3D-like phrasings."""
    tid = rng.choices(range(11), weights=[14, 10, 13, 9, 10, 9, 8, 8, 7, 6, 6])[0]
    obj = rng.choice(OBJECTS)
    color = rng.choice(COLORS)
    mat = rng.choice(MATERIALS)
    style = rng.choice(STYLES)
    size = rng.choice(SIZES)
    wc = rng.choice(WITH_CLAUSES)
    det = rng.choice(DETAILS)

    if tid == 0:                                            # color + object
        s = f"{_art(color)} {color} {obj}."
    elif tid == 1:                                          # material + object
        s = f"{_art(mat)} {mat} {obj}."
    elif tid == 2:                                          # color + material + object
        s = f"{_art(color)} {color} {mat} {obj}."
    elif tid == 3:                                          # style + object
        s = f"{_art(style)} {style} {obj}."
    elif tid == 4:                                          # size + object + with-clause
        s = f"{_art(size)} {size} {obj} {wc}."
    elif tid == 5:                                          # style + object + with-clause
        s = f"{_art(style)} {style} {obj} {wc}."
    elif tid == 6:                                          # 3D-model phrasing
        s = f"A 3D model of {_art(color, cap=False)} {color} {obj}."
    elif tid == 7:                                          # color + object + detail (2 sent.)
        s = f"{_art(color)} {color} {obj}. {det}"
    elif tid == 8:                                          # material + object + detail (2 sent.)
        s = f"{_art(mat)} {mat} {obj}. {det}"
    elif tid == 9:                                          # size + color + object + detail
        s = f"{_art(size)} {size} {color} {obj}. {det}"
    else:                                                   # character wearing clothing
        ch = rng.choice(CHARACTERS)
        cl = rng.choice(CLOTHING)
        if rng.random() < 0.5:
            s = f"{_art(style)} {style} {ch} wearing {cl}."
        else:
            s = f"{_art(color)} {color} {ch} wearing {cl}. {det}"
    return _cap_first(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/baseline_captions_500.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    captions: list[str] = []
    attempts = 0
    max_attempts = args.n * 200
    while len(captions) < args.n and attempts < max_attempts:
        attempts += 1
        c = generate_one(rng)
        if c not in seen:
            seen.add(c)
            captions.append(c)
    if len(captions) < args.n:
        raise RuntimeError(
            f"only produced {len(captions)} unique captions in {attempts} attempts; "
            "expand the vocabulary."
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(captions)} captions -> {args.out} (seed={args.seed})")
    for c in captions[:12]:
        print("  -", c)


if __name__ == "__main__":
    main()
