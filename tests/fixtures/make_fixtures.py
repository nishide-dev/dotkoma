"""Regenerate the hand-drawn fixture sprites.

uv run python tests/fixtures/make_fixtures.py
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent / "sprites"

# '.' transparent; other chars map to colours
COLORS = {
    "k": (20, 20, 30),
    "g": (60, 180, 75),
    "G": (30, 110, 50),
    "w": (240, 240, 240),
    "s": (180, 180, 200),
    "y": (250, 210, 60),
    "o": (240, 130, 40),
    "c": (230, 220, 190),
}

SPRITES = {
    "slime": [
        "................",
        "................",
        "................",
        "................",
        "................",
        "......gggg......",
        "....gggggggg....",
        "...gggggggggg...",
        "..gggkggggkggg..",
        "..gggkggggkggg..",
        "..gggggggggggg..",
        "..GgggggggggggG.",
        "..GGgggggggggGG.",
        "...GGGGGGGGGGG..",
        "................",
        "................",
    ],
    "ghost": [
        "................",
        "................",
        ".....wwwwww.....",
        "....wwwwwwww....",
        "...wwwwwwwwww...",
        "...wwkwwwwkww...",
        "...wwkwwwwkww...",
        "...wwwwwwwwww...",
        "...wwwwwwwwww...",
        "...wwwwwwwwww...",
        "...wwwwwwwwww...",
        "...wwwwwwwwww...",
        "...ssswwwwsss...",
        "...s.ss.ss.ss...",
        "................",
        "................",
    ],
    "candle": [
        "................",
        "................",
        ".......y........",
        "......yoy.......",
        "......yoy.......",
        ".......y........",
        ".......k........",
        "......cccc......",
        "......cccc......",
        "......cccc......",
        "......cccc......",
        "......cccc......",
        "......cccc......",
        ".....ssssss.....",
        "................",
        "................",
    ],
}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row_id, (name, rows) in enumerate(SPRITES.items(), start=1):
        a = np.zeros((16, 16, 4), np.uint8)
        for y, line in enumerate(rows):
            assert len(line) == 16, (name, y)
            for x, ch in enumerate(line):
                if ch != ".":
                    a[y, x] = (*COLORS[ch], 255)
        Image.fromarray(a).save(HERE / f"{name}.png")
        manifest.append(
            {"row_id": row_id, "subject": name, "caption": f"a {name}", "path": f"{name}.png"}
        )
    (HERE / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))


if __name__ == "__main__":
    main()
