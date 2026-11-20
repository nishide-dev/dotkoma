"""Tiling probe: several sprites of one action in one generated video.

Experiment tool for docs/experiments/tiling-01.md, not part of the pipeline yet.

  build: from a prepared single-sprite run (e.g. work/tile-1), write tiled runs
         (work/tile-2: 2x1 at the same scale, work/tile-8: 4x2 at half scale) whose
         jobs.jsonl the normal generate stage can run.
  split: after generate, turn every tiled video into one virtual single-sprite job
         per member (work/<run>-members) so the normal finalize stage can judge each
         sprite on its own grid.

    uv run python scripts/tiling_probe.py build work/tile-1
    uv run python scripts/tiling_probe.py split work/tile-2 work/tile-8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from dotkoma.imaging import SPRITE, Layout, choose_key_color, load_rgba, save_png  # noqa: E402
from dotkoma.records import (  # noqa: E402
    append_jsonl,
    derive_seed,
    job_id,
    read_jsonl,
    write_jsonl_atomic,
)

STYLE = " Static camera, fixed viewpoint, same colors, crisp pixel art."
MOTION = {
    "flicker": "In every sprite only the flame flickers gently; the sprites themselves stay "
    "still and do not move.",
    "steam": "Small wisps of steam rise from every sprite; the sprites themselves stay still "
    "and do not move.",
}
KEYS = ("#00ff00", "#ff00ff", "#0000ff", "#00ffff", "#ffff00", "#ff0000")
# name: (columns, rows, scale, gap in px). Gaps are wider than one cell so a neighbour
# never lands in a member's edge_cut ring.
LAYOUTS = {"tile-2": (2, 1, 24, 32), "tile-8": (4, 2, 12, 16)}


def place(canvas_w: int, canvas_h: int, cols: int, rows: int, scale: int, gap: int):
    size = SPRITE * scale
    w, h = cols * size + (cols - 1) * gap, rows * size + (rows - 1) * gap
    assert w <= canvas_w and h <= canvas_h, (w, h)
    x0, y0 = (canvas_w - w) // 2, (canvas_h - h) // 2
    return [
        Layout(canvas_w, canvas_h, scale, x0 + c * (size + gap), y0 + r * (size + gap))
        for r in range(rows)
        for c in range(cols)
    ]


def build(base: Path) -> None:
    jobs = read_jsonl(base / "jobs.jsonl")
    for name, (cols, rows, scale, gap) in LAYOUTS.items():
        run = base.parent / name
        (run / "inputs").mkdir(parents=True, exist_ok=True)
        out = []
        for action in sorted({j["requested_action"] for j in jobs}):
            members = [j for j in jobs if j["requested_action"] == action]
            for t in range(0, len(members), cols * rows):
                group = members[t : t + cols * rows]
                inf = group[0]["inference"]
                slots = place(inf["width"], inf["height"], cols, rows, scale, gap)
                sprites = [load_rgba(base / m["input_png"]) for m in group]
                key = choose_key_color(np.concatenate(sprites, axis=0), KEYS)
                canvas = np.empty((inf["height"], inf["width"], 3), np.uint8)
                canvas[:] = tuple(int(key[i : i + 2], 16) for i in (1, 3, 5))
                for sp, lay in zip(sprites, slots, strict=False):
                    s = lay.scale
                    big = np.repeat(np.repeat(sp, s, axis=0), s, axis=1).astype(np.float32)
                    a = big[..., 3:4] / 255.0
                    region = canvas[lay.y0 : lay.y0 + SPRITE * s, lay.x0 : lay.x0 + SPRITE * s]
                    region[:] = (big[..., :3] * a + region * (1 - a)).round().astype(np.uint8)
                names = ", ".join(f"a {m['source']['subject']}" for m in group)
                prompt = (
                    f"Pixel art sprites arranged in a grid on a flat solid background: {names}. "
                    + MOTION[action]
                    + STYLE
                )
                rows_ids = [m["source"]["row_id"] for m in group]
                seed = derive_seed(group[0]["source"], rows_ids[0], f"{name}:{action}")
                spec = {
                    "tile": name,
                    "members": [m["job_id"] for m in group],
                    "slots": [s.to_dict() for s in slots[: len(group)]],
                    "prompt": prompt,
                    "seed": seed,
                    "model": group[0]["model"],
                    "inference": inf,
                    "key_color": key,
                }
                jid = job_id(spec)
                ref = run / "inputs" / f"{jid}_ref.png"
                save_png(ref, canvas)
                out.append(
                    {
                        "job_id": jid,
                        "requested_action": action,
                        "prompt": prompt,
                        "negative_prompt": group[0]["negative_prompt"],
                        "seed": seed,
                        "model": group[0]["model"],
                        "inference": inf,
                        "key_color": key,
                        "reference_png": ref.relative_to(run).as_posix(),
                        "tile": {"members": group, "slots": spec["slots"], "base": str(base)},
                    }
                )
        write_jsonl_atomic(run / "jobs.jsonl", out)
        print(name, len(out), "videos for", sum(len(j["tile"]["members"]) for j in out), "sprites")


def split(run: Path) -> None:
    out = run.parent / f"{run.name}-members"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    results = {r["job_id"]: r for r in read_jsonl(run / "results.jsonl")}
    members, base = [], None
    if (out / "results.jsonl").exists():
        (out / "results.jsonl").unlink()
    for j in read_jsonl(run / "jobs.jsonl"):
        r = results.get(j["job_id"])
        if not r or r["status"] != "ok":
            continue
        base = Path(j["tile"]["base"])
        for k, (m, slot) in enumerate(zip(j["tile"]["members"], j["tile"]["slots"], strict=True)):
            mid = f"{j['job_id']}-{k}"
            link = out / "raw" / f"{mid}.mp4"
            if not link.exists():
                os.symlink((run / "raw" / f"{j['job_id']}.mp4").resolve(), link)
            members.append({**m, "job_id": mid, "layout": slot, "key_color": j["key_color"]})
            append_jsonl(out / "results.jsonl", {"job_id": mid, "status": "ok"})
    if base and not (out / "inputs").exists():
        os.symlink((base / "inputs").resolve(), out / "inputs")
    write_jsonl_atomic(out / "jobs.jsonl", members)
    print(out, len(members), "member samples")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "split"])
    p.add_argument("runs", nargs="+", type=Path)
    a = p.parse_args()
    for r in a.runs:
        build(r) if a.cmd == "build" else split(r)


if __name__ == "__main__":
    main()
