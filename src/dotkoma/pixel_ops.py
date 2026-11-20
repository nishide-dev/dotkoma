"""Whole-sprite motion made by moving the 16x16 sprite itself (no video model).

Pilots 01/02 showed the video models do not keep a sprite intact while moving it
as a whole (hop / bob turned into redraws, shadows or a different object). These
motions are exact integer translations of the source, so they are built here.

An action opts in with ``generation_method = "pixel_ops"`` and a ``pixel_ops`` table:

  kind = "bob" | "hop" | "sway" | "custom"
  amplitude = 1          # pixels (bob / hop / sway): peak offset
  frames = 6             # loop length
  ground = 2             # hop only: frames resting before the jump
  frame_ms = 100         # display time of each frame
  offsets = [[dx, dy], ...]  # only for "custom"; frames is then len(offsets)

The expanded offsets and durations (not the preset name) go into job_id, so
changing a preset formula yields new jobs.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

KINDS = ("bob", "hop", "sway", "custom")


def _round(x: float) -> int:
    """Round half away from zero (Python's round() sends 0.5 to 0)."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def expand(spec: dict[str, Any]) -> dict[str, list]:
    """Preset -> explicit per-frame offsets [[dx, dy], ...] and durations in ms."""
    kind = spec["kind"]
    if kind == "custom":
        offsets = [[int(dx), int(dy)] for dx, dy in spec["offsets"]]
    else:
        n, a = int(spec.get("frames", 6)), int(spec.get("amplitude", 1))
        if kind == "bob":  # float up, back, down, back (y grows downwards)
            offsets = [[0, -_round(a * math.sin(2 * math.pi * k / n))] for k in range(n)]
        elif kind == "hop":  # rest on the ground, then a parabolic jump that lands back
            g = int(spec.get("ground", 2))
            if not 1 <= g <= n - 2:
                raise ValueError(f"hop: ground={g} needs 1 <= ground <= frames - 2")
            m = n - g
            arc = [4 * t * (1 - t) for t in ((j + 1) / (m + 1) for j in range(m))]
            offsets = [[0, 0]] * g + [[0, -_round(a * h)] for h in arc]
        else:  # sway: right, back, left, back
            offsets = [[_round(a * math.sin(2 * math.pi * k / n)), 0] for k in range(n)]
    offsets = [[dx + 0, dy + 0] for dx, dy in offsets]  # normalise -0 to 0
    ms = spec.get("frame_ms", 100)
    durations = [int(x) for x in ms] if isinstance(ms, list) else [int(ms)] * len(offsets)
    if len(durations) != len(offsets):
        raise ValueError(f"{len(durations)} durations for {len(offsets)} frames")
    return {"offsets": offsets, "durations_ms": durations}


def validate(spec: dict[str, Any]) -> str | None:
    """Return an error message, or None if the spec is usable."""
    if spec.get("kind") not in KINDS:
        return f"pixel_ops.kind must be one of {KINDS}, got {spec.get('kind')!r}"
    if spec["kind"] == "custom" and not spec.get("offsets"):
        return "pixel_ops.offsets is required for kind = 'custom'"
    try:
        e = expand(spec)
    except (KeyError, TypeError, ValueError) as err:
        return f"pixel_ops: {err}"
    if len(e["offsets"]) < 2:
        return "pixel_ops needs at least 2 frames"
    if all(o == e["offsets"][0] for o in e["offsets"]):
        return "pixel_ops offsets never move the sprite"
    if any(d <= 0 for d in e["durations_ms"]):
        return "pixel_ops.frame_ms must be positive"
    return None


def translate(rgba: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move an RGBA image by (dx, dy); vacated pixels become transparent."""
    out = np.zeros_like(rgba)
    h, w = rgba.shape[:2]
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = rgba[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def render(source: np.ndarray, offsets: list[list[int]]) -> tuple[list[np.ndarray], float]:
    """Frames for each offset, and the largest share of opaque pixels pushed off the grid."""
    opaque = max(1, int((source[..., 3] > 0).sum()))
    frames, lost = [], 0.0
    for dx, dy in offsets:
        f = translate(source, dx, dy)
        frames.append(f)
        lost = max(lost, 1.0 - int((f[..., 3] > 0).sum()) / opaque)
    return frames, lost


def margins(source: np.ndarray) -> dict[str, int]:
    """Transparent rows / columns between the sprite and each edge of the grid."""
    ys, xs = np.nonzero(source[..., 3] > 0)
    h, w = source.shape[:2]
    if len(ys) == 0:
        return {"top": h, "bottom": h, "left": w, "right": w}
    return {
        "top": int(ys.min()),
        "bottom": int(h - 1 - ys.max()),
        "left": int(xs.min()),
        "right": int(w - 1 - xs.max()),
    }


def fits(source: np.ndarray, offsets: list[list[int]]) -> bool:
    """True if every offset keeps the whole sprite on the grid."""
    m = margins(source)
    return all(
        -m["left"] <= dx <= m["right"] and -m["top"] <= dy <= m["bottom"] for dx, dy in offsets
    )


def sprite_digest(rgba: np.ndarray) -> str:
    """sha256 of the RGBA pixels: ties a pixel_ops job to the exact sprite it moves."""
    return hashlib.sha256(np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()).hexdigest()
