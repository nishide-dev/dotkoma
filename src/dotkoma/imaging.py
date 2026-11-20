"""Pixel operations: reference images, background keying, 16x16 reduction, GIF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SPRITE = 16


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def opaque_colors(rgba: np.ndarray) -> np.ndarray:
    """Unique RGB colours of opaque pixels, shape (N, 3) uint8."""
    px = rgba[rgba[..., 3] > 0][:, :3]
    if len(px) == 0:
        return np.zeros((0, 3), np.uint8)
    return np.unique(px, axis=0)


def choose_key_color(rgba: np.ndarray, candidates: tuple[str, ...]) -> str:
    """Candidate background colour farthest from every colour in the sprite."""
    colors = opaque_colors(rgba).astype(np.float32)
    best, best_d = candidates[0], -1.0
    for c in candidates:
        rgb = np.array(hex_to_rgb(c), np.float32)
        d = float(np.min(np.linalg.norm(colors - rgb, axis=1))) if len(colors) else 1e9
        if d > best_d:
            best, best_d = c, d
    return best


@dataclass(frozen=True)
class Layout:
    """Where the 16x16 grid sits on the reference canvas."""

    canvas_w: int
    canvas_h: int
    scale: int
    x0: int
    y0: int

    @classmethod
    def centered(cls, w: int, h: int, scale: int) -> Layout:
        size = SPRITE * scale
        return cls(w, h, scale, (w - size) // 2, (h - size) // 2)

    def to_dict(self) -> dict:
        return {
            "canvas_w": self.canvas_w,
            "canvas_h": self.canvas_h,
            "scale": self.scale,
            "x0": self.x0,
            "y0": self.y0,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Layout:
        return cls(d["canvas_w"], d["canvas_h"], d["scale"], d["x0"], d["y0"])


def make_reference(rgba: np.ndarray, layout: Layout, key_color: str) -> np.ndarray:
    """Nearest-neighbour upscale onto a flat key-colour canvas. Returns RGB uint8."""
    assert rgba.shape == (SPRITE, SPRITE, 4), rgba.shape
    s = layout.scale
    big = np.repeat(np.repeat(rgba, s, axis=0), s, axis=1)
    canvas = np.empty((layout.canvas_h, layout.canvas_w, 3), np.uint8)
    canvas[:] = hex_to_rgb(key_color)
    region = canvas[layout.y0 : layout.y0 + SPRITE * s, layout.x0 : layout.x0 + SPRITE * s]
    a = big[..., 3:4].astype(np.float32) / 255.0
    region[:] = (big[..., :3] * a + region * (1 - a)).round().astype(np.uint8)
    return canvas


# ---------------------------------------------------------------- keying


def measure_background(frame: np.ndarray, ring: int = 4) -> np.ndarray:
    """Median colour of the outer ring of a frame (the model may drift the key)."""
    top, bot = frame[:ring].reshape(-1, 3), frame[-ring:].reshape(-1, 3)
    left, right = frame[:, :ring].reshape(-1, 3), frame[:, -ring:].reshape(-1, 3)
    return np.median(np.concatenate([top, bot, left, right]), axis=0)


def foreground_mask(frame: np.ndarray, bg: np.ndarray, tol: float) -> np.ndarray:
    d = np.linalg.norm(frame.astype(np.float32) - bg.astype(np.float32), axis=-1)
    return d > tol


@dataclass(frozen=True)
class GridBox:
    """The 16x16 grid in video pixel coordinates (video may be resized vs canvas)."""

    x0: float
    y0: float
    cell_w: float
    cell_h: float

    @classmethod
    def from_layout(cls, layout: Layout, video_w: int, video_h: int) -> GridBox:
        sx, sy = video_w / layout.canvas_w, video_h / layout.canvas_h
        return cls(layout.x0 * sx, layout.y0 * sy, layout.scale * sx, layout.scale * sy)

    def cell(self, i: int, j: int, inset: float = 0.25) -> tuple[int, int, int, int]:
        """Pixel bounds (y0, y1, x0, x1) of cell row i, col j, shrunk by ``inset``."""
        y0 = self.y0 + (i + inset) * self.cell_h
        y1 = self.y0 + (i + 1 - inset) * self.cell_h
        x0 = self.x0 + (j + inset) * self.cell_w
        x1 = self.x0 + (j + 1 - inset) * self.cell_w
        y0i, x0i = int(np.floor(y0)), int(np.floor(x0))
        return y0i, max(y0i + 1, int(np.ceil(y1))), x0i, max(x0i + 1, int(np.ceil(x1)))

    def outer(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.y0)),
            int(round(self.y0 + SPRITE * self.cell_h)),
            int(round(self.x0)),
            int(round(self.x0 + SPRITE * self.cell_w)),
        )


def reduce_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    box: GridBox,
    coverage_threshold: float,
) -> np.ndarray:
    """Reduce one video frame to 16x16 RGBA using the same grid for every frame."""
    out = np.zeros((SPRITE, SPRITE, 4), np.uint8)
    H, W = mask.shape
    for i in range(SPRITE):
        for j in range(SPRITE):
            y0, y1, x0, x1 = box.cell(i, j)
            y0, x0 = max(y0, 0), max(x0, 0)
            y1, x1 = min(y1, H), min(x1, W)
            if y1 <= y0 or x1 <= x0:
                continue
            m = mask[y0:y1, x0:x1]
            if m.mean() < coverage_threshold:
                continue
            px = frame[y0:y1, x0:x1][m]
            out[i, j, :3] = np.median(px, axis=0).round().astype(np.uint8)
            out[i, j, 3] = 255
    return out


def snap_to_palette(rgba: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map opaque pixels to the nearest colour of a fixed (global) palette."""
    if len(palette) == 0:
        return rgba
    out = rgba.copy()
    opaque = out[..., 3] > 0
    px = out[opaque][:, :3].astype(np.float32)
    d = np.linalg.norm(px[:, None, :] - palette[None, :, :].astype(np.float32), axis=-1)
    out[..., :3][opaque] = palette[np.argmin(d, axis=1)]
    return out


def stable_new_colors(
    frames: list[np.ndarray],
    palette: np.ndarray,
    min_distance: float,
    min_frame_share: float,
    min_pixels: int,
) -> np.ndarray:
    """Colours absent from ``palette`` that recur across the loop, shape (K, 3) uint8.

    A video model may add a colour the sprite never had (the white of an opened
    eye). Snapping to the source palette alone would erase it. Opaque pixels
    farther than ``min_distance`` from every palette colour are grouped greedily
    (in frame / raster order, so deterministic); a group is kept when it covers at
    least ``min_pixels`` pixels in at least ``min_frame_share`` of the frames.
    One-off colours (a passing artefact, a frame of transformation) are dropped
    and later snapped back to the palette.
    """
    pal = palette.astype(np.float32).reshape(-1, 3)
    groups: list[dict] = []  # {"sum", "members", "frames": Counter-like dict}
    for k, f in enumerate(frames):
        px = f[f[..., 3] > 0][:, :3].astype(np.float32)
        if len(pal):
            px = px[np.linalg.norm(px[:, None] - pal[None], axis=-1).min(axis=1) > min_distance]
        for c in px:
            for g in groups:
                if np.linalg.norm(g["sum"] / len(g["members"]) - c) <= min_distance / 2:
                    break
            else:
                g = {"sum": np.zeros(3, np.float32), "members": [], "frames": {}}
                groups.append(g)
            g["sum"] += c
            g["members"].append(c)
            g["frames"][k] = g["frames"].get(k, 0) + 1
    need = max(1, int(np.ceil(min_frame_share * len(frames))))
    kept = [
        np.median(np.array(g["members"]), axis=0).round()
        for g in groups
        if sum(n >= min_pixels for n in g["frames"].values()) >= need
    ]
    return np.array(kept, np.uint8).reshape(-1, 3)


def edge_cut_fraction(mask: np.ndarray, box: GridBox) -> float:
    """Share of the ring just outside the grid that is foreground.

    Foreground there would be cut off by the 16x16 crop (e.g. a jump that
    leaves the frame of the original sprite).
    """
    H, W = mask.shape
    y0, y1, x0, x1 = box.outer()
    w = max(1, int(round(box.cell_w)))
    ring = np.zeros_like(mask)
    ry0, ry1 = max(0, y0 - w), min(H, y1 + w)
    rx0, rx1 = max(0, x0 - w), min(W, x1 + w)
    ring[ry0:ry1, rx0:rx1] = True
    ring[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)] = False
    n = ring.sum()
    return float((mask & ring).sum() / n) if n else 0.0


def border_foreground_fraction(mask: np.ndarray, ring: int = 4) -> float:
    """Foreground share on the outer frame border: the key was not kept stable."""
    b = np.concatenate(
        [mask[:ring].ravel(), mask[-ring:].ravel(), mask[:, :ring].ravel(), mask[:, -ring:].ravel()]
    )
    return float(b.mean())


# ---------------------------------------------------------------- io


def load_rgba(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA")).copy()


def save_png(path: str | Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path, optimize=True)


def write_gif(
    path: str | Path, frames: list[np.ndarray], durations_ms: list[int], scale: int
) -> None:
    """Transparent looping GIF with a single shared palette (index 0 = transparent)."""
    colors = sorted({tuple(c) for f in frames for c in f[f[..., 3] > 0][:, :3].tolist()})
    if len(colors) > 255:
        raise ValueError(f"{len(colors)} colours exceed the GIF palette")
    lut = {c: k + 1 for k, c in enumerate(colors)}
    flat = [0, 0, 0]
    for c in colors:
        flat.extend(c)
    flat.extend([0, 0, 0] * (256 - 1 - len(colors)))

    ims = []
    for f in frames:
        idx = np.zeros(f.shape[:2], np.uint8)
        ys, xs = np.nonzero(f[..., 3] > 0)
        for y, x in zip(ys, xs, strict=True):
            idx[y, x] = lut[tuple(f[y, x, :3].tolist())]
        idx = np.repeat(np.repeat(idx, scale, axis=0), scale, axis=1)
        im = Image.fromarray(idx, mode="P")
        im.putpalette(flat)
        im.info["transparency"] = 0
        ims.append(im)
    ims[0].save(
        path,
        save_all=True,
        append_images=ims[1:],
        duration=durations_ms,
        loop=0,
        disposal=2,
        transparency=0,
        optimize=False,
    )
