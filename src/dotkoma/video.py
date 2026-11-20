"""Minimal mp4 read/write via PyAV (bundled FFmpeg, no system dependency)."""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def write_mp4(
    path: str | Path,
    frames: np.ndarray | list[np.ndarray],
    fps: int,
    pix_fmt: str = "yuv444p",
    crf: int = 10,
) -> None:
    """Write RGB uint8 frames (F, H, W, 3) atomically: temp file then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.partial{path.suffix}")
    frames = [np.ascontiguousarray(f) for f in frames]
    h, w = frames[0].shape[:2]
    try:
        with av.open(str(tmp), mode="w") as c:
            s = c.add_stream("libx264", rate=Fraction(fps))
            s.width, s.height, s.pix_fmt = w, h, pix_fmt
            s.options = {"crf": str(crf), "preset": "medium"}
            for f in frames:
                for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                    c.mux(pkt)
            for pkt in s.encode():
                c.mux(pkt)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_mp4(path: str | Path) -> tuple[np.ndarray, float]:
    """Return (frames (F, H, W, 3) uint8, fps)."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 0)
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
    if not frames:
        raise ValueError(f"{path}: no video frames")
    return np.stack(frames), fps
