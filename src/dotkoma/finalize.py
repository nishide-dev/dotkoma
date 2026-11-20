"""finalize (local CPU): raw mp4 -> 16x16 RGBA loop + timing + checks + preview GIF.

Outputs per job under ``final/<job_id>/``:
  frames/000.png ...   ordered 16x16 RGBA frames (the canonical training data)
  sample.json          timing, provenance, machine checks
  preview.gif          viewing copy, regenerated from the frames

``finalize.jsonl`` lists every job's machine verdict. Human decisions live in
``review.jsonl`` (written by a person, never touched here).

Frame selection defaults to N evenly spaced frames over the whole clip. To
override after looking at a clip, add a line to ``selections.jsonl``:
  {"job_id": "...", "start": 3, "end": 40}            # window, N frames
  {"job_id": "...", "indices": [3, 9, 15, 21]}         # explicit frames
"""

from __future__ import annotations

import logging
import shutil
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .config import FinalizeConfig
from .generate import raw_path
from .imaging import (
    SPRITE,
    GridBox,
    Layout,
    border_foreground_fraction,
    edge_cut_fraction,
    foreground_mask,
    load_rgba,
    measure_background,
    opaque_colors,
    reduce_frame,
    save_png,
    snap_to_palette,
    write_gif,
)
from .records import latest_by_job, read_jsonl, stable_hash, write_json_atomic, write_jsonl_atomic
from .video import read_mp4

log = logging.getLogger(__name__)

HARD_FAILS = ("wrong_size", "frame_count", "empty_frame", "all_identical")


# ---------------------------------------------------------------- selection


def select_frames(
    n_frames: int, fps: float, sel: dict | None, count: int
) -> tuple[list[int], list[int]]:
    """Return (source frame indices, display durations in ms)."""
    if sel and "indices" in sel:
        idx = [int(i) for i in sel["indices"]]
        if any(i < 0 or i >= n_frames for i in idx) or len(set(idx)) != len(idx):
            raise ValueError(f"bad indices {idx} for {n_frames} frames")
        gaps = [b - a for a, b in pairwise(idx)]
        gaps.append(int(np.median(gaps)) if gaps else 1)
    else:
        start = int(sel.get("start", 0)) if sel else 0
        end = int(sel.get("end", n_frames)) if sel else n_frames
        if not (0 <= start < end <= n_frames):
            raise ValueError(f"bad window [{start}, {end}) for {n_frames} frames")
        count = int(sel.get("count", count)) if sel else count
        span = end - start
        idx = [start + int(round(k * span / count)) for k in range(count)]
        if len(set(idx)) != len(idx):
            raise ValueError(f"window [{start}, {end}) too short for {count} frames")
        gaps = [b - a for a, b in pairwise(idx)] + [end - idx[-1]]
    durations = [_ms(g, fps) for g in gaps]
    return idx, durations


def _ms(gap_frames: int, fps: float) -> int:
    # GIF timing is in 1/100 s; browsers clamp < 20 ms
    return max(20, int(round(gap_frames / fps * 100)) * 10)


# ---------------------------------------------------------------- checks


def frame_diff(a: np.ndarray, b: np.ndarray, tol: float = 24.0) -> int:
    """Number of the 256 pixels that visibly change between two RGBA frames."""
    alpha = (a[..., 3] > 0) != (b[..., 3] > 0)
    both = (a[..., 3] > 0) & (b[..., 3] > 0)
    rgb = (
        np.linalg.norm(a[..., :3].astype(np.float32) - b[..., :3].astype(np.float32), axis=-1) > tol
    )
    return int((alpha | (both & rgb)).sum())


def run_checks(
    frames: list[np.ndarray],
    source: np.ndarray,
    edge_cut: float,
    border_fg: float,
    cfg: FinalizeConfig,
) -> dict[str, Any]:
    n = len(frames)
    adjacent = [frame_diff(a, b) for a, b in pairwise(frames)]
    loop = frame_diff(frames[-1], frames[0]) if n > 1 else 0
    typical = float(np.median(adjacent)) if adjacent else 0.0
    m = {
        "num_frames": n,
        "adjacent_diffs": adjacent,
        "loop_diff": loop,
        "edge_cut_fraction": round(edge_cut, 4),
        "border_foreground_fraction": round(border_fg, 4),
        "first_frame_source_diff": frame_diff(frames[0], source),
        "opaque_pixels": [int((f[..., 3] > 0).sum()) for f in frames],
    }
    flags = []
    if any(f.shape != (SPRITE, SPRITE, 4) for f in frames):
        flags.append("wrong_size")
    if not (cfg.min_frames <= n <= cfg.max_frames):
        flags.append("frame_count")
    if any(c == 0 for c in m["opaque_pixels"]):
        flags.append("empty_frame")
    if n > 1 and max(adjacent + [loop]) == 0:
        flags.append("all_identical")
    if loop > cfg.loop_jump_factor * max(1.0, typical):
        flags.append("loop_jump")
    if edge_cut > cfg.edge_cut_fraction:
        flags.append("edge_cut")
    if border_fg > cfg.background_unstable_fraction:
        flags.append("background_unstable")
    if m["first_frame_source_diff"] > 0.25 * max(1, int((source[..., 3] > 0).sum())):
        flags.append("source_mismatch")
    m["flags"] = flags
    m["auto_verdict"] = (
        "reject" if any(f in HARD_FAILS for f in flags) else ("flagged" if flags else "candidate")
    )
    return m


# ---------------------------------------------------------------- per job


def finalize_job(run_dir: Path, job: dict, cfg: FinalizeConfig, sel: dict | None) -> dict[str, Any]:
    jid = job["job_id"]
    video, fps = read_mp4(raw_path(run_dir, jid))
    fps = fps or float(job["inference"]["fps"])
    layout = Layout.from_dict(job["layout"])
    H, W = video.shape[1:3]
    box = GridBox.from_layout(layout, W, H)
    source = load_rgba(run_dir / job["input_png"])

    idx, durations = select_frames(len(video), fps, sel, cfg.num_frames)
    bg = measure_background(video[0])  # one background colour for the whole clip
    palette = opaque_colors(source) if cfg.palette == "source" else np.zeros((0, 3), np.uint8)

    frames, edge, border = [], 0.0, 0.0
    for i in idx:
        mask = foreground_mask(video[i], bg, cfg.bg_tolerance)
        f = reduce_frame(video[i], mask, box, cfg.coverage_threshold)
        frames.append(snap_to_palette(f, palette))
        edge = max(edge, edge_cut_fraction(mask, box))
        border = max(border, border_foreground_fraction(mask))

    checks = run_checks(frames, source, edge, border, cfg)
    pp_id = stable_hash({"finalize": cfg.__dict__, "selection": sel or {}}, 12)

    out = run_dir / "final" / jid
    tmp = run_dir / "final" / f".{jid}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "frames").mkdir(parents=True)
    for k, f in enumerate(frames):
        save_png(tmp / "frames" / f"{k:03d}.png", f)
    write_gif(tmp / "preview.gif", frames, durations, cfg.preview_scale)
    sample = {
        "job_id": jid,
        "postprocess_id": pp_id,
        "source": job["source"],
        "input_png": job["input_png"],
        "split": job["split"],
        "requested_action": job["requested_action"],
        "generation_method": job["generation_method"],
        "prompt_method": job["prompt_method"],
        "prompt_variant": job["prompt_variant"],
        "frames": [f"frames/{k:03d}.png" for k in range(len(frames))],
        "durations_ms": durations,
        "loop": True,
        "source_frame_indices": idx,
        "source_fps": fps,
        "background_rgb": [round(float(c), 1) for c in bg],
        "checks": checks,
    }
    write_json_atomic(tmp / "sample.json", sample)  # written last: marks completeness
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return sample


def finalize(run_dir: str | Path, cfg: FinalizeConfig) -> list[dict]:
    run_dir = Path(run_dir)
    jobs = read_jsonl(run_dir / "jobs.jsonl")
    results = latest_by_job(read_jsonl(run_dir / "results.jsonl"))
    selections = latest_by_job(read_jsonl(run_dir / "selections.jsonl", unterminated_tail="parse"))

    index = []
    for j in jobs:
        jid = j["job_id"]
        base = {
            "job_id": jid,
            "subject": j["source"]["subject"],
            "requested_action": j["requested_action"],
            "prompt_method": j["prompt_method"],
            "prompt_variant": j["prompt_variant"],
        }
        r = results.get(jid)
        if not r or r["status"] != "ok" or not raw_path(run_dir, jid).exists():
            status = r["status"] if r else "not_generated"
            index.append({**base, "status": "generation_" + status if r else status})
            continue
        try:
            s = finalize_job(run_dir, j, cfg, selections.get(jid))
            c = s["checks"]
            index.append(
                {
                    **base,
                    "status": "ok",
                    "postprocess_id": s["postprocess_id"],
                    "auto_verdict": c["auto_verdict"],
                    "flags": c["flags"],
                }
            )
        except Exception as e:
            log.error("finalize %s failed: %s", jid, e)
            index.append({**base, "status": "finalize_failed", "error": f"{type(e).__name__}: {e}"})
    write_jsonl_atomic(run_dir / "finalize.jsonl", index)
    return index
