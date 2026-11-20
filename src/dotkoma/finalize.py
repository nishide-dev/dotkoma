"""finalize (local CPU): raw mp4 -> 16x16 RGBA loop + timing + checks + preview GIF.

Outputs per job under ``final/<job_id>/``:
  frames/000.png ...   ordered 16x16 RGBA frames (the canonical training data)
  sample.json          timing, provenance, machine checks
  preview.gif          viewing copy, regenerated from the frames

``finalize.jsonl`` lists every job's machine verdict. Human decisions live in
``review.jsonl`` (written by a person, never touched here).

Jobs with ``generation_method = "pixel_ops"`` have no video: their frames are
rendered here from the source sprite (see ``pixel_ops.py``) and go through the
same checks and output format.

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

from . import pixel_ops
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
    rgb_to_hex,
    save_png,
    snap_to_palette,
    stable_new_colors,
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


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = a[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def identity_drift(frame: np.ndarray, source: np.ndarray, max_shift: int) -> float:
    """Share of the source's pixels that differ from ``frame`` under the best shift.

    Translation (hop, bob) is free; redrawing the sprite is not. In the pilot,
    frames that kept the sprite scored <= 0.23 and redrawn / transformed ones >= 0.35.
    """
    opaque = int((source[..., 3] > 0).sum())
    r = range(-max_shift, max_shift + 1)
    best = opaque
    for dy in r:
        for dx in r:
            shifted = _shift(source, dy, dx)
            # source pixels pushed off the 16x16 grid are missing from the frame: count them
            lost = opaque - int((shifted[..., 3] > 0).sum())
            best = min(best, frame_diff(frame, shifted) + lost)
    return best / max(1, opaque)


def lost_detail(frames: list[np.ndarray], source: np.ndarray, max_shift: int) -> int:
    """Source pixels missing from every frame after the first, allowing translation.

    Frame 0 reproduces the reference, so a detail the model drops right away (a
    cookie on a mug) shows up as pixels absent from all later frames. Each frame is
    compared at the shift that keeps most of the source, so a hop or sway is not a
    loss; pixels a shift pushes off the grid count as missing.
    """
    opaque = source[..., 3] > 0
    if len(frames) < 2 or not opaque.any():
        return 0
    missing_everywhere = opaque.copy()
    r = range(-max_shift, max_shift + 1)
    for f in frames[1:]:
        best = None
        for dy in r:
            for dx in r:
                # frame pixel at (y + dy, x + dx) should hold source pixel (y, x)
                back = _shift(f, -dy, -dx)
                miss = opaque & (back[..., 3] == 0)
                if best is None or miss.sum() < best.sum():
                    best = miss
        missing_everywhere &= best
    return int(missing_everywhere.sum())


def run_checks(
    frames: list[np.ndarray],
    source: np.ndarray,
    edge_cut: float,
    border_fg: float,
    cfg: FinalizeConfig,
    motion: str = "global",
) -> dict[str, Any]:
    """Machine checks. ``flags`` decide the verdict; ``notes`` are for the reviewer only."""
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
        "identity_drift": [
            round(identity_drift(f, source, cfg.identity_max_shift), 3) for f in frames
        ],
        "opaque_pixels": [int((f[..., 3] > 0).sum()) for f in frames],
        # pixels changed per step around the loop (including last -> first)
        "motion_median": float(np.median(adjacent + [loop])) if n > 1 else 0.0,
        "detail_lost_pixels": lost_detail(frames, source, cfg.identity_max_shift),
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
    notes = []
    if edge_cut > cfg.edge_cut_fraction:
        if motion == "local" and edge_cut <= cfg.local_edge_limit():
            notes.append("edge_overflow")  # a moving part pokes out a little and is cropped
        else:
            flags.append("edge_cut")
    if border_fg > cfg.background_unstable_fraction:
        flags.append("background_unstable")
    if max(m["identity_drift"]) > cfg.identity_drift_threshold:
        flags.append("identity_drift")
    if n > 1 and "all_identical" not in flags and m["motion_median"] < cfg.low_motion_pixels:
        flags.append("low_motion")  # moves in a step or two, static the rest of the loop
    if m["detail_lost_pixels"] > cfg.detail_lost_pixels:
        flags.append("detail_lost")
    m["flags"] = flags
    m["notes"] = notes
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
    # Wan reproduces frame 0 from the reference and then drifts the flat key colour
    # slightly, so "per_frame" re-measures it on each frame's border (same rule, own value).
    bg0 = measure_background(video[0])

    frames, backgrounds, drift, edge, border = [], [], 0.0, 0.0, 0.0
    for i in idx:
        measured = measure_background(video[i])  # always measured, so drift is real
        drift = max(drift, float(np.linalg.norm(measured - bg0)))
        bg = measured if cfg.background == "per_frame" else bg0
        backgrounds.append([round(float(c), 1) for c in bg])
        mask = foreground_mask(video[i], bg, cfg.bg_tolerance)
        frames.append(reduce_frame(video[i], mask, box, cfg.coverage_threshold))
        edge = max(edge, edge_cut_fraction(mask, box))
        border = max(border, border_foreground_fraction(mask))

    # one palette for the whole loop, decided after seeing every frame
    palette, added = loop_palette(frames, source, cfg)
    # the same membership rule stable_new_colors grouped with
    d = cfg.new_color_distance
    frames = [snap_to_palette(f, palette, added, d / 2, novelty=d) for f in frames]

    checks = run_checks(frames, source, edge, border, cfg, job.get("motion", "global"))
    checks["background_drift"] = round(drift, 1)
    checks["added_colors"] = [rgb_to_hex(c) for c in added]
    # motion decides edge_cut vs edge_overflow, and it can change without a new job_id
    pp_id = stable_hash(
        {"finalize": cfg.__dict__, "selection": sel or {}, "motion": job.get("motion", "global")},
        12,
    )
    extra = {
        "source_frame_indices": idx,
        "source_fps": fps,
        "background_rgb": backgrounds,
    }
    return _write_sample(run_dir, job, frames, durations, checks, pp_id, extra, cfg)


def loop_palette(
    frames: list[np.ndarray], source: np.ndarray, cfg: FinalizeConfig
) -> tuple[np.ndarray, np.ndarray]:
    """(source-derived palette, colours added beyond it) for every frame of the loop."""
    none = np.zeros((0, 3), np.uint8)
    if cfg.palette == "none":
        return none, none
    base = opaque_colors(source)
    if cfg.palette == "source":
        return base, none
    added = stable_new_colors(
        frames,
        base,
        cfg.new_color_distance,
        cfg.new_color_min_frame_share,
        cfg.new_color_min_pixels,
    )
    return base, added


def finalize_pixel_ops_job(run_dir: Path, job: dict, cfg: FinalizeConfig) -> dict[str, Any]:
    """Frames are exact translations of the source sprite; no video is involved."""
    source = load_rgba(run_dir / job["input_png"])
    if pixel_ops.sprite_digest(source) != job["sprite_sha256"]:
        # e.g. prepare was interrupted after rewriting inputs/ but before jobs.jsonl
        raise ValueError(f"{job['input_png']} does not match the sprite this job was made for")
    ops = job["pixel_ops"]
    frames, lost = pixel_ops.render(source, ops["offsets"])
    checks = run_checks(frames, source, lost, 0.0, cfg, job.get("motion", "global"))
    # check thresholds and preview_scale still apply (keying and palette settings do not)
    pp_id = stable_hash(
        {"finalize": cfg.__dict__, "pixel_ops": ops, "motion": job.get("motion", "global")}, 12
    )
    extra = {"pixel_ops": ops}
    return _write_sample(run_dir, job, frames, ops["durations_ms"], checks, pp_id, extra, cfg)


def _write_sample(
    run_dir: Path,
    job: dict,
    frames: list[np.ndarray],
    durations: list[int],
    checks: dict[str, Any],
    pp_id: str,
    extra: dict[str, Any],
    cfg: FinalizeConfig,
) -> dict[str, Any]:
    jid = job["job_id"]
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
        **extra,
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
        pixel = j.get("generation_method") == "pixel_ops"
        r = results.get(jid)
        if not pixel and (not r or r["status"] != "ok" or not raw_path(run_dir, jid).exists()):
            status = r["status"] if r else "not_generated"
            index.append({**base, "status": "generation_" + status if r else status})
            continue
        try:
            if pixel:
                s = finalize_pixel_ops_job(run_dir, j, cfg)
            else:
                s = finalize_job(run_dir, j, cfg, selections.get(jid))
            c = s["checks"]
            index.append(
                {
                    **base,
                    "status": "ok",
                    "postprocess_id": s["postprocess_id"],
                    "auto_verdict": c["auto_verdict"],
                    "flags": c["flags"],
                    "notes": c["notes"],
                }
            )
        except Exception as e:
            log.error("finalize %s failed: %s", jid, e)
            index.append({**base, "status": "finalize_failed", "error": f"{type(e).__name__}: {e}"})
    write_jsonl_atomic(run_dir / "finalize.jsonl", index)
    return index
