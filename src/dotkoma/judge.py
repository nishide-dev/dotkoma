"""judge (GPU): a vision-language model checks each finalized loop against its action.

The machine checks in finalize look at numbers (motion size, colour drift, lost
pixels). They cannot tell *which* motion happened: Pilot 03/04 and Tiling 01 have
candidates where steam turned into sparkles, a flask filled with liquid instead of
bubbling, or a fire pit flickered instead of smoking. The judge looks at the
16x16 result and answers per sample:

  motion_matches  the requested motion visibly happens (not another motion, not ~nothing)
  drawing_kept    apart from the moving part, the frames keep the original drawing

``verdict`` is ``accept`` when both are true, else ``reject``.

Input: one image per sample (``sheet``): the source sprite, then the loop's frames,
all enlarged on the same grid side by side, and (``diff_row``, default
"silhouette") a second row marking in magenta every pixel that differs from the
source. Without that row the VLM mostly repeats the instruction back (Judge 01).
A person reviewing samples should look at the same sheet.

File contract (``work/<run>/``):
  finalize.jsonl, final/<job_id>/  read (from finalize)
  judge.jsonl                      append-only, written here; the last ``ok`` record
                                   per request_id counts. Re-running resumes. With a
                                   sync dir, records only the synced copy has are
                                   taken in first (a fresh VM resumes from it).

``request_id`` hashes the frames, the source pixels, the rendered instruction and
the judge settings, so re-finalizing or changing the judge gives new requests.
Machine verdicts (finalize.jsonl), judge verdicts (judge.jsonl) and human review
(review.jsonl) stay in separate files.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import platform
import re
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .imaging import choose_key_color, hex_to_rgb, load_rgba
from .plan import Planner, QwenPlanner
from .records import append_jsonl, canonical_json, copy_atomic, read_jsonl, stable_hash

log = logging.getLogger(__name__)

# What each action must look like, in object language. Overridable per action
# in the judge config ([judge.criteria]).
DEFAULT_CRITERIA = {
    "flicker": "the flame changes shape or brightness; the candle, holder or torch stays the same",
    "steam": "pale wisps of steam rise above the drink or food and fade; the container "
    "stays the same",
    "smoke": "smoke rises from the chimney or the fire and fades; the object stays the same",
    "bubble": "bubbles appear and pop on the liquid surface; the container stays the same",
    "sparkle": "small glints or sparkles appear and fade on the object; its shape and "
    "colours stay the same",
    "flutter": "the cloth of the flag ripples or waves; the pole stays and no new "
    "patterns appear on the cloth",
    "blink": "the eyes close and open again; the rest of the creature stays the same",
    "wag": "the tail swings; the body and the head stay the same",
}

# Shown to the VLM. {subject}, {action}, {criteria}, {n} are filled per sample;
# {row} becomes the sentence describing the diff row (DIFF_ROW_NOTES).
_INSTRUCTION = """\
The image shows a looping pixel-art animation of "{subject}". The first panel, \
labelled S, is the original 16x16 sprite. Panels 1 to {n} are the frames of the loop \
in order; after frame {n} the loop starts again at frame 1. Every panel is enlarged \
by the same amount.{row}

The requested motion is "{action}": {criteria}.

Judge only what is visible in the panels, and be strict:
- motion_matches: does the requested motion clearly happen across the frames? Answer \
false when the frames barely change, or when a different change happens instead (for \
example sparkles or clouds instead of steam, liquid filling up instead of bubbles, \
the whole object changing colour, the object moving or turning).
- drawing_kept: apart from the part that does the motion, do the frames keep the \
original drawing of panel S (same shape, same colours, no part disappearing, nothing \
added except what the motion itself adds)?

Answer with one JSON object and nothing else:
{{"motion_matches": true or false, "drawing_kept": true or false, "reason": "one \
short sentence about what changes across the frames"}}"""
DIFF_ROW_NOTES = {
    "none": "",
    "faded": " In the bottom row, under each frame, a faded copy of S marks in magenta "
    "every pixel where that frame differs from S.",
    "silhouette": " In the bottom row, under each frame, the shape of S is drawn in flat "
    "grey and every pixel where that frame differs from S is marked in magenta (grey is "
    "not a colour of the frame).",
}
# The default instruction per diff_row, so the text always matches the sheet. These
# are exactly the Judge 01 "v1", "diff" and "diff-sil" instructions.
DEFAULT_INSTRUCTIONS = {
    mode: _INSTRUCTION.replace("{row}", note) for mode, note in DIFF_ROW_NOTES.items()
}
DEFAULT_INSTRUCTION = DEFAULT_INSTRUCTIONS["silhouette"]

# Sheet backgrounds: the one farthest from every colour in the panels is used, so
# white steam is not lost on a white background.
SHEET_BACKGROUNDS = ("#ffffff", "#202020", "#4a6fa5")
DIFF_ROWS = ("none", "faded", "silhouette")
# Bump when ``sheet`` or ``changed_pixels`` draw a different picture for the same
# inputs: it is part of request_id (left out while it is 1, so the requests recorded
# for Judge 01 keep their ids).
SHEET_VERSION = 1
LABEL_H = 14


def judge_spec(j) -> dict[str, Any]:
    """Everything about the judge that changes its answer; part of request_id."""
    spec = {
        "backend": j.backend,
        "id": j.model_id,
        "revision": j.model_revision,
        "max_new_tokens": j.max_new_tokens,
        "scale": j.scale,
        "max_frames": j.max_frames,
        "instruction": j.instruction or DEFAULT_INSTRUCTIONS[j.diff_row],
    }
    # only when on, so requests made before the option keep their ids; "faded" was
    # the first diff row and was recorded as true
    if j.diff_row != "none":
        spec["diff_row"] = True if j.diff_row == "faded" else j.diff_row
    if SHEET_VERSION != 1:
        spec["sheet_version"] = SHEET_VERSION
    return spec


def _diff_row(spec: dict) -> str:
    d = spec.get("diff_row", "none")
    return "faded" if d is True else d


def pick_frames(n: int, max_frames: int) -> list[int]:
    """Evenly spread frame indices (all of them when n <= max_frames)."""
    if n <= max_frames:
        return list(range(n))
    return sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})


def changed_pixels(frame: np.ndarray, source: np.ndarray, tol: float = 30.0) -> np.ndarray:
    """Where a frame differs from the source: opacity flips or the colour moves > tol."""
    fa, sa = frame[..., 3] > 0, source[..., 3] > 0
    dist = np.linalg.norm(frame[..., :3].astype(np.float32) - source[..., :3], axis=-1)
    return (fa != sa) | (fa & sa & (dist > tol))


def _panel(rgba: np.ndarray, bg: np.ndarray, size: int) -> Image.Image:
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + bg * (1 - a)
    return Image.fromarray(rgb.round().astype(np.uint8)).resize((size, size), Image.NEAREST)


def sheet(
    source: np.ndarray,
    frames: list[np.ndarray],
    scale: int,
    gap: int = 8,
    diff_row: str = "none",
):
    """Source sprite + frames side by side, nearest-neighbour enlarged, labelled.

    ``diff_row`` adds a second row with every pixel that differs from the source
    marked in magenta, under each frame, drawn over
      faded       a faded copy of the source (Judge 01 showed the VLM reads its
                  colours as a colour change of the frame), or
      silhouette  the source's shape in flat grey.
    """
    panels = [source, *frames]
    key = choose_key_color(np.concatenate(panels, axis=0), SHEET_BACKGROUNDS)
    bg = np.array(hex_to_rgb(key), np.float32)
    size = source.shape[0] * scale
    if diff_row not in DIFF_ROWS:
        raise ValueError(f"unknown diff_row {diff_row!r}")
    rows = 1 if diff_row == "none" else 2
    w = len(panels) * size + (len(panels) + 1) * gap
    h = rows * (size + LABEL_H + gap) + gap
    img = Image.new("RGB", (w, h), tuple(int(c) for c in bg))
    draw = ImageDraw.Draw(img)
    ink = (0, 0, 0) if bg.mean() > 128 else (255, 255, 255)
    under = source.copy()
    if diff_row == "faded":
        under[..., 3] = (under[..., 3].astype(np.float32) * 0.3).astype(np.uint8)
    else:
        under[..., :3] = 150
    for i, p in enumerate(panels):
        x = gap + i * (size + gap)
        img.paste(_panel(p, bg, size), (x, gap + LABEL_H))
        draw.text((x, gap - 2), "S" if i == 0 else str(i), fill=ink)
        if diff_row != "none" and i > 0:
            d = under.copy()
            d[changed_pixels(p, source)] = (255, 0, 255, 255)
            y = 2 * gap + size + LABEL_H
            img.paste(_panel(d, bg, size), (x, y + LABEL_H))
            draw.text((x, y - 2), f"diff {i}", fill=ink)
    return img


def parse_verdict(text: str) -> dict[str, Any]:
    """The first JSON object in the answer, validated. Raises ValueError otherwise."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in the answer")
    d = json.loads(m.group(0))
    for k in ("motion_matches", "drawing_kept"):
        if not isinstance(d.get(k), bool):
            raise ValueError(f"{k!r} must be true or false")
    ok = d["motion_matches"] and d["drawing_kept"]
    return {
        "changes": str(d.get("changes") or "").strip(),
        "motion_matches": d["motion_matches"],
        "drawing_kept": d["drawing_kept"],
        "reason": str(d.get("reason") or "").strip(),
        "verdict": "accept" if ok else "reject",
    }


# ---------------------------------------------------------------- backends


class FakeJudge:
    """Accepts unless the subject name contains "static"; deterministic. For tests."""

    info = {"backend": "fake"}

    def __call__(self, image: Image.Image, instruction: str) -> str:
        m = re.search(r'animation of "([^"]*)"', instruction)
        ok = not (m and "static" in m.group(1))
        return json.dumps({"motion_matches": ok, "drawing_kept": True, "reason": "fake judge"})


def load_judge(spec: dict) -> Planner:
    if spec["backend"] == "fake":
        return FakeJudge()
    if spec["backend"] == "qwen":
        return QwenPlanner(spec)  # same image + text -> text model as the plan stage
    raise ValueError(f"unknown judge backend {spec['backend']!r}")


# ---------------------------------------------------------------- requests


def _digest(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _load(run_dir: Path, job_id: str) -> tuple[dict, np.ndarray, list[np.ndarray]]:
    sdir = run_dir / "final" / job_id
    sample = json.loads((sdir / "sample.json").read_text())
    frames = [load_rgba(sdir / f) for f in sample["frames"]]
    return sample, load_rgba(run_dir / sample["input_png"]), frames


def build_requests(run_dir: Path, cfg) -> list[dict]:
    """One request per finalized sample whose machine verdict is in ``cfg.verdicts``.

    A sample whose files cannot be read gets a request with ``error`` set (recorded
    as failed by ``judge``) so one broken sample does not stop the others. A missing
    finalize.jsonl or an action without criteria is a setup error and raises.
    """
    index = run_dir / "finalize.jsonl"
    if not index.exists():
        raise FileNotFoundError(f"{index} not found: run finalize first and upload it")
    spec = judge_spec(cfg)
    criteria = {**DEFAULT_CRITERIA, **cfg.criteria}
    out = []
    for r in read_jsonl(index):
        if r["status"] != "ok" or r["auto_verdict"] not in cfg.verdicts:
            continue
        action = r["requested_action"]
        if action not in criteria:
            raise ValueError(f"no judge criteria for action {action!r}: add [judge.criteria]")
        base = {"job_id": r["job_id"], "requested_action": action, "judge": spec}
        try:
            sample, source, frames = _load(run_dir, r["job_id"])
        except Exception as e:
            key = {"job_id": r["job_id"], "postprocess_id": r["postprocess_id"], "judge": spec}
            out.append(
                {
                    **base,
                    "request_id": stable_hash({**key, "unreadable": True}, 16),
                    "postprocess_id": r["postprocess_id"],
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue
        picked = pick_frames(len(frames), cfg.max_frames)
        instruction = spec["instruction"].format(
            subject=sample["source"]["subject"],
            action=action,
            criteria=criteria[action],
            n=len(picked),
        )
        pixels = _digest([source, *frames])
        rid = stable_hash(
            {
                "job_id": r["job_id"],
                "postprocess_id": sample["postprocess_id"],
                "pixels": pixels,
                "frames": picked,
                "judge": spec,
                "instruction": instruction,
            },
            16,
        )
        out.append(
            {
                **base,
                "request_id": rid,
                "postprocess_id": sample["postprocess_id"],
                "pixels": pixels,
                "subject": sample["source"]["subject"],
                "auto_verdict": r["auto_verdict"],
                "frames": picked,
                "instruction": instruction,
            }
        )
    return out


def latest_verdicts(run_dir: Path) -> dict[str, dict]:
    """Per request_id: the last ``ok`` record if there is one, else the last record."""
    out: dict[str, dict] = {}
    for r in read_jsonl(run_dir / "judge.jsonl"):
        prev = out.get(r["request_id"])
        if r["status"] == "ok" or not prev or prev["status"] != "ok":
            out[r["request_id"]] = r
    return out


def judge(
    run_dir: str | Path,
    cfg,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
    sync_dir: str | Path | None = None,
    judge_factory: Callable[[dict], Planner] = load_judge,
    shared: dict | None = None,
) -> dict[str, int]:
    """Judge pending samples of one run. ``shared`` caches the loaded model across runs."""
    run_dir = Path(run_dir)
    _restore(run_dir, sync_dir)
    last = latest_verdicts(run_dir)
    todo = []
    for q in build_requests(run_dir, cfg):
        r = last.get(q["request_id"])
        if r and (r["status"] == "ok" or not retry_failed):
            continue
        todo.append(q)
    if limit is not None:
        todo = todo[:limit]
    counts = {"ok": 0, "failed": 0, "pending": len(todo)}
    _sync(run_dir, sync_dir)
    if not todo:
        return counts

    spec = judge_spec(cfg)
    if all("error" in q for q in todo):
        model = None  # nothing to show the model; record the failures only
        runtime = {"python": platform.python_version(), "host": platform.node()}
    else:
        model, runtime = _model(spec, shared, judge_factory)

    for q in todo:
        rec: dict[str, Any] = {
            "request_id": q["request_id"],
            "job_id": q["job_id"],
            "postprocess_id": q["postprocess_id"],
            "started_at": _now(),
            "runtime": runtime,
        }
        t1 = time.monotonic()
        try:
            if "error" in q:
                raise ValueError(f"sample files unreadable: {q['error']}")
            sample, source, frames = _load(run_dir, q["job_id"])
            if sample["postprocess_id"] != q["postprocess_id"] or q["pixels"] != _digest(
                [source, *frames]
            ):
                # e.g. re-finalized while judging: the verdict must match the request id
                raise ValueError("sample changed since the request was built; run again")
            frames = [frames[i] for i in q["frames"]]
            img = sheet(source, frames, spec["scale"], diff_row=_diff_row(spec))
            raw = model(img, q["instruction"])
            rec["raw"] = raw
            rec.update(status="ok", **parse_verdict(raw))
            counts["ok"] += 1
        except Exception as e:  # recorded per sample; the loop continues
            rec.update(
                status="failed",
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=3),
            )
            counts["failed"] += 1
            log.error("judge %s failed: %s", q["job_id"], rec["error"])
        rec["seconds"] = round(time.monotonic() - t1, 2)
        append_jsonl(run_dir / "judge.jsonl", rec)
        _sync(run_dir, sync_dir)
        counts["pending"] -= 1
    return counts


def _model(spec: dict, shared: dict | None, factory: Callable[[dict], Planner]):
    key = json.dumps(spec, sort_keys=True)
    cache = shared if shared is not None else {}
    if key not in cache:
        t0 = time.monotonic()
        cache[key] = factory(spec)
        log.info("loaded judge %s in %.1fs", spec["id"] or "fake", time.monotonic() - t0)
    model = cache[key]
    return model, {"python": platform.python_version(), "host": platform.node(), **model.info}


def close(shared: dict) -> None:
    for m in shared.values():
        if hasattr(m, "close"):
            m.close()
    shared.clear()
    gc.collect()


def _restore(run_dir: Path, sync_dir: str | Path | None) -> None:
    """Take in records that only the synced copy has (e.g. a fresh VM resuming), so
    the first sync does not overwrite the backup with a shorter log."""
    if not sync_dir:
        return
    backup = Path(sync_dir) / "judge.jsonl"
    if not backup.exists():
        return
    log_path = run_dir / "judge.jsonl"
    have = {canonical_json(r) for r in read_jsonl(log_path)}
    extra = [r for r in read_jsonl(backup) if canonical_json(r) not in have]
    for r in extra:
        append_jsonl(log_path, r)
    if extra:
        log.info("restored %d judge records from %s", len(extra), backup)


def _sync(run_dir: Path, sync_dir: str | Path | None) -> None:
    log_path = run_dir / "judge.jsonl"
    if sync_dir and log_path.exists():
        copy_atomic(log_path, Path(sync_dir) / "judge.jsonl")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
