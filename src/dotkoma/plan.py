"""plan (GPU): a vision-language model writes the motion instruction per sprite.

Pilots 02/03 showed that one prompt template per action does not fit every
sprite: a lamp without a flame cannot flicker, a closed flask does not bubble,
and where the moving part sits differs per sprite. For actions with
``method = "vlm"``, a VLM looks at the sprite and answers, per (sprite, action):

  feasible     can this sprite show the motion at all
  moving_part  which part moves
  motion_text  one or two sentences for the video model

File contract (``work/<run>/``):
  plan_requests.jsonl  written by prepare: every (sprite, action) that needs a plan
  plans.jsonl          append-only, written here; the last ``ok`` record per
                       request_id counts. Re-running resumes.

prepare reads plans.jsonl and turns each feasible plan into a job whose prompt is
the action's ``prompt_frame`` with ``{motion_text}`` filled in. The full prompt is
part of job_id, so a new plan wording gives a new job.

Backends:
  fake  deterministic answers for tests (no model)
  qwen  Qwen3.5 via transformers (imports torch lazily; Colab only)
"""

from __future__ import annotations

import gc
import json
import logging
import platform
import re
import shutil
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .imaging import load_rgba
from .records import append_jsonl, read_jsonl, stable_hash

log = logging.getLogger(__name__)

# Shown to the VLM. {subject}, {caption}, {action}, {description} are filled per request.
DEFAULT_INSTRUCTION = """\
The image is a 16x16 pixel-art sprite of "{subject}" ({caption}), shown enlarged.
We want a short looping animation where the drawing stays exactly as it is and only \
this happens: {action} ({description}).

First decide whether the sprite clearly shows what this motion needs. Be strict: \
answer feasible = false when the needed part is not drawn (for example no flame for \
flickering, no open liquid surface for bubbling, no tail for wagging, no eyes for \
blinking). Do not imagine parts that are not visible.

Answer with one JSON object and nothing else:
{{"feasible": true or false,
 "moving_part": "the drawn part involved, named as an object (e.g. the flame, the tail)",
 "reason": "one short sentence about what you see",
 "motion_text": "one sentence for a video model"}}

Rules for motion_text: talk about objects, never about pixels or pixel colours. If \
the motion adds something new (steam, smoke, sparkles, bubbles), say it appears and \
fades above or on the named part while the drawing itself does not change. If a \
drawn part moves (a flame, a tail, eyelids, cloth), name it and say it returns to its \
original shape. End with: everything else stays perfectly still. Leave motion_text \
empty when feasible is false."""


class Planner(Protocol):
    info: dict[str, Any]

    def __call__(self, image: Image.Image, instruction: str) -> str:
        """Return the model's raw text answer."""


def planner_spec(p) -> dict[str, Any]:
    """Everything about the planner that changes its answer; part of request_id."""
    return {
        "backend": p.backend,
        "id": p.model_id,
        "revision": p.model_revision,
        "max_new_tokens": p.max_new_tokens,
        "image_scale": p.image_scale,
        "instruction": p.instruction or DEFAULT_INSTRUCTION,
    }


def request_id(source: dict, action: dict, spec: dict, instruction: str, sprite_sha256: str) -> str:
    """Everything the VLM sees or that changes its answer: the exact pixels and the
    rendered instruction (subject and caption included), plus the planner settings."""
    return stable_hash(
        {
            "source": source,
            "action": action,
            "planner": spec,
            "instruction": instruction,
            "sprite_sha256": sprite_sha256,
        },
        16,
    )


def sprite_image(rgba: np.ndarray, scale: int, bg: tuple[int, int, int] = (255, 255, 255)):
    """Nearest-neighbour enlargement on a flat background (VLMs see 16x16 poorly)."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + np.array(bg, np.float32) * (1 - a)
    img = Image.fromarray(rgb.round().astype(np.uint8))
    return img.resize((rgba.shape[1] * scale, rgba.shape[0] * scale), Image.NEAREST)


def parse_answer(text: str) -> dict[str, Any]:
    """The first JSON object in the answer, validated. Raises ValueError otherwise."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in the answer")
    d = json.loads(m.group(0))
    if not isinstance(d.get("feasible"), bool):
        raise ValueError("'feasible' must be true or false")
    out = {
        "feasible": d["feasible"],
        "moving_part": str(d.get("moving_part") or "").strip(),
        "reason": str(d.get("reason") or "").strip(),
        "motion_text": " ".join(str(d.get("motion_text") or "").split()),
    }
    if out["feasible"] and not out["motion_text"]:
        raise ValueError("feasible plan without motion_text")
    return out


# ---------------------------------------------------------------- backends


class FakePlanner:
    """Feasible unless the caption says "no {part}"; deterministic text. For tests."""

    info = {"backend": "fake"}

    def __call__(self, image: Image.Image, instruction: str) -> str:
        m = re.search(r'sprite of "([^"]*)" \(([^)]*)\).*?happens: (\w+)', instruction, re.S)
        subject, caption, action = m.groups() if m else ("it", "", "moves")
        feasible = "no " not in caption
        return json.dumps(
            {
                "feasible": feasible,
                "moving_part": f"the part of the {subject} that does {action}",
                "reason": "fake planner",
                "motion_text": f"The {subject} does {action} gently, everything else stays still."
                if feasible
                else "",
            }
        )


class QwenPlanner:
    """Qwen3.5 (image-text-to-text) via transformers, greedy decoding, thinking off."""

    def __init__(self, spec: dict) -> None:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        sha = HfApi().model_info(spec["id"], revision=spec["revision"]).sha
        self.processor = AutoProcessor.from_pretrained(spec["id"], revision=sha)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            spec["id"], revision=sha, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.max_new_tokens = spec["max_new_tokens"]
        self.info = {
            "backend": "qwen",
            "model_id": spec["id"],
            "resolved_revision": sha,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        }

    def __call__(self, image: Image.Image, instruction: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1] :]
        return self.processor.decode(new, skip_special_tokens=True)

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def load_planner(spec: dict) -> Planner:
    if spec["backend"] == "fake":
        return FakePlanner()
    if spec["backend"] == "qwen":
        return QwenPlanner(spec)
    raise ValueError(f"unknown plan backend {spec['backend']!r}")


# ---------------------------------------------------------------- runner


def latest_plans(run_dir: Path) -> dict[str, dict]:
    """Per request_id: the last ``ok`` record if there is one, else the last record.

    A failed retry after a success (e.g. logs merged from resumed runs) must not
    hide the answer that already exists.
    """
    out: dict[str, dict] = {}
    for r in read_jsonl(run_dir / "plans.jsonl"):
        prev = out.get(r["request_id"])
        if r["status"] == "ok" or not prev or prev["status"] != "ok":
            out[r["request_id"]] = r
    return out


def pending_requests(run_dir: Path, retry_failed: bool = False) -> list[dict]:
    path = run_dir / "plan_requests.jsonl"
    if not path.exists():
        # e.g. plan run before prepare, or the file was not uploaded to the VM
        raise FileNotFoundError(f"{path} not found: run prepare first and upload it")
    last = latest_plans(run_dir)
    out = []
    for q in read_jsonl(path):
        r = last.get(q["request_id"])
        if r and r["status"] == "ok":
            continue
        if r and r["status"] == "failed" and not retry_failed:
            continue
        out.append(q)
    return out


def plan(
    run_dir: str | Path,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
    sync_dir: str | Path | None = None,
    planner_factory: Callable[[dict], Planner] = load_planner,
) -> dict[str, int]:
    run_dir = Path(run_dir)
    todo = pending_requests(run_dir, retry_failed)
    if limit is not None:
        todo = todo[:limit]
    counts = {"ok": 0, "failed": 0, "pending": len(todo)}
    if not todo:
        log.info("nothing to plan")
        return counts

    groups: dict[str, list[dict]] = {}
    for q in todo:  # one model load per planner spec
        groups.setdefault(json.dumps(q["planner"], sort_keys=True), []).append(q)
    for reqs in groups.values():
        spec = reqs[0]["planner"]
        t0 = time.monotonic()
        planner = planner_factory(spec)
        load_s = round(time.monotonic() - t0, 2)
        runtime = {"python": platform.python_version(), "host": platform.node(), **planner.info}
        log.info("loaded planner %s in %.1fs; %d requests", spec["id"] or "fake", load_s, len(reqs))
        for q in reqs:
            rec: dict[str, Any] = {
                "request_id": q["request_id"],
                "started_at": _now(),
                "runtime": runtime,
            }
            t1 = time.monotonic()
            try:
                rgba = load_rgba(run_dir / q["input_png"])
                raw = planner(sprite_image(rgba, spec["image_scale"]), q["instruction"])
                rec["raw"] = raw
                rec.update(status="ok", **parse_answer(raw))
                counts["ok"] += 1
            except Exception as e:  # recorded per request; the loop continues
                rec.update(
                    status="failed",
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(limit=3),
                )
                counts["failed"] += 1
                log.error("plan %s failed: %s", q["request_id"], rec["error"])
            rec["seconds"] = round(time.monotonic() - t1, 2)
            append_jsonl(run_dir / "plans.jsonl", rec)
            if sync_dir:  # keep finished answers if the VM goes away
                Path(sync_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_dir / "plans.jsonl", Path(sync_dir) / "plans.jsonl")
            counts["pending"] -= 1
        if hasattr(planner, "close"):
            planner.close()
        del planner
        gc.collect()
    return counts


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
