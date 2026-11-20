"""generate (Colab GPU): run image->video for every pending job in jobs.jsonl.

The model is loaded once per (backend, model id, revision) and jobs run in
order. Each job's outcome is appended to ``results.jsonl``; a job counts as
done only when its last record is ``ok`` *and* the mp4 exists. Re-running
the command resumes where it stopped.

Backends:
  fake       CPU-only synthetic motion from the reference image (stage 0 tests)
  diffusers  Wan2.2 via diffusers (imports torch lazily; Colab only)
"""

from __future__ import annotations

import gc
import logging
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .imaging import Layout, hex_to_rgb, load_rgba
from .records import append_jsonl, canonical_json, latest_by_job, read_jsonl
from .video import write_mp4

log = logging.getLogger(__name__)


class Backend(Protocol):
    info: dict[str, Any]

    def __call__(self, job: dict, reference_rgb: np.ndarray) -> np.ndarray:
        """Return frames (F, H, W, 3) uint8."""


# ---------------------------------------------------------------- fake backend

# vertical / horizontal offset in sprite pixels as a function of phase in [0, 1)
FAKE_MOTIONS: dict[str, Callable[[float], tuple[float, float]]] = {
    "bob": lambda p: (np.sin(2 * np.pi * p), 0.0),
    "hop": lambda p: (-2.0 * abs(np.sin(np.pi * p)), 0.0),
    "sway": lambda p: (0.0, np.sin(2 * np.pi * p)),
    "static": lambda p: (0.0, 0.0),
}


class FakeBackend:
    """Shifts the sprite over a flat background with a known periodic motion.

    The clip covers exactly one period, so frame 0 and frame N would coincide:
    a correct finalize step must produce a seamless loop.
    """

    def __init__(self) -> None:
        self.info = {"backend": "fake", "model_id": "fake", "model_revision": "n/a"}

    def __call__(self, job: dict, reference_rgb: np.ndarray) -> np.ndarray:
        inf = job["inference"]
        layout = Layout.from_dict(job["layout"])
        motion = FAKE_MOTIONS.get(job["requested_action"], FAKE_MOTIONS["bob"])
        key = np.array(hex_to_rgb(job["key_color"]), np.uint8)
        n = inf["num_frames"]
        frames = np.empty((n, *reference_rgb.shape), np.uint8)
        for t in range(n):
            dy, dx = motion(t / n)
            sy, sx = int(round(dy * layout.scale)), int(round(dx * layout.scale))
            frames[t] = _shift(reference_rgb, sy, sx, key)
        return frames


def _shift(img: np.ndarray, dy: int, dx: int, fill: np.ndarray) -> np.ndarray:
    out = np.empty_like(img)
    out[:] = fill
    H, W = img.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = img[ys, xs]
    return out


# ---------------------------------------------------------------- diffusers backend


class DiffusersWanBackend:
    """Wan2.2 image->video through diffusers. Not exercised by the CPU test suite."""

    def __init__(self, model: dict, offload: bool = True) -> None:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from huggingface_hub import HfApi

        model_id = model["id"]
        resolved = HfApi().model_info(model_id, revision=model["revision"]).sha
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32, revision=resolved
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.bfloat16, revision=resolved
        )
        timings = apply_model_options(pipe, model, torch)
        loras = model.get("loras", [])
        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        # Decoding 49 frames at 1280x704 in one pass needs more than an L4's 22 GB
        # (observed OOM in vae.decode); tiled decoding bounds peak memory. The mode is
        # part of job_id via model["vae_decode"] (absent = tiled).
        vae_tiling = model.get("vae_decode", "tiled") == "tiled"
        if vae_tiling:
            pipe.vae.enable_tiling()
        self.pipe, self.torch = pipe, torch
        self.info = {
            "backend": "diffusers",
            "model_id": model_id,
            "model_revision": resolved,
            "offload": offload,
            "vae_tiling": vae_tiling,
            "loras": loras,
            "quantization": model.get("quantization", "none"),
            **timings,
            **_torch_runtime(torch),
        }

    def release_memory(self) -> None:
        """Drop cached CUDA blocks after a failed job so the next job starts clean."""
        import gc

        gc.collect()
        self.torch.cuda.empty_cache()

    def __call__(self, job: dict, reference_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        inf = job["inference"]
        gen = self.torch.Generator(device="cpu").manual_seed(int(job["seed"]))
        out = self.pipe(
            image=Image.fromarray(reference_rgb),
            prompt=job["prompt"],
            negative_prompt=job["negative_prompt"] or None,
            height=inf["height"],
            width=inf["width"],
            num_frames=inf["num_frames"],
            guidance_scale=inf["guidance"],
            **({"guidance_scale_2": inf["guidance_2"]} if "guidance_2" in inf else {}),
            num_inference_steps=inf["steps"],
            generator=gen,
            output_type="np",
        ).frames[0]
        return (np.clip(out, 0.0, 1.0) * 255).round().astype(np.uint8)


def apply_model_options(pipe, model: dict, torch) -> dict[str, float]:
    """Merge LoRAs, then (optionally) store transformer weights in fp8.

    The order matters: fp8 storage must see the merged weights, otherwise the
    LoRA deltas would be applied on top of already-quantised weights.
    """
    t = time.monotonic()
    loras = model.get("loras", [])
    if loras:
        _merge_loras(pipe, loras)
    timings = {"lora_merge_seconds": round(time.monotonic() - t, 2)}
    if model.get("quantization") == "fp8_storage":
        for name in ("transformer", "transformer_2"):
            module = getattr(pipe, name, None)
            if module is not None:
                module.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
                )
    return timings


def _merge_loras(pipe, loras: list[dict]) -> None:
    """Load each LoRA into its expert, fuse it into the weights, then drop the adapters.

    Fusing removes the per-step LoRA overhead and lets fp8 storage cast plain weights.
    """
    by_target: dict[str, list[tuple[str, float]]] = {}
    for i, lora in enumerate(loras):
        name = f"lora{i}"
        extra = {"load_into_transformer_2": True} if lora["target"] == "transformer_2" else {}
        pipe.load_lora_weights(
            lora["repo"],
            weight_name=lora["file"],
            revision=lora["revision"],
            adapter_name=name,
            **extra,
        )
        by_target.setdefault(lora["target"], []).append((name, float(lora["weight"])))
    for target, adapters in by_target.items():
        names = [n for n, _ in adapters]
        pipe.set_adapters(names, adapter_weights=[w for _, w in adapters])
        pipe.fuse_lora(components=[target], adapter_names=names, lora_scale=1.0)
    pipe.unload_lora_weights()


def _torch_runtime(torch) -> dict[str, Any]:
    import diffusers

    info: dict[str, Any] = {"torch": torch.__version__, "diffusers": diffusers.__version__}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1), cuda=torch.version.cuda)
    return info


def load_backend(model: dict, offload: bool) -> Backend:
    if model["backend"] == "fake":
        return FakeBackend()
    if model["backend"] == "diffusers":
        return DiffusersWanBackend(model, offload=offload)
    raise ValueError(f"unknown backend {model['backend']!r}")


# ---------------------------------------------------------------- runner


def raw_path(run_dir: Path, jid: str) -> Path:
    return run_dir / "raw" / f"{jid}.mp4"


def pending_jobs(run_dir: Path, retry_failed: bool = False) -> list[dict]:
    jobs = read_jsonl(run_dir / "jobs.jsonl")
    last = latest_by_job(read_jsonl(run_dir / "results.jsonl"))
    out = []
    for j in jobs:
        if j.get("generation_method", "video_model") != "video_model":
            continue  # made locally by finalize (pixel_ops), never on the GPU
        r = last.get(j["job_id"])
        if r and r["status"] == "ok" and raw_path(run_dir, j["job_id"]).exists():
            continue
        if r and r["status"] == "failed" and not retry_failed:
            continue
        out.append(j)
    return out


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def generate(
    run_dir: str | Path,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
    offload: bool = True,
    pix_fmt: str = "yuv444p",
    crf: int = 10,
    sync_dir: str | Path | None = None,
    backend_factory: Callable[[dict, bool], Backend] = load_backend,
) -> dict[str, int]:
    run_dir = Path(run_dir)
    todo = pending_jobs(run_dir, retry_failed)
    if limit is not None:
        todo = todo[:limit]
    counts = {"ok": 0, "failed": 0, "pending": len(todo)}
    if not todo:
        log.info("nothing to do")
        return counts

    results = run_dir / "results.jsonl"
    groups: dict[str, list[dict]] = {}
    for j in todo:  # one model load per distinct model spec (everything that is in job_id)
        groups.setdefault(canonical_json(j["model"]), []).append(j)

    for jobs in groups.values():
        model_id = jobs[0]["model"]["id"]
        t0 = time.monotonic()
        backend = backend_factory(jobs[0]["model"], offload)
        load_s = round(time.monotonic() - t0, 2)
        log.info("loaded %s in %.1fs; %d jobs", model_id or "fake", load_s, len(jobs))
        runtime = {"python": platform.python_version(), "host": platform.node(), **backend.info}

        for j in jobs:
            jid = j["job_id"]
            rec: dict[str, Any] = {
                "job_id": jid,
                "started_at": _now(),
                "model_load_seconds": load_s,
                "runtime": runtime,
                "inference": j["inference"],
                "encode": {"pix_fmt": pix_fmt, "crf": crf},
            }
            t1 = time.monotonic()
            failed = False
            try:
                ref = load_rgba(run_dir / j["reference_png"])[..., :3]
                frames = backend(j, ref)
                out = raw_path(run_dir, jid)
                write_mp4(out, frames, fps=j["inference"]["fps"], pix_fmt=pix_fmt, crf=crf)
                if sync_dir:
                    dst = Path(sync_dir) / "raw" / out.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(out, dst)
                rec.update(
                    status="ok",
                    raw_path=out.relative_to(run_dir).as_posix(),
                    num_frames=int(len(frames)),
                    frame_size=list(frames.shape[1:3]),
                )
                counts["ok"] += 1
            except Exception as e:  # recorded per job; the loop continues
                rec.update(
                    status="failed",
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(limit=5),
                )
                counts["failed"] += 1
                log.error("job %s failed: %s", jid, rec["error"])
                failed = True
            # Outside the handler: the exception and its traceback (which pin the failed
            # inference's GPU tensors) are gone by now, so the memory can actually be freed.
            if failed and hasattr(backend, "release_memory"):
                backend.release_memory()
            rec["generate_seconds"] = round(time.monotonic() - t1, 2)
            rec["finished_at"] = _now()
            append_jsonl(results, rec)
            if sync_dir:
                shutil.copy2(results, Path(sync_dir) / "results.jsonl")
            counts["pending"] -= 1
        # Drop this model before the next group loads, so two pipelines never coexist.
        del backend
        _free_memory()
    return counts


def _free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")  # only if a GPU backend already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
