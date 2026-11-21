"""Run pending generate jobs on a Colab GPU VM.

Upload the repo source (``src/``, ``scripts/``) and the prepared run directory
(``work/<run>/jobs.jsonl`` + ``work/<run>/inputs/``), then on the VM:

    python scripts/colab_pilot.py work/pilot [--limit N] [--sync-dir DIR] [--no-offload]
    python scripts/colab_pilot.py work/pilot --stage plan   # VLM plans (plan_requests.jsonl)

    python scripts/colab_pilot.py work/pilot-03 work/pilot-04 --stage judge \
        --judge-config configs/judge.toml   # VLM judge (finalize output)

For ``--stage plan`` upload ``plan_requests.jsonl`` + ``inputs/`` instead of jobs.jsonl.
For ``--stage judge`` upload ``finalize.jsonl`` + ``final/`` + ``inputs/`` and the
judge config.

GPU dependencies are installed here, not in the local uv environment. The
versions that actually ran are written to ``<run>/colab_env.json``.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

# Must be set before torch initialises CUDA. Reduces fragmentation: the first L4 OOM
# reported 4.7 GB reserved-but-unallocated memory.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# diffusers' Wan2.2 support may require a recent build; pin once a version is verified.
GPU_DEPS = [
    "diffusers>=0.35",
    "transformers",
    "accelerate",
    "ftfy",
    "peft",  # LoRA loading (step-distill LoRAs)
    "av",
    "huggingface_hub",
    "pillow",
    "numpy",
]
RECORD = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "torchao",
    "av",
    "huggingface_hub",
    "numpy",
]
# PEFT refuses to import with an old torchao installed (Colab shipped 0.10.0; PEFT needs
# > 0.16). dotkoma does not use torchao, so an outdated copy is removed rather than upgraded
# (an upgrade could pull a torch build that does not match the runtime).
TORCHAO_MIN = "0.16"


def require_torch() -> None:
    """PyTorch must come from the environment (Colab ships a CUDA build); never pip it here."""
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch is not installed. Use a GPU runtime that provides a CUDA build of torch.")
    if not torch.cuda.is_available():
        sys.exit(f"torch {torch.__version__} is installed but no CUDA device is available.")


def ensure_deps() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *GPU_DEPS])
    try:
        ver = md.version("torchao")
    except md.PackageNotFoundError:
        return
    from packaging.version import Version

    if Version(ver) <= Version(TORCHAO_MIN):
        print(f"removing torchao {ver} (incompatible with peft)", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])


def record_env(run_dir: Path) -> dict:
    env = {"python": platform.python_version()}
    for name in RECORD:
        try:
            env[name] = md.version(name)
        except md.PackageNotFoundError:
            env[name] = None
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        env["nvidia_smi"] = smi.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["nvidia_smi"] = None
    from dotkoma.records import write_json_atomic

    write_json_atomic(run_dir / "colab_env.json", env)
    return env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, nargs="+", help="several runs share one model load")
    p.add_argument("--limit", type=int)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--sync-dir", help="e.g. /content/drive/MyDrive/dotkoma/pilot")
    p.add_argument("--no-offload", action="store_true", help="keep the whole model on GPU")
    p.add_argument(
        "--offload",
        choices=["model", "none"],
        help="model (CPU offload, default) or none. Resident placement is set in the "
        "config (generation.offload) because it is part of job_id",
    )
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--stage", choices=["generate", "plan", "judge"], default="generate")
    p.add_argument("--judge-config", type=Path, help="judge TOML (for --stage judge)")
    a = p.parse_args()

    require_torch()
    if not a.skip_install:
        ensure_deps()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    names = [r.name for r in a.run_dir]
    if a.sync_dir and len(set(names)) != len(names):
        # each run syncs to <sync-dir>/<name>; equal names would overwrite each other
        sys.exit(f"run dirs need distinct names when syncing: {names}")
    for run_dir in a.run_dir:
        print(json.dumps(record_env(run_dir)), flush=True)
    if a.stage == "judge":
        from dotkoma.config import load_judge_config
        from dotkoma.judge import close, judge

        if not a.judge_config:
            sys.exit("--stage judge needs --judge-config")
        cfg = load_judge_config(a.judge_config)
        many = len(a.run_dir) > 1
        loaded: dict = {}  # one model load for all runs
        try:
            for run_dir in a.run_dir:
                sync = (
                    (Path(a.sync_dir) / run_dir.name if many else a.sync_dir)
                    if a.sync_dir
                    else None
                )
                counts = judge(
                    run_dir,
                    cfg,
                    limit=a.limit,
                    retry_failed=a.retry_failed,
                    sync_dir=sync,
                    shared=loaded,
                )
                print(run_dir, json.dumps(counts), flush=True)
        finally:
            close(loaded)
        return
    if a.stage == "plan":
        from dotkoma.plan import plan

        many = len(a.run_dir) > 1
        for run_dir in a.run_dir:
            sync = (Path(a.sync_dir) / run_dir.name if many else a.sync_dir) if a.sync_dir else None
            counts = plan(run_dir, limit=a.limit, retry_failed=a.retry_failed, sync_dir=sync)
            print(run_dir, json.dumps(counts), flush=True)
        return
    from dotkoma.generate import generate, load_backend, pending_jobs

    offload = a.offload or ("none" if a.no_offload else "model")
    # every pending job per model, across all runs: a resident backend encodes all
    # their prompts up front, before the experts move onto the GPU
    by_model: dict[str, list[dict]] = {}
    for run_dir in a.run_dir:
        todo = pending_jobs(run_dir, a.retry_failed)
        for j in todo[: a.limit] if a.limit is not None else todo:
            by_model.setdefault(json.dumps(j["model"], sort_keys=True), []).append(j)
    loaded: dict[str, object] = {}

    def shared(model: dict, offload: bool | str):
        """Reuse one loaded model across runs with the same model spec."""
        key = json.dumps(model, sort_keys=True)
        if key not in loaded:
            loaded.clear()  # a different model: let the previous one go first
            backend = load_backend(model, offload)
            if hasattr(backend, "prepare"):
                backend.prepare(by_model.get(key, []))
            loaded[key] = backend
        return loaded[key]

    many = len(a.run_dir) > 1
    for run_dir in a.run_dir:
        # one sync folder per run, so runs do not overwrite each other's records
        sync = (Path(a.sync_dir) / run_dir.name if many else a.sync_dir) if a.sync_dir else None
        counts = generate(
            run_dir,
            limit=a.limit,
            retry_failed=a.retry_failed,
            offload=offload,
            sync_dir=sync,
            backend_factory=shared,
        )
        print(run_dir, json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
