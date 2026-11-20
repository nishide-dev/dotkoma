"""Run pending generate jobs on a Colab GPU VM.

Upload the repo source (``src/``, ``scripts/``) and the prepared run directory
(``work/<run>/jobs.jsonl`` + ``work/<run>/inputs/``), then on the VM:

    python scripts/colab_pilot.py work/pilot [--limit N] [--sync-dir DIR] [--no-offload]

GPU dependencies are installed here, not in the local uv environment. The
versions that actually ran are written to ``<run>/colab_env.json``.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

# diffusers' Wan2.2 support may require a recent build; pin once a version is verified.
GPU_DEPS = [
    "diffusers>=0.35",
    "transformers",
    "accelerate",
    "ftfy",
    "av",
    "huggingface_hub",
    "pillow",
    "numpy",
]
RECORD = ["torch", "diffusers", "transformers", "accelerate", "av", "huggingface_hub", "numpy"]


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
    p.add_argument("run_dir", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--sync-dir", help="e.g. /content/drive/MyDrive/dotkoma/pilot")
    p.add_argument("--no-offload", action="store_true", help="keep the whole model on GPU")
    p.add_argument("--skip-install", action="store_true")
    a = p.parse_args()

    require_torch()
    if not a.skip_install:
        ensure_deps()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from dotkoma.generate import generate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(record_env(a.run_dir)), flush=True)
    counts = generate(
        a.run_dir,
        limit=a.limit,
        retry_failed=a.retry_failed,
        offload=not a.no_offload,
        sync_dir=a.sync_dir,
    )
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
