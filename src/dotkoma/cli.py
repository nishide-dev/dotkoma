"""Command line entry: ``dotkoma prepare|plan|generate|finalize|judge``."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from .config import load_config


def _run_dir(args) -> tuple[Path, object]:
    cfg = load_config(args.config, work_dir=args.work_dir)
    return cfg.run_dir, cfg


def cmd_prepare(args) -> None:
    from .prepare import prepare

    _, cfg = _run_dir(args)
    print(prepare(cfg))


def cmd_generate(args) -> None:
    from .generate import generate

    run_dir, cfg = _run_dir(args)
    g = cfg.generation
    counts = generate(
        run_dir,
        limit=args.limit,
        retry_failed=args.retry_failed,
        offload="model" if g.offload == "resident" else g.offload,  # resident: from job
        pix_fmt=g.pix_fmt,
        crf=g.crf,
        sync_dir=args.sync_dir,
    )
    print(json.dumps(counts))


def cmd_plan(args) -> None:
    from .plan import plan

    run_dir, _ = _run_dir(args)
    counts = plan(run_dir, limit=args.limit, retry_failed=args.retry_failed, sync_dir=args.sync_dir)
    print(json.dumps(counts))


def cmd_finalize(args) -> None:
    from .finalize import finalize

    run_dir, cfg = _run_dir(args)
    index = finalize(run_dir, cfg.finalize)
    status = Counter(r["status"] for r in index)
    verdict = Counter(r.get("auto_verdict") for r in index if r["status"] == "ok")
    flags = Counter(f for r in index for f in r.get("flags", []))
    print(json.dumps({"status": status, "auto_verdict": verdict, "flags": flags}, indent=1))


def cmd_judge(args) -> None:
    from .config import load_judge_config
    from .judge import close, judge

    cfg = load_judge_config(args.judge_config)
    names = [Path(r).name for r in args.runs]
    if args.sync_dir and len(set(names)) != len(names):
        raise SystemExit("run dirs must have distinct names when --sync-dir is used")
    shared: dict = {}  # one model load for all runs
    try:
        for run in args.runs:
            sync = Path(args.sync_dir) / Path(run).name if args.sync_dir else None
            counts = judge(
                run,
                cfg,
                limit=args.limit,
                retry_failed=args.retry_failed,
                sync_dir=sync,
                shared=shared,
            )
            print(run, json.dumps(counts))
    finally:
        close(shared)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dotkoma")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("config", help="run config TOML, e.g. configs/pilot.toml")
        sp.add_argument("--work-dir", help="override run.work_dir")
        sp.set_defaults(fn=fn)
        return sp

    add("prepare", cmd_prepare, "pick sprites, write reference PNGs and jobs.jsonl")
    pl = add("plan", cmd_plan, "VLM writes per-sprite motion instructions (Colab GPU, or fake)")
    pl.add_argument("--limit", type=int)
    pl.add_argument("--retry-failed", action="store_true")
    pl.add_argument("--sync-dir", help="copy plans.jsonl here after each answer (e.g. Drive)")
    g = add("generate", cmd_generate, "run pending jobs (Colab GPU, or the fake backend)")
    g.add_argument("--limit", type=int)
    g.add_argument("--retry-failed", action="store_true")
    g.add_argument("--sync-dir", help="copy each finished mp4 + results.jsonl here (e.g. Drive)")
    add("finalize", cmd_finalize, "raw mp4 -> 16x16 RGBA loops, checks, preview GIFs")
    j = sub.add_parser("judge", help="VLM checks finalized loops against their action (Colab GPU)")
    j.add_argument("judge_config", help="judge TOML, e.g. configs/judge.toml")
    j.add_argument("runs", nargs="+", help="run dirs with finalize output, e.g. work/pilot-03")
    j.add_argument("--limit", type=int, help="per run")
    j.add_argument("--retry-failed", action="store_true")
    j.add_argument(
        "--sync-dir", help="copy judge.jsonl to <sync-dir>/<run name>/ after each answer"
    )
    j.set_defaults(fn=cmd_judge)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.fn(args)
