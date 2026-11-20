"""Stage 0 acceptance: prepare -> generate(fake) -> finalize without a GPU."""

import json

import numpy as np

from dotkoma.finalize import finalize
from dotkoma.generate import FakeBackend, generate, load_backend, raw_path
from dotkoma.imaging import load_rgba
from dotkoma.prepare import prepare
from dotkoma.records import append_jsonl, read_jsonl


def _by_action(index, subject, action):
    return next(r for r in index if r["subject"] == subject and r["requested_action"] == action)


def test_prepare_is_deterministic(smoke_cfg):
    p = prepare(smoke_cfg)
    first = p.read_bytes()
    prepare(smoke_cfg)
    assert p.read_bytes() == first
    jobs = read_jsonl(p)
    assert len(jobs) == 6 and len({j["job_id"] for j in jobs}) == 6
    for j in jobs:
        assert (smoke_cfg.run_dir / j["reference_png"]).exists()
        assert j["split"] in ("train", "val", "test")


def test_end_to_end_recovers_known_motion(smoke_cfg):
    run = smoke_cfg.run_dir
    prepare(smoke_cfg)
    assert generate(run) == {"ok": 6, "failed": 0, "pending": 0}
    index = finalize(run, smoke_cfg.finalize)
    assert all(r["status"] == "ok" for r in index)

    hop = _by_action(index, "slime", "hop")
    assert hop["auto_verdict"] == "candidate", hop
    s = json.loads((run / "final" / hop["job_id"] / "sample.json").read_text())
    src = load_rgba(run / s["input_png"])
    frames = [load_rgba(run / "final" / hop["job_id"] / f) for f in s["frames"]]
    np.testing.assert_array_equal(frames[0], src)
    # hop moves the sprite up (two pixels at the apex) and back
    np.testing.assert_array_equal(frames[3][:-2], src[2:])
    assert len(s["durations_ms"]) == len(frames) == 6
    assert (run / "final" / hop["job_id"] / "preview.gif").exists()

    static = _by_action(index, "candle", "static")
    assert static["auto_verdict"] == "reject" and "all_identical" in static["flags"]


def test_resume_only_runs_missing_jobs(smoke_cfg):
    run = smoke_cfg.run_dir
    prepare(smoke_cfg)
    generate(run, limit=2)
    assert len(read_jsonl(run / "results.jsonl")) == 2
    calls = []

    def counting(model, offload):
        b = FakeBackend()

        def call(job, ref):
            calls.append(job["job_id"])
            return b(job, ref)

        call.info = b.info
        return call

    generate(run, backend_factory=counting)
    assert len(calls) == 4
    assert generate(run, backend_factory=counting)["pending"] == 0

    # an ok record whose mp4 vanished is not "done"
    jobs = read_jsonl(run / "jobs.jsonl")
    raw_path(run, jobs[0]["job_id"]).unlink()
    calls.clear()
    generate(run, backend_factory=counting)
    assert calls == [jobs[0]["job_id"]]


def test_failures_are_recorded_and_skipped(smoke_cfg):
    run = smoke_cfg.run_dir
    prepare(smoke_cfg)

    def flaky(model, offload):
        b = FakeBackend()

        def call(job, ref):
            if job["requested_action"] == "sway":
                raise RuntimeError("CUDA out of memory")
            return b(job, ref)

        call.info = b.info
        return call

    assert generate(run, backend_factory=flaky) == {"ok": 4, "failed": 2, "pending": 0}
    failed = [r for r in read_jsonl(run / "results.jsonl") if r["status"] == "failed"]
    assert all("out of memory" in r["error"] for r in failed)
    assert not any(raw_path(run, r["job_id"]).exists() for r in failed)
    assert generate(run, backend_factory=flaky)["pending"] == 0  # not retried by default
    assert (
        generate(run, retry_failed=True, backend_factory=lambda m, o: load_backend(m, o))["ok"] == 2
    )

    index = finalize(run, smoke_cfg.finalize)
    assert {r["status"] for r in index} == {"ok"}


def test_finalize_reuses_raw_with_selection_override(smoke_cfg):
    run = smoke_cfg.run_dir
    prepare(smoke_cfg)
    generate(run)
    finalize(run, smoke_cfg.finalize)
    jid = _by_action(read_jsonl(run / "finalize.jsonl"), "ghost", "bob")["job_id"]
    before = json.loads((run / "final" / jid / "sample.json").read_text())
    mtime = raw_path(run, jid).stat().st_mtime_ns

    append_jsonl(run / "selections.jsonl", {"job_id": jid, "indices": [0, 6, 12, 18]})
    finalize(run, smoke_cfg.finalize)
    after = json.loads((run / "final" / jid / "sample.json").read_text())
    assert after["source_frame_indices"] == [0, 6, 12, 18]
    assert after["postprocess_id"] != before["postprocess_id"]
    assert len(list((run / "final" / jid / "frames").iterdir())) == 4
    assert raw_path(run, jid).stat().st_mtime_ns == mtime  # raw video untouched


def test_motion_leaving_the_grid_is_flagged(smoke_cfg):
    from dotkoma.generate import _shift
    from dotkoma.imaging import Layout, hex_to_rgb

    run = smoke_cfg.run_dir
    prepare(smoke_cfg)

    def big_jump(model, offload):
        def call(job, ref):
            lay = Layout.from_dict(job["layout"])
            key = np.array(hex_to_rgb(job["key_color"]), np.uint8)
            n = job["inference"]["num_frames"]
            return np.stack(
                [_shift(ref, -int(6 * lay.scale * np.sin(np.pi * t / n)), 0, key) for t in range(n)]
            )

        call.info = {"backend": "test"}
        return call

    generate(run, backend_factory=big_jump)
    index = finalize(run, smoke_cfg.finalize)
    assert "edge_cut" in _by_action(index, "ghost", "bob")["flags"]


def test_prompt_variants_become_separate_jobs_with_shared_seed(smoke_cfg):
    import dataclasses

    from dotkoma.config import ActionConfig

    hop = smoke_cfg.actions["hop"]
    actions = {
        **smoke_cfg.actions,
        "hop": ActionConfig("hop", (*hop.prompts, "{subject} jumps up once and lands"), hop.motion),
    }
    sprites = tuple(
        dataclasses.replace(p, prompt_variants="all") if p.subject == "slime" else p
        for p in smoke_cfg.sprites
    )
    cfg = dataclasses.replace(smoke_cfg, actions=actions, sprites=sprites)
    jobs = read_jsonl(prepare(cfg))
    hops = [j for j in jobs if j["requested_action"] == "hop"]
    assert [j["prompt_variant"] for j in hops] == [0, 1]
    assert len({j["job_id"] for j in hops}) == 2
    assert len({j["seed"] for j in hops}) == 1
    assert all(j["prompt_method"] == "template" for j in jobs)
    assert len(jobs) == 7


def test_background_drift_after_first_frame_is_keyed_out(smoke_cfg):
    """Wan reproduces frame 0 exactly, then shifts the key colour slightly."""
    import dataclasses

    from dotkoma.generate import FakeBackend
    from dotkoma.imaging import hex_to_rgb

    run = smoke_cfg.run_dir
    prepare(smoke_cfg)
    fake = FakeBackend()

    def drifting(model, offload):
        def call(job, ref):
            frames = fake(job, ref).astype(np.int16)
            key = np.array(hex_to_rgb(job["key_color"]), np.int16)
            bgmask = (frames == key).all(axis=-1)
            shifted = np.clip(key + np.array([90, -90, 90]), 0, 255)  # > bg_tolerance away
            frames[1:][bgmask[1:]] = shifted
            return frames.astype(np.uint8)

        call.info = fake.info
        return call

    generate(run, backend_factory=drifting)

    def drift(index):
        jid = _by_action(index, "ghost", "bob")["job_id"]
        return json.loads((run / "final" / jid / "sample.json").read_text())["checks"][
            "background_drift"
        ]

    per_frame = finalize(run, smoke_cfg.finalize)
    assert "background_unstable" not in _by_action(per_frame, "ghost", "bob")["flags"]
    assert drift(per_frame) > 60

    first_only = dataclasses.replace(smoke_cfg.finalize, background="first_frame")
    stale = finalize(run, first_only)
    assert "background_unstable" in _by_action(stale, "ghost", "bob")["flags"]
    assert drift(stale) > 60  # drift is measured even when keying uses frame 0


def test_memory_is_released_after_leaving_the_exception_handler(smoke_cfg):
    import sys

    run = smoke_cfg.run_dir
    prepare(smoke_cfg)
    fake = FakeBackend()
    released = []

    class Flaky:
        info = fake.info

        def __call__(self, job, ref):
            if job["requested_action"] == "hop":
                raise RuntimeError("CUDA out of memory")
            return fake(job, ref)

        def release_memory(self):
            released.append(sys.exc_info()[0])  # None once the handler has exited

    counts = generate(run, backend_factory=lambda m, o: Flaky())
    assert counts == {"ok": 5, "failed": 1, "pending": 0}
    assert released == [None]
