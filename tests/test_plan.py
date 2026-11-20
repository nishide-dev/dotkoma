"""plan: a VLM writes per-sprite motion instructions for method = "vlm" actions."""

import json

import pytest

from dotkoma.config import ConfigError, load_config
from dotkoma.plan import FakePlanner, parse_answer, plan
from dotkoma.prepare import prepare
from dotkoma.records import append_jsonl, read_jsonl

PLAN = """
[plan]
backend = "fake"
model_id = ""

[actions.glow]
method = "vlm"
motion = "local"
description = "a part of it glows softly"
prompt_frame = "A pixel art {subject} on a flat background. {motion_text} Static camera."
"""


def _vlm_cfg(smoke_cfg, subjects=("slime", "ghost")):
    path = smoke_cfg.config_path
    sprites = "".join(f'[[sprites]]\nsubject = "{s}"\nactions = ["glow"]\n' for s in subjects)
    text = path.read_text().replace("[actions.hop]", sprites + "\n[actions.hop]", 1)
    path.write_text(text + PLAN)
    return load_config(path, work_dir=smoke_cfg.work_dir)


class _NoGhost(FakePlanner):
    def __call__(self, image, instruction):
        if '"ghost"' in instruction:
            return '{"feasible": false, "moving_part": "", "reason": "nothing to glow"}'
        return super().__call__(image, instruction)


def _vlm_jobs(cfg):
    return [j for j in read_jsonl(prepare(cfg)) if j["prompt_method"] == "vlm"]


def test_parse_answer():
    ok = parse_answer('<think>hm</think>Sure: {"feasible": true, "motion_text": "Flame  moves."}')
    assert ok == {"feasible": True, "moving_part": "", "reason": "", "motion_text": "Flame moves."}
    assert parse_answer('{"feasible": false}')["feasible"] is False
    for bad in ["no json here", '{"feasible": "yes"}', '{"feasible": true, "motion_text": ""}']:
        with pytest.raises(ValueError):
            parse_answer(bad)


def test_prepare_plan_prepare(smoke_cfg):
    cfg = _vlm_cfg(smoke_cfg)
    run = cfg.run_dir
    assert _vlm_jobs(cfg) == []  # nothing planned yet
    reqs = read_jsonl(run / "plan_requests.jsonl")
    assert sorted(q["source"]["subject"] for q in reqs) == ["ghost", "slime"]
    assert all('"slime"' in q["instruction"] or '"ghost"' in q["instruction"] for q in reqs)

    counts = plan(run, planner_factory=lambda spec: _NoGhost())
    assert counts == {"ok": 2, "failed": 0, "pending": 0}
    assert plan(run, planner_factory=lambda spec: _NoGhost())["pending"] == 0  # resumable

    jobs = _vlm_jobs(cfg)  # ghost was judged infeasible: no job
    assert [j["source"]["subject"] for j in jobs] == ["slime"]
    j = jobs[0]
    assert j["prompt"].startswith("A pixel art slime on a flat background. The slime does glow")
    assert j["plan"]["request_id"] in {q["request_id"] for q in reqs}
    assert j["requested_action"] == "glow" and j["prompt_variant"] == 0
    # the template actions are untouched
    assert len([x for x in read_jsonl(run / "jobs.jsonl") if x["prompt_method"] == "template"]) == 6


def test_new_plan_wording_is_a_new_job(smoke_cfg):
    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())
    first = _vlm_jobs(cfg)[0]
    rec = dict(read_jsonl(cfg.run_dir / "plans.jsonl")[-1])
    rec["motion_text"] = "Only the top of the slime shimmers."
    append_jsonl(cfg.run_dir / "plans.jsonl", rec)  # e.g. re-planned with another model
    second = _vlm_jobs(cfg)[0]
    assert second["job_id"] != first["job_id"]
    assert "Only the top of the slime shimmers." in second["prompt"]


def test_failed_answers_are_recorded_and_retried(smoke_cfg):
    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)

    class Broken:
        info = {"backend": "broken"}

        def __call__(self, image, instruction):
            return "I cannot answer in JSON."

    assert plan(cfg.run_dir, planner_factory=lambda s: Broken())["failed"] == 1
    rec = read_jsonl(cfg.run_dir / "plans.jsonl")[-1]
    assert rec["status"] == "failed" and rec["raw"] == "I cannot answer in JSON."
    assert plan(cfg.run_dir, planner_factory=lambda s: FakePlanner())["pending"] == 0
    assert plan(cfg.run_dir, retry_failed=True, planner_factory=lambda s: FakePlanner())["ok"] == 1
    assert len(_vlm_jobs(cfg)) == 1


def test_request_id_tracks_the_planner(smoke_cfg):
    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    first = {q["request_id"] for q in read_jsonl(cfg.run_dir / "plan_requests.jsonl")}
    path = cfg.config_path
    path.write_text(path.read_text().replace('model_id = ""', 'model_id = ""\nimage_scale = 8'))
    prepare(load_config(path, work_dir=cfg.work_dir))
    second = {q["request_id"] for q in read_jsonl(cfg.run_dir / "plan_requests.jsonl")}
    assert first.isdisjoint(second)


@pytest.mark.parametrize(
    "old,new",
    [
        ('description = "a part', 'prompts = ["x"]\ndescription = "a part'),
        ('description = "a part of it glows softly"\n', ""),
        ("{motion_text} Static", "Static"),
        ("{motion_text} Static", "{motion_text} {pose} Static"),
        ('backend = "fake"', 'backend = "qwen"'),  # qwen needs a pinned revision
        ('backend = "fake"', 'backend = "gpt"'),
        ("[plan]", "[unused]"),  # vlm actions need [plan]
    ],
)
def test_vlm_config_rules(smoke_cfg, old, new):
    _vlm_cfg(smoke_cfg)
    path = smoke_cfg.config_path
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))
    with pytest.raises((ConfigError, TypeError)):
        load_config(path)


def test_sample_records_the_plan(smoke_cfg):
    from dotkoma.finalize import finalize
    from dotkoma.generate import generate

    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())
    j = _vlm_jobs(cfg)[0]
    generate(cfg.run_dir)
    finalize(cfg.run_dir, cfg.finalize)
    s = json.loads((cfg.run_dir / "final" / j["job_id"] / "sample.json").read_text())
    assert s["prompt_method"] == "vlm" and s["plan"]["request_id"] == j["plan"]["request_id"]


def _edit_slime(tmp_path):
    from dotkoma.imaging import load_rgba, save_png

    png = tmp_path / "tests" / "fixtures" / "sprites" / "slime.png"
    rgba = load_rgba(png)
    rgba[rgba[..., 3] > 0, 0] ^= 1  # recolour in place: same row_id and path
    save_png(png, rgba)


def test_edited_sprite_or_caption_needs_a_new_plan(smoke_cfg, tmp_path):
    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    rid = lambda: {q["request_id"] for q in read_jsonl(cfg.run_dir / "plan_requests.jsonl")}  # noqa: E731
    first = rid()
    _edit_slime(tmp_path)
    prepare(cfg)
    second = rid()
    assert first.isdisjoint(second)
    manifest = tmp_path / "tests" / "fixtures" / "sprites" / "manifest.jsonl"
    manifest.write_text(manifest.read_text().replace('"a slime"', '"a green slime"'))
    prepare(cfg)
    assert rid().isdisjoint(second)


def test_edited_sprite_gets_a_new_vlm_job_even_with_the_same_text(smoke_cfg, tmp_path):
    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())
    before = _vlm_jobs(cfg)[0]
    _edit_slime(tmp_path)
    prepare(cfg)
    plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())  # same motion_text
    after = _vlm_jobs(cfg)[0]
    assert after["prompt"] == before["prompt"] and after["job_id"] != before["job_id"]


def test_a_later_failure_does_not_hide_an_answer(smoke_cfg):
    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())
    ok = read_jsonl(cfg.run_dir / "plans.jsonl")[-1]
    append_jsonl(cfg.run_dir / "plans.jsonl", {"request_id": ok["request_id"], "status": "failed"})
    assert len(_vlm_jobs(cfg)) == 1
    assert plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())["pending"] == 0


def test_plan_needs_the_request_file(smoke_cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        plan(tmp_path / "nowhere", planner_factory=lambda spec: FakePlanner())
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "plan_requests.jsonl").write_text("")
    assert plan(empty, planner_factory=lambda spec: FakePlanner())["pending"] == 0


def test_plan_syncs_each_answer(smoke_cfg, tmp_path):
    cfg = _vlm_cfg(smoke_cfg)
    prepare(cfg)
    sync = tmp_path / "drive"
    plan(cfg.run_dir, sync_dir=sync, planner_factory=lambda spec: FakePlanner())
    assert (sync / "plans.jsonl").read_bytes() == (cfg.run_dir / "plans.jsonl").read_bytes()
    assert len(read_jsonl(sync / "plans.jsonl")) == 2


def test_plan_refuses_a_sprite_that_does_not_match_the_request(smoke_cfg):
    from dotkoma.imaging import load_rgba, save_png

    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    q = read_jsonl(cfg.run_dir / "plan_requests.jsonl")[0]
    png = cfg.run_dir / q["input_png"]
    rgba = load_rgba(png)
    rgba[rgba[..., 3] > 0, 0] ^= 1  # an older inputs/ uploaded with newer requests
    save_png(png, rgba)
    assert plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())["failed"] == 1
    assert read_jsonl(cfg.run_dir / "plans.jsonl")[-1]["status"] == "failed"


def test_sync_is_retried_on_resume_and_atomic(smoke_cfg, tmp_path, monkeypatch):
    import dotkoma.plan as plan_mod

    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    sync = tmp_path / "drive"
    calls = []
    real = plan_mod.copy_atomic

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 1:  # the copy after the first answer fails
            raise OSError("drive unavailable")
        real(src, dst)

    monkeypatch.setattr(plan_mod, "copy_atomic", flaky)
    with pytest.raises(OSError):
        plan(cfg.run_dir, sync_dir=sync, planner_factory=lambda spec: FakePlanner())
    assert not (sync / "plans.jsonl").exists()
    # nothing left to plan, but the resume still brings the backup up to date
    assert (
        plan(cfg.run_dir, sync_dir=sync, planner_factory=lambda spec: FakePlanner())["pending"] == 0
    )
    assert (sync / "plans.jsonl").read_bytes() == (cfg.run_dir / "plans.jsonl").read_bytes()


def test_copy_atomic_keeps_the_old_copy_on_failure(tmp_path, monkeypatch):
    import shutil

    from dotkoma.records import copy_atomic

    src, dst = tmp_path / "a.jsonl", tmp_path / "out" / "a.jsonl"
    src.write_text("new\n")
    dst.parent.mkdir()
    dst.write_text("old\n")

    def broken(a, b):
        open(b, "w").write("partial")
        raise OSError("vm gone")

    monkeypatch.setattr(shutil, "copy2", broken)
    with pytest.raises(OSError):
        copy_atomic(src, dst)
    assert dst.read_text() == "old\n" and list(dst.parent.iterdir()) == [dst]


def test_generate_refuses_stale_inputs_for_a_vlm_job(smoke_cfg):
    from dotkoma.generate import generate, raw_path
    from dotkoma.imaging import load_rgba, save_png

    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())
    j = _vlm_jobs(cfg)[0]
    assert len(j["sprite_sha256"]) == 64
    ref = cfg.run_dir / j["reference_png"]
    rgba = load_rgba(ref)
    rgba[0, 0, 0] ^= 1  # a reference that was not made from this sprite
    save_png(ref, rgba)
    generate(cfg.run_dir)
    last = [r for r in read_jsonl(cfg.run_dir / "results.jsonl") if r["job_id"] == j["job_id"]]
    assert last[-1]["status"] == "failed" and "does not match" in last[-1]["error"]
    assert not raw_path(cfg.run_dir, j["job_id"]).exists()


def test_finalize_refuses_a_changed_sprite_for_a_vlm_job(smoke_cfg):
    from dotkoma.finalize import finalize
    from dotkoma.generate import generate
    from dotkoma.imaging import load_rgba, save_png

    cfg = _vlm_cfg(smoke_cfg, ("slime",))
    prepare(cfg)
    plan(cfg.run_dir, planner_factory=lambda spec: FakePlanner())
    j = _vlm_jobs(cfg)[0]
    generate(cfg.run_dir)
    png = cfg.run_dir / j["input_png"]
    rgba = load_rgba(png)
    rgba[rgba[..., 3] > 0, 0] ^= 1  # inputs/ rewritten, jobs.jsonl not yet
    save_png(png, rgba)
    index = {r["job_id"]: r for r in finalize(cfg.run_dir, cfg.finalize)}
    assert index[j["job_id"]]["status"] == "finalize_failed"
    assert not (cfg.run_dir / "final" / j["job_id"]).exists()
