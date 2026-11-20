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
