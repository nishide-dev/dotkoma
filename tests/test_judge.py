"""judge: a VLM checks finalized loops against their requested action."""

import json
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dotkoma.config import ConfigError, JudgeConfig, load_judge_config
from dotkoma.finalize import finalize
from dotkoma.generate import generate
from dotkoma.judge import (
    DEFAULT_CRITERIA,
    FakeJudge,
    build_requests,
    judge,
    parse_verdict,
    pick_frames,
    sheet,
)
from dotkoma.prepare import prepare
from dotkoma.records import read_jsonl, write_jsonl_atomic

ROOT = Path(__file__).resolve().parents[1]
FAKE = JudgeConfig(
    backend="fake",
    model_id="",
    criteria={a: f"it does {a}" for a in ("hop", "bob", "static", "sway")},
)


@pytest.fixture
def run(smoke_cfg):
    prepare(smoke_cfg)
    generate(smoke_cfg.run_dir)
    finalize(smoke_cfg.run_dir, smoke_cfg.finalize)
    return smoke_cfg.run_dir


def _counting(answer=None):
    calls = []

    class J(FakeJudge):
        def __call__(self, image, instruction):
            calls.append((image.size, instruction))
            return answer if answer is not None else super().__call__(image, instruction)

    return calls, lambda spec: J()


def test_parse_verdict():
    v = parse_verdict('<think>x</think>ok {"motion_matches": true, "drawing_kept": false}')
    assert v == {
        "changes": "",
        "motion_matches": True,
        "drawing_kept": False,
        "reason": "",
        "verdict": "reject",
    }
    a = parse_verdict('{"motion_matches": true, "drawing_kept": true, "reason": " r "}')
    assert a["verdict"] == "accept" and a["reason"] == "r"
    for bad in [
        "nothing",
        '{"motion_matches": "yes", "drawing_kept": true}',
        '{"drawing_kept": 1}',
    ]:
        with pytest.raises(ValueError):
            parse_verdict(bad)


def test_pick_frames():
    assert pick_frames(6, 8) == [0, 1, 2, 3, 4, 5]
    picked = pick_frames(25, 8)
    assert picked[0] == 0 and picked[-1] == 24 and len(picked) == 8


def test_sheet_keeps_pixels_and_avoids_a_background_like_the_sprite():
    white = np.zeros((16, 16, 4), np.uint8)
    white[4:12, 4:12] = (255, 255, 255, 255)
    img = sheet(white, [white, white], scale=2, gap=4)
    assert img.size == (3 * 32 + 4 * 4, 32 + 14 + 8)
    px = np.asarray(img)
    bg = px[0, 0]
    assert tuple(bg) != (255, 255, 255)  # white steam stays visible
    assert tuple(px[4 + 14 + 8 * 2 + 1, 4 + 8 * 2 + 1]) == (255, 255, 255)  # sprite pixel


def test_judge_records_each_sample_and_resumes(run):
    calls, factory = _counting()
    counts = judge(run, FAKE, judge_factory=factory)
    judged = [r for r in read_jsonl(run / "finalize.jsonl") if r["auto_verdict"] != "reject"]
    assert counts == {"ok": len(judged), "failed": 0, "pending": 0}
    recs = read_jsonl(run / "judge.jsonl")
    assert {r["job_id"] for r in recs} == {r["job_id"] for r in judged}
    assert all(r["verdict"] == "accept" and r["postprocess_id"] for r in recs)
    assert all("The requested motion is" in i for _, i in calls)
    # resume: nothing left
    assert judge(run, FAKE, judge_factory=factory)["pending"] == 0
    assert len(calls) == len(judged)


def test_failures_are_recorded_and_retried(run):
    _, bad = _counting("I cannot tell")
    counts = judge(run, FAKE, judge_factory=bad)
    assert counts["failed"] > 0 and counts["ok"] == 0
    assert all(r["status"] == "failed" for r in read_jsonl(run / "judge.jsonl"))
    _, good = _counting()
    assert judge(run, FAKE, judge_factory=good)["pending"] == 0  # failed are skipped
    assert judge(run, FAKE, retry_failed=True, judge_factory=good)["ok"] == counts["failed"]


def test_request_id_tracks_frames_and_judge_settings(run):
    before = {q["job_id"]: q["request_id"] for q in build_requests(run, FAKE)}
    other = {q["job_id"]: q["request_id"] for q in build_requests(run, replace(FAKE, scale=8))}
    assert before.keys() == other.keys() and not set(before.values()) & set(other.values())
    jid = next(iter(before))
    sdir = run / "final" / jid
    sample = json.loads((sdir / "sample.json").read_text())
    from dotkoma.imaging import load_rgba, save_png

    f = load_rgba(sdir / sample["frames"][1])
    f[0, 0] = (1, 2, 3, 255)
    save_png(sdir / sample["frames"][1], f)
    after = {q["job_id"]: q["request_id"] for q in build_requests(run, FAKE)}
    assert after[jid] != before[jid]
    assert all(after[k] == before[k] for k in before if k != jid)


def test_only_selected_machine_verdicts_are_judged(run):
    rows = read_jsonl(run / "finalize.jsonl")
    rows[0] = {**rows[0], "auto_verdict": "reject"}
    write_jsonl_atomic(run / "finalize.jsonl", rows)
    ids = {q["job_id"] for q in build_requests(run, FAKE)}
    assert rows[0]["job_id"] not in ids
    only = {q["job_id"] for q in build_requests(run, replace(FAKE, verdicts=("reject",)))}
    assert only == {r["job_id"] for r in rows if r["auto_verdict"] == "reject"}


def test_unknown_action_needs_criteria(run):
    with pytest.raises(ValueError, match="no judge criteria"):
        build_requests(run, JudgeConfig(backend="fake", model_id=""))
    assert set(DEFAULT_CRITERIA) >= {"flicker", "steam", "blink", "wag"}


def test_judge_syncs_and_shares_one_model(run, tmp_path):
    loads = Counter()

    def factory(spec):
        loads["n"] += 1
        return FakeJudge()

    shared: dict = {}
    judge(run, FAKE, judge_factory=factory, shared=shared, sync_dir=tmp_path / "sync")
    judge(run, FAKE, retry_failed=True, judge_factory=factory, shared=shared)
    assert loads["n"] == 1
    assert (tmp_path / "sync" / "judge.jsonl").read_bytes() == (run / "judge.jsonl").read_bytes()


def _write(tmp_path, body):
    p = tmp_path / "judge.toml"
    p.write_text(body)
    return p


def test_judge_config_rules(tmp_path):
    ok = load_judge_config(ROOT / "configs" / "judge.toml")
    assert ok.backend == "qwen" and ok.verdicts == ("candidate", "flagged")
    bad = [
        '[judge]\nbackend = "qwen"\n',  # no pinned revision
        '[judge]\nbackend = "other"\n',
        '[judge]\nbackend = "fake"\nverdicts = ["maybe"]\n',
        '[judge]\nbackend = "fake"\nscale = 0\n',
        '[judge]\nbackend = "fake"\nmax_frames = true\n',
        '[judge]\nbackend = "fake"\ndiff_row = true\n',
        '[judge]\nbackend = "fake"\ninstruction = "{nope}"\n',
        '[judge]\nbackend = "fake"\n[judge.criteria]\nhop = ""\n',
        '[judge]\nbackend = "fake"\ncolour = 1\n',
        "[other]\n",
    ]
    for body in bad:
        with pytest.raises(ConfigError):
            load_judge_config(_write(tmp_path, body))


def test_cli_and_agreement_script(run, tmp_path):
    cfg = _write(
        tmp_path,
        '[judge]\nbackend = "fake"\nmodel_id = ""\n[judge.criteria]\n'
        + "".join(f'{a} = "{a}"\n' for a in FAKE.criteria),
    )
    cli = [sys.executable, "-c", "from dotkoma import main; main()", "judge", str(cfg), str(run)]
    out = subprocess.run(cli, capture_output=True, text=True, check=True).stdout
    assert '"failed": 0' in out
    fin = [r for r in read_jsonl(run / "finalize.jsonl") if r["auto_verdict"] != "reject"]
    labels = [
        {
            "job_id": r["job_id"],
            "postprocess_id": r["postprocess_id"],
            "verdict": "accept" if k % 2 else "reject",
            "motion_matches": bool(k % 2),
            "drawing_kept": True,
            "reason": "",
        }
        for k, r in enumerate(fin)
    ]
    write_jsonl_atomic(run / "review.jsonl", labels)
    script = [sys.executable, str(ROOT / "scripts" / "judge_agreement.py"), str(cfg), str(run)]
    rep = subprocess.run(script, capture_output=True, text=True, check=True).stdout
    assert f"{len(fin)} eligible labelled samples" in rep and "| judge (accept) |" in rep


def test_diff_row_marks_changed_pixels():
    src = np.zeros((16, 16, 4), np.uint8)
    src[4:12, 4:12] = (200, 30, 30, 255)
    moved = src.copy()
    moved[2, 2] = (200, 30, 30, 255)  # new pixel
    moved[11, 11, 3] = 0  # vanished pixel
    moved[5, 5] = (205, 30, 30, 255)  # within tolerance
    from dotkoma.judge import changed_pixels

    assert sorted(zip(*np.nonzero(changed_pixels(moved, src)), strict=True)) == [(2, 2), (11, 11)]
    img = np.asarray(sheet(src, [moved], scale=2, gap=4, diff_row="silhouette"))
    top = 4 + 14
    row2 = 2 * 4 + 32 + 14 + 14
    x = 4 + 32 + 4
    assert img.shape[0] == 2 * (32 + 14 + 4) + 4
    assert tuple(img[row2 + 2 * 2, x + 2 * 2]) == (255, 0, 255)
    assert tuple(img[row2 + 11 * 2, x + 11 * 2]) == (255, 0, 255)
    assert tuple(img[row2 + 5 * 2, x + 5 * 2]) != (255, 0, 255)
    assert tuple(img[top + 2 * 2, x + 2 * 2]) == (200, 30, 30)
    assert tuple(img[row2 + 6 * 2, x + 6 * 2]) == (150, 150, 150)  # unchanged: grey shape
    faded = np.asarray(sheet(src, [moved], scale=2, gap=4, diff_row="faded"))
    assert tuple(faded[row2 + 2 * 2, x + 2 * 2]) == (255, 0, 255)
    with pytest.raises(ValueError):
        sheet(src, [moved], scale=2, diff_row="other")


def test_recorded_judge_specs_keep_their_hashes():
    """Judge 01's judge.jsonl records were written with these specs; a change here
    orphans them (bump SHEET_VERSION or add a setting instead of editing silently)."""
    from dotkoma.judge import judge_spec
    from dotkoma.records import stable_hash

    expected = {
        "v1": "0d6ea301bcd3cb09",
        "diff": "6d6b3f848974bb36",
        "diff-sil": "cadf4ab6cf2b746d",
        "describe": "e88d0b93c2c6dd74",
        "both": "465856fe0c388773",
        "both-sil": "b104897013a6482f",
    }
    for name, h in expected.items():
        cfg = load_judge_config(ROOT / "configs" / "judge-ab" / f"{name}.toml")
        assert stable_hash(judge_spec(cfg), 16) == h, name
    default = load_judge_config(ROOT / "configs" / "judge.toml")
    assert stable_hash(judge_spec(default), 16) == expected["diff-sil"]


def test_default_instruction_follows_the_diff_row():
    from dotkoma.judge import DEFAULT_INSTRUCTIONS, judge_spec

    for mode in ("none", "faded", "silhouette"):
        spec = judge_spec(replace(FAKE, diff_row=mode))
        assert spec["instruction"] == DEFAULT_INSTRUCTIONS[mode]
    assert "bottom row" not in DEFAULT_INSTRUCTIONS["none"]
    assert "faded copy" in DEFAULT_INSTRUCTIONS["faded"]
    assert "flat grey" in DEFAULT_INSTRUCTIONS["silhouette"]


def test_missing_finalize_index_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="finalize"):
        judge(tmp_path, FAKE, judge_factory=lambda spec: FakeJudge())


def test_an_unreadable_sample_is_recorded_and_the_rest_are_judged(run):
    ids = [q["job_id"] for q in build_requests(run, FAKE)]
    broken = ids[0]
    (run / "final" / broken / "sample.json").write_text("{not json")
    counts = judge(run, FAKE, judge_factory=lambda spec: FakeJudge())
    assert counts == {"ok": len(ids) - 1, "failed": 1, "pending": 0}
    recs = {r["job_id"]: r for r in read_jsonl(run / "judge.jsonl")}
    assert recs[broken]["status"] == "failed" and "unreadable" in recs[broken]["error"]
    # skipped on resume, retried with --retry-failed
    assert judge(run, FAKE, judge_factory=lambda spec: FakeJudge())["pending"] == 0
    assert judge(run, FAKE, retry_failed=True, judge_factory=lambda s: FakeJudge())["failed"] == 1


def test_a_sample_changed_during_the_run_is_not_judged_under_the_old_id(run):
    from dotkoma.imaging import load_rgba, save_png

    ids = [q["job_id"] for q in build_requests(run, FAKE)]
    victim = ids[-1]

    class Refinalizes(FakeJudge):
        def __call__(self, image, instruction):
            sdir = run / "final" / victim
            frame = json.loads((sdir / "sample.json").read_text())["frames"][1]
            f = load_rgba(sdir / frame)
            f[0, 0] = (9, 9, 9, 255)
            save_png(sdir / frame, f)
            return super().__call__(image, instruction)

    counts = judge(run, FAKE, judge_factory=lambda spec: Refinalizes())
    recs = {r["job_id"]: r for r in read_jsonl(run / "judge.jsonl")}
    assert recs[victim]["status"] == "failed" and "changed" in recs[victim]["error"]
    assert counts["failed"] == 1
    # the next run builds a new request for the new frames and judges it
    assert judge(run, FAKE, judge_factory=lambda spec: FakeJudge())["ok"] == 1


def test_a_fresh_vm_resumes_from_the_synced_log(run, tmp_path):
    sync = tmp_path / "sync"
    first = judge(run, FAKE, limit=2, judge_factory=lambda s: FakeJudge(), sync_dir=sync)
    assert first["ok"] == 2
    backup = (sync / "judge.jsonl").read_bytes()
    (run / "judge.jsonl").unlink()  # a fresh VM: judge.jsonl was not uploaded
    calls, factory = _counting()
    judge(run, FAKE, judge_factory=factory, sync_dir=sync)
    total = len(build_requests(run, FAKE))
    assert len(calls) == total - 2  # the two synced verdicts are not redone
    synced = (sync / "judge.jsonl").read_bytes()
    assert synced.startswith(backup) and len(read_jsonl(sync / "judge.jsonl")) == total


def test_a_malformed_sample_is_recorded_and_the_rest_are_judged(run):
    ids = [q["job_id"] for q in build_requests(run, FAKE)]
    path = run / "final" / ids[0] / "sample.json"
    sample = json.loads(path.read_text())
    del sample["source"]["subject"]
    path.write_text(json.dumps(sample))
    counts = judge(run, FAKE, judge_factory=lambda spec: FakeJudge())
    assert counts == {"ok": len(ids) - 1, "failed": 1, "pending": 0}


def test_a_stale_backup_does_not_override_a_newer_verdict(run, tmp_path):
    sync = tmp_path / "sync"
    judge(run, FAKE, judge_factory=lambda spec: FakeJudge(), sync_dir=sync)
    recs = read_jsonl(run / "judge.jsonl")
    rid = recs[0]["request_id"]
    old = {**recs[0], "started_at": "2000-01-01T00:00:00+00:00", "verdict": "reject"}
    write_jsonl_atomic(sync / "judge.jsonl", [old])  # backup has only an older answer
    judge(run, FAKE, judge_factory=lambda spec: FakeJudge(), sync_dir=sync)
    from dotkoma.judge import latest_verdicts

    assert latest_verdicts(run)[rid]["verdict"] == "accept"
    assert len(read_jsonl(run / "judge.jsonl")) == len(recs) + 1
    assert read_jsonl(sync / "judge.jsonl") == read_jsonl(run / "judge.jsonl")


def test_merge_logs_keeps_the_order_of_both_logs():
    from dotkoma.judge import merge_logs

    a, b, c, d = ({"request_id": "x", "n": i, "started_at": "t"} for i in range(4))
    assert merge_logs([a], [a, b]) == [a, b]  # same second: backup's order holds
    assert merge_logs([c], [a, b]) == [a, b, c]  # fresh VM: backup is older
    assert merge_logs([a, c], [a, b]) == [a, b, c]
    assert merge_logs([a, b], [a]) == [a, b]
    assert merge_logs([a, c], [a, b, c, d]) == [a, b, c, d]
    assert merge_logs([], [a, b]) == [a, b]
