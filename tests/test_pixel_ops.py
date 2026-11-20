"""pixel_ops: whole-sprite motion made by translating the 16x16 sprite."""

import json
from dataclasses import replace

import numpy as np
import pytest

from dotkoma.config import ConfigError, SpritePick, load_config
from dotkoma.finalize import finalize
from dotkoma.generate import generate, pending_jobs
from dotkoma.imaging import load_rgba, save_png
from dotkoma.pixel_ops import expand, fits, render, translate, validate
from dotkoma.prepare import prepare
from dotkoma.records import read_jsonl

PIXEL_ACTIONS = """
[actions.px_bob]
generation_method = "pixel_ops"
pixel_ops = { kind = "bob", amplitude = 1, frames = 6, frame_ms = 120 }
[actions.px_hop]
generation_method = "pixel_ops"
pixel_ops = { kind = "hop", amplitude = 3, frames = 6 }
"""


def _with_pixel_actions(smoke_cfg, sprites_toml: str):
    path = smoke_cfg.config_path
    text = path.read_text().replace("[actions.hop]", sprites_toml + "\n[actions.hop]", 1)
    path.write_text(text + PIXEL_ACTIONS)
    return load_config(path, work_dir=smoke_cfg.work_dir)


def test_presets_are_closed_loops():
    assert expand({"kind": "bob", "amplitude": 1})["offsets"] == [
        [0, 0],
        [0, -1],
        [0, -1],
        [0, 0],
        [0, 1],
        [0, 1],
    ]
    hop = expand({"kind": "hop", "amplitude": 2})["offsets"]
    assert hop == [[0, 0], [0, 0], [0, -1], [0, -2], [0, -2], [0, -1]]
    assert all(dy <= 0 for _, dy in hop)  # never sinks below the ground
    sway = expand({"kind": "sway", "amplitude": 1, "frames": 4, "frame_ms": [80, 90, 80, 90]})
    assert sway == {"offsets": [[0, 0], [1, 0], [0, 0], [-1, 0]], "durations_ms": [80, 90, 80, 90]}
    custom = expand({"kind": "custom", "offsets": [[0, 0], [0, -1]]})
    assert custom["durations_ms"] == [100, 100]


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "spin"},
        {"kind": "custom"},
        {"kind": "bob", "amplitude": 0},
        {"kind": "bob", "frames": 1},
        {"kind": "hop", "frames": 3, "ground": 2},
        {"kind": "bob", "frame_ms": 0},
        {"kind": "sway", "frames": 4, "frame_ms": [100, 100]},
    ],
)
def test_bad_specs_are_rejected(spec):
    assert validate(spec) is not None


def test_translate_and_fit():
    src = np.zeros((16, 16, 4), np.uint8)
    src[14:16, 4:8] = [255, 0, 0, 255]  # touches the bottom edge
    up = translate(src, 0, -2)
    np.testing.assert_array_equal(up[12:14], src[14:16])
    assert not up[14:].any()
    assert fits(src, [[0, 0], [0, -3]])
    assert not fits(src, [[0, 0], [0, 1]])
    frames, lost = render(src, [[0, 0], [0, 1]])
    assert len(frames) == 2 and lost == pytest.approx(0.5)


def test_config_rules(smoke_cfg):
    path = smoke_cfg.config_path
    base = path.read_text()
    pixel = 'generation_method = "pixel_ops"\n'
    bad = [
        pixel + 'prompt = "x"\npixel_ops = { kind = "bob" }',  # prompts on a pixel_ops action
        'prompt = "x"\npixel_ops = { kind = "bob" }',  # pixel_ops without the method
        'generation_method = "magic"\nprompt = "x"',
        pixel + 'pixel_ops = { kind = "bob", frames = 12 }',  # beyond finalize.max_frames
    ]
    for body in bad:
        path.write_text(base + f"\n[actions.extra]\n{body}\n")
        with pytest.raises(ConfigError):
            load_config(path)


def test_pixel_ops_jobs_skip_the_gpu_and_finalize_exactly(smoke_cfg):
    cfg = _with_pixel_actions(
        smoke_cfg,
        '[[sprites]]\nsubject = "slime"\nactions = ["px_bob", "px_hop"]\n'
        # candle has 2 free rows above it: the 3 px hop must be skipped, bob kept
        '[[sprites]]\nsubject = "candle"\nactions = ["px_bob", "px_hop"]\n',
    )
    run = cfg.run_dir
    jobs = read_jsonl(prepare(cfg))
    px = [j for j in jobs if j["generation_method"] == "pixel_ops"]
    assert sorted((j["source"]["subject"], j["requested_action"]) for j in px) == [
        ("candle", "px_bob"),
        ("slime", "px_bob"),
        ("slime", "px_hop"),
    ]
    assert all("model" not in j and j["prompt_variant"] is None for j in px)

    # the GPU stage never sees them
    assert {j["job_id"] for j in pending_jobs(run)}.isdisjoint(j["job_id"] for j in px)
    generate(run)
    assert not any(
        r["job_id"] in {j["job_id"] for j in px} for r in read_jsonl(run / "results.jsonl")
    )

    index = {r["job_id"]: r for r in finalize(run, cfg.finalize)}
    for j in px:
        r = index[j["job_id"]]
        assert r["status"] == "ok" and r["auto_verdict"] == "candidate", r
        s = json.loads((run / "final" / j["job_id"] / "sample.json").read_text())
        src = load_rgba(run / s["input_png"])
        frames = [load_rgba(run / "final" / j["job_id"] / f) for f in s["frames"]]
        for f, (dx, dy) in zip(frames, j["pixel_ops"]["offsets"], strict=True):
            np.testing.assert_array_equal(f, translate(src, dx, dy))
        assert s["durations_ms"] == j["pixel_ops"]["durations_ms"]
        assert s["generation_method"] == "pixel_ops"
        assert (run / "final" / j["job_id"] / "preview.gif").exists()
    bob = next(j for j in px if j["requested_action"] == "px_bob")
    assert bob["pixel_ops"]["durations_ms"] == [120] * 6


def test_pixel_ops_job_id_tracks_the_motion(smoke_cfg):
    cfg = _with_pixel_actions(smoke_cfg, '[[sprites]]\nsubject = "slime"\nactions = ["px_bob"]\n')
    first = {j["job_id"] for j in read_jsonl(prepare(cfg))}
    path = cfg.config_path
    path.write_text(path.read_text().replace("frame_ms = 120", "frame_ms = 140"))
    second = {j["job_id"] for j in read_jsonl(prepare(load_config(path, work_dir=cfg.work_dir)))}
    assert len(first - second) == 1  # only the pixel_ops job changed


def test_generation_section_is_optional_only_for_pixel_ops(smoke_cfg):
    path = smoke_cfg.config_path
    text = path.read_text()
    head, _, rest = text.partition("[generation]")
    no_gen = head + "[reference]" + rest.partition("[reference]")[2]
    path.write_text(no_gen)
    with pytest.raises(ConfigError, match="need a \\[generation\\] backend"):
        load_config(path)  # the smoke sprites still use video actions
    sprites = '[[sprites]]\nsubject = "slime"\nactions = ["px_hop"]\n'
    body = no_gen[: no_gen.index("[[sprites]]")] + sprites + no_gen[no_gen.index("[actions.hop]") :]
    path.write_text(body + PIXEL_ACTIONS)
    cfg = load_config(path, work_dir=smoke_cfg.work_dir)
    assert cfg.generation.backend == "none"
    jobs = read_jsonl(prepare(cfg))
    assert [j["requested_action"] for j in jobs] == ["px_hop"]
    assert not list((cfg.run_dir / "inputs").glob("*_ref_*.png"))  # no video, no reference


def test_each_pixel_motion_picks_its_own_sprite(smoke_cfg):
    from dotkoma.prepare import _sprite_groups
    from dotkoma.source import Sprite

    tall = np.zeros((16, 16, 4), np.uint8)
    tall[:, 6:10] = 255  # room left/right only: sway fits, hop does not
    low = np.zeros((16, 16, 4), np.uint8)
    low[6:16, 0:16] = 255  # room above only: hop fits, sway does not
    pool = [Sprite(1, "x", "", tall), Sprite(2, "x", "", low)]
    cfg = _with_pixel_actions(smoke_cfg, "")
    sway = '[actions.px_sway]\ngeneration_method = "pixel_ops"\npixel_ops = { kind = "sway" }\n'
    cfg.config_path.write_text(cfg.config_path.read_text() + sway)
    cfg = load_config(cfg.config_path, work_dir=cfg.work_dir)

    pick = SpritePick("x", ("hop", "px_hop", "px_sway"))
    groups = [([sp.row_id for sp in p], a) for p, a in _sprite_groups(pick, pool, cfg)]
    # video actions keep the whole pool in order; each motion gets the sprites it fits
    assert groups == [([1, 2], ("hop",)), ([2], ("px_hop",)), ([1], ("px_sway",))]
    pinned = SpritePick("x", ("px_hop",), row_id=1)
    assert [a for _, a in _sprite_groups(pinned, pool[:1], cfg)] == [("px_hop",)]


def test_video_job_ids_unchanged_from_main(smoke_cfg):
    # job_ids of configs/smoke.toml before pixel_ops existed (main @ 684c7dd)
    assert sorted(j["job_id"] for j in read_jsonl(prepare(smoke_cfg))) == [
        "4947ed203f3e31ed",
        "9f5d10ffd8a4f3ce",
        "b59d4fff98100f14",
        "b5ce6150b8d5fd6a",
        "c798335cdc692b15",
        "c7b16f663db000f4",
    ]


def test_edited_sprite_gets_a_new_pixel_ops_job(smoke_cfg, tmp_path):
    cfg = _with_pixel_actions(smoke_cfg, '[[sprites]]\nsubject = "slime"\nactions = ["px_bob"]\n')
    px = lambda: {  # noqa: E731
        j["job_id"] for j in read_jsonl(prepare(cfg)) if j["generation_method"] == "pixel_ops"
    }
    before = px()
    assert len(before) == 1
    png = tmp_path / "tests" / "fixtures" / "sprites" / "slime.png"
    rgba = load_rgba(png)
    rgba[rgba[..., 3] > 0, 0] ^= 1  # recolour in place, same row_id and path
    save_png(png, rgba)
    assert px().isdisjoint(before)


def test_pixel_ops_postprocess_id_tracks_check_settings(smoke_cfg):
    cfg = _with_pixel_actions(smoke_cfg, '[[sprites]]\nsubject = "slime"\nactions = ["px_bob"]\n')
    prepare(cfg)
    first = {r["job_id"]: r.get("postprocess_id") for r in finalize(cfg.run_dir, cfg.finalize)}
    stricter = replace(cfg.finalize, identity_drift_threshold=0.1)
    second = {r["job_id"]: r.get("postprocess_id") for r in finalize(cfg.run_dir, stricter)}
    jobs = read_jsonl(cfg.run_dir / "jobs.jsonl")
    px = next(j["job_id"] for j in jobs if j["generation_method"] == "pixel_ops")
    assert first[px] and second[px] and first[px] != second[px]
