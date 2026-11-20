"""Loop palette: source colours plus new colours that recur across the loop."""

from dataclasses import replace

import numpy as np
import pytest

from dotkoma.config import ConfigError, FinalizeConfig, load_config
from dotkoma.finalize import loop_palette
from dotkoma.imaging import snap_to_palette, stable_new_colors, write_gif

DARK = (30, 30, 40)


def _frame(extra: dict[tuple[int, int], tuple[int, int, int]]) -> np.ndarray:
    f = np.zeros((16, 16, 4), np.uint8)
    f[4:12, 4:12] = (*DARK, 255)
    for (y, x), rgb in extra.items():
        f[y, x] = (*rgb, 255)
    return f


SOURCE = _frame({})
EYES = {(6, 6): (250, 250, 250), (6, 9): (255, 255, 252)}  # white, slightly noisy


def test_recurring_new_colour_is_kept():
    frames = [SOURCE] + [_frame(EYES)] * 4 + [SOURCE]  # open in 4 of 6 frames
    added = stable_new_colors(frames, np.array([DARK], np.uint8), 80, 0.5, 2)
    assert added.shape == (1, 3) and added.min() >= 250


def test_one_off_and_tiny_colours_are_dropped():
    flash = [_frame(EYES)] + [SOURCE] * 5  # 1 of 6 frames
    speck = [_frame({(6, 6): (250, 250, 250)})] * 6  # 1 pixel per frame
    near = [_frame({(6, 6): (60, 60, 70), (6, 7): (60, 60, 70)})] * 6  # close to DARK
    pal = np.array([DARK], np.uint8)
    for frames in (flash, speck, near):
        assert len(stable_new_colors(frames, pal, 80, 0.5, 2)) == 0


def test_separate_new_colours_stay_separate():
    two = {
        (6, 6): (250, 250, 250),
        (6, 7): (250, 250, 250),
        (9, 6): (230, 40, 40),
        (9, 7): (230, 40, 40),
    }
    added = stable_new_colors([_frame(two)] * 4, np.array([DARK], np.uint8), 80, 0.5, 2)
    assert sorted(map(tuple, added.tolist())) == [(230, 40, 40), (250, 250, 250)]


def test_loop_palette_modes():
    frames = [_frame(EYES)] * 4
    cfg = FinalizeConfig()
    assert cfg.palette == "source+stable"
    pal, added = loop_palette(frames, SOURCE, cfg)
    assert len(added) == 1 and pal.tolist() == [list(DARK)]
    pal, added = loop_palette(frames, SOURCE, replace(cfg, palette="source"))
    assert pal.tolist() == [list(DARK)] and len(added) == 0
    pal, added = loop_palette(frames, SOURCE, replace(cfg, palette="none"))
    assert len(pal) == 0 and len(added) == 0


def test_one_off_colour_is_not_pulled_onto_an_added_colour():
    base, added = np.array([DARK], np.uint8), np.array([[250, 250, 250]], np.uint8)
    f = _frame({(6, 6): (252, 250, 249), (8, 8): (180, 180, 120)})  # eye, and a flash
    out = snap_to_palette(f, base, added, 40)
    assert tuple(out[6, 6, :3]) == (250, 250, 250)  # within the group radius: kept
    assert tuple(out[8, 8, :3]) == DARK  # nearest is white, but outside its radius


def test_preview_survives_more_than_255_colours(tmp_path):
    f = np.zeros((16, 16, 4), np.uint8)
    f[..., 0] = np.arange(256).reshape(16, 16)  # 256 distinct opaque colours
    f[..., 3] = 255
    write_gif(tmp_path / "p.gif", [f, f], [100, 100], 1)
    assert (tmp_path / "p.gif").stat().st_size > 0


@pytest.mark.parametrize(
    "line",
    [
        "new_color_distance = 0",
        "new_color_distance = 500",
        "new_color_min_frame_share = 0",
        "new_color_min_frame_share = 2.0",
        "new_color_min_pixels = 0",
        "new_color_min_pixels = 1.5",
    ],
)
def test_bad_new_colour_settings_are_rejected(smoke_cfg, line):
    path = smoke_cfg.config_path
    path.write_text(path.read_text().replace("[finalize]\n", f"[finalize]\n{line}\n"))
    with pytest.raises(ConfigError):
        load_config(path)


def test_boundary_new_colour_settings_load(smoke_cfg):
    path = smoke_cfg.config_path
    lines = "new_color_distance = 442\nnew_color_min_frame_share = 1\nnew_color_min_pixels = 1\n"
    path.write_text(path.read_text().replace("[finalize]\n", f"[finalize]\n{lines}"))
    assert load_config(path).finalize.new_color_min_frame_share == 1
