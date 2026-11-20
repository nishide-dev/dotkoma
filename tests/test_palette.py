"""Loop palette: source colours plus new colours that recur across the loop."""

from dataclasses import replace

import numpy as np

from dotkoma.config import FinalizeConfig
from dotkoma.finalize import loop_palette
from dotkoma.imaging import stable_new_colors

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
    assert len(added) == 1 and len(pal) == 2
    pal, added = loop_palette(frames, SOURCE, replace(cfg, palette="source"))
    assert pal.tolist() == [list(DARK)] and len(added) == 0
    pal, added = loop_palette(frames, SOURCE, replace(cfg, palette="none"))
    assert len(pal) == 0 and len(added) == 0
