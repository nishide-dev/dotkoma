import numpy as np
import pytest

from dotkoma.config import FinalizeConfig
from dotkoma.finalize import run_checks, select_frames


def test_uniform_window_gives_seamless_timing():
    idx, ms = select_frames(24, 12.0, None, 6)
    assert idx == [0, 4, 8, 12, 16, 20]
    assert ms == [330] * 6


def test_window_and_explicit_indices():
    idx, ms = select_frames(49, 24.0, {"start": 8, "end": 32, "count": 4}, 6)
    assert idx == [8, 14, 20, 26] and ms == [250] * 4
    idx, ms = select_frames(49, 24.0, {"indices": [0, 6, 18]}, 6)
    assert idx == [0, 6, 18] and ms == [250, 500, 380]


def test_bad_selection_raises():
    with pytest.raises(ValueError):
        select_frames(10, 24.0, {"indices": [0, 12]}, 6)
    with pytest.raises(ValueError):
        select_frames(10, 24.0, {"start": 0, "end": 3, "count": 6}, 6)


def _sprite(x):
    f = np.zeros((16, 16, 4), np.uint8)
    f[6:10, x : x + 4] = (200, 50, 50, 255)
    return f


def test_loop_jump_and_identical_flags():
    cfg = FinalizeConfig()
    smooth = [_sprite(x) for x in (4, 5, 6, 5)]
    assert "loop_jump" not in run_checks(smooth, smooth[0], 0, 0, cfg)["flags"]
    jumpy = [_sprite(x) for x in (0, 1, 2, 3, 4, 5, 6, 7)]
    assert "loop_jump" in run_checks(jumpy, jumpy[0], 0, 0, cfg)["flags"]
    same = [_sprite(4)] * 4
    r = run_checks(same, same[0], 0, 0, cfg)
    assert "all_identical" in r["flags"] and r["auto_verdict"] == "reject"
    empty = [np.zeros((16, 16, 4), np.uint8)] + smooth[1:]
    assert "empty_frame" in run_checks(empty, smooth[0], 0, 0, cfg)["flags"]
