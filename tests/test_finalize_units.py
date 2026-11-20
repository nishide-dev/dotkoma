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


def test_identity_drift_ignores_translation_but_not_redrawing():
    from dotkoma.finalize import identity_drift

    src = _sprite(4)
    assert identity_drift(_sprite(6), src, 4) == 0.0  # moved, same drawing
    other = np.zeros((16, 16, 4), np.uint8)
    other[2:14, 7:9] = (50, 50, 200, 255)  # a different shape and colour
    assert identity_drift(other, src, 4) > 0.9
    flags = run_checks([src, other, src, src], src, 0, 0, FinalizeConfig())["flags"]
    assert "identity_drift" in flags


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right"])
def test_identity_drift_counts_pixels_cut_at_the_grid_edge(edge):
    from dotkoma.finalize import identity_drift

    src = np.zeros((16, 16, 4), np.uint8)
    sl = {
        "top": (slice(0, 4), slice(6, 10)),
        "bottom": (slice(12, 16), slice(6, 10)),
        "left": (slice(6, 10), slice(0, 4)),
        "right": (slice(6, 10), slice(12, 16)),
    }[edge]
    src[sl] = (200, 50, 50, 255)
    assert identity_drift(src, src, 4) == 0.0
    cut = src.copy()
    row_or_col = {
        "top": (0, slice(None)),
        "bottom": (15, slice(None)),
        "left": (slice(None), 0),
        "right": (slice(None), 15),
    }[edge]
    cut[row_or_col] = 0  # the outermost row/column is missing
    assert identity_drift(cut, src, 4) == pytest.approx(4 / 16)


def test_small_overflow_of_a_local_motion_is_a_note():
    src = np.zeros((16, 16, 4), np.uint8)
    src[4:12, 4:12] = 255
    moved = src.copy()
    moved[4, 4] = 0
    frames = [src, moved, src, moved]
    cfg = FinalizeConfig()

    def check(edge, motion):
        r = run_checks(frames, src, edge, 0, cfg, motion)
        return r["flags"], r["notes"]

    assert check(0.01, "local") == ([], [])
    assert check(0.05, "local") == ([], ["edge_overflow"])
    assert check(0.05, "global") == (["edge_cut"], [])  # a hop that leaves the grid
    assert check(0.10, "local") == (["edge_cut"], [])  # too much is still a flag
