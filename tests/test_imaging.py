from pathlib import Path

import numpy as np
from PIL import Image

from dotkoma.imaging import (
    GridBox,
    Layout,
    choose_key_color,
    foreground_mask,
    hex_to_rgb,
    load_rgba,
    make_reference,
    measure_background,
    reduce_frame,
    write_gif,
)

FIX = Path(__file__).parent / "fixtures" / "sprites"
KEYS = ("#00ff00", "#ff00ff", "#0000ff")


def test_key_color_avoids_sprite_colours():
    slime = load_rgba(FIX / "slime.png")  # green
    assert choose_key_color(slime, KEYS) != "#00ff00"


def test_reference_roundtrip_is_exact():
    for name in ("slime", "ghost", "candle"):
        src = load_rgba(FIX / f"{name}.png")
        key = choose_key_color(src, KEYS)
        layout = Layout.centered(320, 256, 10)
        ref = make_reference(src, layout, key)
        assert ref.shape == (256, 320, 3)
        bg = measure_background(ref)
        assert tuple(bg) == hex_to_rgb(key)
        out = reduce_frame(
            ref, foreground_mask(ref, bg, 60), GridBox.from_layout(layout, 320, 256), 0.5
        )
        np.testing.assert_array_equal(out, src)


def test_grid_box_follows_resized_video():
    src = load_rgba(FIX / "ghost.png")
    layout = Layout.centered(320, 256, 10)
    ref = make_reference(src, layout, "#00ff00")
    half = np.asarray(Image.fromarray(ref).resize((160, 128), Image.NEAREST))
    bg = measure_background(half)
    out = reduce_frame(
        half, foreground_mask(half, bg, 60), GridBox.from_layout(layout, 160, 128), 0.5
    )
    np.testing.assert_array_equal(out, src)


def test_gif_keeps_frames_timing_and_transparency(tmp_path):
    src = load_rgba(FIX / "candle.png")
    frames = [src, np.roll(src, 1, axis=1)]
    p = tmp_path / "a.gif"
    write_gif(p, frames, [100, 200], scale=8)
    with Image.open(p) as im:
        assert im.n_frames == 2 and im.size == (128, 128)
        assert im.info["loop"] == 0
        durs = []
        for k in range(im.n_frames):
            im.seek(k)
            durs.append(im.info["duration"])
            rgba = np.asarray(im.convert("RGBA"))[::8, ::8]
            np.testing.assert_array_equal(rgba[..., 3] > 0, frames[k][..., 3] > 0)
        assert durs == [100, 200]
