import pytest

from dotkoma.config import ConfigError, load_config

BASE = """
version = "t"
[source]
kind = "local"
path = "."
[[sprites]]
subject = "slime"
actions = ["hop"]
{pick}
[actions.hop]
{action}
[generation]
backend = "fake"
width = 320
height = 256
[reference]
scale = 10
"""


def _load(tmp_path, action, pick=""):
    p = tmp_path / "c.toml"
    p.write_text(BASE.format(action=action, pick=pick))
    return load_config(p)


def test_single_prompt_is_shorthand_for_one_variant(tmp_path):
    cfg = _load(tmp_path, 'prompt = "hop"')
    assert cfg.actions["hop"].prompts == ("hop",)
    assert cfg.sprites[0].prompt_variants == (0,)


def test_variants_all_and_indices(tmp_path):
    cfg = _load(tmp_path, 'prompts = ["a", "b", "c"]', 'prompt_variants = "all"')
    assert cfg.actions["hop"].variant_indices(cfg.sprites[0].prompt_variants) == (0, 1, 2)
    cfg = _load(tmp_path, 'prompts = ["a", "b", "c"]', "prompt_variants = [2]")
    assert cfg.actions["hop"].variant_indices(cfg.sprites[0].prompt_variants) == (2,)


@pytest.mark.parametrize(
    "action, pick",
    [
        ('prompt = "a"\nprompts = ["b"]', ""),
        ("prompts = []", ""),
        ('prompts = ["a"]', "prompt_variants = [1]"),
        ('prompts = ["a"]', 'prompt_variants = "first"'),
    ],
)
def test_invalid_prompt_settings(tmp_path, action, pick):
    with pytest.raises(ConfigError):
        _load(tmp_path, action, pick)
