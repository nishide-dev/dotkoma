"""Load and validate the TOML run configuration.

The config is the single source of truth for one run. Everything that affects
a generated video is folded into the job_id (see ``records.job_id``), and
everything that affects post-processing is folded into ``postprocess_id``.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    kind: str  # "atlas" | "local"
    # atlas
    repo_id: str = "Angais/pixel-art-atlas-1.4m"
    revision: str = ""
    shards: tuple[str, ...] = ()
    # local: a directory with manifest.jsonl ({row_id, subject, caption, path})
    path: str = ""


@dataclass(frozen=True)
class SpritePick:
    subject: str
    actions: tuple[str, ...]
    row_id: int | None = None  # pin a specific sprite; otherwise lowest row_id wins
    count: int = 1  # how many distinct sprites of this subject to take
    # which prompt wordings of each action to run: "all" or indices into ActionConfig.prompts
    prompt_variants: str | tuple[int, ...] = (0,)


@dataclass(frozen=True)
class ActionConfig:
    """One motion. ``prompts`` holds alternative wordings of the same instruction.

    Variants are an experiment variable: compare acceptance rate per variant, then
    keep the best wording for mass generation. TOML accepts ``prompt = "..."`` as a
    shorthand for a single variant.
    """

    name: str
    prompts: tuple[str, ...]
    motion: str = "global"  # "local" | "global": informational, used for coverage stats
    method: str = "template"  # how the prompts were written: "template" | "vlm" | ...

    def variant_indices(self, selection: str | tuple[int, ...]) -> tuple[int, ...]:
        if selection == "all":
            return tuple(range(len(self.prompts)))
        return tuple(selection)


@dataclass(frozen=True)
class GenerationConfig:
    backend: str  # "diffusers" | "fake"
    model_id: str = ""
    model_revision: str = "main"
    width: int = 1280
    height: int = 704
    num_frames: int = 49
    fps: int = 24
    steps: int = 40
    guidance: float = 5.0
    negative_prompt: str = ""
    seed: int | None = None  # fixed seed; None derives one per (row_id, action)
    offload: bool = True
    # raw video encoding; yuv444p avoids chroma blur at sprite edges
    pix_fmt: str = "yuv444p"
    crf: int = 10


@dataclass(frozen=True)
class ReferenceConfig:
    scale: int = 24  # integer nearest-neighbour upscale of the 16x16 sprite
    key_colors: tuple[str, ...] = (
        "#00ff00",
        "#ff00ff",
        "#0000ff",
        "#00ffff",
        "#ffff00",
        "#ff0000",
    )


@dataclass(frozen=True)
class FinalizeConfig:
    num_frames: int = 6
    min_frames: int = 4
    max_frames: int = 8
    bg_tolerance: float = 60.0  # RGB distance to the measured background colour
    background: str = "per_frame"  # "per_frame" | "first_frame": where the key colour is measured
    coverage_threshold: float = 0.5  # foreground share of a cell to become opaque
    palette: str = "source"  # "source": snap colours to the sprite's palette, "none"
    preview_scale: int = 8
    # soft-check thresholds
    loop_jump_factor: float = 2.5
    edge_cut_fraction: float = 0.02
    background_unstable_fraction: float = 0.05
    identity_drift_threshold: float = 0.3  # see finalize.identity_drift
    identity_max_shift: int = 4


@dataclass(frozen=True)
class SplitConfig:
    val: float = 0.1
    test: float = 0.1
    salt: str = "dotkoma-v1"


@dataclass(frozen=True)
class Config:
    version: str
    run_name: str
    work_dir: Path
    source: SourceConfig
    sprites: tuple[SpritePick, ...]
    actions: dict[str, ActionConfig]
    generation: GenerationConfig
    reference: ReferenceConfig
    finalize: FinalizeConfig
    split: SplitConfig
    config_path: Path = field(default=Path("."))

    @property
    def run_dir(self) -> Path:
        return self.work_dir / self.run_name

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["work_dir"] = str(self.work_dir)
        d["config_path"] = str(self.config_path)
        return d


class ConfigError(ValueError):
    pass


def _tuple(v: Any) -> tuple:
    return tuple(v) if v is not None else ()


def _action(name: str, spec: dict) -> ActionConfig:
    spec = dict(spec)
    if "prompt" in spec and "prompts" in spec:
        raise ConfigError(f"action {name!r}: set either prompt or prompts, not both")
    if "prompt" in spec:
        spec["prompts"] = [spec.pop("prompt")]
    prompts = _tuple(spec.pop("prompts", None))
    if not prompts:
        raise ConfigError(f"action {name!r}: needs at least one prompt")
    return ActionConfig(name=name, prompts=prompts, **spec)


def _variants(v: Any) -> str | tuple[int, ...]:
    if v == "all":
        return "all"
    if isinstance(v, list) and v and all(isinstance(i, int) for i in v):
        return tuple(v)
    raise ConfigError(f'prompt_variants must be "all" or a non-empty list of ints, got {v!r}')


def load_config(path: str | Path, work_dir: str | Path | None = None) -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    try:
        run = raw.get("run", {})
        src = dict(raw["source"])
        src["shards"] = _tuple(src.get("shards"))
        source = SourceConfig(**src)
        if source.kind == "local" and source.path:
            # resolve relative to the config file so tests/CI work from any cwd
            p = Path(source.path)
            if not p.is_absolute():
                source = SourceConfig(**{**src, "path": str((path.parent / p).resolve())})

        actions = {name: _action(name, spec) for name, spec in raw["actions"].items()}
        sprites = tuple(
            SpritePick(
                subject=s["subject"],
                actions=_tuple(s["actions"]),
                row_id=s.get("row_id"),
                count=s.get("count", 1),
                prompt_variants=_variants(s.get("prompt_variants", [0])),
            )
            for s in raw["sprites"]
        )
        ref = dict(raw.get("reference", {}))
        if "key_colors" in ref:
            ref["key_colors"] = _tuple(ref["key_colors"])
        cfg = Config(
            version=raw["version"],
            run_name=run.get("name", path.stem),
            work_dir=Path(work_dir or run.get("work_dir", "work")),
            source=source,
            sprites=sprites,
            actions=actions,
            generation=GenerationConfig(**raw["generation"]),
            reference=ReferenceConfig(**ref),
            finalize=FinalizeConfig(**raw.get("finalize", {})),
            split=SplitConfig(**raw.get("split", {})),
            config_path=path,
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"{path}: invalid config: {e}") from e

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.source.kind not in ("atlas", "local"):
        raise ConfigError(f"unknown source.kind {cfg.source.kind!r}")
    if cfg.source.kind == "atlas" and not cfg.source.revision:
        raise ConfigError("source.revision must pin a dataset commit for atlas sources")
    if cfg.generation.backend not in ("diffusers", "fake"):
        raise ConfigError(f"unknown generation.backend {cfg.generation.backend!r}")
    for pick in cfg.sprites:
        for a in pick.actions:
            if a not in cfg.actions:
                raise ConfigError(f"sprite {pick.subject!r} uses undefined action {a!r}")
            n = len(cfg.actions[a].prompts)
            bad = [
                i for i in cfg.actions[a].variant_indices(pick.prompt_variants) if not 0 <= i < n
            ]
            if bad:
                raise ConfigError(
                    f"sprite {pick.subject!r}: action {a!r} has {n} prompts, no variant {bad}"
                )
    g, r = cfg.generation, cfg.reference
    if 16 * r.scale > min(g.width, g.height):
        raise ConfigError(f"16*scale={16 * r.scale} does not fit the {g.width}x{g.height} canvas")
    f = cfg.finalize
    if not (f.min_frames <= f.num_frames <= f.max_frames):
        raise ConfigError("finalize.num_frames must lie in [min_frames, max_frames]")
    if f.background not in ("per_frame", "first_frame"):
        raise ConfigError(f"unknown finalize.background {f.background!r}")
    if f.palette not in ("source", "none"):
        raise ConfigError(f"unknown finalize.palette {f.palette!r}")
