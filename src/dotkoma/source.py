"""Sprite sources: Pixel Art Atlas parquet shards, or a local fixture directory."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SourceConfig
from .imaging import SPRITE, load_rgba


@dataclass(frozen=True)
class Sprite:
    row_id: int
    subject: str
    caption: str
    rgba: np.ndarray

    def __post_init__(self):
        if self.rgba.shape != (SPRITE, SPRITE, 4):
            raise ValueError(f"row {self.row_id}: expected 16x16 RGBA, got {self.rgba.shape}")


def source_ref(cfg: SourceConfig) -> dict:
    """What gets written into every job to trace the sprite back to its origin."""
    if cfg.kind == "atlas":
        return {"kind": "atlas", "repo_id": cfg.repo_id, "revision": cfg.revision}
    return {"kind": "local", "path": Path(cfg.path).name}


def load_sprites(cfg: SourceConfig, subjects: set[str]) -> dict[str, list[Sprite]]:
    """All sprites whose subject is in ``subjects``, sorted by row_id per subject."""
    if cfg.kind == "atlas":
        found = _load_atlas(cfg, subjects)
    else:
        found = _load_local(cfg, subjects)
    return {k: sorted(v, key=lambda s: s.row_id) for k, v in found.items()}


def _load_local(cfg: SourceConfig, subjects: set[str]) -> dict[str, list[Sprite]]:
    root = Path(cfg.path)
    out: dict[str, list[Sprite]] = {}
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["subject"] not in subjects:
            continue
        out.setdefault(r["subject"], []).append(
            Sprite(
                int(r["row_id"]), r["subject"], r.get("caption", ""), load_rgba(root / r["path"])
            )
        )
    return out


def _load_atlas(cfg: SourceConfig, subjects: set[str]) -> dict[str, list[Sprite]]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    shards = list(cfg.shards)
    if not shards:
        files = HfApi().list_repo_files(cfg.repo_id, repo_type="dataset", revision=cfg.revision)
        shards = sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))

    out: dict[str, list[Sprite]] = {}
    wanted = list(subjects)
    for shard in shards:
        local = hf_hub_download(cfg.repo_id, shard, repo_type="dataset", revision=cfg.revision)
        table = pq.read_table(local, columns=["image", "caption", "subject", "row_id"])
        table = table.filter(pc.is_in(table["subject"], value_set=_arrow_strings(wanted)))
        for r in table.to_pylist():
            with Image.open(io.BytesIO(r["image"]["bytes"])) as im:
                rgba = np.asarray(im.convert("RGBA")).copy()
            out.setdefault(r["subject"], []).append(
                Sprite(int(r["row_id"]), r["subject"], r["caption"], rgba)
            )
    return out


def _arrow_strings(values: list[str]):
    import pyarrow as pa

    return pa.array(values, type=pa.string())
