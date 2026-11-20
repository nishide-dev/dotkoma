import shutil
from pathlib import Path

import pytest

from dotkoma.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke_cfg(tmp_path):
    """Smoke config with its work dir redirected into a temp directory."""
    cfg_path = tmp_path / "configs" / "smoke.toml"
    cfg_path.parent.mkdir()
    shutil.copy(ROOT / "configs" / "smoke.toml", cfg_path)
    shutil.copytree(ROOT / "tests" / "fixtures", tmp_path / "tests" / "fixtures")
    return load_config(cfg_path, work_dir=tmp_path / "work")
