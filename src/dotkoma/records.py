"""File contracts shared by prepare / generate / finalize.

All records are JSON Lines. Writers are atomic (write temp file, then rename)
so an interrupted process never leaves a half-written file that looks done.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def derive_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary parts."""
    return int(stable_hash(list(parts), 8), 16) & 0x7FFFFFFF


def job_id(generation_inputs: dict[str, Any]) -> str:
    """ID of a generation job.

    ``generation_inputs`` must contain everything that changes the raw video
    (source sprite, action prompt, seed, model and inference params, reference
    layout). Same inputs -> same id; any change -> new id.
    """
    return stable_hash(generation_inputs, 16)


# ---------------------------------------------------------------- jsonl io


def read_jsonl(path: str | Path, *, unterminated_tail: str = "drop") -> list[dict[str, Any]]:
    """Read a JSON Lines file. A record is complete only when it ends with a newline.

    ``unterminated_tail`` decides what happens to a final line without a newline:
      "drop"   machine-written logs (append_jsonl): it is an interrupted append,
               ignored here and cut off by the next append. Same rule on both sides.
      "parse"  hand-edited files: editors often omit the final newline.
    Any complete line that is not valid UTF-8 JSON is real corruption and raises.
    """
    path = Path(path)
    if not path.exists():
        return []
    *complete, tail = path.read_bytes().split(b"\n")
    if tail.strip() and unterminated_tail == "parse":
        complete.append(tail)
    elif unterminated_tail not in ("drop", "parse"):
        raise ValueError(f"unknown unterminated_tail {unterminated_tail!r}")
    out = []
    for lineno, raw in enumerate(complete, 1):
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{lineno}: corrupt jsonl line") from e
    return out


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(canonical_json(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one record and fsync. Used for the append-only results log.

    A crash mid-append can leave a torn last line. It is cut off before writing,
    so the next record never lands after a corrupt line in the middle of the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _drop_torn_tail(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical_json(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _drop_torn_tail(path: Path) -> None:
    if not path.exists():
        return
    with path.open("rb+") as f:
        size = f.seek(0, os.SEEK_END)
        if size == 0:
            return
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return
        # scan back to the last complete line
        pos, chunk = size, 4096
        while pos > 0:
            start = max(0, pos - chunk)
            f.seek(start)
            nl = f.read(pos - start).rfind(b"\n")
            if nl != -1:
                f.truncate(start + nl + 1)
                break
            pos = start
        else:
            f.truncate(0)
        f.flush()
        os.fsync(f.fileno())


def copy_atomic(src: str | Path, dst: str | Path) -> None:
    """Copy via a temp file in the destination directory, then rename over ``dst``.

    A crash mid-copy leaves the previous ``dst`` intact (e.g. a Drive backup).
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def latest_by_job(results: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Last record per job_id in an append-only results log."""
    out: dict[str, dict[str, Any]] = {}
    for r in results:
        out[r["job_id"]] = r
    return out


def iter_jobs(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)
