"""
On-disk layout and atomic persistence of per-provider registry files.

    <data_dir>/<provider>.json            current registry (shipped in the wheel)
    <data_dir>/<provider>.previous.json   last known-good copy (never shipped)
    <data_dir>/<provider>.drift.json      drift report of the last successful update
    <data_dir>/cache/<provider>/          HTTP cache (never shipped)

The default data dir is the package's own `data/` directory; override with
`data_dir=` / `--data-dir` / `MODELROSTER_DATA_DIR`.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
ENV_DATA_DIR = "MODELROSTER_DATA_DIR"


def data_dir(override: str | os.PathLike | None = None) -> Path:
    if override:
        return Path(override)
    env = os.getenv(ENV_DATA_DIR)
    if env:
        return Path(env)
    return PACKAGE_DATA_DIR


def registry_path(provider: str, base: str | os.PathLike | None = None) -> Path:
    return data_dir(base) / ("%s.json" % provider)


def previous_path(provider: str, base: str | os.PathLike | None = None) -> Path:
    return data_dir(base) / ("%s.previous.json" % provider)


def drift_path(provider: str, base: str | os.PathLike | None = None) -> Path:
    return data_dir(base) / ("%s.drift.json" % provider)


def cache_dir(provider: str, base: str | os.PathLike | None = None) -> Path:
    return data_dir(base) / "cache" / provider


def available_providers(base: str | os.PathLike | None = None) -> list[str]:
    d = data_dir(base)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json")
                  if not p.name.endswith((".previous.json", ".drift.json")))


def read_json(path: str | os.PathLike) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return None


def write_json_atomic(path: str | os.PathLike, data: Any, keep_previous: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if keep_previous and path.exists():
        shutil.copy2(path, path.with_name(path.stem + ".previous.json"))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False, ensure_ascii=False), "utf-8")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_text_atomic(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, "utf-8")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
