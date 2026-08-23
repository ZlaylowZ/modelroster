"""
Discovery: candidate models from broad public registries.

Results are `ModelRecord`s with `tier="discovered"`, mostly-None capabilities,
and `relationship="unknown"`. They live in a separate tier (written to
`<data_dir>/discovered/<source>.json`) and never enter the verified catalog;
`Registry.models()` excludes them unless `tier="discovered"` or `tier=None`
is requested explicitly.

    from modelroster import discover
    discover.sources()                      # ["huggingface", "nvidia_nim", "ollama_library"]
    discover.run("huggingface", http, limit=100)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ..schema import ModelRecord

_SOURCES = {
    "huggingface": "modelroster.discover.huggingface:HuggingFaceSource",
    "ollama_library": "modelroster.discover.ollama_library:OllamaLibrarySource",
    "nvidia_nim": "modelroster.discover.nvidia_nim:NvidiaNimSource",
}


def sources() -> list[str]:
    return sorted(_SOURCES)


def get(name: str):
    if name not in _SOURCES:
        raise KeyError("unknown discovery source %r (known: %s)" % (name, ", ".join(sources())))
    mod, _, cls = _SOURCES[name].partition(":")
    return getattr(import_module(mod), cls)()


def run(name: str, http: Any, **kw: Any) -> list[ModelRecord]:
    return get(name).discover(http, **kw)


__all__ = ["sources", "get", "run"]
