"""
Provider plugin interface.

A provider knows how to (1) list the model ids its API currently exposes and
(2) enrich those ids with capability metadata from official sources. Both
steps receive the shared HTTP layer so they can be replayed offline.

Third parties register providers either by calling
`modelroster.providers.register(provider)` or by declaring an entry point in
the `modelroster.providers` group whose value is a zero-argument factory (a
class or function returning a Provider instance).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..schema import ModelRecord


class ProviderError(Exception):
    """Raised by a provider when its listing/enrichment cannot proceed."""


class SkipProvider(ProviderError):
    """Raised when a provider cannot run in this environment (no key, local
    daemon not running). The updater reports it as skipped, not failed."""


class MissingCredentials(SkipProvider):
    """Raised when a provider needs an API key that is not set."""


@dataclass
class ProviderResult:
    """What `Provider.enrich` returns."""
    records: list[ModelRecord]
    model_order: list[str] = field(default_factory=list)     # provider's own ordering
    sources: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)      # provider-specific validation state

    def __post_init__(self) -> None:
        if not self.model_order:
            self.model_order = [r.model_id for r in self.records]


@runtime_checkable
class Provider(Protocol):
    name: str
    auth: tuple[str, ...]

    def list_models(self, http: Any) -> list[dict]: ...

    def enrich(self, raw: list[dict], http: Any) -> ProviderResult: ...

    def fixtures(self, root: Path) -> dict[str, str | Path] | None: ...


class BaseProvider:
    """
    Convenience base: credential lookup, default fixtures()/validate().
    Subclasses set `name`, `auth`, and implement `list_models` / `enrich`.
    """

    name: str = ""
    auth: tuple[str, ...] = ()
    #: Human description of the sources, for `modelroster providers`.
    describe: str = ""

    def api_key(self) -> str | None:
        for var in self.auth:
            val = os.getenv(var)
            if val:
                return val
        return None

    def require_key(self) -> str:
        key = self.api_key()
        if not key:
            raise MissingCredentials("%s: none of %s is set" % (self.name, ", ".join(self.auth) or "(no env var)"))
        return key

    def fixtures(self, root: Path) -> dict[str, str | Path] | None:
        """{url: fixture path} so the whole pipeline can be replayed offline
        from captured responses under `root` (tests/fixtures by default)."""
        return None

    def validate(self, current: dict, previous: dict | None) -> tuple[list[str], list[str]]:
        """Provider-specific gates beyond the generic ones. (errors, warnings)."""
        return [], []

    def __repr__(self) -> str:
        return "<Provider %s>" % self.name


def fixture_dir() -> Path | None:
    """tests/fixtures of a source checkout, if this package is imported from one."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "tests" / "fixtures"
        if cand.exists():
            return cand
    return None
