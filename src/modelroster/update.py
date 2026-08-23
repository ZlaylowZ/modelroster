"""
Refresh pipeline, one provider at a time:

    list_models -> enrich -> validate(current, previous) -> diff ->
    atomic write (<provider>.json + .previous.json) -> drift report
    (<provider>.drift.json + printed)

Providers are independent: one failing never blocks the others. Each run
returns an `UpdateResult` whose `code` follows the CLI contract
(0 ok / 2 validation refused / 3 fetch failure / 4 usage); `run_many`
reports the worst code. A provider whose API key is absent is *skipped*
(code 0, `skipped=True`), not failed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import providers as provider_registry
from .http import FetchError, Fetcher, FixtureFetcher
from .providers.base import ProviderError, ProviderResult, SkipProvider
from .schema import PARSER_VERSION, SCHEMA_VERSION, utc_now_iso
from .store import cache_dir, drift_path, previous_path, read_json, registry_path, write_json_atomic
from .validate import diff, format_drift, validate

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_FETCH = 3
EXIT_USAGE = 4


@dataclass
class UpdateResult:
    provider: str
    code: int
    registry: dict | None = None
    drift: dict | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    written: bool = False
    path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.code == EXIT_OK

    def summary(self) -> str:
        if self.skipped:
            return "%s: skipped (%s)" % (self.provider, "; ".join(self.errors) or "no credentials")
        if self.code == EXIT_REFUSED:
            return "%s: REFUSED — %s" % (self.provider, "; ".join(self.errors))
        if self.code == EXIT_FETCH:
            return "%s: FETCH FAILED — %s" % (self.provider, "; ".join(self.errors))
        n = len((self.registry or {}).get("models") or {})
        return "%s: ok, %d model(s)%s" % (self.provider, n, "" if self.written else " (not written)")


def load_dotenv_if_available(path: str | os.PathLike | None = None) -> None:
    """Populate os.environ from a .env file without overriding existing values (optional dependency)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path, override=False) if path else load_dotenv(override=False)


def build_envelope(provider_name: str, result: ProviderResult, retrieved_at: str | None = None) -> dict:
    models = {}
    for rec in result.records:
        if rec.model_id in models:
            rec.warn("duplicate model id in listing; later occurrence kept")
        models[rec.model_id] = rec.to_dict()
    return {
        "provider": provider_name,
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "retrieved_at": retrieved_at or utc_now_iso(),
        "sources": result.sources,
        "model_order": [m for m in result.model_order if m in models],
        "models": models,
        "stats": dict(result.stats, models=len(models)),
        "warnings": list(result.warnings),
        "extra": result.extra,
    }


def make_http(provider: Any, *, data_dir=None, offline=False, no_cache=False, fixtures_root=None,
              concurrency=6):
    """The HTTP seam: fixtures replay when a fixtures root is given, else the shared Fetcher."""
    if fixtures_root is not None:
        pages = provider.fixtures(Path(fixtures_root)) if hasattr(provider, "fixtures") else None
        if not pages:
            raise ProviderError("%s: provider ships no fixtures" % provider.name)
        return FixtureFetcher(pages)
    return Fetcher(cache_dir(provider.name, data_dir), offline=offline, no_cache=no_cache,
                   concurrency=concurrency)


def run_provider(provider: Any, *, data_dir=None, offline: bool = False, no_cache: bool = False,
                 dry_run: bool = False, fixtures_root=None, http=None, quiet: bool = False,
                 out=None) -> UpdateResult:
    out = out or sys.stdout
    if isinstance(provider, str):
        try:
            provider = provider_registry.get(provider)
        except KeyError as exc:
            return UpdateResult(str(provider), EXIT_USAGE, errors=[str(exc)])
    name = provider.name
    path = registry_path(name, data_dir)
    previous = read_json(path)
    res = UpdateResult(name, EXIT_OK, path=path)

    own_http = http is None
    try:
        if own_http:
            try:
                http = make_http(provider, data_dir=data_dir, offline=offline, no_cache=no_cache,
                                 fixtures_root=fixtures_root)
            except ProviderError as exc:
                res.code, res.errors = EXIT_USAGE, [str(exc)]
                return res
        try:
            raw = provider.list_models(http)
            result = provider.enrich(raw, http)
        except SkipProvider as exc:
            res.skipped = True
            res.errors = [str(exc)]
            if not quiet:
                print("%s: skipped — %s" % (name, exc), file=out)
            return res
        except FetchError as exc:
            res.code, res.errors = EXIT_FETCH, ["fetch failed: %s" % exc]
            print("ERROR: %s: %s" % (name, exc), file=sys.stderr)
            return res
        except ProviderError as exc:
            res.code, res.errors = EXIT_REFUSED, [str(exc)]
            print("ERROR: %s: %s" % (name, exc), file=sys.stderr)
            return res
        except Exception as exc:  # a parser crash is a refusal, never a half-written file
            res.code, res.errors = EXIT_REFUSED, ["%s: %s" % (type(exc).__name__, exc)]
            print("ERROR: %s: %s: %s" % (name, type(exc).__name__, exc), file=sys.stderr)
            return res
    finally:
        if own_http and http is not None:
            http.close()

    current = build_envelope(name, result)
    errors, warnings = validate(current, previous, provider)
    current["warnings"] = warnings
    res.registry, res.warnings = current, warnings
    if errors:
        res.code, res.errors = EXIT_REFUSED, errors
        for e in errors:
            print("ERROR: %s: %s" % (name, e), file=sys.stderr)
        print("%s: registry NOT written; previous data preserved at %s" % (name, path), file=sys.stderr)
        return res

    drift_report = diff(previous, current)
    res.drift = drift_report
    if not dry_run:
        write_json_atomic(path, current)
        write_json_atomic(drift_path(name, data_dir), drift_report, keep_previous=False)
        res.written = True
    if not quiet:
        print("", file=out)
        print("%s registry %s" % (name, "(dry run — not written)" if dry_run else "updated -> %s" % path), file=out)
        print(format_drift(drift_report), file=out)
    return res


def run_many(names: list[str] | None = None, **kw: Any) -> list[UpdateResult]:
    names = names or provider_registry.names()
    return [run_provider(n, **kw) for n in names]


def worst_code(results: list[UpdateResult]) -> int:
    return max([r.code for r in results] + [EXIT_OK])


def refresh(providers: list[str] | None = None, *, data_dir=None, offline: bool = False,
            dry_run: bool = False, quiet: bool = True) -> dict[str, dict]:
    """
    Convenience for long-running agents: refresh the given providers (default:
    all) and return {provider: drift-or-status}. Never raises for a single
    provider's failure; the status dict carries `code` and `errors`.
    """
    load_dotenv_if_available()
    out = {}
    for r in run_many(providers, data_dir=data_dir, offline=offline, dry_run=dry_run, quiet=quiet):
        out[r.provider] = {"code": r.code, "skipped": r.skipped, "written": r.written,
                           "errors": r.errors, "drift": r.drift}
    return out


def previous_registry(provider: str, data_dir=None) -> dict | None:
    return read_json(previous_path(provider, data_dir))
