"""
Consumer API over the shipped (or locally refreshed) registry data.

    import modelroster
    r = modelroster.load()                      # every provider with data
    r.models(tool_calling=True, reasoning=True) # documented-supported only
    r.models(provider="openai", streaming=True, strict=False)
    r.get("gpt-5.4"); r.get("anthropic/claude-opus-5"); r.get(ModelRef(...))
    r.resolve("gpt-4o-2024-08-06")              # -> the family's canonical record

Filters are tri-state aware: `capability=True` matches records whose source
documents support; `capability=False` matches documented non-support;
`strict=False` (the default) means None ("not documented") never matches a
True/False filter — pass `unknown_ok=True` to let None pass as well.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from .ref import ModelRef, RetiredModelError, UnknownModelError
from .schema import CAPABILITY_FIELDS, MODALITIES, ModelRecord
from .store import available_providers, read_json, registry_path

_FILTER_MODALITY = {"%s_%s" % (m, d): (m, d) for m in MODALITIES for d in ("input", "output")}


@dataclass
class ProviderInfo:
    provider: str
    retrieved_at: str | None
    parser_version: str | None
    schema_version: int | None
    models: int
    stats: dict


class Registry:
    def __init__(self, envelopes: dict[str, dict]):
        self._env = envelopes
        self._records: dict[str, dict[str, ModelRecord]] = {}
        self._order: dict[str, list[str]] = {}
        for prov, env in envelopes.items():
            recs = {}
            for mid, d in (env.get("models") or {}).items():
                recs[mid] = ModelRecord.from_dict(d)
            self._records[prov] = recs
            order = [m for m in env.get("model_order") or [] if m in recs]
            order += [m for m in recs if m not in set(order)]
            self._order[prov] = order

    # ── introspection ──
    def providers(self) -> list[str]:
        return sorted(self._records)

    def info(self, provider: str | None = None) -> dict[str, ProviderInfo] | ProviderInfo:
        def one(p):
            env = self._env[p]
            return ProviderInfo(p, env.get("retrieved_at"), env.get("parser_version"),
                                env.get("schema_version"), len(self._records[p]), env.get("stats") or {})
        if provider:
            return one(provider)
        return {p: one(p) for p in self.providers()}

    def envelope(self, provider: str) -> dict:
        return self._env[provider]

    def __len__(self) -> int:
        return sum(len(r) for r in self._records.values())

    def __iter__(self):
        for p in self.providers():
            for mid in self._order[p]:
                yield self._records[p][mid]

    def __contains__(self, item) -> bool:
        return self.get(item) is not None

    # ── lookup ──
    def _lookup(self, model_id: str, provider: str | None = None) -> list[ModelRecord]:
        provs = [provider] if provider else self.providers()
        return [self._records[p][model_id] for p in provs if p in self._records and model_id in self._records[p]]

    def get(self, ref: str | ModelRef, provider: str | None = None) -> ModelRecord | None:
        """Exact-id lookup. 'provider/model_id' strings and ModelRefs are honoured."""
        if isinstance(ref, ModelRef):
            provider, model_id = ref.provider, ref.model_id
        else:
            model_id = ref
            if provider is None and "/" in ref:
                head, _, tail = ref.partition("/")
                if head in self._records and tail in self._records[head]:
                    return self._records[head][tail]
        hits = self._lookup(model_id, provider)
        if len(hits) == 1:
            return hits[0]
        if not hits:
            return None
        # same id in several providers: prefer the one that owns it canonically, else first by provider name
        for h in hits:
            if h.relationship == "canonical":
                return h
        return hits[0]

    def find_providers(self, model_id: str) -> list[str]:
        return [p for p in self.providers() if model_id in self._records[p]]

    def resolve(self, ref: str | ModelRef, provider: str | None = None) -> ModelRecord | None:
        """Follow aliases/snapshots/fine-tune inheritance to the canonical record (or the record itself)."""
        rec = self.get(ref, provider)
        if rec is None:
            return None
        if rec.family and rec.family != rec.model_id:
            fam = self.get(rec.family, rec.provider)
            if fam is not None:
                return fam
        return rec

    def ref(self, ref: str | ModelRef) -> ModelRef:
        """Parse and validate an identifier against this registry."""
        return ModelRef.parse(ref, registry=self).validate(self)

    # ── filtering ──
    def models(self, provider: str | None = None, *, strict: bool = False, unknown_ok: bool = False,
               tier: str | None = "verified", include_retired: bool = True,
               relationship: str | Iterable[str] | None = None,
               endpoint: str | None = None, builtin_tool: str | None = None,
               **filters: Any) -> list[ModelRecord]:
        """
        Return records matching every filter. Keyword filters are capability
        names (`tool_calling=True`, `reasoning=False`), modality flags
        (`image_input=True`), or any `capabilities.extra` key. A True/False
        filter matches only a documented value unless `unknown_ok=True`, in
        which case None also passes. `strict=True` additionally requires the
        record itself to have no warnings and a canonical/snapshot/alias
        relationship (i.e. not an unmatched listing id).
        """
        provs = [provider] if provider else self.providers()
        if relationship is not None and isinstance(relationship, str):
            relationship = (relationship,)
        out = []
        for p in provs:
            for mid in self._order.get(p, []):
                rec = self._records[p][mid]
                if tier and rec.tier != tier:
                    continue
                if not include_retired and rec.deprecated is True:
                    continue
                if relationship is not None and rec.relationship not in relationship:
                    continue
                if strict and (rec.relationship == "unknown"):
                    continue
                if endpoint is not None and not _match(rec.endpoints.get(endpoint), True, unknown_ok):
                    continue
                if builtin_tool is not None:
                    tools = rec.builtin_tools or {}
                    if not _match(tools.get(builtin_tool) if rec.builtin_tools is not None else None, True, unknown_ok):
                        continue
                ok = True
                for name, want in filters.items():
                    if name in _FILTER_MODALITY:
                        m, d = _FILTER_MODALITY[name]
                        have = rec.modality(m, d)
                    else:
                        have = rec.capabilities.get(name)
                    if not _match(have, want, unknown_ok):
                        ok = False
                        break
                if ok:
                    out.append(rec)
        return out

    def ids(self, provider: str | None = None, **kw: Any) -> list[str]:
        return [r.model_id for r in self.models(provider, **kw)]

    def refs(self, provider: str | None = None, **kw: Any) -> list[ModelRef]:
        return [ModelRef(r.provider, r.model_id) for r in self.models(provider, **kw)]


def _match(have: Any, want: Any, unknown_ok: bool) -> bool:
    if want is None:
        return have is None
    if isinstance(want, (list, tuple, set)):
        return have in want or (unknown_ok and have is None)
    if have is None:
        return unknown_ok
    return have == want


_cache: dict[tuple, Registry] = {}


def load(provider: str | Iterable[str] | None = None, data_dir: str | os.PathLike | None = None,
         *, force: bool = False) -> Registry:
    """Load the registry for one, several, or (default) every provider with data."""
    if isinstance(provider, str):
        provs = [provider]
    elif provider is None:
        provs = available_providers(data_dir)
    else:
        provs = list(provider)
    key = (tuple(provs), str(data_dir) if data_dir else None)
    if key in _cache and not force:
        return _cache[key]
    envelopes = {}
    for p in provs:
        env = read_json(registry_path(p, data_dir))
        if env is None:
            raise FileNotFoundError("no registry data for provider %r (looked in %s)" % (p, registry_path(p, data_dir).parent))
        envelopes[p] = env
    reg = Registry(envelopes)
    _cache[key] = reg
    return reg


def clear_cache() -> None:
    _cache.clear()


__all__ = ["Registry", "ProviderInfo", "load", "clear_cache", "ModelRef", "UnknownModelError", "RetiredModelError"]
