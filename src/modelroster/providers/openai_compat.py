"""
Base class for providers that expose an OpenAI-compatible `GET /v1/models`
listing: `{"object": "list", "data": [{"id": ..., "created": ..., "owned_by": ...}]}`.

Such listings answer *availability* only. Every capability stays None unless
the subclass overrides `enrich_record` with data from an official per-model
metadata source (Mistral's `capabilities` object, Google's native model
resource, ...). Adding a new provider of this kind is a subclass with
`name`, `base_url`, `auth`, and (optionally) `enrich_record` — no core edits.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..schema import ModelRecord, prov, utc_now_iso
from .base import BaseProvider, ProviderResult


class OpenAICompatProvider(BaseProvider):
    base_url: str = ""
    models_path: str = "/models"
    #: header used to send the key; "bearer" -> Authorization: Bearer <key>
    auth_style: str = "bearer"
    auth_header: str = "Authorization"
    #: endpoint keys that the listing implies (e.g. an OpenAI-compatible chat
    #: endpoint). Left empty by default: a listing does not prove support.
    implied_endpoints: dict[str, bool] = {}

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + self.models_path

    def headers(self) -> dict[str, str]:
        if not self.auth:
            return {}
        key = self.require_key()
        if self.auth_style == "bearer":
            return {"Authorization": "Bearer " + key}
        return {self.auth_header: key}

    def list_models(self, http: Any) -> list[dict]:
        data = http.get_json(self.models_url, headers=self.headers())
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("%s: unexpected listing shape from %s" % (self.name, self.models_url))
        out = [m for m in items if isinstance(m, dict) and m.get("id")]
        out.sort(key=lambda m: m["id"])
        return out

    def fixtures(self, root: Path) -> dict[str, str | Path] | None:
        return {self.models_url: root / "listings" / ("%s_models.json" % self.name)}

    def base_record(self, raw: dict, retrieved_at: str) -> ModelRecord:
        rec = ModelRecord(provider=self.name, model_id=raw["id"], relationship="unknown", retrieved_at=retrieved_at,
                          sources={"listing": self.models_url, "documentation": None})
        rec.raw = {k: raw.get(k) for k in ("created", "owned_by", "object") if k in raw}
        created = raw.get("created")
        if isinstance(created, (int, float)) and created > 10_000_000:
            rec.released = time.strftime("%Y-%m-%d", time.gmtime(created))
            rec.provenance["released"] = prov("api:" + self.models_path, "created timestamp")
        for k, v in self.implied_endpoints.items():
            rec.endpoints[k] = v
            rec.provenance["endpoints." + k] = prov("api:" + self.models_path, "listed by the OpenAI-compatible models endpoint")
        return rec

    def enrich_record(self, rec: ModelRecord, raw: dict, http: Any, context: dict) -> None:
        """Override to add official per-model metadata. `context` is shared across records."""

    def enrich_context(self, raw: list[dict], http: Any) -> dict:
        """Override to fetch provider-wide metadata once (e.g. a richer second endpoint)."""
        return {}

    def enrich(self, raw: list[dict], http: Any) -> ProviderResult:
        now = utc_now_iso()
        context = self.enrich_context(raw, http)
        records = []
        for m in raw:
            rec = self.base_record(m, now)
            self.enrich_record(rec, m, http, context)
            records.append(rec)
        return ProviderResult(records=records, sources={"listing": self.models_url, **context.get("_sources", {})},
                              stats={"api_models": len(records)}, warnings=list(context.get("_warnings", [])))


def _set(rec: ModelRecord, field: str, value: Any, section: str, evidence: str) -> None:
    """Set a capability (or top-level field) with provenance, leaving None alone."""
    if value is None:
        return
    if hasattr(rec.capabilities, field) and field != "extra":
        setattr(rec.capabilities, field, value)
    elif hasattr(rec, field):
        setattr(rec, field, value)
    else:
        rec.capabilities.extra[field] = value
    rec.provenance[field] = prov(section, evidence)
