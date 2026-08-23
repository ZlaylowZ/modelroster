"""
Ollama (local): installed models from `GET http://localhost:11434/api/tags`,
enriched with `POST /api/show` per model — Ollama's own metadata, which
reports a `capabilities` list (completion, tools, vision, thinking, embedding)
and the model's context length. No generation request is made.

Set OLLAMA_HOST to point at another daemon. No API key is involved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..schema import ModelRecord, prov, utc_now_iso
from ..http import FetchError
from .base import BaseProvider, ProviderResult, SkipProvider

KNOWN_CAPABILITIES = ("completion", "tools", "vision", "thinking", "embedding", "insert")


def host() -> str:
    h = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not h.startswith("http"):
        h = "http://" + h
    return h


class OllamaProvider(BaseProvider):
    name = "ollama"
    auth = ()
    describe = "local daemon: GET /api/tags + POST /api/show (capabilities, context length)"

    def list_models(self, http: Any) -> list[dict]:
        try:
            data = http.get_json(host() + "/api/tags")
        except FetchError as exc:
            if exc.status is None:   # transport-level: daemon not running here
                raise SkipProvider("ollama: no daemon reachable at %s (%s)" % (host(), exc)) from exc
            raise
        out = [m for m in (data.get("models") or []) if isinstance(m, dict) and (m.get("model") or m.get("name"))]
        out.sort(key=lambda m: m.get("model") or m.get("name"))
        return out

    def fixtures(self, root: Path):
        base = host()
        pages: dict[str, str | Path] = {base + "/api/tags": root / "listings" / "ollama_tags.json"}
        show_dir = root / "listings" / "ollama_show"
        if show_dir.exists():
            for p in show_dir.glob("*.json"):
                name = p.stem.replace("__", "/").replace("--", ":")
                pages[base + '/api/show#{"model": "%s"}' % name] = p
        return pages

    def enrich(self, raw: list[dict], http: Any) -> ProviderResult:
        now = utc_now_iso()
        records, warnings = [], []
        for m in raw:
            name = m.get("model") or m.get("name")
            rec = ModelRecord(provider="ollama", model_id=name, family=name, relationship="canonical", retrieved_at=now,
                              sources={"listing": host() + "/api/tags", "documentation": None})
            details = m.get("details") or {}
            rec.raw = {"size": m.get("size"), "digest": m.get("digest"), "modified_at": m.get("modified_at"),
                       "details": details}
            rec.display_name = name
            try:
                show = http.post_json(host() + "/api/show", {"model": name})
            except Exception as exc:
                rec.warn("/api/show failed: %s" % exc)
                records.append(rec)
                continue
            self._apply_show(rec, show)
            records.append(rec)
        return ProviderResult(records=records, sources={"listing": host() + "/api/tags", "show": host() + "/api/show"},
                              stats={"installed_models": len(records)}, warnings=warnings)

    @staticmethod
    def _apply_show(rec: ModelRecord, show: dict) -> None:
        src = "api:/api/show"
        caps = show.get("capabilities")
        if isinstance(caps, list):
            rec.capabilities.tool_calling = "tools" in caps
            rec.capabilities.reasoning = "thinking" in caps
            rec.capabilities.extended_thinking = "thinking" in caps
            rec.modalities["image"]["input"] = "vision" in caps
            rec.modalities["text"]["input"] = True
            rec.modalities["text"]["output"] = "completion" in caps
            rec.endpoints["chat"] = "completion" in caps
            rec.endpoints["embeddings"] = "embedding" in caps
            for k in ("tool_calling", "reasoning", "extended_thinking", "modalities.image.input", "endpoints.chat", "endpoints.embeddings"):
                rec.provenance[k] = prov(src, "capabilities list")
            for c in caps:
                if c not in KNOWN_CAPABILITIES:
                    rec.warn("unrecognised capability %r (kept)" % c)
                rec.capabilities.extra["ollama." + c] = True
            for c in KNOWN_CAPABILITIES:
                rec.capabilities.extra.setdefault("ollama." + c, False)
        info = show.get("model_info") or {}
        for k, v in info.items():
            if k.endswith(".context_length") and isinstance(v, int):
                rec.context_window = v
                rec.provenance["context_window"] = prov(src, "model_info." + k)
                break
        details = show.get("details") or {}
        if details:
            rec.raw["details"] = details
        rec.raw["parameters"] = show.get("parameters")
        rec.sources["show"] = host() + "/api/show"
