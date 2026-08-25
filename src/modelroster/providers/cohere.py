"""
Cohere: `GET https://api.cohere.com/v1/models` (native listing) reports per
model the supported `endpoints` (chat, generate, embed, rerank, ...),
`context_length`, `supports_vision`, and a `features` list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import ModelRecord, prov, utc_now_iso
from .base import BaseProvider, ProviderResult

MODELS_URL = "https://api.cohere.com/v1/models"
KNOWN_FEATURES = ("tools", "strict_tools", "json_mode", "json_schema", "safety_modes", "citations",
                  "reasoning", "logprobs", "tool_choice", "tool_images", "vision")


class CohereProvider(BaseProvider):
    name = "cohere"
    auth = ("COHERE_API_KEY", "CO_API_KEY")
    describe = "GET /v1/models with endpoints, context_length, features"

    def list_models(self, http: Any) -> list[dict]:
        headers = {"Authorization": "Bearer " + self.require_key()}
        out, token = [], None
        for _ in range(20):
            params: dict[str, Any] = {"page_size": 1000}
            if token:
                params["page_token"] = token
            data = http.get_json(MODELS_URL, headers=headers, params=params)
            out.extend(m for m in (data.get("models") or []) if isinstance(m, dict) and m.get("name"))
            token = data.get("next_page_token")
            if not token:
                break
        out.sort(key=lambda m: m["name"])
        return out

    def fixtures(self, root: Path):
        return {MODELS_URL + "?page_size=1000": root / "listings" / "cohere_models.json"}

    def enrich(self, raw: list[dict], http: Any) -> ProviderResult:
        now = utc_now_iso()
        records = [self._record(m, now) for m in raw]
        return ProviderResult(records=records, sources={"listing": MODELS_URL}, stats={"api_models": len(records)})

    def _record(self, m: dict, now: str) -> ModelRecord:
        src = "api:/v1/models"
        rec = ModelRecord(provider="cohere", model_id=m["name"], family=m["name"], relationship="canonical",
                          retrieved_at=now, sources={"listing": MODELS_URL, "documentation": None})
        eps = m.get("endpoints")
        if isinstance(eps, list):
            for e in ("chat", "generate", "embed", "rerank", "classify", "summarize"):
                rec.endpoints[e] = e in eps
            for e in eps:
                if e not in rec.endpoints:
                    rec.endpoints[e] = True
            rec.provenance["endpoints"] = prov(src, "endpoints list")
            if "chat" in eps:
                rec.modalities["text"] = {"input": True, "output": True}
        if isinstance(m.get("context_length"), (int, float)):
            rec.context_window = int(m["context_length"])
            rec.provenance["context_window"] = prov(src, "context_length")
        if "supports_vision" in m:
            rec.modalities["image"]["input"] = bool(m["supports_vision"])
            rec.provenance["modalities.image.input"] = prov(src, "supports_vision")
        feats = m.get("features")
        if isinstance(feats, list):
            rec.capabilities.tool_calling = "tools" in feats
            rec.provenance["tool_calling"] = prov(src, "features list", evidence_detail="listed" if "tools" in feats else "absent_from_list")
            rec.capabilities.structured_outputs = "json_schema" in feats
            rec.provenance["structured_outputs"] = prov(src, "features list")
            if "citations" in feats:
                rec.capabilities.citations = True
                rec.provenance["citations"] = prov(src, "features list")
            if "reasoning" in feats:
                rec.capabilities.reasoning = True
                rec.provenance["reasoning"] = prov(src, "features list")
            for f in feats:
                rec.capabilities.extra["cohere." + f] = True
                if f not in KNOWN_FEATURES:
                    rec.warn("unrecognised feature %r (kept)" % f)
            for f in KNOWN_FEATURES:
                rec.capabilities.extra.setdefault("cohere." + f, False)
        if m.get("finetuned") is True:
            rec.relationship = "fine_tune_inherited"
        rec.capabilities.fine_tuning = None
        rec.raw = {k: m.get(k) for k in ("finetuned", "tokenizer_url", "default_endpoints") if k in m}
        return rec
