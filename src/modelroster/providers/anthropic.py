"""
Anthropic: `GET https://api.anthropic.com/v1/models` (paginated) already
returns a structured `capabilities` object per model, so this adapter
normalises rather than parses. A capability block the API omits is None.

Provider-wide facts (see docs/DESIGN.md, "Provider-wide facts policy"):
Anthropic's tool-use and streaming documentation states that every Claude
model served by the Messages API supports developer-defined tools and
streaming. Those two capabilities are set to True for every listed model with
provenance `section="provider_docs"` so consumers can tell them apart from
per-model API facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import ModelRecord, prov, tri, utc_now_iso
from .base import BaseProvider, ProviderResult

MODELS_URL = "https://api.anthropic.com/v1/models"
API_VERSION = "2023-06-01"
TOOL_USE_DOCS = "https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview"
STREAMING_DOCS = "https://docs.anthropic.com/en/docs/build-with-claude/streaming"

EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
THINKING_TYPES = ("enabled", "adaptive")
CONTEXT_MANAGEMENT_KEYS = ("clear_tool_uses_20250919", "clear_thinking_20251015", "compact_20260112")
SIMPLE_CAPS = ("batch", "citations", "code_execution", "image_input", "pdf_input", "structured_outputs")


def _tri_supported(block: Any) -> bool | None:
    if not isinstance(block, dict) or "supported" not in block:
        return None
    return bool(block.get("supported"))


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    auth = ("ANTHROPIC_API_KEY",)
    describe = "GET /v1/models (paginated) with the per-model `capabilities` object"

    def __init__(self, page_size: int = 100, max_pages: int = 50):
        self.page_size = page_size
        self.max_pages = max_pages

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.require_key(), "anthropic-version": API_VERSION}

    def list_models(self, http: Any) -> list[dict]:
        out: list[dict] = []
        after_id = None
        headers = self._headers()
        for _ in range(self.max_pages):
            params: dict[str, Any] = {"limit": self.page_size}
            if after_id:
                params["after_id"] = after_id
            data = http.get_json(MODELS_URL, headers=headers, params=params)
            out.extend(data.get("data") or [])
            if not data.get("has_more"):
                break
            after_id = data.get("last_id")
            if not after_id:
                break
        return out

    def fixtures(self, root: Path) -> dict[str, str | Path] | None:
        return {MODELS_URL + "?limit=%d" % self.page_size: root / "listings" / "anthropic_models.json"}

    def enrich(self, raw: list[dict], http: Any) -> ProviderResult:
        now = utc_now_iso()
        records = [normalise_model(m, retrieved_at=now) for m in raw]
        return ProviderResult(
            records=records,
            model_order=[r.model_id for r in records],      # API order: newest first
            sources={"listing": MODELS_URL},
            stats={"api_models": len(records)},
        )

    def validate(self, current: dict, previous: dict | None) -> tuple[list[str], list[str]]:
        warnings = []
        for mid, rec in (current.get("models") or {}).items():
            if rec.get("max_input_tokens") is None or rec.get("max_output_tokens") is None:
                warnings.append("%s: missing max_input_tokens/max_tokens" % mid)
        return [], warnings


def normalise_model(raw: dict, retrieved_at: str | None = None) -> ModelRecord:
    caps = raw.get("capabilities") or {}
    rec = ModelRecord(provider="anthropic", model_id=raw.get("id"), display_name=raw.get("display_name"),
                      family=raw.get("id"), relationship="canonical", retrieved_at=retrieved_at)
    created = raw.get("created_at")
    rec.released = (created or "")[:10] or None
    rec.max_input_tokens = raw.get("max_input_tokens")
    rec.context_window = raw.get("max_input_tokens")
    rec.max_output_tokens = raw.get("max_tokens")
    rec.raw = {"created_at": created, "type": raw.get("type"), "capabilities": caps}
    rec.sources = {"listing": MODELS_URL, "documentation": None}
    api_prov = prov("api:/v1/models", "capabilities object")
    for k in ("max_input_tokens", "max_output_tokens", "context_window"):
        rec.provenance[k] = prov("api:/v1/models", "model object field")
    rec.provenance["context_window"]["note"] = "max_input_tokens reported as context window"

    c = rec.capabilities
    # effort
    effort_block = caps.get("effort")
    if isinstance(effort_block, dict):
        supported = _tri_supported(effort_block)
        levels = [lvl for lvl in EFFORT_ORDER if _tri_supported(effort_block.get(lvl))]
        for k in effort_block:
            if k != "supported" and k not in EFFORT_ORDER:
                rec.warn("unrecognised effort level key %r" % k)
                if _tri_supported(effort_block.get(k)):
                    levels.append(k)
        c.extra["effort_supported"] = supported
        c.reasoning_efforts = levels if supported else (None if supported is None else [])
        rec.provenance["reasoning_efforts"] = api_prov
    # thinking
    thinking_block = caps.get("thinking")
    if isinstance(thinking_block, dict):
        c.extended_thinking = _tri_supported(thinking_block)
        c.reasoning = c.extended_thinking
        types_block = thinking_block.get("types") or {}
        types = [t for t in THINKING_TYPES if _tri_supported(types_block.get(t))]
        for k in types_block:
            if k not in THINKING_TYPES:
                rec.warn("unrecognised thinking type %r" % k)
                if _tri_supported(types_block.get(k)):
                    types.append(k)
        c.extra["thinking_types"] = types
        rec.provenance["reasoning"] = prov("api:/v1/models", "capabilities.thinking.supported")
        rec.provenance["extended_thinking"] = rec.provenance["reasoning"]
    # context management
    cm_block = caps.get("context_management")
    if isinstance(cm_block, dict):
        c.extra["context_management"] = _tri_supported(cm_block)
        for k in CONTEXT_MANAGEMENT_KEYS:
            c.extra["context_management." + k] = _tri_supported(cm_block.get(k))
        for k in cm_block:
            if k != "supported" and k not in CONTEXT_MANAGEMENT_KEYS:
                c.extra["context_management." + k] = _tri_supported(cm_block.get(k))
                rec.warn("unrecognised context_management key %r" % k)
    # simple blocks
    simple = {k: _tri_supported(caps.get(k)) for k in SIMPLE_CAPS}
    c.batch, c.citations, c.code_execution = simple["batch"], simple["citations"], simple["code_execution"]
    c.pdf_input, c.structured_outputs = simple["pdf_input"], simple["structured_outputs"]
    for k in ("batch", "citations", "code_execution", "pdf_input", "structured_outputs"):
        if simple[k] is not None:
            rec.provenance[k] = prov("api:/v1/models", "capabilities.%s.supported" % k)
    if simple["image_input"] is not None:
        rec.modalities["image"]["input"] = simple["image_input"]
        rec.provenance["modalities.image.input"] = prov("api:/v1/models", "capabilities.image_input.supported")
    if caps:
        rec.modalities["text"] = {"input": True, "output": True}
        rec.provenance["modalities.text"] = prov("api:/v1/models", "Messages API model (text in/out)")
    for k in caps:
        if k not in SIMPLE_CAPS + ("effort", "thinking", "context_management"):
            rec.warn("unrecognised capability block %r" % k)
            c.extra[k] = tri(_tri_supported(caps.get(k)))
    if raw.get("type") == "model":
        c.tool_calling = True
        rec.provenance["tool_calling"] = prov("provider_docs", "provider-wide statement", url=TOOL_USE_DOCS)
        c.streaming = True
        rec.provenance["streaming"] = prov("provider_docs", "provider-wide statement", url=STREAMING_DOCS)
        rec.endpoints["messages"] = True
        rec.provenance["endpoints.messages"] = prov("api:/v1/models", "listed by the Messages API models endpoint")
    return rec
