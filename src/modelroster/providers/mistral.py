"""
Mistral: `GET https://api.mistral.ai/v1/models` is OpenAI-compatible *and*
carries a per-model `capabilities` object (completion_chat, function_calling,
vision, fine_tuning, ...), `max_context_length`, `aliases`, and `deprecation`.
"""

from __future__ import annotations

from ..schema import ModelRecord, prov
from .openai_compat import OpenAICompatProvider

KNOWN_CAPS = ("completion_chat", "completion_fim", "function_calling", "fine_tuning", "vision",
              "classification", "ocr", "audio", "moderation")


class MistralProvider(OpenAICompatProvider):
    name = "mistral"
    auth = ("MISTRAL_API_KEY",)
    base_url = "https://api.mistral.ai/v1"
    describe = "GET /v1/models with the per-model `capabilities` object"

    def enrich_record(self, rec: ModelRecord, raw: dict, http, context: dict) -> None:
        src = "api:/v1/models"
        rec.display_name = raw.get("name") or None
        rec.description = raw.get("description") or None
        caps = raw.get("capabilities")
        if isinstance(caps, dict):
            def t(k):
                v = caps.get(k)
                return None if v is None else bool(v)
            rec.capabilities.tool_calling = t("function_calling")
            rec.capabilities.fine_tuning = t("fine_tuning")
            if rec.capabilities.tool_calling is not None:
                rec.provenance["tool_calling"] = prov(src, "capabilities.function_calling")
            if rec.capabilities.fine_tuning is not None:
                rec.provenance["fine_tuning"] = prov(src, "capabilities.fine_tuning")
            if t("completion_chat") is not None:
                rec.endpoints["chat_completions"] = t("completion_chat")
                rec.provenance["endpoints.chat_completions"] = prov(src, "capabilities.completion_chat")
            if t("completion_fim") is not None:
                rec.endpoints["fim_completions"] = t("completion_fim")
            if t("vision") is not None:
                rec.modalities["image"]["input"] = t("vision")
                rec.provenance["modalities.image.input"] = prov(src, "capabilities.vision")
            if t("completion_chat"):
                rec.modalities["text"] = {"input": True, "output": True}
            for k, v in caps.items():
                if k not in KNOWN_CAPS:
                    rec.warn("unrecognised capability key %r (kept in extra)" % k)
                rec.capabilities.extra["mistral." + k] = None if v is None else bool(v)
        ctx = raw.get("max_context_length")
        if isinstance(ctx, int):
            rec.context_window = ctx
            rec.provenance["context_window"] = prov(src, "max_context_length")
        aliases = [a for a in (raw.get("aliases") or []) if a != rec.model_id]
        rec.aliases = aliases
        rec.family = rec.model_id
        rec.relationship = "canonical"
        dep = raw.get("deprecation")
        if dep:
            rec.deprecated = True
            rec.shutdown_date = str(dep)[:10]
            rec.provenance["shutdown_date"] = prov(src, "deprecation")
        elif "deprecation" in raw:
            rec.deprecated = False
        rec.raw["type"] = raw.get("type")
