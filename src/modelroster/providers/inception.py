"""
Inception Labs: `GET https://api.inceptionlabs.ai/v1/models` is publicly
listable and richer than plain OpenAI-compat: modalities, context_length,
max_output_length, supported_features (tools, json_mode, structured_outputs),
supported_sampling_parameters, and pricing per token (converted to USD per
million tokens).
"""

from __future__ import annotations

from ..schema import MODALITIES, ModelRecord, prov
from .openai_compat import OpenAICompatProvider

KNOWN_FEATURES = ("tools", "json_mode", "structured_outputs", "streaming", "reasoning", "vision")


class InceptionProvider(OpenAICompatProvider):
    name = "inception"
    auth = ("INCEPTION_API_KEY",)
    base_url = "https://api.inceptionlabs.ai/v1"
    describe = "GET /v1/models (public; modalities, limits, supported_features, pricing)"

    def headers(self):
        key = self.api_key()
        return {"Authorization": "Bearer " + key} if key else {}

    def enrich_record(self, rec: ModelRecord, raw: dict, http, context: dict) -> None:
        src = "api:/v1/models"
        rec.display_name = raw.get("name")
        rec.description = raw.get("description")
        rec.family = rec.model_id
        rec.relationship = "canonical"
        for direction, key in (("input", "input_modalities"), ("output", "output_modalities")):
            mods = raw.get(key)
            if isinstance(mods, list):
                for m in MODALITIES:
                    rec.modalities[m][direction] = m in mods
                rec.provenance["modalities." + direction] = prov(src, key)
        if isinstance(raw.get("context_length"), int):
            rec.context_window = raw["context_length"]
            rec.provenance["context_window"] = prov(src, "context_length")
        if isinstance(raw.get("max_output_length"), int):
            rec.max_output_tokens = raw["max_output_length"]
            rec.provenance["max_output_tokens"] = prov(src, "max_output_length")
        feats = raw.get("supported_features")
        if isinstance(feats, list):
            rec.capabilities.tool_calling = "tools" in feats
            rec.capabilities.structured_outputs = "structured_outputs" in feats
            for k in ("tool_calling", "structured_outputs"):
                rec.provenance[k] = prov(src, "supported_features list")
            for f in feats:
                rec.capabilities.extra["inception." + f] = True
                if f not in KNOWN_FEATURES:
                    rec.warn("unrecognised supported_features value %r (kept)" % f)
            for f in KNOWN_FEATURES:
                rec.capabilities.extra.setdefault("inception." + f, False)
        pricing = raw.get("pricing")
        if isinstance(pricing, dict):
            def per_m(k):
                try:
                    return round(float(pricing[k]) * 1_000_000, 6) if pricing.get(k) is not None else None
                except (TypeError, ValueError):
                    return None
            rec.pricing = {"input": per_m("prompt"), "output": per_m("completion"),
                           "cached_input": per_m("input_cache_reads"), "currency": "USD", "per": "1M tokens"}
            rec.provenance["pricing"] = prov(src, "pricing per token x 1e6")
        rec.endpoints["chat_completions"] = True
        rec.provenance["endpoints.chat_completions"] = prov(src, "listed by the OpenAI-compatible models endpoint")
        rec.raw["supported_sampling_parameters"] = raw.get("supported_sampling_parameters")
