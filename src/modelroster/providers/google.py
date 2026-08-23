"""
Google (Gemini API): availability from the OpenAI-compatibility shim
`GET https://generativelanguage.googleapis.com/v1beta/openai/models`
(ids are "models/<name>", exactly as the shim reports them) enriched from the
native `GET /v1beta/models` resource, which is official per-model metadata:
displayName, description, inputTokenLimit, outputTokenLimit,
supportedGenerationMethods. Both use the same "models/<name>" id, so the join
is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import ModelRecord, prov
from .openai_compat import OpenAICompatProvider

NATIVE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
METHOD_TO_ENDPOINT = {
    "generateContent": "generate_content", "streamGenerateContent": "stream_generate_content",
    "countTokens": "count_tokens", "embedContent": "embed_content", "batchEmbedContents": "batch_embed_contents",
    "generateAnswer": "generate_answer", "predict": "predict", "predictLongRunning": "predict_long_running",
    "bidiGenerateContent": "bidi_generate_content", "createCachedContent": "create_cached_content",
    "batchGenerateContent": "batch_generate_content", "countTextTokens": "count_text_tokens",
    "generateText": "generate_text", "embedText": "embed_text", "createTunedModel": "create_tuned_model",
}


class GoogleProvider(OpenAICompatProvider):
    name = "google"
    auth = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    describe = "OpenAI-compat shim /v1beta/openai/models + native /v1beta/models metadata"

    def fixtures(self, root: Path):
        return {self.models_url: root / "listings" / "google_models.json",
                NATIVE_MODELS_URL + "?pageSize=1000": root / "listings" / "google_native_models.json"}

    def enrich_context(self, raw, http) -> dict:
        ctx: dict[str, Any] = {"_sources": {"native": NATIVE_MODELS_URL}, "_warnings": [], "by_name": {}}
        headers = {"x-goog-api-key": self.require_key()}
        token = None
        for _ in range(20):
            params = {"pageSize": 1000}
            if token:
                params["pageToken"] = token
            try:
                data = http.get_json(NATIVE_MODELS_URL, headers=headers, params=params)
            except Exception as exc:
                ctx["_warnings"].append("native models endpoint unavailable: %s" % exc)
                break
            for m in data.get("models") or []:
                if isinstance(m, dict) and m.get("name"):
                    ctx["by_name"][m["name"]] = m
            token = data.get("nextPageToken")
            if not token:
                break
        return ctx

    def enrich_record(self, rec: ModelRecord, raw: dict, http, context: dict) -> None:
        m = context.get("by_name", {}).get(rec.model_id)
        rec.endpoints["chat_completions"] = True
        rec.provenance["endpoints.chat_completions"] = prov("api:/v1beta/openai/models", "listed by the OpenAI-compatibility shim")
        if not m:
            rec.warn("not present in the native /v1beta/models listing; no capability metadata")
            return
        src = "api:/v1beta/models"
        rec.sources["native"] = NATIVE_MODELS_URL
        rec.display_name = m.get("displayName")
        rec.description = m.get("description")
        rec.family = rec.model_id
        rec.relationship = "canonical"
        if isinstance(m.get("inputTokenLimit"), int):
            rec.context_window = rec.max_input_tokens = m["inputTokenLimit"]
            rec.provenance["context_window"] = prov(src, "inputTokenLimit")
            rec.provenance["max_input_tokens"] = rec.provenance["context_window"]
        if isinstance(m.get("outputTokenLimit"), int):
            rec.max_output_tokens = m["outputTokenLimit"]
            rec.provenance["max_output_tokens"] = prov(src, "outputTokenLimit")
        methods = m.get("supportedGenerationMethods")
        if isinstance(methods, list):
            for meth, key in METHOD_TO_ENDPOINT.items():
                rec.endpoints[key] = meth in methods
            for meth in methods:
                if meth not in METHOD_TO_ENDPOINT:
                    key = "".join("_" + ch.lower() if ch.isupper() else ch for ch in meth)
                    rec.endpoints[key] = True
                    rec.warn("unrecognised generation method %r (stored as endpoint %r)" % (meth, key))
            rec.provenance["endpoints"] = prov(src, "supportedGenerationMethods")
            if "generateContent" in methods:
                rec.modalities["text"] = {"input": True, "output": True}
                rec.capabilities.streaming = "streamGenerateContent" in methods
                rec.provenance["streaming"] = prov(src, "supportedGenerationMethods contains streamGenerateContent")
        if isinstance(m.get("thinking"), bool):
            rec.capabilities.reasoning = rec.capabilities.extended_thinking = m["thinking"]
            rec.provenance["reasoning"] = prov(src, "thinking")
            rec.provenance["extended_thinking"] = rec.provenance["reasoning"]
        rec.raw["native"] = {k: v for k, v in m.items() if k not in ("description",)}
