"""
Mistral: `GET https://api.mistral.ai/v1/models` is OpenAI-compatible *and*
carries a per-model `capabilities` object, `max_context_length`, `aliases`,
and `deprecation`.

Capability keys observed live 2026-08-24: completion_chat, completion_fim,
function_calling, fine_tuning, vision, classification, ocr, audio,
audio_speech, audio_transcription, audio_transcription_realtime, moderation,
reasoning, unified_resources. Every key is also mirrored verbatim into
`capabilities.extra["mistral.<key>"]`; unknown keys are kept there with a
warning so new vocabulary is noticed, never lost.
"""

from __future__ import annotations

from ..schema import ModelRecord, prov, tri
from .openai_compat import OpenAICompatProvider

KNOWN_CAPS = (
    "completion_chat", "completion_fim", "function_calling", "fine_tuning", "vision",
    "classification", "ocr", "audio", "audio_speech", "audio_transcription",
    "audio_transcription_realtime", "moderation", "reasoning", "unified_resources",
)

# capability key -> stable endpoint key (same vocabulary as the OpenAI route map)
_ENDPOINT_CAPS = {
    "completion_chat": "chat_completions",
    "completion_fim": "fim_completions",
    "audio_speech": "speech_generation",
    "audio_transcription": "transcription",
    "audio_transcription_realtime": "realtime_transcription",
    "ocr": "ocr",
    "moderation": "moderation",
    "classification": "classification",
}


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
                return tri(caps.get(k)) if k in caps else None

            rec.capabilities.tool_calling = t("function_calling")
            rec.capabilities.fine_tuning = t("fine_tuning")
            rec.capabilities.reasoning = t("reasoning")
            for field, key in (("tool_calling", "function_calling"), ("fine_tuning", "fine_tuning"),
                               ("reasoning", "reasoning")):
                if getattr(rec.capabilities, field) is not None:
                    rec.provenance[field] = prov(src, "capabilities.%s" % key)
            for key, endpoint in _ENDPOINT_CAPS.items():
                if t(key) is not None:
                    rec.endpoints[endpoint] = t(key)
                    rec.provenance["endpoints." + endpoint] = prov(src, "capabilities.%s" % key)
            if t("vision") is not None:
                rec.modalities["image"]["input"] = t("vision")
                rec.provenance["modalities.image.input"] = prov(src, "capabilities.vision")
            if t("audio") is not None:
                rec.modalities["audio"]["input"] = t("audio")
                rec.provenance["modalities.audio.input"] = prov(src, "capabilities.audio")
            if t("completion_chat"):
                rec.modalities["text"] = {"input": True, "output": True}
            for k, v in caps.items():
                if k not in KNOWN_CAPS:
                    rec.warn("unrecognised capability key %r (kept in extra)" % k)
                rec.capabilities.extra["mistral." + k] = tri(v)
        ctx = raw.get("max_context_length")
        if isinstance(ctx, int):
            rec.context_window = ctx
            rec.provenance["context_window"] = prov(src, "max_context_length")
        rec.aliases = [a for a in (raw.get("aliases") or []) if a != rec.model_id]
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
