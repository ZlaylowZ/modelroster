"""
xAI: `GET https://api.x.ai/v1/models` (OpenAI-compatible, availability) plus
`GET /v1/language-models`, xAI's own richer listing that reports input/output
modalities and aliases per model. Prices from that endpoint are kept raw
(their unit is provider-specific) and not converted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import MODALITIES, ModelRecord, prov
from .openai_compat import OpenAICompatProvider

LANGUAGE_MODELS_URL = "https://api.x.ai/v1/language-models"


class XAIProvider(OpenAICompatProvider):
    name = "xai"
    auth = ("XAI_API_KEY",)
    base_url = "https://api.x.ai/v1"
    describe = "GET /v1/models (availability) + GET /v1/language-models (modalities, aliases)"

    def fixtures(self, root: Path):
        return {self.models_url: root / "listings" / "xai_models.json",
                LANGUAGE_MODELS_URL: root / "listings" / "xai_language_models.json"}

    def enrich_context(self, raw, http) -> dict:
        ctx: dict[str, Any] = {"_sources": {"language_models": LANGUAGE_MODELS_URL}, "_warnings": [], "by_id": {}}
        try:
            data = http.get_json(LANGUAGE_MODELS_URL, headers=self.headers())
        except Exception as exc:  # the richer endpoint is optional
            ctx["_warnings"].append("language-models endpoint unavailable: %s" % exc)
            return ctx
        for m in data.get("models") or []:
            if isinstance(m, dict) and m.get("id"):
                ctx["by_id"][m["id"]] = m
                for a in m.get("aliases") or []:
                    ctx["by_id"].setdefault(a, dict(m, _alias_of=m["id"]))
        return ctx

    def enrich_record(self, rec: ModelRecord, raw: dict, http, context: dict) -> None:
        m = context.get("by_id", {}).get(rec.model_id)
        if not m:
            return
        src = "api:/v1/language-models"
        canonical = m.get("_alias_of") or m["id"]
        rec.family = canonical
        rec.relationship = "alias" if m.get("_alias_of") else "canonical"
        rec.aliases = [a for a in (m.get("aliases") or []) if a != rec.model_id] if not m.get("_alias_of") else []
        rec.sources["language_models"] = LANGUAGE_MODELS_URL
        ins = m.get("input_modalities")
        outs = m.get("output_modalities")
        if isinstance(ins, list):
            for mod in MODALITIES:
                rec.modalities[mod]["input"] = mod in ins
            rec.provenance["modalities.input"] = prov(src, "input_modalities")
            for x in ins:
                if x not in MODALITIES:
                    rec.warn("unrecognised input modality %r" % x)
        if isinstance(outs, list):
            for mod in MODALITIES:
                rec.modalities[mod]["output"] = mod in outs
            rec.provenance["modalities.output"] = prov(src, "output_modalities")
        rec.raw["language_model"] = {k: v for k, v in m.items() if k != "_alias_of"}
        rec.endpoints["chat_completions"] = True
        rec.provenance["endpoints.chat_completions"] = prov(src, "listed as a language model")
