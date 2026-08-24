"""
xAI, three official sources joined by exact id:

  1. `GET https://api.x.ai/v1/models` (OpenAI-compatible) — availability.
  2. `GET /v1/language-models` — xAI's richer listing: modalities, aliases,
     raw token prices (kept in `raw`, unit is provider-specific).
  3. `https://docs.x.ai/developers/models/<model id>.md` — the official
     per-model documentation page, served as Markdown, with an explicit
     `## Capabilities` section (Function calling / Structured outputs /
     Reasoning: Yes|No) and an `## At a glance` block (modalities, context
     window, aliases, Batch API).

The documentation URL is derived from the exact listing id — no guessing; a
missing page leaves capabilities None with a warning. Where the API and the
docs both state a fact, the API wins and a contradiction is warned.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..http import FetchError
from ..schema import MODALITIES, ModelRecord, prov
from .base import ProviderResult
from .openai_compat import OpenAICompatProvider

LANGUAGE_MODELS_URL = "https://api.x.ai/v1/language-models"
DOCS_BASE = "https://docs.x.ai/developers/models/"

# "## Capabilities" bullet label -> schema capability field
CAPABILITY_KEYS = {
    "function calling": "tool_calling",
    "structured outputs": "structured_outputs",
    "reasoning": "reasoning",
}

_RE_H1 = re.compile(r"^#\s+(?!#)(.+?)\s*$")
_RE_H2 = re.compile(r"^##\s+(?!#)(.+?)\s*$")
_RE_BULLET = re.compile(r"^-\s+\*\*(.+?):\*\*\s*(.+?)\s*$")
_RE_CONTEXT = re.compile(r"^([\d,]+)\s+tokens$")
_RE_BACKTICK = re.compile(r"`([^`]+)`")

SECTION_GLANCE = "at a glance"
SECTION_CAPABILITIES = "capabilities"


class DocParseError(Exception):
    """Raised when a page does not look like an xAI model documentation page."""


def parse_model_page(markdown: str, model_id: str | None = None) -> dict:
    """One docs.x.ai model page -> a documentation record (plain dict)."""
    rec: dict[str, Any] = {
        "display_name": None, "description": None, "model_name": None, "aliases": [],
        "context_window": None, "input_modalities": None, "output_modalities": None,
        "batch": None, "capabilities": {}, "capabilities_section": False,
        "extra": {}, "warnings": [],
    }
    section = None
    saw_sections: set[str] = set()
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        m = _RE_H2.match(line)
        if m:
            section = m.group(1).strip().lower()
            saw_sections.add(section)
            continue
        if section is None:
            m = _RE_H1.match(line)
            if m and rec["display_name"] is None:
                rec["display_name"] = m.group(1).strip()
                continue
            if line and not line.startswith("#") and rec["display_name"] is not None and rec["description"] is None:
                rec["description"] = line.strip()
            continue
        if section == SECTION_GLANCE:
            m = _RE_BULLET.match(line)
            if not m:
                continue
            label, value = m.group(1).strip().lower(), m.group(2).strip()
            if label == "modalities":
                parts = value.split("→")
                if len(parts) == 2:
                    rec["input_modalities"] = [x.strip().lower() for x in parts[0].split(",") if x.strip()]
                    rec["output_modalities"] = [x.strip().lower() for x in parts[1].split(",") if x.strip()]
                else:
                    rec["warnings"].append("unrecognised Modalities value %r" % value)
            elif label == "context window":
                mm = _RE_CONTEXT.match(value)
                if mm:
                    rec["context_window"] = int(mm.group(1).replace(",", ""))
                else:
                    rec["warnings"].append("unrecognised Context window value %r" % value)
            elif label == "model name":
                rec["model_name"] = value.strip("`")
            elif label == "aliases":
                rec["aliases"] = _RE_BACKTICK.findall(value)
            elif label == "batch api":
                low = value.lower()
                if low == "not supported":
                    rec["batch"] = False
                elif low.startswith("supported"):
                    rec["batch"] = True
                else:
                    rec["warnings"].append("unexpected Batch API value %r" % value)
            elif label == "knowledge cutoff":
                rec["extra"]["knowledge_cutoff_raw"] = value
            else:
                rec["warnings"].append("unrecognised At a glance bullet %r" % label)
            continue
        if section == SECTION_CAPABILITIES:
            m = _RE_BULLET.match(line)
            if not m:
                continue
            label, value = m.group(1).strip().lower(), m.group(2).strip().lower()
            if value == "yes":
                val = True
            elif value == "no":
                val = False
            else:
                val = None
                rec["warnings"].append("unexpected capability value %r for %r" % (value, label))
            key = CAPABILITY_KEYS.get(label)
            if key:
                rec["capabilities"][key] = val
            else:
                rec["extra"]["docs." + re.sub(r"[^a-z0-9]+", "_", label)] = val
                rec["warnings"].append("unrecognised capability %r (kept)" % label)
            continue
        # Pricing / Rate limits / Regions: intentionally not parsed here.

    if SECTION_GLANCE not in saw_sections:
        raise DocParseError("page has no 'At a glance' section (%s)" % (model_id or "?"))
    rec["capabilities_section"] = SECTION_CAPABILITIES in saw_sections
    if model_id and rec["model_name"] and rec["model_name"] != model_id:
        rec["warnings"].append("documented model name %r differs from listing id %r" % (rec["model_name"], model_id))
    return rec


class XAIProvider(OpenAICompatProvider):
    name = "xai"
    auth = ("XAI_API_KEY",)
    base_url = "https://api.x.ai/v1"
    describe = "GET /v1/models + /v1/language-models + docs.x.ai per-model pages (capabilities)"

    def doc_url(self, model_id: str) -> str:
        return DOCS_BASE + model_id + ".md"

    def fixtures(self, root: Path):
        pages: dict[str, str | Path] = {
            self.models_url: root / "listings" / "xai_models.json",
            LANGUAGE_MODELS_URL: root / "listings" / "xai_language_models.json",
        }
        docs = root / "xai_docs"
        if docs.exists():
            for p in docs.glob("*.md"):
                pages[self.doc_url(p.stem)] = p
        return pages

    def enrich_context(self, raw, http) -> dict:
        ctx: dict[str, Any] = {"_sources": {"language_models": LANGUAGE_MODELS_URL,
                                            "documentation": DOCS_BASE + "<model id>.md"},
                               "_warnings": [], "by_id": {}}
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
        ins, outs = m.get("input_modalities"), m.get("output_modalities")
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

    def apply_documentation(self, rec: ModelRecord, doc: dict) -> None:
        src = "docs:developers/models"
        url = self.doc_url(rec.model_id)
        rec.sources["documentation"] = url
        rec.warnings.extend(doc["warnings"])
        if rec.display_name is None:
            rec.display_name = doc["display_name"]
        if rec.description is None:
            rec.description = doc["description"]
        for key, val in doc["capabilities"].items():
            setattr(rec.capabilities, key, val)
            rec.provenance[key] = prov(src, "Capabilities bullet")
        if doc["batch"] is not None:
            rec.capabilities.batch = doc["batch"]
            rec.provenance["batch"] = prov(src, "At a glance: Batch API")
        if doc["context_window"] is not None:
            if rec.context_window is None:
                rec.context_window = doc["context_window"]
                rec.provenance["context_window"] = prov(src, "At a glance: Context window")
            elif rec.context_window != doc["context_window"]:
                rec.warn("context window differs between API (%s) and docs (%s); keeping API" % (
                    rec.context_window, doc["context_window"]))
        for direction, mods in (("input", doc["input_modalities"]), ("output", doc["output_modalities"])):
            if mods is None:
                continue
            for mod in MODALITIES:
                current = rec.modalities[mod][direction]
                documented = mod in mods
                if current is None:
                    rec.modalities[mod][direction] = documented
                    rec.provenance.setdefault("modalities." + direction, prov(src, "At a glance: Modalities"))
                elif current != documented:
                    rec.warn("%s modality %r differs between API and docs; keeping API" % (direction, mod))
        if doc["aliases"]:
            if not rec.aliases:
                rec.aliases = list(doc["aliases"])
                rec.provenance["aliases"] = prov(src, "At a glance: Aliases")
                if rec.relationship == "unknown":
                    rec.family = rec.model_id
                    rec.relationship = "canonical"
            elif set(rec.aliases) != set(doc["aliases"]):
                rec.warn("alias list differs between API and docs; keeping API")
        if doc["extra"].get("knowledge_cutoff_raw"):
            from ..schema import parse_cutoff
            rec.knowledge_cutoff_raw = doc["extra"]["knowledge_cutoff_raw"]
            rec.knowledge_cutoff = parse_cutoff(rec.knowledge_cutoff_raw)
            rec.provenance["knowledge_cutoff"] = prov(src, "At a glance: Knowledge cutoff")
        for k, v in doc["extra"].items():
            if k != "knowledge_cutoff_raw":
                rec.capabilities.extra[k] = v

    def enrich(self, raw, http) -> ProviderResult:
        result = super().enrich(raw, http)
        urls = {self.doc_url(r.model_id): r for r in result.records}
        fetched = http.get_many(list(urls), expect_content_type="text/markdown")
        docs_pages: dict[str, str] = {}
        for url, rec in urls.items():
            res = fetched.get(url)
            if isinstance(res, Exception) or res is None:
                docs_pages[rec.model_id] = "missing"
                rec.warn("no documentation page (%s); capabilities not documented" % url)
                continue
            try:
                doc = parse_model_page(res[0], model_id=rec.model_id)
            except DocParseError as exc:
                docs_pages[rec.model_id] = "parse_failed"
                rec.warn("documentation page did not parse: %s" % exc)
                continue
            docs_pages[rec.model_id] = "ok" if doc["capabilities_section"] else "ok_no_capabilities"
            self.apply_documentation(rec, doc)
        result.extra["docs_pages"] = docs_pages
        return result

    def validate(self, current: dict, previous: dict | None) -> tuple[list[str], list[str]]:
        """Refuse when the docs stage collapses: pages that previously parsed
        (or previously carried a Capabilities section) mostly stop doing so."""
        errors: list[str] = []
        warnings: list[str] = []
        cur = (current.get("extra") or {}).get("docs_pages") or {}
        prev = ((previous or {}).get("extra") or {}).get("docs_pages") or {}
        bad = sorted(mid for mid, st in cur.items() if st in ("missing", "parse_failed"))
        if bad:
            warnings.append("model page(s) without parseable documentation: %s" % ", ".join(bad))
        if prev:
            had = [mid for mid, st in prev.items() if st.startswith("ok") and mid in cur]
            lost = [mid for mid in had if not cur[mid].startswith("ok")]
            if had and len(lost) / len(had) >= 0.5:
                errors.append("XAI DOCS REGRESSION suspected — %d/%d previously-documented pages no longer parse: %s"
                              % (len(lost), len(had), ", ".join(sorted(lost))))
            had_caps = [mid for mid, st in prev.items() if st == "ok" and mid in cur]
            lost_caps = [mid for mid in had_caps if cur[mid] != "ok"]
            if had_caps and len(lost_caps) / len(had_caps) >= 0.5:
                errors.append("XAI DOCS REGRESSION suspected — %d/%d pages lost their Capabilities section: %s"
                              % (len(lost_caps), len(had_caps), ", ".join(sorted(lost_caps))))
        return errors, warnings
