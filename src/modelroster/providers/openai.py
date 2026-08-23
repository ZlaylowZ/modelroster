"""
OpenAI: availability from `GET https://api.openai.com/v1/models`, capabilities
from the official Markdown documentation (models.md catalog -> one page per
documented model), joined by exact id through an explicit alias index.

  1. Parse every documented page (openai_docs.py).
  2. Build the alias index: canonical id, explicit aliases and explicit
     snapshots map to the canonical record. Conflicts are resolved
     deterministically (the page whose own Model ID equals the contested id
     wins, else lexicographically first) and reported.
  3. exact API id -> alias index -> documentation record. No fuzzy matching.
  4. `ft:<base>:<org>::<id>` ids may inherit from the documented base
     (relationship "fine_tune_inherited").
  5. Unmatched API ids are kept with all capabilities None and a warning.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..http import FetchError
from ..schema import KNOWN_FEATURE_KEYS, ModelRecord, prov, utc_now_iso
from .base import BaseProvider, ProviderResult
from .openai_docs import (
    CATALOG_URL, DocParseError, discover_model_pages, header_facts, is_empty_parse, parse_model_page,
)

MODELS_URL = "https://api.openai.com/v1/models"
MAX_PAGE_FAILURE_FRACTION = 0.10
MAX_REGRESSION_FRACTION = 0.25
MAX_REGRESSION_COUNT = 10
MAX_HEADER_LOSS_FRACTION = 0.25

_RE_FINE_TUNE = re.compile(r"^ft:([^:]+):")


def fine_tune_base(model_id: str | None) -> str | None:
    m = _RE_FINE_TUNE.match(model_id or "")
    return m.group(1) if m else None


def build_alias_index(doc_records: dict[str, dict]):
    """
    Pure function (does not mutate its input).
    Returns (alias_index, conflicts, routes_to) where
      alias_index = {id: {"canonical_model_id", "relationship"}}
      conflicts   = [{"id", "claimants", "resolved_to"}]
      routes_to   = {losing_canonical_id: winning_canonical_id}
    """
    claims: dict[str, list[tuple[str, str]]] = {}

    def claim(id_, canonical, relationship):
        claims.setdefault(id_, []).append((canonical, relationship))

    for canonical, rec in doc_records.items():
        claim(canonical, canonical, "canonical")
        for a in rec.get("aliases") or []:
            claim(a, canonical, "alias")
        for s in rec.get("snapshots") or []:
            if s != canonical:
                claim(s, canonical, "snapshot")

    alias_index, conflicts, routes_to = {}, [], {}
    for id_, claimants in claims.items():
        distinct = sorted({c for c, _ in claimants})
        if len(distinct) == 1:
            canonical = distinct[0]
            alias_index[id_] = {"canonical_model_id": canonical,
                                "relationship": _best_relationship([r for c, r in claimants if c == canonical])}
            continue
        owners = [c for c in distinct if c == id_]
        resolved = owners[0] if owners else distinct[0]
        alias_index[id_] = {"canonical_model_id": resolved,
                            "relationship": _best_relationship([r for c, r in claimants if c == resolved])}
        conflicts.append({"id": id_, "claimants": distinct, "resolved_to": resolved})
        for loser in distinct:
            if loser != resolved and loser in doc_records:
                routes_to[loser] = resolved
    return alias_index, conflicts, routes_to


def _best_relationship(rels):
    for r in ("canonical", "snapshot", "alias"):
        if r in rels:
            return r
    return rels[0] if rels else "unknown"


def doc_to_record(doc: dict, model_id: str, api: dict | None, relationship: str,
                  retrieved_at: str, inherited_from: str | None = None,
                  routes_to: str | None = None) -> ModelRecord:
    """Project a documentation record (+ API listing fields) onto a ModelRecord."""
    rec = ModelRecord(provider="openai", model_id=model_id, display_name=doc["display_name"],
                      description=doc["description"], family=doc["canonical_model_id"],
                      aliases=list(doc["aliases"]), snapshots=list(doc["snapshots"]),
                      default_snapshot=doc["default_snapshot"], routes_to=routes_to,
                      relationship=relationship, retrieved_at=retrieved_at)
    rec.context_window = doc["context_window"]
    rec.max_input_tokens = doc["max_input_tokens"]
    rec.max_output_tokens = doc["max_output_tokens"]
    rec.knowledge_cutoff = doc["knowledge_cutoff"]
    rec.knowledge_cutoff_raw = doc["knowledge_cutoff_raw"]
    rec.modalities = {k: dict(v) for k, v in doc["modalities"].items()}
    rec.endpoints = dict(doc["endpoints"])
    rec.builtin_tools = dict(doc["tools"]) if doc["tools"] is not None else None
    rec.pricing = dict(doc["pricing"]) if doc.get("pricing") else None
    c = rec.capabilities
    c.reasoning = doc["reasoning"]["supported"]
    c.reasoning_efforts = list(doc["reasoning"]["efforts"]) if doc["reasoning"]["efforts"] is not None else None
    c.default_effort = doc["reasoning"]["default_effort"]
    c.tool_calling = doc["function_calling"]
    c.structured_outputs = doc["structured_outputs"]
    c.streaming = doc["streaming"]
    c.fine_tuning = doc["fine_tuning"]
    c.prompt_caching = doc["prompt_caching"]
    c.batch = doc["endpoints"].get("batch")
    feats = doc["features"] or {}
    for k in KNOWN_FEATURE_KEYS:
        c.extra["features." + k] = feats.get(k)
    for k, v in feats.items():
        if k not in KNOWN_FEATURE_KEYS:
            c.extra["features." + k] = v
    if feats.get("image_input") is True and rec.modalities["image"]["input"] is None:
        rec.modalities["image"]["input"] = True
    rec.provenance = dict(doc["provenance"])
    if "function_calling" in rec.provenance:
        rec.provenance["tool_calling"] = rec.provenance["function_calling"]
    if "reasoning.supported" in rec.provenance:
        rec.provenance["reasoning"] = rec.provenance["reasoning.supported"]
    if "reasoning.efforts" in rec.provenance:
        rec.provenance["reasoning_efforts"] = rec.provenance["reasoning.efforts"]
    if rec.endpoints.get("batch") is not None:
        rec.provenance["batch"] = rec.provenance.get("endpoints.batch")
    rec.sources = {"listing": MODELS_URL if api is not None else None, "documentation": doc["documentation_url"]}
    rec.raw["endpoints_raw"] = dict(doc["endpoints_raw"])
    rec.warnings = list(doc["warnings"])
    if api is not None:
        rec.raw.update({"created": api.get("created"), "owned_by": api.get("owned_by"),
                        "shutdown_date": api.get("shutdown_date")})
        if api.get("created"):
            rec.released = time.strftime("%Y-%m-%d", time.gmtime(api["created"]))
            rec.provenance["released"] = prov("api:/v1/models", "created timestamp")
        if api.get("shutdown_date"):
            rec.shutdown_date = api["shutdown_date"]
            rec.deprecated = True
            rec.provenance["shutdown_date"] = prov("api:/v1/models", "shutdown_date field")
    if inherited_from:
        rec.raw["inherited_from"] = inherited_from
        rec.warn("capabilities inherited from base model %r documentation; not independently documented for this fine-tune" % inherited_from)
    return rec


def unmatched_record(model_id: str, api: dict, retrieved_at: str) -> ModelRecord:
    rec = ModelRecord(provider="openai", model_id=model_id, relationship="unknown", retrieved_at=retrieved_at,
                      sources={"listing": MODELS_URL, "documentation": None})
    rec.raw = {"created": api.get("created"), "owned_by": api.get("owned_by"), "shutdown_date": api.get("shutdown_date")}
    if api.get("created"):
        rec.released = time.strftime("%Y-%m-%d", time.gmtime(api["created"]))
    if api.get("shutdown_date"):
        rec.shutdown_date = api["shutdown_date"]
        rec.deprecated = True
    rec.warn("No official model documentation record matched this API model ID.")
    base = fine_tune_base(model_id)
    if base:
        rec.raw["fine_tune_base"] = base
        rec.warn("fine-tuned model whose base %r has no documentation record" % base)
    return rec


class OpenAIProvider(BaseProvider):
    name = "openai"
    auth = ("OPENAI_API_KEY",)
    describe = "GET /v1/models for availability; developers.openai.com Markdown pages for capabilities"

    def __init__(self, catalog_url: str = CATALOG_URL):
        self.catalog_url = catalog_url

    def list_models(self, http: Any) -> list[dict]:
        data = http.get_json(MODELS_URL, headers={"Authorization": "Bearer " + self.require_key()})
        out = []
        for m in data.get("data") or []:
            out.append({"id": m.get("id"), "created": m.get("created"), "owned_by": m.get("owned_by"),
                        "shutdown_date": m.get("shutdown_date")})
        out.sort(key=lambda x: x["id"] or "")
        return out

    def fixtures(self, root: Path) -> dict[str, str | Path]:
        docs = root / "openai_docs"
        pages: dict[str, str | Path] = {self.catalog_url: docs / "models.md",
                                        MODELS_URL: root / "listings" / "openai_models.json"}
        for p in docs.glob("*.md"):
            if p.stem != "models":
                pages["https://developers.openai.com/api/docs/models/%s.md" % p.stem] = p
        return pages

    # ── documentation stage ──
    def fetch_documentation(self, http: Any) -> tuple[dict[str, dict], list[dict], int]:
        """Returns (doc_records by canonical id, failures, pages_discovered)."""
        catalog_md, _ = http.get_text(self.catalog_url, expect_content_type="text/markdown")
        pages = discover_model_pages(catalog_md)
        if not pages:
            raise DocParseError("no model documentation links discovered in models.md — format change?")
        url_to_slug = {url: slug for slug, url in pages}
        fetched = http.get_many(list(url_to_slug), expect_content_type="text/markdown")
        doc_records, failed = {}, []
        for url in url_to_slug:
            slug = url_to_slug[url]
            res = fetched.get(url)
            if isinstance(res, Exception) or res is None:
                failed.append({"slug": slug, "reason": "fetch: %s" % res})
                continue
            text, _meta = res
            try:
                rec = parse_model_page(text, documentation_url=url, slug=slug)
            except DocParseError as exc:
                failed.append({"slug": slug, "reason": "parse: %s" % exc})
                continue
            cid = rec["canonical_model_id"]
            if cid in doc_records:
                failed.append({"slug": slug, "reason": "duplicate canonical id %r" % cid})
                continue
            doc_records[cid] = rec
        return doc_records, failed, len(pages)

    def enrich(self, raw: list[dict], http: Any) -> ProviderResult:
        retrieved_at = utc_now_iso()
        doc_records, failed, discovered = self.fetch_documentation(http)
        alias_index, conflicts, routes_to = build_alias_index(doc_records)
        for loser, winner in routes_to.items():
            doc_records[loser]["routes_to"] = winner
            doc_records[loser]["warnings"].append("claims an id owned by %r; recorded as routes_to" % winner)

        records, unmatched, fine_tuned = [], [], []
        for am in raw:
            sdk_id = am["id"]
            hit = alias_index.get(sdk_id)
            relationship = None
            inherited_from = None
            if hit is None:
                base = fine_tune_base(sdk_id)
                if base and base in alias_index:
                    hit = alias_index[base]
                    relationship = "fine_tune_inherited"
                    inherited_from = base
            if hit is None:
                records.append(unmatched_record(sdk_id, am, retrieved_at))
                unmatched.append(sdk_id)
                if fine_tune_base(sdk_id):
                    fine_tuned.append(sdk_id)
                continue
            canonical = hit["canonical_model_id"]
            doc = doc_records[canonical]
            rec = doc_to_record(doc, sdk_id, am, relationship or hit["relationship"], retrieved_at,
                                inherited_from=inherited_from, routes_to=doc.get("routes_to"))
            if inherited_from:
                fine_tuned.append(sdk_id)
            records.append(rec)

        api_ids = {am["id"] for am in raw}
        documentation_only = sorted(cid for cid in doc_records if cid not in api_ids)
        warnings = []
        for c in conflicts:
            warnings.append("id %r claimed by %s; resolved to %r" % (c["id"], ", ".join(c["claimants"]), c["resolved_to"]))
        if unmatched:
            warnings.append("%d API model id(s) have no documentation record: %s" % (len(unmatched), ", ".join(unmatched)))
        empty_pages = sorted(cid for cid, r in doc_records.items() if is_empty_parse(r))
        if empty_pages:
            warnings.append("pages with no usable capability data: %s" % ", ".join(empty_pages))

        return ProviderResult(
            records=records,
            sources={"listing": MODELS_URL, "documentation_catalog": self.catalog_url},
            stats={"api_models": len(raw), "documented_model_families": len(doc_records),
                   "matched_api_models": len(raw) - len(unmatched), "unmatched_api_models": len(unmatched),
                   "pages_discovered": discovered, "page_failures": len(failed)},
            warnings=warnings,
            extra={"documentation_models": doc_records, "alias_index": alias_index,
                   "alias_conflicts": conflicts, "page_failures": failed, "pages_discovered": discovered,
                   "unmatched_api_models": unmatched, "fine_tuned_api_models": fine_tuned,
                   "documentation_only_ids": documentation_only,
                   "header_facts": {cid: header_facts(r) for cid, r in doc_records.items()}},
        )

    # ── gates ──
    def validate(self, current: dict, previous: dict | None) -> tuple[list[str], list[str]]:
        errors, warnings = [], []
        extra = current.get("extra") or {}
        docs = extra.get("documentation_models") or {}
        if not docs:
            errors.append("no documentation models were parsed at all")
        failed = extra.get("page_failures") or []
        discovered = extra.get("pages_discovered") or 0
        if failed:
            for f in failed:
                warnings.append("page %s: %s" % (f["slug"], f["reason"]))
            if discovered and len(failed) / float(discovered) > MAX_PAGE_FAILURE_FRACTION:
                errors.append("%d/%d documentation pages failed to fetch/parse — refusing to update" % (len(failed), discovered))
        for cid, rec in docs.items():
            for w in rec.get("warnings") or []:
                warnings.append("%s: %s" % (cid, w))

        prev_extra = (previous or {}).get("extra") or {}
        prev_docs = prev_extra.get("documentation_models") or {}
        if prev_docs:
            regressed = []
            for cid, prev in prev_docs.items():
                cur = docs.get(cid)
                if cur is None:
                    continue
                prev_has = bool(prev.get("endpoints")) or prev.get("context_window") is not None
                cur_has = bool(cur.get("endpoints")) or cur.get("context_window") is not None
                if prev_has and not cur_has:
                    regressed.append(cid)
            if regressed:
                frac = len(regressed) / max(1, len(prev_docs))
                msg = "%d previously-documented model(s) now parse to no capabilities: %s" % (
                    len(regressed), ", ".join(sorted(regressed)))
                if frac > MAX_REGRESSION_FRACTION or len(regressed) >= MAX_REGRESSION_COUNT:
                    errors.append("PARSER REGRESSION suspected — " + msg)
                else:
                    warnings.append(msg)
            if len(prev_docs) >= 10 and len(docs) < len(prev_docs) * 0.5:
                errors.append("documentation catalog shrank from %d to %d pages — refusing to overwrite" % (len(prev_docs), len(docs)))

            # Header-region facts: pages that previously yielded an effort
            # sentence / prose alias and are still documented must still yield it.
            prev_hf = prev_extra.get("header_facts") or {}
            cur_hf = extra.get("header_facts") or {}
            for fact in ("effort_sentence", "prose_alias"):
                had = [cid for cid, f in prev_hf.items() if f.get(fact) and cid in cur_hf]
                lost = [cid for cid in had if not cur_hf[cid].get(fact)]
                if had and lost:
                    msg = "%d/%d page(s) lost the header-region fact %r: %s" % (len(lost), len(had), fact, ", ".join(sorted(lost)))
                    if len(lost) / float(len(had)) >= MAX_HEADER_LOSS_FRACTION:
                        errors.append("HEADER PARSE REGRESSION suspected — " + msg)
                    else:
                        warnings.append(msg)
        return errors, warnings
