"""
Parse OpenAI's official Markdown model documentation pages into
documentation records (plain dicts; the provider turns them into ModelRecords).

Heading-driven state machine: walk the page line by line, track the current
"## " section, apply narrow section-local rules. Nothing depends on absolute
line positions; unrecognised content becomes a warning, never a silent drop.

Observed page structure (96 pages, 2026-08-18):

    # <Display name>
    > <tagline>
    Model ID: `<id>`
    <prose; may contain "Reasoning.effort supports: ..." and
            "The `<alias>` alias routes requests to ...">
    ## Model details        (Default snapshot, Input/Output modalities, N context
                             window, Maximum input tokens: N, N max output tokens,
                             <date> knowledge cutoff, Reasoning token support)
    ## Pricing              (### Text tokens table: Input / Cached input / Output)
    ## Endpoints            (| Endpoint | Route | Support |)
    ## Supported features   (positive list; optional)
    ## Supported tools      (positive list; optional)
    ## Quick comparison     (ignored)
    ## Snapshots            (bullets of `ids`)
    ## Rate limits          (ignored)

Tri-state rules:
  * Endpoints table: Supported -> True, Not supported -> False, else None+warning.
  * Supported features / tools / modalities / reasoning marker are positive
    enumerations: section PRESENT -> listed key True, known absent key False
    (evidence "absent_from_list"); section ABSENT -> whole group None.

Header-region facts (effort sentence, prose alias) are only looked for before
the first H2; `header_facts(rec)` reports which ones a page yielded so the
updater can detect their loss across runs.
"""

from __future__ import annotations

import re

from ..schema import (
    ENDPOINT_ROUTE_TO_KEY, KNOWN_EFFORTS, KNOWN_FEATURE_KEYS, KNOWN_TOOL_KEYS, MODALITIES,
    empty_modalities, parse_cutoff, prov,
)

DOCS_BASE = "https://developers.openai.com"
CATALOG_URL = DOCS_BASE + "/api/docs/models.md"
MODEL_PAGE_PREFIX = "/api/docs/models/"

SECTION_MODEL_DETAILS = "model details"
SECTION_ENDPOINTS = "endpoints"
SECTION_FEATURES = "supported features"
SECTION_TOOLS = "supported tools"
SECTION_SNAPSHOTS = "snapshots"
SECTION_PRICING = "pricing"

_RE_H1 = re.compile(r"^#\s+(?!#)(.+?)\s*$")
_RE_H2 = re.compile(r"^##\s+(?!#)(.+?)\s*$")
_RE_H3 = re.compile(r"^###\s+(?!#)(.+?)\s*$")
_RE_MODEL_ID = re.compile(r"^Model ID:\s*`([^`]+)`\s*$")
_RE_TAGLINE = re.compile(r"^>\s+(?!For the complete documentation index)(.+?)\s*$")
_RE_EFFORT = re.compile(r"Reasoning\.effort supports:\s*(.+?)\.(?:\s|$)")
_RE_PROSE_ALIAS = re.compile(r"The `([A-Za-z0-9._:-]+)` alias routes requests to")
_RE_BULLET = re.compile(r"^-\s+(.+?)\s*$")
_RE_DEFAULT_SNAPSHOT = re.compile(r"^Default snapshot:\s*`([^`]+)`$")
_RE_INPUT_MODS = re.compile(r"^Input modalities:\s*(.+)$")
_RE_OUTPUT_MODS = re.compile(r"^Output modalities:\s*(.+)$")
_RE_CONTEXT = re.compile(r"^([\d,]+)\s+context window$")
_RE_MAX_INPUT = re.compile(r"^Maximum input tokens:\s*([\d,]+)$")
_RE_MAX_OUTPUT = re.compile(r"^([\d,]+)\s+max output tokens$")
_RE_CUTOFF = re.compile(r"^(.+?)\s+knowledge cutoff$")
_RE_REASONING_MARKER = re.compile(r"^Reasoning token support$")
_RE_TABLE_ROW = re.compile(r"^\|\s*(.+?)\s*\|\s*`([^`]+)`\s*\|\s*(.+?)\s*\|\s*$")
_RE_PRICE_ROW = re.compile(r"^\|\s*(.+?)\s*\|\s*\$?([\d.,]+)\s*\|\s*(.+?)\s*\|\s*$")
_RE_BACKTICK_ID = re.compile(r"^`([^`]+)`$")
_RE_CATALOG_LINK = re.compile(r"\((" + re.escape(MODEL_PAGE_PREFIX) + r"[A-Za-z0-9._-]+?)(?:\.md)?\)")

PRICE_LABELS = {"input": "input", "cached input": "cached_input", "output": "output"}


class DocParseError(Exception):
    """Raised when a page does not look like a model documentation page at all."""


def empty_doc_record() -> dict:
    return {
        "canonical_model_id": None, "display_name": None, "description": None,
        "documentation_url": None, "aliases": [], "snapshots": [], "default_snapshot": None,
        "routes_to": None,
        "reasoning": {"supported": None, "efforts": None, "default_effort": None},
        "function_calling": None, "structured_outputs": None, "streaming": None,
        "fine_tuning": None, "prompt_caching": None,
        "features": {}, "modalities": empty_modalities(),
        "endpoints": {}, "endpoints_raw": {}, "tools": None,
        "context_window": None, "max_input_tokens": None, "max_output_tokens": None,
        "knowledge_cutoff": None, "knowledge_cutoff_raw": None,
        "pricing": None,
        "provenance": {}, "warnings": [],
    }


def discover_model_pages(catalog_markdown: str) -> list[tuple[str, str]]:
    """Ordered, de-duplicated [(slug, absolute .md url)] from models.md."""
    seen: list[str] = []
    for path in _RE_CATALOG_LINK.findall(catalog_markdown):
        slug = path[len(MODEL_PAGE_PREFIX):]
        if slug and slug not in seen:
            seen.append(slug)
    return [(s, DOCS_BASE + MODEL_PAGE_PREFIX + s + ".md") for s in seen]


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _split_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_effort_sentence(sentence: str):
    """'none (default), low, medium, high and xhigh' -> (efforts, default, warnings)."""
    warnings = []
    s = sentence.strip().rstrip(".")
    s = re.sub(r",?\s+and\s+", ", ", s)
    efforts, default = [], None
    for tok in _split_list(s):
        is_default = "(default)" in tok
        tok = tok.replace("(default)", "").strip().lower()
        if not tok:
            continue
        if tok not in KNOWN_EFFORTS:
            warnings.append("unrecognised reasoning effort value %r" % tok)
        efforts.append(tok)
        if is_default:
            default = tok
    return efforts, default, warnings


def parse_model_page(markdown: str, documentation_url: str | None = None, slug: str | None = None) -> dict:
    """Parse one official model Markdown page. Raises DocParseError for non-model pages."""
    rec = empty_doc_record()
    rec["documentation_url"] = documentation_url
    warnings = rec["warnings"]

    section = None
    subsection = None
    saw_sections: set[str] = set()
    features_listed: list[str] = []
    tools_listed: list[str] = []
    snapshots: list[str] = []
    pricing: dict[str, float] = {}
    input_mods = output_mods = None

    for raw in markdown.splitlines():
        line = raw.rstrip()

        m = _RE_H2.match(line)
        if m:
            section = m.group(1).strip().lower()
            subsection = None
            saw_sections.add(section)
            continue
        m = _RE_H3.match(line)
        if m:
            subsection = m.group(1).strip().lower()
            continue

        if section is None:
            m = _RE_H1.match(line)
            if m and rec["display_name"] is None:
                rec["display_name"] = m.group(1).strip()
                continue
            m = _RE_MODEL_ID.match(line)
            if m:
                rec["canonical_model_id"] = m.group(1).strip()
                continue
            m = _RE_TAGLINE.match(line)
            if m and rec["description"] is None:
                rec["description"] = m.group(1).strip()
                continue
            m = _RE_EFFORT.search(line)
            if m:
                efforts, default, w = parse_effort_sentence(m.group(1))
                rec["reasoning"]["efforts"] = efforts
                rec["reasoning"]["default_effort"] = default
                rec["reasoning"]["supported"] = True
                rec["provenance"]["reasoning.efforts"] = prov("header", "Reasoning.effort supports sentence")
                warnings.extend(w)
            m = _RE_PROSE_ALIAS.search(line)
            if m:
                alias = m.group(1)
                if alias not in rec["aliases"]:
                    rec["aliases"].append(alias)
                rec["provenance"].setdefault("aliases", []).append(
                    prov("header", "prose alias sentence", alias=alias))
            continue

        if section == SECTION_MODEL_DETAILS:
            m = _RE_BULLET.match(line)
            if not m:
                continue
            b = m.group(1).strip()
            mm = _RE_DEFAULT_SNAPSHOT.match(b)
            if mm:
                rec["default_snapshot"] = mm.group(1)
                continue
            mm = _RE_INPUT_MODS.match(b)
            if mm:
                input_mods = [x.lower() for x in _split_list(mm.group(1))]
                continue
            mm = _RE_OUTPUT_MODS.match(b)
            if mm:
                output_mods = [x.lower() for x in _split_list(mm.group(1))]
                continue
            mm = _RE_CONTEXT.match(b)
            if mm:
                rec["context_window"] = _int(mm.group(1))
                rec["provenance"]["context_window"] = prov("Model details", "bullet")
                continue
            mm = _RE_MAX_INPUT.match(b)
            if mm:
                rec["max_input_tokens"] = _int(mm.group(1))
                rec["provenance"]["max_input_tokens"] = prov("Model details", "bullet")
                continue
            mm = _RE_MAX_OUTPUT.match(b)
            if mm:
                rec["max_output_tokens"] = _int(mm.group(1))
                rec["provenance"]["max_output_tokens"] = prov("Model details", "bullet")
                continue
            mm = _RE_CUTOFF.match(b)
            if mm:
                rec["knowledge_cutoff_raw"] = mm.group(1).strip()
                rec["knowledge_cutoff"] = parse_cutoff(rec["knowledge_cutoff_raw"])
                if rec["knowledge_cutoff"] is None:
                    warnings.append("unparseable knowledge cutoff %r (raw text kept)" % rec["knowledge_cutoff_raw"])
                rec["provenance"]["knowledge_cutoff"] = prov("Model details", "bullet")
                continue
            if _RE_REASONING_MARKER.match(b):
                rec["reasoning"]["supported"] = True
                rec["provenance"]["reasoning.supported"] = prov("Model details", "Reasoning token support bullet")
                continue
            warnings.append("unrecognised Model details bullet: %r" % b)
            continue

        if section == SECTION_PRICING:
            if subsection == "text tokens":
                m = _RE_PRICE_ROW.match(line)
                if m and m.group(3).strip().lower() == "1m tokens":
                    label = m.group(1).strip().lower()
                    key = PRICE_LABELS.get(label)
                    if key and key not in pricing:
                        try:
                            pricing[key] = float(m.group(2).replace(",", ""))
                        except ValueError:
                            warnings.append("unparseable price %r for %r" % (m.group(2), label))
            continue

        if section == SECTION_ENDPOINTS:
            m = _RE_TABLE_ROW.match(line)
            if not m:
                continue
            name, route, support = m.group(1), m.group(2).strip(), m.group(3).strip()
            if name.lower() == "endpoint":
                continue
            key = ENDPOINT_ROUTE_TO_KEY.get(route)
            if key is None:
                key = re.sub(r"[^a-z0-9]+", "_", route.lower()).strip("_")
                warnings.append("unrecognised endpoint route %r (stored as %r)" % (route, key))
            sup = support.lower()
            if sup == "supported":
                val = True
            elif sup == "not supported":
                val = False
            else:
                val = None
                warnings.append("unexpected endpoint support value %r for %s" % (support, route))
            rec["endpoints"][key] = val
            rec["endpoints_raw"][route] = support
            rec["provenance"]["endpoints." + key] = prov("Endpoints", "table row", route=route)
            continue

        if section == SECTION_FEATURES:
            m = _RE_BULLET.match(line)
            if m:
                features_listed.append(m.group(1).strip())
            continue

        if section == SECTION_TOOLS:
            m = _RE_BULLET.match(line)
            if m:
                tools_listed.append(m.group(1).strip())
            continue

        if section == SECTION_SNAPSHOTS:
            m = _RE_BULLET.match(line)
            if m:
                mm = _RE_BACKTICK_ID.match(m.group(1).strip())
                if mm:
                    snapshots.append(mm.group(1))
                else:
                    warnings.append("unrecognised Snapshots bullet: %r" % m.group(1))
            continue

    # ── structural sanity ──
    if rec["canonical_model_id"] is None and not (saw_sections & {SECTION_MODEL_DETAILS, SECTION_ENDPOINTS, SECTION_SNAPSHOTS}):
        raise DocParseError("page does not look like a model documentation page (%s)" % (documentation_url or slug))
    if rec["canonical_model_id"] is None:
        warnings.append("no 'Model ID:' line found; falling back to slug %r" % slug)
        rec["canonical_model_id"] = slug
    elif slug and rec["canonical_model_id"] != slug:
        warnings.append("Model ID %r differs from documentation slug %r" % (rec["canonical_model_id"], slug))

    # ── modalities ──
    if input_mods is not None:
        for mod in MODALITIES:
            rec["modalities"][mod]["input"] = mod in input_mods
        for unknown in sorted(set(input_mods) - set(MODALITIES)):
            warnings.append("unrecognised input modality %r" % unknown)
        rec["provenance"]["modalities.input"] = prov("Model details", "Input modalities bullet")
    if output_mods is not None:
        for mod in MODALITIES:
            rec["modalities"][mod]["output"] = mod in output_mods
        for unknown in sorted(set(output_mods) - set(MODALITIES)):
            warnings.append("unrecognised output modality %r" % unknown)
        rec["provenance"]["modalities.output"] = prov("Model details", "Output modalities bullet")

    # ── reasoning marker lives in an always-present section ──
    if SECTION_MODEL_DETAILS in saw_sections and rec["reasoning"]["supported"] is None:
        rec["reasoning"]["supported"] = False
        rec["provenance"]["reasoning.supported"] = prov("Model details", "absent_from_list",
                                                        note="no 'Reasoning token support' bullet")

    # ── features ──
    if SECTION_FEATURES in saw_sections:
        feats = {k: (k in features_listed) for k in KNOWN_FEATURE_KEYS}
        for k in features_listed:
            if k not in KNOWN_FEATURE_KEYS:
                feats[k] = True
                warnings.append("unrecognised feature key %r (kept)" % k)
        rec["features"] = feats
        for k in ("function_calling", "structured_outputs", "streaming", "prompt_caching"):
            rec[k] = feats.get(k)
            rec["provenance"][k] = prov("Supported features", "listed" if feats.get(k) else "absent_from_list")
        rec["provenance"]["features"] = prov("Supported features", "positive list")
    else:
        rec["features"] = {k: None for k in KNOWN_FEATURE_KEYS}
        rec["provenance"]["features"] = prov(None, "section absent")

    ep_ft = rec["endpoints"].get("fine_tuning")
    feat_ft = rec["features"].get("fine_tuning")
    if ep_ft is not None:
        rec["fine_tuning"] = ep_ft
        rec["provenance"]["fine_tuning"] = prov("Endpoints", "table row", route="v1/fine-tuning")
        if feat_ft is True and ep_ft is False:
            warnings.append("contradiction: 'fine_tuning' listed under Supported features but v1/fine-tuning is Not supported")
    elif feat_ft is not None:
        rec["fine_tuning"] = feat_ft
        rec["provenance"]["fine_tuning"] = prov("Supported features", "listed" if feat_ft else "absent_from_list")

    # ── tools ──
    if SECTION_TOOLS in saw_sections:
        tools = {k: (k in tools_listed) for k in KNOWN_TOOL_KEYS}
        for k in tools_listed:
            if k not in KNOWN_TOOL_KEYS:
                tools[k] = True
                warnings.append("unrecognised tool key %r (kept)" % k)
        rec["tools"] = tools
        rec["provenance"]["tools"] = prov("Supported tools", "positive list")
    else:
        rec["tools"] = None
        rec["provenance"]["tools"] = prov(None, "section absent")

    # ── pricing ──
    if pricing:
        rec["pricing"] = {"input": pricing.get("input"), "output": pricing.get("output"),
                          "cached_input": pricing.get("cached_input"), "currency": "USD", "per": "1M tokens"}
        rec["provenance"]["pricing"] = prov("Pricing", "Text tokens table")

    # ── snapshots ──
    rec["snapshots"] = snapshots
    if snapshots:
        rec["provenance"]["snapshots"] = prov("Snapshots", "bullet list")
    if rec["default_snapshot"] and rec["default_snapshot"] not in snapshots:
        warnings.append("default snapshot %r is not in the Snapshots list" % rec["default_snapshot"])

    if SECTION_ENDPOINTS not in saw_sections:
        warnings.append("no Endpoints section found")
    if SECTION_MODEL_DETAILS not in saw_sections:
        warnings.append("no Model details section found")
    return rec


def is_empty_parse(rec: dict) -> bool:
    """True when a page yielded no usable capability information at all."""
    return (not rec["endpoints"]) and rec["context_window"] is None and not rec["snapshots"] \
        and rec["reasoning"]["supported"] is None and rec["tools"] is None


def header_facts(rec: dict) -> dict[str, bool]:
    """Which header-region (pre-H2) facts a page produced; tracked across runs."""
    return {
        "effort_sentence": rec["reasoning"]["efforts"] is not None,
        "prose_alias": bool(rec["aliases"]),
    }
