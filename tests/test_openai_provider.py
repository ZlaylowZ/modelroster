"""Alias index, API overlay, fine-tune handling, unmatched ids, OpenAI gates. No network."""

import json

from helpers import FIXTURES, api, docs, listing
from modelroster.http import FixtureFetcher
from modelroster.providers.openai import OpenAIProvider, build_alias_index, doc_to_record, fine_tune_base
from modelroster.update import build_envelope


def overlay(d, ids):
    """Run the overlay stage of the provider with pre-parsed docs and fake API ids."""
    prov = OpenAIProvider()
    prov.fetch_documentation = lambda http: (d, [], len(d))
    res = prov.enrich(api(*ids) if isinstance(ids[0], str) else ids, http=None)
    return build_envelope("openai", res)


def test_alias_index_contains_canonical_aliases_and_snapshots():
    idx, conflicts, routes = build_alias_index(docs("gpt-4o", "gpt-5.6-sol"))
    assert idx["gpt-4o"] == {"canonical_model_id": "gpt-4o", "relationship": "canonical"}
    assert idx["gpt-4o-2024-08-06"] == {"canonical_model_id": "gpt-4o", "relationship": "snapshot"}
    assert idx["gpt-5.6"] == {"canonical_model_id": "gpt-5.6-sol", "relationship": "alias"}
    assert conflicts == [] and routes == {}


def test_duplicate_ownership_resolves_to_page_whose_model_id_matches_without_mutating_input():
    d = docs("gpt-5.6-sol", "daybreak-blue-latest")
    before = json.dumps(d, sort_keys=True)
    idx, conflicts, routes = build_alias_index(d)
    assert json.dumps(d, sort_keys=True) == before          # pure function
    assert idx["gpt-5.6-sol"]["canonical_model_id"] == "gpt-5.6-sol"
    assert idx["daybreak-blue-latest"]["canonical_model_id"] == "daybreak-blue-latest"
    assert len(conflicts) == 1 and conflicts[0]["id"] == "gpt-5.6-sol"
    assert routes == {"daybreak-blue-latest": "gpt-5.6-sol"}


def test_duplicate_with_no_owner_is_deterministic_and_reported():
    a = {"canonical_model_id": "a-latest", "aliases": [], "snapshots": ["shared-007"], "warnings": []}
    b = {"canonical_model_id": "b-stable", "aliases": [], "snapshots": ["shared-007"], "warnings": []}
    idx, conflicts, _ = build_alias_index({"a-latest": a, "b-stable": b})
    assert idx["shared-007"]["canonical_model_id"] == "a-latest"
    assert conflicts[0]["claimants"] == ["a-latest", "b-stable"]


def test_api_ids_preserved_exactly_and_snapshot_inherits_family_capabilities():
    reg = overlay(docs("gpt-4o"), ["gpt-4o", "gpt-4o-2024-08-06"])
    snap = reg["models"]["gpt-4o-2024-08-06"]
    assert snap["model_id"] == "gpt-4o-2024-08-06" and snap["family"] == "gpt-4o"
    assert snap["relationship"] == "snapshot"
    assert snap["capabilities"]["tool_calling"] is True
    assert snap["sources"]["documentation"].endswith("/gpt-4o.md")
    assert reg["models"]["gpt-4o"]["relationship"] == "canonical"
    assert reg["models"]["gpt-4o"]["pricing"]["input"] == 2.5


def test_no_regex_guessing_for_date_suffixed_ids():
    reg = overlay(docs("gpt-4o"), ["gpt-4o-2099-01-01"])
    rec = reg["models"]["gpt-4o-2099-01-01"]
    assert rec["family"] is None and rec["relationship"] == "unknown"
    assert rec["capabilities"]["tool_calling"] is None
    assert "gpt-4o-2099-01-01" in reg["extra"]["unmatched_api_models"]
    assert rec["warnings"] == ["No official model documentation record matched this API model ID."]


def test_unmatched_api_model_is_kept_not_discarded():
    reg = overlay(docs("gpt-4o"), ["tts-1-1106"])
    assert "tts-1-1106" in reg["models"]
    assert reg["stats"]["unmatched_api_models"] == 1


def test_fine_tune_base_extraction():
    assert fine_tune_base("ft:gpt-4o-2024-08-06:acme::abc123") == "gpt-4o-2024-08-06"
    assert fine_tune_base("gpt-4o") is None and fine_tune_base("") is None


def test_fine_tuned_model_inherits_marked_not_claimed_as_documented():
    reg = overlay(docs("gpt-4o"), ["ft:gpt-4o-2024-08-06:acme::abc123"])
    rec = reg["models"]["ft:gpt-4o-2024-08-06:acme::abc123"]
    assert rec["relationship"] == "fine_tune_inherited"
    assert rec["raw"]["inherited_from"] == "gpt-4o-2024-08-06"
    assert rec["family"] == "gpt-4o"
    assert rec["capabilities"]["tool_calling"] is True
    assert any("inherited" in w for w in rec["warnings"])
    assert "ft:gpt-4o-2024-08-06:acme::abc123" in reg["extra"]["fine_tuned_api_models"]


def test_fine_tuned_model_with_unknown_base_stays_unknown():
    reg = overlay(docs("gpt-4o"), ["ft:mystery-9:acme::zzz"])
    rec = reg["models"]["ft:mystery-9:acme::zzz"]
    assert rec["capabilities"]["tool_calling"] is None and rec["relationship"] == "unknown"
    assert rec["raw"]["fine_tune_base"] == "mystery-9"


def test_documentation_only_ids_are_listed():
    reg = overlay(docs("gpt-4o", "o3"), ["gpt-4o"])
    assert reg["extra"]["documentation_only_ids"] == ["o3"]


def test_stats():
    reg = overlay(docs("gpt-4o"), ["gpt-4o", "gpt-4o-2024-05-13", "weird"])
    st = reg["stats"]
    assert (st["api_models"], st["documented_model_families"], st["matched_api_models"], st["unmatched_api_models"]) == (3, 1, 2, 1)


def test_shutdown_date_marks_deprecated():
    reg = overlay(docs("gpt-4o"), [{"id": "gpt-4o", "created": 1, "owned_by": "x", "shutdown_date": "2026-10-23"}])
    rec = reg["models"]["gpt-4o"]
    assert rec["deprecated"] is True and rec["shutdown_date"] == "2026-10-23"


def test_doc_to_record_maps_function_calling_to_tool_calling_with_provenance():
    d = docs("gpt-5.4")["gpt-5.4"]
    rec = doc_to_record(d, "gpt-5.4", None, "canonical", "2026-01-01T00:00:00Z")
    assert rec.capabilities.tool_calling is True
    assert rec.provenance["tool_calling"]["section"] == "Supported features"
    assert rec.capabilities.extra["features.function_calling"] is True
    assert rec.raw["endpoints_raw"]["v1/responses"] == "Supported"
    assert rec.builtin_tools["computer_use"] is True


def test_full_fixture_pipeline_lists_real_listing(fake_keys):
    prov = OpenAIProvider()
    http = FixtureFetcher(prov.fixtures(FIXTURES))
    raw = prov.list_models(http)
    assert len(raw) == len(listing("openai_models.json")["data"])
    res = prov.enrich(raw, http)
    assert res.stats["page_failures"] == 0
    assert res.stats["documented_model_families"] == 96


# ── OpenAI-specific gates ──

def _env_with_docs(d, ids=("gpt-4o",)):
    return overlay(d, list(ids))


def test_validate_refuses_parser_regression():
    prev = _env_with_docs(docs("gpt-4o", "o3", "gpt-5.4", "gpt-5-mini"))
    cur_docs = docs("gpt-4o", "o3", "gpt-5.4", "gpt-5-mini")
    for r in cur_docs.values():
        r["endpoints"] = {}
        r["context_window"] = None
    cur = _env_with_docs(cur_docs)
    errors, _ = OpenAIProvider().validate(cur, prev)
    assert any("PARSER REGRESSION" in e for e in errors)


def test_validate_refuses_catastrophic_catalog_shrink():
    many = {("m%02d" % i): dict(docs("gpt-4o")["gpt-4o"], canonical_model_id="m%02d" % i) for i in range(12)}
    prev = _env_with_docs(many, ids=["m00"])
    cur = _env_with_docs(docs("gpt-4o"))
    errors, _ = OpenAIProvider().validate(cur, prev)
    assert any("shrank" in e for e in errors)


def test_validate_refuses_too_many_page_failures():
    cur = _env_with_docs(docs("gpt-4o"))
    cur["extra"]["page_failures"] = [{"slug": "x%d" % i, "reason": "fetch"} for i in range(20)]
    cur["extra"]["pages_discovered"] = 96
    errors, _ = OpenAIProvider().validate(cur, None)
    assert any("failed to fetch/parse" in e for e in errors)


def test_validate_refuses_loss_of_header_facts():
    prev = _env_with_docs(docs("gpt-5.4", "gpt-5.6-sol", "gpt-5-mini", "o3"))
    cur_docs = docs("gpt-5.4", "gpt-5.6-sol", "gpt-5-mini", "o3")
    for r in cur_docs.values():
        r["reasoning"]["efforts"] = None       # effort sentences silently lost
    cur = _env_with_docs(cur_docs)
    errors, _ = OpenAIProvider().validate(cur, prev)
    assert any("HEADER PARSE REGRESSION" in e for e in errors)


def test_validate_reports_conflicts_and_unmatched_as_warnings_not_errors():
    cur = overlay(docs("gpt-5.6-sol", "daybreak-blue-latest"), ["gpt-5.6-sol", "daybreak-blue-latest", "tts-1"])
    from modelroster.validate import validate
    errors, warnings = validate(cur, None, OpenAIProvider())
    assert errors == []
    assert any("claimed by" in w for w in warnings)
    assert any("no documentation record" in w for w in warnings)
