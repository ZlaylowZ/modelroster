"""OpenAI documentation parser — fixture-based, no network (ported from the prototype)."""

import pytest

from helpers import DOCS, load_doc, parse
from modelroster.providers.openai_docs import (
    DocParseError, discover_model_pages, header_facts, is_empty_parse, parse_effort_sentence, parse_model_page,
)
from modelroster.schema import KNOWN_FEATURE_KEYS, KNOWN_TOOL_KEYS, MODALITIES, parse_cutoff


def test_discover_model_pages_dedupes_md_and_bare_links():
    pages = discover_model_pages(load_doc("models"))
    slugs = [s for s, _ in pages]
    assert len(slugs) == len(set(slugs))
    assert "gpt-5.4" in slugs and "gpt-5.6-sol" in slugs and "text-embedding-3-small" in slugs
    assert all(url.endswith(".md") for _, url in pages)
    assert len(slugs) > 80


def test_every_catalogued_page_has_a_fixture():
    missing = [s for s, _ in discover_model_pages(load_doc("models")) if not (DOCS / (s + ".md")).exists()]
    assert missing == []


def test_canonical_id_and_display_name():
    r = parse("gpt-5.4")
    assert r["canonical_model_id"] == "gpt-5.4"
    assert r["display_name"] == "GPT-5.4"
    assert r["description"] == "A more affordable model for coding and professional work."
    assert r["documentation_url"].endswith("/gpt-5.4.md")


def test_reasoning_model_with_effort_sentence():
    r = parse("gpt-5.4")
    assert r["reasoning"] == {"supported": True, "efforts": ["none", "low", "medium", "high", "xhigh"], "default_effort": "none"}
    assert r["provenance"]["reasoning.efforts"]["section"] == "header"
    assert header_facts(r) == {"effort_sentence": True, "prose_alias": False}


def test_reasoning_model_without_effort_sentence_keeps_efforts_unknown():
    r = parse("o3")
    assert r["reasoning"]["supported"] is True
    assert r["reasoning"]["efforts"] is None
    assert r["reasoning"]["default_effort"] is None


def test_non_reasoning_model():
    r = parse("gpt-4o")
    assert r["reasoning"]["supported"] is False
    assert r["provenance"]["reasoning.supported"]["evidence"] == "absent_from_list"


def test_effort_sentence_with_max_and_default_in_middle():
    r = parse("gpt-5.6-sol")
    assert r["reasoning"]["efforts"] == ["none", "low", "medium", "high", "xhigh", "max"]
    assert r["reasoning"]["default_effort"] == "medium"


@pytest.mark.parametrize("sentence,efforts,default", [
    ("none (default), low, medium, high and xhigh", ["none", "low", "medium", "high", "xhigh"], "none"),
    ("medium, high (default) and xhigh", ["medium", "high", "xhigh"], "high"),
    ("minimal, low, medium, and high", ["minimal", "low", "medium", "high"], None),
    ("none, low, medium (default), high, xhigh, and max.", ["none", "low", "medium", "high", "xhigh", "max"], "medium"),
])
def test_parse_effort_sentence_variants(sentence, efforts, default):
    e, d, w = parse_effort_sentence(sentence)
    assert e == efforts and d == default and w == []


def test_unknown_effort_value_is_kept_with_warning():
    e, d, w = parse_effort_sentence("low, ultra (default), high")
    assert e == ["low", "ultra", "high"] and d == "ultra"
    assert any("ultra" in x for x in w)


def test_function_calling_supported():
    r = parse("gpt-5.4")
    assert r["function_calling"] is True and r["structured_outputs"] is True and r["streaming"] is True
    assert r["provenance"]["function_calling"] == {"section": "Supported features", "evidence": "listed"}


def test_function_calling_not_listed_is_false_when_section_present():
    r = parse("gpt-3.5-turbo")
    assert r["function_calling"] is False
    assert r["provenance"]["function_calling"]["evidence"] == "absent_from_list"
    assert r["features"]["fine_tuning"] is True


def test_missing_features_section_is_unknown_not_false():
    r = parse("text-embedding-3-small")
    assert r["function_calling"] is None and r["structured_outputs"] is None
    assert all(r["features"][k] is None for k in KNOWN_FEATURE_KEYS)
    assert r["provenance"]["features"]["evidence"] == "section absent"


def test_each_builtin_tool_parsed_independently():
    r = parse("gpt-5.4")
    assert all(r["tools"][k] is True for k in KNOWN_TOOL_KEYS)
    r4o = parse("gpt-4o")
    assert r4o["tools"]["web_search"] is True
    assert r4o["tools"]["hosted_shell"] is False and r4o["tools"]["computer_use"] is False


def test_tools_section_absent_is_unknown():
    r = parse("gpt-3.5-turbo")
    assert r["tools"] is None and r["function_calling"] is False


def test_modalities_by_direction():
    r = parse("gpt-5.4")
    assert r["modalities"]["text"] == {"input": True, "output": True}
    assert r["modalities"]["image"] == {"input": True, "output": False}
    assert r["modalities"]["audio"] == {"input": False, "output": False}
    assert set(r["modalities"]) == set(MODALITIES)


def test_endpoint_compatibility():
    r = parse("gpt-5.4")
    assert r["endpoints"]["chat_completions"] is True and r["endpoints"]["responses"] is True
    assert r["endpoints"]["batch"] is True and r["endpoints"]["realtime"] is False
    assert r["endpoints_raw"]["v1/chat/completions"] == "Supported"
    emb = parse("text-embedding-3-small")
    assert emb["endpoints"]["embeddings"] is True and emb["endpoints"]["chat_completions"] is False


def test_fine_tuning_from_endpoint_table():
    assert parse("gpt-4o")["fine_tuning"] is True
    assert parse("gpt-5.4")["fine_tuning"] is False
    assert parse("gpt-5.4")["provenance"]["fine_tuning"]["section"] == "Endpoints"


def test_context_window_and_output_limit():
    r = parse("gpt-5.4")
    assert r["context_window"] == 1_050_000 and r["max_output_tokens"] == 128_000 and r["max_input_tokens"] is None
    assert parse("gpt-5.6-sol")["max_input_tokens"] == 922_000
    e = parse("text-embedding-3-small")
    assert e["context_window"] is None and e["max_output_tokens"] is None


def test_knowledge_cutoff_parsed_and_raw_kept():
    r = parse("gpt-5.4")
    assert r["knowledge_cutoff_raw"] == "Aug 31, 2025"
    assert r["knowledge_cutoff"] == "2025-08-31"


@pytest.mark.parametrize("raw,iso", [
    ("Aug 31, 2025", "2025-08-31"), ("October 2023", "2023-10-01"), ("Sep 30, 2024", "2024-09-30"),
    ("2024-06", "2024-06-01"), ("2024-06-15", "2024-06-15"), ("unknown", None), ("", None), (None, None),
])
def test_parse_cutoff(raw, iso):
    assert parse_cutoff(raw) == iso


def test_pricing_from_text_tokens_table():
    r = parse("gpt-5.4")
    assert r["pricing"] == {"input": 2.5, "output": 15.0, "cached_input": 0.25, "currency": "USD", "per": "1M tokens"}
    assert r["provenance"]["pricing"]["section"] == "Pricing"


def test_snapshots_and_default_snapshot():
    r = parse("gpt-4o")
    assert r["snapshots"] == ["gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13"]
    assert r["default_snapshot"] == "gpt-4o-2024-08-06"


def test_prose_alias_is_captured_with_provenance():
    r = parse("gpt-5.6-sol")
    assert r["aliases"] == ["gpt-5.6"]
    assert r["provenance"]["aliases"][0]["evidence"] == "prose alias sentence"
    assert header_facts(r)["prose_alias"] is True


def test_alias_page_claims_other_models_snapshot():
    r = parse("daybreak-blue-latest")
    assert r["canonical_model_id"] == "daybreak-blue-latest"
    assert r["snapshots"] == ["gpt-5.6-sol"]
    assert r["endpoints"]["chat_completions"] is False


def test_unknown_endpoint_support_value_becomes_none_with_warning():
    md = load_doc("gpt-4o").replace("| Chat Completions | `v1/chat/completions` | Supported |",
                                    "| Chat Completions | `v1/chat/completions` | Preview |")
    r = parse_model_page(md, slug="gpt-4o")
    assert r["endpoints"]["chat_completions"] is None
    assert any("unexpected endpoint support value" in w for w in r["warnings"])


def test_unknown_feature_key_kept_with_warning():
    md = load_doc("gpt-4o").replace("- function_calling\n", "- function_calling\n- teleportation\n")
    r = parse_model_page(md, slug="gpt-4o")
    assert r["features"]["teleportation"] is True
    assert any("teleportation" in w for w in r["warnings"])


def test_unknown_model_details_bullet_warns_but_does_not_fail():
    md = load_doc("gpt-4o").replace("- Output modalities: text\n", "- Output modalities: text\n- Quantum entanglement: yes\n")
    r = parse_model_page(md, slug="gpt-4o")
    assert any("unrecognised Model details bullet" in w for w in r["warnings"])
    assert r["context_window"] == 128_000


def test_missing_model_id_line_falls_back_to_slug_with_warning():
    md = load_doc("gpt-4o").replace("Model ID: `gpt-4o`\n", "")
    r = parse_model_page(md, slug="gpt-4o")
    assert r["canonical_model_id"] == "gpt-4o"
    assert any("no 'Model ID:' line" in w for w in r["warnings"])


def test_model_id_slug_mismatch_is_warned_not_rewritten():
    r = parse_model_page(load_doc("gpt-4o"), slug="something-else")
    assert r["canonical_model_id"] == "gpt-4o"
    assert any("differs from documentation slug" in w for w in r["warnings"])


def test_completely_unrelated_page_raises():
    with pytest.raises(DocParseError):
        parse_model_page("# Pricing\n\nSome prose without any model sections.\n", slug="pricing")


def test_format_change_removing_sections_yields_empty_parse_signal():
    r = parse_model_page("# Mystery\n\nModel ID: `mystery-1`\n\n## Something new\n\n- foo\n", slug="mystery-1")
    assert is_empty_parse(r)
    assert any("no Endpoints section" in w for w in r["warnings"])
    assert any("no Model details section" in w for w in r["warnings"])


def test_contradiction_feature_vs_endpoint_is_warned():
    md = load_doc("gpt-4o").replace("| Fine-tuning | `v1/fine-tuning` | Supported |",
                                    "| Fine-tuning | `v1/fine-tuning` | Not supported |")
    r = parse_model_page(md, slug="gpt-4o")
    assert r["fine_tuning"] is False
    assert any("contradiction" in w for w in r["warnings"])


def test_all_fixture_pages_parse_without_empty_results():
    from modelroster.providers.openai_docs import discover_model_pages
    empties = []
    for slug, _ in discover_model_pages(load_doc("models")):
        r = parse(slug)
        if is_empty_parse(r):
            empties.append(slug)
    assert empties == []
