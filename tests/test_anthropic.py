"""Anthropic normalisation from the /v1/models capabilities object. No network."""

from helpers import FIXTURES, listing
from modelroster.http import FixtureFetcher
from modelroster.providers.anthropic import AnthropicProvider, normalise_model

SAMPLE = {
    "type": "model", "id": "claude-opus-5", "display_name": "Claude Opus 5", "created_at": "2026-07-24T00:00:00Z",
    "max_input_tokens": 1000000, "max_tokens": 128000,
    "capabilities": {
        "batch": {"supported": True}, "citations": {"supported": True}, "code_execution": {"supported": True},
        "context_management": {"supported": True, "clear_tool_uses_20250919": {"supported": True},
                               "clear_thinking_20251015": {"supported": True}, "compact_20260112": {"supported": True}},
        "effort": {"supported": True, "low": {"supported": True}, "medium": {"supported": True}, "high": {"supported": True},
                   "xhigh": {"supported": True}, "max": {"supported": True}},
        "image_input": {"supported": True}, "pdf_input": {"supported": True}, "structured_outputs": {"supported": True},
        "thinking": {"supported": True, "types": {"enabled": {"supported": False}, "adaptive": {"supported": True}}},
    },
}


def test_normalisation():
    rec = normalise_model(SAMPLE)
    assert rec.model_id == "claude-opus-5" and rec.released == "2026-07-24"
    assert rec.max_input_tokens == 1_000_000 and rec.context_window == 1_000_000 and rec.max_output_tokens == 128_000
    assert rec.capabilities.reasoning_efforts == ["low", "medium", "high", "xhigh", "max"]
    assert rec.capabilities.reasoning is True and rec.capabilities.extended_thinking is True
    assert rec.capabilities.extra["thinking_types"] == ["adaptive"]
    assert rec.capabilities.structured_outputs is True and rec.capabilities.pdf_input is True
    assert rec.modalities["image"]["input"] is True
    assert rec.capabilities.extra["context_management.compact_20260112"] is True
    assert rec.provenance["structured_outputs"]["section"] == "api:/v1/models"


def test_provider_wide_facts_are_marked_as_such():
    rec = normalise_model(SAMPLE)
    assert rec.capabilities.tool_calling is True and rec.capabilities.streaming is True
    assert rec.provenance["tool_calling"]["section"] == "provider_docs"
    assert rec.provenance["tool_calling"]["evidence"] == "provider-wide statement"
    assert "url" in rec.provenance["streaming"]
    # everything the API does not report stays unknown
    assert rec.capabilities.prompt_caching is None and rec.capabilities.fine_tuning is None


def test_missing_capability_blocks_are_unknown():
    rec = normalise_model(dict(SAMPLE, capabilities={}))
    assert rec.capabilities.reasoning is None and rec.capabilities.reasoning_efforts is None
    assert rec.capabilities.batch is None and rec.modalities["image"]["input"] is None


def test_unknown_blocks_are_kept_with_warnings():
    caps = dict(SAMPLE["capabilities"], teleport={"supported": True})
    caps["effort"] = dict(caps["effort"], ultra={"supported": True})
    rec = normalise_model(dict(SAMPLE, capabilities=caps))
    assert rec.capabilities.extra["teleport"] is True
    assert "ultra" in rec.capabilities.reasoning_efforts
    assert any("teleport" in w for w in rec.warnings) and any("ultra" in w for w in rec.warnings)


def test_fixture_listing_round_trip(fake_keys):
    prov = AnthropicProvider()
    http = FixtureFetcher(prov.fixtures(FIXTURES))
    raw = prov.list_models(http)
    assert [m["id"] for m in raw] == [m["id"] for m in listing("anthropic_models.json")["data"]]
    res = prov.enrich(raw, http)
    assert res.model_order[0] == raw[0]["id"]          # API order preserved (newest first)
    assert all(r.capabilities.tool_calling is True for r in res.records)


def test_pagination_follows_after_id(fake_keys):
    from modelroster.providers.anthropic import MODELS_URL
    import json
    pages = {
        MODELS_URL + "?limit=2": json.dumps({"data": [{"id": "a", "type": "model"}, {"id": "b", "type": "model"}], "has_more": True, "last_id": "b"}),
        MODELS_URL + "?limit=2&after_id=b": json.dumps({"data": [{"id": "c", "type": "model"}], "has_more": False}),
    }
    raw = AnthropicProvider(page_size=2).list_models(FixtureFetcher(pages))
    assert [m["id"] for m in raw] == ["a", "b", "c"]


def test_missing_key_skips_provider(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from modelroster.update import run_provider
    r = run_provider("anthropic", data_dir=tmp_path, fixtures_root=FIXTURES, quiet=True)
    assert r.skipped and r.ok and not (tmp_path / "anthropic.json").exists()
