"""OpenAI-compatible listing providers, Ollama, and the plugin interface. No network."""

from pathlib import Path

import pytest

from helpers import FIXTURES, run
from modelroster import providers
from modelroster.http import FixtureFetcher
from modelroster.providers.base import BaseProvider, ProviderResult
from modelroster.providers.openai_compat import OpenAICompatProvider
from modelroster.schema import ModelRecord


def records(name):
    p = providers.get(name)
    http = FixtureFetcher(p.fixtures(FIXTURES))
    return {r.model_id: r for r in p.enrich(p.list_models(http), http).records}


@pytest.mark.usefixtures("fake_keys")
class TestProviders:
    def test_xai_aliases_and_modalities(self):
        recs = records("xai")
        g4 = recs["grok-4"]
        assert g4.relationship == "canonical" and g4.aliases == ["grok-4-latest"]
        assert recs["grok-4-latest"].relationship == "alias" and recs["grok-4-latest"].family == "grok-4"
        assert g4.modalities["image"]["input"] is True and g4.modalities["audio"]["input"] is False
        assert g4.capabilities.tool_calling is None                 # listing does not say
        assert recs["grok-2-image-1212"].modalities["image"]["input"] is None   # not a language model

    def test_mistral_capabilities_object(self):
        recs = records("mistral")
        m = recs["mistral-large-latest"]
        assert m.capabilities.tool_calling is True and m.capabilities.fine_tuning is False
        assert m.context_window == 131072 and m.aliases == ["mistral-large-2411"]
        assert m.endpoints["chat_completions"] is True and m.deprecated is False
        assert recs["pixtral-12b-2409"].modalities["image"]["input"] is True
        old = recs["open-mistral-7b"]
        assert old.deprecated is True and old.shutdown_date == "2025-03-30"
        assert recs["mistral-embed"].endpoints["chat_completions"] is False
        assert m.capabilities.reasoning is None

    def test_google_native_enrichment(self):
        recs = records("google")
        g = recs["models/gemini-2.5-pro"]
        assert g.display_name == "Gemini 2.5 Pro" and g.context_window == 1048576 and g.max_output_tokens == 65536
        assert g.endpoints["generate_content"] is True and g.endpoints["embed_content"] is False
        assert g.capabilities.streaming is False     # streamGenerateContent not in the fixture's list
        assert g.capabilities.reasoning is True
        assert recs["models/gemini-embedding-001"].endpoints["embed_content"] is True
        assert g.capabilities.tool_calling is None

    def test_cohere_endpoints_and_features(self):
        recs = records("cohere")
        c = recs["command-a-03-2025"]
        assert c.endpoints["chat"] is True and c.endpoints["embed"] is False
        assert c.capabilities.tool_calling is True and c.capabilities.structured_outputs is True
        assert c.context_window == 256000
        assert recs["embed-v4.0"].capabilities.tool_calling is None        # no features list
        assert recs["command-a-vision-07-2025"].modalities["image"]["input"] is True

    def test_nvidia_listing_only(self):
        recs = records("nvidia")
        assert len(recs) > 50
        r = recs["meta/llama-3.1-8b-instruct"]
        assert r.relationship == "unknown" and r.capabilities.tool_calling is None
        assert r.raw["owned_by"] == "meta"

    def test_inception_rich_listing(self):
        r = records("inception")["mercury-2"]
        assert r.capabilities.tool_calling is True and r.capabilities.structured_outputs is True
        assert r.context_window == 128000 and r.max_output_tokens == 50000
        assert r.pricing["input"] == 0.25 and r.pricing["output"] == 0.75 and r.pricing["cached_input"] == 0.025
        assert r.modalities["text"] == {"input": True, "output": True}

    def test_ollama_show_capabilities(self):
        recs = records("ollama")
        q = recs["qwen3:8b"]
        assert q.capabilities.tool_calling is True and q.capabilities.reasoning is True
        assert q.context_window == 40960
        assert recs["llama3.1:8b"].capabilities.reasoning is False
        e = recs["nomic-embed-text:latest"]
        assert e.endpoints["embeddings"] is True and e.endpoints["chat"] is False

    def test_every_provider_declares_auth_and_fixtures(self):
        for n in providers.names():
            p = providers.get(n)
            assert isinstance(p.auth, tuple)
            assert p.fixtures(FIXTURES), n


def test_no_key_needed_for_public_listings(monkeypatch, tmp_path):
    for v in ("NVIDIA_API_KEY", "INCEPTION_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert run("nvidia", tmp_path).ok and run("inception", tmp_path).ok


def test_plugin_provider_needs_no_core_edits(fake_keys, tmp_path):
    class AcmeProvider(OpenAICompatProvider):
        name = "acme"
        auth = ("ACME_API_KEY",)
        base_url = "https://api.acme.test/v1"

        def fixtures(self, root: Path):
            return {self.models_url: '{"object":"list","data":[{"id":"acme-1","created":1700000000,"owned_by":"acme"}]}'}

    providers.register(AcmeProvider(), replace=True)
    import os
    os.environ["ACME_API_KEY"] = "x"
    r = run("acme", tmp_path)
    assert r.ok and (tmp_path / "acme.json").exists()
    from modelroster import load
    reg = load("acme", data_dir=tmp_path)
    assert reg.get("acme-1").provider == "acme"


def test_custom_base_provider_minimal():
    class Mini(BaseProvider):
        name = "mini"
        auth = ()

        def list_models(self, http):
            return [{"id": "m"}]

        def enrich(self, raw, http):
            return ProviderResult(records=[ModelRecord(provider="mini", model_id="m")])

    res = Mini().enrich(Mini().list_models(None), None)
    assert res.model_order == ["m"] and isinstance(Mini(), providers.Provider)


def test_ollama_daemon_not_running_is_a_skip(tmp_path):
    from modelroster.http import FetchError
    from modelroster.update import run_provider

    class Down:
        def get_json(self, *a, **k):
            raise FetchError("connection refused", None)
        def close(self):
            pass

    r = run_provider("ollama", data_dir=tmp_path, http=Down(), quiet=True)
    assert r.skipped and r.ok
