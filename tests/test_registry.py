"""Consumer API: Registry, filters (tri-state), predicates, ModelRef. No network."""

import pytest

import modelroster
from modelroster import ModelRef, RetiredModelError, UnknownModelError, load
from modelroster.ref import AmbiguousModelError, infer_provider, is_retired
from modelroster.schema import Capabilities, ModelRecord


def test_load_all_and_info(registry):
    assert set(registry.providers()) == {"anthropic", "openai", "xai", "mistral", "google", "cohere", "nvidia", "inception", "ollama"}
    info = registry.info("openai")
    assert info.models == 126 and info.parser_version == modelroster.PARSER_VERSION
    assert len(registry) == sum(i.models for i in registry.info().values())


def test_load_single_provider(populated):
    reg = load("anthropic", data_dir=populated)
    assert reg.providers() == ["anthropic"] and reg.get("gpt-5.4") is None


def test_load_unknown_provider_raises(populated):
    with pytest.raises(FileNotFoundError):
        load("nope", data_dir=populated)


def test_get_by_id_prefixed_id_and_ref(registry):
    assert registry.get("gpt-5.4").provider == "openai"
    assert registry.get("openai/gpt-5.4").model_id == "gpt-5.4"
    assert registry.get(ModelRef("anthropic", "claude-opus-5")).display_name == "Claude Opus 5"
    assert registry.get("models/gemini-2.5-pro").provider == "google"      # slash inside the id
    assert registry.get("nope") is None and "gpt-5.4" in registry


def test_filters_are_tri_state(registry):
    both = registry.models(tool_calling=True, reasoning=True)
    assert both and all(r.capabilities.tool_calling is True and r.capabilities.reasoning is True for r in both)
    assert {r.provider for r in both} >= {"openai", "anthropic"}
    # None never satisfies True unless unknown_ok
    nv = registry.models(provider="nvidia", tool_calling=True)
    assert nv == []
    assert len(registry.models(provider="nvidia", tool_calling=True, unknown_ok=True)) == 102
    # explicit False and explicit None filters
    assert all(r.capabilities.tool_calling is False for r in registry.models(tool_calling=False))
    assert all(r.capabilities.tool_calling is None for r in registry.models(tool_calling=None))


def test_filter_modalities_endpoints_tools_relationship(registry):
    img = registry.models(provider="openai", image_input=True, endpoint="responses")
    assert "gpt-5.4" in [r.model_id for r in img]
    cu = registry.models(provider="openai", builtin_tool="computer_use")
    assert "gpt-5.4" in [r.model_id for r in cu] and "gpt-4o" not in [r.model_id for r in cu]
    canon = registry.models(provider="openai", relationship="canonical")
    assert all(r.relationship == "canonical" for r in canon)
    strict = registry.models(provider="openai", strict=True)
    assert all(r.relationship != "unknown" for r in strict)
    assert registry.models(provider="openai", extended_thinking=True) == []
    assert registry.models(tier="discovered") == []


def test_retired_excluded_on_request(registry):
    dep = [r for r in registry.models(provider="openai") if r.deprecated]
    assert dep, "fixture listing carries shutdown_date fields"
    assert not [r for r in registry.models(provider="openai", include_retired=False) if r.deprecated]


def test_extra_capability_filter(registry):
    assert [r.model_id for r in registry.models(provider="anthropic", **{"context_management.compact_20260112": True})]


def test_resolve_follows_snapshots_and_aliases(registry):
    assert registry.resolve("gpt-4o-2024-08-06").model_id == "gpt-4o"
    assert registry.resolve("grok-4.3-latest").model_id == "grok-4.3"
    assert registry.resolve("claude-opus-5").model_id == "claude-opus-5"


def test_predicates_are_provider_agnostic(populated, monkeypatch):
    monkeypatch.setenv("MODELROSTER_DATA_DIR", str(populated))
    modelroster.clear_cache()
    assert modelroster.supports_tool_calling("gpt-5.4") is True
    assert modelroster.supports_tool_calling("claude-opus-5") is True
    assert modelroster.supports_tool_calling("gpt-3.5-turbo") is False
    assert modelroster.supports_tool_calling("text-embedding-3-small") is None
    assert modelroster.supports_tool_calling("never-heard-of-it") is None
    assert modelroster.supports_reasoning("gpt-4o") is False and modelroster.supports_reasoning("claude-opus-5") is True
    assert modelroster.supported_reasoning_efforts("gpt-5.4") == ["none", "low", "medium", "high", "xhigh"]
    assert modelroster.supported_reasoning_efforts("claude-opus-5") == ["low", "medium", "high", "xhigh", "max"]
    assert modelroster.supports_builtin_tool("gpt-5.4", "computer_use") is True
    assert modelroster.supports_builtin_tool("gpt-3.5-turbo", "computer_use") is None
    assert modelroster.supports_endpoint("text-embedding-3-small", "chat_completions") is False
    assert modelroster.context_window("gpt-5.4") == 1_050_000 and modelroster.max_output_tokens("gpt-5.4") == 128_000
    assert modelroster.context_window("claude-opus-5") == 1_000_000 and modelroster.max_output_tokens("claude-opus-5") == 128_000
    assert modelroster.supports_modality("gpt-5.4", "image") is True
    assert "gpt-5.4" in modelroster.models_supporting_tool_calling("openai")
    assert "gpt-4o" not in modelroster.models_supporting_reasoning("openai")
    assert modelroster.models_supporting_builtin_tool("computer_use", "openai")
    assert modelroster.supports("claude-opus-5", "pdf_input") is True
    assert modelroster.get("gpt-5.4").family == "gpt-5.4"


# ── ModelRef ──

def test_modelref_parse_forms(registry):
    assert ModelRef.parse("openai/gpt-5.4", registry) == ModelRef("openai", "gpt-5.4")
    r = ModelRef.parse("gpt-5.4", registry)
    assert r.provider == "openai" and not r.inferred
    assert ModelRef.parse("models/gemini-2.5-pro", registry) == ModelRef("google", "models/gemini-2.5-pro")
    assert ModelRef.parse("meta/llama-3.1-8b-instruct", registry).provider == "nvidia"
    assert str(ModelRef("a", "b")) == "a/b"
    assert ModelRef.parse(ModelRef("a", "b"), registry) == ModelRef("a", "b")


def test_modelref_heuristic_is_fallback_and_marked(registry):
    r = ModelRef.parse("gpt-99", registry)
    assert r.provider == "openai" and r.inferred
    assert not r.is_valid(registry)
    with pytest.raises(UnknownModelError):
        ModelRef.parse("totally-unknown", registry)
    with pytest.raises(UnknownModelError):
        ModelRef.parse("totally-unknown", registry, heuristics=True)
    assert ModelRef.parse("totally-unknown", registry, default_provider="acme") == ModelRef("acme", "totally-unknown")
    assert infer_provider("claude-x") == "anthropic" and infer_provider("zzz") is None


def test_modelref_validate_and_resolve(registry):
    assert ModelRef("openai", "gpt-5.4").validate(registry) == ModelRef("openai", "gpt-5.4")
    with pytest.raises(UnknownModelError, match="not a known model id"):
        ModelRef("openai", "claude-opus-5").validate(registry)
    with pytest.raises(UnknownModelError, match="unknown provider"):
        ModelRef("acme", "x").validate(registry)
    assert ModelRef("openai", "gpt-4o-2024-08-06").resolve(registry).model_id == "gpt-4o"
    assert registry.ref("claude-opus-5") == ModelRef("anthropic", "claude-opus-5")


def test_modelref_retired():
    rec = ModelRecord(provider="p", model_id="m", deprecated=True, shutdown_date="2000-01-01")
    assert is_retired(rec)
    future = ModelRecord(provider="p", model_id="m", deprecated=True, shutdown_date="2999-01-01")
    assert not is_retired(future)
    assert not is_retired(ModelRecord(provider="p", model_id="m"))
    from modelroster.registry import Registry
    env = {"p": {"models": {"m": rec.to_dict()}, "model_order": ["m"]}}
    reg = Registry(env)
    with pytest.raises(RetiredModelError):
        ModelRef("p", "m").validate(reg)
    assert ModelRef("p", "m").validate(reg, allow_retired=True)


def test_modelref_ambiguous():
    from modelroster.registry import Registry
    env = {p: {"models": {"shared": ModelRecord(provider=p, model_id="shared").to_dict()}} for p in ("a", "b")}
    reg = Registry(env)
    with pytest.raises(AmbiguousModelError):
        ModelRef.parse("shared", reg)
    assert ModelRef.parse("shared", reg, default_provider="b").provider == "b"
    assert ModelRef.parse("a/shared", reg).provider == "a"


def test_record_round_trip_and_unknown_fields():
    rec = ModelRecord(provider="p", model_id="m", capabilities=Capabilities(tool_calling=True, extra={"x": None}))
    d = rec.to_dict()
    d["capabilities"]["novel_cap"] = True
    d["novel_field"] = 1
    back = ModelRecord.from_dict(d)
    assert back.capabilities.tool_calling is True and back.capabilities.extra == {"x": None, "novel_cap": True}
    assert back.raw["_unknown_fields"] == {"novel_field": 1}
    assert back.supports("tool_calling") is True and back.supports("nothing") is None
    assert back.ref == "p/m"


# ── alias/snapshot index (documented ids the listing does not carry) ──

def test_documented_alias_absent_from_listing_resolves(registry):
    # "gpt-5.6" is gpt-5.6-sol's prose alias and is NOT an id in the /v1/models fixture
    assert "gpt-5.6" not in [r.model_id for r in registry.models("openai")]
    rec = registry.resolve("gpt-5.6")
    assert rec is not None and rec.model_id == "gpt-5.6-sol" and rec.relationship == "canonical"
    assert registry.get("gpt-5.6").model_id == "gpt-5.6-sol"
    assert registry.get("openai/gpt-5.6").model_id == "gpt-5.6-sol"
    assert registry.alias_owner("gpt-5.6", "openai") == ("gpt-5.6-sol", "alias")
    assert "openai" in registry.find_providers("gpt-5.6")
    # and it now parses/validates as a ModelRef
    assert ModelRef.parse("gpt-5.6", registry) == ModelRef("openai", "gpt-5.6")
    assert ModelRef("openai", "gpt-5.6").validate(registry)
    assert ModelRef("openai", "gpt-5.6").resolve(registry).model_id == "gpt-5.6-sol"


def test_snapshot_id_resolves_to_family_record(registry):
    # snapshot that IS in the listing: resolves through its own record's family link
    fam = registry.resolve("gpt-4o-2024-11-20")
    assert fam.model_id == "gpt-4o" and fam.relationship == "canonical"
    # documented snapshot that is NOT in the listing: resolves through the index
    listed = {r.model_id for r in registry.models("openai")}
    assert "gpt-4-0314" not in listed
    assert registry.resolve("gpt-4-0314").model_id == "gpt-4"


def test_exact_record_always_beats_alias_claim(registry):
    # daybreak-blue-latest's Snapshots list claims gpt-5.6-sol, which has its own page
    assert registry.get("gpt-5.6-sol").relationship == "canonical"
    assert registry.resolve("gpt-5.6-sol").model_id == "gpt-5.6-sol"


def test_fine_tune_id_resolves_via_base(registry):
    rec = registry.resolve("ft:gpt-4o-2024-08-06:acme::abc123")
    assert rec.model_id == "gpt-4o"
    assert registry.get("ft:gpt-4o-2024-08-06:acme::abc123").family == "gpt-4o"
    assert registry.resolve("ft:mystery-9:acme::zzz") is None


def test_alias_index_conflict_is_deterministic():
    from modelroster.registry import Registry
    def rec(mid, snaps):
        return ModelRecord(provider="p", model_id=mid, family=mid, relationship="canonical", snapshots=snaps).to_dict()
    env = {"p": {"models": {"b-stable": rec("b-stable", ["shared-007"]), "a-latest": rec("a-latest", ["shared-007"])},
                 "model_order": ["b-stable", "a-latest"]}}
    reg = Registry(env)
    assert reg.alias_owner("shared-007", "p") == ("a-latest", "snapshot")   # lexicographically first owner
    assert reg.resolve("shared-007").model_id == "a-latest"
    # a claimed id that is itself a record is never indexed
    env["p"]["models"]["shared-007"] = rec("shared-007", [])
    reg2 = Registry(env)
    assert reg2.alias_owner("shared-007", "p") is None
    assert reg2.resolve("shared-007").model_id == "shared-007"


def test_snapshot_claim_outranks_alias_claim_from_same_owner():
    from modelroster.registry import Registry
    d = ModelRecord(provider="p", model_id="fam", family="fam", relationship="canonical",
                    aliases=["x"], snapshots=["x"]).to_dict()
    reg = Registry({"p": {"models": {"fam": d}, "model_order": ["fam"]}})
    assert reg.alias_owner("x", "p") == ("fam", "snapshot")
