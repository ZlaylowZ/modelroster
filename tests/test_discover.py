"""Discovery tier: separate from verified, mostly-None capabilities."""

import pytest

from helpers import FIXTURES
from modelroster import discover
from modelroster.http import FixtureFetcher


@pytest.mark.parametrize("name", discover.sources())
def test_sources_run_from_fixtures(name):
    src = discover.get(name)
    recs = src.discover(FixtureFetcher(src.fixtures(FIXTURES)), limit=50)
    assert 0 < len(recs) <= 50
    for r in recs:
        assert r.tier == "discovered" and r.relationship == "unknown" and r.provider == name
        assert r.capabilities.tool_calling is None and r.capabilities.reasoning is None
        assert any("not verified" in w for w in r.warnings)


def test_huggingface_metadata():
    src = discover.get("huggingface")
    recs = src.discover(FixtureFetcher(src.fixtures(FIXTURES)), limit=50)
    assert recs[0].model_id == "Qwen/Qwen3-0.6B" and recs[0].raw["downloads"] > 0
    assert recs[0].sources["documentation"].startswith("https://huggingface.co/")


def test_nvidia_nim_only_nim_org():
    src = discover.get("nvidia_nim")
    recs = src.discover(FixtureFetcher(src.fixtures(FIXTURES)), limit=100)
    assert recs and all(r.model_id.startswith("nim/") for r in recs)


def test_ollama_library_format_change_is_loud():
    src = discover.get("ollama_library")
    with pytest.raises(ValueError, match="format changed"):
        src.discover(FixtureFetcher({"https://ollama.com/library": "<html>no links</html>"}))


def test_discovered_tier_never_enters_verified_registry(populated, tmp_path, fake_keys):
    from modelroster.cli import main
    from modelroster import load
    assert main(["--data-dir", str(populated), "discover", "huggingface", "--fixtures", str(FIXTURES), "--limit", "50", "--write"]) == 0
    assert (populated / "discovered" / "huggingface.json").exists()
    reg = load(data_dir=populated, force=True)
    assert "huggingface" not in reg.providers()
    assert reg.models(tier="discovered") == []
