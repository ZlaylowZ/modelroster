"""Pipeline behaviour: gates, atomic writes, byte-for-byte preservation, exit codes, drift. No network."""

import json

import pytest

from helpers import DOCS, FIXTURES, run
from modelroster.http import FetchError, FixtureFetcher
from modelroster.providers.openai import OpenAIProvider
from modelroster.store import write_json_atomic
from modelroster.update import EXIT_FETCH, EXIT_REFUSED, EXIT_USAGE, run_many, run_provider, worst_code
from modelroster.validate import diff, validate


def test_write_json_atomic_keeps_previous(tmp_path):
    p = tmp_path / "x.json"
    write_json_atomic(p, {"v": 1})
    write_json_atomic(p, {"v": 2})
    assert json.loads(p.read_text())["v"] == 2
    assert json.loads((tmp_path / "x.previous.json").read_text())["v"] == 1
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.usefixtures("fake_keys")
class TestPipeline:
    def test_dry_run_writes_nothing_and_reports_zero_failures(self, data_dir):
        r = run("openai", data_dir, dry_run=True)
        assert r.ok and not r.written
        assert r.registry["stats"]["page_failures"] == 0
        assert not (data_dir / "openai.json").exists()

    def test_update_writes_registry_and_drift(self, data_dir):
        r = run("openai", data_dir)
        assert r.ok and (data_dir / "openai.json").exists() and (data_dir / "openai.drift.json").exists()
        d = json.loads((data_dir / "openai.drift.json").read_text())
        assert d["first_run"] and d["counts"]["current"] == 126

    def test_second_run_keeps_previous_and_reports_no_drift(self, data_dir):
        run("openai", data_dir)
        r = run("openai", data_dir)
        assert (data_dir / "openai.previous.json").exists()
        assert not r.drift["first_run"] and r.drift["added_models"] == [] and r.drift["changed_capabilities"] == {}

    def test_deliberate_parser_break_refuses_and_preserves_data(self, data_dir, tmp_path):
        run("openai", data_dir)
        before = (data_dir / "openai.json").read_bytes()
        # Break the format: rename the H2 that carries every capability table.
        broken = tmp_path / "broken"
        (broken / "openai_docs").mkdir(parents=True)
        (broken / "listings").mkdir()
        for p in DOCS.glob("*.md"):
            (broken / "openai_docs" / p.name).write_text(
                p.read_text("utf-8").replace("## Endpoints", "## Where it runs").replace("## Model details", "## Facts"), "utf-8")
        (broken / "listings" / "openai_models.json").write_bytes((FIXTURES / "listings" / "openai_models.json").read_bytes())
        r = run("openai", data_dir, fixtures_root=broken)
        assert r.code == EXIT_REFUSED
        assert any("PARSER REGRESSION" in e for e in r.errors)
        assert (data_dir / "openai.json").read_bytes() == before

    def test_missing_pages_beyond_threshold_refuse(self, data_dir, tmp_path):
        prov = OpenAIProvider()
        pages = prov.fixtures(FIXTURES)
        keep = {u: p for u, p in pages.items() if "/api/docs/models/" not in u or u.endswith(("gpt-4o.md", "o3.md"))}
        r = run_provider(prov, data_dir=data_dir, http=FixtureFetcher(keep), quiet=True)
        assert r.code == EXIT_REFUSED and any("failed to fetch/parse" in e for e in r.errors)

    def test_shrink_gate(self, data_dir):
        run("anthropic", data_dir)
        from modelroster.providers.anthropic import AnthropicProvider
        url, path = next(iter(AnthropicProvider().fixtures(FIXTURES).items()))
        one = json.loads(path.read_text())["data"][:1]
        small = FixtureFetcher({url: json.dumps({"data": one, "has_more": False})})
        r = run_provider("anthropic", data_dir=data_dir, http=small, quiet=True)
        assert r.code == EXIT_REFUSED and any("shrank" in e for e in r.errors)
        assert json.loads((data_dir / "anthropic.json").read_text())["stats"]["models"] == 10

    def test_fetch_failure_exit_3(self, data_dir):
        class Dead:
            def get_json(self, *a, **k):
                raise FetchError("boom", 503)
            get_text = get_json
            def get_many(self, urls, **k):
                return {u: FetchError("boom") for u in urls}
            def close(self):
                pass
        r = run_provider("anthropic", data_dir=data_dir, http=Dead(), quiet=True)
        assert r.code == EXIT_FETCH and not (data_dir / "anthropic.json").exists()

    def test_unknown_provider_is_usage_error(self, data_dir):
        assert run("nope", data_dir).code == EXIT_USAGE

    def test_one_failure_does_not_block_others_and_worst_code_wins(self, data_dir, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY")   # openai skipped
        results = run_many(["openai", "anthropic", "nope"], data_dir=data_dir, fixtures_root=FIXTURES, quiet=True)
        assert [r.provider for r in results] == ["openai", "anthropic", "nope"]
        assert results[0].skipped and results[1].ok and results[2].code == EXIT_USAGE
        assert worst_code(results) == EXIT_USAGE
        assert (data_dir / "anthropic.json").exists()

    def test_refresh_helper_returns_drift(self, data_dir, monkeypatch):
        import modelroster
        from modelroster import update as upd
        calls = {}

        def fake_run_many(names, **kw):
            calls["names"] = names
            return [run(n, data_dir) for n in names]

        monkeypatch.setattr(upd, "run_many", fake_run_many)
        out = modelroster.refresh(["anthropic"], data_dir=data_dir)
        assert calls["names"] == ["anthropic"]
        assert out["anthropic"]["code"] == 0 and out["anthropic"]["drift"]["provider"] == "anthropic"


def test_generic_validate_and_diff():
    cur = {"provider": "p", "models": {"a": {"capabilities": {"tool_calling": True}, "warnings": ["w"]}}, "warnings": []}
    prev = {"provider": "p", "models": {"a": {"capabilities": {"tool_calling": False}}, "b": {"capabilities": {}}}}
    errors, warnings = validate(cur, prev)
    assert errors == [] and warnings == ["a: w"]
    d = diff(prev, cur)
    assert d["removed_models"] == ["b"]
    assert ["capabilities.tool_calling", "false", "true"] in d["changed_capabilities"]["a"]
    big_prev = {"provider": "p", "models": {str(i): {} for i in range(10)}}
    errors, _ = validate(cur, big_prev)
    assert any("shrank" in e for e in errors)
    assert validate({"provider": "p", "models": {}}, None)[0]


def test_diff_prints_unknown_for_none():
    from modelroster.validate import fmt
    assert fmt(None) == "unknown" and fmt(True) == "true" and fmt(("a", "b")) == "[a, b]"
