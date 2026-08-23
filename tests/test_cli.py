"""CLI contract: subcommands, exit codes."""

import json

import pytest

from helpers import FIXTURES
from modelroster.cli import main


@pytest.mark.usefixtures("fake_keys")
class TestCLI:
    def test_update_dry_run_fixtures_exit_0(self, data_dir, capsys):
        assert main(["--data-dir", str(data_dir), "update", "--provider", "openai", "--dry-run", "--fixtures", str(FIXTURES)]) == 0
        out = capsys.readouterr().out
        assert "openai: ok, 126 model(s) (not written)" in out
        assert not (data_dir / "openai.json").exists()

    def test_update_all_then_list_show_diff_validate_emit(self, data_dir, capsys):
        assert main(["--data-dir", str(data_dir), "update", "--fixtures", str(FIXTURES), "-q"]) == 0
        assert main(["--data-dir", str(data_dir), "list", "--provider", "openai", "-c", "reasoning", "-c", "tool_calling"]) == 0
        out = capsys.readouterr().out
        assert "gpt-5.4" in out and "gpt-4o " not in out
        assert main(["--data-dir", str(data_dir), "list", "-c", "tool_calling", "--json"]) == 0
        assert all(r["capabilities"]["tool_calling"] for r in json.loads(capsys.readouterr().out))
        assert main(["--data-dir", str(data_dir), "show", "gpt-5.4", "--provenance"]) == 0
        out = capsys.readouterr().out
        assert "reasoning_efforts:   none, low" in out and "Supported features" in out
        assert main(["--data-dir", str(data_dir), "show", "anthropic/claude-opus-5", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["capabilities"]["tool_calling"] is True
        assert main(["--data-dir", str(data_dir), "diff", "--provider", "openai"]) == 0
        assert "first run" in capsys.readouterr().out
        assert main(["--data-dir", str(data_dir), "validate"]) == 0
        assert main(["--data-dir", str(data_dir), "emit", "--out", str(data_dir / "snap.py"), "-c", "tool_calling=true"]) == 0
        assert (data_dir / "snap.py").exists()
        assert main(["--data-dir", str(data_dir), "providers", "-v"]) == 0
        assert "anthropic" in capsys.readouterr().out

    def test_validate_detects_corruption(self, data_dir, capsys):
        main(["--data-dir", str(data_dir), "update", "--provider", "anthropic", "--fixtures", str(FIXTURES), "-q"])
        p = data_dir / "anthropic.json"
        env = json.loads(p.read_text())
        env["models"] = {}
        p.write_text(json.dumps(env))
        assert main(["--data-dir", str(data_dir), "validate", "--provider", "anthropic"]) == 2

    def test_usage_errors_exit_4(self, data_dir, capsys):
        assert main(["--data-dir", str(data_dir), "update", "--provider", "nope", "--fixtures", str(FIXTURES), "-q"]) == 4
        assert main(["bogus"]) == 4
        main(["--data-dir", str(data_dir), "update", "--provider", "openai", "--fixtures", str(FIXTURES), "-q"])
        assert main(["--data-dir", str(data_dir), "show", "does-not-exist"]) == 4
        assert main(["--data-dir", str(data_dir), "list", "-c", "reasoning=maybe"]) == 4

    def test_parser_break_exit_2_via_cli(self, data_dir, tmp_path):
        main(["--data-dir", str(data_dir), "update", "--provider", "openai", "--fixtures", str(FIXTURES), "-q"])
        before = (data_dir / "openai.json").read_bytes()
        broken = tmp_path / "broken"
        (broken / "openai_docs").mkdir(parents=True)
        (broken / "listings").mkdir()
        for p in (FIXTURES / "openai_docs").glob("*.md"):
            (broken / "openai_docs" / p.name).write_text(p.read_text("utf-8").replace("## Endpoints", "## Routes").replace("## Model details", "## Details"), "utf-8")
        (broken / "listings" / "openai_models.json").write_bytes((FIXTURES / "listings" / "openai_models.json").read_bytes())
        assert main(["--data-dir", str(data_dir), "update", "--provider", "openai", "--fixtures", str(broken), "-q"]) == 2
        assert (data_dir / "openai.json").read_bytes() == before

    def test_version(self, capsys):
        assert main(["--version"]) == 0
        assert "modelroster" in capsys.readouterr().out
