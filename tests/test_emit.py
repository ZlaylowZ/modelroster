"""Generated-module emitter."""

from modelroster.emit import compile_check, emit, render_module


def test_rendered_module_compiles_and_round_trips(registry, tmp_path):
    text = render_module(registry, ["openai", "anthropic"], tool_calling=True)
    ns = compile_check(text)
    assert ns["OPENAI_MODELS"]["gpt-5.4"]["context_window"] == 1_050_000
    assert ns["OPENAI_MODELS"]["gpt-5.4"]["reasoning_efforts"] == ("none", "low", "medium", "high", "xhigh")
    assert "gpt-3.5-turbo" not in ns["OPENAI_MODELS"]            # tool_calling False filtered out
    assert ns["ANTHROPIC_MODELS"]["claude-opus-5"]["max_output_tokens"] == 128_000
    assert ns["MODELS"]["openai/gpt-5.4"]["provider"] == "openai"
    assert ns["REFRESHED"]["openai"]
    assert "DO NOT EDIT" in text and "nukez" not in text.lower()
    out = tmp_path / "snap.py"
    emit(registry, out, ["openai"])
    assert out.exists() and not list(tmp_path.glob("*.tmp"))


def test_none_stays_none_in_snapshot(registry):
    ns = compile_check(render_module(registry, ["openai"]))
    assert ns["OPENAI_MODELS"]["text-embedding-3-small"]["tool_calling"] is None
