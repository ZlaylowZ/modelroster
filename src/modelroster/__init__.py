"""
modelroster — accurate, current LLM model identifiers and capabilities for
every provider, shipped as data, refreshed from official sources.

    import modelroster
    r = modelroster.load()
    r.models(tool_calling=True, reasoning=True)
    modelroster.ModelRef.parse("openai/gpt-5.4").validate()
    modelroster.context_window("claude-opus-5")

Every capability is tri-state: True / False / None, where None means "the
source does not say" — never "no".
"""

from __future__ import annotations

from typing import Any

from ._version import __version__
from .ref import AmbiguousModelError, ModelRef, RetiredModelError, UnknownModelError, is_retired
from .registry import Registry, clear_cache, load
from .schema import CAPABILITY_FIELDS, PARSER_VERSION, SCHEMA_VERSION, Capabilities, ModelRecord


def _rec(model: str | ModelRef | ModelRecord, provider: str | None = None) -> ModelRecord | None:
    if isinstance(model, ModelRecord):
        return model
    try:
        return load().get(model, provider)
    except FileNotFoundError:
        return None


def get(model: str | ModelRef, provider: str | None = None) -> ModelRecord | None:
    """Exact-id lookup across the shipped registry."""
    return _rec(model, provider)


def supports(model: str | ModelRef | ModelRecord, capability: str, provider: str | None = None) -> bool | None:
    """Tri-state: True / False / None (unknown model or undocumented capability)."""
    rec = _rec(model, provider)
    return None if rec is None else rec.supports(capability)


def supports_tool_calling(model, provider=None):
    return supports(model, "tool_calling", provider)


def supports_reasoning(model, provider=None):
    return supports(model, "reasoning", provider)


def supports_structured_outputs(model, provider=None):
    return supports(model, "structured_outputs", provider)


def supports_streaming(model, provider=None):
    return supports(model, "streaming", provider)


def supported_reasoning_efforts(model, provider=None) -> list[str] | None:
    rec = _rec(model, provider)
    return None if rec is None else rec.capabilities.reasoning_efforts


def supports_endpoint(model, endpoint: str, provider=None) -> bool | None:
    rec = _rec(model, provider)
    return None if rec is None else rec.endpoints.get(endpoint)


def supports_builtin_tool(model, tool: str, provider=None) -> bool | None:
    rec = _rec(model, provider)
    if rec is None or rec.builtin_tools is None:
        return None
    return rec.builtin_tools.get(tool)


def supports_modality(model, modality: str, direction: str = "input", provider=None) -> bool | None:
    rec = _rec(model, provider)
    return None if rec is None else rec.modality(modality, direction)


def context_window(model, provider=None) -> int | None:
    rec = _rec(model, provider)
    return None if rec is None else rec.context_window


def max_output_tokens(model, provider=None) -> int | None:
    rec = _rec(model, provider)
    return None if rec is None else rec.max_output_tokens


def max_input_tokens(model, provider=None) -> int | None:
    rec = _rec(model, provider)
    return None if rec is None else rec.max_input_tokens


def models_supporting(capability: str, provider: str | None = None, **kw: Any) -> list[str]:
    """Ids whose source documents support for `capability` (None never counts)."""
    return load().ids(provider, **{capability: True}, **kw)


def models_supporting_tool_calling(provider=None):
    return models_supporting("tool_calling", provider)


def models_supporting_reasoning(provider=None):
    return models_supporting("reasoning", provider)


def models_supporting_endpoint(endpoint: str, provider=None) -> list[str]:
    return load().ids(provider, endpoint=endpoint)


def models_supporting_builtin_tool(tool: str, provider=None) -> list[str]:
    return load().ids(provider, builtin_tool=tool)


def available_models(provider: str | None = None) -> list[str]:
    return load().ids(provider)


def info(provider: str | None = None):
    return load().info(provider)


def refresh(providers: list[str] | None = None, **kw: Any) -> dict[str, dict]:
    """Refresh registry data from the providers' official sources; returns per-provider drift/status."""
    from .update import refresh as _refresh
    out = _refresh(providers, **kw)
    clear_cache()
    return out


__all__ = [
    "__version__", "PARSER_VERSION", "SCHEMA_VERSION", "CAPABILITY_FIELDS",
    "Registry", "ModelRecord", "Capabilities", "ModelRef", "load", "clear_cache", "get",
    "UnknownModelError", "RetiredModelError", "AmbiguousModelError", "is_retired",
    "supports", "supports_tool_calling", "supports_reasoning", "supports_structured_outputs",
    "supports_streaming", "supported_reasoning_efforts", "supports_endpoint", "supports_builtin_tool",
    "supports_modality", "context_window", "max_output_tokens", "max_input_tokens",
    "models_supporting", "models_supporting_tool_calling", "models_supporting_reasoning",
    "models_supporting_endpoint", "models_supporting_builtin_tool", "available_models", "info", "refresh",
]
