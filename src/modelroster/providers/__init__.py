"""
Provider registry.

Built-in providers are registered here; third-party providers register via
`register()` or the `modelroster.providers` entry-point group (value: a
zero-argument factory returning a Provider).
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points

from .base import BaseProvider, MissingCredentials, Provider, ProviderError, ProviderResult, SkipProvider, fixture_dir

ENTRY_POINT_GROUP = "modelroster.providers"

# name -> dotted "module:Class" for lazy import
_BUILTIN = {
    "anthropic": "modelroster.providers.anthropic:AnthropicProvider",
    "openai": "modelroster.providers.openai:OpenAIProvider",
    "xai": "modelroster.providers.xai:XAIProvider",
    "mistral": "modelroster.providers.mistral:MistralProvider",
    "google": "modelroster.providers.google:GoogleProvider",
    "cohere": "modelroster.providers.cohere:CohereProvider",
    "nvidia": "modelroster.providers.nvidia:NvidiaProvider",
    "inception": "modelroster.providers.inception:InceptionProvider",
    "ollama": "modelroster.providers.ollama:OllamaProvider",
}
_registry: dict[str, Provider] = {}
_loaded_entry_points = False


def register(provider: Provider, *, replace: bool = False) -> Provider:
    name = provider.name
    if not name:
        raise ValueError("provider has no name")
    if name in _registry and not replace and _registry[name] is not provider:
        raise ValueError("provider %r already registered" % name)
    _registry[name] = provider
    return provider


def _load_entry_points() -> None:
    global _loaded_entry_points
    if _loaded_entry_points:
        return
    _loaded_entry_points = True
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        eps = entry_points().get(ENTRY_POINT_GROUP, [])
    for ep in eps:
        try:
            factory = ep.load()
            prov = factory() if callable(factory) else factory
            register(prov, replace=False)
        except Exception as exc:  # a broken plugin must not take the core down
            import warnings
            warnings.warn("modelroster: could not load provider entry point %r: %s" % (ep.name, exc))


def get(name: str) -> Provider:
    name = name.lower()
    if name in _registry:
        return _registry[name]
    if name in _BUILTIN:
        mod, _, cls = _BUILTIN[name].partition(":")
        prov = getattr(import_module(mod), cls)()
        _registry[name] = prov
        return prov
    _load_entry_points()
    if name in _registry:
        return _registry[name]
    raise KeyError("unknown provider %r (known: %s)" % (name, ", ".join(names())))


def names() -> list[str]:
    _load_entry_points()
    return sorted(set(_BUILTIN) | set(_registry))


def all_providers() -> list[Provider]:
    return [get(n) for n in names()]


__all__ = ["BaseProvider", "MissingCredentials", "SkipProvider", "Provider", "ProviderError", "ProviderResult",
           "register", "get", "names", "all_providers", "fixture_dir", "ENTRY_POINT_GROUP"]
