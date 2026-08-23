"""
`ModelRef` — a typed model identifier: the "type annotation for model names".

    ModelRef("openai", "gpt-5.4")
    ModelRef.parse("openai/gpt-5.4")
    ModelRef.parse("gpt-5.4")                    # provider found in the registry
    ModelRef.parse("claude-opus-5").validate()   # raises UnknownModelError / RetiredModelError

Provider inference is an exact lookup across the loaded registry. Only when
the registry has never heard of the id does a *documented* prefix heuristic
run (`gpt-`/`o1`…→openai, `claude-`→anthropic, `grok-`→xai, `mistral-`/
`codestral`…→mistral, `gemini-`→google, `command-`→cohere, `mercury`→inception);
a ref produced that way has `inferred=True` and `validate()` still fails,
because the heuristic is a convenience for error messages, not a source of
truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .registry import Registry
    from .schema import ModelRecord


class UnknownModelError(LookupError):
    pass


class RetiredModelError(LookupError):
    pass


class AmbiguousModelError(LookupError):
    pass


PREFIX_HEURISTICS: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"), ("chatgpt-", "openai"),
    ("text-embedding-", "openai"), ("dall-e", "openai"), ("whisper", "openai"), ("tts-", "openai"),
    ("ft:", "openai"), ("claude-", "anthropic"), ("grok-", "xai"), ("mistral-", "mistral"),
    ("codestral", "mistral"), ("ministral", "mistral"), ("magistral", "mistral"), ("pixtral", "mistral"),
    ("devstral", "mistral"), ("open-mistral", "mistral"), ("open-mixtral", "mistral"),
    ("gemini-", "google"), ("models/gemini", "google"), ("command", "cohere"), ("embed-", "cohere"),
    ("rerank-", "cohere"), ("mercury", "inception"),
)


def infer_provider(model_id: str) -> str | None:
    low = model_id.lower()
    for prefix, provider in PREFIX_HEURISTICS:
        if low.startswith(prefix):
            return provider
    return None


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model_id: str
    inferred: bool = field(default=False, compare=False)

    def __str__(self) -> str:
        return "%s/%s" % (self.provider, self.model_id)

    @classmethod
    def parse(cls, value: "str | ModelRef", registry: "Registry | None" = None,
              default_provider: str | None = None, heuristics: bool = True) -> "ModelRef":
        """
        'provider/model_id' -> ModelRef; 'model_id' -> provider looked up in
        the registry (loaded on demand), then `default_provider`, then the
        documented prefix heuristics (marked `inferred=True`).
        """
        if isinstance(value, ModelRef):
            return value
        if not isinstance(value, str) or not value.strip():
            raise UnknownModelError("empty model reference")
        value = value.strip()
        if registry is None:
            from .registry import load
            try:
                registry = load()
            except FileNotFoundError:
                registry = None
        if "/" in value:
            head, _, tail = value.partition("/")
            if registry is None:
                return cls(head, tail)
            if head in registry.providers():
                return cls(head, tail)
            # ids like "models/gemini-2.5-pro" or "meta/llama-3.1-8b-instruct"
            # contain slashes themselves: fall through to a whole-string lookup.
        if registry is not None:
            provs = registry.find_providers(value)
            if len(provs) == 1:
                return cls(provs[0], value)
            if len(provs) > 1:
                if default_provider in provs:
                    return cls(default_provider, value)
                raise AmbiguousModelError("%r exists in several providers: %s — use provider/model_id" % (value, ", ".join(provs)))
        if default_provider:
            return cls(default_provider, value)
        if heuristics:
            guess = infer_provider(value)
            if guess:
                return cls(guess, value, inferred=True)
        raise UnknownModelError("cannot determine the provider of %r; use provider/model_id" % value)

    def record(self, registry: "Registry | None" = None) -> "ModelRecord | None":
        registry = registry or _default_registry()
        return registry.get(self) if registry is not None else None

    def resolve(self, registry: "Registry | None" = None) -> "ModelRecord":
        """The canonical record behind this id (follows aliases/snapshots). Raises if unknown."""
        registry = registry or _default_registry()
        self.validate(registry)
        return registry.resolve(self)

    def validate(self, registry: "Registry | None" = None, allow_retired: bool = False) -> "ModelRef":
        registry = registry or _default_registry()
        if registry is None:
            raise UnknownModelError("no registry data available to validate %s" % self)
        if self.provider not in registry.providers():
            raise UnknownModelError("unknown provider %r in %s (known: %s)" % (
                self.provider, self, ", ".join(registry.providers())))
        rec = registry.get(self)
        if rec is None:
            hint = ""
            others = registry.find_providers(self.model_id)
            if others:
                hint = " (but it exists under %s)" % ", ".join(others)
            raise UnknownModelError("%s is not a known model id for provider %r%s" % (self, self.provider, hint))
        if not allow_retired and is_retired(rec):
            raise RetiredModelError("%s is retired%s" % (
                self, " (shut down %s)" % rec.shutdown_date if rec.shutdown_date else ""))
        return self

    def is_valid(self, registry: "Registry | None" = None) -> bool:
        try:
            self.validate(registry)
            return True
        except LookupError:
            return False


def is_retired(rec: "ModelRecord", today: str | None = None) -> bool:
    """Deprecated with no shutdown date, or a shutdown date that has passed."""
    if rec.deprecated is not True:
        return False
    if not rec.shutdown_date:
        return True
    today = today or date.today().isoformat()
    return rec.shutdown_date[:10] <= today


def _default_registry() -> "Registry | None":
    from .registry import load
    try:
        return load()
    except FileNotFoundError:
        return None
