"""
Normalized schema shared by every provider.

Design rules:
  * Capabilities are TRI-STATE: True / False / None.  None means "the source
    does not say" and is never collapsed into False.
  * `model_id` is the exact string the provider's API accepts; it is never
    rewritten.
  * Canonical / alias / snapshot relationships come only from explicit
    provider statements, never from guesses on the id string.
  * Every derived fact carries a provenance record.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

# Bump when any parser's output for the same input could change.
PARSER_VERSION = "2026.08.24-1"
SCHEMA_VERSION = 2

MODALITIES = ("text", "image", "audio", "video")
RELATIONSHIPS = ("canonical", "snapshot", "alias", "fine_tune_inherited", "unknown")
TIERS = ("verified", "discovered")

# Reasoning effort values currently documented by any provider; others are
# kept but warned about.
KNOWN_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# OpenAI "## Supported features" bullet keys (observed 2026-08-18).
KNOWN_FEATURE_KEYS = (
    "function_calling", "streaming", "image_input", "structured_outputs", "prompt_caching",
    "file_search", "web_search", "file_uploads", "fine_tuning", "evals", "stored_completions",
    "predicted_outputs", "inpainting", "mcp", "image_generation", "system_messages",
)

# OpenAI "## Supported tools" bullet keys (Responses API built-in tools).
KNOWN_TOOL_KEYS = (
    "web_search", "mcp", "function_calling", "file_search", "code_interpreter", "image_generation",
    "hosted_shell", "skills", "apply_patch", "tool_search", "computer_use",
)

# OpenAI "## Endpoints" table: raw route -> stable key.
ENDPOINT_ROUTE_TO_KEY = {
    "v1/chat/completions": "chat_completions",
    "v1/responses": "responses",
    "v1/realtime": "realtime",
    "v1/realtime/translations": "realtime_translation",
    "v1/realtime/transcription_sessions": "realtime_transcription",
    "v1/assistants": "assistants",
    "v1/batch": "batch",
    "v1/fine-tuning": "fine_tuning",
    "v1/embeddings": "embeddings",
    "v1/images/generations": "image_generation",
    "v1/videos": "videos",
    "v1/images/edits": "image_edit",
    "v1/audio/speech": "speech_generation",
    "v1/audio/transcriptions": "transcription",
    "v1/audio/translations": "translation",
    "v1/moderations": "moderation",
    "v1/completions": "completions_legacy",
}

# Capability names that Registry.models(**filters) accepts directly.
CAPABILITY_FIELDS = (
    "reasoning", "extended_thinking", "tool_calling", "structured_outputs", "streaming",
    "prompt_caching", "fine_tuning", "batch", "citations", "code_execution", "pdf_input",
)


class SchemaError(ValueError):
    """Raised when a record cannot be built from a dict."""


def tri(value: Any) -> bool | None:
    """Coerce to a tri-state bool (True/False/None)."""
    if value is None:
        return None
    return bool(value)


def prov(section: str | None, evidence: str, **extra: Any) -> dict[str, Any]:
    """Provenance record: which document section / API field produced a value."""
    p: dict[str, Any] = {"section": section, "evidence": evidence}
    p.update(extra)
    return p


def empty_modalities() -> dict[str, dict[str, bool | None]]:
    return {m: {"input": None, "output": None} for m in MODALITIES}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── knowledge cutoff parsing ────────────────────────────────────────────

_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_RE_MDY = re.compile(r"^([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})$")
_RE_MY = re.compile(r"^([A-Za-z]{3})[a-z]*\.?\s+(\d{4})$")
_RE_ISO = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")


def parse_cutoff(raw: str | None) -> str | None:
    """
    'Aug 31, 2025' -> '2025-08-31'; 'October 2023' -> '2023-10-01' (first of
    month, precision loss is recorded by keeping the raw text beside it);
    '2024-06' -> '2024-06-01'. Unparseable text -> None (raw text retained by
    the caller).
    """
    if not raw:
        return None
    s = raw.strip().rstrip(".")
    m = _RE_MDY.match(s)
    if m and m.group(1).lower() in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2))).isoformat()
        except ValueError:
            return None
    m = _RE_MY.match(s)
    if m and m.group(1).lower() in _MONTHS:
        return date(int(m.group(2)), _MONTHS[m.group(1).lower()], 1).isoformat()
    m = _RE_ISO.match(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)).isoformat()
        except ValueError:
            return None
    return None


# ── dataclasses ─────────────────────────────────────────────────────────

@dataclass
class Capabilities:
    reasoning: bool | None = None
    reasoning_efforts: list[str] | None = None
    default_effort: str | None = None
    extended_thinking: bool | None = None
    tool_calling: bool | None = None
    structured_outputs: bool | None = None
    streaming: bool | None = None
    prompt_caching: bool | None = None
    fine_tuning: bool | None = None
    batch: bool | None = None
    citations: bool | None = None
    code_execution: bool | None = None
    pdf_input: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        """Tri-state lookup by name; unknown names fall through to `extra`."""
        if name in CAPABILITY_FIELDS or name in ("reasoning_efforts", "default_effort"):
            return getattr(self, name)
        return self.extra.get(name)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Capabilities":
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        extra = dict(d.pop("extra", None) or {})
        for k in list(d):
            if k not in known:
                extra[k] = d.pop(k)
        return cls(extra=extra, **d)


@dataclass
class ModelRecord:
    provider: str
    model_id: str
    display_name: str | None = None
    description: str | None = None
    family: str | None = None                 # canonical id of the documented family
    aliases: list[str] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)
    default_snapshot: str | None = None
    routes_to: str | None = None
    relationship: str = "unknown"
    released: str | None = None               # ISO date
    deprecated: bool | None = None
    shutdown_date: str | None = None
    context_window: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    knowledge_cutoff: str | None = None       # ISO date when parseable
    knowledge_cutoff_raw: str | None = None
    modalities: dict[str, dict[str, bool | None]] = field(default_factory=empty_modalities)
    capabilities: Capabilities = field(default_factory=Capabilities)
    endpoints: dict[str, bool | None] = field(default_factory=dict)
    builtin_tools: dict[str, bool | None] | None = None
    pricing: dict[str, float | None] | None = None
    tier: str = "verified"
    provenance: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    retrieved_at: str | None = None
    parser_version: str = PARSER_VERSION
    warnings: list[str] = field(default_factory=list)

    # ── convenience ──
    def supports(self, capability: str) -> bool | None:
        return self.capabilities.get(capability)

    def modality(self, name: str, direction: str = "input") -> bool | None:
        return (self.modalities.get(name) or {}).get(direction)

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model_id}"

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["capabilities"] = self.capabilities.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelRecord":
        if not isinstance(d, dict) or "provider" not in d or "model_id" not in d:
            raise SchemaError("record needs provider and model_id: %r" % (d,))
        d = dict(d)
        caps = d.pop("capabilities", None)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = {k: d.pop(k) for k in list(d) if k not in known}
        rec = cls(capabilities=Capabilities.from_dict(caps), **d)
        if unknown:
            rec.raw.setdefault("_unknown_fields", {}).update(unknown)
        if rec.relationship not in RELATIONSHIPS:
            rec.warn("unrecognised relationship %r" % rec.relationship)
        return rec


def new_record(provider: str, model_id: str, **kw: Any) -> ModelRecord:
    return ModelRecord(provider=provider, model_id=model_id, **kw)
