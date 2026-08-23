"""
Provider-agnostic validation gates and drift reporting over the normalized
registry envelope. Providers add their own gates through `Provider.validate`.

validate(current, previous, provider=None) -> (errors, warnings)
    errors block the write; warnings are reported.
diff(previous, current) -> drift report (JSON-serializable)
format_drift(drift) -> text
"""

from __future__ import annotations

from typing import Any

from .schema import CAPABILITY_FIELDS, MODALITIES

SHRINK_MIN_MODELS = 5
SHRINK_FRACTION = 0.5

DIFF_SCALARS = ("display_name", "family", "relationship", "context_window", "max_input_tokens",
                "max_output_tokens", "knowledge_cutoff", "shutdown_date", "deprecated", "default_snapshot")


def flatten(rec: dict) -> dict[str, Any]:
    """Flatten a record dict into {dotted_key: hashable scalar} for diffing."""
    out: dict[str, Any] = {}
    for k in DIFF_SCALARS:
        out[k] = rec.get(k)
    caps = rec.get("capabilities") or {}
    for k in CAPABILITY_FIELDS:
        out["capabilities." + k] = caps.get(k)
    eff = caps.get("reasoning_efforts")
    out["capabilities.reasoning_efforts"] = tuple(eff) if eff is not None else None
    out["capabilities.default_effort"] = caps.get("default_effort")
    for k, v in (caps.get("extra") or {}).items():
        if isinstance(v, (bool, int, str, type(None))):
            out["capabilities.extra." + k] = v
        elif isinstance(v, list):
            out["capabilities.extra." + k] = tuple(v)
    for k, v in (rec.get("endpoints") or {}).items():
        out["endpoints." + k] = v
    tools = rec.get("builtin_tools")
    if tools is not None:
        for k, v in tools.items():
            out["builtin_tools." + k] = v
    for mod in MODALITIES:
        dirs = (rec.get("modalities") or {}).get(mod) or {}
        for d in ("input", "output"):
            out["modalities.%s.%s" % (mod, d)] = dirs.get(d)
    pricing = rec.get("pricing")
    if pricing:
        for k in ("input", "output", "cached_input"):
            out["pricing." + k] = pricing.get(k)
    out["aliases"] = tuple(rec.get("aliases") or ())
    out["snapshots"] = tuple(rec.get("snapshots") or ())
    return out


def fmt(v: Any) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, tuple):
        return "[" + ", ".join(str(x) for x in v) + "]"
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


def validate(current: dict, previous: dict | None, provider: Any = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    models = current.get("models") or {}
    if not models:
        errors.append("%s: listing returned no models" % current.get("provider"))
    for mid, rec in models.items():
        for w in rec.get("warnings") or []:
            warnings.append("%s: %s" % (mid, w))
    warnings.extend(current.get("warnings") or [])
    if previous:
        pm = previous.get("models") or {}
        if len(pm) >= SHRINK_MIN_MODELS and len(models) < len(pm) * SHRINK_FRACTION:
            errors.append("%s: model list shrank from %d to %d — refusing to overwrite" % (
                current.get("provider"), len(pm), len(models)))
    if provider is not None and hasattr(provider, "validate"):
        e, w = provider.validate(current, previous)
        errors.extend(e)
        warnings.extend(w)
    # de-duplicate while preserving order
    seen: set[str] = set()
    warnings = [w for w in warnings if not (w in seen or seen.add(w))]
    return errors, warnings


def diff(previous: dict | None, current: dict) -> dict:
    prev = previous or {}
    pmodels, cmodels = prev.get("models") or {}, current.get("models") or {}

    def snaps(models):
        s = set()
        for rec in models.values():
            s.update(rec.get("snapshots") or [])
        return s

    def families(models):
        return {rec.get("family") for rec in models.values() if rec.get("family")}

    changed = {}
    for mid in sorted(set(pmodels) & set(cmodels)):
        a, b = flatten(pmodels[mid]), flatten(cmodels[mid])
        deltas = [[k, fmt(a.get(k)), fmt(b.get(k))] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        if deltas:
            changed[mid] = deltas

    return {
        "provider": current.get("provider"),
        "first_run": not bool(prev),
        "previous_retrieved_at": prev.get("retrieved_at"),
        "retrieved_at": current.get("retrieved_at"),
        "parser_version": current.get("parser_version"),
        "previous_parser_version": prev.get("parser_version"),
        "counts": {"previous": len(pmodels), "current": len(cmodels)},
        "added_models": sorted(set(cmodels) - set(pmodels)),
        "removed_models": sorted(set(pmodels) - set(cmodels)),
        "added_families": sorted(families(cmodels) - families(pmodels)),
        "removed_families": sorted(families(pmodels) - families(cmodels)),
        "new_snapshots": sorted(snaps(cmodels) - snaps(pmodels)),
        "removed_snapshots": sorted(snaps(pmodels) - snaps(cmodels)),
        "changed_capabilities": changed,
        "warnings": list(current.get("warnings") or []),
    }


def format_drift(d: dict, max_warnings: int = 40) -> str:
    lines = []
    p = d.get("provider")
    lines.append("%s: %d model(s)%s" % (p, d["counts"]["current"],
                                        " (first run — nothing to compare against)" if d.get("first_run")
                                        else " (previously %d)" % d["counts"]["previous"]))

    def block(title, items, sign):
        if items:
            lines.append("")
            lines.append(title)
            for i in items:
                lines.append("  %s %s" % (sign, i))

    block("New models:", d["added_models"], "+")
    block("Removed models:", d["removed_models"], "-")
    block("New families:", d["added_families"], "+")
    block("Removed families:", d["removed_families"], "-")
    block("New snapshots:", d["new_snapshots"], "+")
    block("Removed snapshots:", d["removed_snapshots"], "-")
    if d["changed_capabilities"]:
        lines.append("")
        lines.append("Capability changes:")
        for mid, deltas in d["changed_capabilities"].items():
            lines.append("  %s" % mid)
            for k, a, b in deltas:
                lines.append("    %-36s %s -> %s" % (k + ":", a, b))
    w = d.get("warnings") or []
    if w:
        lines.append("")
        lines.append("Warnings (%d):" % len(w))
        for x in w[:max_warnings]:
            lines.append("  %s" % x)
        if len(w) > max_warnings:
            lines.append("  … %d more (see the drift report / registry warnings)" % (len(w) - max_warnings))
    return "\n".join(lines)
