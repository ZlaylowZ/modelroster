"""
modelroster command-line interface.

    modelroster update [--provider X ...|--all] [--offline] [--dry-run] [--data-dir D] [--fixtures DIR]
    modelroster list [--provider X] [--capability reasoning --capability tool_calling ...] [--json]
    modelroster show <model_id|provider/model_id> [--json]
    modelroster diff [--provider X]
    modelroster validate [--provider X] [--data-dir D]
    modelroster emit --out FILE [--provider X ...] [--capability ...]
    modelroster discover <source> [--limit N] [--json]
    modelroster providers
    modelroster capture --provider X --out DIR      # save live listing responses as fixtures

Exit status: 0 ok · 2 validation refused (previous data preserved) · 3 fetch
failure · 4 usage error. One provider's failure never blocks the others; the
exit status is the worst stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__, providers as provider_registry
from .ref import ModelRef
from .registry import load
from .schema import CAPABILITY_FIELDS
from .store import available_providers, data_dir as resolve_data_dir, drift_path, read_json, registry_path
from .update import EXIT_OK, EXIT_REFUSED, EXIT_USAGE, load_dotenv_if_available, run_provider, worst_code
from .validate import format_drift, validate


def _parse_capability_filters(values: list[str] | None) -> dict[str, Any]:
    """['reasoning', 'tool_calling=true', 'streaming=false', 'batch=unknown'] -> filters."""
    out: dict[str, Any] = {}
    for v in values or []:
        name, _, val = v.partition("=")
        val = val.strip().lower()
        if val in ("", "true", "yes", "1"):
            out[name.strip()] = True
        elif val in ("false", "no", "0"):
            out[name.strip()] = False
        elif val in ("none", "unknown", "null"):
            out[name.strip()] = None
        else:
            raise argparse.ArgumentTypeError("capability filter %r must be name[=true|false|unknown]" % v)
    return out


def _tri(v: Any) -> str:
    return "unknown" if v is None else ("yes" if v is True else ("no" if v is False else str(v)))


def cmd_update(args: argparse.Namespace) -> int:
    load_dotenv_if_available()
    names = args.provider or provider_registry.names()
    results = []
    for n in names:
        results.append(run_provider(n, data_dir=args.data_dir, offline=args.offline, no_cache=args.no_cache,
                                    dry_run=args.dry_run, fixtures_root=args.fixtures, quiet=args.quiet))
    print()
    for r in results:
        print(r.summary())
    code = worst_code(results)
    if args.emit and code == EXIT_OK and not args.dry_run:
        from .emit import emit
        emit(load(data_dir=args.data_dir, force=True), args.emit)
        print("snapshot module written -> %s" % args.emit)
    return code


def cmd_list(args: argparse.Namespace) -> int:
    filters = _parse_capability_filters(args.capability)
    reg = load(args.provider, data_dir=args.data_dir) if args.provider else load(data_dir=args.data_dir)
    tier = None if args.all_tiers else ("discovered" if args.discovered else "verified")
    recs = reg.models(args.provider, unknown_ok=args.unknown_ok, include_retired=not args.exclude_retired,
                      tier=tier, endpoint=args.endpoint, **filters)
    if args.json:
        print(json.dumps([r.to_dict() for r in recs], indent=1))
        return EXIT_OK
    if not recs:
        print("no models match", file=sys.stderr)
        return EXIT_OK
    width = max(len(r.model_id) for r in recs)
    print("%-10s %-*s %-9s %-9s %-9s %-9s %-9s %s" % ("provider", width, "model_id", "reason", "tools", "struct", "stream", "context", "rel"))
    for r in recs:
        c = r.capabilities
        print("%-10s %-*s %-9s %-9s %-9s %-9s %-9s %s" % (
            r.provider, width, r.model_id, _tri(c.reasoning), _tri(c.tool_calling), _tri(c.structured_outputs),
            _tri(c.streaming), r.context_window if r.context_window is not None else "unknown", r.relationship))
    print("\n%d model(s)" % len(recs))
    return EXIT_OK


def cmd_show(args: argparse.Namespace) -> int:
    reg = load(data_dir=args.data_dir)
    try:
        ref = ModelRef.parse(args.model, registry=reg)
    except LookupError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_USAGE
    rec = reg.get(ref)
    if rec is None:
        print("error: %s is not in the registry%s" % (ref, " (provider guessed from the id prefix)" if ref.inferred else ""), file=sys.stderr)
        return EXIT_USAGE
    if args.json:
        print(json.dumps(rec.to_dict(), indent=1))
        return EXIT_OK
    print("%s/%s" % (rec.provider, rec.model_id))
    for k in ("display_name", "description", "family", "relationship", "aliases", "snapshots", "default_snapshot",
              "routes_to", "released", "deprecated", "shutdown_date", "context_window", "max_input_tokens",
              "max_output_tokens", "knowledge_cutoff", "knowledge_cutoff_raw", "tier", "retrieved_at", "parser_version"):
        v = getattr(rec, k)
        if v not in (None, [], ""):
            print("  %-20s %s" % (k + ":", v))
    print("  capabilities:")
    for k in CAPABILITY_FIELDS:
        print("    %-20s %s" % (k + ":", _tri(rec.capabilities.get(k))))
    if rec.capabilities.reasoning_efforts is not None:
        print("    %-20s %s (default %s)" % ("reasoning_efforts:", ", ".join(rec.capabilities.reasoning_efforts), rec.capabilities.default_effort))
    for k, v in rec.capabilities.extra.items():
        print("    %-20s %s" % (k + ":", _tri(v)))
    print("  modalities:")
    for m, d in rec.modalities.items():
        print("    %-20s in=%s out=%s" % (m + ":", _tri(d.get("input")), _tri(d.get("output"))))
    if rec.endpoints:
        print("  endpoints:")
        for k, v in rec.endpoints.items():
            print("    %-20s %s" % (k + ":", _tri(v)))
    if rec.builtin_tools:
        print("  builtin_tools:")
        for k, v in rec.builtin_tools.items():
            print("    %-20s %s" % (k + ":", _tri(v)))
    if rec.pricing:
        print("  pricing (USD / 1M tokens): input=%s output=%s cached_input=%s" % (
            rec.pricing.get("input"), rec.pricing.get("output"), rec.pricing.get("cached_input")))
    print("  sources:")
    for k, v in rec.sources.items():
        print("    %-20s %s" % (k + ":", v))
    if args.provenance:
        print("  provenance:")
        for k, v in rec.provenance.items():
            print("    %-28s %s" % (k + ":", json.dumps(v)))
    if rec.warnings:
        print("  warnings:")
        for w in rec.warnings:
            print("    - %s" % w)
    return EXIT_OK


def cmd_diff(args: argparse.Namespace) -> int:
    names = args.provider or available_providers(args.data_dir)
    found = False
    for n in names:
        d = read_json(drift_path(n, args.data_dir))
        if d is None:
            print("%s: no drift report (run `modelroster update --provider %s`)" % (n, n))
            continue
        found = True
        if args.json:
            print(json.dumps(d, indent=1))
        else:
            print(format_drift(d))
            print()
    return EXIT_OK if found or not names else EXIT_USAGE


def cmd_validate(args: argparse.Namespace) -> int:
    names = args.provider or available_providers(args.data_dir)
    worst = EXIT_OK
    for n in names:
        cur = read_json(registry_path(n, args.data_dir))
        if cur is None:
            print("%s: no data" % n)
            worst = max(worst, EXIT_USAGE)
            continue
        prev = read_json(registry_path(n, args.data_dir).with_name(n + ".previous.json"))
        try:
            prov = provider_registry.get(n)
        except KeyError:
            prov = None
        errors, warnings = validate(cur, prev, prov)
        n_models = len(cur.get("models") or {})
        print("%s: %d model(s), retrieved %s, parser %s — %s" % (
            n, n_models, cur.get("retrieved_at"), cur.get("parser_version"),
            "OK" if not errors else "INVALID"))
        for e in errors:
            print("  ERROR: %s" % e)
        if args.verbose:
            for w in warnings:
                print("  warning: %s" % w)
        elif warnings:
            print("  %d warning(s) (use -v to list)" % len(warnings))
        if errors:
            worst = max(worst, EXIT_REFUSED)
    return worst


def cmd_emit(args: argparse.Namespace) -> int:
    from .emit import emit
    filters = _parse_capability_filters(args.capability)
    reg = load(data_dir=args.data_dir)
    try:
        ns = emit(reg, args.out, args.provider, **filters)
    except Exception as exc:
        print("error: generated module does not compile: %s" % exc, file=sys.stderr)
        return EXIT_REFUSED
    print("wrote %s (%d models)" % (args.out, len(ns.get("MODELS") or {})))
    return EXIT_OK


def cmd_discover(args: argparse.Namespace) -> int:
    from . import discover
    from .http import Fetcher, FixtureFetcher
    from .store import write_json_atomic
    from .update import build_envelope
    from .providers.base import ProviderResult
    src = discover.get(args.source)
    if args.fixtures:
        http = FixtureFetcher(src.fixtures(Path(args.fixtures)) or {})
    else:
        http = Fetcher(resolve_data_dir(args.data_dir) / "cache" / "discover" / src.name, offline=args.offline)
    try:
        recs = src.discover(http, limit=args.limit)
    finally:
        http.close()
    if args.json:
        print(json.dumps([r.to_dict() for r in recs], indent=1))
    else:
        for r in recs:
            print("%-40s %s" % (r.model_id, r.display_name or ""))
        print("\n%d candidate(s) from %s (tier: discovered, not verified)" % (len(recs), src.name))
    if args.write:
        env = build_envelope(src.name, ProviderResult(records=recs, sources={"listing": src.describe}))
        env["tier"] = "discovered"
        out = resolve_data_dir(args.data_dir) / "discovered" / (src.name + ".json")
        write_json_atomic(out, env, keep_previous=False)
        print("written -> %s" % out)
    return EXIT_OK


def cmd_providers(args: argparse.Namespace) -> int:
    have = set(available_providers(args.data_dir))
    for n in provider_registry.names():
        p = provider_registry.get(n)
        env = read_json(registry_path(n, args.data_dir)) if n in have else None
        print("%-10s auth=%-32s data=%s" % (
            n, ",".join(p.auth) or "(none)",
            "%d models @ %s" % (len(env.get("models") or {}), env.get("retrieved_at")) if env else "—"))
        if args.verbose and getattr(p, "describe", ""):
            print("           %s" % p.describe)
    return EXIT_OK


def cmd_capture(args: argparse.Namespace) -> int:
    """Fetch a provider's live source URLs and save them under --out using the fixture names."""
    from .http import Fetcher
    load_dotenv_if_available()
    p = provider_registry.get(args.provider)
    pages = p.fixtures(Path(args.out)) or {}
    http = Fetcher(None)
    code = EXIT_OK
    try:
        headers = {}
        if hasattr(p, "headers"):
            try:
                headers = p.headers()
            except Exception:
                headers = {}
        elif hasattr(p, "_headers"):
            headers = p._headers()
        for url, path in pages.items():
            if "#" in url or "/api/docs/models/" in url:
                continue  # POST bodies and the per-page docs are not captured here
            try:
                text, _ = http.get_text(url, headers=headers, cache=False)
            except Exception as exc:
                print("  ! %s: %s" % (url, exc))
                code = 3
                continue
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, "utf-8")
            print("  %s -> %s" % (url, path))
    finally:
        http.close()
    return code


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="modelroster", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version="modelroster " + __version__)
    ap.add_argument("--data-dir", default=None, help="registry data directory (default: package data, or $MODELROSTER_DATA_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("update", help="refresh registry data from official sources")
    u.add_argument("--provider", action="append", help="provider name (repeatable); default: all")
    u.add_argument("--all", action="store_true", help="explicitly all providers (the default)")
    u.add_argument("--offline", action="store_true", help="serve every request from the on-disk cache; never open a socket")
    u.add_argument("--no-cache", action="store_true", help="ignore and do not write the HTTP cache")
    u.add_argument("--dry-run", action="store_true", help="fetch, parse, validate, report; write nothing")
    u.add_argument("--fixtures", default=None, help="replay captured fixtures from this directory instead of the network")
    u.add_argument("--emit", default=None, help="also write a dependency-free snapshot module to this path")
    u.add_argument("--quiet", "-q", action="store_true")
    u.set_defaults(func=cmd_update)

    l = sub.add_parser("list", help="list models, optionally filtered by capability")
    l.add_argument("--provider", default=None)
    l.add_argument("--capability", "-c", action="append", help="name[=true|false|unknown] (repeatable)")
    l.add_argument("--endpoint", default=None, help="require this endpoint key to be supported")
    l.add_argument("--unknown-ok", action="store_true", help="let undocumented (None) values pass True/False filters")
    l.add_argument("--exclude-retired", action="store_true")
    l.add_argument("--discovered", action="store_true", help="list the discovered tier instead of verified")
    l.add_argument("--all-tiers", action="store_true")
    l.add_argument("--json", action="store_true")
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show one model record")
    s.add_argument("model")
    s.add_argument("--json", action="store_true")
    s.add_argument("--provenance", action="store_true", help="print the provenance of every field")
    s.set_defaults(func=cmd_show)

    d = sub.add_parser("diff", help="print the drift report of the last update")
    d.add_argument("--provider", action="append")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_diff)

    v = sub.add_parser("validate", help="re-run the validation gates on the stored data")
    v.add_argument("--provider", action="append")
    v.add_argument("-v", "--verbose", action="store_true")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("emit", help="write a dependency-free Python snapshot module")
    e.add_argument("--out", required=True)
    e.add_argument("--provider", action="append")
    e.add_argument("--capability", "-c", action="append")
    e.set_defaults(func=cmd_emit)

    di = sub.add_parser("discover", help="scan a broad registry for candidate models (discovered tier)")
    di.add_argument("source", help="huggingface | ollama_library | nvidia_nim")
    di.add_argument("--limit", type=int, default=None)
    di.add_argument("--offline", action="store_true")
    di.add_argument("--fixtures", default=None)
    di.add_argument("--write", action="store_true", help="write <data-dir>/discovered/<source>.json")
    di.add_argument("--json", action="store_true")
    di.set_defaults(func=cmd_discover)

    pr = sub.add_parser("providers", help="list known providers and their data status")
    pr.add_argument("-v", "--verbose", action="store_true")
    pr.set_defaults(func=cmd_providers)

    c = sub.add_parser("capture", help="save a provider's live listing responses as test fixtures")
    c.add_argument("--provider", required=True)
    c.add_argument("--out", default="tests/fixtures")
    c.set_defaults(func=cmd_capture)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:  # argparse uses 2 for usage errors; our contract says 4
        code = exc.code if isinstance(exc.code, int) else 0
        return EXIT_USAGE if code == 2 else code
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_USAGE
    except argparse.ArgumentTypeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_USAGE
    except KeyError as exc:
        print("error: %s" % exc.args[0], file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
