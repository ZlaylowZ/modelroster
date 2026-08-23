# Where modelroster intentionally differs from the prototype

Prototype: `nukez-agentic-testing/model_registry/` (≈2,800 lines, 58 tests).
Every item below is a deliberate change; behaviour not listed here was ported
as-is (parser rules, tri-state semantics, alias-conflict resolution, gate
thresholds, atomic writes, exit-code contract).

## Schema and records

1. **One record shape for every provider.** The prototype had two
   (OpenAI nested under `capabilities`, Anthropic flat) and `if provider ==
   "anthropic"` in every consumer function. `ModelRecord` + `Capabilities`
   dataclasses are used by all nine providers; the consumer layer has no
   provider branches.
2. **`tool_calling` is the single name.** The prototype stored
   `function_calling` *and* a `tool_calling` synonym. OpenAI's
   `function_calling` feature maps onto `tool_calling`; the raw feature map
   survives as `capabilities.extra["features.function_calling"]` etc.
3. **`available_to_api_key` is gone.** It was always `True`. Presence in the
   listing is the statement; `sources["listing"]` says which listing.
4. **`knowledge_cutoff` is an ISO date** when parseable (`"Aug 31, 2025"` →
   `"2025-08-31"`, month-only text → first of month), with the original text in
   `knowledge_cutoff_raw`. The prototype kept free text only.
5. **Pricing is parsed** from the `### Text tokens` table of OpenAI's
   `## Pricing` section (input / cached input / output, USD per 1M tokens) and
   from Inception's per-token prices. The prototype parsed past it. No other
   pricing sources were added (non-goal).
6. **`deprecated` / `shutdown_date`** are first-class. OpenAI's listing carries
   `shutdown_date`; Mistral's carries `deprecation`. `ModelRef.validate()` treats
   a model as retired only once the shutdown date has passed.
7. **`relationship` vocabulary**: `prose_alias` → `alias`; unmatched ids are
   `unknown` (the prototype used `None`); `fine_tune_unknown_base` is expressed
   as `unknown` + `raw["fine_tune_base"]` + a warning.
8. **`routes_to` and alias conflicts no longer mutate the parser's input.**
   `build_alias_index` is pure and returns a `routes_to` map that the provider
   applies explicitly.
9. **Per-provider JSON envelope** `{provider, schema_version, parser_version,
   retrieved_at, sources, model_order, models, stats, warnings, extra}`.
   OpenAI-specific state (`documentation_models`, `alias_index`,
   `alias_conflicts`, `page_failures`, `header_facts`, …) lives under `extra`
   instead of at the top level. `SCHEMA_VERSION` is 2.
10. **`endpoints_raw`** moved from capabilities to `raw["endpoints_raw"]`.

## Provider-wide documentation facts (new policy)

The prototype deliberately left Anthropic `function_calling = None`. Policy
now: an adapter may set a capability from provider-wide official documentation
only when the statement covers every model the listing returns, and must record
`provenance = {"section": "provider_docs", "evidence": "provider-wide
statement", "url": ...}`. Anthropic sets `tool_calling=True` and
`streaming=True` this way. Consumers who want per-model evidence only can
filter on provenance. Nothing else is inferred (Anthropic `prompt_caching`,
`fine_tuning`, OpenAI-compat providers' capabilities stay `None`).

## Providers and HTTP

11. **Provider plugin interface** replaces the four hard-coded provider
    lists. `Provider` protocol, `BaseProvider`, `OpenAICompatProvider`,
    explicit `register()` and the `modelroster.providers` entry-point group.
12. **One HTTP stack.** The `openai` SDK and the hand-rolled Anthropic loop are
    gone; every call goes through `Fetcher` (retries, jitter, 30 s cap,
    `Retry-After`, cache, offline, stale fallback, concurrency). Neither SDK
    is a dependency.
13. **`--offline` covers listing calls too**, and a `FixtureFetcher` replays
    captured responses (`--fixtures DIR`) for the whole pipeline.
14. **Content-hash change detection** (`meta["changed"]`) alongside
    ETag/Last-Modified; the OpenAI docs host sends neither, so this is what
    actually tells you a page changed. POST requests (`/api/show`) are cached
    under URL + body hash.
15. **Missing API key = skip, not error** (exit 0, `skipped=True`). The
    prototype raised.
16. **New providers**: xai, mistral, google, cohere, nvidia, inception,
    ollama; **discovery tier**: huggingface, ollama_library, nvidia_nim,
    written to `<data-dir>/discovered/` and excluded from `Registry.models()`
    by default.
17. **OpenAI listing via REST** (`GET /v1/models` with `httpx`) rather than
    `client.models.list()`; ids, `created`, `owned_by`, `shutdown_date` are
    taken verbatim.

## Validation and reporting

18. **Generic gates** run for every provider (empty listing, >50 % shrink when
    previous ≥ 5 models); OpenAI keeps its page-failure, parser-regression and
    catalog-shrink gates. The page-failure gate moved from `run_openai` into
    `OpenAIProvider.validate` so all gates live in one place.
19. **Header-fact loss gate** (new): pages that previously yielded a
    reasoning-effort sentence or prose alias must still yield it; ≥ 25 % loss
    refuses the write.
20. **Drift reports are written** (`<provider>.drift.json`) as well as printed,
    and `diff()` is provider-agnostic over the flattened record (so endpoint,
    tool, modality, pricing, alias and snapshot changes are all reported).
21. **Exit code 4 is reachable** (unknown provider, bad filter, unknown id).
22. **A parser crash is a refusal (2)**, never a traceback with a half-written
    file.

## Consumer API

23. `load()` returns a `Registry` object with `get / resolve / models(**filters)
    / providers / info / ids / refs` instead of module-level functions keyed by
    a `provider=` argument. The prototype's `supports_*` / `models_supporting_*`
    / `context_window` / `max_output_tokens` predicates remain, provider-agnostic.
24. **`ModelRef`** is new: parse (`provider/id` or bare id resolved by exact
    registry lookup; documented prefix heuristic only as a flagged fallback),
    `validate`, `resolve`, `is_valid`.
25. **Filter semantics**: `capability=True` matches documented support only;
    `unknown_ok=True` lets `None` pass; `strict=True` drops unmatched listing
    ids. The prototype's catalog hard-coded one cut (chat + function calling).
26. **Emitter** produces a generic, provider-keyed snapshot (`MODELS`,
    `<PROVIDER>_MODELS`, `<PROVIDER>_MODEL_IDS`, `REFRESHED`) chosen by the
    same filters as `Registry.models`; the Nukez-specific header, `--require-
    reasoning`, and the "both registries required" rule are gone.
27. **`refresh()`** helper for scheduled agents.
27b. **Runtime alias index.** The prototype's `get_model` was exact-only, so a
    documented alias the listing does not carry (`gpt-5.6`) or a retired
    snapshot (`gpt-4-0314`) returned `None`. `Registry` now builds an
    in-memory index of every record's `aliases[]`/`snapshots[]` at load time
    (same deterministic conflict rule as the update pipeline; real records
    always win) and `get`/`resolve`/`ModelRef` consult it, with `ft:<base>:...`
    falling back to the base. Matching is still exact — the index only
    contains ids the provider itself documented.

## Packaging and naming

28. `src/` layout, `pyproject.toml`, `httpx` as the only dependency, optional
    `python-dotenv` (`.env` from the working directory, `override=False`) —
    no repo-relative `.env`.
29. Data ships inside the wheel (`modelroster/data/<provider>.json`);
    `.previous.json`, drift reports, caches and discovered tiers are excluded.
30. `USER_AGENT` is `modelroster/<version>`; `NUKEZ_MODEL_REGISTRY_LIVE` is
    `MODELROSTER_LIVE`; data dir override is `--data-dir` / `MODELROSTER_DATA_DIR`.
31. Test helpers are consolidated in `tests/helpers.py` + `conftest.py`; the
    fixture set grew from 9 to 96 OpenAI pages (all pages linked from
    `models.md`) plus captured listings, so the full pipeline runs offline.

## Not carried over

* `--require-reasoning` (use `-c reasoning` / `reasoning=True`).
* `--no-catalog` / `--catalog-path` (use `--emit FILE`).
* Writing a downstream project's `model_catalog.py` — downstream projects now
  depend on the package or run `modelroster emit`.
