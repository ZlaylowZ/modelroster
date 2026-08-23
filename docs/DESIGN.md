# modelroster — design note

Status: confirmed before bulk implementation (2026-08-23). Package/import name
`modelroster` (PyPI name checked free 2026-08-23; re-check at first publish).

## What it is

One pip-installable registry of *current* LLM model identifiers and their
capabilities, for every provider, shipped with generated data so it works
offline with no keys, plus an updater that refreshes the data from official
listing endpoints and official documentation and refuses to overwrite good
data with a broken parse.

Non-negotiables carried over from the prototype: tri-state capabilities
(`True`/`False`/`None`, `None` = "source does not say", never collapsed to
`False`); no generation probes; provenance on every fact; exact-id alias
resolution with deterministic conflict handling; heading-driven parsing;
validation gates with byte-for-byte preservation on refusal; atomic writes
with a last-known-good copy; injectable HTTP; offline fixture tests.

## Schema (one record shape for every provider)

`modelroster.schema.ModelRecord` — a plain dataclass, JSON round-trippable
via `to_dict()/from_dict()`, no Pydantic dependency.

```
ModelRecord
  provider, model_id                  exact string the provider's API accepts
  display_name, description, family   family == canonical id (the documented page id)
  aliases[], snapshots[], default_snapshot, routes_to
  relationship                        canonical | snapshot | alias | fine_tune_inherited | unknown
  released, deprecated (bool|None), shutdown_date
  context_window, max_input_tokens, max_output_tokens
  knowledge_cutoff (ISO date|None), knowledge_cutoff_raw (str|None)
  modalities                          {text,image,audio,video} x {input,output}, tri-state
  capabilities : Capabilities         reasoning, reasoning_efforts, default_effort, extended_thinking,
                                      tool_calling, structured_outputs, streaming, prompt_caching,
                                      fine_tuning, batch, citations, code_execution, pdf_input,
                                      extra{}  (provider-specific, tri-state, e.g. Anthropic
                                                context_management.*, OpenAI features.*)
  endpoints{}                         stable key -> tri-state (prototype route->key map)
  builtin_tools                       None (unknown) | {key -> tri-state}
  pricing                             None | {input, output, cached_input} USD per 1M tokens
  tier                                "verified" | "discovered"
  provenance{}                        dotted field -> {section, evidence, ...}
  sources{}                           {"listing": ..., "documentation": ...}
  raw{}                               untouched provider fields worth keeping (created, owned_by, ...)
  retrieved_at, parser_version, warnings[]
```

Per-provider data file (`data/<provider>.json`) envelope:
`{provider, schema_version, parser_version, retrieved_at, sources, model_order[],
models{model_id -> record}, stats{}, warnings[], extra{}}`. `extra` holds
provider-specific validation state (OpenAI: `documentation_models`,
`alias_index`, `alias_conflicts`, `page_failures`, `header_facts`).

`tool_calling` is the single name (the prototype's `function_calling`
duplicate is gone; OpenAI's feature key is mapped onto it and the raw feature
map survives in `capabilities.extra["features.*"]`).

**Provider-wide facts policy.** An adapter may set a capability from
provider-wide official documentation (not per-model) only when the statement
covers every model the listing returns and the provenance records
`{"section": "provider_docs", "evidence": "provider-wide statement", "url": ...}`.
Anthropic uses this for `tool_calling=True` and `streaming=True`. Consumers can
see the difference in provenance; nothing else is inferred.

## Typed identifiers

`ModelRef(provider, model_id)` with `ModelRef.parse("openai/gpt-5.4")` or
`parse("gpt-5.4")` (provider found by exact lookup across the loaded registry;
a documented prefix heuristic is the fallback and is flagged in the result),
`resolve(registry)` → canonical record, `validate(registry)` → raises
`UnknownModelError` / `RetiredModelError`.

## Provider interface

```python
class Provider(Protocol):
    name: str
    auth: tuple[str, ...]          # env var names; () for local services
    def list_models(self, http: Fetcher) -> list[dict]: ...
    def enrich(self, raw: list[dict], http: Fetcher) -> ProviderResult: ...
    def fixtures(self) -> dict[str, str] | None: ...     # url -> fixture file, for offline tests
    def validate(self, current: dict, previous: dict | None) -> tuple[list, list]: ...  # optional extra gates
```

`ProviderResult` = records + provider `extra` + stats + warnings. Providers are
found through an explicit in-package registry plus the `modelroster.providers`
entry-point group. `OpenAICompatProvider` is a configurable base class
(`base_url`, `auth`, optional enrich hook) so adding xAI/Mistral/etc. is a
~30-line module plus a fixture. Anthropic and OpenAI are full adapters. Ollama
uses `/api/tags` + `/api/show` (official metadata, not a probe).

## HTTP

One `Fetcher` (ported): timeouts, retry on 429/5xx/transport with jittered
exponential backoff capped at 30 s, `Retry-After`, no retry on other 4xx,
conditional GET when ETags exist, content-hash change detection otherwise,
disk cache (`<sha1>.body` + `.meta.json`), `offline=True` covering *every*
call, stale-cache fallback, bounded concurrency, `User-Agent: modelroster/<ver>`.
`get_json()` wraps `get_text()` for API calls with per-call headers.

## Pipeline and gates

`update.run_provider(provider, ...)` → fetch → enrich → validate(current,
previous) → diff → atomic write (`<provider>.json`, `<provider>.previous.json`
under `--data-dir`, default: the package data dir) → drift report printed and
written to `<provider>.drift.json`. Providers run independently; the CLI exit
code is the worst stage (0 ok, 2 validation refused, 3 fetch failure, 4 usage).
A missing API key skips the provider (exit 0, reported).

Generic gates (all providers): empty result; shrink >50% when previous had ≥5
models. OpenAI gates: >10% page fetch/parse failures; parser regression
(>25% or ≥10 previously-understood pages now empty); documentation catalog
shrink >50%; loss of header-region facts (effort sentences / prose aliases)
on ≥25% of pages that previously had them. Emitted module must compile.

## Consumer API

`load(provider=None, data_dir=None) -> Registry`; `Registry.get`, `.models(provider=None,
strict=False, **filters)`, `.resolve`, `.providers()`, `.info()`, `.refs()`;
module-level predicates `supports(model, "tool_calling")`, `context_window`,
`max_output_tokens`, `models_supporting(...)`; `modelroster.emit.render_module`
for a dependency-free snapshot; `modelroster.refresh()` returning drift.

## Layout

```
src/modelroster/{__init__,schema,http,registry,ref,validate,store,update,emit,cli}.py
src/modelroster/providers/{__init__,base,anthropic,openai,openai_docs,openai_compat,xai,mistral,google,cohere,nvidia,inception,ollama}.py
src/modelroster/discover/{__init__,huggingface,ollama_library,nvidia_nim}.py
src/modelroster/data/<provider>.json          shipped; caches/previous files never shipped
tests/{conftest.py,test_*.py,fixtures/}
```
