# modelroster

**Accurate, current LLM model identifiers and capabilities for every provider —
shipped as data, refreshed from official sources, never guessed.**

```bash
pip install modelroster
```

```python
import modelroster

r = modelroster.load()                                   # works offline, no keys
for m in r.models(tool_calling=True, reasoning=True):
    print(m.ref, m.context_window, m.capabilities.reasoning_efforts)

ref = modelroster.ModelRef.parse("openai/gpt-5.4").validate()   # raises on unknown/retired ids
modelroster.context_window("claude-opus-5")              # 1000000
modelroster.supports_tool_calling("text-embedding-3-small")     # None  (not documented — NOT False)
```

## Why

Providers add, rename, alias, snapshot and retire models constantly, and each
publishes its list differently. Every project that calls an LLM ends up with a
hand-maintained model table that silently rots. `modelroster` is one package
that any project — or any agent — installs to get the exact id strings a
provider accepts today and what each model supports, with enough validation
that an upstream format change fails loudly instead of shipping a broken
catalog.

## Principles

1. **Tri-state capabilities.** Every capability is `True` / `False` / `None`.
   `None` means *the source does not say* and is never collapsed into `False`.
   `False` only arises from an explicit "not supported" statement, or from
   absence in a positively enumerated list whose section is present.
2. **No generation probes.** Availability comes from listing endpoints;
   capabilities come from official documentation or official API metadata.
   The registry never sends a completion request to find out what a model does.
3. **Provenance on every fact.** Each field carries `{section, evidence, ...}`
   naming the document section or API field it came from
   (`modelroster show gpt-5.4 --provenance`).
4. **Exact ids only.** Aliases, snapshots and fine-tune bases resolve through an
   explicit index built from provider statements; there is no fuzzy matching and
   no inference from date suffixes.
5. **Refuse rather than rot.** Validation gates refuse to overwrite good data
   when a parser stops understanding a page; the previous data survives
   byte-for-byte and a drift report says what changed.

## What is in the box

| Provider | Availability | Capabilities | Key |
|---|---|---|---|
| `anthropic` | `GET /v1/models` (paginated) | the same call's `capabilities` object | `ANTHROPIC_API_KEY` |
| `openai` | `GET /v1/models` | official Markdown docs (`developers.openai.com/api/docs/models/*.md`), 96 pages | `OPENAI_API_KEY` |
| `xai` | `GET /v1/models` | `GET /v1/language-models` (modalities, aliases) | `XAI_API_KEY` |
| `mistral` | `GET /v1/models` | the listing's `capabilities` object | `MISTRAL_API_KEY` |
| `google` | OpenAI-compat shim `/v1beta/openai/models` | native `/v1beta/models` (limits, methods) | `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| `cohere` | `GET /v1/models` | the same call (endpoints, features, context) | `COHERE_API_KEY` |
| `nvidia` | `GET integrate.api.nvidia.com/v1/models` (public) | — (ids only) | optional |
| `inception` | `GET api.inceptionlabs.ai/v1/models` (public) | the same call (modalities, limits, features, pricing) | optional |
| `ollama` | local `GET /api/tags` | local `POST /api/show` (capabilities, context) | none (`OLLAMA_HOST`) |

The wheel ships data for every provider that could be refreshed without a
private key at release time (`modelroster providers` shows what is loaded and
when it was retrieved). Run `modelroster update` with your keys to add the rest;
a missing key skips that provider, it is never an error.

A separate **discovery tier** (`modelroster discover huggingface|ollama_library|nvidia_nim`)
lists candidate models from broad registries with mostly-unknown capabilities.
They are labelled `tier="discovered"` and never enter the verified catalog.

## Consumer API

```python
r = modelroster.load()                       # every provider with data
r = modelroster.load("openai")               # one provider
r = modelroster.load(data_dir="~/my/data")   # a refreshed copy (also $MODELROSTER_DATA_DIR)

r.providers(); r.info()                      # retrieved_at, parser_version, counts
r.get("gpt-5.4"); r.get("openai/gpt-5.4"); r.get(ModelRef("openai", "gpt-5.4"))
r.resolve("gpt-4o-2024-08-06")               # -> the gpt-4o family record
r.models(provider="openai", tool_calling=True, image_input=True, endpoint="responses")
r.models(reasoning=True, unknown_ok=True)    # let None pass too
r.models(relationship="canonical", include_retired=False, strict=True)
r.ids(...); r.refs(...)                      # plain ids / ModelRefs
```

Filters accept every capability name (`reasoning`, `reasoning_efforts`,
`extended_thinking`, `tool_calling`, `structured_outputs`, `streaming`,
`prompt_caching`, `fine_tuning`, `batch`, `citations`, `code_execution`,
`pdf_input`), modality flags (`image_input`, `audio_output`, …), any
`capabilities.extra` key, and `endpoint=` / `builtin_tool=`. A `True`/`False`
filter matches only a *documented* value; pass `unknown_ok=True` to let `None`
through.

Module-level predicates mirror the record fields and are provider-agnostic:
`supports(model, cap)`, `supports_tool_calling`, `supports_reasoning`,
`supported_reasoning_efforts`, `supports_endpoint`, `supports_builtin_tool`,
`supports_modality`, `context_window`, `max_input_tokens`, `max_output_tokens`,
`models_supporting(cap, provider)`, `available_models(provider)`.

### `ModelRef` — a type for model names

```python
from modelroster import ModelRef, UnknownModelError, RetiredModelError

ModelRef.parse("openai/gpt-5.4")        # explicit
ModelRef.parse("gpt-5.4")               # provider found by exact lookup in the registry
ModelRef.parse("gpt-99").inferred       # True: only the documented prefix heuristic matched
ModelRef("openai", "gpt-5.4").validate()            # raises UnknownModelError / RetiredModelError
ModelRef("openai", "gpt-4o-2024-08-06").resolve()   # canonical family record
```

### Dependency-free snapshot

```bash
modelroster emit --out my_models.py --provider openai --provider anthropic -c tool_calling
```

writes a compiled-checked module with `MODELS`, `OPENAI_MODELS`,
`OPENAI_MODEL_IDS`, … for projects that vendor a file instead of depending on
`modelroster`.

### Scheduled refresh from an agent

```python
report = modelroster.refresh(["openai", "anthropic"])   # {provider: {code, drift, errors, ...}}
```

## CLI

```
modelroster update [--provider X ...] [--offline] [--dry-run] [--no-cache] [--fixtures DIR] [--emit FILE]
modelroster list [--provider X] [-c reasoning -c tool_calling[=true|false|unknown]] [--endpoint K] [--json]
modelroster show <id | provider/id> [--provenance] [--json]
modelroster diff [--provider X]          # last drift report
modelroster validate [--provider X] [-v] # re-run the gates on stored data
modelroster emit --out FILE [--provider X] [-c ...]
modelroster discover <huggingface|ollama_library|nvidia_nim> [--limit N] [--write]
modelroster providers [-v]
modelroster capture --provider X         # save live listing responses as test fixtures
```

Exit status: `0` ok · `2` validation refused the write (previous data preserved)
· `3` fetch failure · `4` usage. Providers run independently; the exit status
is the worst stage.

`--offline` serves *every* request — documentation pages and listing calls —
from the on-disk cache (`<data-dir>/cache/<provider>/`) and never opens a
socket. `--fixtures tests/fixtures` replays the captured fixtures instead.

Keys are read from the environment (`<PROVIDER>_API_KEY`), optionally from a
`.env` in the working directory when `python-dotenv` is installed
(`pip install modelroster[dotenv]`); existing environment variables are never
overridden. No key is ever written anywhere.

## Record shape

```
ModelRecord
  provider, model_id                exact string the API accepts
  display_name, description, family (canonical id), aliases, snapshots,
  default_snapshot, routes_to, relationship (canonical|snapshot|alias|fine_tune_inherited|unknown)
  released, deprecated, shutdown_date
  context_window, max_input_tokens, max_output_tokens
  knowledge_cutoff (ISO), knowledge_cutoff_raw
  modalities {text,image,audio,video} x {input,output}
  capabilities  reasoning, reasoning_efforts, default_effort, extended_thinking, tool_calling,
                structured_outputs, streaming, prompt_caching, fine_tuning, batch, citations,
                code_execution, pdf_input, extra{provider-specific keys}
  endpoints {key: tri}, builtin_tools {key: tri} | None, pricing {input, output, cached_input} | None
  tier (verified|discovered), provenance, sources, raw, retrieved_at, parser_version, warnings
```

**Provider-wide facts.** An adapter may set a capability from provider-wide
official documentation (rather than a per-model source) only when the statement
covers every model the listing returns; such values carry
`provenance = {"section": "provider_docs", "evidence": "provider-wide statement", "url": ...}`.
Anthropic uses this for `tool_calling` and `streaming`. Nothing else is inferred.

## Validation gates

Generic: empty listing; model count shrinks by more than half when the previous
run had at least 5 models. OpenAI: more than 10 % of documentation pages fail
to fetch/parse; parser regression (more than 25 % or at least 10
previously-understood pages now parse to nothing); documentation catalog shrinks
by more than half; loss of header-region facts (reasoning-effort sentence, prose
alias) on at least 25 % of the pages that previously carried them. The emitted
snapshot module must compile. On refusal the previous file is untouched and the
CLI exits 2.

Every successful update writes `<provider>.drift.json` beside the data: added /
removed models and families, new / removed snapshots, per-model capability
deltas (`None` printed as `unknown`), and warnings.

## Adding a provider

Subclass `OpenAICompatProvider` (or `BaseProvider`), set `name`, `base_url`,
`auth`, override `enrich_record` if the provider publishes per-model metadata,
and point `fixtures()` at a captured response. Register it with
`modelroster.providers.register(MyProvider())` or via the entry-point group
`modelroster.providers`. No core file changes are needed (see
`tests/test_compat_providers.py::test_plugin_provider_needs_no_core_edits`).

## Development

```bash
pip install -e ".[dev]"
pytest -q                                   # offline, ~1 s
MODELROSTER_LIVE=1 pytest tests/test_live.py -q    # hits the real endpoints
modelroster update --fixtures tests/fixtures --dry-run   # full pipeline on fixtures
```

See [MAINTAINERS.md](MAINTAINERS.md) for the refresh loop, `docs/DESIGN.md`
for the design note, and `docs/DIFFERENCES.md` for where this package
intentionally differs from the prototype it was ported from.

## License

MIT.
