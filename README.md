# modelroster

**Delete your hand-maintained model list.** One `pip install` gives you every
provider's current model ids and capabilities — refreshed daily from official
sources, shipped as data, queryable in one line.

```bash
pip install modelroster
```

```python
import modelroster

r = modelroster.load()        # current data for every provider — no keys, no network

r.models(provider="anthropic")                        # every Claude model, newest first
modelroster.context_window("grok-4.6")                # 500000
modelroster.supported_reasoning_efforts("gpt-5.4")    # ['none', 'low', 'medium', 'high', 'xhigh']

# choose models by what your code needs, not by memorised names:
for m in r.models(tool_calling=True, reasoning=True):
    print(m.ref, m.context_window)
# anthropic/claude-opus-5 1000000
# openai/gpt-5.4 1050000
# xai/grok-4.6 500000
# mistral/magistral-medium-latest 40960
# cohere/command-a-reasoning-08-2025 256000 ...

# and never ship a typo'd or retired model id again:
modelroster.ModelRef.parse("openai/gpt-5.4").validate()   # raises on unknown/retired ids
```

## The problem it deletes

Every project that calls an LLM grows a hand-maintained table: model ids,
context windows, "supports tools?" comments. Providers rename, alias,
snapshot, and retire models constantly, so that table silently rots — the
stale id fails in production, the context window is from two releases ago,
and switching providers means researching a new set of names.

`modelroster` makes that someone else's job:

* **Install and go.** The wheel ships current data for eight providers
  (anthropic, openai, xai, mistral, google, cohere, nvidia, inception —
  382 models). No API keys, no network, no setup. `modelroster providers` shows what you have and when
  it was retrieved.
* **Stays current without you.** A daily pipeline refreshes the data from
  each provider's own listing APIs and official documentation, with
  validation gates that refuse to publish a broken parse. Release versions
  snapshot that data; `modelroster update` refreshes your local copy any time.
* **One structure for every provider.** Every model is the same
  `ModelRecord` — id, family, aliases, context window, modalities,
  capabilities, endpoints, pricing — so swapping `anthropic/claude-opus-5`
  for `openai/gpt-5.4` or `mistral/magistral-medium-latest` is a value
  change, not a research project.
* **Answers you can trust.** No generation probes, no fuzzy matching, no
  guessing from model names. Every fact is traceable to the API field or
  documentation section that stated it (`modelroster show gpt-5.4
  --provenance`), and every capability is honestly tri-state: `True`,
  `False`, or `None` for "the source does not say" — unknown is never
  dressed up as an answer.

## Querying

```python
r = modelroster.load()

r.models()                                   # every model, every provider
r.models(provider="mistral")                 # one provider's full list
r.models(tool_calling=True, reasoning=True)  # documented support for both
r.models(reasoning=True, include_retired=False)
r.models(provider="openai", image_input=True, endpoint="responses")
r.get("gpt-5.4")                             # one record (also "openai/gpt-5.4", ModelRef)
r.resolve("gpt-4o-2024-08-06")               # snapshot/alias -> the family's canonical record
```

Capability filters match **documented** values only: a provider whose source
doesn't state a capability (`None`) won't match `capability=True` — that's the
honesty guarantee, not a gap. Add `unknown_ok=True` to also accept
undocumented models. Filters accept every capability (`reasoning`,
`tool_calling`, `structured_outputs`, `streaming`, `prompt_caching`,
`fine_tuning`, `batch`, `citations`, `code_execution`, `pdf_input`, …),
modality flags (`image_input`, `audio_output`, …), any provider-specific
`capabilities.extra` key, plus `endpoint=` and `builtin_tool=`.

One-line predicates for the common questions (provider-agnostic — they find
the model wherever it lives):

```python
modelroster.context_window("claude-opus-5")           # 1000000
modelroster.max_output_tokens("gpt-5.4")              # 128000
modelroster.supports_tool_calling("gpt-3.5-turbo")    # False (documented)
modelroster.supports_tool_calling("some-embedding")   # None  (undocumented — NOT False)
modelroster.models_supporting("reasoning", "google")  # 34 Gemini ids
```

### What each provider's sources document

Coverage differs because providers publish different amounts of metadata.
This table is what determines which providers appear under a given capability
filter (counts from the 0.1.3 data):

| Provider | Models | tool_calling | reasoning | structured_outputs | context window | Source |
|---|---:|---:|---:|---:|---:|---|
| anthropic | 10 | 10 | 10 | 10 | 10 | `/v1/models` capabilities object (+ provider-wide tool/streaming docs) |
| openai | 126 | 103 | 120 | 103 | 97 | official Markdown docs, 96 pages |
| mistral | 56 | 56 | 56 | — | 56 | `/v1/models` capabilities object |
| cohere | 31 | 15 | 3 | 15 | 31 | `/v1/models` endpoints + features lists |
| xai | 12 | 7 | 7 | 7 | 7 | docs.x.ai per-model pages + `/v1/language-models` |
| google | 51 | — | 34 | — | 50 | native `/v1beta/models` (no per-model tool field) |
| inception | 1 | 1 | — | 1 | 1 | `/v1/models` supported_features |
| nvidia | 95 | — | — | — | — | ids-only public listing |

"—" means the provider's official source simply doesn't state it; those
models are reachable via `r.models()`, provider listings, or `unknown_ok=True`.

### `ModelRef` — a type for model names

```python
from modelroster import ModelRef, UnknownModelError, RetiredModelError

ref = ModelRef.parse("openai/gpt-5.4")     # or bare "gpt-5.4" — provider found by exact lookup
ref.validate()                             # raises UnknownModelError / RetiredModelError
ref.resolve()                              # aliases, snapshots, ft: ids -> the canonical record
```

Aliases, snapshots, and fine-tune bases resolve through an index built from
provider statements — including documented aliases the listing API doesn't
carry (`gpt-5.6`, `grok-4.3-latest`). Never fuzzy, never guessed from
date-looking suffixes.

## Keeping data fresh yourself

The shipped data is refreshed at every release. For fresher data between
releases:

```bash
modelroster update                    # refresh every provider you have keys for
modelroster update --provider ollama  # your local Ollama daemon (no key needed)
modelroster diff                      # what changed since last time
```

Keys are read from `<PROVIDER>_API_KEY` environment variables (or a `.env`
with the `dotenv` extra); a missing key skips that provider — never an error.
Long-running agents can call `modelroster.refresh()` on a schedule and act on
the returned drift report. The data directory is overridable
(`--data-dir` / `$MODELROSTER_DATA_DIR`), so refreshed data can live outside
the installed package.

One provider needs something from you: **ollama** is inherently local — run
`modelroster update --provider ollama` against your own daemon. (Cohere's
listing requires an account with billing enabled; the shipped data covers it.)

## What is in the box

| Provider | Availability | Capabilities | Key |
|---|---|---|---|
| `anthropic` | `GET /v1/models` (paginated) | the same call's `capabilities` object | `ANTHROPIC_API_KEY` |
| `openai` | `GET /v1/models` | official Markdown docs (`developers.openai.com/api/docs/models/*.md`), 96 pages | `OPENAI_API_KEY` |
| `xai` | `GET /v1/models` | docs.x.ai per-model pages (function calling, structured outputs, reasoning, batch, context) + `GET /v1/language-models` (modalities, aliases) | `XAI_API_KEY` |
| `mistral` | `GET /v1/models` | the listing's `capabilities` object | `MISTRAL_API_KEY` |
| `google` | OpenAI-compat shim `/v1beta/openai/models` | native `/v1beta/models` (limits, methods, thinking) | `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| `cohere` | `GET /v1/models` | the same call (endpoints, features, context) | `COHERE_API_KEY` (account with billing) |
| `nvidia` | `GET integrate.api.nvidia.com/v1/models` (public) | — (ids only) | optional |
| `inception` | `GET api.inceptionlabs.ai/v1/models` (public) | the same call (modalities, limits, features, pricing) | optional |
| `ollama` | local `GET /api/tags` | local `POST /api/show` (capabilities, context) | none (`OLLAMA_HOST`) |

A separate **discovery tier** (`modelroster discover huggingface|ollama_library|nvidia_nim`)
lists candidate models from broad registries with mostly-unknown capabilities.
They are labelled `tier="discovered"` and never enter the verified catalog.

## Principles

1. **Tri-state capabilities.** `True` / `False` / `None`, where `None` means
   *the source does not say* and is never collapsed into `False`.
2. **No generation probes.** Availability from listing endpoints;
   capabilities from official documentation or official API metadata only.
3. **Provenance on every fact** — the exact section or field that stated it.
4. **Exact ids only.** No fuzzy matching, no inference from date suffixes.
5. **Refuse rather than rot.** Validation gates refuse to overwrite good
   data when an upstream format changes; the previous data survives
   byte-for-byte and a drift report says what changed.

## CLI

```
modelroster update [--provider X ...] [--offline] [--dry-run] [--no-cache] [--fixtures DIR] [--emit FILE]
modelroster list [--provider X] [-c reasoning -c tool_calling[=true|false|unknown]] [--endpoint K] [--json]
modelroster show <id | provider/id> [--provenance] [--json]
modelroster diff [--provider X]          # last drift report
modelroster validate [--provider X] [-v] # re-run the gates on stored data
modelroster emit --out FILE [--provider X] [-c ...]   # dependency-free vendorable snapshot module
modelroster discover <huggingface|ollama_library|nvidia_nim> [--limit N] [--write]
modelroster providers [-v]
modelroster capture --provider X         # save live listing responses as test fixtures
```

Exit status: `0` ok · `2` validation refused the write (previous data
preserved) · `3` fetch failure · `4` usage. Providers run independently; the
exit status is the worst stage. `--offline` serves every request from the
on-disk cache and never opens a socket.

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

Records are dataclasses with `to_dict()` / `from_dict()`, and the package
ships a `py.typed` marker. **Provider-wide facts** policy: an adapter may set
a capability from provider-wide official documentation only when the
statement covers every model the listing returns, and marks it
`provenance.section = "provider_docs"` (Anthropic: `tool_calling`,
`streaming`). Nothing else is inferred.

## Adding a provider

Subclass `OpenAICompatProvider` (or `BaseProvider`), set `name`, `base_url`,
`auth`, override `enrich_record` if the provider publishes per-model
metadata, and point `fixtures()` at a captured response. Register with
`modelroster.providers.register(...)` or the `modelroster.providers`
entry-point group — no core changes needed.

## Development

```bash
pip install -e ".[dev]"
pytest -q                                          # offline, ~1 s
MODELROSTER_LIVE=1 pytest tests/test_live.py -q    # hits the real endpoints
modelroster update --fixtures tests/fixtures --dry-run   # full pipeline on captured fixtures
```

See [MAINTAINERS.md](MAINTAINERS.md) for the refresh loop, `docs/DESIGN.md`
for the design note, and `docs/DIFFERENCES.md` for where this package
intentionally differs from the prototype it was ported from.

## License

MIT.
