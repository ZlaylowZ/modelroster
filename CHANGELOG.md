# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

Two version numbers matter:

* the **package version** (`modelroster.__version__`) — API/CLI compatibility;
* the **parser version** (`modelroster.PARSER_VERSION`, stamped into every
  record) — bumped whenever any adapter's output for the same input could change.

## [Unreleased]

## [0.1.3] — 2026-08-25

### Added
- Cohere is live: shipped data now covers eight providers (382 models). The
  first real listing brought 31 models with documented tool calling (15),
  structured outputs (15), reasoning (3), context windows, and endpoints —
  `r.models(tool_calling=True, reasoning=True)` now spans five providers,
  84 models.
- Cohere feature vocabulary extended (`logprobs`, `tool_choice`,
  `tool_images`, `vision`); parser 2026.08.25-1.

### Fixed
- A persistent nightly failure no longer opens a duplicate GitHub issue per
  day; one open refresh-failure issue is the single tracker.
## [0.1.2] — 2026-08-24

### Added
- xAI capabilities are now documented, not unknown: the adapter parses each
  model's official docs.x.ai page (`/developers/models/<id>.md`) — function
  calling, structured outputs, reasoning, Batch API, context window,
  modalities, aliases — joined by exact id, with a docs-regression gate.
  `r.models(tool_calling=True, reasoning=True)` now spans anthropic, openai,
  mistral, and xai. Parser 2026.08.24-2.
- xAI test fixtures upgraded to real captures (listing reconstructed from the
  2026-08-24 live data; docs pages verbatim).

### Changed
- README rewritten around the core value proposition, with a per-provider
  documented-capability coverage table explaining exactly which providers a
  capability filter can return.
## [0.1.1] — 2026-08-24

### Added
- Shipped data now covers seven providers: anthropic, openai, xai, mistral,
  google, nvidia, inception (refreshed live 2026-08-24). Cohere pends an
  account with billing enabled; ollama remains local-only.
- Mistral adapter maps the full capability vocabulary observed live
  (parser 2026.08.24-1): `reasoning` -> `capabilities.reasoning`; `audio` ->
  audio input modality; `audio_speech`/`audio_transcription`/
  `audio_transcription_realtime`/`ocr`/`moderation`/`classification` ->
  stable endpoint keys. Keys remain mirrored in `capabilities.extra`.

### Fixed
- Daily refresh commits the successful providers' data even when one
  provider fails, and failure issues are actually created (label existed
  check).
## [0.1.0] — 2026-08-23

Initial release, ported from the `model_registry` prototype.

### Added
- One normalized `ModelRecord` schema with tri-state capabilities, modalities,
  endpoints, built-in tools, pricing, provenance, and warnings.
- Providers: anthropic, openai (docs parser + listing), xai, mistral, google,
  cohere, nvidia, inception, ollama; `modelroster.providers` entry-point group.
- Discovery tier: huggingface, ollama_library, nvidia_nim.
- Consumer API: `load()`, `Registry.models(**filters)`, `ModelRef`, predicates.
- CLI: `update`, `list`, `show`, `diff`, `validate`, `emit`, `discover`,
  `providers`, `capture`.
- Shared HTTP layer with retries, caching, content-hash change detection,
  full `--offline` coverage, and fixture replay.
- Validation gates (page failures, parser regression, catalog shrink,
  header-fact loss, generic shrink) with byte-for-byte preservation on refusal.
- Shipped data for anthropic, openai, nvidia, inception (retrieved 2026-08-23);
  xai, mistral, cohere, google populate via `modelroster update` with keys.
- Runtime alias index: documented aliases, snapshots, and `ft:` bases resolve
  through `Registry.get`/`resolve`/`ModelRef` even when the listing does not
  carry them.
- `py.typed` marker for downstream type-checkers.
- GitHub Actions: offline tests on push, daily live refresh, PyPI publish on tags.
