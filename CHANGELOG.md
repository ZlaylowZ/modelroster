# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

Two version numbers matter:

* the **package version** (`modelroster.__version__`) — API/CLI compatibility;
* the **parser version** (`modelroster.PARSER_VERSION`, stamped into every
  record) — bumped whenever any adapter's output for the same input could change.

## [Unreleased]

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
