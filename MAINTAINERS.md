# Maintaining modelroster

This file is written for the agent (or person) operating the refresh loop.
Everything below is scriptable; nothing requires judgment until a gate fires.

## The loop

```
modelroster update --all            # 1. refresh every provider you have keys for
modelroster diff                    # 2. read the drift reports
modelroster validate -v             # 3. inspect warnings
```

1. **Run the update.** One provider failing never blocks the others. Exit
   status is the worst stage: `0` everything written, `2` at least one provider
   refused (its previous data is untouched), `3` a fetch failed, `4` usage.
   A provider whose key is absent is *skipped* with exit 0 — check the summary
   line (`xai: skipped (...)`) to make sure that is what you expected.
2. **Read the drift report** (`modelroster diff`, or `<provider>.drift.json`).
   New/removed models and snapshots are normal. Post the report where the team
   will see it. A `changed_capabilities` entry that flips a documented value
   (`true -> false`) on many models at once is a parser problem until proven
   otherwise — see "Deciding whether the parser broke".
3. **Inspect warnings.** The categories you will see:
   * `No official model documentation record matched this API model ID` —
     the listing has ids the docs don't (search/tts variants). Informational.
   * `id X claimed by A, B; resolved to A` — alias pages claiming another
     family's snapshot; resolved deterministically. Informational.
   * `contradiction: ...` — the page says two things. Informational, but
     worth a glance.
   * `unrecognised <thing> ... (kept)` — **new vocabulary**. The value is
     retained, so nothing is lost, but the known-vocabulary tuples in
     `schema.py` / the provider module should be extended so the value gets a
     stable key and absence starts meaning `False` where a positive list is
     present. This is a parser change → bump `PARSER_VERSION`.
   * `unparseable knowledge cutoff` — extend `parse_cutoff`. Parser change.
   * `no Endpoints section found` / `no Model details section found` on
     pages that used to have them — format change. See below.

## Deciding whether the parser broke

A gate firing (exit 2) means the data on disk is still the last good run. Do
**not** delete `<provider>.json` or `.previous.json` to "fix" it.

* `XAI DOCS REGRESSION suspected` — fetch one page
  (`curl https://docs.x.ai/developers/models/grok-4.6.md`) and compare with
  `tests/fixtures/xai_docs/`; adjust `parse_model_page` in
  `providers/xai.py`, refresh the fixture, bump `PARSER_VERSION`.
* `PARSER REGRESSION suspected` / `HEADER PARSE REGRESSION suspected` —
  fetch one affected page (`curl https://developers.openai.com/api/docs/models/<slug>.md`)
  and compare with `tests/fixtures/openai_docs/<slug>.md`. If a heading was
  renamed or a bullet format changed, update the regexes / section constants
  in `providers/openai_docs.py`, refresh the fixture, add a test, bump
  `PARSER_VERSION`.
* `documentation catalog shrank` — open `models.md`; the link format probably
  changed (`discover_model_pages`).
* `N/M documentation pages failed to fetch/parse` — transient (rerun, the
  fetcher retries and falls back to cache) unless the failures are `parse:` —
  then a format change.
* `model list shrank` — compare the raw listing (`modelroster capture`) with
  the previous data. Real mass retirements happen; if it is real, the gate can
  be overridden by deleting *only* `<provider>.previous.json`… but first post
  the diff and get a human ack.

## Parser version and releases

`PARSER_VERSION` (`schema.py`) is stamped into every record. Bump it whenever
an adapter's output for the same input could change: new vocabulary, a new
regex, a changed mapping, a new provider-wide fact. The date-based format is
`YYYY.MM.DD-N`.

Release when data or code changed in a way consumers should pick up:

1. `CHANGELOG.md`: move items from Unreleased under a new version heading.
2. Bump `src/modelroster/_version.py` (semver: data-only refresh = patch;
   new provider/field = minor; schema or API break = major — also bump
   `SCHEMA_VERSION`).
3. `pytest -q && modelroster validate`.
4. Tag `vX.Y.Z` and push the tag — `publish.yml` builds, re-validates, checks
   the tag matches `__version__`, and publishes via PyPI trusted publishing
   (configure the `pypi` environment once in the repository settings).

Before the **first** publish, confirm the name is still free:
`pip index versions modelroster` should report nothing.

## Automation in place

* `ci.yml` — offline suite on every push (3.11–3.13), fixture replay of the
  full OpenAI/Anthropic pipeline, shipped-data validation, and a clean-venv
  install test of the built wheel.
* `refresh.yml` — daily at 06:17 UTC: `modelroster update --all --no-cache`
  with whatever secrets are configured, the live test suite, then commits
  `src/modelroster/data/*.json` (data + drift reports) on success. On exit ≠ 0
  it opens a GitHub issue labelled `refresh-failure` with the log tail and
  fails the run.
* `publish.yml` — on tags.

Secrets to configure for the refresh: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
and optionally `XAI_API_KEY`, `MISTRAL_API_KEY`, `GOOGLE_API_KEY`,
`COHERE_API_KEY`, `NVIDIA_API_KEY`, `INCEPTION_API_KEY`. Missing ones skip
the provider.

## Fixtures

`tests/fixtures/README.md` records the provenance of every fixture. When a
provider's response shape changes or a key becomes available for a provider
whose fixture is reference-shaped, run `modelroster capture --provider <name>`
(writes under `tests/fixtures/listings/`) and commit the result. OpenAI
documentation pages are refreshed by copying `<data-dir>/cache/openai/*.body`
files to `tests/fixtures/openai_docs/<slug>.md` (the `.meta.json` beside each
body records its URL); xAI pages likewise live in `tests/fixtures/xai_docs/`
(one `<model id>.md` per id in the xai listing fixture — refresh with
`curl https://docs.x.ai/developers/models/<id>.md`).

## Adding a provider

1. `src/modelroster/providers/<name>.py` — subclass `OpenAICompatProvider`
   or `BaseProvider`; set `name`, `auth`, `describe`; implement
   `enrich_record`/`enrich`; point `fixtures()` at captured responses.
2. Add it to `_BUILTIN` in `providers/__init__.py` (third parties use the
   entry-point group instead).
3. `tests/fixtures/listings/<name>_models.json` + a test in
   `tests/test_compat_providers.py` that asserts the tri-state mapping.
4. Document the row in the README table; add the key to `refresh.yml`.

## Things that must stay true

* No generation probes. Availability from listings, capabilities from
  official docs/metadata only.
* `None` is never turned into `False`.
* Exact-id matching only.
* A refused update leaves the previous file byte-identical.
* No key is ever read from anywhere but the environment / `.env`, and never
  written to disk (the HTTP cache stores response bodies, not request headers).
