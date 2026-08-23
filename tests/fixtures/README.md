# Test fixtures

All tests except `test_live.py` run against these files only.

## openai_docs/

Verbatim copies of OpenAI's official Markdown documentation pages
(`https://developers.openai.com/api/docs/models/<slug>.md`) plus `models.md`
(the catalog). `gpt-3.5-turbo`, `gpt-4o`, `gpt-5-mini`, `gpt-5.4`,
`gpt-5.6-sol`, `daybreak-blue-latest`, `o3`, `text-embedding-3-small` and
`models.md` were captured 2026-08-18 (the prototype's fixtures); the other 88
pages were captured 2026-08-19 from the same URLs (byte-identical for the
overlapping pages). Together they are the complete set linked from models.md,
so an offline `modelroster update --provider openai --fixtures tests/fixtures
--dry-run` exercises every page.

## listings/

Captured 2026-08-23 (verbatim responses):

* `anthropic_models.json` — `GET https://api.anthropic.com/v1/models?limit=100`
* `openai_models.json` — `GET https://api.openai.com/v1/models`
* `nvidia_models.json` — `GET https://integrate.api.nvidia.com/v1/models`
* `inception_models.json` — `GET https://api.inceptionlabs.ai/v1/models`

Reference-shaped (no API key was available to capture them; hand-written from
each provider's public API reference, same field names and types, a handful
of representative models). **Replace with captured responses when a key is
available** — `modelroster capture --provider <name>` writes them:

* `xai_models.json`, `xai_language_models.json` — `GET /v1/models`, `GET /v1/language-models`
* `mistral_models.json` — `GET https://api.mistral.ai/v1/models`
* `google_models.json`, `google_native_models.json` — OpenAI-compat shim and native `/v1beta/models`
* `cohere_models.json` — `GET https://api.cohere.com/v1/models?page_size=1000`
* `ollama_tags.json`, `ollama_show/<name>.json` — local daemon `GET /api/tags`, `POST /api/show`
  (file name: `:` -> `--`, `/` -> `__`)

## discovery/

Captured 2026-08-23: `hf_models.json` (Hugging Face `/api/models?pipeline_tag=text-generation&sort=downloads&limit=50`),
`ollama_library.html` (`https://ollama.com/library`), `ngc_nim.json` (NGC catalog search, query "nim", page size 50).
