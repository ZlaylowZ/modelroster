"""Shared helpers: fixture loading, fake keys, offline pipeline runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from modelroster import clear_cache
from modelroster.providers.openai_docs import parse_model_page

FIXTURES = Path(__file__).parent / "fixtures"
DOCS = FIXTURES / "openai_docs"
LISTINGS = FIXTURES / "listings"

FAKE_KEYS = {"ANTHROPIC_API_KEY": "test", "OPENAI_API_KEY": "test", "XAI_API_KEY": "test",
             "MISTRAL_API_KEY": "test", "GOOGLE_API_KEY": "test", "COHERE_API_KEY": "test"}


def load_doc(slug: str) -> str:
    return (DOCS / (slug + ".md")).read_text("utf-8")


def parse(slug: str) -> dict:
    return parse_model_page(load_doc(slug), documentation_url="https://developers.openai.com/api/docs/models/%s.md" % slug, slug=slug)


def docs(*slugs: str) -> dict[str, dict]:
    out = {}
    for s in slugs:
        r = parse(s)
        out[r["canonical_model_id"]] = r
    return out


def api(*ids: str, created: int = 1700000000) -> list[dict]:
    return [{"id": i, "created": created + n, "owned_by": "system", "shutdown_date": None} for n, i in enumerate(ids)]


def listing(name: str):
    return json.loads((LISTINGS / name).read_text("utf-8"))


@pytest.fixture
def fake_keys(monkeypatch):
    for k, v in FAKE_KEYS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MODELROSTER_DATA_DIR", raising=False)
    yield


@pytest.fixture(autouse=True)
def _clear_registry_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


def run(provider: str, data_dir, **kw):
    from modelroster.update import run_provider
    kw.setdefault("fixtures_root", FIXTURES)
    kw.setdefault("quiet", True)
    return run_provider(provider, data_dir=data_dir, **kw)


@pytest.fixture(scope="session")
def populated(tmp_path_factory):
    """A data dir with every provider updated from fixtures (session-scoped: fast)."""
    d = tmp_path_factory.mktemp("data")
    old = {k: os.environ.get(k) for k in FAKE_KEYS}
    os.environ.update(FAKE_KEYS)
    try:
        for p in ("anthropic", "openai", "xai", "mistral", "google", "cohere", "nvidia", "inception", "ollama"):
            r = run(p, d)
            assert r.ok, (p, r.errors)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return d


@pytest.fixture
def registry(populated):
    from modelroster import load
    return load(data_dir=populated, force=True)
