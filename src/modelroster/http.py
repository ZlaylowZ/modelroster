"""
The single HTTP layer every provider and discovery source uses.

  * descriptive User-Agent naming the package and version
  * connect/read timeouts, redirects followed
  * retries with exponential backoff + jitter (capped at 30 s) for transport
    errors, HTTP 429 and 5xx; `Retry-After` honoured; NO retry on other 4xx
  * on-disk cache keyed by URL hash (`<sha1>.body` + `<sha1>.meta.json`)
  * conditional GET (ETag / Last-Modified) when upstream supports it and
    content-hash change detection when it does not (`meta["changed"]`)
  * `offline=True` serves every request from cache and never opens a socket
  * stale-cache fallback once retries are exhausted
  * bounded concurrency for `get_many`

Responses are cached by URL only; requests that carry credentials are cached
too (so `--offline` can replay them) but the credentials themselves are never
written to disk. Set `cache=False` on a call to opt out.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import httpx

from ._version import __version__

USER_AGENT = "modelroster/%s (+https://github.com/ZlaylowZ/modelroster)" % __version__
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF = 30.0


class FetchError(Exception):
    """Raised when a resource cannot be fetched after retries."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Fetcher:
    def __init__(self, cache_dir: str | os.PathLike | None = None, *, max_retries: int = 4,
                 backoff_base: float = 0.8, concurrency: int = 6, offline: bool = False,
                 no_cache: bool = False, timeout: httpx.Timeout | float | None = None,
                 transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.concurrency = max(1, int(concurrency))
        self.offline = offline
        self.no_cache = no_cache or self.cache_dir is None
        self._sleep_fn = sleep
        self._client = httpx.Client(headers={"User-Agent": USER_AGENT},
                                    timeout=timeout or DEFAULT_TIMEOUT,
                                    follow_redirects=True, transport=transport)

    # ── cache helpers ──────────────────────────────────────────────
    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        assert self.cache_dir is not None
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / (key + ".body"), self.cache_dir / (key + ".meta.json")

    def _read_cache(self, url: str):
        if self.no_cache:
            return None, None
        body_p, meta_p = self._cache_paths(url)
        if not body_p.exists() or not meta_p.exists():
            return None, None
        try:
            return body_p.read_text("utf-8"), json.loads(meta_p.read_text("utf-8"))
        except Exception:
            return None, None

    def _write_cache(self, url: str, text: str, response: httpx.Response, content_hash: str) -> None:
        if self.no_cache:
            return
        body_p, meta_p = self._cache_paths(url)
        meta = {
            "url": url,
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
            "content_type": response.headers.get("content-type"),
            "content_hash": content_hash,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": response.status_code,
        }
        tmp_b = body_p.with_suffix(".body.tmp")
        tmp_m = meta_p.with_suffix(".meta.json.tmp")
        tmp_b.write_text(text, "utf-8")
        tmp_m.write_text(json.dumps(meta, indent=1), "utf-8")
        os.replace(tmp_b, body_p)
        os.replace(tmp_m, meta_p)

    # ── core fetch ─────────────────────────────────────────────────
    def get_text(self, url: str, *, headers: dict[str, str] | None = None,
                 params: dict[str, Any] | None = None, expect_content_type: str | None = None,
                 cache: bool = True) -> tuple[str, dict[str, Any]]:
        """
        Return (text, meta). `meta` has at least `from_cache`, `status`, and
        `changed` (False when the body hash equals the cached one).
        """
        return self._request("GET", url, headers=headers, params=params,
                             expect_content_type=expect_content_type, cache=cache)

    def post_json(self, url: str, body: Any, *, headers: dict[str, str] | None = None,
                  cache: bool = True) -> Any:
        """POST a JSON body and decode the JSON reply (cached under url + body hash)."""
        text, _ = self._request("POST", url, headers=headers, json_body=body, cache=cache)
        try:
            return json.loads(text)
        except ValueError as exc:
            raise FetchError("invalid JSON from %s: %s" % (url, exc)) from exc

    def _request(self, method: str, url: str, *, headers=None, params=None, json_body=None,
                 expect_content_type=None, cache=True) -> tuple[str, dict[str, Any]]:
        if params:
            url = str(httpx.URL(url, params=params))
        cache_key = url if json_body is None else url + "#" + hashlib.sha1(
            json.dumps(json_body, sort_keys=True).encode("utf-8")).hexdigest()
        cached_text, cached_meta = self._read_cache(cache_key) if cache else (None, None)

        if self.offline:
            if cached_text is None:
                raise FetchError("offline mode and no cached copy of %s" % url)
            return cached_text, {"from_cache": True, "status": 200, "changed": False,
                                 "etag": (cached_meta or {}).get("etag")}

        req_headers = dict(headers or {})
        if cached_meta:
            if cached_meta.get("etag"):
                req_headers["If-None-Match"] = cached_meta["etag"]
            if cached_meta.get("last_modified"):
                req_headers["If-Modified-Since"] = cached_meta["last_modified"]

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(method, url, headers=req_headers, json=json_body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                self._sleep(attempt)
                continue

            if resp.status_code == 304 and cached_text is not None:
                return cached_text, {"from_cache": True, "status": 304, "changed": False,
                                     "etag": (cached_meta or {}).get("etag")}

            if resp.status_code in RETRY_STATUSES:
                last_exc = FetchError("HTTP %d for %s" % (resp.status_code, url), resp.status_code)
                self._sleep(attempt, resp.headers.get("retry-after"))
                continue

            if 400 <= resp.status_code < 500:
                raise FetchError("HTTP %d for %s: %s" % (resp.status_code, url, resp.text[:200]),
                                 resp.status_code)

            if resp.status_code != 200:
                last_exc = FetchError("HTTP %d for %s" % (resp.status_code, url), resp.status_code)
                self._sleep(attempt)
                continue

            ctype = (resp.headers.get("content-type") or "").lower()
            if expect_content_type and expect_content_type not in ctype:
                raise FetchError("unexpected content-type %r for %s" % (ctype, url), resp.status_code)

            text = resp.text
            if not text or not text.strip():
                raise FetchError("empty body for %s" % url, resp.status_code)

            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            changed = (cached_meta or {}).get("content_hash") != digest
            if cache:
                self._write_cache(cache_key, text, resp, digest)
            return text, {"from_cache": False, "status": 200, "changed": changed,
                          "etag": resp.headers.get("etag"), "content_hash": digest}

        if cached_text is not None:
            return cached_text, {"from_cache": True, "status": 0, "stale": True, "changed": False,
                                 "error": str(last_exc)}
        raise FetchError("failed to fetch %s: %s" % (url, last_exc),
                         getattr(last_exc, "status", None))

    def get_json(self, url: str, **kw: Any) -> Any:
        text, _meta = self.get_text(url, **kw)
        try:
            return json.loads(text)
        except ValueError as exc:
            raise FetchError("invalid JSON from %s: %s" % (url, exc)) from exc

    def get_many(self, urls: Iterable[str], **kw: Any) -> dict[str, Any]:
        """Fetch many URLs concurrently. Returns {url: (text, meta) | FetchError}."""
        results: dict[str, Any] = {}

        def one(u: str):
            try:
                return u, self.get_text(u, **kw)
            except FetchError as exc:
                return u, exc

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for u, r in ex.map(one, list(urls)):
                results[u] = r
        return results

    def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        if attempt >= self.max_retries:
            return
        delay = None
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = None
        if delay is None:
            delay = self.backoff_base * (2 ** attempt) + random.uniform(0, 0.3)
        self._sleep_fn(min(delay, MAX_BACKOFF))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FixtureFetcher:
    """
    Offline stand-in used by tests and by `--offline` fixture runs: serves
    bodies from a {url: text-or-path} map and raises FetchError for anything
    else. Shares the `get_text`/`get_json`/`get_many` surface of `Fetcher`.
    """

    def __init__(self, pages: dict[str, str | os.PathLike], content_type: str = "text/markdown"):
        self.pages = dict(pages)
        self.content_type = content_type
        self.requested: list[str] = []

    def _body(self, url: str) -> str:
        src = self.pages[url]
        if isinstance(src, (str,)) and ("\n" in src or not os.path.exists(src)):
            return src
        return Path(src).read_text("utf-8")

    def get_text(self, url: str, *, headers=None, params=None, expect_content_type=None, cache=True):
        if params:
            url = str(httpx.URL(url, params=params))
        self.requested.append(url)
        if url not in self.pages:
            raise FetchError("no fixture for %s" % url, 404)
        return self._body(url), {"from_cache": True, "status": 200, "changed": False, "fixture": True}

    def get_json(self, url: str, **kw: Any) -> Any:
        text, _ = self.get_text(url, **kw)
        return json.loads(text)

    def post_json(self, url: str, body: Any, *, headers=None, cache=True) -> Any:
        """Fixture lookup for POSTs: '<url>#<json body>' first, then the bare url."""
        key = url + "#" + json.dumps(body, sort_keys=True)
        self.requested.append(key)
        src = key if key in self.pages else url
        if src not in self.pages:
            raise FetchError("no fixture for POST %s" % key, 404)
        return json.loads(self._body(src))

    def get_many(self, urls, **kw):
        out = {}
        for u in urls:
            try:
                out[u] = self.get_text(u, **kw)
            except FetchError as exc:
                out[u] = exc
        return out

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
