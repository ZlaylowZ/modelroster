"""Fetcher behaviour with a mock transport: retries, Retry-After, 4xx, cache, offline, content-hash."""

import json

import httpx
import pytest

from modelroster.http import Fetcher, FetchError, FixtureFetcher


def make(responses, tmp_path, **kw):
    calls = []

    def handler(request):
        calls.append(request)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        status, body, headers = r
        return httpx.Response(status, text=body, headers=headers, request=request)

    f = Fetcher(tmp_path / "cache", transport=httpx.MockTransport(handler), sleep=lambda s: kw.setdefault("slept", []).append(s), **{k: v for k, v in kw.items() if k != "slept"})
    return f, calls


def test_retry_on_5xx_then_success(tmp_path):
    f, calls = make([(503, "", {"retry-after": "2"}), (200, "hello", {"content-type": "text/plain"})], tmp_path)
    text, meta = f.get_text("https://x.test/a")
    assert text == "hello" and meta["changed"] and not meta["from_cache"] and len(calls) == 2


def test_no_retry_on_404(tmp_path):
    f, calls = make([(404, "nope", {})], tmp_path)
    with pytest.raises(FetchError) as ei:
        f.get_text("https://x.test/a")
    assert ei.value.status == 404 and len(calls) == 1


def test_cache_and_content_hash_change_detection(tmp_path):
    f, calls = make([(200, "v1", {}), (200, "v1", {}), (200, "v2", {})], tmp_path)
    assert f.get_text("https://x.test/a")[1]["changed"] is True
    assert f.get_text("https://x.test/a")[1]["changed"] is False
    assert f.get_text("https://x.test/a")[1]["changed"] is True
    metas = list((tmp_path / "cache").glob("*.meta.json"))
    assert len(metas) == 1 and json.loads(metas[0].read_text())["content_hash"]


def test_etag_conditional_get(tmp_path):
    f, calls = make([(200, "v1", {"etag": '"abc"'}), (304, "", {})], tmp_path)
    f.get_text("https://x.test/a")
    text, meta = f.get_text("https://x.test/a")
    assert text == "v1" and meta["status"] == 304 and meta["from_cache"]
    assert calls[1].headers["if-none-match"] == '"abc"'


def test_offline_serves_cache_only(tmp_path):
    f, _ = make([(200, "v1", {})], tmp_path)
    f.get_text("https://x.test/a")
    off = Fetcher(tmp_path / "cache", offline=True, transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(AssertionError("network!"))))
    assert off.get_text("https://x.test/a")[0] == "v1"
    with pytest.raises(FetchError, match="offline"):
        off.get_text("https://x.test/other")
    with pytest.raises(FetchError, match="offline"):
        off.post_json("https://x.test/p", {"a": 1})


def test_stale_cache_fallback_after_retries(tmp_path):
    f, _ = make([(200, "v1", {})] + [(500, "", {})] * 5, tmp_path, max_retries=4)
    f.get_text("https://x.test/a")
    text, meta = f.get_text("https://x.test/a")
    assert text == "v1" and meta.get("stale")


def test_transport_error_exhausts_to_error(tmp_path):
    f, _ = make([httpx.ConnectError("down")] * 3, tmp_path, max_retries=2)
    with pytest.raises(FetchError, match="down"):
        f.get_text("https://x.test/a")


def test_backoff_capped_and_retry_after_honoured(tmp_path):
    slept = []
    f = Fetcher(tmp_path / "c", max_retries=3, backoff_base=100, sleep=slept.append,
                transport=httpx.MockTransport(lambda r: httpx.Response(429, headers={"retry-after": "7"} if len(slept) == 0 else {})))
    with pytest.raises(FetchError):
        f.get_text("https://x.test/a")
    assert slept[0] == 7.0 and all(s <= 30.0 for s in slept)


def test_content_type_check_and_empty_body(tmp_path):
    f, _ = make([(200, "<html>", {"content-type": "text/html"})], tmp_path)
    with pytest.raises(FetchError, match="content-type"):
        f.get_text("https://x.test/a", expect_content_type="text/markdown")
    f, _ = make([(200, "  ", {})], tmp_path)
    with pytest.raises(FetchError, match="empty"):
        f.get_text("https://x.test/b")


def test_get_many_and_json_and_post(tmp_path):
    f, calls = make([(200, "a", {}), (200, "b", {}), (200, '{"k": 1}', {}), (200, '{"p": 2}', {}), (200, "{", {})], tmp_path, concurrency=1)
    out = f.get_many(["https://x.test/1", "https://x.test/2"])
    assert out["https://x.test/1"][0] == "a" and out["https://x.test/2"][0] == "b"
    assert f.get_json("https://x.test/j") == {"k": 1}
    assert f.post_json("https://x.test/p", {"model": "m"}) == {"p": 2}
    assert calls[-1].method == "POST" and json.loads(calls[-1].content) == {"model": "m"}
    with pytest.raises(FetchError, match="invalid JSON"):
        f.get_json("https://x.test/bad")


def test_user_agent_names_package(tmp_path):
    f, calls = make([(200, "x", {})], tmp_path)
    f.get_text("https://x.test/a")
    assert calls[0].headers["user-agent"].startswith("modelroster/")


def test_fixture_fetcher_records_requests():
    ff = FixtureFetcher({"https://x.test/a": "body", 'https://x.test/p#{"m": 1}': '{"ok": true}'})
    assert ff.get_text("https://x.test/a")[0] == "body"
    assert ff.post_json("https://x.test/p", {"m": 1}) == {"ok": True}
    with pytest.raises(FetchError):
        ff.get_text("https://x.test/missing")
    assert ff.requested[0] == "https://x.test/a"
