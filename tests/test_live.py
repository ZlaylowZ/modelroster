"""
Live integration check — hits the real provider endpoints. Skipped unless
MODELROSTER_LIVE=1. Providers without credentials are skipped by the updater,
so the test passes with any subset of keys; it fails only on fetch failures,
validation refusals, or a shipped-data regression.

    MODELROSTER_LIVE=1 pytest tests/test_live.py -q
"""

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("MODELROSTER_LIVE") != "1", reason="set MODELROSTER_LIVE=1 to run live checks")


def test_live_dry_run_all_providers(tmp_path):
    from modelroster.update import load_dotenv_if_available, run_many, worst_code
    load_dotenv_if_available()
    results = run_many(None, data_dir=tmp_path, dry_run=True, quiet=True)
    for r in results:
        print(r.summary())
    assert worst_code(results) == 0
    ran = [r for r in results if not r.skipped]
    assert ran, "no provider had credentials or a public listing"


def test_live_discovery(tmp_path):
    from modelroster import discover
    from modelroster.http import Fetcher
    with Fetcher(tmp_path / "cache") as http:
        for name in discover.sources():
            assert discover.run(name, http, limit=5)
