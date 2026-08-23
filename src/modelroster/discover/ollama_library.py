"""
Ollama public library: `https://ollama.com/library` has no JSON API, so the
HTML is scanned for `/library/<name>` links. Only names are recorded; an
`ollama pull` plus the local provider is the verified path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import DiscoverySource

LIBRARY_URL = "https://ollama.com/library"
_RE_LINK = re.compile(r'href="/library/([a-z0-9][a-z0-9._-]*)"')


class OllamaLibrarySource(DiscoverySource):
    name = "ollama_library"
    describe = "names linked from ollama.com/library"
    default_limit = 500

    def fixtures(self, root: Path):
        return {LIBRARY_URL: root / "discovery" / "ollama_library.html"}

    def discover(self, http: Any, limit: int | None = None, **kw: Any):
        html, _ = http.get_text(LIBRARY_URL)
        names: list[str] = []
        for n in _RE_LINK.findall(html):
            if n not in names:
                names.append(n)
        if not names:
            raise ValueError("no /library/<name> links found on %s — page format changed?" % LIBRARY_URL)
        out = []
        for n in names[: limit or self.default_limit]:
            rec = self.record(n, display_name=n, sources={"listing": LIBRARY_URL, "documentation": LIBRARY_URL + "/" + n})
            out.append(rec)
        return out
