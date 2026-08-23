"""
NVIDIA NIM catalog (NGC): `https://api.ngc.nvidia.com/v2/search/catalog/resources/CONTAINER`
search restricted to the `nim` org. Records carry the NGC resource id,
display name, description, labels and dates; capabilities remain None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from ..http import FetchError
from .base import DiscoverySource

SEARCH_URL = "https://api.ngc.nvidia.com/v2/search/catalog/resources/CONTAINER"
PAGE_SIZE = 50


class NvidiaNimSource(DiscoverySource):
    name = "nvidia_nim"
    describe = "NGC catalog containers published under the nim org"
    default_limit = 200

    def url(self, page: int = 0, page_size: int = PAGE_SIZE, query: str = "nim") -> str:
        q: dict[str, Any] = {"query": query, "pageSize": page_size}
        if page:
            q["page"] = page
        return str(httpx.URL(SEARCH_URL, params={"q": json.dumps(q, separators=(",", ":"))}))

    def fixtures(self, root: Path):
        return {self.url(): root / "discovery" / "ngc_nim.json"}

    def discover(self, http: Any, limit: int | None = None, max_pages: int = 20, **kw: Any):
        limit = limit or self.default_limit
        out, seen = [], set()
        for page in range(max_pages):
            try:
                data = http.get_json(self.url(page=page))
            except FetchError:
                if page == 0:
                    raise
                break  # partial catalog is still useful for discovery
            resources = []
            for group in data.get("results") or []:
                resources.extend(group.get("resources") or [])
            if not resources:
                break
            for r in resources:
                rid = r.get("resourceId")
                if not rid or r.get("orgName") != "nim" or rid in seen:
                    continue
                seen.add(rid)
                rec = self.record(rid, display_name=r.get("displayName"), description=r.get("description"),
                                  sources={"listing": SEARCH_URL, "documentation": "https://catalog.ngc.nvidia.com/orgs/nim/containers/" + rid.split("/", 1)[-1]})
                rec.released = (r.get("dateCreated") or "")[:10] or None
                rec.raw = {"team": r.get("teamName"), "name": r.get("name"), "labels": r.get("labels"),
                           "date_modified": r.get("dateModified"), "is_public": r.get("isPublic")}
                out.append(rec)
                if len(out) >= limit:
                    return out
            total_pages = data.get("resultPageTotal")
            if total_pages is not None and page + 1 >= total_pages:
                break
        return out
