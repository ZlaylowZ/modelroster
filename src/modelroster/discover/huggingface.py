"""
Hugging Face Hub: `GET https://huggingface.co/api/models` filtered by
pipeline tag (default text-generation), sorted by downloads, capped.
Only Hub metadata is recorded (tags, downloads, likes, gated, library);
capabilities remain None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ..schema import prov
from .base import DiscoverySource

API_URL = "https://huggingface.co/api/models"
MAX_LIMIT = 1000


class HuggingFaceSource(DiscoverySource):
    name = "huggingface"
    describe = "Hub /api/models by pipeline_tag, sorted by downloads"
    default_limit = 200

    def url(self, pipeline_tag: str = "text-generation", limit: int | None = None, **params: Any) -> str:
        q = {"pipeline_tag": pipeline_tag, "sort": "downloads", "direction": -1,
             "limit": min(limit or self.default_limit, MAX_LIMIT)}
        q.update(params)
        return str(httpx.URL(API_URL, params=q))

    def fixtures(self, root: Path):
        return {self.url(limit=50): root / "discovery" / "hf_models.json"}

    def discover(self, http: Any, limit: int | None = None, pipeline_tag: str = "text-generation", **kw: Any):
        data = http.get_json(self.url(pipeline_tag=pipeline_tag, limit=limit, **kw))
        out = []
        for m in data if isinstance(data, list) else []:
            mid = m.get("id") or m.get("modelId")
            if not mid:
                continue
            rec = self.record(mid, display_name=mid, sources={"listing": API_URL, "documentation": "https://huggingface.co/" + mid})
            tags = m.get("tags") or []
            rec.raw = {"downloads": m.get("downloads"), "likes": m.get("likes"), "pipeline_tag": m.get("pipeline_tag"),
                       "library_name": m.get("library_name"), "gated": m.get("gated"), "private": m.get("private"),
                       "created_at": m.get("createdAt"), "tags": tags}
            if m.get("createdAt"):
                rec.released = str(m["createdAt"])[:10]
            if "conversational" in tags:
                rec.modalities["text"] = {"input": True, "output": True}
                rec.provenance["modalities.text"] = prov("hub:tags", "conversational tag")
            if m.get("pipeline_tag") == "text-generation":
                rec.capabilities.extra["hf.pipeline_tag"] = "text-generation"
            out.append(rec)
        return out
