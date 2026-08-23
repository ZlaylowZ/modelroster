from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import ModelRecord, utc_now_iso


class DiscoverySource:
    name: str = ""
    describe: str = ""
    default_limit: int = 200

    def discover(self, http: Any, limit: int | None = None, **kw: Any) -> list[ModelRecord]:
        raise NotImplementedError

    def fixtures(self, root: Path) -> dict[str, str | Path] | None:
        return None

    def record(self, model_id: str, **kw: Any) -> ModelRecord:
        rec = ModelRecord(provider=self.name, model_id=model_id, tier="discovered", relationship="unknown",
                          retrieved_at=utc_now_iso(), **kw)
        rec.warn("discovered from %s; capabilities not verified against an official per-model source" % self.name)
        return rec
