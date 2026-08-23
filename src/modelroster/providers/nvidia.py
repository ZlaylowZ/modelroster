"""
NVIDIA NIM hosted endpoints: `GET https://integrate.api.nvidia.com/v1/models`
(OpenAI-compatible; publicly listable, a key is only needed to call models).
The listing carries ids and owners only, so capabilities stay None.
"""

from __future__ import annotations

from .openai_compat import OpenAICompatProvider


class NvidiaProvider(OpenAICompatProvider):
    name = "nvidia"
    auth = ("NVIDIA_API_KEY",)
    base_url = "https://integrate.api.nvidia.com/v1"
    describe = "GET /v1/models (public OpenAI-compatible listing; ids only)"

    def headers(self):
        key = self.api_key()
        return {"Authorization": "Bearer " + key} if key else {}
