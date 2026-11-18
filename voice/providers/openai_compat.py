"""Streaming LLM adapter for any OpenAI-compatible endpoint.

Works with Smallest.ai Electron (base_url=https://api.smallest.ai/waves/v1,
model="electron"), Groq, OpenRouter, Gemini's OpenAI-compatible endpoint, etc.
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx


class OpenAICompatLLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, max_tokens: int = 150, timeout_s: float = 10.0):
        self.base_url = (base_url or os.environ["LLM_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["LLM_API_KEY"]
        self.model = model or os.environ.get("LLM_MODEL", "electron")
        self.max_tokens = max_tokens
        self.name = f"openai-compat:{self.model}"
        self._client = httpx.AsyncClient(timeout=timeout_s)  # reused: keeps TLS warm

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": self.max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                       json=body, headers=headers) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
