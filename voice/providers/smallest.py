"""Smallest.ai streaming adapters (protocols per docs.smallest.ai, Nov 2026).

Pulse STT : wss://api.smallest.ai/waves/v1/stt/live?model=pulse&language=en&encoding=linear16&sample_rate=16000
            binary PCM16 frames in; JSON {transcript, is_final, is_last} out; {"type":"close_stream"} to finish.
Lightning : wss://api.smallest.ai/waves/v1/tts/live
            JSON {text, voice_id, model, sample_rate, output_format:"pcm"} in;
            {"status":"chunk","data":{"audio":<b64 PCM16>}} ... {"status":"complete"} out.
Electron  : OpenAI-compatible -> use providers.openai_compat.OpenAICompatLLM(base_url=".../waves/v1", model="electron").
Auth: Authorization: Bearer <SMALLEST_API_KEY>.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import AsyncIterator

import websockets

from ..audio import FRAME_BYTES

STT_URL = "wss://api.smallest.ai/waves/v1/stt/live"
TTS_URL = "wss://api.smallest.ai/waves/v1/tts/live"


def _key(k):
    return k or os.environ["SMALLEST_API_KEY"]


class PulseStream:
    """One streaming STT session per user turn. Partials update continuously; finalize()
    sends close_stream and waits for is_last."""

    def __init__(self, ws):
        self.ws = ws
        self._final_parts: list[str] = []
        self._partial = ""
        self._last = asyncio.Event()
        self._reader = asyncio.create_task(self._read())
        self._buf = b""

    async def _read(self):
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                text = (m.get("transcript") or "").strip()
                if m.get("is_final"):
                    if text:
                        self._final_parts.append(text)
                    self._partial = ""
                else:
                    self._partial = text
                if m.get("is_last"):
                    break
        finally:
            self._last.set()

    async def send(self, pcm: bytes) -> None:
        self._buf += pcm
        if len(self._buf) >= 4096:  # docs recommend ~4 KB chunks
            await self.ws.send(self._buf)
            self._buf = b""

    def partial(self) -> str:
        return " ".join(self._final_parts + ([self._partial] if self._partial else []))

    async def finalize(self, timeout_s: float = 5.0) -> str:
        if self._buf:
            await self.ws.send(self._buf)
            self._buf = b""
        await self.ws.send(json.dumps({"type": "close_stream"}))
        await asyncio.wait_for(self._last.wait(), timeout_s)
        return " ".join(self._final_parts).strip()

    async def aclose(self) -> None:
        self._reader.cancel()
        await self.ws.close()


class PulseSTT:
    name = "smallest-pulse"

    def __init__(self, api_key: str | None = None, language: str = "en", url: str = STT_URL):
        self.key, self.language, self.url = _key(api_key), language, url

    async def open(self) -> PulseStream:
        q = f"?model=pulse&language={self.language}&encoding=linear16&sample_rate=16000"
        ws = await websockets.connect(self.url + q, additional_headers={"Authorization": f"Bearer {self.key}"},
                                      max_size=None)
        return PulseStream(ws)


class LightningTTS:
    name = "smallest-lightning"

    def __init__(self, api_key: str | None = None, voice_id: str = "meher", model: str = "lightning_v3.1",
                 url: str = TTS_URL):
        self.key, self.voice_id, self.model, self.url = _key(api_key), voice_id, model, url

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yields 16 kHz PCM16 in exact 20 ms frames (re-chunked from the server's variable chunks)."""
        async with websockets.connect(self.url, additional_headers={"Authorization": f"Bearer {self.key}"},
                                      max_size=None) as ws:
            await ws.send(json.dumps({"text": text, "voice_id": self.voice_id, "model": self.model,
                                      "sample_rate": 16000, "output_format": "pcm"}))
            buf = b""
            async for raw in ws:
                m = json.loads(raw)
                if m.get("status") == "chunk":
                    buf += base64.b64decode(m["data"]["audio"])
                    while len(buf) >= FRAME_BYTES:
                        yield buf[:FRAME_BYTES]
                        buf = buf[FRAME_BYTES:]
                elif m.get("status") == "complete":
                    break
                elif m.get("status") == "error" or "error" in m:
                    raise RuntimeError(f"lightning error: {m}")
            if buf:
                yield buf + b"\x00" * (FRAME_BYTES - len(buf))
