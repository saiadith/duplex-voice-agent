"""Provider interfaces. Every STT/LLM/TTS backend implements one of these, so
providers (Smallest.ai, Deepgram, Groq, local CPU fallbacks, mocks) are swappable
by config and comparable in the same benchmark."""
from __future__ import annotations

from typing import AsyncIterator, Protocol


class STTStream(Protocol):
    async def send(self, pcm: bytes) -> None: ...
    def partial(self) -> str: ...
    async def finalize(self) -> str: ...
    async def aclose(self) -> None: ...


class STTProvider(Protocol):
    name: str

    async def open(self) -> STTStream: ...


class LLMProvider(Protocol):
    name: str

    def stream(self, messages: list[dict]) -> AsyncIterator[str]: ...


class TTSProvider(Protocol):
    name: str

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield 16 kHz PCM16 audio in 20 ms frames."""
        ...
