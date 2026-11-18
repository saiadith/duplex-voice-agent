"""Mock providers with configurable latency, jitter and failure rate.
Used for tests, load tests and chaos tests at zero cost."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..audio import FRAME_BYTES, FRAME_MS


@dataclass
class Latency:
    mean_ms: float
    jitter_ms: float = 0.0
    fail_rate: float = 0.0

    async def wait(self, rng: random.Random) -> None:
        if rng.random() < self.fail_rate:
            raise ConnectionError("injected provider failure")
        d = max(0.0, rng.gauss(self.mean_ms, self.jitter_ms))
        await asyncio.sleep(d / 1000)


class MockSTTStream:
    def __init__(self, text: str, latency: Latency, rng: random.Random, revise_rate: float = 0.0):
        self._text, self._lat, self._rng = text, latency, rng
        self._bytes = 0
        # real streaming STT sometimes rewrites the partial at finalisation
        self._final = text + " please" if rng.random() < revise_rate else text

    async def send(self, pcm: bytes) -> None:
        self._bytes += len(pcm)

    def partial(self) -> str:
        # reveal words proportionally to audio received (~6 words/sec)
        words = self._text.split()
        n = min(len(words), int(self._bytes / 32_000 * 6))
        return " ".join(words[:n])

    async def finalize(self) -> str:
        await self._lat.wait(self._rng)
        return self._final

    async def aclose(self) -> None:
        pass


@dataclass
class MockSTT:
    transcripts: list[str]
    latency: Latency = field(default_factory=lambda: Latency(80, 15))
    seed: int = 0
    revise_rate: float = 0.0
    name: str = "mock-stt"

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._i = 0

    async def open(self) -> MockSTTStream:
        text = self.transcripts[self._i % len(self.transcripts)]
        self._i += 1
        return MockSTTStream(text, self.latency, self._rng, self.revise_rate)


@dataclass
class MockLLM:
    reply: str = "Sure, I can help with that. Your balance is due on the fifth. Anything else?"
    ttft: Latency = field(default_factory=lambda: Latency(120, 30))
    per_token_ms: float = 8.0
    seed: int = 0
    name: str = "mock-llm"

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        await self.ttft.wait(self._rng)
        for i, w in enumerate(self.reply.split(" ")):
            if i:
                await asyncio.sleep(self.per_token_ms / 1000)
            yield (" " if i else "") + w


@dataclass
class MockTTS:
    ttfb: Latency = field(default_factory=lambda: Latency(90, 20))
    ms_per_char: float = 65.0  # ~15 chars/s speaking rate
    seed: int = 0
    name: str = "mock-tts"

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        await self.ttfb.wait(self._rng)
        n_frames = max(1, int(len(text) * self.ms_per_char / FRAME_MS))
        frame = b"\x10\x00" * (FRAME_BYTES // 2)
        for i in range(n_frames):
            if i and i % 10 == 0:
                await asyncio.sleep(0)  # synthesis faster than real time
            yield frame
