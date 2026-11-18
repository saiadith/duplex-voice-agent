"""VoiceSession: the full-duplex conversation loop for one call.

Inbound:  20 ms PCM frames -> VAD -> (user turn) -> STT stream -> endpointer
Outbound: STT final -> LLM token stream -> clause chunker -> TTS stream ->
          paced audio out (playback clock)

Barge-in: if the user starts speaking while the agent is talking, the response
task is cancelled (LLM + TTS), the client buffer is cleared, and the assistant
turn in the history is truncated to what the user actually *heard*, estimated
from the playback clock.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable

from .audio import FRAME_MS
from .metrics import TurnMetrics
from .providers.base import LLMProvider, STTProvider, TTSProvider
from .turn import VAD, Endpointer

SendAudio = Callable[[bytes], Awaitable[None]]
SendEvent = Callable[[dict], Awaitable[None]]

_CLAUSE_END = re.compile(r"[.!?;:]\s*$")


@dataclass
class SessionConfig:
    system_prompt: str = "You are a concise, friendly phone agent. Reply in one or two short sentences."
    min_speech_frames: int = 3       # 60 ms of speech to count as onset (debounce)
    preroll_frames: int = 10         # 200 ms of audio before onset sent to STT
    max_lead_ms: int = 200           # how far ahead of real time we push audio to the client
    clause_min_chars: int = 30       # flush at a comma once buffer is this long
    barge_in_enabled: bool = True
    speculate_ms: int | None = None  # start LLM on the partial transcript after this much silence


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split())


class _Speculation:
    """An LLM call started on a partial transcript; tokens are buffered until the
    endpoint confirms (reuse) or the user keeps talking / STT revises (discard)."""

    def __init__(self, text: str, tokens: AsyncIterator[str]):
        self.text, self.q, self.n_tokens = text, asyncio.Queue(), 0
        self.task = asyncio.create_task(self._pump(tokens))

    async def _pump(self, tokens):
        try:
            async for t in tokens:
                self.n_tokens += 1
                await self.q.put(t)
        finally:
            self.q.put_nowait(None)

    async def stream(self):
        while (t := await self.q.get()) is not None:
            yield t

    def cancel(self):
        self.task.cancel()


async def clause_chunks(tokens: AsyncIterator[str], min_chars: int, on_first_token=None):
    """Group streamed tokens into speakable chunks: sentence ends always flush;
    commas flush once the buffer is long enough. Smaller first chunk = lower TTFB."""
    buf = ""
    async for tok in tokens:
        if on_first_token:
            on_first_token()
            on_first_token = None
        buf += tok
        if _CLAUSE_END.search(buf) or (buf.rstrip().endswith(",") and len(buf) >= min_chars):
            yield buf.strip()
            buf = ""
    if buf.strip():
        yield buf.strip()


class VoiceSession:
    def __init__(self, stt: STTProvider, llm: LLMProvider, tts: TTSProvider, vad: VAD,
                 endpointer: Endpointer, send_audio: SendAudio,
                 send_event: SendEvent | None = None, cfg: SessionConfig | None = None):
        self.stt, self.llm, self.tts, self.vad, self.endpointer = stt, llm, tts, vad, endpointer
        self.send_audio = send_audio
        self.send_event = send_event or (lambda e: asyncio.sleep(0))
        self.cfg = cfg or SessionConfig()
        self.history: list[dict] = [{"role": "system", "content": self.cfg.system_prompt}]
        self.turns: list[TurnMetrics] = []
        self.errors: list[str] = []

        self._preroll: deque[bytes] = deque(maxlen=self.cfg.preroll_frames)
        self._onset_frames = 0
        self._silence_frames = 0
        self._last_voiced_ts = 0.0
        self._stt_stream = None
        self._turn: TurnMetrics | None = None
        self._response_task: asyncio.Task | None = None
        self._spec: _Speculation | None = None
        self.spec_stats = {"started": 0, "hit": 0, "miss_revised": 0, "miss_resumed": 0, "wasted_tokens": 0}
        # playback bookkeeping for the current response
        self._play_start: float | None = None
        self._sent_ms = 0.0
        self._spoken: list[list] = []  # [chunk text, audio ms sent, complete]

    # ------------------------------------------------------------------ inbound
    @property
    def agent_speaking(self) -> bool:
        return self._response_task is not None and not self._response_task.done()

    async def push_frame(self, frame: bytes) -> None:
        now = time.perf_counter()
        voiced = self.vad.is_speech(frame)

        if self._stt_stream is None:  # not in a user turn
            self._preroll.append(frame)
            self._onset_frames = self._onset_frames + 1 if voiced else 0
            if voiced:
                self._last_voiced_ts = now
            if self._onset_frames >= self.cfg.min_speech_frames:
                await self._start_user_turn()
            return

        await self._stt_stream.send(frame)
        if voiced:
            self._silence_frames = 0
            self._last_voiced_ts = now
            if self._spec:  # user resumed: the speculative reply is for an unfinished turn
                self._drop_spec("miss_resumed")
            return
        self._silence_frames += 1
        silence_ms = self._silence_frames * FRAME_MS
        partial = self._stt_stream.partial()
        if (self.cfg.speculate_ms is not None and self._spec is None and partial
                and silence_ms >= self.cfg.speculate_ms):
            msgs = list(self.history) + [{"role": "user", "content": partial}]
            self._spec = _Speculation(partial, self.llm.stream(msgs))
            self.spec_stats["started"] += 1
        if self.endpointer.should_end(silence_ms, partial):
            await self._end_user_turn(now)

    async def _start_user_turn(self) -> None:
        if self.agent_speaking:
            if not self.cfg.barge_in_enabled:
                self._onset_frames = 0
                return
            await self._barge_in()
        self._turn = TurnMetrics(turn_id=len(self.turns))
        self.turns.append(self._turn)
        self._stt_stream = await self.stt.open()
        for f in self._preroll:
            await self._stt_stream.send(f)
        self._preroll.clear()
        self._onset_frames = self._silence_frames = 0
        await self.send_event({"type": "user_turn_start", "turn": self._turn.turn_id})

    def _drop_spec(self, reason: str) -> None:
        self._spec.cancel()
        self.spec_stats[reason] += 1
        self.spec_stats["wasted_tokens"] += self._spec.n_tokens
        self._spec = None

    async def _end_user_turn(self, now: float) -> None:
        turn, stream = self._turn, self._stt_stream
        turn.mark("speech_end", self._last_voiced_ts)
        turn.mark("endpoint", now)
        self._stt_stream = None
        spec, self._spec = self._spec, None
        self._response_task = asyncio.create_task(self._respond(stream, turn, spec))

    # ---------------------------------------------------------------- outbound
    async def _respond(self, stream, turn: TurnMetrics, spec: _Speculation | None = None) -> None:
        try:
            text = await stream.finalize()
        except Exception as e:  # provider failure: degrade, don't crash the call
            self.errors.append(f"stt: {e!r}")
            if spec:
                spec.cancel()
            return
        finally:
            await stream.aclose()
        turn.mark("stt_final", time.perf_counter())
        turn.user_text = text
        if not text.strip():
            return
        self.history.append({"role": "user", "content": text})
        self._play_start, self._sent_ms, self._spoken = None, 0.0, []

        if spec and _norm(spec.text) == _norm(text):
            tokens = spec.stream()
            turn.spec_hit = True
            self.spec_stats["hit"] += 1
        else:
            if spec:
                spec.cancel()
                self.spec_stats["miss_revised"] += 1
                self.spec_stats["wasted_tokens"] += spec.n_tokens
            tokens = None
        try:
            tokens = tokens or self.llm.stream(list(self.history))
            first = lambda: turn.mark("llm_first_token", time.perf_counter())
            async for chunk in clause_chunks(tokens, self.cfg.clause_min_chars, first):
                turn.mark("first_chunk", time.perf_counter())
                entry = [chunk, 0.0, False]  # text, audio ms sent, complete?
                self._spoken.append(entry)
                async for frame in self.tts.synthesize(chunk):
                    turn.mark("tts_first_byte", time.perf_counter())
                    await self._paced_send(frame)
                    turn.mark("first_audio_out", time.perf_counter())
                    entry[1] += FRAME_MS
                entry[2] = True
            self.history.append({"role": "assistant", "content": " ".join(e[0] for e in self._spoken)})
            await self.send_event({"type": "turn_metrics", "turn": turn.turn_id, **turn.stages_ms()})
        except asyncio.CancelledError:
            if spec:
                spec.cancel()
            turn.interrupted = True
            self.history.append({"role": "assistant", "content": self._heard_text(), "interrupted": True})
            raise
        except Exception as e:
            self.errors.append(f"response: {e!r}")
            await self.send_event({"type": "error", "where": "response", "detail": repr(e)})

    async def _paced_send(self, frame: bytes) -> None:
        """Keep at most `max_lead_ms` of audio buffered at the client, so a barge-in
        'clear' discards little audio and the playback clock stays accurate."""
        if self._play_start is None:
            self._play_start = time.perf_counter()
        elapsed_ms = (time.perf_counter() - self._play_start) * 1000
        ahead = self._sent_ms - elapsed_ms
        if ahead > self.cfg.max_lead_ms:
            await asyncio.sleep((ahead - self.cfg.max_lead_ms) / 1000)
        await self.send_audio(frame)
        self._sent_ms += FRAME_MS

    def _heard_ms(self) -> float:
        if self._play_start is None:
            return 0.0
        return min(self._sent_ms, (time.perf_counter() - self._play_start) * 1000)

    def _heard_text(self) -> str:
        """Map heard audio duration back onto the spoken text (proportional by chars)."""
        heard, out = self._heard_ms(), []
        done = [(t, ms) for t, ms, c in self._spoken if c and ms]
        # speaking rate from completed chunks; fallback ~15 chars/s
        ms_per_char = (sum(ms for _, ms in done) / sum(len(t) for t, _ in done)) if done else 65.0
        for text, ms, complete in self._spoken:
            if complete and heard >= ms:
                out.append(text)
                heard -= ms
            else:
                # partially synthesised chunk: its full duration is estimated from the rate
                total = ms if complete else len(text) * ms_per_char
                n = int(len(text) * min(1.0, heard / total)) if total else 0
                if n:
                    out.append(text[:n].rsplit(" ", 1)[0] + "—")
                break
        return " ".join(out)

    async def _barge_in(self) -> None:
        task = self._response_task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._response_task = None
        await self.send_event({"type": "clear"})  # client/Twilio drops buffered audio

    async def close(self) -> None:
        if self.agent_speaking:
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
        if self._stt_stream:
            await self._stt_stream.aclose()

    async def wait_idle(self) -> None:
        if self._response_task:
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
