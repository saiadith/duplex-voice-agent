"""Voice activity detection and end-of-turn (endpointing) policies.

`Endpointer` is the seam where the learned semantic turn detector plugs in:
it sees silence duration *and* the partial transcript, so a model can wait
longer after "my account number is..." than after "thanks, bye."
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class VAD(Protocol):
    def is_speech(self, frame: bytes) -> bool: ...


class EnergyVAD:
    """RMS-threshold VAD. Zero-dependency baseline; use SileroVAD in production."""

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def is_speech(self, frame: bytes) -> bool:
        x = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x))) > self.threshold


class SileroVAD:
    """Silero VAD via ONNX (CPU, <1 ms/frame). Requires `pip install silero-vad`."""

    def __init__(self, threshold: float = 0.5):
        from silero_vad import load_silero_vad  # lazy import
        import torch

        self._torch = torch
        self.model = load_silero_vad(onnx=True)
        self.threshold = threshold
        self._buf = b""

    def is_speech(self, frame: bytes) -> bool:
        # Silero needs 512-sample windows at 16 kHz; keep last result between windows.
        self._buf += frame
        prob = getattr(self, "_last", 0.0)
        while len(self._buf) >= 1024:
            chunk, self._buf = self._buf[:1024], self._buf[1024:]
            x = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
            prob = float(self.model(self._torch.from_numpy(x), 16000))
        self._last = prob
        return prob > self.threshold


class Endpointer(Protocol):
    def should_end(self, silence_ms: int, partial_text: str) -> bool: ...


class SilenceEndpointer:
    def __init__(self, silence_ms: int = 500):
        self.silence_ms = silence_ms

    def should_end(self, silence_ms: int, partial_text: str) -> bool:
        return silence_ms >= self.silence_ms


class HeuristicSemanticEndpointer:
    """Baseline 'semantic' endpointer: shorter wait when the partial transcript
    looks complete, longer when it trails off. The learned model replaces this."""

    INCOMPLETE_TAIL = ("and", "but", "so", "the", "a", "my", "is", "to", "of", "um", "uh", "because")

    def __init__(self, fast_ms: int = 250, slow_ms: int = 900, default_ms: int = 500):
        self.fast_ms, self.slow_ms, self.default_ms = fast_ms, slow_ms, default_ms

    def should_end(self, silence_ms: int, partial_text: str) -> bool:
        t = partial_text.strip().lower()
        if not t:
            return silence_ms >= self.default_ms
        if t.endswith(("?", ".", "!")) or t.endswith(("thanks", "thank you", "bye")):
            return silence_ms >= self.fast_ms
        if t.split()[-1] in self.INCOMPLETE_TAIL:
            return silence_ms >= self.slow_ms
        return silence_ms >= self.default_ms
