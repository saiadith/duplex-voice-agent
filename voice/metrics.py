"""Per-turn latency tracing. Defines the latency budget precisely:

  mouth-to-ear = first agent audio sent  -  last voiced user frame received

broken into stages so every millisecond is attributable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STAGES = [
    ("endpoint", "speech_end", "endpoint"),        # waiting to be sure user finished
    ("stt_final", "endpoint", "stt_final"),        # STT finalisation
    ("llm_ttft", "stt_final", "llm_first_token"),  # LLM time-to-first-token
    ("first_chunk", "llm_first_token", "first_chunk"),  # tokens until a speakable clause
    ("tts_ttfb", "first_chunk", "tts_first_byte"), # TTS time-to-first-byte
    ("send", "tts_first_byte", "first_audio_out"),
]


@dataclass
class TurnMetrics:
    turn_id: int
    t: dict[str, float] = field(default_factory=dict)
    interrupted: bool = False
    user_text: str = ""
    spec_hit: bool = False

    def mark(self, name: str, ts: float) -> None:
        self.t.setdefault(name, ts)

    def stages_ms(self) -> dict[str, float]:
        out = {}
        for name, a, b in STAGES:
            if a in self.t and b in self.t:
                out[name] = (self.t[b] - self.t[a]) * 1000
        if "speech_end" in self.t and "first_audio_out" in self.t:
            out["mouth_to_ear"] = (self.t["first_audio_out"] - self.t["speech_end"]) * 1000
        return out


def percentiles(values: list[float], ps=(50, 95, 99)) -> dict[str, float]:
    import numpy as np

    if not values:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(values, p)) for p in ps}
