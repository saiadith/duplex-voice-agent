"""Audio utilities: G.711 mu-law codec, resampling, PCM helpers.

Implemented in numpy because `audioop` was removed in Python 3.13.
Internal format everywhere: 16 kHz, mono, signed 16-bit little-endian PCM,
20 ms frames (640 bytes).
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * 2  # 640

_BIAS = 0x84
_CLIP = 32635


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def mulaw_encode(pcm: bytes) -> bytes:
    """16-bit linear PCM -> 8-bit G.711 mu-law (ITU-T reference algorithm)."""
    s = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    mag = np.minimum(np.abs(s), _CLIP) + _BIAS
    exponent = np.floor(np.log2(mag)).astype(np.int32) - 7
    exponent = np.clip(exponent, 0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def mulaw_decode(ulaw: bytes) -> bytes:
    u = ~np.frombuffer(ulaw, dtype=np.uint8).astype(np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -mag, mag).astype("<i2").tobytes()


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler. Adequate for telephony (8k<->16k);
    swap for soxr/polyphase if you need better anti-aliasing."""
    if src_rate == dst_rate:
        return pcm
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if x.size == 0:
        return b""
    n_out = int(round(x.size * dst_rate / src_rate))
    t_out = np.arange(n_out) * (src_rate / dst_rate)
    y = np.interp(t_out, np.arange(x.size), x)
    return np.clip(y, -32768, 32767).astype("<i2").tobytes()


def frames(pcm: bytes, frame_bytes: int = FRAME_BYTES):
    for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        yield pcm[i : i + frame_bytes]


def tone(ms: int, freq: float = 220.0, amp: float = 0.3) -> bytes:
    """Synthetic 'speech-like' signal for tests and load generation."""
    t = np.arange(SAMPLE_RATE * ms // 1000) / SAMPLE_RATE
    rng = np.random.default_rng(0)
    x = amp * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(t.size)
    return float_to_pcm16(x)


def silence(ms: int) -> bytes:
    return b"\x00\x00" * (SAMPLE_RATE * ms // 1000)
