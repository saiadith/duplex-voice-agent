"""Transports.

/ws      browser client: binary 16 kHz PCM16 frames in, binary frames + JSON events out
/twilio  Twilio Media Streams: JSON with base64 8 kHz mu-law; we resample and
         send 'clear' on barge-in so Twilio drops buffered agent audio.
"""
from __future__ import annotations

import base64
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .audio import FRAME_BYTES, mulaw_decode, mulaw_encode, resample
from .providers.mock import MockLLM, MockSTT, MockTTS
from .session import VoiceSession
from .turn import EnergyVAD, HeuristicSemanticEndpointer

app = FastAPI(title="voice-agent")


def build_endpointer(session_history=lambda: []):
    """ENDPOINTER=learned uses the int8 ONNX turn detector (turn_detector/artifacts/eot)."""
    if os.getenv("ENDPOINTER") == "learned":
        from turn_detector.model import EOTModel, LearnedEndpointer
        model = EOTModel(os.getenv("EOT_DIR", "turn_detector/artifacts/eot"))
        last_agent = lambda: next((m["content"] for m in reversed(session_history()) if m["role"] == "assistant"), "")
        return LearnedEndpointer(model, min_ms=150, max_ms=1600, gamma=0.5, prev_turn=last_agent)
    return HeuristicSemanticEndpointer()


def build_providers():
    """Choose providers from env. Defaults to mocks so the repo runs with no keys."""
    if os.getenv("PROVIDERS") == "smallest":  # needs SMALLEST_API_KEY
        from .providers.openai_compat import OpenAICompatLLM
        from .providers.smallest import LightningTTS, PulseSTT
        key = os.environ["SMALLEST_API_KEY"]
        return (PulseSTT(key), OpenAICompatLLM("https://api.smallest.ai/waves/v1", key, "electron"),
                LightningTTS(key, voice_id=os.getenv("VOICE_ID", "meher")))
    stt = MockSTT(["Hi, I want to check my loan balance."])
    tts = MockTTS()
    if os.getenv("LLM_BASE_URL"):
        from .providers.openai_compat import OpenAICompatLLM
        llm = OpenAICompatLLM()
    else:
        llm = MockLLM()
    return stt, llm, tts


@app.websocket("/ws")
async def ws_browser(ws: WebSocket):
    await ws.accept()
    stt, llm, tts = build_providers()

    async def send_audio(frame: bytes):
        await ws.send_bytes(frame)

    async def send_event(ev: dict):
        await ws.send_text(json.dumps(ev))

    holder = {}
    s = VoiceSession(stt, llm, tts, EnergyVAD(), build_endpointer(lambda: holder["s"].history), send_audio, send_event)
    holder["s"] = s
    buf = b""
    try:
        while True:
            buf += await ws.receive_bytes()
            while len(buf) >= FRAME_BYTES:
                frame, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                await s.push_frame(frame)
    except WebSocketDisconnect:
        pass
    finally:
        await s.close()


@app.websocket("/twilio")
async def ws_twilio(ws: WebSocket):
    await ws.accept()
    stt, llm, tts = build_providers()
    stream_sid = None

    async def send_audio(frame16k: bytes):
        payload = base64.b64encode(mulaw_encode(resample(frame16k, 16000, 8000))).decode()
        await ws.send_text(json.dumps({"event": "media", "streamSid": stream_sid,
                                       "media": {"payload": payload}}))

    async def send_event(ev: dict):
        if ev.get("type") == "clear":
            await ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))

    holder = {}
    s = VoiceSession(stt, llm, tts, EnergyVAD(), build_endpointer(lambda: holder["s"].history), send_audio, send_event)
    holder["s"] = s
    buf = b""
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg["event"] == "start":
                stream_sid = msg["start"]["streamSid"]
            elif msg["event"] == "media":
                pcm8 = mulaw_decode(base64.b64decode(msg["media"]["payload"]))
                buf += resample(pcm8, 8000, 16000)
                while len(buf) >= FRAME_BYTES:
                    frame, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                    await s.push_frame(frame)
            elif msg["event"] == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        await s.close()
