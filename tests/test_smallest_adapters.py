"""Protocol tests for the Smallest.ai adapters against a local mock server that speaks
the documented message formats (no network, no API key)."""
import asyncio
import base64
import json

import pytest
import websockets

from voice.audio import FRAME_BYTES
from voice.providers.smallest import LightningTTS, PulseSTT


async def mock_stt(ws):
    assert ws.request.headers["Authorization"] == "Bearer test"
    assert "encoding=linear16" in ws.request.path and "sample_rate=16000" in ws.request.path
    got = 0
    async for msg in ws:
        if isinstance(msg, bytes):
            got += len(msg)
            if got >= 8192:
                await ws.send(json.dumps({"transcript": "check my", "is_final": False}))
        elif json.loads(msg).get("type") == "close_stream":
            await ws.send(json.dumps({"transcript": "check my balance", "is_final": True}))
            await ws.send(json.dumps({"transcript": "", "is_final": True, "is_last": True}))
            return


async def mock_tts(ws):
    req = json.loads(await ws.recv())
    assert req["sample_rate"] == 16000 and req["output_format"] == "pcm" and req["voice_id"]
    for n in (1000, 333, 2500):  # deliberately not frame-aligned
        await ws.send(json.dumps({"status": "chunk", "data": {"audio": base64.b64encode(b"\x01\x00" * (n // 2)).decode()}}))
    await ws.send(json.dumps({"status": "complete"}))


@pytest.mark.asyncio
async def test_pulse_protocol():
    async with websockets.serve(mock_stt, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        stt = PulseSTT(api_key="test", url=f"ws://127.0.0.1:{port}/stt")
        s = await stt.open()
        for _ in range(20):
            await s.send(b"\x00" * 640)
        await asyncio.sleep(0.05)
        assert s.partial() == "check my"
        assert await s.finalize() == "check my balance"
        await s.aclose()


@pytest.mark.asyncio
async def test_lightning_reframes_to_20ms():
    async with websockets.serve(mock_tts, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        tts = LightningTTS(api_key="test", url=f"ws://127.0.0.1:{port}/tts")
        frames = [f async for f in tts.synthesize("hello")]
        assert all(len(f) == FRAME_BYTES for f in frames)
        assert len(frames) == -(-3832 // FRAME_BYTES)  # ceil(total bytes / frame)
