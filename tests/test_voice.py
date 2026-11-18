import asyncio

import numpy as np
import pytest

from voice.audio import frames, mulaw_decode, mulaw_encode, resample, silence, tone
from voice.providers.mock import Latency, MockLLM, MockSTT, MockTTS
from voice.session import SessionConfig, VoiceSession, clause_chunks
from voice.turn import EnergyVAD, HeuristicSemanticEndpointer, SilenceEndpointer


def test_mulaw_roundtrip_error_small():
    x = tone(200, amp=0.5)
    y = mulaw_decode(mulaw_encode(x))
    a = np.frombuffer(x, "<i2").astype(float)
    b = np.frombuffer(y, "<i2").astype(float)
    snr = 10 * np.log10(np.sum(a**2) / np.sum((a - b) ** 2))
    assert snr > 30  # G.711 gives ~38 dB on loud signals


def test_mulaw_known_values():
    assert mulaw_encode(np.array([0], "<i2").tobytes()) == b"\xff"
    assert mulaw_decode(b"\xff") == np.array([0], "<i2").tobytes()


def test_resample_lengths():
    x = tone(100)  # 1600 samples
    assert len(resample(x, 16000, 8000)) == 1600
    assert len(resample(resample(x, 16000, 8000), 8000, 16000)) == 3200


def test_vad_and_endpointers():
    vad = EnergyVAD()
    assert vad.is_speech(next(frames(tone(20))))
    assert not vad.is_speech(next(frames(silence(20))))
    assert SilenceEndpointer(500).should_end(500, "")
    ep = HeuristicSemanticEndpointer()
    assert ep.should_end(260, "thanks, bye")
    assert not ep.should_end(600, "my account number is")


@pytest.mark.asyncio
async def test_clause_chunker():
    async def toks():
        for t in ["Sure", ",", " your", " balance", " is", " due", " soon", ".", " Bye", "."]:
            yield t
    out = [c async for c in clause_chunks(toks(), min_chars=10)]
    assert out == ["Sure, your balance is due soon.", "Bye."]


def make_session(sent, events, **kw):
    async def sa(f): sent.append(f)
    async def se(e): events.append(e)
    return VoiceSession(MockSTT(["hello there"], latency=Latency(5)),
                        MockLLM(ttft=Latency(5), per_token_ms=1),
                        MockTTS(ttfb=Latency(5)), EnergyVAD(), SilenceEndpointer(300),
                        sa, se, SessionConfig(max_lead_ms=50, **kw))


async def feed(s, pcm, realtime=False):
    for f in frames(pcm):
        await s.push_frame(f)
        await asyncio.sleep(0.02 if realtime else 0)


@pytest.mark.asyncio
async def test_full_turn_produces_audio_and_metrics():
    sent, events = [], []
    s = make_session(sent, events)
    await feed(s, tone(500) + silence(400))
    await s.wait_idle()
    assert sent, "agent produced no audio"
    t = s.turns[0]
    st = t.stages_ms()
    assert {"stt_final", "llm_ttft", "tts_ttfb", "mouth_to_ear"} <= st.keys()
    assert s.history[-1]["role"] == "assistant" and not t.interrupted
    assert any(e["type"] == "turn_metrics" for e in events)


@pytest.mark.asyncio
async def test_no_turn_on_short_noise():
    sent, events = [], []
    s = make_session(sent, events)
    await feed(s, tone(40) + silence(600))  # 2 frames < min_speech_frames
    assert not s.turns


@pytest.mark.asyncio
async def test_barge_in_cancels_and_truncates_history():
    sent, events = [], []
    s = make_session(sent, events)
    await feed(s, tone(500) + silence(400))
    await asyncio.sleep(0.6)  # let agent talk for a while (reply is ~5 s of audio)
    assert s.agent_speaking
    await feed(s, tone(200), realtime=True)  # user interrupts
    assert any(e["type"] == "clear" for e in events)
    interrupted = [m for m in s.history if m.get("interrupted")]
    assert interrupted, "assistant turn not marked interrupted"
    full = MockLLM().reply
    heard = interrupted[0]["content"]
    assert 0 < len(heard) < len(full), heard
    assert s.turns[0].interrupted
    await s.close()


@pytest.mark.asyncio
async def test_barge_in_disabled_ignores_speech():
    sent, events = [], []
    s = make_session(sent, events, barge_in_enabled=False)
    await feed(s, tone(500) + silence(400))
    await asyncio.sleep(0.2)
    await feed(s, tone(200), realtime=True)
    assert not any(e["type"] == "clear" for e in events)
    await s.close()


@pytest.mark.asyncio
async def test_provider_failure_degrades_gracefully():
    sent, events = [], []
    async def sa(f): sent.append(f)
    s = VoiceSession(MockSTT(["hi"], latency=Latency(1, fail_rate=1.0)), MockLLM(), MockTTS(),
                     EnergyVAD(), SilenceEndpointer(300), sa)
    await feed(s, tone(300) + silence(400))
    await s.wait_idle()
    assert s.errors and not sent


@pytest.mark.asyncio
async def test_speculation_hit_cuts_ttft():
    sent = []
    async def sa(f): sent.append(f)
    llm = MockLLM(ttft=Latency(150), per_token_ms=1)
    s = VoiceSession(MockSTT(["check my balance"], latency=Latency(60)), llm, MockTTS(ttfb=Latency(5)),
                     EnergyVAD(), SilenceEndpointer(500), sa, cfg=SessionConfig(speculate_ms=200))
    await feed(s, tone(800) + silence(700), realtime=True)
    await s.wait_idle()
    t = s.turns[0]
    assert t.spec_hit and s.spec_stats["hit"] == 1
    # LLM ran during the silence window, so no LLM wait after STT final
    assert t.stages_ms()["llm_ttft"] < 20


@pytest.mark.asyncio
async def test_speculation_discarded_when_user_resumes():
    sent = []
    async def sa(f): sent.append(f)
    s = VoiceSession(MockSTT(["check my balance"], latency=Latency(5)), MockLLM(ttft=Latency(5)),
                     MockTTS(ttfb=Latency(5)), EnergyVAD(), SilenceEndpointer(600), sa,
                     cfg=SessionConfig(speculate_ms=200))
    await feed(s, tone(800) + silence(300) + tone(400) + silence(700), realtime=True)
    await s.wait_idle()
    assert s.spec_stats["miss_resumed"] == 1 and len(s.turns) == 1
    assert s.history[1]["content"] == "check my balance"


def test_server_endpoints_with_learned_endpointer(monkeypatch):
    import os
    from pathlib import Path
    if not (Path("turn_detector/artifacts/eot/eot_int8.onnx")).exists():
        pytest.skip("model artifacts not downloaded")
    monkeypatch.setenv("ENDPOINTER", "learned")
    from fastapi.testclient import TestClient
    from voice.server import app
    with TestClient(app).websocket_connect("/ws") as ws:
        for f in frames(tone(600) + silence(1700)):
            ws.send_bytes(f)
        got_audio = False
        for _ in range(400):
            m = ws.receive()
            if m.get("bytes"):
                got_audio = True
                break
        assert got_audio


def test_learned_endpointer_policy():
    from pathlib import Path
    if not Path("turn_detector/artifacts/eot/eot_int8.onnx").exists():
        pytest.skip("model artifacts not downloaded")
    from turn_detector.model import EOTModel, LearnedEndpointer
    ep = LearnedEndpointer(EOTModel("turn_detector/artifacts/eot"), min_ms=150, max_ms=1600, gamma=0.5,
                           prev_turn=lambda: "how can i help you today")
    # same 1 s of silence: respond to the complete turn, keep waiting on the trailing one
    assert ep.should_end(1000, "i want to check the balance on my account")
    assert not ep.should_end(1000, "i want to check the balance on my")
    assert ep.should_end(1000, "yes")
