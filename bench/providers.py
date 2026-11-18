"""Honest provider benchmark: time-to-first-audio, real-time factor, STT finalisation latency,
WER and failure rate, over a fixed utterance set, measured from the client.

    SMALLEST_API_KEY=... python -m bench.providers --runs 3            # Smallest Pulse + Lightning + Electron
    (optional) GROQ_API_KEY=...  adds an LLM comparison via Groq's OpenAI-compatible endpoint

The STT test audio is produced by the TTS under test and fed back in real time, so WER here measures
STT on clean synthetic speech (an upper bound). Note this caveat in any write-up. Results -> bench/results.json.
Run at several times of day; report p50/p95 with the client's region and network.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time

import numpy as np

from voice.audio import FRAME_MS

UTTERANCES = [
    "Hi, I'd like to check the balance on my savings account.",
    "Your next payment of four thousand two hundred rupees is due on the fifteenth.",
    "Sure, I can move that to Friday. Is there anything else I can help with?",
    "Could you please confirm the last four digits of your registered mobile number?",
    "Thanks, that's all for today.",
]


def wer(ref: str, hyp: str) -> float:
    norm = lambda s: "".join(c.lower() if c.isalnum() or c == " " else " " for c in s).split()
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int); d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


async def bench_tts(tts, text):
    t0 = time.perf_counter(); first = None; frames = []
    async for f in tts.synthesize(text):
        first = first or time.perf_counter()
        frames.append(f)
    total = time.perf_counter() - t0
    audio_s = len(frames) * FRAME_MS / 1000
    return {"ttfa_ms": (first - t0) * 1000, "total_ms": total * 1000, "rtf": total / max(audio_s, 1e-6)}, b"".join(frames)


async def bench_stt(stt, pcm, ref):
    s = await stt.open()
    for i in range(0, len(pcm), 640):  # real time: 20 ms per frame
        await s.send(pcm[i:i + 640]); await asyncio.sleep(0.02)
    t = time.perf_counter(); hyp = await s.finalize(); fin = (time.perf_counter() - t) * 1000
    await s.aclose()
    return {"finalize_ms": fin, "wer": wer(ref, hyp), "hyp": hyp}


async def bench_llm(llm):
    t0 = time.perf_counter(); first = None; n = 0
    async for _ in llm.stream([{"role": "system", "content": "You are a concise phone agent."},
                               {"role": "user", "content": "When is my next loan payment due?"}]):
        first = first or time.perf_counter(); n += 1
    return {"ttft_ms": (first - t0) * 1000, "total_ms": (time.perf_counter() - t0) * 1000, "chunks": n}


def summarise(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if k in r]
        if v:
            out[k] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "n": len(v)}
    return out


async def main(a):
    from voice.providers.openai_compat import OpenAICompatLLM
    from voice.providers.smallest import LightningTTS, PulseSTT
    import httpx
    key = os.environ["SMALLEST_API_KEY"]
    stt = PulseSTT()
    ttss = {"lightning_v3.1/lauren": LightningTTS(voice_id="lauren", model="lightning_v3.1")}
    try:  # Pro voices come from their own catalog
        r = httpx.get("https://api.smallest.ai/waves/v1/lightning-v3.1-pro/get_voices",
                      headers={"Authorization": f"Bearer {key}"}, timeout=15)
        voices = r.json().get("voices", [])
        en = [v for v in voices if "english" in [l.lower() for l in v.get("tags", {}).get("language", [])]]
        if en:
            vid = en[0]["voiceId"]
            ttss[f"lightning_v3.1_pro/{vid}"] = LightningTTS(voice_id=vid, model="lightning_v3.1_pro")
        print("pro voices:", len(voices), "english:", len(en))
    except Exception as e:
        print("could not list pro voices:", repr(e))
    llms = {"smallest-electron": OpenAICompatLLM("https://api.smallest.ai/waves/v1", os.environ["SMALLEST_API_KEY"], "electron")}
    if os.getenv("GROQ_API_KEY"):
        llms["groq-llama-3.1-8b"] = OpenAICompatLLM("https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"], "llama-3.1-8b-instant")
    R = {"env": {"host": platform.node(), "when": time.strftime("%Y-%m-%d %H:%M %Z")}, "tts": {}, "stt": [], "llm": {}, "errors": []}
    for run in range(a.runs):
        for text in UTTERANCES:
            for name, tts in ttss.items():
                try:
                    m, pcm = await bench_tts(tts, text); R["tts"].setdefault(name, []).append(m)
                    if name.startswith("lightning_v3.1/"):  # STT fed with the standard voice
                        R["stt"].append(await bench_stt(stt, pcm, text))
                except Exception as e:
                    R["errors"].append(f"{name}: {e!r}")
        for name, llm in llms.items():
            try:
                R["llm"].setdefault(name, []).append(await bench_llm(llm))
            except Exception as e:
                R["errors"].append(f"{name}: {e!r}")
    S = {"tts": {k: summarise(v, ["ttfa_ms", "total_ms", "rtf"]) for k, v in R["tts"].items()},
         "stt_pulse": summarise(R["stt"], ["finalize_ms", "wer"]),
         "llm": {k: summarise(v, ["ttft_ms", "total_ms"]) for k, v in R["llm"].items()},
         "n_errors": len(R["errors"]), "error_kinds": sorted({e.split(":")[0] + ": " + e.split("(")[0][-40:] for e in R["errors"]}),
         "env": R["env"]}
    os.makedirs("bench", exist_ok=True)
    json.dump({"summary": S, "raw": R}, open("bench/results.json", "w"), indent=2)
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=3)
    asyncio.run(main(ap.parse_args()))
