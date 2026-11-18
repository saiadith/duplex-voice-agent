"""Load test: N concurrent simulated callers in real time against mock providers.

    python -m loadtest.run --calls 100 --turns 3 --barge-in 0.3

Each caller speaks ~1.2 s, pauses, listens; a fraction of callers interrupt the
agent mid-reply. Reports per-stage p50/p95/p99 and event-loop lag (the real
bottleneck of a single-process async server).
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time

from voice.audio import frames, silence, tone
from voice.metrics import STAGES, percentiles
from voice.providers.mock import Latency, MockLLM, MockSTT, MockTTS
from voice.session import VoiceSession
from voice.turn import EnergyVAD, HeuristicSemanticEndpointer, SilenceEndpointer

UTTERANCES = ["I want to check my loan balance.", "When is my next payment due?",
              "Can you move it to the fifteenth?", "Okay thanks, bye."]


async def caller(i: int, turns: int, barge_p: float, fail_rate: float, endpointer: str, speculate: int | None, revise: float):
    rng = random.Random(i)

    async def send_audio(_):  # network send cost not modelled
        pass

    ep = HeuristicSemanticEndpointer() if endpointer == "semantic" else SilenceEndpointer(500)
    from voice.session import SessionConfig
    s = VoiceSession(MockSTT(UTTERANCES, seed=i, latency=Latency(80, 20, fail_rate), revise_rate=revise),
                     MockLLM(seed=i, ttft=Latency(150, 40, fail_rate)),
                     MockTTS(seed=i, ttfb=Latency(90, 25, fail_rate)),
                     EnergyVAD(), ep, send_audio, cfg=SessionConfig(speculate_ms=speculate))
    speech, gap = tone(1200), silence(1000)
    for _ in range(turns):
        for f in frames(speech + gap):
            await s.push_frame(f)
            await asyncio.sleep(0.02)
        # listen; maybe barge in mid-reply
        if rng.random() < barge_p:
            await asyncio.sleep(rng.uniform(0.6, 1.5))
            for f in frames(tone(400) + silence(700)):
                await s.push_frame(f)
                await asyncio.sleep(0.02)
        await s.wait_idle()
        await asyncio.sleep(rng.uniform(0.2, 0.6))
    await s.close()
    return s


async def loop_lag_probe(stop: asyncio.Event, out: list[float]):
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.01)
        out.append((time.perf_counter() - t - 0.01) * 1000)


async def main(a):
    stop, lag = asyncio.Event(), []
    probe = asyncio.create_task(loop_lag_probe(stop, lag))
    t0 = time.perf_counter()
    sessions = await asyncio.gather(*(caller(i, a.turns, a.barge_in, a.fail_rate, a.endpointer, a.speculate, a.revise)
                                      for i in range(a.calls)))
    stop.set(); await probe
    turns = [t for s in sessions for t in s.turns]
    done = [t for t in turns if "first_audio_out" in t.t]
    print(f"calls={a.calls} turns={len(turns)} answered={len(done)} "
          f"interrupted={sum(t.interrupted for t in turns)} "
          f"errors={sum(len(s.errors) for s in sessions)} wall={time.perf_counter()-t0:.1f}s")
    print(f"{'stage':<14}{'p50':>8}{'p95':>8}{'p99':>8}  (ms)")
    for name in [n for n, _, _ in STAGES] + ["mouth_to_ear"]:
        vals = [t.stages_ms()[name] for t in done if name in t.stages_ms()]
        p = percentiles(vals)
        print(f"{name:<14}{p['p50']:>8.0f}{p['p95']:>8.0f}{p['p99']:>8.0f}")
    if a.speculate is not None:
        agg = {k: sum(s.spec_stats[k] for s in sessions) for k in sessions[0].spec_stats}
        print("speculation:", agg)
    p = percentiles(lag)
    print(f"{'loop_lag':<14}{p['p50']:>8.1f}{p['p95']:>8.1f}{p['p99']:>8.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--barge-in", type=float, default=0.3)
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--endpointer", choices=["silence", "semantic"], default="semantic")
    ap.add_argument("--speculate", type=int, default=None, help="ms of silence before speculative LLM start")
    ap.add_argument("--revise", type=float, default=0.1, help="rate at which STT final differs from partial")
    asyncio.run(main(ap.parse_args()))
