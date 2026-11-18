# Provider benchmark: Smallest.ai (first run)

Setup: client on a Kaggle CPU VM (Google Cloud, region not disclosed, likely US), 2026-11-18 07:46 UTC, one run window.
5 call-centre utterances × 3 runs. Measured client-side by `bench/providers.py`, so all numbers include network RTT.
Raw data: `bench/results_2026-11-18_kaggle.json`.

| | p50 | p95 | n |
|---|---|---|---|
| Lightning v3.1 (`lauren`): time to first audio | 399 ms | 572 ms | 15 |
| Lightning v3.1 Pro (`ben`): time to first audio | 387 ms | 651 ms | 15 |
| Pulse STT: finalisation after end of audio (`close_stream` → last transcript) | 97 ms | 101 ms | 15 |
| Pulse STT: WER on Lightning-synthesised speech | 0.00 | 0.33* | 15 |
| Electron LLM | not available on this account's plan (HTTP 403) | | |

\* The only "errors" are normalisation artefacts: "today" → "to-day" (counted as 2 words) on the shortest utterance.
This is clean synthetic speech, so it is an upper bound on STT accuracy, not a phone-line result.

## Findings
1. TTS time-to-first-audio is about 400 ms from this client. The docs state about 200 ms server TTFB in-region, so roughly half
   of what we saw is network distance. For a sub-400 ms agent, co-locating the agent with Smallest's region is the
   biggest single win, bigger than any model change. That's the next measurement: rerun from an India-region VM.
2. Pulse finalises in about 100 ms, very consistently (p95 101 ms). With speculative LLM start, STT is not on the critical path.
3. Audio streams at about real-time, not in bursts. Steady-state delivery rate (time after first audio ÷ audio duration):
   v3.1 p50 0.90, p95 1.92; Pro p50 1.15, p95 1.57. Values above 1 mean audio arrived slower than it plays
   (6/15 calls for v3.1, 10/15 for Pro). A client that plays immediately will stutter. Our session's playback pacer must
   hold a jitter buffer (start playback after about 150–250 ms of audio), trading a little latency for smooth audio.
   (This may be server-side pacing or network throughput from this VM; repeat from closer to the servers.)
4. Projected mouth-to-ear with real providers (silence endpointing, speculative LLM, an LLM with a 150 ms TTFT):
   endpoint 200–500 + STT 0 (hidden by speculation) + LLM ≈ 150 + first clause ≈ 40 + TTS first audio ≈ 400 + jitter buffer ≈ 200
   → ≈ 1.0–1.3 s from this region. Getting under 400 ms needs: learned endpointing (≈150 ms), co-location (TTS ≈ 200),
   and a smaller buffer once delivery is faster than real time.

## Caveats
Single client location and time window; no concurrency test (the docs allow 5 concurrent TTS connections per unit);
Electron is untested. Rerun at several times of day and from an India-region VM before quoting.
