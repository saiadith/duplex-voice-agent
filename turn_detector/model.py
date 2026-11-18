"""Runtime: learned end-of-turn endpointer (int8 ONNX, CPU).

The model outputs P(turn complete | agent's last turn, user partial). The endpointer
maps it to a silence budget:  wait = min_ms + (max_ms - min_ms) * (1 - p) ** gamma
so confident-complete turns get a fast response and trailing-off turns get patience.
Predictions are cached per partial transcript (the partial only changes every ~100 ms
while silence is counted every 20 ms, so the model runs once per new partial).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np


def stt_norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", s).strip()


class EOTModel:
    def __init__(self, model_dir: str | Path, threads: int = 1):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads  # one call per session; keep cores for other calls
        self.sess = ort.InferenceSession(str(model_dir / "eot_int8.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def prob(self, prev: str, partial: str) -> float:
        prev = " ".join(stt_norm(prev).split()[-20:])
        cur = " ".join(stt_norm(partial).split()[-40:])
        enc = self.tok([prev], [cur], truncation="longest_first", max_length=96, return_tensors="np")
        feed = {k: enc[k].astype(np.int64) for k in self.inputs if k in enc}
        logits = self.sess.run(None, feed)[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())

    def latency_ms(self, n: int = 200) -> dict:
        ts = []
        for i in range(n):
            t = time.perf_counter()
            self.prob("how can i help you today", "i want to check the balance on my account " * (1 + i % 3))
            ts.append((time.perf_counter() - t) * 1000)
        return {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95))}


class LearnedEndpointer:
    def __init__(self, model: EOTModel, min_ms=150, max_ms=1200, gamma=1.5, prev_turn=lambda: ""):
        self.m, self.min_ms, self.max_ms, self.gamma, self.prev_turn = model, min_ms, max_ms, gamma, prev_turn
        self._cache: tuple[str, float] | None = None

    def wait_ms(self, p: float) -> float:
        return self.min_ms + (self.max_ms - self.min_ms) * (1 - p) ** self.gamma

    def should_end(self, silence_ms: int, partial_text: str) -> bool:
        if not partial_text:
            return silence_ms >= 600
        if self._cache is None or self._cache[0] != partial_text:
            self._cache = (partial_text, self.m.prob(self.prev_turn(), partial_text))
        return silence_ms >= self.wait_ms(self._cache[1])
