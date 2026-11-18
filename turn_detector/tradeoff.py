"""False-interruption vs response-delay trade-off on held-out turns.

For each held-out user turn we have the model's P(end) on the COMPLETE turn and on a
TRUNCATED prefix (where the user pauses mid-thought). Pause model: mid-turn pauses are
log-normal (median 450 ms, sigma 0.6), a typical range for hesitation pauses in conversation.

  false interruption: policy's wait budget on the prefix < the mid-turn pause
  response delay    : policy's wait budget on the complete turn

Policies: silence-only thresholds (200..1400 ms) vs the learned mapping
wait = min + (max - min) * (1 - p)^gamma swept over (min, max, gamma).
Outputs: tradeoff.json and tradeoff.png (lower-left is better).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np


def load_pairs(path):
    rows = [json.loads(l) for l in open(path)]
    pairs = []
    for a, b in zip(rows[::2], rows[1::2]):  # train.py writes (complete, truncated) per turn
        if a["label"] == 1 and b["label"] == 0:
            pairs.append((a["p"], b["p"]))
    return np.array(pairs)


def evaluate(p_full, p_trunc, wait_fn, pauses):
    w_full, w_trunc = wait_fn(p_full), wait_fn(p_trunc)
    return float((w_trunc < pauses).mean()), float(np.median(w_full)), float(np.percentile(w_full, 90))


def main(val_path, out_dir="."):
    P = load_pairs(val_path)
    rng = np.random.default_rng(0)
    pauses = rng.lognormal(np.log(450), 0.6, len(P))
    res = {"n_turns": int(len(P)), "silence": [], "learned": []}
    for t in range(200, 1500, 100):
        fi, d50, d90 = evaluate(P[:, 0], P[:, 1], lambda p: np.full_like(p, t, dtype=float), pauses)
        res["silence"].append({"threshold_ms": t, "false_interrupt": fi, "delay_p50": d50, "delay_p90": d90})
    for mn, mx, g in itertools.product([100, 150, 250], [900, 1200, 1600, 2000], [0.5, 1, 1.5, 2, 3]):
        fn = lambda p, mn=mn, mx=mx, g=g: mn + (mx - mn) * (1 - p) ** g
        fi, d50, d90 = evaluate(P[:, 0], P[:, 1], fn, pauses)
        res["learned"].append({"min": mn, "max": mx, "gamma": g, "false_interrupt": fi, "delay_p50": d50, "delay_p90": d90})

    # headline: at the false-interruption rate of the common 500 ms silence rule, how much faster is the model?
    base = next(r for r in res["silence"] if r["threshold_ms"] == 500)
    ok = [r for r in res["learned"] if r["false_interrupt"] <= base["false_interrupt"]]
    best = min(ok, key=lambda r: r["delay_p50"]) if ok else None
    res["headline"] = {"silence_500": base, "best_learned_at_same_or_fewer_interruptions": best}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "tradeoff.json").write_text(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    s = res["silence"]
    ax.plot([r["delay_p50"] for r in s], [r["false_interrupt"] for r in s], "o-", label="silence threshold")
    L = sorted(res["learned"], key=lambda r: r["delay_p50"])
    front, best_fi = [], 1.0  # Pareto front of the learned policies
    for r in L:
        if r["false_interrupt"] < best_fi:
            front.append(r); best_fi = r["false_interrupt"]
    ax.plot([r["delay_p50"] for r in front], [r["false_interrupt"] for r in front], "s-", label="learned EOT (Pareto)")
    ax.set(xlabel="Median response delay at true end of turn (ms)", ylabel="False-interruption rate",
           title="Turn-taking trade-off (lower-left is better)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "tradeoff.png", dpi=150)
    print(json.dumps(res["headline"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "turn_detector/results")
