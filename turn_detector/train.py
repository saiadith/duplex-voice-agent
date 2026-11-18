"""Train a semantic end-of-turn (EOT) classifier (Kaggle GPU, ~20-30 min on a T4).

Task: given the agent's previous turn and the user's *partial* transcript (STT-style:
lowercase, no punctuation), predict P(user has finished their turn).

Data (allenai/soda, 1.5M synthetic social dialogues):
  positive = a complete utterance
  negative = the same utterance truncated at a random word boundary (incomplete)
Known label noise: some prefixes are themselves complete ("yes" in "yes i think so").
We measure this by hand-checking a sample (see README) rather than hiding it.

Outputs: eot.onnx (int8-quantised), tokenizer/, metrics.json
"""
from __future__ import annotations

import json
import os
import random
import re
import time

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.getenv("EOT_BASE", "nreimers/MiniLM-L6-H384-uncased")
N_DIALOGS = int(os.getenv("N_DIALOGS", 40000))
OUT = os.getenv("OUT", "/kaggle/working/eot")
MAXLEN = 96
os.makedirs(OUT, exist_ok=True)
random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def stt_norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


COMPLETE_SHORT = {"yes", "no", "yeah", "sure", "okay", "ok", "thanks", "thank you", "right", "exactly",
                  "of course", "definitely", "maybe", "absolutely", "nope", "yep", "fine", "great"}


def build(split_dialogs):
    X, y = [], []
    for d in split_dialogs:
        turns = [stt_norm(t) for t in d["dialogue"]]
        for i in range(1, len(turns)):
            prev, cur = turns[i - 1], turns[i]
            w = cur.split()
            if not w:
                continue
            X.append((prev, cur)); y.append(1)  # includes one-word answers ("yes", "sure")
            if len(w) < 2:
                continue  # nothing to truncate
            k = random.randint(1, len(w) - 1)  # truncated -> incomplete
            prefix = " ".join(w[:k])
            if prefix in COMPLETE_SHORT:  # "yes" in "yes i think so" is not a reliable negative
                continue
            X.append((prev, prefix)); y.append(0)
    return X, y


def encode(tok, pairs):
    # keep the END of the agent turn and the whole partial (truncate from the left of context)
    prev = [" ".join(p.split()[-20:]) for p, _ in pairs]
    cur = [" ".join(c.split()[-40:]) for _, c in pairs]  # the END of the partial carries the signal
    return tok(prev, cur, truncation="longest_first", max_length=MAXLEN, padding="max_length", return_tensors="pt")


def main():
    ds = load_dataset("allenai/soda", split="train").shuffle(seed=0).select(range(N_DIALOGS))
    n_val = N_DIALOGS // 20
    tr_d, va_d = ds.select(range(n_val, N_DIALOGS)), ds.select(range(n_val))
    Xtr, ytr = build(tr_d); Xva, yva = build(va_d)
    print("train", len(Xtr), "val", len(Xva), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    enc_tr, enc_va = encode(tok, Xtr), encode(tok, Xva)
    tr = list(zip(enc_tr.input_ids, enc_tr.attention_mask, enc_tr.token_type_ids, torch.tensor(ytr)))
    dl = DataLoader(tr, batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 2
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-5, total_steps=epochs * len(dl), pct_start=0.06)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for i, (ids, am, tt, lab) in enumerate(dl):
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(input_ids=ids.to(dev), attention_mask=am.to(dev), token_type_ids=tt.to(dev),
                             labels=lab.to(dev)).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if i % 200 == 0:
                print(f"ep{ep} {i}/{len(dl)} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)

    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(yva), 512):
            out = model(input_ids=enc_va.input_ids[i:i+512].to(dev), attention_mask=enc_va.attention_mask[i:i+512].to(dev),
                        token_type_ids=enc_va.token_type_ids[i:i+512].to(dev)).logits
            probs.append(torch.softmax(out.float(), -1)[:, 1].cpu().numpy())
    p = np.concatenate(probs); yv = np.array(yva)
    from sklearn.metrics import roc_auc_score
    metrics = {"val_auroc": float(roc_auc_score(yv, p)), "val_acc": float(((p > .5) == yv).mean()),
               "n_train": len(ytr), "n_val": len(yva), "base": MODEL, "train_s": time.time() - t0}
    print(metrics, flush=True)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
    with open(f"{OUT}/val_examples.jsonl", "w") as f:
        for (prev, cur), lab, pr in zip(Xva[:3000], yva[:3000], p[:3000]):
            f.write(json.dumps({"prev": prev, "cur": cur, "label": lab, "p": float(pr)}) + "\n")

    # export ONNX (fp32) then dynamic int8 quantisation for CPU
    model.cpu()
    dummy = encode(tok, [("how can i help you", "i want to check my")])
    torch.onnx.export(model, (dummy.input_ids, dummy.attention_mask, dummy.token_type_ids), f"{OUT}/eot_fp32.onnx",
                      input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["logits"],
                      dynamic_axes={**{k: {0: "b", 1: "s"} for k in ["input_ids", "attention_mask", "token_type_ids"]}, "logits": {0: "b"}},
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(f"{OUT}/eot_fp32.onnx", f"{OUT}/eot_int8.onnx", weight_type=QuantType.QInt8)
    tok.save_pretrained(f"{OUT}/tokenizer")
    print("exported", os.listdir(OUT), flush=True)


if __name__ == "__main__":
    main()
