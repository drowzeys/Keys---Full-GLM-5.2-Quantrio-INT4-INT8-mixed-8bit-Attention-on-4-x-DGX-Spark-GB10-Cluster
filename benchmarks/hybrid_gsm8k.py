#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Part of: github.com/drowzeys (full GLM-5.2 on 4xGB10 recipe)
"""Concurrent GSM8K runner with robust numeric-answer grading (reasoning-aware).
Usage: hybrid_gsm8k.py <base_url> <model> <out.json> [n_samples] [concurrency]"""
import sys, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
CONC = int(sys.argv[5]) if len(sys.argv) > 5 else 4
MAXTOK = 2048  # non-limiting; never truncate reasoning CoT

data = [json.loads(l) for l in open("/home/keyspark/logs/he_runs/gsm8k_test.jsonl")][:N]

def gold(ans):
    m = re.search(r"####\s*([\-\d,\.]+)", ans)
    return m.group(1).replace(",", "").rstrip(".") if m else None

def _clean(s): return s.replace(",", "").rstrip(".") if s else s
def extract_pred(text):
    if not text: return None
    # 1) instructed format "answer is X" -> LAST occurrence (answer is at the end)
    ms = re.findall(r"answer\s+is\s*:?\s*\$?(-?\d[\d,]*\.?\d*)", text, re.I)
    if ms: return _clean(ms[-1])
    # 2) \boxed{X} -> last
    bs = re.findall(r"\\boxed\{([^}]*)\}", text)
    if bs:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", bs[-1])
        if nums: return _clean(nums[-1])
    # 3) any "answer/result/total ... N" -> LAST occurrence
    ms = re.findall(r"(?:answer|result|total)\D{0,15}?(-?\d[\d,]*\.?\d*)", text, re.I)
    if ms: return _clean(ms[-1])
    # 4) fallback: last number in text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _clean(nums[-1]) if nums else None

def num_eq(a, b):
    try: return abs(float(a) - float(b)) < 1e-4
    except: return str(a) == str(b)

PROMPT = ("Solve this math problem. Show brief reasoning, then end with a line "
          "exactly: 'The answer is <number>'.\n\n{q}")

def run(i, q, g):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
            "max_tokens": MAXTOK, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    txt = ""; fin = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            ch = r["choices"][0]; txt = ch["message"].get("content") or ""; fin = ch.get("finish_reason")
            if fin == "length":   # truncated -> retry, don't score as wrong
                txt = ""; time.sleep(1); continue
            break
        except Exception as e:
            txt = f"ERR {e}"; time.sleep(1)
    pred = extract_pred(txt)
    return {"i": i, "gold": g, "pred": pred, "fin": fin, "resp": txt[-400:],
            "ok": bool(pred and num_eq(pred, g)), "empty": not txt or txt.startswith("ERR")}

items = [(i, d["question"], gold(d["answer"])) for i, d in enumerate(data)]
print(f"{MODEL} @ {BASE}: GSM8K n={len(items)} conc={CONC}", flush=True)
res = []; t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs = [ex.submit(run, i, q, g) for i, q, g in items]
    for f in as_completed(futs):
        res.append(f.result()); done += 1
        if done % 40 == 0: print(f"  {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
correct = sum(1 for r in res if r["ok"]); empty = sum(1 for r in res if r["empty"])
acc = correct / len(res)
json.dump({"model": MODEL, "n": len(res), "correct": correct, "acc": round(acc, 4),
           "empty": empty, "secs": round(time.time()-t0), "rows": res}, open(OUT, "w"), indent=1)
print(f"DONE: acc={acc:.4f} ({correct}/{len(res)}) empty={empty} {time.time()-t0:.0f}s -> {OUT}", flush=True)
