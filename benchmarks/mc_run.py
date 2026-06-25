#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Part of: github.com/drowzeys (full GLM-5.2 on 4xGB10 recipe)
"""Multiple-choice runner for GPQA-Diamond + MMLU-Pro (reasoning-aware).
Hardened: 32K budget, finish_reason verify+retry, robust letter extraction.
Usage: mc_run.py <bench: gpqa|mmlu_pro> <base_url> <model> <out.json> [n] [conc]"""
import sys, json, re, time, urllib.request, string
from concurrent.futures import ThreadPoolExecutor, as_completed

BENCH, BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
N    = int(sys.argv[5]) if len(sys.argv) > 5 else 100
CONC = int(sys.argv[6]) if len(sys.argv) > 6 else 8
import os
MAXTOK = int(os.environ.get("MC_MAXTOK", "2048"))
REQ_TIMEOUT = int(os.environ.get("MC_TIMEOUT", "1800"))
LETTERS = string.ascii_uppercase

if BENCH == "gpqa":
    rows = [json.loads(l) for l in open("/home/keyspark/logs/he_runs/gpqa_diamond.jsonl")]
    def fmt(r): return r["question"], r["answer"].strip().upper()      # question embeds A-D, answer=letter
elif BENCH == "mmlu_pro":
    rows = [json.loads(l) for l in open("/home/keyspark/logs/he_runs/mmlu_pro_test.jsonl")]
    def fmt(r):
        opts = r["options"]
        body = r["question"] + "\n\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        return body, r["answer"].strip().upper()
else:
    sys.exit("bench must be gpqa|mmlu_pro")

rows = rows[:N]
PROMPT = ("Answer the following multiple-choice question. Think briefly, then end with "
          "a line exactly: 'The answer is X' where X is the single correct option letter.\n\n{q}")

def extract_letter(text):
    if not text: return None
    m = re.findall(r"answer\s+is\s*:?\s*\(?([A-J])\b", text, re.I)
    if m: return m[-1].upper()
    m = re.findall(r"\\boxed\{\s*\(?([A-J])", text)
    if m: return m[-1].upper()
    m = re.findall(r"\b(?:option|choice)\s*\(?([A-J])\b", text, re.I)
    if m: return m[-1].upper()
    # last standalone capital letter A-J near the end
    m = re.findall(r"\b([A-J])\b", text[-60:])
    return m[-1].upper() if m else None

def run(i, q, gold):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
            "max_tokens": MAXTOK, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    txt = ""; fin = None
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(BASE,
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=REQ_TIMEOUT))
            ch = r["choices"][0]; txt = ch["message"].get("content") or ""; fin = ch.get("finish_reason")
            if fin == "length": txt = ""; time.sleep(1); continue
            break
        except Exception as e:
            txt = f"ERR {e}"; time.sleep(1)
    pred = extract_letter(txt)
    return {"i": i, "gold": gold, "pred": pred, "fin": fin,
            "ok": pred == gold, "empty": not txt or txt.startswith("ERR"), "resp": txt[-300:]}

items = [(i, *fmt(r)) for i, r in enumerate(rows)]
print(f"{BENCH} | {MODEL} @ {BASE} | n={len(items)} conc={CONC}", flush=True)
res = []; t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs = [ex.submit(run, i, q, g) for i, q, g in items]
    for f in as_completed(futs):
        res.append(f.result()); done += 1
        if done % 25 == 0: print(f"  {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
correct = sum(1 for r in res if r["ok"]); empty = sum(1 for r in res if r["empty"])
acc = correct/len(res)
import math
se = math.sqrt(acc*(1-acc)/len(res))
json.dump({"bench": BENCH, "model": MODEL, "n": len(res), "correct": correct, "acc": round(acc,4),
           "se": round(se,4), "empty": empty, "secs": round(time.time()-t0), "rows": res},
          open(OUT, "w"), indent=1)
print(f"DONE {BENCH} {MODEL}: acc={acc:.4f} ±{1.96*se:.3f} ({correct}/{len(res)}) empty={empty} {time.time()-t0:.0f}s", flush=True)
