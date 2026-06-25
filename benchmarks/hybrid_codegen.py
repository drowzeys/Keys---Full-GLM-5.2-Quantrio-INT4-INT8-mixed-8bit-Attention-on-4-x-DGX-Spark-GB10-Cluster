#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Part of: github.com/drowzeys (full GLM-5.2 on 4xGB10 recipe)
"""Concurrent HumanEval+ generator -> evalplus jsonl. Official grading via
`evalplus.evaluate` afterward. Reasoning-aware (high token budget, sanitize
handles CoT). Usage: hybrid_codegen.py <base_url> <model> <out.jsonl> [concurrency]"""
import sys, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 4
MAXTOK = 2048  # non-limiting: models have 160K-1M ctx; never truncate CoT
PROMPT_TMPL = ("Please provide a self-contained Python solution to the following problem "
               "in a single markdown code block. Implement the function exactly as specified:\n"
               "```python\n{p}\n```")

def _one_call(prompt, entry):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT_TMPL.format(p=prompt)}],
            "max_tokens": MAXTOK, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    ch = r["choices"][0]
    raw = ch["message"].get("content") or ""
    fin = ch.get("finish_reason")
    return raw, fin

def gen(task_id, prompt, entry):
    """Retry on truncation(length)/empty/error up to 3x. finish must be 'stop'."""
    t0 = time.time(); sol = ""; fin = None; last_err = ""
    for attempt in range(3):
        try:
            raw, fin = _one_call(prompt, entry)
            if fin == "length":            # truncated -> invalid, retry
                last_err = "length"; continue
            sol = sanitize(raw, entry)
            if sol.strip():
                return task_id, sol, time.time() - t0, fin, "ok"
            last_err = "empty-after-sanitize"
        except Exception as e:
            last_err = str(e)[:80]
        time.sleep(1)
    return task_id, sol, time.time() - t0, fin, last_err

problems = get_human_eval_plus()
items = [(tid, p["prompt"], p["entry_point"]) for tid, p in problems.items()]
print(f"{MODEL} @ {BASE}: {len(items)} problems, conc={CONC}", flush=True)
results = {}; flags = {}
t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs = {ex.submit(gen, tid, pr, en): tid for tid, pr, en in items}
    for f in as_completed(futs):
        tid, sol, dt, fin, status = f.result()
        results[tid] = sol; flags[tid] = (fin, status)
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
empty = sum(1 for s in results.values() if not s.strip())
truncated = sum(1 for fin, st in flags.values() if st == "length")
with open(OUT, "w") as fh:
    for tid, _, _ in items:
        fh.write(json.dumps({"task_id": tid, "solution": results.get(tid, "")}) + "\n")
bad = {t: flags[t] for t in flags if not results[t].strip()}
print(f"DONE: wrote {OUT} | {len(items)} solutions | {empty} empty | {truncated} still-truncated | {time.time()-t0:.0f}s", flush=True)
if bad: print("UNRESOLVED:", json.dumps(bad)[:500], flush=True)
