#!/bin/bash
# True-token bench for MTP serves (usage-based, not chunk-count). Args: <max_tokens> <concurrency> <prompt_repeat>
URL=http://10.100.10.4:8000/v1/chat/completions
MODEL=${MODEL:-glm-5.2-quanttrio}
MAXTOK=${1:-300}; CONC=${2:-1}; REP=${3:-1}

one(){
python3 - "$URL" "$MODEL" "$MAXTOK" "$REP" <<'PY'
import sys,json,time,urllib.request
url,model,maxtok,rep=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
filler=("Cache coherence and memory bandwidth are central to GPU performance. "*40+"\n")*rep
prompt=filler+"Write a detailed multi-paragraph explanation of the key idea and its implications for large language model inference."
body=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],
  "max_tokens":maxtok,"temperature":0.6,"stream":True,
  "stream_options":{"include_usage":True},
  "chat_template_kwargs":{"enable_thinking":False}}).encode()
req=urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"})
t0=time.time(); ttft=None; comp=0
with urllib.request.urlopen(req,timeout=600) as r:
    for raw in r:
        line=raw.decode().strip()
        if not line.startswith("data:"): continue
        d=line[5:].strip()
        if d=="[DONE]": break
        try: o=json.loads(d)
        except: continue
        ch=o.get("choices") or []
        if ch and (ch[0].get("delta",{}).get("content") or ch[0].get("delta",{}).get("reasoning_content")):
            if ttft is None: ttft=time.time()-t0
        u=o.get("usage")
        if u and u.get("completion_tokens"): comp=u["completion_tokens"]
t=time.time()-t0
if ttft is None: print("NOOUT"); sys.exit(0)
dec=(comp)/(t-ttft) if t>ttft and comp else 0
print(f"{ttft:.3f} {t:.3f} {comp} {dec:.2f}")
PY
}
echo "=== TRUE-TOKEN bench max_tokens=$MAXTOK conc=$CONC rep=$REP ==="
if [ "$CONC" = 1 ]; then
  set -- $(one); echo "TTFT=${1}s total=${2}s true_tokens=${3} decode=${4} tok/s"
else
  tmp=$(mktemp -d); ws=$(date +%s.%N)
  for i in $(seq 1 $CONC); do one >"$tmp/$i" & done; wait
  we=$(date +%s.%N); wall=$(echo "$we - $ws"|bc)
  agg=0; for f in "$tmp"/*; do set -- $(cat "$f"); [ "$1" != "NOOUT" ] && { echo "  TTFT=${1}s decode=${4} tok/s tokens=${3}"; agg=$((agg+${3:-0})); }; done
  echo "AGGREGATE: $agg true tokens in ${wall}s = $(echo "scale=2;$agg/$wall"|bc) tok/s across $CONC streams"
  rm -rf "$tmp"
fi
