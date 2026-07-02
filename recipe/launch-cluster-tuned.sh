#!/usr/bin/env bash
# Launch a GLM-5.2 experiment on the whole cluster. Env vars pass through to glm52-exp-launch.sh.
# Usage: EAGER=0 MTPK=3 CTX=65536 SEQS=2 UTIL=0.86 BTOK=4096 ./glm52-exp-all.sh
set -uo pipefail
ENVSTR="EAGER=${EAGER:-1} MTPK=${MTPK:-3} CTX=${CTX:-65536} SEQS=${SEQS:-2} UTIL=${UTIL:-0.88} BTOK=${BTOK:-} KVD=${KVD:-fp8} IMG=${IMG:-vllm-glm52-cuda130:full} MODELDIR=${MODELDIR:-models--QuantTrio--GLM-5.2-Int4-Int8Mix} SERVEDNAME=${SERVEDNAME:-glm-5.2-quanttrio} EXTRA='${EXTRA:-}'"
echo "=== experiment: $ENVSTR ==="
# rank map: head/.4=0, .1=1, .2=2, .3=3
declare -A RANKS=( [10.100.10.4]=0 [10.100.10.1]=1 [10.100.10.2]=2 [10.100.10.3]=3 )
echo "--- gpu-clear all nodes ---"
bash ~/gpu-clear.sh
for ip in 10.100.10.1 10.100.10.2 10.100.10.3; do ssh keyspark@$ip 'bash ~/gpu-clear.sh' & done; wait
echo "--- pre-launch drop_caches (unified memory: raise startup free-mem for vLLM preflight) ---"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
for ip in 10.100.10.1 10.100.10.2 10.100.10.3; do ssh keyspark@$ip 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null' & done; wait
free -g | awk 'NR==2{print "head free now: "$4"G avail: "$7"G"}'
echo "--- sync launcher to workers ---"
for ip in 10.100.10.1 10.100.10.2 10.100.10.3; do scp -q ~/glm52-exp-launch.sh keyspark@$ip:~/ & done; wait
echo "--- launch rank0 (head) ---"
eval "$ENVSTR bash ~/glm52-exp-launch.sh 0"
for ip in 10.100.10.1 10.100.10.2 10.100.10.3; do
  echo "--- launch rank ${RANKS[$ip]} ($ip) ---"
  ssh keyspark@$ip "$ENVSTR bash ~/glm52-exp-launch.sh ${RANKS[$ip]}" &
done
wait
echo "=== all ranks launched; waiting for /v1/models on head:8000 (up to 30 min) ==="
for i in $(seq 1 180); do
  curl -s -m 3 http://10.100.10.4:8000/v1/models >/dev/null 2>&1 && { echo "SERVER READY after ~$((i*10))s"; exit 0; }
  # bail early if head container died
  docker ps --format '{{.Names}}' | grep -q glm_qt || { echo "HEAD CONTAINER DIED — logs:"; docker logs glm_qt 2>&1 | tail -40; exit 1; }
  sleep 10
done
echo "TIMEOUT waiting for server"; docker logs glm_qt 2>&1 | tail -40; exit 1
