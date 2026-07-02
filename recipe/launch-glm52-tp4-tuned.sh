#!/usr/bin/env bash
# GLM-5.2 QuantTrio TP=4 experiment launcher. Arg1 = node rank (0..3).
# Env overrides: EAGER=1|0  MTPK=3  CTX=65536  SEQS=2  UTIL=0.88  BTOK=  KVD=fp8  IMG=vllm-glm52-cuda130:full  EXTRA=
set -uo pipefail
NODE_RANK="${1:?rank}"; MASTER=10.100.10.4; IF=enp1s0f1np1; HCA=rocep1s0f1
EAGER="${EAGER:-1}"; MTPK="${MTPK:-3}"; CTX="${CTX:-65536}"; SEQS="${SEQS:-2}"
UTIL="${UTIL:-0.88}"; BTOK="${BTOK:-}"; KVD="${KVD:-fp8}"; IMG="${IMG:-vllm-glm52-cuda130:full}"; EXTRA="${EXTRA:-}"
MODELDIR="${MODELDIR:-models--QuantTrio--GLM-5.2-Int4-Int8Mix}"; SERVEDNAME="${SERVEDNAME:-glm-5.2-quanttrio}"
SELFIP=$(ip -4 addr show $IF 2>/dev/null|awk '/inet /{print $2}'|cut -d/ -f1); SELFIP=${SELFIP:-$MASTER}
HEADLESS=""; [ "$NODE_RANK" != "0" ] && HEADLESS="--headless"
EAGERFLAG=""; [ "$EAGER" = "1" ] && EAGERFLAG="--enforce-eager"
BTOKFLAG=""; [ -n "$BTOK" ] && BTOKFLAG="--max-num-batched-tokens $BTOK"
SNAP=$(ls -d "$HOME/.cache/huggingface/hub/$MODELDIR/snapshots/"*/ 2>/dev/null | head -1)
MODEL=/root/.cache/huggingface/hub/$MODELDIR/snapshots/$(basename "$SNAP")
docker rm -f glm_qt 2>/dev/null || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g --ulimit memlock=-1 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm-glm-exp:/root/.cache/vllm" --name glm_qt \
  -e VLLM_HOST_IP=$SELFIP -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e HF_HUB_OFFLINE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  --entrypoint /opt/nvidia/nvidia_entrypoint.sh "$IMG" \
  vllm serve $MODEL --served-model-name $SERVEDNAME --host 0.0.0.0 --port 8000 \
    --trust-remote-code --tensor-parallel-size 4 --pipeline-parallel-size 1 --distributed-executor-backend mp \
    --nnodes 4 --node-rank $NODE_RANK --master-addr $MASTER --master-port 29501 $HEADLESS \
    --kv-cache-dtype $KVD --max-model-len $CTX --max-num-seqs $SEQS --gpu-memory-utilization $UTIL \
    $EAGERFLAG $BTOKFLAG $EXTRA \
    --reasoning-parser glm45 --enable-auto-tool-choice --tool-call-parser glm47 \
    $([ "$MTPK" != "0" ] && echo "--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$MTPK}")
sleep 2; docker ps --format '{{.Names}}|{{.Status}}'|grep glm_qt||echo "rank $NODE_RANK notup"
# GB10 unified-memory fix: page cache from streaming the checkpoint looks like
# consumed GPU memory to vLLM's profiler. Keep dropping caches during the load
# window so the KV-sizing step sees true free memory.
nohup bash -c 'for i in $(seq 1 75); do sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1; sleep 20; done' >/dev/null 2>&1 &
