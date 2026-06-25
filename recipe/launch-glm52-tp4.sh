#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Part of: github.com/drowzeys (full GLM-5.2 on 4xGB10 recipe)
set -uo pipefail
NODE_RANK="${1:?rank}"; MASTER=<HEAD_IP>; IF=enp1s0f1np1; HCA=rocep1s0f1
SELFIP=$(ip -4 addr show $IF 2>/dev/null|awk '/inet /{print $2}'|cut -d/ -f1); SELFIP=${SELFIP:-$MASTER}
HEADLESS=""; [ "$NODE_RANK" != "0" ] && HEADLESS="--headless"
MODEL=/root/.cache/huggingface/hub/models--QuantTrio--GLM-5.2-Int4-Int8Mix/snapshots/1d3bcfe5ec549ecd000fd80b37f191183842e983
docker rm -f glm_qt 2>/dev/null || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g --ulimit memlock=-1 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband -v "$HOME/.cache/huggingface:/root/.cache/huggingface" --name glm_qt \
  -e VLLM_HOST_IP=$SELFIP -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e HF_HUB_OFFLINE=1 \
  --entrypoint /opt/nvidia/nvidia_entrypoint.sh vllm-glm52-cuda130:full \
  vllm serve $MODEL --served-model-name glm-5.2-quanttrio --host 0.0.0.0 --port 8000 \
    --trust-remote-code --tensor-parallel-size 4 --pipeline-parallel-size 1 --distributed-executor-backend mp \
    --nnodes 4 --node-rank $NODE_RANK --master-addr $MASTER --master-port 29501 $HEADLESS \
    --kv-cache-dtype fp8 --max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.86 \
    --enforce-eager --num-gpu-blocks-override 768 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
sleep 2; docker ps --format '{{.Names}}|{{.Status}}'|grep glm_qt||echo "rank $NODE_RANK notup"
