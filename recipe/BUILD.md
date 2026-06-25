# Reproducible build — full GLM-5.2 on 4×GB10

Driver assumed: **580.159.03 / CUDA 13.0** (the DGX Spark default; do NOT upgrade — no CUDA-13.2
Spark driver exists, and `cuda-compat` is unsupported on the GB10 SoC). aarch64, Ubuntu 24.04.

## 0. Clone upstreams (not vendored here)
```bash
git clone https://github.com/eugr/spark-vllm-docker        # the CUDA-13.0 vLLM image builder
git clone https://github.com/CosmicRaisins/glm-5.2-gb10    # the portable Triton sparse-MLA kernels
```

## 1. Build the base image at the GLM-5.2 ref (CUDA 13.0 → fixes the Marlin PTX wall)
```bash
cd spark-vllm-docker
./build-and-copy.sh \
  --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 \   # post-0.23.0, GLM-5.2 + indexer/MTP
  -t vllm-glm52-cuda130 --tf5                              # transformers>=5 for the GLM-5.2 config
```
The Dockerfile uses `FROM nvidia/cuda:13.0.2-devel-ubuntu24.04` + torch cu130 → `_C` is CUDA-13.0
native → no 13.2-PTX JIT → **Marlin works on the 13.0 driver.**

## 2. Layer the two DSA mods → `vllm-glm52-cuda130:full`
Build via **Dockerfile `FROM vllm-glm52-cuda130:latest`** (NOT `docker commit` — commit mangles
ENTRYPOINT to `sleep`). The Dockerfile copies `glm-5.2-gb10/kernels/` in and runs both mods:

```dockerfile
FROM vllm-glm52-cuda130:latest
COPY patches/sm12x-deepgemm-bypass /opt/mods/sm12x
COPY <glm-5.2-gb10>/kernels       /opt/mods/kernels
RUN bash /opt/mods/sm12x/run.sh                 # DSA indexer: Triton DeepGEMM-bypass
COPY patches/sparse-mla-sm121a    /opt/mods/mla
RUN bash /opt/mods/mla/run.sh                   # FLASHMLA_SPARSE: portable Triton + un-gate sm_121a + b12x==0.23.0
# ENTRYPOINT inherited (/opt/nvidia/nvidia_entrypoint.sh) — do not override
```
(Optional) for asymmetric AWQ models instead of symmetric: also `RUN bash /opt/mods/asym-moe-marlin/run.sh`.

## 3. Distribute the image + model to all 4 nodes
- Image: `docker save vllm-glm52-cuda130:full | ssh <node> 'docker load'` (fast over the 200G fabric;
  use `-c aes128-gcm@openssh.com -o Compression=no`).
- Model: download once to the head (4 TB library node) then fabric-copy with parallel rsync streams —
  the WAN is a single shared uplink, so download-once + fabric-copy beats per-node download.

## 4. Launch TP=4 (per node; mp multi-node, NOT Ray)
Use [`launch-glm52-tp4.sh`](launch-glm52-tp4.sh): workers (ranks 3,2,1) first, then head (rank 0).
Key flags: `--tensor-parallel-size 4 --distributed-executor-backend mp --nnodes 4 --node-rank N
--master-addr <head> --master-port 29501 [--headless for workers] --kv-cache-dtype fp8
--max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.86 --enforce-eager
--num-gpu-blocks-override 768 --reasoning-parser glm45 --tool-call-parser glm47
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.

### Memory notes (the head-rank gotcha)
- Rank 0 runs the API server (~15 GiB extra) → only ~106 GiB usable vs ~125 on workers.
  **util 0.88 fails the preflight on head; 0.86 works.** Stop any other GPU service on the head first.
- Fully drain prior containers + drop caches before relaunch (GPU memory releases with lag → transient
  preflight failures if you relaunch too fast).

Serves an OpenAI-compatible API on `:8000` as `glm-5.2-quanttrio`.
