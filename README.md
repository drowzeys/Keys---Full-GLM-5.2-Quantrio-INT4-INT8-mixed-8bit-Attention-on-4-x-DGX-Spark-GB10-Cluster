# GLM-5.2 (full, non-pruned) on a 4×DGX-Spark / GB10 cluster (Updated: tuned recipe — 15.3 tok/s single / 28.9 tok/s @ C4)

Serving **full GLM-5.2** (753B MoE with DeepSeek Sparse Attention) TP=4 across four
NVIDIA DGX Spark / GB10 (sm_121a, aarch64) nodes — on a **vLLM rebuilt from source**
with two custom sparse-attention mods and a working Marlin MoE path.

To our knowledge this is the first public recipe that runs the **full** (un-pruned)
GLM-5.2 on a *rebuilt* sm_121a vLLM with both DSA sparse mods **and** Marlin working —
prior GB10 recipes rely on a 15 % expert prune to fit and use a prebuilt image.

## 🆕 Tuned serving results (2026-07) — full writeup in [`docs/TUNING-2026-07.md`](docs/TUNING-2026-07.md)

| Test | Tuned result | Previously documented |
|---|---|---|
| Single-stream decode | **15.3 tok/s** | "~13" (undercounted — see below) |
| C2 aggregate | **19.8 tok/s** | — |
| **C4 aggregate** | **28.9 tok/s** | — |
| Decode @ 18K-deep context | **13.7 tok/s** | — |
| TTFT @ 18K prompt | 67 s (~270 tok/s prefill) | — |
| KV pool @ util 0.86 | 73,728 tokens (64K ctx, 4 streams) | — |
| Worst-case RAM headroom | 2–5 GiB/node | was 0–1 GiB |

**What was tuned:** `max-num-seqs 2→4` (concurrency), **`--num-gpu-blocks-override 1152`**
(vLLM's KV estimator computes **negative** blocks on GB10 unified memory — the override is
mandatory, not optional), `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` + `max-num-batched-tokens 2048`
(long-prompt RAM safety at zero TTFT cost), and a **corrected benchmark** — the old bench counted
SSE chunks, but MTP delivers ~2–3 tokens per chunk, so every earlier speed was an undercount
([`benchmarks/bench-true-tokens.sh`](benchmarks/bench-true-tokens.sh) counts `usage.completion_tokens`).
Closed for good: **CUDA graphs / torch.compile are ~2× slower than eager** for GLM-5.2 DSA on
sm_121a (3 variants tested), and a **root-zombie cleanup bug** (crashed runs held 106 GiB/node
through non-root `kill`; cleanup must `sudo kill -9` + verify `nvidia-smi --query-compute-apps`
per node). Tuned launchers: [`recipe/launch-cluster-tuned.sh`](recipe/launch-cluster-tuned.sh) +
[`recipe/launch-glm52-tp4-tuned.sh`](recipe/launch-glm52-tp4-tuned.sh).

## NVFP4 4-bit KV cache → 100K context

Added **`--kv-cache-dtype nvfp4_ds_mla`** — to our knowledge the **first NVFP4 4-bit KV cache
on a DeepSeek-Sparse-Attention / MLA path on consumer Blackwell (GB10)**. Numerically validated
(0.095 rel-Frobenius vs fp32, byte-exact vs flashinfer `nvfp4_kv_dequantize`) → now **serving
coherently** TP=4.

| Metric | fp8 baseline | **NVFP4 KV (new)** |
|---|---|---|
| Per-token KV | ~53 KiB | **~33 KiB (416 B/layer)** |
| Max context (full model) | ~64K | **~112K (serving 100K)** |
| KV pool @ util 0.87 | — | **118,720 tokens** |
| Coherence | ✓ | **✓ (code + DSA reasoning correct)** |
| Speed | ~13 tok/s | ~11.5 tok/s (single-stream + MTP) |

Layout: 512×E2M1 NoPE latent + per-16 fp8 block scales (`amax/6`) + bf16 RoPE. The mod, kernels,
numerical test, and idempotent patcher are in [`patches/nvfp4-mla-kv/`](patches/nvfp4-mla-kv/).
**128K+ on the full model is weight-blocked** (needs util 0.88 > head's preflight ceiling) — the path
to true 128K-256K is REAP-504B (frees ~24 GB/node of weights) **+** this NVFP4-KV mod.

## Quality results (thinking OFF, single-stream + MTP — 15.3 tok/s true-token rate)

| Category | Benchmark | Score |
|---|---|---|
| **Coding** | HumanEval pass@1 | **96.3 %** (base) / 92.7 % (plus) |
| **Math** | GSM8K | **97.5 %** |
| **Reasoning** | MMLU-Pro | **82.5 %** |
| **Hard science** | GPQA-Diamond | **75.0 %** |

Frontier-grade quality, confirming the 4-bit-expert / 8-bit-attention quant **and** the
first-run portable Triton sparse-MLA kernels produce correct output (not garbage).

## Hardware / model
- 4× DGX Spark / GB10, sm_121a, 128 GB unified memory/node, ~273 GB/s, CX7 200G RoCE fabric.
- Model: [`QuantTrio/GLM-5.2-Int4-Int8Mix`](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
  (full GlmMoeDsa, 256 experts/8 active; **experts 4-bit symmetric → Marlin MoE**, attention/dense 8-bit).
  ~96.98 GiB weights/node.

## Why this was hard — the nine walls
Stock vLLM cannot serve GLM-5.2 DSA on sm_121a. Each wall and its fix is documented in
[`docs/NINE-WALLS.md`](docs/NINE-WALLS.md). The three structural ones:

1. **Marlin PTX vs driver** — vLLM `_C` ships sm_121a **PTX built with CUDA 13.2**, but the only
   driver available for DGX Spark is **580.159.03 = CUDA 13.0**, which cannot JIT 13.2 PTX
   (`cudaErrorUnsupportedPtxVersion`). No CUDA-13.2 driver exists for Spark; `cuda-compat` is
   unsupported on GB10. **Fix: rebuild vLLM from a `nvidia/cuda:13.0.2` base.**
2. **DSA indexer requires DeepGEMM** (arch-gated to sm_90/sm_100). **Fix:** `patches/sm12x-deepgemm-bypass`
   routes the indexer to portable Triton fallbacks on sm_121a.
3. **Sparse-MLA attention backend** `FLASHMLA_SPARSE: compute capability not supported`. **Fix:**
   `patches/sparse-mla-sm121a` installs the portable Triton sparse-MLA kernels (no `_flashmla_C`)
   and un-gates the backend for capability family 12.x.

## Quick start
See [`recipe/BUILD.md`](recipe/BUILD.md) for the full reproducible build (clone upstreams →
build CUDA-13.0 base → apply mods → distribute → launch). Launch script:
[`recipe/launch-glm52-tp4.sh`](recipe/launch-glm52-tp4.sh).

## Contents
- `recipe/` — reproducible build + the per-node TP=4 launch script.
- `patches/` — our vLLM mods: DSA-indexer DeepGEMM-bypass, sparse-MLA-sm121a, **asymmetric-AWQ Marlin MoE
  patch** (`asym-moe-marlin`, lets the symmetric-only CT Marlin MoE wrapper accept asymmetric AWQ via
  zero-points), and a **SGLang TileLang DSA** route patch (`sglang-tilelang-dsa`).
- `docs/` — the nine-walls writeup, the SGLang TileLang DSA breakthrough, and the optimization analysis.
- `benchmarks/` — runners + raw results.

## Optimization (see [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) + [`docs/TUNING-2026-07.md`](docs/TUNING-2026-07.md))
2026-07 campaign settled the open items: **applied** — seqs 4, blocks-override 1152, BTOK 2048 +
indexer logits cap, MTP k=3 (per-position acceptance shows k=4 not worth it). **Dead ends on this
platform:** CUDA graphs/compile (~2× slower than eager), W4A4-NVFP4-via-Marlin (slower + garbage;
native FP4 GEMM hardware-rejects sm_121a), **DFlash** (no GLM-5.2 drafter exists), and
**Expert Parallel** (no memory win at TP=4 + throughput loss on the 200G fabric).

## Upcoming research & optimization

Roadmap in [`docs/UPCOMING-RESEARCH.md`](docs/UPCOMING-RESEARCH.md) — three tracks:
- **Faster decode:** **JetSpec** parallel-tree speculative decoding (train a GLM-5.2 draft head; up to ~9× on Qwen3-8B). *Plan auto-generated by GLM-5.2 self-researching on this cluster.*
- **Bigger context:** `nvidia/GLM-5.2-NVFP4` via b12x + our `nvfp4_ds_mla` KV → 128K path.
- **The model:** REAP-504B (frees ~24 GB/node) + NVFP4-KV → true 128K-256K.

## Attribution & license
Built on the work of others — see [`ATTRIBUTION.md`](ATTRIBUTION.md). The portable Triton sparse-MLA /
DeepGEMM-fallback kernels are from **CosmicRaisins/glm-5.2-gb10** (Apache-2.0); the base image build is
**eugr/spark-vllm-docker**; sparse kernels derive from **vLLM / jasl** (Apache-2.0) and **lukealonso/b12x**.
Our own contributions (the mods' wiring, the asym-MoE Marlin patch, the SGLang TileLang route, the recipe,
benchmarks, docs) are released under Apache-2.0. Serves MIT-licensed GLM-5.2 weights (Z.ai).
