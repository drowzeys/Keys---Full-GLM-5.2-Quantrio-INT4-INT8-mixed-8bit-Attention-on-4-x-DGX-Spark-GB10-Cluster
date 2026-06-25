# Attribution

This work integrates and builds on:

- **CosmicRaisins/glm-5.2-gb10** (Apache-2.0) — the portable Triton sparse-MLA + DeepGEMM-fallback
  kernels (`sm12x_*`, `flashmla_sparse.py`, `sparse_mla_kernels.py`, etc.). Our `patches/*/run.sh`
  install/wire these; the `kernels-from-cosmicraisins/` copies retain their Apache-2.0 license.
- **eugr/spark-vllm-docker** — the CUDA-13.0 sm_121a vLLM image builder (`build-and-copy.sh`, base Dockerfile).
- **vLLM** (Apache-2.0) and **jasl**'s sparse-MLA-sm120 lineage (PR #41834) — the sparse-MLA approach.
- **lukealonso/b12x** — sm_120/121 CuTe DSL kernels (installed for the decode fast path).
- **Z.ai / GLM-5.2** (MIT weights), **QuantTrio** (the Int4-Int8Mix quant).

## Our contributions (Apache-2.0)
- The two-mod **Dockerfile recipe** rebuilding vLLM on CUDA 13.0 to fix the Marlin sm_121a PTX wall.
- `patches/asym-moe-marlin` — wiring the symmetric-only compressed-tensors Marlin MoE wrapper to accept
  **asymmetric AWQ** via zero-points (CT-layout `weight_zero_point` → marlin zp conversion).
- `patches/sglang-tilelang-dsa` — routing SGLang's DSA indexer to its in-tree **TileLang** `fp8_index`
  kernel on sm_121a (proven to run; see `docs/SGLANG-TILELANG-DSA.md`).
- The multi-node TP=4 launch recipe, memory tuning, benchmark harness + results, and documentation.
