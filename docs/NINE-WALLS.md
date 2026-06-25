# The nine walls — getting full GLM-5.2 DSA to serve on sm_121a

Each launch attempt went one layer deeper. Documented so others don't re-derive them.

| # | Wall | Symptom | Fix |
|---|---|---|---|
| 1 | Wrong quant (asymmetric) | `AssertionError: Only symmetric quantization is supported for MoE` | Use a **symmetric** expert quant (QuantTrio Int4-Int8Mix) → Marlin MoE; OR apply `patches/asym-moe-marlin` to wire zero-points for asymmetric AWQ |
| 2 | Config schema | `layer_types ... got ['deepseek_sparse_attention']` | transformers≥5 (`--tf5`) understands it; or map → `compressed_sparse_attention` |
| 3 | Preflight util too high | `Free memory (108) < desired util (0.90→109)` | Lower util; head rank needs the most headroom |
| 4 | **Marlin PTX vs driver** | `cudaErrorUnsupportedPtxVersion` on every Marlin op | **Rebuild vLLM from `nvidia/cuda:13.0.2`** — `_C` ships only sm_121a *PTX built with CUDA 13.2*; the 580/CUDA-13.0 Spark driver can't JIT it. No CUDA-13.2 Spark driver exists; cuda-compat unsupported on GB10 |
| 5 | JSON/entrypoint | `sleep: unrecognized option --served-model-name` | `docker commit` mangled ENTRYPOINT→`sleep`; build mods via **Dockerfile `FROM`** instead. Also: clean-JSON the `--speculative-config` (shell escaping) |
| 6 | **DSA indexer needs DeepGEMM** | `Sparse Attention Indexer CUDA op requires DeepGEMM support` | `patches/sm12x-deepgemm-bypass`: route `fp8_*_mqa_logits` to portable Triton fallbacks + un-gate the constructor on cap family 12.x |
| 7 | **Sparse-MLA attention backend** | `No valid attention backend ... FLASHMLA_SPARSE: compute capability not supported` | `patches/sparse-mla-sm121a`: install portable Triton sparse-MLA (no `_flashmla_C`) + `supports_compute_capability` → `{9,10,12}` + `pip install b12x==0.23.0` |
| 8 | Head preflight (rank 0) | `Free memory (104-106) < util 0.88 (107)` | Rank 0 = API server (~15 GiB overhead) → use **util 0.86**; stop other GPU services on head; full-drain + drop caches before relaunch (release lag) |
| 9 | Benchmark harness | all responses empty / HTTP 400 | `max_tokens` was set to the full `max_model_len` (32768) → `prompt+max_tokens > ctx` → 400. Cap `max_tokens` (e.g. 2048) |

After wall 8 the model loads (96.98 GiB weights/node), KV-cache + DSA indexer initialize, and it
serves. The portable Triton sparse-MLA kernels' first real generation produced **correct** output —
the flagged "silently-wrong-attention" risk did not materialize (validated by coherent code + the
benchmark scores in `../benchmarks/`).

## Things that did NOT work (so you can skip them)
- **Driver upgrade to CUDA 13.2** — NVIDIA has not released a CUDA-13.2 driver supported on DGX Spark
  (DGX OS 7.5 still ships 580/13.0; 595 explicitly "not yet supported on Spark"). The only 595 route
  is a beta sbsa stack whose failure mode is an SSD-erasing reflash. Don't.
- **`cuda-compat`** forward-compat package — unsupported on GB10 (Grace SoC not on the supported list).
- **CosmicRaisins-15pct prune** — fits + hits 256K ctx + 22 tok/s, but it's a 15 % expert prune
  (quality unverified). This recipe keeps the **full** model deliberately.
