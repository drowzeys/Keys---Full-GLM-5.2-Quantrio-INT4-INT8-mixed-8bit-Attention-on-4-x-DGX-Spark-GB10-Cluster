# BREAKTHROUGH: SGLang DSA on GB10/sm_121a via TileLang (2026-06-25)

## The verdict everyone (incl. our own research) reached was BEATABLE
"SGLang GLM-5.2 DSA is blocked on GB10" — because `nsa_indexer.py` calls DeepGEMM (18×),
DeepGEMM is arch-gated to {9,10} (sm_100 tcgen05), no sm_121a fallback (#23657). TRUE for the
DEFAULT CUDA path. But SGLang ALSO ships a non-DeepGEMM DSA kernel that nobody wired to CUDA.

## The discovery
`nsa_indexer.py` has TWO indexer implementations:
1. `_get_topk_paged` / `_get_topk_ragged` — **DeepGEMM** (the CUDA default → blocks GB10)
2. `forward_indexer` (line 947) — **TileLang `fp8_index`** (`tilelang_kernel.py`), the `else` branch
   for non-CUDA. Pure `T.gemm`+`T.max`+`T.reduce_sum`, NO DeepGEMM, NO tcgen05. TMA-lower DISABLED
   (avoids Hopper-specific instr → portable).

The dispatch (forward_cuda line 1266):
```
if _is_cuda or _is_hip:   topk = self._get_topk_paged(...)   # DeepGEMM
else:                     topk = self.forward_indexer(...)   # TileLang
```

## PROVEN: TileLang fp8_index runs on sm_121a
Isolated test in the SGLang image on a GB10 (torch cap (12,1)):
```
fp8_index(q[1,2,64,128]fp8, q_s, k[1,128,128]fp8, k_s) -> out (1,2,128) f32, finite, real values
===== TILELANG fp8_index RUNS ON sm_121a =====
```
The TileLang JIT compiled the DSA-indexer kernel for sm_121a and produced valid logits. **The
DeepGEMM arch-gate is bypassable with SGLang's OWN code.**

## THE PATCH (minimal — patches/route-sm121-tilelang/run.sh)
1. Add `_is_sm121a = _is_cuda and torch.cuda.get_device_capability()==(12,1)`.
2. `if _is_cuda or _is_hip:` → `if (_is_cuda or _is_hip) and not _is_sm121a:` at the dispatch.
   → sm_121a takes the `else` branch = `forward_indexer` (TileLang). One condition flip.
Fast path `_forward_cuda_k_only` is already DeepGEMM-free (dummy_logits+topk_transform) — no patch needed.

## Validation status
- ✅ Kernel proven on sm_121a (isolated).
- ⏳ End-to-end serve needs GLM-5.2 in SGLang on sm_121a → NVFP4-REAP-504B (309GB, modelopt_fp4,
  77GB/node TP=4, SGLang-compatible). Downloading.
- Open validation risks: forward_indexer assumes page_size==64; topk format parity with the CUDA
  path; NSA KV-pool index_k layout. All checkable once it loads.

## Why this matters
First known route to run DeepSeek-Sparse-Attention models (GLM-5.2, DeepSeek-V4) on SGLang on
consumer Blackwell (sm_121a/GB10) — using SGLang's in-tree TileLang kernel, no b12x port, no
DeepGEMM. A genuine frontier result for the 4×GB10 mesh.
