# nvfp4-mla-kv — NVFP4 4-bit KV cache for GLM-5.2's DeepSeek-Sparse-Attention (MLA) path on sm_121a

Adds a new `--kv-cache-dtype nvfp4_ds_mla` for the `flashmla_sparse` (DSA/MLA) backend — to our
knowledge the first NVFP4 KV cache on an MLA/sparse-attention path on consumer Blackwell (GB10).

## Layout (`nvfp4_ds_mla`, 416 B/token vs fp8_ds_mla's 656)
- bytes [0:256]   — 512 × E2M1 (4-bit float, 2/byte): the quantized NoPE MLA latent (kv_lora_rank=512)
- bytes [256:288] — 32 × fp8(e4m3) block scales, one per 16 NoPE elems (scale = amax/6, the NVFP4 grid max)
- bytes [288:416] — 64 × bf16 RoPE (kept full-precision; small + sensitive)
Dequant: `nope[i] = E2M1_LUT[nibble(i)] * fp8_block_scale[i//16]` (global scale fixed 1.0).

## How it works
- **Store** (`_nvfp4ds_store`): flashinfer `nvfp4_kv_quantize` packs the latent → E2M1 + per-16 fp8 BSF;
  a Triton scatter writes the 416-B record. Dispatched from `SparseMLAAttentionImpl.do_kv_cache_update`.
- **Read** (`_gather_dequant_nvfp4ds` / fused kernel): the gather-by-global-index sparse-MLA kernels
  unpack E2M1 + apply the per-16 fp8 block scale, keeping RoPE bf16. No split-KV (gather design), so the
  4-bit split-KV corruption hazard (FlashInfer #3684) cannot occur on this path.
- Calibration: per-block `amax/6` (NOT `/448` like fp8 — the #1 NVFP4-KV pitfall).

## Numerical validation (the gate)
`test_nvfp4_roundtrip.py` on GB10: NVFP4 store→dequant vs fp32 reference = **0.095 rel-Frobenius**
(in the established ~0.09-0.10 band), Triton dequant **byte-identical** to flashinfer `nvfp4_kv_dequantize`,
fused attention vs dense fp32 ref = 0.001 (bf16 noise). RoPE lossless.

## Apply
`bash run.sh` inside a `flashmla_sparse`-capable vLLM image (copies the kernels + patches the gate sites:
dtype maps, MLA page-size → block_size*416, supported dtypes, shape, store dispatch). Idempotent, AST-checked.
Then serve with `--kv-cache-dtype nvfp4_ds_mla`.

Built on the portable sparse-MLA Triton kernels (CosmicRaisins/glm-5.2-gb10) + the NVFP4 quant/dequant
pattern from tonyd2wild's DiffKV mod; calibration + split-KV-gate insights from @Hikari_07_jp's vLLM/SGLang NVFP4-KV work.
