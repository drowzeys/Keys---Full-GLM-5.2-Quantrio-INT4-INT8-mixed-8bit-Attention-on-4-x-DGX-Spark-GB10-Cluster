#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# nvfp4_ds_mla — apply the GLM-5.2 sparse-MLA NVFP4 KV-cache mod into the
# in-image vLLM tree. Idempotent + AST-checked. Run INSIDE the container, or
# via: docker run --rm -v <repo>:/work --entrypoint bash IMAGE /work/run.sh
#
# Wires the gate sites:
#   1. flashmla_sparse.py        : supported_kv_cache_dtypes += nvfp4_ds_mla
#   2. flashmla_sparse.py        : get_kv_cache_shape -> 416 for nvfp4_ds_mla
#   3. flashmla_sparse.py        : relax the fp8-only assert
#   4. flashmla_sparse.py        : use_fp8_kv_cache / decode routing accept nvfp4
#   5. sm12x_sparse_mla_attn.py  : kernels + width-based decode routing (file copy)
#   6. config/cache.py           : CacheDType literal += nvfp4_ds_mla
#   7. kv_cache_interface.py     : get_kv_quant_mode -> NVFP4 for nvfp4_ds_mla
#   7b.kv_cache_interface.py     : MLA/SWA real_page_size_bytes -> 416 (THE fix
#                                  for the PAGE-SIZE mismatch crash; mirrors the
#                                  fp8_ds_mla 656 override)
#   7c.utils/torch_utils.py      : STR_DTYPE_TO_TORCH_DTYPE += nvfp4_ds_mla:uint8
#                                  (indexed in platforms/interface.py before alloc)
#   8. mla_attention.py          : nvfp4 -> nvfp4_ds_mla routing for FLASHMLA_SPARSE
#   9. backend.py                : SparseMLAAttentionImpl.do_kv_cache_update store dispatch
# (is_quantized_kv_cache already True via the nvfp4 prefix -> get_kv_quant_mode)
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
echo "[run.sh] repo=$REPO  vllm=$V"

ast_check () { python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$1"; }

# --- 5. Copy the kernel-bearing extracted files over the in-image versions ----
for f in sm12x_sparse_mla_attn.py flashmla_sparse.py; do
  dst="$V/v1/attention/backends/mla/$f"
  if [ -f "$REPO/src/$f" ] && [ -f "$dst" ]; then
    cp "$REPO/src/$f" "$dst"
    ast_check "$dst"
    echo "[run.sh] copied src/$f -> $dst"
  fi
done

# --- Python patcher for the in-image gate sites (idempotent, AST-checked) -----
python3 - "$V" <<'PYEOF'
import io, os, sys
V = sys.argv[1]

def patch(path, edits, marker):
    s = io.open(path, encoding="utf-8").read()
    if marker in s:
        print(f"[run.sh] {os.path.basename(path)}: already patched")
        return
    for old, new in edits:
        assert old in s, f"anchor not found in {path}:\n{old[:120]}"
        s = s.replace(old, new, 1)
    import ast; ast.parse(s)
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"[run.sh] patched {os.path.basename(path)}")

# 1. flashmla_sparse: supported dtypes + shape + assert
fm = f"{V}/v1/attention/backends/mla/flashmla_sparse.py"
patch(fm, [
    ('''        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]''',
     '''        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
        "nvfp4_ds_mla",  # GLM-5.2 sparse-MLA NVFP4 KV cache (416-B record)
    ]'''),
    ('''        if cache_dtype_str == "fp8_ds_mla":
            # V3.2 main MLA: 656-byte custom storage format. See module docstring.
            return (num_blocks, block_size, 656)''',
     '''        if cache_dtype_str == "nvfp4_ds_mla":
            # GLM-5.2 sparse-MLA NVFP4: 416-byte record (E2M1|fp8 BSF|bf16 RoPE).
            return (num_blocks, block_size, 416)
        if cache_dtype_str == "fp8_ds_mla":
            # V3.2 main MLA: 656-byte custom storage format. See module docstring.
            return (num_blocks, block_size, 656)'''),
    ('''            assert kv_cache_dtype == "fp8_ds_mla", (
                "FlashMLA Sparse Attention backend fp8 only supports "
                "fp8_ds_mla kv-cache dtype"
            )''',
     '''            assert kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla"), (
                "FlashMLA Sparse Attention quantized KV only supports "
                "fp8_ds_mla / nvfp4_ds_mla kv-cache dtype"
            )'''),
    ('''        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"''',
     '''        self.use_fp8_kv_cache = cache_config.cache_dtype in (
            "fp8_ds_mla", "nvfp4_ds_mla")'''),
    ('''        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"''',
     '''        use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")'''),
], marker="nvfp4_ds_mla")

# 6. config/cache.py: CacheDType literal
cc = f"{V}/config/cache.py"
patch(cc, [
    ('''    "fp8_ds_mla",
''',
     '''    "fp8_ds_mla",
    "nvfp4_ds_mla",
'''),
], marker='"nvfp4_ds_mla"')

# 7. kv_cache_interface.py: map nvfp4_ds_mla -> NVFP4 quant mode + page size.
ki = f"{V}/v1/kv_cache_interface.py"
patch(ki, [
    ('''    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4''',
     '''    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if kv_cache_dtype == "nvfp4_ds_mla":
        return KVQuantMode.NVFP4'''),
], marker="nvfp4_ds_mla")

# 7b. kv_cache_interface.py: MLAAttentionSpec / SlidingWindowMLASpec page size.
#     The raw KV-cache tensor is allocated at `real_page_size_bytes`, but the
#     view in _reshape_kv_cache_tensors uses get_kv_cache_shape's last dim (416
#     for nvfp4_ds_mla). The generic MLA formula yields block_size * head_size
#     (= 576 for kv_lora 512 + rope 64), so the view fails. Force 416, mirroring
#     fp8_ds_mla's 656. Gate on BOTH cache_dtype_str (the layer-built spec at
#     mla_attention.py passes it) AND kv_quant_mode.is_nvfp4 (the page-size-1
#     validation spec at platforms/interface.py passes only kv_quant_mode).
#     416 = 256 E2M1 + 32 fp8 BSF + 128 bf16 RoPE.  Idempotent via the "* 416"
#     marker (distinct from the get_kv_quant_mode patch above).
if "* 416" in io.open(ki, encoding="utf-8").read():
    print("[run.sh] kv_cache_interface.py: page-size already patched")
else:
    s = io.open(ki, encoding="utf-8").read()
    # MLAAttentionSpec.real_page_size_bytes — insert nvfp4_ds_mla branch first.
    mla_anchor = '''    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":'''
    mla_repl = '''    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "nvfp4_ds_mla" or (
            self.cache_dtype_str is None and self.kv_quant_mode.is_nvfp4
        ):
            # GLM-5.2 sparse-MLA NVFP4: 416-B record (256 E2M1 + 32 fp8 BSF +
            # 128 bf16 RoPE). Must equal flashmla_sparse.get_kv_cache_shape's
            # last dim so _reshape_kv_cache_tensors' view succeeds.
            return self.block_size * 416
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":'''
    assert mla_anchor in s, "MLAAttentionSpec.real_page_size_bytes anchor not found"
    assert s.count(mla_anchor) == 1, "MLAAttentionSpec anchor not unique"
    s = s.replace(mla_anchor, mla_repl, 1)

    # SlidingWindowMLASpec.real_page_size_bytes — same twin (sparse SWA path).
    swa_anchor = '''    @property
    def real_page_size_bytes(self) -> int:
        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":'''
    swa_repl = '''    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "nvfp4_ds_mla" or (
            self.cache_dtype_str is None and self.kv_quant_mode.is_nvfp4
        ):
            # GLM-5.2 sparse-MLA NVFP4 SWA cache: 416-B record (same layout).
            return self.storage_block_size * 416
        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":'''
    assert swa_anchor in s, "SlidingWindowMLASpec.real_page_size_bytes anchor not found"
    assert s.count(swa_anchor) == 1, "SlidingWindowMLASpec anchor not unique"
    s = s.replace(swa_anchor, swa_repl, 1)

    import ast; ast.parse(s)
    io.open(ki, "w", encoding="utf-8").write(s)
    print("[run.sh] patched kv_cache_interface.py (MLA/SWA page_size -> 416)")

# 7c. torch_utils.py: STR_DTYPE_TO_TORCH_DTYPE must map nvfp4_ds_mla -> uint8.
#     Indexed directly in platforms/interface.py for the page-size-1 spec; a
#     KeyError there pre-empts KV-cache allocation entirely.
tu = f"{V}/utils/torch_utils.py"
patch(tu, [
    ('''    "nvfp4": torch.uint8,
}''',
     '''    "nvfp4": torch.uint8,
    "nvfp4_ds_mla": torch.uint8,
}'''),
], marker='"nvfp4_ds_mla": torch.uint8')

# 8. mla_attention.py: convert generic nvfp4 -> nvfp4_ds_mla for FLASHMLA_SPARSE
ma = f"{V}/model_executor/layers/attention/mla_attention.py"
patch(ma, [
    ('''        if (
            self.attn_backend.get_name() == "FLASHMLA_SPARSE"
            and is_quantized_kv_cache(kv_cache_dtype)
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"''',
     '''        if (
            self.attn_backend.get_name() == "FLASHMLA_SPARSE"
            and kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla")
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "nvfp4_ds_mla"
            kv_cache_dtype = "nvfp4_ds_mla"
            logger.info_once(
                "Using GLM-5.2 NVFP4 sparse-MLA KV cache (nvfp4_ds_mla, 416-B)."
            )
        elif (
            self.attn_backend.get_name() == "FLASHMLA_SPARSE"
            and is_quantized_kv_cache(kv_cache_dtype)
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"'''),
], marker="nvfp4_ds_mla")

# 9. backend.py: SparseMLAAttentionImpl.do_kv_cache_update store dispatch.
bk = f"{V}/v1/attention/backend.py"
# Target only the SparseMLAAttentionImpl copy (the LAST do_kv_cache_update with concat).
s = io.open(bk, encoding="utf-8").read()
SPARSE_STORE_MARK = "nvfp4_ds_mla store"
if SPARSE_STORE_MARK in s:
    print("[run.sh] backend.py: already patched")
else:
    needle = '''    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )'''
    # There are two identical copies; the SparseMLAAttentionImpl one is the LAST.
    cnt = s.count(needle)
    assert cnt >= 1, "do_kv_cache_update needle not found in backend.py"
    repl = '''    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype == "nvfp4_ds_mla":  # nvfp4_ds_mla store
            from vllm.v1.attention.backends.mla.sm12x_sparse_mla_attn import (
                _nvfp4ds_store,
            )
            _nvfp4ds_store(
                kv_c_normed,
                k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe,
                kv_cache.view(torch.uint8).reshape(-1, 416),
                slot_mapping.flatten(),
            )
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )'''
    # Replace the LAST occurrence (SparseMLAAttentionImpl).
    idx = s.rfind(needle)
    s = s[:idx] + repl + s[idx + len(needle):]
    import ast; ast.parse(s)
    io.open(bk, "w", encoding="utf-8").write(s)
    print("[run.sh] patched backend.py (SparseMLAAttentionImpl.do_kv_cache_update)")
PYEOF

echo "[run.sh] AST-check final import sanity..."
python3 -c "import vllm.config.cache, vllm.v1.kv_cache_interface; print('[run.sh] config + kv_cache_interface import OK')"
echo "[run.sh] DONE"
