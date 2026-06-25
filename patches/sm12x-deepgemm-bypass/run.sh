#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Part of: github.com/drowzeys (full GLM-5.2 on 4xGB10 recipe)
# glm52-sm12x-sparse :: DSA-indexer DeepGEMM bypass for GB10 / sm_121 (cap family 120)
#
# Wires the vendored sm12x DeepGEMM fallback kernels into the vLLM tree and
# rewrites the two gates that otherwise force the DeepSeek Sparse Attention
# (DSA) indexer down the DeepGEMM-only path:
#
#   1. vllm/v1/attention/ops/deepseek_v4_ops/{sm12x_mqa,sm12x_deep_gemm_fallbacks,
#      b12x_sparse_helpers}.py  -- copied in (the fallback impls).
#   2. vllm/utils/deep_gemm.py  -- fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits /
#      tf32_hc_prenorm_gemm short-circuit to the sm12x_* fallbacks on cap family
#      120 BEFORE the _missing() / has_deep_gemm() gate.
#   3. vllm/model_executor/layers/sparse_attn_indexer.py -- the SparseAttnIndexer
#      constructor gate no longer requires has_deep_gemm() on cap family 120.
#
# Idempotent: re-running is a no-op. Every patched file is AST-checked.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS_SRC="${MOD_DIR}/kernels"

VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
DG_OPS_DIR="${VLLM_DIR}/v1/attention/ops/deepseek_v4_ops"
DEEP_GEMM_PY="${VLLM_DIR}/utils/deep_gemm.py"
INDEXER_PY="${VLLM_DIR}/model_executor/layers/sparse_attn_indexer.py"

echo "[glm52-sm12x-sparse] vLLM tree: ${VLLM_DIR}"

# ---------------------------------------------------------------------------
# 1. Copy the fallback kernels into deepseek_v4_ops/
# ---------------------------------------------------------------------------
mkdir -p "${DG_OPS_DIR}"
[ -f "${DG_OPS_DIR}/__init__.py" ] || touch "${DG_OPS_DIR}/__init__.py"
for f in sm12x_mqa.py sm12x_deep_gemm_fallbacks.py b12x_sparse_helpers.py; do
    install -m 0644 "${KERNELS_SRC}/${f}" "${DG_OPS_DIR}/${f}"
    echo "[glm52-sm12x-sparse] installed ${DG_OPS_DIR}/${f}"
done

# ---------------------------------------------------------------------------
# 2 & 3. In-place patches via a Python AST/text patcher (idempotent).
# ---------------------------------------------------------------------------
DEEP_GEMM_PY="${DEEP_GEMM_PY}" INDEXER_PY="${INDEXER_PY}" python3 - <<'PYEOF'
import ast
import os
import sys

MARK = "GLM52_SM12X_SPARSE"  # idempotency marker

def read(p):
    with open(p, "r") as fh:
        return fh.read()

def write(p, s):
    ast.parse(s)  # fail loudly before we clobber the file
    with open(p, "w") as fh:
        fh.write(s)
    print(f"[glm52-sm12x-sparse] patched + AST-OK: {p}")

# ===========================================================================
# deep_gemm.py
# ===========================================================================
dg_path = os.environ["DEEP_GEMM_PY"]
src = read(dg_path)

if MARK not in src:
    # Helper inserted just before the first wrapper we patch. It dispatches the
    # three DeepGEMM-only ops to the vendored sm12x fallbacks on cap family 120,
    # and is a no-op (returns None / False) on every other platform so the
    # original DeepGEMM path is preserved untouched off-GB10.
    helper = '''
# --- GLM52_SM12X_SPARSE: sm_121 (cap family 120) DeepGEMM fallback dispatch ---
def _glm52_sm12x_active() -> bool:
    """True only on CUDA cap family 120 (consumer Blackwell / GB10 sm_121)."""
    try:
        return bool(
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        )
    except Exception:
        return False


def _glm52_fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits):
    from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_deep_gemm_fallbacks import (
        _fp8_mqa_logits_sm12x,
    )
    return _fp8_mqa_logits_sm12x(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )


def _glm52_fp8_fp4_paged_mqa_logits(
    q, kv_cache, weights, context_lens, block_tables, max_model_len
):
    from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_deep_gemm_fallbacks import (
        _fp8_paged_mqa_logits_sm12x,
    )
    return _fp8_paged_mqa_logits_sm12x(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def _glm52_tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split):
    from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_deep_gemm_fallbacks import (
        _tf32_hc_prenorm_gemm_sm12x,
    )
    return _tf32_hc_prenorm_gemm_sm12x(x, fn, out, sqrsum, num_split)
# --- end GLM52_SM12X_SPARSE helper ---

'''
    anchor = "def fp8_fp4_mqa_logits("
    assert anchor in src, "deep_gemm.py: fp8_fp4_mqa_logits anchor not found"
    src = src.replace(anchor, helper + anchor, 1)

    # Short-circuit fp8_fp4_mqa_logits: route to sm12x BEFORE _lazy_init/_missing.
    old = (
        '''    _lazy_init()
    if _fp8_fp4_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_mqa_logits_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )'''
    )
    new = (
        '''    if _glm52_sm12x_active():  # GLM52_SM12X_SPARSE
        return _glm52_fp8_fp4_mqa_logits(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
        )
    _lazy_init()
    if _fp8_fp4_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_mqa_logits_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )'''
    )
    assert old in src, "deep_gemm.py: fp8_fp4_mqa_logits body not found verbatim"
    src = src.replace(old, new, 1)

    # Short-circuit fp8_fp4_paged_mqa_logits. The sm12x fallback ignores
    # schedule_metadata (the indexer leaves its buffer unpopulated when
    # has_deep_gemm() is False) and clean_logits (the indexer passes False;
    # the topk kernels re-mask). Drop both at the boundary.
    old = (
        '''    _lazy_init()
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_paged_mqa_logits_impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )'''
    )
    new = (
        '''    if _glm52_sm12x_active():  # GLM52_SM12X_SPARSE
        return _glm52_fp8_fp4_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )
    _lazy_init()
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_paged_mqa_logits_impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )'''
    )
    assert old in src, "deep_gemm.py: fp8_fp4_paged_mqa_logits body not found verbatim"
    src = src.replace(old, new, 1)

    # Short-circuit tf32_hc_prenorm_gemm.
    old = (
        '''    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )'''
    )
    new = (
        '''    if _glm52_sm12x_active():  # GLM52_SM12X_SPARSE
        return _glm52_tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split)
    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )'''
    )
    assert old in src, "deep_gemm.py: tf32_hc_prenorm_gemm body not found verbatim"
    src = src.replace(old, new, 1)

    write(dg_path, src)
else:
    print(f"[glm52-sm12x-sparse] already patched (marker present): {dg_path}")

# ===========================================================================
# sparse_attn_indexer.py  -- constructor gate
# ===========================================================================
ix_path = os.environ["INDEXER_PY"]
src = read(ix_path)

if MARK not in src:
    old = (
        '''        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )'''
    )
    new = (
        '''        # GLM52_SM12X_SPARSE: on cap family 120 (sm_121 / GB10 consumer
        # Blackwell) the DSA indexer runs on the vendored sm12x Triton
        # fallbacks wired into vllm.utils.deep_gemm, so DeepGEMM is not needed.
        _glm52_sm12x = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        )
        if current_platform.is_cuda() and not has_deep_gemm() and not _glm52_sm12x:
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )'''
    )
    assert old in src, "sparse_attn_indexer.py: constructor gate not found verbatim"
    src = src.replace(old, new, 1)
    write(ix_path, src)
else:
    print(f"[glm52-sm12x-sparse] already patched (marker present): {ix_path}")
PYEOF

# ---------------------------------------------------------------------------
# Final AST/byte-compile sweep on everything we touched.
# ---------------------------------------------------------------------------
python3 -m py_compile \
    "${DEEP_GEMM_PY}" \
    "${INDEXER_PY}" \
    "${DG_OPS_DIR}/sm12x_mqa.py" \
    "${DG_OPS_DIR}/sm12x_deep_gemm_fallbacks.py" \
    "${DG_OPS_DIR}/b12x_sparse_helpers.py"
echo "[glm52-sm12x-sparse] py_compile OK for all patched/installed files"
echo "[glm52-sm12x-sparse] DONE"
