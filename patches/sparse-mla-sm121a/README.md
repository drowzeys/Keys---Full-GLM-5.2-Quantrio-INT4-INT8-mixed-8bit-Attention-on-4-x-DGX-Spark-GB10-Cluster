# sparse-mla-sm121a (the glm52-b12x-sparse mod)

Installs CosmicRaisins' portable Triton sparse-MLA kernels into vLLM's `v1/attention/backends/mla/`
(`flashmla_sparse.py`, `sm12x_sparse_mla_attn.py`, `sparse_mla_kernels.py`, `sparse_mla_env.py`,
`b12x_sparse_helpers.py`) + applies `patch_flashmla_ops.py` (rebinds `flash_mla_sparse_fwd` /
`flash_mla_with_kvcache` to Triton — no `_flashmla_C`), and `pip install --no-deps b12x==0.23.0`.

The decisive one-line un-gate: in `flashmla_sparse.py`,
`supports_compute_capability` returns `capability.major in [9, 10]` → `[9, 10, 12]` so sm_121a passes.

Kernels: clone https://github.com/CosmicRaisins/glm-5.2-gb10 and copy `kernels/`. Apply via the
Dockerfile in `recipe/BUILD.md` (step 2).
