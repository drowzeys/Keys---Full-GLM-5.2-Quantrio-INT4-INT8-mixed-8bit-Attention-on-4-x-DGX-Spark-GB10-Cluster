#!/usr/bin/env python3
"""NVFP4 ds_mla round-trip numerical gate.

STORE  : random latent x[N,512] bf16 + rope[N,64] bf16  ->  416-B record
DEQUANT: 416-B record -> bf16 [N,576] (NoPE 512 + RoPE 64)
Compare reconstructed NoPE to the fp32 reference x. Also runs the existing
fp8_ds_mla path as a baseline.

ACCEPTANCE: nvfp4 rel-Frobenius err <= 0.12 (MiMo band ~0.09-0.10).

Run on a GB10:
  docker run --rm --gpus all -v <worktree>:/work --entrypoint python3 \
      vllm-glm52-cuda130:full /work/tests/test_nvfp4_roundtrip.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

import torch
import sm12x_sparse_mla_attn as M


def fp4_amax_quant_dequant_reference(x_f32: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference of nvfp4 (E2M1, per-16 amax/6 fp8 scale, GS=1)."""
    N, D = x_f32.shape
    blk = 16
    pos_grid = torch.tensor([0.0, .5, 1., 1.5, 2., 3., 4., 6.],
                            device=x_f32.device, dtype=torch.float32)
    xb = x_f32.reshape(N, D // blk, blk)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    scale = amax / 6.0
    scale_fp8 = scale.to(torch.float8_e4m3fn).to(torch.float32)
    scale_safe = torch.where(scale_fp8 > 0, scale_fp8, torch.ones_like(scale_fp8))
    xn = xb / scale_safe
    sign = torch.sign(xn)
    mag = xn.abs()
    diff = (mag.unsqueeze(-1) - pos_grid).abs()
    nidx = diff.argmin(dim=-1)
    q = pos_grid[nidx] * sign
    return (q * scale_fp8).reshape(N, D)


def metrics(name, recon, ref):
    recon = recon.float(); ref = ref.float()
    max_abs = (recon - ref).abs().max().item()
    rel = ((recon - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(recon, ref, dim=1)
    print(f"  [{name}] max-abs={max_abs:.4e}  rel-Frob={rel:.5f}  "
          f"cos(min/mean)={cos.min().item():.5f}/{cos.mean().item():.5f}")
    return rel


def fp8_baseline(x, rope, slots, total_slots, dev):
    N = x.shape[0]
    cache = torch.zeros(total_slots, M._FP8DS_ENTRY_BYTES, dtype=torch.uint8, device=dev)
    xf = x.float()
    tile = M._FP8DS_QUANT_TILE; nsc = M._FP8DS_NUM_SCALES
    finfo = torch.finfo(torch.float8_e4m3fn)
    for i in range(N):
        s = int(slots[i].item())
        if s < 0:
            continue
        row = xf[i]
        scales = torch.empty(nsc, dtype=torch.float32, device=dev)
        for t in range(nsc):
            seg = row[t * tile:(t + 1) * tile]
            scale = (seg.abs().max() / finfo.max).clamp(min=1e-12)
            scales[t] = scale
            q = (seg / scale).to(torch.float8_e4m3fn)
            cache[s, t * tile:(t + 1) * tile] = q.view(torch.uint8)
        cache[s, 512:512 + 4 * nsc] = scales.view(torch.uint8)
        cache[s, 528:528 + 2 * 64] = rope[i].to(torch.bfloat16).view(torch.uint8)
    idx = slots.reshape(N, 1).to(torch.int32)
    kv, _ = M._gather_dequant_fp8ds(cache, idx)
    kv = kv.reshape(N, 576)
    vm = (slots >= 0)
    return metrics("fp8_ds_mla NoPE", kv[:, :512][vm], xf[vm])


def run():
    assert torch.cuda.is_available(), "needs a GPU (GB10)"
    dev = "cuda"
    torch.manual_seed(0)
    N = 257
    total_slots = 512
    x = (torch.randn(N, 512, device=dev) * 4.0).to(torch.bfloat16)
    rope = (torch.randn(N, 64, device=dev) * 2.0).to(torch.bfloat16)
    slots = torch.arange(N, dtype=torch.int32, device=dev)
    slots[5] = -1; slots[100] = -1

    cache = torch.zeros(total_slots, M._NVFP4DS_ENTRY_BYTES, dtype=torch.uint8, device=dev)
    M._nvfp4ds_store(x, rope, cache, slots)

    idx = slots.reshape(N, 1).to(torch.int32)
    kv, valid = M._gather_dequant_nvfp4ds(cache, idx)
    kv = kv.reshape(N, 576)
    recon_nope = kv[:, :512]; recon_rope = kv[:, 512:]
    vm = (slots >= 0); xf = x.float()

    print("== nvfp4_ds_mla round-trip ==")
    rel_nv = metrics("nvfp4 NoPE", recon_nope[vm], xf[vm])
    rel_rope = metrics("nvfp4 RoPE (lossless bf16)", recon_rope[vm], rope.float()[vm])
    ref = fp4_amax_quant_dequant_reference(xf[vm])
    metrics("nvfp4 vs pyref", recon_nope[vm], ref)
    if (~vm).any():
        print(f"  invalid-slot NoPE max-abs (want 0): {recon_nope[~vm].abs().max().item():.4e}")

    rel_fp8 = None
    try:
        rel_fp8 = fp8_baseline(x, rope, slots, total_slots, dev)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  [fp8 baseline skipped] {e}")

    print()
    print(f"RESULT nvfp4 rel-Frob = {rel_nv:.5f}  (accept <= 0.12)")
    if rel_fp8 is not None:
        print(f"BASELINE fp8_ds_mla rel-Frob = {rel_fp8:.5f}")
    print(f"RoPE rel-Frob = {rel_rope:.6f}  (want <1e-2)")
    ok = (rel_nv <= 0.12) and (rel_rope < 1e-2)
    print("GATE:", "GREEN" if ok else "RED")
    return 0 if ok else 1


def run_fused_attention_check():
    """Validate the FUSED nvfp4 attend kernel against a dense fp32 reference
    attention computed from the (lossy) nvfp4-reconstructed KV. This isolates
    the kernel's softmax/dot logic from the quantization error."""
    dev = "cuda"
    torch.manual_seed(1)
    T, H, topk = 8, 16, 64
    total_slots = 256
    x = (torch.randn(total_slots, 512, device=dev) * 3.0).to(torch.bfloat16)
    rope = (torch.randn(total_slots, 64, device=dev) * 1.5).to(torch.bfloat16)
    slots = torch.arange(total_slots, dtype=torch.int32, device=dev)
    cache = torch.zeros(total_slots, M._NVFP4DS_ENTRY_BYTES, dtype=torch.uint8, device=dev)
    M._nvfp4ds_store(x, rope, cache, slots)

    q = (torch.randn(T, H, 576, device=dev) * 0.5).to(torch.bfloat16)
    idx = torch.randint(0, total_slots, (T, topk), dtype=torch.int32, device=dev)
    lens = torch.full((T,), topk, dtype=torch.int32, device=dev)
    scale = 576 ** -0.5

    max_score = torch.full((T, H), float("-inf"), dtype=torch.float32, device=dev)
    denom = torch.zeros((T, H), dtype=torch.float32, device=dev)
    acc = torch.zeros((T, H, 512), dtype=torch.float32, device=dev)
    M._fused_gather_dequant_attend_nvfp4ds(q, cache, idx, lens, scale, max_score, denom, acc)
    out_fused = (acc / denom.clamp_min(1e-30).unsqueeze(-1)).float()

    # reference: dequant the same slots, dense attention in fp32
    kv, _ = M._gather_dequant_nvfp4ds(cache, idx)  # [T, topk, 576]
    kvf = kv.float()
    qf = q.float()
    scores = torch.einsum("thd,tkd->thk", qf, kvf) * scale  # [T,H,topk]
    w = torch.softmax(scores, dim=-1)
    out_ref = torch.einsum("thk,tkd->thd", w, kvf[:, :, :512])

    rel = ((out_fused - out_ref).norm() / out_ref.norm()).item()
    print(f"  [fused-attn vs dense-ref] rel-Frob={rel:.5f} (bf16-dot tol; want <2e-2)")
    return rel < 2e-2


if __name__ == "__main__":
    rc = run()
    print()
    print("== fused nvfp4 attention kernel check ==")
    ok2 = True
    if os.environ.get("RUN_FUSED", "1") == "1":
        try:
            ok2 = run_fused_attention_check()
            print("FUSED-ATTN:", "OK" if ok2 else "FAIL")
        except Exception as e:
            import traceback; traceback.print_exc(); ok2 = False
    sys.exit(rc if (rc != 0 or ok2) else 1)
