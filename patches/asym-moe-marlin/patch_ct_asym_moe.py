#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Part of: github.com/drowzeys (full GLM-5.2 on 4xGB10 recipe)
"""
In-place patch: enable ASYMMETRIC (AWQ-style, sym=false) compressed-tensors
WNA16 MoE on the Marlin kernel.

Target file:
  vllm/model_executor/layers/quantization/compressed_tensors/
      compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py

What this does (4-bit group-quantized asymmetric only):
  * Removes the hard `assert weight_quant.symmetric` block; instead sets
    self.has_zp and only allows asymmetric when num_bits == 4 and strategy
    is group/channel (the AWQ-int4 case this model uses).
  * Sets self.quant_type = scalar_types.uint4 (asym) instead of uint4b8 (sym).
  * Registers w13_weight_zero_point / w2_weight_zero_point int32 params whose
    *checkpoint* shape is [out//pack, num_groups]; after the FusedMoE loader's
    is_transposed .t(), the layer param holds [E, num_groups, out//pack]
    (the same orientation as AWQMarlinMoEMethod.w13_qzeros).
  * In process_weights_after_loading, converts the loaded compressed-tensors
    zero-points to the Marlin zp layout, per expert, via:
        marlin_zero_points(unpack_cols(zp_e, 4, num_groups, size_n),
                           size_k=num_groups, size_n=size_n, num_bits=4)
    NOTE: this is the compressed-tensors conversion (the same one the CT linear
    Marlin kernel uses), NOT moe_awq_to_marlin_zero_points. CT does NOT
    pre-interleave the zp bits the way AWQ checkpoints do, so we must skip the
    undo-interleave that moe_awq_to_marlin_zero_points performs.
  * Passes the converted zp through get_fused_moe_quant_config (w1_zp/w2_zp)
    and the legacy apply() fused_marlin_moe call (w1_zeros/w2_zeros).

The patch is idempotent: it bails out if the marker is already present.
"""
import io
import re
import sys

F = (
    "/opt/env/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "quantization/compressed_tensors/compressed_tensors_moe/"
    "compressed_tensors_moe_wna16_marlin.py"
)

src = io.open(F, "r", encoding="utf-8").read()

MARKER = "# ASYM_MOE_PATCH_APPLIED"
if MARKER in src:
    print("Patch already applied; nothing to do.")
    sys.exit(0)

orig = src

# ---------------------------------------------------------------------------
# 1) Imports: scalar_types, marlin_zero_points, unpack_cols.
#    Hook onto the existing marlin_utils import block.
# ---------------------------------------------------------------------------
imp_anchor = "from vllm.model_executor.layers.quantization.utils.marlin_utils import (\n"
assert imp_anchor in src, "marlin_utils import anchor not found"
src = src.replace(
    imp_anchor,
    imp_anchor + "    marlin_zero_points,\n",
    1,
)

# add unpack_cols + scalar_types imports right after that import block's logger line
logger_anchor = "logger = init_logger(__name__)\n"
assert logger_anchor in src, "logger anchor not found"
src = src.replace(
    logger_anchor,
    "from vllm.model_executor.layers.quantization.utils.quant_utils import (\n"
    "    unpack_cols,\n"
    ")\n"
    "from vllm.scalar_type import scalar_types\n"
    + logger_anchor,
    1,
)

# ---------------------------------------------------------------------------
# 2) Replace the symmetric assertion with asym-allow + has_zp.
# ---------------------------------------------------------------------------
assert_block = (
    '        assert weight_quant.symmetric, (\n'
    '            "Only symmetric quantization is supported for MoE"\n'
    "        )\n"
)
assert assert_block in src, "symmetric assert block not found"
replacement_assert = (
    "        " + MARKER + "\n"
    "        self.has_zp = not weight_quant.symmetric\n"
    "        if self.has_zp:\n"
    "            assert weight_quant.num_bits == 4, (\n"
    '                "Asymmetric compressed-tensors MoE is only supported for "\n'
    '                "4-bit weights"\n'
    "            )\n"
    "            assert weight_quant.strategy in (\"group\", \"channel\"), (\n"
    '                "Asymmetric compressed-tensors MoE requires group/channel "\n'
    '                "quantization"\n'
    "            )\n"
)
src = src.replace(assert_block, replacement_assert, 1)

# ---------------------------------------------------------------------------
# 3) quant_type: uint4 (asym) vs uint4b8 (sym).
# ---------------------------------------------------------------------------
qt_line = "        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[self.num_bits]\n"
assert qt_line in src, "quant_type line not found"
src = src.replace(
    qt_line,
    "        if self.has_zp:\n"
    "            self.quant_type = scalar_types.uint4\n"
    "        else:\n"
    + qt_line.replace("        ", "            ", 1),
    1,
)

# ---------------------------------------------------------------------------
# 4) create_weights: register zero-point params (asym only).
#    Insert right before `layer.a13_scale = None`.
# ---------------------------------------------------------------------------
cw_anchor = "        layer.a13_scale = None\n"
assert cw_anchor in src, "create_weights tail anchor not found"
zp_create = (
    "        if self.has_zp:\n"
    "            # compressed-tensors stores weight_zero_point with checkpoint\n"
    "            # shape [out // pack_factor, num_groups] (packed along the\n"
    "            # output dim). The FusedMoE loader transposes 2D CT tensors\n"
    "            # (is_transposed), so the registered param uses the\n"
    "            # post-transpose orientation [E, num_groups, out // pack].\n"
    "            w13_qzeros = torch.nn.Parameter(\n"
    "                torch.zeros(\n"
    "                    num_experts,\n"
    "                    num_groups_w13,\n"
    "                    (2 if self.moe.is_act_and_mul else 1)\n"
    "                    * intermediate_size_per_partition\n"
    "                    // self.packed_factor,\n"
    "                    dtype=torch.int32,\n"
    "                ),\n"
    "                requires_grad=False,\n"
    "            )\n"
    "            layer.register_parameter(\"w13_weight_zero_point\", w13_qzeros)\n"
    "            set_weight_attrs(w13_qzeros, extra_weight_attrs)\n"
    "\n"
    "            w2_qzeros = torch.nn.Parameter(\n"
    "                torch.zeros(\n"
    "                    num_experts,\n"
    "                    num_groups_w2,\n"
    "                    hidden_size // self.packed_factor,\n"
    "                    dtype=torch.int32,\n"
    "                ),\n"
    "                requires_grad=False,\n"
    "            )\n"
    "            layer.register_parameter(\"w2_weight_zero_point\", w2_qzeros)\n"
    "            set_weight_attrs(w2_qzeros, extra_weight_attrs)\n"
    "\n"
)
src = src.replace(cw_anchor, zp_create + cw_anchor, 1)

# ---------------------------------------------------------------------------
# 5) process_weights_after_loading: convert zp to marlin layout.
#    Insert right before `layer.workspace = marlin_make_workspace_new`.
# ---------------------------------------------------------------------------
pw_anchor = "        layer.workspace = marlin_make_workspace_new(device, 4)\n"
assert pw_anchor in src, "process_weights workspace anchor not found"
zp_process = (
    "        if self.has_zp:\n"
    "            # Convert compressed-tensors zero-points -> Marlin zp layout.\n"
    "            # Loaded param orientation is [E, num_groups, out // pack].\n"
    "            # This mirrors the CT *linear* Marlin kernel which does\n"
    "            #   marlin_zero_points(unpack_cols(zp.t(), ...))\n"
    "            # except the .t() is unnecessary here because the FusedMoE\n"
    "            # loader already transposed the 2D CT tensor. We must NOT use\n"
    "            # moe_awq_to_marlin_zero_points: CT does not pre-interleave\n"
    "            # the zp bits the way AWQ checkpoints do.\n"
    "            def _ct_zp_to_marlin(qzeros, size_n):\n"
    "                n_exp = qzeros.shape[0]\n"
    "                num_groups = qzeros.shape[1]\n"
    "                outs = []\n"
    "                for e in range(n_exp):\n"
    "                    unpacked = unpack_cols(\n"
    "                        qzeros[e], self.num_bits, num_groups, size_n\n"
    "                    )\n"
    "                    mz = marlin_zero_points(\n"
    "                        unpacked,\n"
    "                        size_k=num_groups,\n"
    "                        size_n=size_n,\n"
    "                        num_bits=self.num_bits,\n"
    "                    )\n"
    "                    outs.append(mz)\n"
    "                return torch.stack(outs, dim=0).contiguous()\n"
    "\n"
    "            size_n_w13 = (\n"
    "                layer.w13_weight_zero_point.shape[2] * self.packed_factor\n"
    "            )\n"
    "            size_n_w2 = (\n"
    "                layer.w2_weight_zero_point.shape[2] * self.packed_factor\n"
    "            )\n"
    "            marlin_w13_zp = _ct_zp_to_marlin(\n"
    "                layer.w13_weight_zero_point, size_n_w13\n"
    "            )\n"
    "            marlin_w2_zp = _ct_zp_to_marlin(\n"
    "                layer.w2_weight_zero_point, size_n_w2\n"
    "            )\n"
    "            replace_parameter(layer, \"w13_weight_zero_point\", marlin_w13_zp)\n"
    "            replace_parameter(layer, \"w2_weight_zero_point\", marlin_w2_zp)\n"
    "\n"
)
src = src.replace(pw_anchor, zp_process + pw_anchor, 1)

# ---------------------------------------------------------------------------
# 6) get_fused_moe_quant_config: pass w1_zp/w2_zp.
# ---------------------------------------------------------------------------
gfq_old = (
    "            w1_zp=None,\n"
    "            w2_zp=None,\n"
)
assert gfq_old in src, "quant_config w1_zp=None anchor not found"
gfq_new = (
    "            w1_zp=getattr(layer, \"w13_weight_zero_point\", None)\n"
    "            if self.has_zp\n"
    "            else None,\n"
    "            w2_zp=getattr(layer, \"w2_weight_zero_point\", None)\n"
    "            if self.has_zp\n"
    "            else None,\n"
)
src = src.replace(gfq_old, gfq_new, 1)

# ---------------------------------------------------------------------------
# 7) Legacy apply(): pass w1_zeros/w2_zeros into fused_marlin_moe.
#    Insert after the two w*_weight_scale positional args.
# ---------------------------------------------------------------------------
apply_old = (
    "            layer.w13_weight_scale,\n"
    "            layer.w2_weight_scale,\n"
    "            topk_weights,\n"
    "            topk_ids,\n"
)
assert apply_old in src, "apply() scale args anchor not found"
apply_new = (
    "            layer.w13_weight_scale,\n"
    "            layer.w2_weight_scale,\n"
    "            topk_weights,\n"
    "            topk_ids,\n"
    "            w1_zeros=getattr(layer, \"w13_weight_zero_point\", None)\n"
    "            if self.has_zp\n"
    "            else None,\n"
    "            w2_zeros=getattr(layer, \"w2_weight_zero_point\", None)\n"
    "            if self.has_zp\n"
    "            else None,\n"
)
src = src.replace(apply_old, apply_new, 1)

assert src != orig, "no substitutions made"

io.open(F, "w", encoding="utf-8").write(src)
print("Patch applied successfully to:")
print("  " + F)
