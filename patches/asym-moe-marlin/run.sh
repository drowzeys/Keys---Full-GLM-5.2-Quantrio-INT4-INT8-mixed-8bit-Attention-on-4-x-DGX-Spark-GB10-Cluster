#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Part of: github.com/drowzeys (full GLM-5.2 on 4xGB10 recipe)
# Apply the asymmetric compressed-tensors WNA16 Marlin MoE patch inside the
# vLLM container, then run syntax + import validation.
#
# Usage (inside the container):
#   bash /path/to/run.sh
#
# This script expects patch_ct_asym_moe.py to sit next to it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER="${HERE}/patch_ct_asym_moe.py"
TARGET="/opt/env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py"

echo "== Backing up target =="
cp -n "${TARGET}" "${TARGET}.orig" || true

echo "== Applying patch =="
python3 "${PATCHER}"

echo "== Syntax check =="
python3 -c "import ast; ast.parse(open('${TARGET}').read()); print('AST OK')"

echo "== Import check =="
python3 -c "import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin as m; print('IMPORT OK')"

echo "== Done =="
echo
echo "NOTE: End-to-end Marlin numeric validation on GB10 is currently blocked by a"
echo "driver/PTX mismatch (vllm _C ships sm_121a PTX built with CUDA 13.2; driver"
echo "580.159.03 only consumes CUDA 13.0 PTX -> cudaErrorUnsupportedPtxVersion on"
echo "ANY Marlin op, symmetric or asymmetric). The zp-layout conversion was instead"
echo "validated structurally as byte-identical to vLLM's production CT *linear*"
echo "Marlin asym conversion. Run numtest.py once the image/driver supports sm_121a"
echo "Marlin SASS/PTX to get the numeric symmetric-vs-asymmetric comparison."
