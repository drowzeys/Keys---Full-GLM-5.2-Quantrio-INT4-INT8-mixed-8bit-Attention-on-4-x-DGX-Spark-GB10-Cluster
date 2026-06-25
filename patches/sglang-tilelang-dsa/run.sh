#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of: github.com/drowzeys (full GLM-5.2 on 4xGB10 recipe)
# FRONTIER PATCH: route GLM-5.2 DSA indexer through SGLang's TileLang fp8_index on sm_121a (GB10),
# bypassing the DeepGEMM arch-gate. The TileLang forward_indexer path is already in-tree (the 'else'
# branch); we just make sm_121a take it. Proven: tilelang fp8_index compiles+runs on sm_121a.
set -uo pipefail
F=$(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__)+'/srt/layers/attention/nsa/nsa_indexer.py')")
echo "[route-sm121] patching $F"
python3 - "$F" <<'PY'
import sys,re
p=sys.argv[1]; s=open(p).read()
if "_is_sm121a" not in s:
    # add detection right after _is_cuda assignment
    s=s.replace("_is_cuda = is_cuda()",
                "_is_cuda = is_cuda()\n"
                "try:\n"
                "    import torch as _t_sm\n"
                "    _is_sm121a = _is_cuda and _t_sm.cuda.is_available() and _t_sm.cuda.get_device_capability()==(12,1)\n"
                "except Exception:\n"
                "    _is_sm121a = False",1)
# reroute the topk dispatch: sm_121a -> TileLang forward_indexer (else branch)
s=s.replace("        if _is_cuda or _is_hip:\n            assert forward_batch.seq_lens_cpu is not None",
            "        if (_is_cuda or _is_hip) and not _is_sm121a:\n            assert forward_batch.seq_lens_cpu is not None",1)
open(p,"w").write(s)
print("[route-sm121] _is_sm121a added:", "_is_sm121a" in s)
import ast; ast.parse(s); print("[route-sm121] syntax OK")
PY
echo "[route-sm121] done"
