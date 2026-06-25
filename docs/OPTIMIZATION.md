# Optimization analysis — full GLM-5.2 on 4×GB10

Grounded in the live engine logs of the serving deployment (TP=4, MTP k=3, util 0.86, ~13 tok/s
single-stream decode). KV measured at ~53.5 KiB/token (fp8_ds_mla, block_size 64, 78 layers).

## Ranked levers
| # | Lever | Expected gain | Verdict |
|---|---|---|---|
| 1 | `num-gpu-blocks-override` 768 → 819 | free KV (you're capping below vLLM's own safe estimate) | **Do it** |
| 2 | Thinking-off default (chat/coding/tools) | biggest wall-clock win (~38 s/CoT-block @ 13 tok/s) | **Do it** — keep ON only for hard math/reasoning |
| 3 | `--max-num-batched-tokens 4096` | clears vLLM's MTP `max_num_scheduled_tokens=2048` warning | **Do it** |
| 4 | MTP-k sweep {2,3,4,5} | maybe +10-25% single-stream IF acceptance holds | **Test** — GLM-5.2 has ONE MTP layer reused k×; k=3 is the GB10 community default |
| 5 | 64K context | +~3.3 GiB KV; fits. 128K only at concurrency 1.0× + Nemotron stopped | Optional |
| 6 | `max-num-seqs` 1→2 | aggregate throughput (batched expert GEMMs); single-stream unchanged | Multi-client |
| 7 | **DFlash** | — | **N/A — no GLM-5.2 drafter exists** (DFlash is target-trained; only gated GLM-5.1). Wiring ready for when one ships |
| 8 | **Expert Parallel** | — | **Trap** — experts already sharded at TP=4 (no memory win); naive all-to-all over 200G is a throughput loss; risks W4A16 Marlin TP bugs |

## "MTP2" clarified
= `num_speculative_tokens` (autoregressive reuse count of the single MTP head), NOT 2 MTP layers.
GLM-5.2 ships `num_nextn_predict_layers=1`.

## Recommended next configs
**(A) Max single-stream:** `--num-gpu-blocks-override 819 --max-num-batched-tokens 4096
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` + thinking-off default. Then sweep k.

**(B) Max aggregate throughput:** `--max-num-seqs 2 --num-gpu-blocks-override 1536
--max-num-batched-tokens 4096 --speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
Do NOT enable EP.
