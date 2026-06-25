# Benchmark results — GLM-5.2 (QuantTrio Int4-Int8Mix, full) TP=4 on 4×GB10

Config: thinking OFF, temperature 0, single-stream + MTP k=3, ~13 tok/s decode. Raw JSON in `results/`.

| Category | Benchmark | Score | n |
|---|---|---|---|
| Coding | HumanEval pass@1 (evalplus) | **0.963** base / 0.927 plus | 164 |
| Math | GSM8K | **0.975** | 40 |
| Reasoning | MMLU-Pro | **0.825** | 40 |
| Hard science | GPQA-Diamond | **0.750** | 20 |

Thinking-ON would likely raise the reasoning/hard-science scores (GLM-5.2's strength) at higher latency.
Runners: `hybrid_gsm8k.py`, `mc_run.py` (mmlu_pro|gpqa), `hybrid_codegen.py` (+ evalplus). All send
`chat_template_kwargs:{enable_thinking:false}` and cap `max_tokens` below `max_model_len`.
