# GLM-5.2 empty-response fix (agents / OpenAI clients e.g. Hermes)

## Symptom
An agent/chat client gets **empty `content`** from the GLM-5.2 OpenAI endpoint.

## Cause
GLM-5.2 defaults to **thinking mode ON**. With `--reasoning-parser glm45`, the chain-of-thought goes to
`reasoning_content` and the answer to `content`. Two ways that leaves `content` empty:
1. The client reads only `message.content` — the reasoning sits in `reasoning_content`.
2. A modest `max_tokens` is consumed entirely by the thinking block, so the answer is never reached
   (`finish_reason: "length"`).

## Fix (best → fallback)
1. **Send thinking-off in the request body** — direct answers in `content`, faster, no CoT tax:
   ```json
   {"model":"glm-5.2-quanttrio","messages":[...],
    "chat_template_kwargs": {"enable_thinking": false},
    "max_tokens": 1024}
   ```
2. If the client can't add custom kwargs: read `message.reasoning_content` as a fallback when
   `content` is empty, AND raise `max_tokens` (>=1024) so the answer isn't cut off mid-think.
3. Server-side default: bake `enable_thinking=false` into the served chat template so all clients get content.

## Verified
`thinking_off → content="The capital of France is Paris."` (non-empty); thinking-on can return empty
`content` on longer prompts. Keep thinking ON only for hard reasoning/math (GLM-5.2's strength), OFF for
chat / coding / tool-call loops.

---

## Also: agent tool-calling — HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice`

### Symptom
A tool-using agent (e.g. Hermes) sends `tool_choice: "auto"` and gets:
```
HTTP 400: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

### Cause
vLLM requires **both** flags for automatic tool selection. `--tool-call-parser glm47` alone is not
enough — `--enable-auto-tool-choice` is its mandatory partner.

### Fix
Launch the server with both (GLM-5.2 uses the `glm47` tool-call parser):
```
--reasoning-parser glm45 --enable-auto-tool-choice --tool-call-parser glm47
```
Then `tool_choice: "auto"` requests succeed and the model emits `tool_calls`. Verified with a
`get_weather` function call over the NVFP4 100K endpoint. (See `recipe/launch-glm52-tp4.sh`.)
