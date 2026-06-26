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
