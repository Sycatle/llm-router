# llm-router

**Stop picking models by hand. Let the router pick the cheapest one that can do the job.**

One local endpoint for OpenCode (or any OpenAI-compatible client). Typos go to a fast, cheap model.
Deadlock hunts go to a heavyweight. Out of quota on one provider? It switches to another. You never touch the model picker.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/rust-2024-orange.svg)

| You type | Router picks | Why |
|---|---|---|
| "fix this typo" | **FAST** (e.g. Haiku) | trivial, low latency matters |
| "add pagination to this endpoint" | **STANDARD** (e.g. Sonnet) | routine feature work |
| "find the intermittent deadlock in our async code" | **REASONING / FRONTIER** | subtle, multi-file, high ambiguity |

## Why

- **Cut your bill.** Stop paying frontier prices for one-line edits.
- **Stay fast.** Simple asks get quick models.
- **Never get stuck.** Provider down, rate-limited, or 5xx? The router falls back automatically, same tier first, then higher.
- **Limit-proof.** Hit your Claude, OpenAI or Mistral rate limit or quota? The router detects it, honours `Retry-After`, sidelines the model (or the whole provider when credits run out) and keeps working on the next best one.
- **One endpoint, every provider.** Claude, OpenAI, Mistral, Ollama, llama.cpp, OpenRouter... anything OpenAI-compatible.
- **Predictable.** An AI classifier only emits *signals*. A pure, deterministic, unit-tested policy makes the actual decision.
- **Stable.** Hysteresis avoids flip-flopping models between messages; tool loops keep their model.
- **Inspectable.** Every decision is logged and served at `GET /debug/routes`.

## Quick start

```sh
export ANTHROPIC_API_KEY=...        # Claude
export TYPESAFE_API_KEY=...         # Jev classifier (without it, everything goes to STANDARD)
export OPENAI_API_KEY=...           # optional
export MISTRAL_API_KEY=...          # optional; providers without a key are skipped

cargo run                           # reads ./router.toml
```

Edit `router.toml` first: model ids and prices are examples. Then point OpenCode at it:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "router": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LLM Router",
      "options": { "baseURL": "http://127.0.0.1:8787/v1" },
      "models": {
        "auto": { "name": "Auto" },
        "auto-fast": { "name": "Fast" },
        "auto-standard": { "name": "Standard" },
        "auto-reasoning": { "name": "Reasoning" },
        "auto-frontier": { "name": "Frontier" }
      }
    }
  }
}
```

Then `/models` and choose **LLM Router / Auto**. That's it.

## Modes

| `model` value | Behavior |
|---|---|
| `auto` | Full routing |
| `auto-fast` / `auto-standard` / `auto-reasoning` / `auto-frontier` | Force a tier (fallback still applies) |
| `anthropic/opus` (any catalog id) | Force one model, no fallback (add it to `models` in `opencode.json`) |

Header `x-router-min-tier: reasoning` sets a tier floor. Responses include `x-router-model`, `x-router-tier` and `x-router-request-id`.

## How it works

```
OpenCode --> POST /v1/chat/completions
                |
                v
         Router service ---- tool-loop continuation? keep tier, skip classification
                |
                +--> Classifier (trait) -- Jev (TypeSafe System One)
                |       signals only: task_type, complexity, reasoning, tool_intensity,
                |       latency_sensitivity, ambiguity, confidence
                |       failure => neutral signals, confidence 0
                v
         Policy engine (pure, deterministic)
                |       score -> tier -> hysteresis -> ordered candidates
                |       filters context window + tool support, demotes unhealthy models
                v
         Provider (trait) --+-- anthropic  (Messages API <-> OpenAI translation, SSE)
                            +-- openai     (pass-through: OpenAI, Mistral, Ollama, OpenRouter...)
                |
                v
         Metrics: SQLite decision log + cooldown/latency  -->  GET /debug/routes
```

### Policy

`score = 0.40 reasoning + 0.30 complexity + 0.15 tool_intensity + 0.15 ambiguity`, adjusted by task type
(architecture +0.10, debugging/refactor +0.05, simple tasks -0.05), context size (large +0.05, huge +0.15) and
`-0.10 latency_sensitivity`. Tier bands come from `routing.thresholds`.

- **Hysteresis.** A new session takes the target tier. In an existing one the score must leave the current band by
  `stickiness` to switch, or by `switch_threshold` if that tier was used in the last 3 requests (no FAST, STANDARD, FAST).
- **Low confidence** (Jev down or no key): keep the current tier, or STANDARD.
- **Stickiness.** Within a tier the current model, then the current provider, wins. Tool loops keep model and tier.
- **Fallback.** Same tier, then higher tiers. Models in cooldown go last. Only possible before the first streamed byte.
- **Limits.** Errors are classified, whatever the provider's wording:
  - rate limit (429/529): that model is skipped for `Retry-After` (default 60s);
  - quota, credits or usage cap exhausted (402, OpenAI `insufficient_quota`, Anthropic "out of extra usage"):
    the whole provider is skipped (default 15 min, or `Retry-After`), so the router goes straight to another provider;
  - 5xx/network/auth: short exponential cooldown; 400/413/422: no cooldown (the request is at fault).
  Cooldowns are capped at 6h and cleared by the first success.

### Example log

```
route: routed request_id=req-f6ae0740e4ea session_id=ses_f0bd... task_type=Some("Other") previous_model=None
  tier=standard model=Some("local/m") reason=low confidence: default standard latency_ms=Some(2)
  tokens_in=Some(10) tokens_out=Some(4) cost=Some(0.0) success=true fallback=false jev={...}
```

With Jev, `reason` reads like `score 0.77 is 0.22 outside standard (margin 0.15): switch`.
`curl localhost:8787/debug/routes?limit=20` returns the same records as JSON.

## Anthropic subscription (experimental, off by default)

`auth = "oauth_opencode"` on the Anthropic provider reuses the Claude subscription token saved by `opencode /connect`
(read-only, no refresh). **Warning:** Anthropic's terms may prohibit using subscription tokens outside their own
clients, and this sends a Claude Code system prefix. In testing, small requests passed but OpenCode-sized ones were
rejected (HTTP 400 "out of extra usage"). Use an API key unless you accept that risk.

## Tests

`cargo test` runs 36 tests: the policy (every tier, Jev down, degraded provider, rate-limit and quota handling, fallback chain, stickiness, threshold,
anti-flap, force model, oversized context), the Jev client against a real local HTTP server, Anthropic translation
(history, tools, SSE) and end-to-end router tests against real local HTTP upstreams.

## Status

- Verified with OpenCode 1.18.33: streaming, tool-call round trip, session header `x-session-id`, tool-loop continuation.
- **Jev is not verified against the live API** (no key during development). Its request format follows TypeSafe's docs
  and is tested against a local fake. Calibrate `thresholds` with your own key.

## Known limits

- Model catalog, prices and context windows are static in `router.toml`; no dynamic discovery, no budget cap.
- OAuth is Anthropic-only, experimental, no refresh. OpenAI is API key only.
- OpenCode's title-generation call shares the session id and is classified like a normal request.
- Reasoning/thinking parameters and image parts on OpenAI upstreams are passed through, not translated.
- Sessions live in memory (lost on restart).
- Jev receives a truncated dossier (last user message 3000 chars, first 500, tool names): that data leaves your machine.
- Anthropic `max_tokens` defaults to 8192 when the client sends none.

## Roadmap

1. Calibrate thresholds and weights from real `/debug/routes` data.
2. Test the Anthropic adapter end to end with OpenCode's tools.
3. Stream idle timeout; `Retry-After` handling for 429.
4. Budget caps; use measured latency in candidate ordering.
5. Dynamic model discovery, OpenRouter, local model health probes.
6. Jev extras: MCP tool filtering, subagent suggestion.
7. Persistent sessions; CLI to inspect decisions.

## License

MIT
