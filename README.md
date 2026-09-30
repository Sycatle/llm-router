# llm-router

Local router for OpenCode (or any OpenAI-compatible client). It exposes one OpenAI-compatible endpoint and
relays to **Anthropic (Claude)**, **OpenAI**, **Mistral** and any OpenAI-compatible server (Ollama, llama.cpp,
OpenRouter...). A classifier (Jev) emits structured signals, a deterministic policy picks the tier, then the model.

```
OpenCode --> POST /v1/chat/completions   (model = auto | auto-fast | auto-standard | auto-reasoning
                |                                 | auto-frontier | <provider/model> = forced)
                v
          api/openai_compat            session id = x-session-id header (OpenCode sends it)
                v
          router::RouterService  -- tool-loop continuation? keep tier, skip classification
                |
                +--> RouterClassifier (trait) -- JevClassifier -- POST api.typesafe.ai/v1/systemone
                |        signals only (task_type, complexity, reasoning, tool_intensity,
                |        latency_sensitivity, ambiguity, confidence); error => neutral, confidence 0
                v
          router::policy (pure)  score -> tier -> hysteresis -> ordered candidates
                |        filters: context window, tools; degraded models last; escalates to higher tiers
                v
          LlmProvider (trait) --+-- anthropic  (Messages API <-> OpenAI translation, SSE)
                                +-- openai     (pass-through: OpenAI, Mistral, Ollama, OpenRouter... any compatible API)
                v
          metrics: SQLite decisions + in-memory cooldown/latency  -->  GET /debug/routes
```

## Run

```sh
export ANTHROPIC_API_KEY=...       # Claude
export TYPESAFE_API_KEY=...        # Jev; without it everything is routed to STANDARD
export OPENAI_API_KEY=... MISTRAL_API_KEY=...   # optional; a provider without credentials is skipped
cargo run                          # reads ./router.toml (or: cargo run -- path/to/router.toml)
RUST_LOG=debug cargo run           # also logs the request headers
```

Edit `router.toml` first: model ids and prices are examples. Anthropic reads `ANTHROPIC_API_KEY`.
Optional and off by default: `auth = "oauth_opencode"` on the Anthropic provider reuses the Claude subscription
token saved by `opencode /connect` (read-only, no refresh). **Warning:** Anthropic's terms may prohibit using
subscription tokens outside their own clients, and it sends a Claude Code system prefix. In testing, small requests
passed but OpenCode-sized ones were rejected (HTTP 400 "out of extra usage"). Use at your own risk.

## OpenCode config (`opencode.json`)

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

Then `/models` -> `LLM Router / Auto`. Modes: `auto`, `auto-<tier>` (forced tier), or any catalog id such as
`anthropic/opus` (FORCE_MODEL, no fallback; add it to `models` above). Header `x-router-min-tier: reasoning`
forces a tier floor. Responses carry `x-router-model`, `x-router-tier`, `x-router-request-id`.

## Policy

`score = 0.40 reasoning + 0.30 complexity + 0.15 tool_intensity + 0.15 ambiguity`, plus task-type bias
(architecture +0.10, debugging/refactor +0.05, simple tasks -0.05), context bias (large +0.05, huge +0.15),
minus `0.10 latency_sensitivity`. Bands from `routing.thresholds`.

- Hysteresis: a new session takes the target tier. In an existing one the score must be `stickiness` outside the
  current band to switch, `switch_threshold` if the target tier was already used in the last 3 requests (no FAST->STANDARD->FAST).
- Confidence below `min_confidence` (Jev down, no key): keep the current tier, or STANDARD.
- Within a tier the current model, then the current provider, is preferred. Tool-loop continuations keep model and tier.
- Fallback order: same tier, then higher tiers, models in cooldown (429/5xx/network/auth: 30-60s, exponential) last.
  Fallback is only possible before the first streamed byte.

## Example decision logs (`RUST_LOG=info`, also stored in `router.db`)

```
route: routed request_id=req-f6ae0740e4ea session_id=ses_f0bd24975ffe... task_type=Some("Other") previous_model=None
  tier=standard model=Some("local/m") reason=low confidence: default standard latency_ms=Some(2) tokens_in=Some(10)
  tokens_out=Some(4) cost=Some(0.0) success=true error=None fallback=false jev={"confidence":0.0,...}
route: routed ... previous_model=Some("local/m") tier=standard reason=tool-loop continuation: keep tier ...
```

With Jev, `reason` reads e.g. `score 0.77 is 0.22 outside standard (margin 0.15): switch` and `jev` holds the scores.
`curl localhost:8787/debug/routes?limit=20` returns the same records as JSON.

## Tests

`cargo test` (31): policy (tiers, Jev down, degraded provider, fallback chain, stickiness, threshold, anti-flap, force model,
context too large), Jev client against a real local HTTP server, Anthropic translation (history, tools, SSE), and
end-to-end router tests against real local HTTP upstreams (fallback on 5xx, 502, streaming + usage, 400 on oversized context).

## Verified / not verified

- Verified with real OpenCode 1.18.33: streaming, tool call round trip, session header `x-session-id`, tool-loop continuation.
- **Not verified: Jev** (no `TYPESAFE_API_KEY` here): the request/response format follows TypeSafe's docs and is tested
  against a local fake only. Check the actual REASONING/FRONTIER routing of your two example prompts with your key.

## Known limits

- OAuth is Anthropic-only, experimental, no refresh; no ChatGPT-subscription backend (OpenAI is API key only).
- Model catalog, prices and context windows are static in `router.toml`; no dynamic discovery, no budget cap.
- OpenCode's title-generation call shares the session id and is routed like a normal request (it is classified separately).
- Reasoning/thinking parameters and image parts on OpenAI upstreams are passed through unchanged, not translated.
- Session state is in memory (lost on restart). Client-disconnect is detected by dropping the upstream stream.
- Jev sees a truncated dossier (last user message 3000 chars, first 500, tool names); this leaves your machine.
- Anthropic `max_tokens` defaults to 8192 when the client sends none.

## TODO (priority order)

1. Run with a Jev key, calibrate `thresholds` and weights from `/debug/routes` data.
2. Test the Anthropic adapter end to end with OpenCode's tools.
3. Idle timeout on streams; retry-after handling for 429.
4. Budget/cost caps in the policy; use measured latency in candidate ordering.
5. Dynamic model discovery (`/v1/models` of upstreams); OpenRouter; local models health probe.
6. Jev extras: MCP tool filtering, subagent suggestion.
7. Persist sessions; CLI `llm-router routes` to inspect decisions.

## License

MIT
