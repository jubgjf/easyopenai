# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

`easyopenai` is an async multi-provider aggregation scheduler for OpenAI-compatible
APIs. It load-balances chat-completion requests across multiple providers with
automatic failover, circuit-breaker health monitoring, stream/non-stream
normalization, and reasoning-model content extraction.

Full feature docs and API reference are in `README.md`. Developer architecture
details are in `docs/DEVELOPMENT.md`.

## Commands

```bash
uv sync                                   # install deps
uv run pytest -v                          # all unit tests (no API cost, ~30s)
uv run pytest tests/test_health.py -v     # one file
uv run pytest -k "reasoning" -v           # by name
uv run python examples/simple_chat.py     # needs config.yaml + .env

# Smoke test against real APIs (costs a couple of cents per run):
BLTCY_API_KEY=... YUNWU_API_KEY=... uv run pytest tests/test_smoke_real_api.py -v -s

# Integration test against local vLLM (requires server at localhost:8000):
no_proxy=localhost,127.0.0.1 uv run pytest tests/test_logprobs_integration.py -v -s
```

The smoke test and integration test auto-skip when their servers/env vars are
absent. **Keep them minimal — real API calls cost money.**

## Architecture

Request lifecycle: `Client` loads `AppConfig` -> constructs one `Provider` per
config entry -> `Client.stream()` builds a fresh `Scheduler` per call ->
scheduler spawns N worker coroutines per provider (N = `max_concurrency`) -> each
worker pulls `Task`s from the shared queue, checks
`supports(model) AND name not in attempted AND health.can_serve()`, and either
calls or re-queues the task.

The stream/non-stream decision is made in the **backend layer** (`backends.py`),
not in `Provider`. `OpenAIBackend.call()` checks `_force_stream_for(model)` and
either streams (funneled through `aggregate_stream()`) or calls directly. Both
paths return the same dict shape. `Provider.call()` just invokes
`self._backend.call(task)`.

### Backend engines

- **`openai`** (default): requires `base_url` + `api_key`. Uses the OpenAI
  Python SDK (`AsyncOpenAI`).
- **`hmwrangler`**: requires `sub_account_name`. Calls `hm_aigc.aigc_managed()`
  in a thread. Does not support streaming or logprobs — logprobs requests will
  fail validation and trigger failover.

### Per-model overrides

`force_stream`, `max_concurrency`, and `max_rpm` can be set at the model level
in config YAML, overriding provider-level defaults. When set, `Provider` creates
separate `asyncio.Semaphore` / `AsyncLimiter` instances per model. Resolution
order: model-level (if not None) -> provider-level.

### Retry mechanism

`Provider.call()` uses tenacity: `stop_after_attempt(3)` with
`wait_random_exponential(min=1, max=10)`. Retryable errors: `RateLimitError`,
`APITimeoutError`, `APIConnectionError`, and `APIStatusError` with
`status_code >= 500`. The rate limiter is re-acquired on each retry attempt.
A single 5xx test can take 10+ seconds due to backoff — this is expected.

### Key invariants

- **Failure routing**: on `ProviderError`, the scheduler appends the provider
  to `task.attempted_providers` and re-queues so a *different* provider picks
  it up. A task tried by every supporting provider returns a `Result` with
  `error` set.
- **Stream/non-stream symmetry**: `aggregate_stream()` produces the same dict
  shape as `ChatCompletion.model_dump()`. The rest of the pipeline never knows
  the difference.
- **Reasoning parser is fail-fast**: when `is_reasoning: true` but no
  `reasoning_content` / `reasoning` field / `<think>` block can be extracted,
  `parse_message` asserts. Do not soften this.
- **Health monitor** (`health.py`) is a 3-state circuit breaker
  (CLOSED -> OPEN -> HALF_OPEN). HALF_OPEN admits exactly one probe; the next
  `record()` decides recovery vs. re-open.
- **Token accounting**: `reasoning_tokens` from
  `usage.completion_tokens_details.reasoning_tokens`;
  `answer_tokens = completion_tokens - reasoning_tokens` (clamped to 0).
- **AssertionError propagates unwrapped**: all other exceptions from
  `Provider.call()` are wrapped in `ProviderError`. This distinguishes
  programming bugs from operational failures.

### Logprobs

- Requested via `task.extra = {"logprobs": True, "top_logprobs": N}`.
- After the response, `_validate_logprobs_match_text` reconstructs text from
  `choice.logprobs.content` entries and checks it against candidate target
  strings covering all reasoning+answer wrapping variants (think tags,
  concatenation, etc.).
- Trailing EOS tokens (e.g. `<|im_end|>`) are stripped before comparison.
- Validation failure raises `ProviderError`, triggering normal failover.
- In streaming mode, logprobs for reasoning and answer tokens all accumulate
  into a single `choice.logprobs.content` array.

## Config & secrets

- `config.yaml` and `.env` are gitignored; `config.example.yaml` and
  `.env.example` are the templates.
- API keys in YAML use `${ENV_VAR}` interpolation, resolved by
  `config._walk_and_interpolate`. Missing env vars fail fast with an assert.
- All log output goes through loguru (`logging.setup_logging`); a regex filter
  masks `sk-...` and `Bearer ...` tokens in messages. **Do not import the stdlib
  `logging` module** — the project standardizes on `from loguru import logger`.

## Conventions

- Python 3.10, pydantic v2, asyncio throughout. `pytest-asyncio` runs in
  `auto` mode (set in `pyproject.toml`), so `async def test_*` works without
  decorators.
- `assert` is used liberally for fail-fast invariants per the README. Don't
  replace asserts with soft validation unless the user asks.
- Public surface is `easyopenai/__init__.py` — only `Client`, `Task`,
  `Result`, `TokenUsage`, `setup_logging` are exported. Keep it that way.

## Testing notes

- Fault injection tests use `respx` to mock httpx at the network layer. Both
  providers' `/models` routes must be mocked for `Client.__aenter__` ping.
- Do not set `respx.mock(assert_all_called=True)` — circuit-breaker tests
  intentionally skip some routes.
- `test_logprobs_integration.py` covers the 2x2x2 matrix of logprobs,
  reasoning, and streaming against a local vLLM. Needs `no_proxy` set to
  bypass macOS system proxy for localhost.
