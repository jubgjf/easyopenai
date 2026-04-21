# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

`easyopenai` is a thin async wrapper around the OpenAI Python SDK that fans out
chat-completion requests across multiple OpenAI-compatible providers. The full
goals (multi-provider scheduling, stream→non-stream normalization, reasoning
parsing, fail-fast semantics, GitHub-private publishing) live in `README.md` —
read it first when making non-trivial changes.

## Commands

```bash
uv sync                                   # install deps
uv run pytest -v                          # all unit tests (no API cost)
uv run pytest tests/test_health.py -v     # one file
uv run pytest -k "reasoning" -v           # by name
uv run python examples/basic.py           # needs config.yaml + .env

# Smoke test against real APIs (costs a couple of cents per run):
BLTCY_API_KEY=... YUNWU_API_KEY=... uv run pytest tests/test_smoke_real_api.py -v -s
```

The smoke test auto-skips when the env vars are absent. **Keep it minimal —
real API calls cost money.** Do not add iterations or larger payloads without
asking.

## Architecture

Request lifecycle: `Client` loads `AppConfig` → constructs one `Provider` per
config entry → `Client.stream()` builds a fresh `Scheduler` per call →
scheduler spawns N worker coroutines per provider (N = `max_concurrency`) → each
worker pulls `Task`s from the shared queue, checks
`supports(model) ∧ name not in attempted ∧ health.can_serve()`, and either
calls or re-queues the task.

Key invariants worth preserving:

- **Failure routing**: on `ProviderError`, the scheduler appends the provider
  to `task.attempted_providers` and re-queues so a *different* provider picks
  it up. A task that has been tried by every supporting provider returns a
  `Result` with `error` set.
- **Stream/non-stream symmetry**: `Provider._single_call` branches on
  `cfg.force_stream`. The stream branch is funneled through
  `aggregate_stream()` which produces the same dict shape as
  `ChatCompletion.model_dump()`, so the rest of the pipeline never knows the
  difference.
- **Reasoning parser is fail-fast**: when a model is configured
  `is_reasoning: true` but no `reasoning_content` / `reasoning` field /
  `<think>` block can be extracted, `parse_message` asserts. Do not soften
  this without changing the README contract.
- **Health monitor** (`health.py`) is a 3-state circuit breaker
  (CLOSED→OPEN→HALF_OPEN). HALF_OPEN admits exactly one probe; the next
  `record()` decides recovery vs. re-open. Be careful editing — the
  in-flight flag is the part that's easy to get wrong.
- **Token accounting**: `reasoning_tokens` comes from
  `usage.completion_tokens_details.reasoning_tokens` (per the example
  responses in `example.txt`); `answer_tokens = completion_tokens -
  reasoning_tokens`.

## Config & secrets

- `config.yaml` and `.env` are gitignored; `config.example.yaml` and
  `.env.example` are the templates.
- API keys in YAML use `${ENV_VAR}` interpolation, resolved by
  `config._walk_and_interpolate`. Missing env vars fail fast with an assert.
- All log output goes through loguru (`logging.setup_logging`); a regex filter
  masks `sk-…` and `Bearer …` tokens in messages. **Do not import the stdlib
  `logging` module** — the project standardizes on `from loguru import logger`.

## Conventions

- Python 3.12, pydantic v2, asyncio throughout. `pytest-asyncio` runs in
  `auto` mode (set in `pyproject.toml`), so `async def test_*` works without
  decorators.
- `assert` is used liberally for fail-fast invariants per the README. Don't
  replace asserts with soft validation unless the user asks.
- Public surface is `easyopenai/__init__.py` — only `Client`, `Task`,
  `Result`, `TokenUsage`, `setup_logging` are exported. Keep it that way.
