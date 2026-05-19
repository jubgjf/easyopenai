"""Integration tests: logprobs × reasoning × streaming (2×2×2 matrix).

Requires a local vLLM server at http://localhost:8000/v1 serving qwen3-0.6b.
Auto-skips when the server is not reachable.
"""

from __future__ import annotations

import os

os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import httpx
import pytest

from easyopenai import Client, Task
from easyopenai.config import (
    AppConfig,
    LoggingConfig,
    ModelConfig,
    ProviderConfig,
    SchedulerConfig,
)

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen3-0.6b"


def _server_available() -> bool:
    try:
        r = httpx.get(f"{VLLM_BASE_URL}/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_available(), reason="local vLLM not running")


def _make_config(*, force_stream: bool, is_reasoning: bool) -> AppConfig:
    return AppConfig(
        logging=LoggingConfig(stats_interval_s=600),
        scheduler=SchedulerConfig(),
        providers=[
            ProviderConfig(
                name="local-vllm",
                engine="openai",
                base_url=VLLM_BASE_URL,
                api_key="dummy",
                max_concurrency=1,
                max_rpm=60,
                force_stream=force_stream,
                models=[ModelConfig(name=MODEL, is_reasoning=is_reasoning)],
            ),
        ],
    )


def _make_task(*, logprobs: bool, is_reasoning: bool) -> Task:
    prompt = "What is 2+3? Answer with just the number."
    if not is_reasoning:
        prompt += " /no_think"
    extra: dict = {}
    if logprobs:
        extra["logprobs"] = True
        extra["top_logprobs"] = 3
    return Task(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=500, extra=extra)


@pytest.mark.parametrize(
    "logprobs,is_reasoning,force_stream",
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
    ids=[
        "nolp-noreason-nostream",
        "nolp-noreason-stream",
        "nolp-reason-nostream",
        "nolp-reason-stream",
        "lp-noreason-nostream",
        "lp-noreason-stream",
        "lp-reason-nostream",
        "lp-reason-stream",
    ],
)
async def test_logprobs_matrix(logprobs: bool, is_reasoning: bool, force_stream: bool):
    cfg = _make_config(force_stream=force_stream, is_reasoning=is_reasoning)
    task = _make_task(logprobs=logprobs, is_reasoning=is_reasoning)

    async with Client(config=cfg) as client:
        results = [r async for r in client.stream([task])]

    assert len(results) == 1
    result = results[0]
    assert result.ok, f"request failed: {result.error}"

    assert result.answer_content, "answer_content should be non-empty"

    if is_reasoning:
        assert result.reasoning_content, "reasoning_content should be non-empty for reasoning model"

    if not logprobs:
        assert result.logprobs is None
    else:
        assert result.logprobs is not None
        content_entries = result.logprobs.get("content", [])
        assert len(content_entries) > 0, "logprobs.content should have entries"
        for entry in content_entries:
            assert "token" in entry
            assert "logprob" in entry
