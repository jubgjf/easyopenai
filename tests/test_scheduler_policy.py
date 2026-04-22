"""Dispatch policy tests: greedy (default) vs round_robin.

Uses respx to mock two identical providers and verifies how tasks are
distributed between them under each policy.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import httpx
import pytest
import respx

from easyopenai.client import Client
from easyopenai.config import (
    AppConfig,
    HealthConfig,
    LoggingConfig,
    ModelConfig,
    ProviderConfig,
    SchedulerConfig,
)
from easyopenai.types import Task


def _make_config(policy: str) -> AppConfig:
    return AppConfig(
        logging=LoggingConfig(stats_interval_s=3600),
        scheduler=SchedulerConfig(max_retries_per_task=3, dispatch_policy=policy),
        providers=[
            ProviderConfig(
                name="p1",
                base_url="https://p1.test/v1",
                api_key="sk-p1-0000000000000000",
                max_concurrency=1,
                max_rpm=600,
                force_stream=False,
                health=HealthConfig(window_size=20, failure_threshold=0.9, cooldown_s=60),
                models=[ModelConfig(name="m1", is_reasoning=False)],
            ),
            ProviderConfig(
                name="p2",
                base_url="https://p2.test/v1",
                api_key="sk-p2-0000000000000000",
                max_concurrency=1,
                max_rpm=600,
                force_stream=False,
                health=HealthConfig(window_size=20, failure_threshold=0.9, cooldown_s=60),
                models=[ModelConfig(name="m1", is_reasoning=False)],
            ),
        ],
    )


def _ok_response() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "m1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def _slow_ok(request):
    # Simulate real latency so that provider concurrency genuinely caps at 1.
    await asyncio.sleep(0.2)
    return httpx.Response(200, json=_ok_response())


@pytest.fixture
def mock_routes():
    with respx.mock(assert_all_called=False) as r:
        r.get("https://p1.test/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "m1"}], "object": "list"})
        )
        r.get("https://p2.test/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "m1"}], "object": "list"})
        )
        r.post("https://p1.test/v1/chat/completions").mock(side_effect=_slow_ok)
        r.post("https://p2.test/v1/chat/completions").mock(side_effect=_slow_ok)
        yield r


async def _run_batch(cfg: AppConfig, n: int) -> list:
    tasks = [Task(task_id=f"t{i}", messages=[{"role": "user", "content": "x"}], model="m1") for i in range(n)]
    results = []
    async with Client(config=cfg) as client:
        async for r in client.stream(tasks):
            results.append(r)
    return results


async def test_greedy_prefers_first_provider(mock_routes):
    """Under greedy, workers of the first provider tend to grab tasks first,
    so p1 should handle >= as many tasks as p2 (often strictly more)."""
    cfg = _make_config("greedy")
    results = await _run_batch(cfg, n=6)
    assert all(r.error is None for r in results)
    dist = Counter(r.provider for r in results)
    # Defining property: greedy leans toward the first provider.
    assert dist["p1"] >= dist["p2"]


async def test_round_robin_distributes_evenly(mock_routes):
    """Under round_robin, a 6-task batch across 2 providers should split 3/3."""
    cfg = _make_config("round_robin")
    results = await _run_batch(cfg, n=6)
    assert all(r.error is None for r in results)
    dist = Counter(r.provider for r in results)
    # Strict fairness: dispatcher rotates, both providers are healthy and
    # support the model, so distribution must be exactly even.
    assert dist["p1"] == 3
    assert dist["p2"] == 3


async def test_round_robin_default_is_greedy():
    """SchedulerConfig() without explicit dispatch_policy stays greedy
    (backwards compat)."""
    cfg = SchedulerConfig()
    assert cfg.dispatch_policy == "greedy"
