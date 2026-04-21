"""Fault injection: use respx to mock httpx-level responses from fake providers
and verify the scheduler's failover, circuit breaker, and exhaustion paths.

No real API calls here — these tests must run offline.
"""
from __future__ import annotations

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


def _make_config(force_stream_for: set[str] | None = None) -> AppConfig:
    force_stream_for = force_stream_for or set()
    return AppConfig(
        logging=LoggingConfig(stats_interval_s=3600),
        scheduler=SchedulerConfig(max_retries_per_task=5),
        providers=[
            ProviderConfig(
                name="bad",
                base_url="https://bad.test/v1",
                api_key="sk-bad-0000000000000000",
                max_concurrency=2,
                max_rpm=60,
                force_stream="bad" in force_stream_for,
                health=HealthConfig(window_size=3, failure_threshold=0.5, cooldown_s=60),
                models=[ModelConfig(name="m1", is_reasoning=False)],
            ),
            ProviderConfig(
                name="good",
                base_url="https://good.test/v1",
                api_key="sk-good-000000000000000",
                max_concurrency=2,
                max_rpm=60,
                force_stream="good" in force_stream_for,
                health=HealthConfig(window_size=3, failure_threshold=0.5, cooldown_s=60),
                models=[ModelConfig(name="m1", is_reasoning=False)],
            ),
        ],
    )


def _ok_response(content: str = "hello") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "m1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }


@pytest.fixture
def mock_routes():
    with respx.mock(assert_all_called=False) as r:
        # Models (ping) — both up
        r.get("https://bad.test/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "m1"}], "object": "list"})
        )
        r.get("https://good.test/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "m1"}], "object": "list"})
        )
        yield r


async def test_failover_routes_to_healthy_provider(mock_routes):
    """Bad provider returns 500; task must succeed via good provider."""
    mock_routes.post("https://bad.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    mock_routes.post("https://good.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_response("ok from good"))
    )

    cfg = _make_config()
    results = []
    async with Client(config=cfg) as client:
        async for r in client.stream(
            [Task(task_id="t1", messages=[{"role": "user", "content": "hi"}], model="m1")]
        ):
            results.append(r)

    assert len(results) == 1
    r = results[0]
    assert r.error is None
    assert r.provider == "good"
    assert r.answer_content == "ok from good"


async def test_all_providers_fail_returns_error_result(mock_routes):
    """Every provider 500s → scheduler exhausts retries and emits error Result."""
    mock_routes.post("https://bad.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom1"})
    )
    mock_routes.post("https://good.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom2"})
    )

    cfg = _make_config()
    results = []
    async with Client(config=cfg) as client:
        async for r in client.stream(
            [Task(task_id="doomed", messages=[{"role": "user", "content": "x"}], model="m1")]
        ):
            results.append(r)

    assert len(results) == 1
    assert results[0].error is not None
    assert results[0].task_id == "doomed"


async def test_circuit_breaker_opens_and_skips_provider(mock_routes):
    """Enough consecutive failures on 'bad' must open its breaker; subsequent
    tasks route to 'good' without even touching 'bad'."""
    bad_route = mock_routes.post("https://bad.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    good_route = mock_routes.post("https://good.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_response("good"))
    )

    cfg = _make_config()
    # Shrink window so breaker opens quickly
    for p in cfg.providers:
        p.health = HealthConfig(window_size=2, failure_threshold=0.5, cooldown_s=60)

    results = []
    async with Client(config=cfg) as client:
        # Fire 6 tasks; after the first couple of failures on 'bad', its health
        # should flip to OPEN and later tasks skip it.
        tasks = [
            Task(task_id=f"t{i}", messages=[{"role": "user", "content": "x"}], model="m1")
            for i in range(6)
        ]
        async for r in client.stream(tasks):
            results.append(r)

        bad_provider = next(p for p in client.providers if p.name == "bad")
        assert bad_provider.stats.requests_failed >= 2
        # Breaker should be OPEN by end of run.
        assert bad_provider.health.state.value == "open"

    assert all(r.error is None for r in results), [r.error for r in results if r.error]
    assert all(r.provider == "good" for r in results)
    # 'bad' got hit a bounded number of times (window_size retries roughly);
    # 'good' handled everything successful.
    assert bad_route.call_count >= 2
    assert good_route.call_count == len(results)


async def test_tenacity_retries_transient_5xx_within_provider(mock_routes):
    """A single provider that fails twice then succeeds — tenacity inside Provider
    must absorb the transient failures without the scheduler re-routing."""
    side_effects = [
        httpx.Response(500, json={"error": "try1"}),
        httpx.Response(500, json={"error": "try2"}),
        httpx.Response(200, json=_ok_response("third time's the charm")),
    ]
    mock_routes.post("https://bad.test/v1/chat/completions").mock(side_effect=side_effects)
    mock_routes.post("https://good.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_response("unused"))
    )

    cfg = _make_config()
    # Only keep 'bad' to force the task to land there.
    cfg.providers = [cfg.providers[0]]

    results = []
    async with Client(config=cfg) as client:
        async for r in client.stream(
            [Task(task_id="retry", messages=[{"role": "user", "content": "hi"}], model="m1")]
        ):
            results.append(r)

    assert len(results) == 1
    assert results[0].error is None
    assert results[0].answer_content == "third time's the charm"
    assert results[0].provider == "bad"
