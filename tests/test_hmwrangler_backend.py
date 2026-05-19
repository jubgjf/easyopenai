"""Tests for hmwrangler backend integration.

Covers config validation, HmwranglerBackend unit tests (with mocked hmwrangler
module), Provider integration, scheduler failover, and reasoning parsing.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from easyopenai.config import (
    AppConfig,
    HealthConfig,
    LoggingConfig,
    ModelConfig,
    ProviderConfig,
    SchedulerConfig,
)
from easyopenai.provider import Provider, ProviderError
from easyopenai.types import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hmwrangler_provider_config(**overrides) -> ProviderConfig:
    defaults = dict(
        name="yibu",
        engine="hmwrangler",
        sub_account_name="acc0501",
        max_concurrency=2,
        max_rpm=60,
        health=HealthConfig(window_size=3, failure_threshold=0.5, cooldown_s=60),
        models=[ModelConfig(name="deepseek-v3.2", is_reasoning=False)],
    )
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _ok_response(content: str = "mock response", model: str = "deepseek-v3.2") -> dict:
    return {
        "id": "mock-123",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _logprobs_for_text(text: str, *, trailing_token: str | None = None) -> dict:
    entries = [
        {
            "token": token,
            "logprob": -0.1,
            "bytes": list(token.encode("utf-8")),
            "top_logprobs": [{"token": token, "logprob": -0.1, "bytes": list(token.encode("utf-8"))}],
        }
        for token in text
    ]
    if trailing_token is not None:
        entries.append(
            {
                "token": trailing_token,
                "logprob": -0.2,
                "bytes": list(trailing_token.encode("utf-8")),
                "top_logprobs": [
                    {"token": trailing_token, "logprob": -0.2, "bytes": list(trailing_token.encode("utf-8"))}
                ],
            }
        )
    return {"content": entries}


def _reasoning_response(reasoning: str = "let me think", answer: str = "42") -> dict:
    return {
        "id": "mock-reason",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-v3.2",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                    "reasoning_content": reasoning,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 15},
        },
    }


def _install_mock_hmwrangler(aigc_managed_side_effect=None, aigc_managed_return=None):
    """Install fake hmwrangler_init and hmwrangler.hm_aigc modules into sys.modules."""
    mock_init = types.ModuleType("hmwrangler_init")
    mock_hmwrangler = types.ModuleType("hmwrangler")
    mock_hm_aigc = types.ModuleType("hmwrangler.hm_aigc")

    mock_fn = MagicMock()
    if aigc_managed_side_effect is not None:
        mock_fn.side_effect = aigc_managed_side_effect
    elif aigc_managed_return is not None:
        mock_fn.return_value = aigc_managed_return
    else:
        mock_fn.return_value = _ok_response()

    mock_hm_aigc.aigc_managed = mock_fn
    mock_hmwrangler.hm_aigc = mock_hm_aigc

    sys.modules["hmwrangler_init"] = mock_init
    sys.modules["hmwrangler"] = mock_hmwrangler
    sys.modules["hmwrangler.hm_aigc"] = mock_hm_aigc

    return mock_fn


def _cleanup_mock_hmwrangler():
    for mod_name in ("hmwrangler_init", "hmwrangler", "hmwrangler.hm_aigc"):
        sys.modules.pop(mod_name, None)


@pytest.fixture(autouse=True)
def _clean_hmwrangler_modules():
    yield
    _cleanup_mock_hmwrangler()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_engine_default_is_openai(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://example.com/v1",
            api_key="sk-test0000000000000",
            models=[ModelConfig(name="m1")],
        )
        assert cfg.engine == "openai"

    def test_hmwrangler_requires_sub_account_name(self):
        with pytest.raises((AssertionError, Exception), match="sub_account_name"):
            ProviderConfig(
                name="test",
                engine="hmwrangler",
                models=[ModelConfig(name="m1")],
            )

    def test_hmwrangler_with_sub_account_name_ok(self):
        cfg = _hmwrangler_provider_config()
        assert cfg.engine == "hmwrangler"
        assert cfg.sub_account_name == "acc0501"

    def test_openai_requires_base_url(self):
        with pytest.raises((AssertionError, Exception), match="base_url"):
            ProviderConfig(
                name="test",
                engine="openai",
                api_key="sk-test0000000000000",
                models=[ModelConfig(name="m1")],
            )

    def test_openai_requires_api_key(self):
        with pytest.raises((AssertionError, Exception), match="api_key"):
            ProviderConfig(
                name="test",
                engine="openai",
                base_url="https://example.com/v1",
                models=[ModelConfig(name="m1")],
            )

    def test_hmwrangler_base_url_optional(self):
        cfg = _hmwrangler_provider_config()
        assert cfg.base_url is None

    def test_hmwrangler_api_key_optional(self):
        cfg = _hmwrangler_provider_config()
        assert cfg.api_key is None


# ---------------------------------------------------------------------------
# HmwranglerBackend unit tests
# ---------------------------------------------------------------------------


class TestHmwranglerBackend:
    async def test_call_returns_correct_response(self):
        expected = _ok_response("hello from hmwrangler")
        mock_fn = _install_mock_hmwrangler(aigc_managed_return=expected)

        from easyopenai.backends import HmwranglerBackend

        cfg = _hmwrangler_provider_config()
        backend = HmwranglerBackend(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        result = await backend.call(task)
        assert result == expected
        mock_fn.assert_called_once()

    async def test_call_passes_correct_arguments(self):
        mock_fn = _install_mock_hmwrangler()

        from easyopenai.backends import HmwranglerBackend

        cfg = _hmwrangler_provider_config()
        backend = HmwranglerBackend(cfg)
        task = Task(
            messages=[{"role": "user", "content": "test"}],
            model="deepseek-v3.2",
            temperature=0.7,
            max_tokens=100,
        )

        await backend.call(task)

        call_kwargs = mock_fn.call_args
        assert call_kwargs.kwargs["model_agent"] == "yibu"
        assert call_kwargs.kwargs["sub_account_name"] == "acc0501"
        assert call_kwargs.kwargs["model"] == "deepseek-v3.2"
        assert call_kwargs.kwargs["timeout"] == 300
        req_data = call_kwargs.kwargs["req_data"]
        assert req_data["model"] == "deepseek-v3.2"
        assert req_data["temperature"] == 0.7
        assert req_data["max_tokens"] == 100

    async def test_call_extracts_timeout_from_extra(self):
        mock_fn = _install_mock_hmwrangler()

        from easyopenai.backends import HmwranglerBackend

        cfg = _hmwrangler_provider_config()
        backend = HmwranglerBackend(cfg)
        task = Task(
            messages=[{"role": "user", "content": "test"}],
            model="deepseek-v3.2",
            extra={"timeout": 600},
        )

        await backend.call(task)

        call_kwargs = mock_fn.call_args
        assert call_kwargs.kwargs["timeout"] == 600
        assert "timeout" not in call_kwargs.kwargs["req_data"]

    async def test_call_propagates_exception(self):
        _install_mock_hmwrangler(aigc_managed_side_effect=RuntimeError("connection lost"))

        from easyopenai.backends import HmwranglerBackend

        cfg = _hmwrangler_provider_config()
        backend = HmwranglerBackend(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        with pytest.raises(RuntimeError, match="connection lost"):
            await backend.call(task)

    async def test_ping_returns_true(self):
        from easyopenai.backends import HmwranglerBackend

        cfg = _hmwrangler_provider_config()
        backend = HmwranglerBackend(cfg)
        assert await backend.ping() is True

    async def test_close_is_noop(self):
        from easyopenai.backends import HmwranglerBackend

        cfg = _hmwrangler_provider_config()
        backend = HmwranglerBackend(cfg)
        await backend.close()


# ---------------------------------------------------------------------------
# make_backend factory
# ---------------------------------------------------------------------------


class TestMakeBackend:
    def test_openai_engine(self):
        from easyopenai.backends import OpenAIBackend, make_backend

        cfg = ProviderConfig(
            name="test",
            engine="openai",
            base_url="https://example.com/v1",
            api_key="sk-test0000000000000",
            models=[ModelConfig(name="m1")],
        )
        backend = make_backend(cfg)
        assert isinstance(backend, OpenAIBackend)

    def test_hmwrangler_engine(self):
        from easyopenai.backends import HmwranglerBackend, make_backend

        cfg = _hmwrangler_provider_config()
        backend = make_backend(cfg)
        assert isinstance(backend, HmwranglerBackend)

    def test_unknown_engine_asserts(self):
        from easyopenai.backends import make_backend

        cfg = ProviderConfig.__new__(ProviderConfig)
        object.__setattr__(cfg, "engine", "unknown")
        object.__setattr__(cfg, "name", "test")
        with pytest.raises(AssertionError, match="Unknown backend engine"):
            make_backend(cfg)


# ---------------------------------------------------------------------------
# Provider integration (with patched backend)
# ---------------------------------------------------------------------------


class TestProviderWithHmwrangler:
    async def test_provider_call_success(self):
        _install_mock_hmwrangler(aigc_managed_return=_ok_response("works"))

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        result = await provider.call(task)
        assert result.ok
        assert result.answer_content == "works"
        assert result.logprobs is None
        assert result.provider == "yibu"
        assert provider.stats.requests_success == 1
        assert provider.stats.requests_total == 1

    async def test_provider_call_failure_raises_provider_error(self):
        _install_mock_hmwrangler(aigc_managed_side_effect=RuntimeError("fail"))

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        with pytest.raises(ProviderError):
            await provider.call(task)

        assert provider.stats.requests_failed == 1

    async def test_provider_health_recorded_on_success(self):
        _install_mock_hmwrangler(aigc_managed_return=_ok_response())

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        await provider.call(task)
        assert provider.health.state.value == "closed"

    async def test_provider_health_recorded_on_failure(self):
        _install_mock_hmwrangler(aigc_managed_side_effect=RuntimeError("fail"))

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        with pytest.raises(ProviderError):
            await provider.call(task)

    async def test_provider_usage_parsed(self):
        _install_mock_hmwrangler(aigc_managed_return=_ok_response())

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        result = await provider.call(task)
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 3
        assert result.usage.reasoning_tokens == 0

    async def test_logprobs_exposed_when_requested_and_matching(self):
        response = _ok_response("答案")
        response["choices"][0]["logprobs"] = _logprobs_for_text("答案")
        _install_mock_hmwrangler(aigc_managed_return=response)

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v3.2",
            extra={"logprobs": True, "top_logprobs": 3},
        )

        result = await provider.call(task)
        assert result.ok
        assert result.answer_content == "答案"
        assert result.logprobs == response["choices"][0]["logprobs"]

    async def test_logprobs_validation_allows_trailing_stop_token(self):
        response = _ok_response("ok")
        response["choices"][0]["logprobs"] = _logprobs_for_text("ok", trailing_token="<|im_end|>")
        _install_mock_hmwrangler(aigc_managed_return=response)

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v3.2",
            extra={"logprobs": True},
        )

        result = await provider.call(task)
        assert result.logprobs == response["choices"][0]["logprobs"]

    async def test_logprobs_validation_allows_empty_think_wrapper(self):
        response = _ok_response("ok")
        response["choices"][0]["logprobs"] = _logprobs_for_text("<think>\n\n</think>\n\nok")
        _install_mock_hmwrangler(aigc_managed_return=response)

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v3.2",
            extra={"logprobs": True},
        )

        result = await provider.call(task)
        assert result.answer_content == "ok"
        assert result.logprobs == response["choices"][0]["logprobs"]

    async def test_logprobs_validation_allows_open_think_prefix(self):
        response = _ok_response("ok")
        response["choices"][0]["logprobs"] = _logprobs_for_text("<think>ok")
        _install_mock_hmwrangler(aigc_managed_return=response)

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v3.2",
            extra={"logprobs": True},
        )

        result = await provider.call(task)
        assert result.answer_content == "ok"
        assert result.logprobs == response["choices"][0]["logprobs"]

    async def test_logprobs_requested_missing_raises_provider_error(self):
        _install_mock_hmwrangler(aigc_managed_return=_ok_response("ok"))

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v3.2",
            extra={"logprobs": True},
        )

        with pytest.raises(ProviderError, match="logprobs=True"):
            await provider.call(task)

    async def test_logprobs_mismatch_raises_provider_error(self):
        response = _ok_response("ok")
        response["choices"][0]["logprobs"] = _logprobs_for_text("no")
        _install_mock_hmwrangler(aigc_managed_return=response)

        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        task = Task(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v3.2",
            extra={"logprobs": True},
        )

        with pytest.raises(ProviderError, match="does not reconstruct"):
            await provider.call(task)

    async def test_reasoning_logprobs_can_match_thinking_text(self):
        response = _reasoning_response("thinking...", "")
        response["choices"][0]["logprobs"] = _logprobs_for_text("<think>\nthinking...")
        _install_mock_hmwrangler(aigc_managed_return=response)

        cfg = _hmwrangler_provider_config(
            models=[ModelConfig(name="deepseek-v3.2", is_reasoning=True)],
        )
        provider = Provider(cfg)
        task = Task(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v3.2",
            extra={"logprobs": True},
        )

        result = await provider.call(task)
        assert result.reasoning_content == "thinking..."
        assert result.answer_content == ""
        assert result.logprobs == response["choices"][0]["logprobs"]

    async def test_reasoning_logprobs_can_match_think_wrapped_answer(self):
        response = _reasoning_response("thinking...", "answer")
        response["choices"][0]["logprobs"] = _logprobs_for_text("<think>\nthinking...\n</think>\n\nanswer")
        _install_mock_hmwrangler(aigc_managed_return=response)

        cfg = _hmwrangler_provider_config(
            models=[ModelConfig(name="deepseek-v3.2", is_reasoning=True)],
        )
        provider = Provider(cfg)
        task = Task(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v3.2",
            extra={"logprobs": True},
        )

        result = await provider.call(task)
        assert result.reasoning_content == "thinking..."
        assert result.answer_content == "answer"
        assert result.logprobs == response["choices"][0]["logprobs"]

    async def test_provider_ping(self):
        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        assert await provider.ping() is True

    async def test_provider_close(self):
        cfg = _hmwrangler_provider_config()
        provider = Provider(cfg)
        await provider.close()


# ---------------------------------------------------------------------------
# Reasoning parsing with hmwrangler response
# ---------------------------------------------------------------------------


class TestReasoningWithHmwrangler:
    async def test_reasoning_content_extracted(self):
        _install_mock_hmwrangler(aigc_managed_return=_reasoning_response("thinking...", "answer"))

        cfg = _hmwrangler_provider_config(
            models=[ModelConfig(name="deepseek-v3.2", is_reasoning=True)],
        )
        provider = Provider(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        result = await provider.call(task)
        assert result.reasoning_content == "thinking..."
        assert result.answer_content == "answer"
        assert result.usage.reasoning_tokens == 15

    async def test_think_tag_reasoning_extracted(self):
        response = _ok_response()
        response["choices"][0]["message"]["content"] = "<think>step by step</think>final answer"
        _install_mock_hmwrangler(aigc_managed_return=response)

        cfg = _hmwrangler_provider_config(
            models=[ModelConfig(name="deepseek-v3.2", is_reasoning=True)],
        )
        provider = Provider(cfg)
        task = Task(messages=[{"role": "user", "content": "hi"}], model="deepseek-v3.2")

        result = await provider.call(task)
        assert result.reasoning_content == "step by step"
        assert result.answer_content == "final answer"


# ---------------------------------------------------------------------------
# Scheduler-level failover with hmwrangler providers
# ---------------------------------------------------------------------------


class TestSchedulerFailoverWithHmwrangler:
    async def test_failover_to_second_provider(self):
        call_count = 0

        def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("model_agent") == "bad_hm":
                raise RuntimeError("provider down")
            return _ok_response("from good_hm")

        _install_mock_hmwrangler(aigc_managed_side_effect=_side_effect)

        from easyopenai.client import Client

        cfg = AppConfig(
            logging=LoggingConfig(stats_interval_s=3600),
            scheduler=SchedulerConfig(max_retries_per_task=5),
            providers=[
                ProviderConfig(
                    name="bad_hm",
                    engine="hmwrangler",
                    sub_account_name="acc_bad",
                    max_concurrency=2,
                    max_rpm=60,
                    health=HealthConfig(window_size=3, failure_threshold=0.5, cooldown_s=60),
                    models=[ModelConfig(name="m1", is_reasoning=False)],
                ),
                ProviderConfig(
                    name="good_hm",
                    engine="hmwrangler",
                    sub_account_name="acc_good",
                    max_concurrency=2,
                    max_rpm=60,
                    health=HealthConfig(window_size=3, failure_threshold=0.5, cooldown_s=60),
                    models=[ModelConfig(name="m1", is_reasoning=False)],
                ),
            ],
        )

        results = []
        async with Client(config=cfg) as client:
            async for r in client.stream(
                [Task(task_id="t1", messages=[{"role": "user", "content": "hi"}], model="m1")]
            ):
                results.append(r)

        assert len(results) == 1
        assert results[0].ok
        assert results[0].provider == "good_hm"
        assert results[0].answer_content == "from good_hm"

    async def test_all_providers_fail_returns_error(self):
        _install_mock_hmwrangler(aigc_managed_side_effect=RuntimeError("all down"))

        from easyopenai.client import Client

        cfg = AppConfig(
            logging=LoggingConfig(stats_interval_s=3600),
            scheduler=SchedulerConfig(max_retries_per_task=3),
            providers=[
                ProviderConfig(
                    name="hm1",
                    engine="hmwrangler",
                    sub_account_name="acc1",
                    max_concurrency=2,
                    max_rpm=60,
                    health=HealthConfig(window_size=3, failure_threshold=0.5, cooldown_s=60),
                    models=[ModelConfig(name="m1", is_reasoning=False)],
                ),
                ProviderConfig(
                    name="hm2",
                    engine="hmwrangler",
                    sub_account_name="acc2",
                    max_concurrency=2,
                    max_rpm=60,
                    health=HealthConfig(window_size=3, failure_threshold=0.5, cooldown_s=60),
                    models=[ModelConfig(name="m1", is_reasoning=False)],
                ),
            ],
        )

        results = []
        async with Client(config=cfg) as client:
            async for r in client.stream(
                [Task(task_id="doomed", messages=[{"role": "user", "content": "x"}], model="m1")]
            ):
                results.append(r)

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].task_id == "doomed"


# ---------------------------------------------------------------------------
# Mixed engine: openai + hmwrangler
# ---------------------------------------------------------------------------


class TestMixedEngines:
    async def test_hmwrangler_and_openai_coexist_in_config(self):
        cfg = AppConfig(
            logging=LoggingConfig(stats_interval_s=3600),
            scheduler=SchedulerConfig(),
            providers=[
                ProviderConfig(
                    name="oai",
                    engine="openai",
                    base_url="https://api.example.com/v1",
                    api_key="sk-test0000000000000",
                    models=[ModelConfig(name="gpt-4")],
                ),
                ProviderConfig(
                    name="hm",
                    engine="hmwrangler",
                    sub_account_name="acc01",
                    models=[ModelConfig(name="deepseek-v3.2")],
                ),
            ],
        )
        assert cfg.providers[0].engine == "openai"
        assert cfg.providers[1].engine == "hmwrangler"
