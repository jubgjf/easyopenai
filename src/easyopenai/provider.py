"""Provider abstraction: wraps AsyncOpenAI, applies rate limiting, retries,
health tracking, stream aggregation, and reasoning parsing."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from aiolimiter import AsyncLimiter
from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from easyopenai.config import ProviderConfig
from easyopenai.health import HealthMonitor
from easyopenai.parser import parse_message
from easyopenai.stream import aggregate_stream
from easyopenai.types import ProviderStats, Result, Task, TokenUsage

_RETRYABLE = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)


def _is_retryable_status(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


class ProviderError(Exception):
    pass


class Provider:
    def __init__(self, cfg: ProviderConfig):
        assert cfg.max_concurrency > 0
        assert cfg.max_rpm > 0
        self.cfg = cfg
        self.name = cfg.name
        self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self.limiter = AsyncLimiter(cfg.max_rpm, 60)
        self.semaphore = asyncio.Semaphore(cfg.max_concurrency)
        self.health = HealthMonitor(cfg.health)
        self.stats = ProviderStats()
        self._models = {m.name: m for m in cfg.models}

    def supports(self, model: str) -> bool:
        return model in self._models

    def model_is_reasoning(self, model: str) -> bool:
        return self._models[model].is_reasoning

    def has_capacity(self) -> bool:
        """Best-effort check: are we below max_concurrency?"""
        return self.semaphore._value > 0  # type: ignore[attr-defined]

    async def ping(self) -> bool:
        try:
            await self.client.models.list()
            logger.info("[{}] ping OK", self.name)
            return True
        except Exception as e:
            logger.warning("[{}] ping FAILED: {}", self.name, e)
            return False

    async def call(self, task: Task) -> Result:
        assert self.supports(task.model), f"Provider {self.name} does not serve {task.model}"
        model_info = self._models[task.model]
        started = time.monotonic()

        async with self.semaphore:
            self.stats.inflight += 1
            self.stats.requests_total += 1
            try:

                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_random_exponential(min=1, max=10),
                    retry=retry_if_exception_type(_RETRYABLE)
                    | retry_if_exception_type(APIStatusError),
                    reraise=True,
                )
                async def _do() -> dict:
                    async with self.limiter:
                        return await self._single_call(task)

                try:
                    response = await _do()
                except APIStatusError as e:
                    # Only 5xx should have been retried; re-raise 4xx without wrapping
                    if not _is_retryable_status(e):
                        raise
                    raise

                choice = response["choices"][0]
                reasoning, answer = parse_message(
                    choice["message"], is_reasoning=model_info.is_reasoning
                )
                usage_raw = response.get("usage") or {}
                usage = self._parse_usage(usage_raw, reasoning_text=reasoning, answer_text=answer)

                self.stats.requests_success += 1
                self.stats.prompt_tokens += usage.prompt_tokens
                self.stats.completion_tokens += usage.completion_tokens
                self.health.record(True)

                return Result(
                    task_id=task.task_id,
                    provider=self.name,
                    model=task.model,
                    reasoning_content=reasoning,
                    answer_content=answer,
                    usage=usage,
                    latency_s=time.monotonic() - started,
                )
            except Exception as e:
                self.stats.requests_failed += 1
                self.health.record(False)
                logger.warning("[{}] call failed for task={}: {}", self.name, task.task_id, e)
                raise ProviderError(str(e)) from e
            finally:
                self.stats.inflight -= 1

    async def _single_call(self, task: Task) -> dict:
        kwargs: dict[str, Any] = {
            "model": task.model,
            "messages": task.messages,
        }
        if task.temperature is not None:
            kwargs["temperature"] = task.temperature
        if task.max_tokens is not None:
            kwargs["max_tokens"] = task.max_tokens
        kwargs.update(task.extra)

        if self.cfg.force_stream:
            kwargs["stream"] = True
            # Ask for usage in final chunk where supported.
            kwargs.setdefault("stream_options", {"include_usage": True})
            stream = await self.client.chat.completions.create(**kwargs)
            return await aggregate_stream(stream)
        else:
            resp = await self.client.chat.completions.create(**kwargs)
            return resp.model_dump()

    @staticmethod
    def _parse_usage(raw: dict, reasoning_text: str, answer_text: str) -> TokenUsage:
        prompt = int(raw.get("prompt_tokens", 0) or 0)
        completion = int(raw.get("completion_tokens", 0) or 0)
        details = raw.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
        answer_tokens = completion - reasoning_tokens if completion else 0
        if answer_tokens < 0:
            answer_tokens = 0
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
        )
