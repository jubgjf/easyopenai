"""Provider abstraction: wraps a Backend, applies rate limiting, retries,
health tracking, and reasoning parsing."""

from __future__ import annotations

import asyncio
import time

from aiolimiter import AsyncLimiter
from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from easyopenai.backends import make_backend
from easyopenai.config import ProviderConfig
from easyopenai.health import HealthMonitor
from easyopenai.parser import parse_message
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
        self._backend = make_backend(cfg)
        self._default_limiter = AsyncLimiter(cfg.max_rpm, 60)
        self._default_semaphore = asyncio.Semaphore(cfg.max_concurrency)
        self.health = HealthMonitor(cfg.health)
        self.stats = ProviderStats()
        self._models = {m.name: m for m in cfg.models}
        self._model_semaphores: dict[str, asyncio.Semaphore] = {}
        self._model_limiters: dict[str, AsyncLimiter] = {}
        for m in cfg.models:
            if m.max_concurrency is not None:
                self._model_semaphores[m.name] = asyncio.Semaphore(m.max_concurrency)
            if m.max_rpm is not None:
                self._model_limiters[m.name] = AsyncLimiter(m.max_rpm, 60)

    def _semaphore_for(self, model: str) -> asyncio.Semaphore:
        return self._model_semaphores.get(model, self._default_semaphore)

    def _limiter_for(self, model: str) -> AsyncLimiter:
        return self._model_limiters.get(model, self._default_limiter)

    def supports(self, model: str) -> bool:
        return model in self._models

    def model_is_reasoning(self, model: str) -> bool:
        return self._models[model].is_reasoning

    def has_capacity(self, model: str | None = None) -> bool:
        sem = self._semaphore_for(model) if model else self._default_semaphore
        return sem._value > 0  # type: ignore[attr-defined]

    async def ping(self) -> bool:
        try:
            ok = await self._backend.ping()
            if ok:
                logger.info("[{}] ping OK", self.name)
            else:
                logger.warning("[{}] ping FAILED", self.name)
            return ok
        except Exception as e:
            logger.warning("[{}] ping FAILED: {}", self.name, e)
            return False

    async def call(self, task: Task) -> Result:
        assert self.supports(task.model), f"Provider {self.name} does not serve {task.model}"
        model_info = self._models[task.model]
        semaphore = self._semaphore_for(task.model)
        limiter = self._limiter_for(task.model)
        started = time.monotonic()

        async with semaphore:
            self.stats.inflight += 1
            self.stats.requests_total += 1
            try:

                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_random_exponential(min=1, max=10),
                    retry=retry_if_exception_type(_RETRYABLE) | retry_if_exception_type(APIStatusError),
                    reraise=True,
                )
                async def _do() -> dict:
                    async with limiter:
                        return await self._backend.call(task)

                try:
                    response = await _do()
                except APIStatusError as e:
                    if not _is_retryable_status(e):
                        raise
                    raise

                choice = response["choices"][0]
                reasoning, answer = parse_message(choice["message"], is_reasoning=model_info.is_reasoning)
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

    async def close(self) -> None:
        await self._backend.close()

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
