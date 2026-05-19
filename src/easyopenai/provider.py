"""Provider abstraction: wraps a Backend, applies rate limiting, retries,
health tracking, and reasoning parsing."""

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
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
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


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


class ProviderError(Exception):
    pass


_IGNORABLE_TRAILING_TOKENS = {
    "",
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "</s>",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return None


def _logprobs_requested(task: Task) -> bool:
    return task.extra.get("logprobs") is True


def _decode_logprob_entry(entry: dict[str, Any]) -> str:
    raw_bytes = entry.get("bytes")
    if raw_bytes is not None:
        try:
            return bytes(raw_bytes).decode("utf-8")
        except Exception as e:
            raise ValueError(f"invalid logprobs bytes for token {entry.get('token')!r}: {raw_bytes!r}") from e

    token = entry.get("token")
    if token is None:
        raise ValueError(f"logprobs content entry has neither bytes nor token: {entry!r}")
    return str(token)


def _is_ignorable_trailing_entry(entry: dict[str, Any], text: str) -> bool:
    token = str(entry.get("token") or "")
    return text in _IGNORABLE_TRAILING_TOKENS or token in _IGNORABLE_TRAILING_TOKENS


def _trim_trailing_ignorable_entries(
    entries: list[dict[str, Any]],
    texts: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    while entries and _is_ignorable_trailing_entry(entries[-1], texts[-1]):
        entries = entries[:-1]
        texts = texts[:-1]
    return entries, texts


def _logprobs_text_targets(message: Any, reasoning: str, answer: str) -> list[str]:
    targets: list[str] = []

    raw_content = _get(message, "content")
    if isinstance(raw_content, str) and raw_content:
        targets.append(raw_content)
    if answer:
        targets.append(answer)
    if reasoning:
        if answer:
            targets.extend(
                [
                    reasoning + answer,
                    f"{reasoning}\n{answer}",
                    f"{reasoning}\n\n{answer}",
                ]
            )
        targets.extend(
            [
                reasoning,
                f"<think>\n{reasoning}",
                f"<think>{reasoning}",
            ]
        )
        if answer:
            targets.extend(
                [
                    f"<think>\n{reasoning}\n</think>\n{answer}",
                    f"<think>\n{reasoning}\n</think>\n\n{answer}",
                    f"<think>{reasoning}</think>{answer}",
                ]
            )

    unique_targets: list[str] = []
    for target in targets:
        if target not in unique_targets:
            unique_targets.append(target)
    return unique_targets


def _reconstructed_text_variants(text: str) -> list[str]:
    variants = [text]

    if text.startswith("<think>"):
        after_open = text[len("<think>") :]
        variants.append(after_open.lstrip("\n"))

        if "</think>" in after_open:
            reasoning_part, answer_part = after_open.split("</think>", 1)
            reasoning = reasoning_part.strip("\n")
            answer = answer_part.lstrip("\n")
            variants.extend(
                [
                    reasoning,
                    answer,
                    reasoning + answer,
                    f"{reasoning}\n{answer}",
                    f"{reasoning}\n\n{answer}",
                ]
            )

    unique_variants: list[str] = []
    for variant in variants:
        if variant not in unique_variants:
            unique_variants.append(variant)
    return unique_variants


def _validate_logprobs_match_text(
    logprobs: dict[str, Any] | None,
    message: Any,
    reasoning: str,
    answer: str,
) -> None:
    if logprobs is None:
        raise ValueError("logprobs=True was requested but response choice.logprobs is missing")

    content = logprobs.get("content")
    if not isinstance(content, list):
        raise ValueError("logprobs=True was requested but response choice.logprobs.content is missing")
    targets = _logprobs_text_targets(message, reasoning, answer)
    if not content and targets:
        raise ValueError("logprobs.content is empty but response text is not empty")

    entries: list[dict[str, Any]] = []
    texts: list[str] = []

    for raw_entry in content:
        entry = _as_dict(raw_entry)
        if entry is None:
            raise ValueError(f"logprobs.content entry is not a mapping: {raw_entry!r}")
        entries.append(entry)
        texts.append(_decode_logprob_entry(entry))

    _entries, texts = _trim_trailing_ignorable_entries(entries, texts)
    reconstructed = "".join(texts)
    for variant in _reconstructed_text_variants(reconstructed):
        if variant in targets:
            return

    if not reconstructed and not targets:
        return

    target_preview = targets[0] if targets else ""
    raise ValueError(
        "logprobs.content does not reconstruct response text: "
        f"got {reconstructed!r}, expected one of {len(targets)} target(s), first target {target_preview!r}"
    )


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
        """Best-effort check: are we below max_concurrency?"""
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
                    retry=retry_if_exception(_is_retryable),
                    reraise=True,
                )
                async def _do() -> dict:
                    async with limiter:
                        return await self._backend.call(task)

                response = await _do()

                choice = response["choices"][0]
                reasoning, answer = parse_message(choice["message"], is_reasoning=model_info.is_reasoning)
                logprobs = _as_dict(_get(choice, "logprobs"))
                if _logprobs_requested(task):
                    _validate_logprobs_match_text(logprobs, choice["message"], reasoning, answer)
                usage_raw = response.get("usage") or {}
                usage = self._parse_usage(usage_raw)

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
                    logprobs=logprobs,
                    usage=usage,
                    latency_s=time.monotonic() - started,
                )
            except AssertionError:
                raise
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
    def _parse_usage(raw: dict) -> TokenUsage:
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
