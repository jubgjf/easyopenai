"""Backend protocol and implementations."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from loguru import logger
from openai import AsyncOpenAI

from easyopenai.config import ModelConfig, ProviderConfig
from easyopenai.stream import aggregate_stream
from easyopenai.types import Task


@runtime_checkable
class Backend(Protocol):
    async def call(self, task: Task) -> dict: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


class OpenAIBackend:
    def __init__(self, cfg: ProviderConfig) -> None:
        assert cfg.base_url is not None, f"Provider '{cfg.name}': engine=openai requires base_url"
        assert cfg.api_key is not None, f"Provider '{cfg.name}': engine=openai requires api_key"
        self._client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self._cfg = cfg
        self._models: dict[str, ModelConfig] = {m.name: m for m in cfg.models}

    def _force_stream_for(self, model: str) -> bool:
        m = self._models[model]
        if m.force_stream is not None:
            return m.force_stream
        return self._cfg.force_stream

    async def call(self, task: Task) -> dict:
        kwargs: dict[str, Any] = {"model": task.model, "messages": task.messages}
        if task.temperature is not None:
            kwargs["temperature"] = task.temperature
        if task.max_tokens is not None:
            kwargs["max_tokens"] = task.max_tokens
        kwargs.update(task.extra)

        if self._force_stream_for(task.model):
            kwargs["stream"] = True
            kwargs.setdefault("stream_options", {"include_usage": True})
            stream = await self._client.chat.completions.create(**kwargs)
            return await aggregate_stream(stream)
        else:
            resp = await self._client.chat.completions.create(**kwargs)
            return resp.model_dump()

    async def ping(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()


class HmwranglerBackend:
    def __init__(self, cfg: ProviderConfig) -> None:
        assert cfg.sub_account_name is not None, f"Provider '{cfg.name}': engine=hmwrangler requires sub_account_name"
        self._cfg = cfg

    async def call(self, task: Task) -> dict:
        req_data: dict[str, Any] = {"model": task.model, "messages": task.messages}
        if task.temperature is not None:
            req_data["temperature"] = task.temperature
        if task.max_tokens is not None:
            req_data["max_tokens"] = task.max_tokens
        extra = {k: v for k, v in task.extra.items() if k != "timeout"}
        req_data.update(extra)
        timeout = task.extra.get("timeout", 300)

        cfg = self._cfg

        def _sync_call() -> dict:
            import hmwrangler_init  # noqa: F401
            from hmwrangler import hm_aigc

            return hm_aigc.aigc_managed(
                model_agent=cfg.name,
                req_data=req_data,
                sub_account_name=cfg.sub_account_name,
                model=task.model,
                timeout=timeout,
            )

        return await asyncio.to_thread(_sync_call)

    async def ping(self) -> bool:
        logger.info("[{}] hmwrangler ping (assumed OK)", self._cfg.name)
        return True

    async def close(self) -> None:
        pass


def make_backend(cfg: ProviderConfig) -> Backend:
    if cfg.engine == "openai":
        return OpenAIBackend(cfg)
    if cfg.engine == "hmwrangler":
        return HmwranglerBackend(cfg)
    assert False, f"Unknown backend engine: {cfg.engine!r}"
