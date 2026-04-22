"""User-facing Client — the library's single public entrypoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from loguru import logger

from easyopenai.config import AppConfig, load_config
from easyopenai.logging import setup_logging
from easyopenai.provider import Provider
from easyopenai.scheduler import Scheduler
from easyopenai.stats import stats_printer
from easyopenai.types import Result, Task


class Client:
    def __init__(
        self,
        config_path: str | Path | None = None,
        config: AppConfig | None = None,
        log_level: str = "INFO",
    ):
        setup_logging(level=log_level)
        assert (config_path is None) != (config is None), "Pass exactly one of config_path / config"
        self.cfg: AppConfig = config or load_config(config_path)  # type: ignore[arg-type]
        self.providers: list[Provider] = [Provider(pc) for pc in self.cfg.providers]
        self._stats_task: asyncio.Task | None = None

    async def __aenter__(self) -> Client:
        await self._ping_all()
        self._stats_task = asyncio.create_task(stats_printer(self.providers, self.cfg.logging.stats_interval_s))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except (asyncio.CancelledError, Exception):
                pass
        for p in self.providers:
            await p.client.close()

    async def _ping_all(self) -> None:
        results = await asyncio.gather(*(p.ping() for p in self.providers))
        alive = [p.name for p, ok in zip(self.providers, results) if ok]
        assert alive, "No provider is reachable at startup"
        logger.info("Startup health check: {}/{} providers up: {}", len(alive), len(self.providers), alive)

    async def stream(self, tasks: Iterable[Task | dict]) -> AsyncIterator[Result]:
        """Main API: submit tasks and asynchronously yield Results as they complete."""
        scheduler = Scheduler(self.providers, self.cfg.scheduler)
        async for result in scheduler.stream(tasks):
            yield result

    async def achat(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra,
    ) -> Result:
        task = Task(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )
        async for r in self.stream([task]):
            return r
        raise RuntimeError("stream exhausted without yielding a result")

    def chat(self, messages: list[dict], model: str, **kw) -> Result:
        """Synchronous convenience wrapper — spins up its own event loop."""

        async def _run() -> Result:
            async with self as c:
                return await c.achat(messages, model, **kw)

        return asyncio.run(_run())
