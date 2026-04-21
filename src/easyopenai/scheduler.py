"""Task scheduler: dispatches tasks from an input queue across providers,
with failover and retry-on-failure routing to untried providers."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from loguru import logger

from easyopenai.config import SchedulerConfig
from easyopenai.provider import Provider, ProviderError
from easyopenai.types import Result, Task, TokenUsage


_SENTINEL = object()


class Scheduler:
    def __init__(self, providers: list[Provider], cfg: SchedulerConfig):
        assert len(providers) > 0
        self.providers = providers
        self.cfg = cfg
        self._task_q: asyncio.Queue = asyncio.Queue()
        self._result_q: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._outstanding: int = 0
        self._lock = asyncio.Lock()
        self._all_submitted = asyncio.Event()

    async def submit_many(self, tasks: Iterable[Task | dict]) -> int:
        n = 0
        for t in tasks:
            if isinstance(t, dict):
                t = Task(**t)
            async with self._lock:
                self._outstanding += 1
            await self._task_q.put(t)
            n += 1
        self._all_submitted.set()
        return n

    async def stream(self, tasks: Iterable[Task | dict]) -> AsyncIterator[Result]:
        total = await self.submit_many(tasks)
        self._start_workers()
        yielded = 0
        while yielded < total:
            result = await self._result_q.get()
            yielded += 1
            yield result
        await self.shutdown()

    def _start_workers(self) -> None:
        if self._workers:
            return
        for p in self.providers:
            for _ in range(p.cfg.max_concurrency):
                self._workers.append(asyncio.create_task(self._worker_loop(p)))

    async def _worker_loop(self, provider: Provider) -> None:
        while True:
            try:
                task = await self._task_q.get()
            except asyncio.CancelledError:
                return
            if task is _SENTINEL:
                return
            try:
                # Check if this provider is a valid candidate for this task.
                if (
                    not provider.supports(task.model)
                    or provider.name in task.attempted_providers
                    or not provider.health.can_serve()
                ):
                    # Not our task — put it back and yield.
                    await self._task_q.put(task)
                    await asyncio.sleep(0.05)
                    continue

                task.attempted_providers.add(provider.name)
                try:
                    result = await provider.call(task)
                    await self._result_q.put(result)
                    async with self._lock:
                        self._outstanding -= 1
                except ProviderError as e:
                    task.retry_count += 1
                    untried = [
                        p
                        for p in self.providers
                        if p.supports(task.model) and p.name not in task.attempted_providers
                    ]
                    if untried and task.retry_count <= self.cfg.max_retries_per_task:
                        logger.info(
                            "Rerouting task {} (attempt {}) after [{}] failure: {}",
                            task.task_id,
                            task.retry_count,
                            provider.name,
                            e,
                        )
                        await self._task_q.put(task)
                    else:
                        logger.error(
                            "Task {} exhausted all providers. Last error: {}",
                            task.task_id,
                            e,
                        )
                        await self._result_q.put(
                            Result(
                                task_id=task.task_id,
                                provider=provider.name,
                                model=task.model,
                                usage=TokenUsage(),
                                error=str(e),
                            )
                        )
                        async with self._lock:
                            self._outstanding -= 1
            finally:
                self._task_q.task_done()

    async def shutdown(self) -> None:
        for _ in self._workers:
            await self._task_q.put(_SENTINEL)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
