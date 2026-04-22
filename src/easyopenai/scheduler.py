"""Task scheduler: dispatches tasks from an input queue across providers,
with failover and retry-on-failure routing to untried providers.

Two dispatch policies (configurable via `SchedulerConfig.dispatch_policy`):

- `greedy` (default): one shared queue, all workers race to pull. Lowest
  latency / highest throughput, but the first provider in the list tends
  to absorb most traffic until it saturates.

- `round_robin`: each provider has its own private queue. A dedicated
  dispatcher coroutine pulls from the shared queue and rotates tasks
  across providers, skipping ones that don't support the model, have
  already attempted the task, or are circuit-broken. Smooths load so
  every provider gets roughly equal traffic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from loguru import logger

from easyopenai.config import SchedulerConfig
from easyopenai.provider import Provider, ProviderError
from easyopenai.types import Result, Task, TokenUsage


class Scheduler:
    def __init__(self, providers: list[Provider], cfg: SchedulerConfig):
        assert len(providers) > 0
        self.providers = providers
        self.cfg = cfg
        # Shared queue: always used for initial submission and for failure
        # re-routing. In greedy mode, workers read from it directly.
        # In round_robin mode, the dispatcher reads from it and fans out
        # to per-provider queues.
        self._task_q: asyncio.Queue = asyncio.Queue()
        self._result_q: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._dispatcher: asyncio.Task | None = None
        self._provider_qs: dict[str, asyncio.Queue] = {}
        self._rr_cursor: int = 0
        self._lock = asyncio.Lock()
        self._outstanding: int = 0
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
        try:
            while yielded < total:
                result = await self._result_q.get()
                yielded += 1
                yield result
        finally:
            await self.shutdown()

    def _start_workers(self) -> None:
        if self._workers:
            return
        if self.cfg.dispatch_policy == "round_robin":
            for p in self.providers:
                self._provider_qs[p.name] = asyncio.Queue()
            self._dispatcher = asyncio.create_task(self._dispatcher_loop())
            for p in self.providers:
                src_q = self._provider_qs[p.name]
                for _ in range(p.cfg.max_concurrency):
                    self._workers.append(asyncio.create_task(self._worker_loop(p, src_q)))
        else:
            for p in self.providers:
                for _ in range(p.cfg.max_concurrency):
                    self._workers.append(asyncio.create_task(self._worker_loop(p, self._task_q)))

    async def _dispatcher_loop(self) -> None:
        """round_robin only: pull from shared _task_q and assign each task
        to the next eligible provider in rotation."""
        n = len(self.providers)
        while True:
            try:
                task = await self._task_q.get()
            except asyncio.CancelledError:
                return
            try:
                chosen: Provider | None = None
                # Try up to n providers starting from current cursor.
                for offset in range(n):
                    idx = (self._rr_cursor + offset) % n
                    p = self.providers[idx]
                    if p.supports(task.model) and p.name not in task.attempted_providers and p.health.can_serve():
                        chosen = p
                        self._rr_cursor = (idx + 1) % n
                        break
                if chosen is None:
                    # No eligible provider right now — re-queue and yield.
                    # This covers transient cases (all circuit-broken briefly,
                    # or queued task for a model only served by a currently
                    # unhealthy provider). Permanent exhaustion is detected
                    # by workers via ProviderError -> "all attempted".
                    await self._task_q.put(task)
                    await asyncio.sleep(0.05)
                else:
                    await self._provider_qs[chosen.name].put(task)
            finally:
                self._task_q.task_done()

    async def _worker_loop(self, provider: Provider, src_q: asyncio.Queue) -> None:
        while True:
            try:
                task = await src_q.get()
            except asyncio.CancelledError:
                return
            try:
                # Check if this provider is a valid candidate for this task.
                # In greedy mode, tasks come from the shared queue and any
                # worker may pick them up. In round_robin mode, the dispatcher
                # has already filtered, but the task may have sat in our queue
                # long enough for our health to change — so we re-check.
                if (
                    not provider.supports(task.model)
                    or provider.name in task.attempted_providers
                    or not provider.health.can_serve()
                ):
                    # Not our task — put it back on the shared queue so the
                    # dispatcher (round_robin) or another worker (greedy)
                    # can route it elsewhere.
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
                        p for p in self.providers if p.supports(task.model) and p.name not in task.attempted_providers
                    ]
                    if untried and task.retry_count <= self.cfg.max_retries_per_task:
                        logger.info(
                            "Rerouting task {} (attempt {}) after [{}] failure: {}",
                            task.task_id,
                            task.retry_count,
                            provider.name,
                            e,
                        )
                        # Re-route via the shared queue so the dispatcher
                        # (or a different worker in greedy mode) assigns it.
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
                src_q.task_done()

    async def shutdown(self) -> None:
        if self._dispatcher is not None:
            self._dispatcher.cancel()
        for w in self._workers:
            w.cancel()
        pending = list(self._workers)
        if self._dispatcher is not None:
            pending.append(self._dispatcher)
        await asyncio.gather(*pending, return_exceptions=True)
        self._workers.clear()
        self._dispatcher = None
