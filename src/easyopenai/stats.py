"""Periodic provider statistics logger."""
from __future__ import annotations

import asyncio

from loguru import logger

from easyopenai.provider import Provider


async def stats_printer(providers: list[Provider], interval_s: float) -> None:
    last_totals: dict[str, tuple[int, int, int]] = {p.name: (0, 0, 0) for p in providers}
    while True:
        await asyncio.sleep(interval_s)
        for p in providers:
            s = p.stats
            prev_total, prev_prompt, prev_comp = last_totals[p.name]
            d_req = s.requests_total - prev_total
            d_prompt = s.prompt_tokens - prev_prompt
            d_comp = s.completion_tokens - prev_comp
            last_totals[p.name] = (s.requests_total, s.prompt_tokens, s.completion_tokens)
            rpm = d_req / interval_s * 60
            tpm = (d_prompt + d_comp) / interval_s * 60
            success_rate = (
                s.requests_success / s.requests_total * 100 if s.requests_total else 0.0
            )
            logger.info(
                "[stats:{}] state={} inflight={} rpm={:.0f} tpm={:.0f} "
                "success={:.0f}% total={} fail={}",
                p.name,
                p.health.state.value,
                s.inflight,
                rpm,
                tpm,
                success_rate,
                s.requests_total,
                s.requests_failed,
            )
