"""End-to-end smoke test against the two configured providers.

Runs only when both BLTCY_API_KEY and YUNWU_API_KEY env vars are present.
Sends 4 small tasks (one per provider × model combo) — keep this minimal,
real API calls cost money.
"""

import os
import textwrap
from pathlib import Path

import pytest

from easyopenai import Client, Task

REQUIRED_ENV = ["BLTCY_API_KEY", "YUNWU_API_KEY"]
pytestmark = pytest.mark.skipif(
    not all(os.environ.get(v) for v in REQUIRED_ENV),
    reason="Real-API smoke test — set BLTCY_API_KEY and YUNWU_API_KEY to enable",
)


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        textwrap.dedent(
            """
            logging:
              stats_interval_s: 5
            scheduler:
              max_retries_per_task: 2
            providers:
              - name: bltcy
                base_url: https://api.bltcy.ai/v1
                api_key: ${BLTCY_API_KEY}
                max_concurrency: 2
                max_rpm: 30
                force_stream: true
                models:
                  - name: qwen3-8b
                    is_reasoning: true
              - name: yunwu
                base_url: https://yunwu.ai/v1
                api_key: ${YUNWU_API_KEY}
                max_concurrency: 2
                max_rpm: 30
                force_stream: false
                models:
                  - name: deepseek-r1
                    is_reasoning: true
            """
        ),
        encoding="utf-8",
    )
    return p


async def test_smoke_e2e_one_per_provider(cfg_path: Path):
    tasks = [
        Task(
            task_id="bltcy-qwen",
            messages=[{"role": "user", "content": "用一个字回答：1+1="}],
            model="qwen3-8b",
            max_tokens=200,
        ),
        Task(
            task_id="yunwu-r1",
            messages=[{"role": "user", "content": "用一个字回答：2+2="}],
            model="deepseek-r1",
            max_tokens=200,
        ),
    ]
    results: list = []
    async with Client(config_path=cfg_path, log_level="INFO") as client:
        async for r in client.stream(tasks):
            results.append(r)

    assert len(results) == 2
    by_id = {r.task_id: r for r in results}

    for tid in ("bltcy-qwen", "yunwu-r1"):
        r = by_id[tid]
        assert r.error is None, f"{tid} failed: {r.error}"
        assert r.answer_content, f"{tid} empty answer"
        assert r.reasoning_content, f"{tid} reasoning model returned no thinking"
        assert r.usage.prompt_tokens > 0
        assert r.usage.completion_tokens > 0
        assert r.usage.reasoning_tokens > 0
        assert r.usage.answer_tokens >= 0
        assert r.latency_s > 0
        print(
            f"\n[{tid}] provider={r.provider} answer={r.answer_content!r} "
            f"reasoning_len={len(r.reasoning_content)} usage={r.usage}"
        )
