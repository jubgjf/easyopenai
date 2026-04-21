"""Reasoning-model content parser.

Handles three wire-format variants we've observed:

1. `message.reasoning_content` + `message.content`  (DeepSeek / Qwen3 / SiliconFlow)
2. `message.reasoning` + `message.content`          (OpenRouter and some others)
3. `<think>...</think>` inlined in `message.content` (some local backends)

For reasoning models where none of these yield anything, we fail fast with
an assertion — per user requirement in README.md.
"""
from __future__ import annotations

import re
from typing import Any

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def parse_message(message: Any, is_reasoning: bool) -> tuple[str, str]:
    """Return ``(reasoning_content, answer_content)``.

    ``message`` may be a pydantic model from openai SDK or a plain dict.
    """
    reasoning = _get(message, "reasoning_content")
    if not reasoning:
        reasoning = _get(message, "reasoning")
    content = _get(message, "content") or ""

    if reasoning:
        return str(reasoning), str(content)

    # Fallback: extract <think>...</think> from content.
    m = _THINK_RE.search(content)
    if m:
        reasoning = m.group(1)
        answer = _THINK_RE.sub("", content).strip()
        return reasoning, answer

    if is_reasoning:
        assert False, (
            "Declared reasoning model but no reasoning content found. "
            "Checked message.reasoning_content, message.reasoning, and <think> tags. "
            f"message snapshot: {message!r}"
        )
    return "", content
