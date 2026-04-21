"""Aggregate an async stream of chat completion chunks into a unified response dict.

Works with ``openai.AsyncStream[ChatCompletionChunk]`` and any async iterable
that yields objects/dicts with the same structure.
"""
from __future__ import annotations

import io
from typing import Any, AsyncIterator


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def aggregate_stream(stream: AsyncIterator[Any]) -> dict:
    """Consume an async stream of SSE chunks and return a dict resembling a
    non-streaming ``ChatCompletion`` response."""
    content_buf = io.StringIO()
    reasoning_buf = io.StringIO()
    finish_reason = None
    model = ""
    response_id = ""
    usage: dict | None = None

    async for chunk in stream:
        response_id = _get(chunk, "id", response_id)
        model = _get(chunk, "model", model)

        choices = _get(chunk, "choices") or []
        for choice in choices:
            delta = _get(choice, "delta")
            if delta is None:
                continue
            c = _get(delta, "content")
            if c:
                content_buf.write(c)
            rc = _get(delta, "reasoning_content")
            if rc:
                reasoning_buf.write(rc)
            fr = _get(choice, "finish_reason")
            if fr:
                finish_reason = fr

        chunk_usage = _get(chunk, "usage")
        if chunk_usage and chunk_usage != {}:
            usage = (
                chunk_usage
                if isinstance(chunk_usage, dict)
                else chunk_usage.model_dump() if hasattr(chunk_usage, "model_dump") else dict(chunk_usage)
            )

    reasoning_text = reasoning_buf.getvalue()
    content_text = content_buf.getvalue()

    result: dict = {
        "id": response_id,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_text,
                    "reasoning_content": reasoning_text or None,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {},
    }
    return result
