import pytest

from easyopenai.stream import aggregate_stream


async def _async_iter(items):
    for it in items:
        yield it


async def test_aggregate_basic_content():
    chunks = [
        {"id": "x", "model": "m", "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"id": "x", "model": "m", "choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"id": "x", "model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"id": "x", "model": "m", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]
    out = await aggregate_stream(_async_iter(chunks))
    assert out["choices"][0]["message"]["content"] == "Hello world"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] == 5


async def test_aggregate_reasoning_split():
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]},
        {"choices": [{"delta": {"reasoning_content": "ing"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "ans"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "wer"}, "finish_reason": "stop"}]},
    ]
    out = await aggregate_stream(_async_iter(chunks))
    msg = out["choices"][0]["message"]
    assert msg["reasoning_content"] == "thinking"
    assert msg["content"] == "answer"
