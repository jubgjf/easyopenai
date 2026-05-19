from easyopenai.stream import aggregate_stream


async def _async_iter(items):
    for it in items:
        yield it


def _logprob_entry(token: str) -> dict:
    return {
        "token": token,
        "logprob": -0.1,
        "bytes": list(token.encode("utf-8")),
        "top_logprobs": [{"token": token, "logprob": -0.1, "bytes": list(token.encode("utf-8"))}],
    }


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


async def test_aggregate_delta_logprobs_for_answer_only():
    chunks = [
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "thinking", "logprobs": {"content": []}},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"content": "ok", "logprobs": {"content": [_logprob_entry("o"), _logprob_entry("k")]}},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
    ]
    out = await aggregate_stream(_async_iter(chunks))
    choice = out["choices"][0]
    assert choice["message"]["reasoning_content"] == "thinking"
    assert choice["message"]["content"] == "ok"
    assert [entry["token"] for entry in choice["logprobs"]["content"]] == ["o", "k"]


async def test_aggregate_choice_logprobs():
    chunks = [
        {
            "choices": [
                {
                    "delta": {"content": "a"},
                    "logprobs": {"content": [_logprob_entry("a")]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"content": "b"},
                    "logprobs": {"content": [_logprob_entry("b")]},
                    "finish_reason": "stop",
                }
            ]
        },
    ]
    out = await aggregate_stream(_async_iter(chunks))
    assert out["choices"][0]["message"]["content"] == "ab"
    assert [entry["token"] for entry in out["choices"][0]["logprobs"]["content"]] == ["a", "b"]
