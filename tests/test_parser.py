import pytest

from easyopenai.parser import parse_message


def test_reasoning_content_field():
    msg = {"reasoning_content": "thought", "content": "answer"}
    assert parse_message(msg, is_reasoning=True) == ("thought", "answer")


def test_reasoning_field_alt():
    msg = {"reasoning": "thought2", "content": "ans"}
    assert parse_message(msg, is_reasoning=True) == ("thought2", "ans")


def test_think_tag_extraction():
    msg = {"content": "<think>internal</think>final answer"}
    r, a = parse_message(msg, is_reasoning=True)
    assert r == "internal"
    assert a == "final answer"


def test_non_reasoning_passthrough():
    msg = {"content": "hello"}
    assert parse_message(msg, is_reasoning=False) == ("", "hello")


def test_reasoning_missing_fails_fast():
    msg = {"content": "no thinking here"}
    with pytest.raises(AssertionError, match="reasoning"):
        parse_message(msg, is_reasoning=True)


def test_object_attribute_access():
    class M:
        reasoning_content = "T"
        content = "A"

    assert parse_message(M(), is_reasoning=True) == ("T", "A")
