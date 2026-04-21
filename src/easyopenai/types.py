from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict]
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    # Runtime state (not serialized as part of request)
    attempted_providers: set[str] = Field(default_factory=set, exclude=True)
    retry_count: int = Field(default=0, exclude=True)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0


class Result(BaseModel):
    task_id: str
    provider: str
    model: str
    reasoning_content: str = ""
    answer_content: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ProviderStats(BaseModel):
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inflight: int = 0
