from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

import os

# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)}")


def _interpolate_env(value: str) -> str:
    def _replace(m: re.Match) -> str:
        var = m.group(1)
        resolved = os.environ.get(var)
        assert resolved is not None, f"Environment variable ${{{var}}} is not set"
        return resolved

    return _ENV_VAR_RE.sub(_replace, value)


def _walk_and_interpolate(obj: Any) -> Any:
    if isinstance(obj, str):
        return _interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_interpolate(v) for v in obj]
    return obj


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return key[:3] + "****" + key[-4:]


class ModelConfig(BaseModel):
    name: str
    is_reasoning: bool = False


class HealthConfig(BaseModel):
    window_size: int = 50
    failure_threshold: float = 0.5
    cooldown_s: float = 60


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key: str
    max_concurrency: int = 8
    max_rpm: int = 300
    force_stream: bool = False
    health: HealthConfig = Field(default_factory=HealthConfig)
    models: list[ModelConfig] = Field(default_factory=list)


class SchedulerConfig(BaseModel):
    max_retries_per_task: int = 3


class LoggingConfig(BaseModel):
    stats_interval_s: float = 30


class AppConfig(BaseModel):
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    providers: list[ProviderConfig]


def load_config(path: str | Path, dotenv_path: str | Path | None = None) -> AppConfig:
    load_dotenv(dotenv_path or Path(path).parent / ".env", override=False)
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), "Config root must be a YAML mapping"
    data = _walk_and_interpolate(data)
    cfg = AppConfig.model_validate(data)
    assert len(cfg.providers) > 0, "At least one provider must be configured"
    for p in cfg.providers:
        assert len(p.models) > 0, f"Provider '{p.name}' must have at least one model"
        logger.info(
            "Loaded provider '{}' base_url={} api_key={} models={}",
            p.name,
            p.base_url,
            mask_key(p.api_key),
            [m.name for m in p.models],
        )
    return cfg
