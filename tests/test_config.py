import textwrap
from pathlib import Path

import pytest

from easyopenai.config import load_config, mask_key


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_env_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-abcdef1234567890")
    cfg_path = write(
        tmp_path,
        "config.yaml",
        """
        providers:
          - name: p1
            base_url: https://x/v1
            api_key: ${MY_KEY}
            models:
              - name: m
                is_reasoning: false
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.providers[0].api_key == "sk-abcdef1234567890"


def test_missing_env_var_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg_path = write(
        tmp_path,
        "config.yaml",
        """
        providers:
          - name: p1
            base_url: https://x/v1
            api_key: ${MISSING_KEY}
            models:
              - name: m
        """,
    )
    with pytest.raises(AssertionError, match="MISSING_KEY"):
        load_config(cfg_path)


def test_mask_key():
    assert mask_key("sk-abcdef1234567890") == "sk-****7890"
    assert mask_key("short") == "****"


def test_no_providers_fails(tmp_path):
    cfg_path = write(
        tmp_path,
        "config.yaml",
        """
        providers: []
        """,
    )
    with pytest.raises(Exception):
        load_config(cfg_path)


def test_per_model_overrides(tmp_path, monkeypatch):
    """Model-level force_stream / max_concurrency / max_rpm override provider defaults;
    unset fields fall back to provider values."""
    monkeypatch.setenv("K", "sk-abcdef1234567890")
    cfg_path = write(
        tmp_path,
        "config.yaml",
        """
        providers:
          - name: p1
            base_url: https://x/v1
            api_key: ${K}
            max_concurrency: 8
            max_rpm: 100
            force_stream: true
            models:
              - name: m_default
                is_reasoning: false
              - name: m_override
                is_reasoning: true
                force_stream: false
                max_concurrency: 2
                max_rpm: 30
        """,
    )
    cfg = load_config(cfg_path)
    p = cfg.providers[0]
    # Provider-level defaults
    assert p.force_stream is True
    assert p.max_concurrency == 8
    assert p.max_rpm == 100
    # Model without overrides — fields are None (Provider resolves at runtime)
    m_def = next(m for m in p.models if m.name == "m_default")
    assert m_def.force_stream is None
    assert m_def.max_concurrency is None
    assert m_def.max_rpm is None
    # Model with overrides
    m_ovr = next(m for m in p.models if m.name == "m_override")
    assert m_ovr.force_stream is False
    assert m_ovr.max_concurrency == 2
    assert m_ovr.max_rpm == 30
