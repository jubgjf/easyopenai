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
