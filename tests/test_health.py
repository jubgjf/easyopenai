import time

import pytest

from easyopenai.config import HealthConfig
from easyopenai.health import HealthMonitor, State


def _mon(**kw) -> HealthMonitor:
    return HealthMonitor(HealthConfig(window_size=4, failure_threshold=0.5, cooldown_s=0.05, **kw))


def test_closed_allows_traffic():
    m = _mon()
    assert m.state == State.CLOSED
    assert m.can_serve()


def test_opens_after_window_failure_rate():
    m = _mon()
    for _ in range(3):
        m.record(False)
    # window not full yet -> still closed
    assert m.state == State.CLOSED
    m.record(False)  # window filled with all failures
    assert m.state == State.OPEN
    assert not m.can_serve()


def test_half_open_after_cooldown_and_recover():
    m = _mon()
    for _ in range(4):
        m.record(False)
    assert m.state == State.OPEN
    time.sleep(0.06)
    assert m.can_serve()  # transitions to HALF_OPEN, allows one probe
    assert m.state == State.HALF_OPEN
    assert not m.can_serve()  # second probe gated out
    m.record(True)
    assert m.state == State.CLOSED


def test_half_open_probe_failure_reopens():
    m = _mon()
    for _ in range(4):
        m.record(False)
    time.sleep(0.06)
    assert m.can_serve()
    m.record(False)
    assert m.state == State.OPEN
