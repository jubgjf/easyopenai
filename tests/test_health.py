import time


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


def test_half_open_probe_timeout_resets_in_flight():
    """If a HALF_OPEN probe is lost (no record() call), the in-flight flag
    should auto-reset after cooldown_s so the provider isn't stuck forever."""
    m = _mon()
    for _ in range(4):
        m.record(False)
    assert m.state == State.OPEN
    time.sleep(0.06)
    # First probe enters HALF_OPEN
    assert m.can_serve()
    assert m.state == State.HALF_OPEN
    # Simulate lost probe: no record() call, just wait for cooldown
    assert not m.can_serve()  # still blocked
    time.sleep(0.06)
    # After cooldown_s, the stale in-flight should be cleared
    assert m.can_serve()  # new probe allowed
    assert m.state == State.HALF_OPEN
