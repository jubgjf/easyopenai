"""Sliding-window failure-rate circuit breaker.

State machine: CLOSED -> OPEN (after threshold breach) -> HALF_OPEN (after cooldown)
-> CLOSED (on success probe) | OPEN (on failure probe).
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum

from easyopenai.config import HealthConfig


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class HealthMonitor:
    def __init__(self, cfg: HealthConfig):
        assert cfg.window_size > 0
        assert 0 < cfg.failure_threshold <= 1
        assert cfg.cooldown_s > 0
        self._cfg = cfg
        self._window: deque[bool] = deque(maxlen=cfg.window_size)
        self._state: State = State.CLOSED
        self._opened_at: float = 0.0
        self._half_open_in_flight: bool = False

    @property
    def state(self) -> State:
        return self._state

    def record(self, success: bool) -> None:
        self._window.append(success)
        if self._state == State.HALF_OPEN:
            self._half_open_in_flight = False
            if success:
                self._state = State.CLOSED
                self._window.clear()
            else:
                self._state = State.OPEN
                self._opened_at = time.monotonic()
            return

        if self._state == State.CLOSED and len(self._window) >= self._cfg.window_size:
            failures = sum(1 for ok in self._window if not ok)
            if failures / len(self._window) >= self._cfg.failure_threshold:
                self._state = State.OPEN
                self._opened_at = time.monotonic()

    def can_serve(self) -> bool:
        if self._state == State.CLOSED:
            return True
        if self._state == State.OPEN:
            if time.monotonic() - self._opened_at >= self._cfg.cooldown_s:
                self._state = State.HALF_OPEN
                self._half_open_in_flight = True
                return True
            return False
        # HALF_OPEN: only one probe at a time
        if not self._half_open_in_flight:
            self._half_open_in_flight = True
            return True
        return False
