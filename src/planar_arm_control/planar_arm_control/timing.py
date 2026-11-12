"""Timer-jitter / loop-time measurement for periodic callbacks."""

import threading
import time
from collections import deque

import numpy as np


class LoopTimer:
    """
    Measures the real period between successive calls of a periodic callback
    and how long the callback body takes.

        tick = monitor.start()      # at the top of the callback, returns dt
        ...
        monitor.stop()              # at the bottom
    """

    def __init__(self, target_period, window=1000):
        self.target_period = float(target_period)
        self._periods = deque(maxlen=window)
        self._exec = deque(maxlen=window)
        self._last = None
        self._t_start = None
        self._lock = threading.Lock()

    def start(self):
        now = time.perf_counter()
        dt = self.target_period if self._last is None else now - self._last
        if self._last is not None:
            with self._lock:
                self._periods.append(dt)
        self._last = now
        self._t_start = now
        return dt

    def stop(self):
        if self._t_start is not None:
            with self._lock:
                self._exec.append(time.perf_counter() - self._t_start)

    def reset(self):
        with self._lock:
            self._periods.clear()
            self._exec.clear()
        self._last = None

    def stats(self):
        """Dict of timing statistics in milliseconds."""
        with self._lock:
            p = np.asarray(self._periods, dtype=float)
            e = np.asarray(self._exec, dtype=float)
        out = {"target_ms": self.target_period * 1e3, "n": int(p.size),
               "mean_ms": 0.0, "std_ms": 0.0, "max_ms": 0.0, "p99_jitter_ms": 0.0,
               "exec_mean_ms": 0.0, "exec_max_ms": 0.0}
        if p.size:
            jitter = np.abs(p - self.target_period)
            out.update(mean_ms=p.mean() * 1e3, std_ms=p.std() * 1e3,
                       max_ms=p.max() * 1e3,
                       p99_jitter_ms=float(np.percentile(jitter, 99)) * 1e3)
        if e.size:
            out.update(exec_mean_ms=e.mean() * 1e3, exec_max_ms=e.max() * 1e3)
        return out
