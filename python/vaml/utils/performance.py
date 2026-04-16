import time
from contextlib import contextmanager


class PerformanceTracker:
    def __init__(self):
        self._start_times: dict[str, float] = {}
        self._total_times: dict[str, float] = {}

    def start(self, name: str):
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if name not in self._total_times:
            self._total_times[name] = 0.0
        self._total_times[name] += time.perf_counter() - self._start_times[name]

    @contextmanager
    def time(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def total_time_percentages(self):
        total_time = sum(self._total_times.values())
        return {name: time / total_time for name, time in self._total_times.items()}

    def reset(self):
        self._start_times.clear()
        self._total_times.clear()
