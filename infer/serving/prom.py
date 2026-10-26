"""Prometheus metrics in the plain-text exposition format, with no extra dependency.

Prometheus scrapes GET /metrics/prometheus every few seconds. The format is just
lines of `name{labels} value`; a histogram is a set of cumulative bucket counters
(`_bucket{le="0.5"}` = observations <= 0.5 s) plus `_sum` and `_count`, from which
Prometheus computes percentiles with histogram_quantile().

Counters only go up (Prometheus handles restarts). Gauges are read at scrape time.
"""

from __future__ import annotations

import bisect
import threading
from collections import defaultdict

LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0)


class Histogram:
    def __init__(self, buckets: tuple[float, ...] = LATENCY_BUCKETS) -> None:
        self.buckets = buckets
        self.counts = [0] * len(buckets)  # observations in (buckets[i-1], buckets[i]]
        self.over = 0  # observations above the largest bucket
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        i = bisect.bisect_left(self.buckets, value)
        if i < len(self.buckets):
            self.counts[i] += 1
        else:
            self.over += 1
        self.sum += value
        self.count += 1

    def lines(self, name: str) -> list[str]:
        out, running = [], 0
        for le, n in zip(self.buckets, self.counts, strict=True):
            running += n
            out.append(f'{name}_bucket{{le="{le}"}} {running}')
        out.append(f'{name}_bucket{{le="+Inf"}} {self.count}')
        out.append(f"{name}_sum {self.sum:.6f}")
        out.append(f"{name}_count {self.count}")
        return out


class ServerMetrics:
    """Counters and histograms updated by the engine thread and the request handlers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ttft = Histogram()
        self.latency = Histogram()
        self.finished: dict[str, int] = defaultdict(int)  # by finish_reason
        self.rejected: dict[str, int] = defaultdict(int)  # by reason

    def request_finished(self, reason: str, ttft: float | None, latency: float | None) -> None:
        with self._lock:
            self.finished[reason] += 1
            if ttft is not None:
                self.ttft.observe(ttft)
            if latency is not None:
                self.latency.observe(latency)

    def request_rejected(self, reason: str) -> None:
        with self._lock:
            self.rejected[reason] += 1

    def render(self, gauges: dict[str, tuple[str, float]], counters: dict[str, tuple[str, float]]) -> str:
        """gauges and counters read from the engine: name -> (help text, current value)."""
        lines: list[str] = []

        def head(name: str, kind: str, help_text: str) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")

        for kind, values in (("gauge", gauges), ("counter", counters)):
            for name, (help_text, value) in values.items():
                head(name, kind, help_text)
                lines.append(f"{name} {value}")
        with self._lock:
            head("infer_requests_total", "counter", "Finished requests by finish reason")
            for reason, n in sorted(self.finished.items()):
                lines.append(f'infer_requests_total{{finish_reason="{reason}"}} {n}')
            head("infer_rejected_requests_total", "counter", "Requests turned away, by reason")
            for reason, n in sorted(self.rejected.items()):
                lines.append(f'infer_rejected_requests_total{{reason="{reason}"}} {n}')
            head("infer_time_to_first_token_seconds", "histogram", "Arrival to first token")
            lines += self.ttft.lines("infer_time_to_first_token_seconds")
            head("infer_request_latency_seconds", "histogram", "Arrival to last token")
            lines += self.latency.lines("infer_request_latency_seconds")
        return "\n".join(lines) + "\n"
