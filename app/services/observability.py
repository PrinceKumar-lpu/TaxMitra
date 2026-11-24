"""
Observability and Metrics Service for TaxMitra.
Interview "Why":
- Why structured JSON logs? Standard text logs are difficult to parse in ELK/Datadog/CloudWatch. Structured JSON
  allows querying by session_id, latency, tool calls, and error rates seamlessly.
- Why an in-memory /metrics endpoint? Enables real-time health and performance monitoring (P95/P99 latency,
  error percentage, tool call breakdown) without requiring external heavy monitoring infrastructure.
"""

import time
import json
import logging
from typing import Dict, Any, List
from threading import Lock

logger = logging.getLogger("taxmitra.observability")


class MetricsTracker:
    def __init__(self):
        self._lock = Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.latencies: List[float] = []
        self.tool_calls_counts: Dict[str, int] = {
            "calculate_tax": 0,
            "compare_regimes": 0
        }
        self.retrieval_mode_counts: Dict[str, int] = {
            "dense": 0,
            "hybrid": 0,
            "hybrid+rerank": 0
        }
        self.guardrail_rejections = 0
        self.feedback_counts = {"up": 0, "down": 0}
        self.start_time = time.time()

    def record_request(
        self,
        latency_ms: float,
        success: bool,
        retrieval_mode: str = "hybrid+rerank",
        tools_used: List[str] = None,
        rejected_by_guardrail: bool = False
    ):
        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            self.total_latency_ms += latency_ms
            self.latencies.append(latency_ms)
            if len(self.latencies) > 500:
                self.latencies.pop(0)

            if rejected_by_guardrail:
                self.guardrail_rejections += 1

            if retrieval_mode in self.retrieval_mode_counts:
                self.retrieval_mode_counts[retrieval_mode] += 1
            else:
                self.retrieval_mode_counts[retrieval_mode] = 1

            if tools_used:
                for tool in tools_used:
                    self.tool_calls_counts[tool] = self.tool_calls_counts.get(tool, 0) + 1

    def record_feedback(self, rating: str):
        with self._lock:
            if rating in self.feedback_counts:
                self.feedback_counts[rating] += 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            uptime_seconds = round(time.time() - self.start_time, 2)
            avg_latency = round(self.total_latency_ms / max(1, self.total_requests), 2)
            error_rate = round((self.failed_requests / max(1, self.total_requests)) * 100, 2)

            # Calculate p95 if available
            p95_latency = 0.0
            if self.latencies:
                sorted_lat = sorted(self.latencies)
                idx = int(len(sorted_lat) * 0.95)
                p95_latency = round(sorted_lat[min(idx, len(sorted_lat) - 1)], 2)

            return {
                "uptime_seconds": uptime_seconds,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "error_rate_pct": error_rate,
                "average_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "guardrail_rejections": self.guardrail_rejections,
                "tool_calls": self.tool_calls_counts,
                "retrieval_modes": self.retrieval_mode_counts,
                "user_feedback": self.feedback_counts
            }


metrics_tracker = MetricsTracker()


def log_structured_event(event_type: str, data: Dict[str, Any]):
    """Logs a single line JSON formatted event."""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": event_type,
        **data
    }
    logger.info(json.dumps(log_entry))
