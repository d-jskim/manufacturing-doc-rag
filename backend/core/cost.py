"""실행 비용(단계별 시간, 요청별 시간, 토큰 수)을 모든 실험이 같은 구조로 기록한다."""

import platform
import statistics
import time
from contextlib import contextmanager


def summarize_seconds(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(ordered[max(0, round(0.95 * len(ordered)) - 1)], 4),
    }


def environment() -> dict:
    """시간은 장비에 따라 달라지므로 실행 환경을 함께 남긴다."""
    info = {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor(), "gpu": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


class CostRecorder:
    def __init__(self) -> None:
        self.stage_seconds: dict[str, float] = {}
        self.request_seconds: dict[str, list[float]] = {}
        self.token_counts: dict[str, int | None] = {}

    @contextmanager
    def stage(self, name: str):
        """with 블록 안의 작업 시간을 단계 이름으로 더한다(단계끼리 겹치게 쓰지 않음)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stage_seconds[name] = round(self.stage_seconds.get(name, 0.0) + elapsed, 3)

    def add_request_seconds(self, name: str, seconds: float) -> None:
        """질문 하나를 처리한 시간처럼 요청 단위 시간을 모은다."""
        self.request_seconds.setdefault(name, []).append(seconds)

    def add_tokens(self, name: str, count: int | None) -> None:
        """확인하지 못한 사용량은 0이 아니라 None(확인 불가)으로 남긴다."""
        if count is None:
            self.token_counts.setdefault(name, None)
            return
        self.token_counts[name] = (self.token_counts.get(name) or 0) + count

    def to_dict(self) -> dict:
        return {
            "stage_seconds": self.stage_seconds,
            "total_seconds": round(sum(self.stage_seconds.values()), 3),
            "request_seconds": {name: summarize_seconds(v) for name, v in self.request_seconds.items()},
            "token_counts": self.token_counts,
            "environment": environment(),
        }