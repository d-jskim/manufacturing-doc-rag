"""Langfuse 서비스 대시보드 데이터 처리를 검증한다."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from frontend.views import admin_dashboard
from frontend.views.admin_dashboard import (
    confirmed_usage_count,
    fetch_service_requests,
    filter_requests,
    mode_summary,
    normal_count,
    normalize_observation,
    recent_rows,
)


class FakeStreamlit:
    def __init__(self):
        self.messages = []

    def title(self, text):
        self.messages.append(("title", text))

    def warning(self, text):
        self.messages.append(("warning", text))

    def caption(self, text):
        self.messages.append(("caption", text))

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index]


def observation(source: str, mode: str = "hybrid", status: str = "answered"):
    return SimpleNamespace(
        trace_id=f"trace-{source}-{mode}",
        start_time=datetime(2026, 11, 13, tzinfo=timezone.utc),
        user_id="demo_field",
        session_id="session-1",
        input=json.dumps({"question": "질문"}),
        output=json.dumps(
            {
                "status": status,
                "settings": {"chunk_version": "c3"},
                "timings": {"total_seconds": 2.0},
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "retrieval_results": [{"chunk_id": "MB01_c3_0002"}],
                "citations": [],
                "error": None,
            }
        ),
        metadata={"source": source, "retrieval_mode": mode},
        usage_details=None,
        latency=2.0,
        status_message=None,
    )


def test_fetch_service_requests_uses_observations_api_and_excludes_other_sources():
    captured = {}

    def get_many(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            data=[observation("service"), observation("evaluation")]
        )

    client = SimpleNamespace(
        api=SimpleNamespace(
            observations=SimpleNamespace(get_many=get_many),
        )
    )
    start = datetime(2026, 11, 12, tzinfo=timezone.utc)
    end = datetime(2026, 11, 13, tzinfo=timezone.utc)

    rows = fetch_service_requests(client, start, end)

    assert len(rows) == 1
    assert rows[0]["source"] == "service"
    assert rows[0]["user_id"] == "demo_field"
    assert rows[0]["session_id"] == "session-1"
    assert rows[0]["input_tokens"] == 100
    assert captured["name"] == "rag_query"
    assert captured["is_root_observation"] is True
    assert '"value": "service"' in captured["filter"]


def test_dashboard_filters_and_summarizes_modes():
    rows = [
        {
            **fetch_row,
            "retrieval_mode": mode,
            "status": status,
            "latency_seconds": latency,
        }
        for fetch_row, mode, status, latency in (
            ({"user_id": "demo_field"}, "hybrid", "answered", 2.0),
            ({"user_id": "demo_field"}, "hybrid", "error", 4.0),
            ({"user_id": "demo_quality"}, "naive", "answered", 1.0),
        )
    ]

    filtered = filter_requests(rows, "demo_field", "전체", "전체")
    summary = mode_summary(filtered)

    assert len(filtered) == 2
    assert summary == [
        {
            "검색 방식": "(R2) Hybrid",
            "요청 수": 2,
            "평균 응답 시간(초)": 3.0,
            "오류율": "50.0%",
        }
    ]


def test_dashboard_does_not_convert_unknown_usage_or_latency_to_zero():
    row = {
        "trace_id": "trace-1",
        "start_time": datetime(2026, 11, 13, tzinfo=timezone.utc),
        "user_id": "demo_field",
        "session_id": "session-1",
        "question": "질문",
        "retrieval_mode": "hybrid",
        "status": "answered",
        "latency_seconds": None,
        "input_tokens": 100,
        "output_tokens": None,
    }

    assert recent_rows([row])[0]["토큰 수"] == "-"
    assert confirmed_usage_count([row]) == 0
    assert mode_summary([row])[0]["평균 응답 시간(초)"] == "-"


def test_dashboard_merges_retrieval_settings_and_normalizes_missing_error_status():
    item = observation("service")
    item.output = json.dumps(
        {
            "settings": {
                "dataset": "manufacturing_basic",
                "chunk_version": "c3",
                "embedding": "m1",
                "prompt": "p1",
            },
            "error": "pipeline_error:RuntimeError",
        }
    )
    item.metadata = {
        "source": "service",
        "retrieval": "r2",
        "retrieval_mode": "hybrid",
    }

    row = normalize_observation(item)

    assert row["status"] == "error"
    assert row["settings"]["retrieval"] == "r2"
    assert row["settings"]["retrieval_mode"] == "hybrid"
    assert normal_count([row]) == 0


def test_dashboard_counts_only_documented_normal_statuses():
    rows = [
        {"status": "answered"},
        {"status": "needs_clarification"},
        {"status": "abstained"},
        {"status": "error"},
        {"status": "-"},
    ]

    assert normal_count(rows) == 3


def test_dashboard_handles_missing_configuration(monkeypatch):
    streamlit = FakeStreamlit()
    monkeypatch.setattr(admin_dashboard, "st", streamlit)
    monkeypatch.setattr(admin_dashboard, "configured", lambda: False)

    admin_dashboard.render()

    assert ("warning", "Langfuse 연결 정보가 설정되지 않았습니다.") in streamlit.messages


def test_dashboard_handles_connection_failure(monkeypatch):
    streamlit = FakeStreamlit()
    monkeypatch.setattr(admin_dashboard, "st", streamlit)
    monkeypatch.setattr(admin_dashboard, "configured", lambda: True)
    monkeypatch.setattr(admin_dashboard, "get_client", lambda: object())
    monkeypatch.setattr(
        admin_dashboard,
        "fetch_service_requests",
        lambda client, start, end: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    admin_dashboard.render()

    assert (
        "warning",
        "Langfuse 서비스 기록을 불러오지 못했습니다.",
    ) in streamlit.messages


def test_dashboard_handles_empty_results(monkeypatch):
    streamlit = FakeStreamlit()
    monkeypatch.setattr(admin_dashboard, "st", streamlit)
    monkeypatch.setattr(admin_dashboard, "configured", lambda: True)
    monkeypatch.setattr(admin_dashboard, "get_client", lambda: object())
    monkeypatch.setattr(
        admin_dashboard, "fetch_service_requests", lambda client, start, end: []
    )

    admin_dashboard.render()

    assert (
        "caption",
        "조회 기간에 서비스 요청이 없습니다.",
    ) in streamlit.messages
