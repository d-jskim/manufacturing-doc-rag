"""AI 어시스턴트의 API 요청 세션 흐름을 검증한다."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest


class StopRender(Exception):
    pass


def load_service_chat(monkeypatch):
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    fake_streamlit = ModuleType("streamlit")

    def stop_at_render(*args, **kwargs):
        raise StopRender

    fake_streamlit.title = stop_at_render
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.syspath_prepend(str(frontend))
    path = frontend / "views" / "service_chat.py"
    spec = importlib.util.spec_from_file_location("service_chat_flow", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except StopRender:
        pass
    return module


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.payloads = []

    def post(self, url, json):
        self.payloads.append(json)
        if self.fail:
            raise httpx.RequestError("offline")
        mode = json["retrieval_mode"]
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"request_id": f"request-{mode}"},
        )


def test_new_session_api_failure_does_not_create_empty_session(monkeypatch):
    module = load_service_chat(monkeypatch)
    created = []
    monkeypatch.setattr(module, "create_session", lambda *args: created.append(args))

    with pytest.raises(httpx.RequestError):
        module.run_queries(
            FakeClient(fail=True),
            "질문",
            ["hybrid"],
            "c3",
            "demo_field",
            None,
        )

    assert created == []


def test_existing_session_is_reused_without_creating_another(monkeypatch):
    module = load_service_chat(monkeypatch)
    client = FakeClient()
    session_id, _ = module.run_queries(
        client,
        "질문",
        ["hybrid"],
        "c3",
        "demo_field",
        "existing-session",
    )

    assert session_id == "existing-session"
    assert client.payloads[0]["session_id"] == "existing-session"


def test_comparison_requests_share_user_and_session_but_keep_distinct_modes(
    monkeypatch,
):
    module = load_service_chat(monkeypatch)
    client = FakeClient()
    session_id, results = module.run_queries(
        client,
        "질문",
        ["naive", "hybrid", "rerank"],
        "c3",
        "demo_field",
        None,
    )

    assert {payload["user_id"] for payload in client.payloads} == {"demo_field"}
    assert {payload["session_id"] for payload in client.payloads} == {session_id}
    assert [payload["retrieval_mode"] for payload in client.payloads] == [
        "naive",
        "hybrid",
        "rerank",
    ]
    assert {result["request_id"] for result in results} == {
        "request-naive",
        "request-hybrid",
        "request-rerank",
    }


def test_comparison_summary_uses_the_same_citations_as_document_sources(monkeypatch):
    module = load_service_chat(monkeypatch)
    result = {
        "retrieval_mode": "hybrid",
        "status": "answered",
        "citations": [
            {
                "file_name": "MB01.pdf",
                "page_start": 2,
                "page_end": 2,
            }
        ],
        "retrieval_results": [
            {
                "file_name": "MB99.pdf",
                "page_start": 9,
                "page_end": 9,
                "used_in_answer": True,
            }
        ],
        "metrics": {"latency_seconds": 1.2},
    }

    assert module.document_sources(result) == result["citations"]
    summary = module.comparison_summary([result])[0]
    assert summary["1순위 출처"] == "MB01.pdf · 2쪽"
    assert summary["근거 수"] == 1


def test_document_sources_falls_back_to_used_retrieval_results(monkeypatch):
    module = load_service_chat(monkeypatch)
    result = {
        "citations": [],
        "retrieval_results": [
            {
                "file_name": "MB01.pdf",
                "page_start": 2,
                "page_end": 2,
                "used_in_answer": True,
            },
            {
                "file_name": "MB02.pdf",
                "page_start": 4,
                "page_end": 4,
                "used_in_answer": False,
            },
        ],
    }

    assert module.document_sources(result) == [result["retrieval_results"][0]]
