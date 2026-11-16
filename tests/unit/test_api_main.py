"""API가 서비스 식별 정보를 파이프라인에 전달하는지 검증한다."""

import json

from backend.api import main as api_main
from backend.api.schemas import QueryRequest


class FakePipeline:
    config = {
        "dataset": "manufacturing_basic",
        "chunk_version": "c3",
        "embedding": "m1",
        "prompt": "p1",
        "default_retrieval_mode": "hybrid",
    }
    retrieval_modes = ["naive", "hybrid", "rerank"]
    embedder = type("Embedder", (), {"model_code": "m1"})()
    generator = type("Generator", (), {"prompt_version": "p1"})()

    def __init__(self):
        self.kwargs = None

    def query(self, **kwargs):
        self.kwargs = kwargs
        return {
            "request_id": kwargs["request_id"],
            "status": "answered",
            "answer": "답변",
            "reason": "근거 확인",
            "citations": [],
            "retrieval_mode": kwargs["retrieval_mode"],
            "retrieval_results": [],
            "settings": {
                "dataset": "manufacturing_basic",
                "chunk_version": "c3",
                "embedding": "m1",
                "prompt": "p1",
            },
            "generation_seconds": 0.8,
            "attempts": 1,
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "retrieval": {
                "seconds": 0.2,
                "context_chunk_ids": [],
                "context_token_count": 0,
                "score_type": "rrf",
            },
            "latency_seconds": 1.0,
        }


def test_query_passes_service_user_and_session(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(api_main, "pipeline", pipeline)
    request = QueryRequest(
        question="질문",
        source="service",
        retrieval_mode="hybrid",
        chunk_version="c3",
        user_id="demo_field",
        session_id="session-1",
    )

    response = api_main.query(request)
    body = json.loads(response.body)

    assert pipeline.kwargs["user_id"] == "demo_field"
    assert pipeline.kwargs["session_id"] == "session-1"
    assert pipeline.kwargs["source"] == "service"
    assert body["status"] == "answered"
