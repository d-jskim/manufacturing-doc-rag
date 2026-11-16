"""평가 요청과 운영 지표 응답 형식을 확인한다."""

import pytest
from pydantic import ValidationError

from backend.api.schemas import QueryRequest, QueryResponse


def test_service_request_does_not_require_execution_metadata():
    request = QueryRequest(
        question="CV-20 기능을 알려 주세요.",
        retrieval_mode="hybrid",
        chunk_version="c3",
        user_id="demo_field",
        session_id="session-1",
    )

    assert request.source == "service"
    assert request.chunk_version == "c3"
    assert request.user_id == "demo_field"
    assert request.session_id == "session-1"
    assert request.run_id is None
    assert request.question_id is None


def test_evaluation_request_accepts_execution_metadata():
    request = QueryRequest(
        question="CV-20 기능을 알려 주세요.",
        retrieval_mode="rerank",
        source="evaluation",
        run_id="e04_test",
        question_id="M-DEV-01",
    )

    assert request.source == "evaluation"
    assert request.run_id == "e04_test"
    assert request.question_id == "M-DEV-01"
    assert request.user_id is None
    assert request.session_id is None


@pytest.mark.parametrize(
    ("missing_field", "values"),
    [
        (
            "run_id",
            {
                "question_id": "M-DEV-01",
            },
        ),
        (
            "question_id",
            {
                "run_id": "e04_test",
            },
        ),
    ],
)
def test_evaluation_request_requires_execution_metadata(
    missing_field,
    values,
):
    with pytest.raises(
        ValidationError,
        match="run_id와 question_id",
    ):
        QueryRequest(
            question="CV-20 기능을 알려 주세요.",
            retrieval_mode="rerank",
            source="evaluation",
            **values,
        )


def test_query_response_accepts_operational_metrics():
    response = QueryResponse(
        request_id="request-1",
        status="answered",
        answer="답변",
        reason="근거 확인",
        citations=[],
        retrieval_results=[{
            "rank": 1,
            "chunk_id": "MB01_c3_0002",
            "book_id": "MB01",
            "file_name": "MB01.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "CV-20은 네트워크 수신을 지원합니다.",
            "score": 0.9,
            "score_type": "cosine",
            "used_in_answer": True,
        }],
        retrieval_mode="hybrid",
        settings={
            "dataset": "manufacturing_basic",
            "chunk_version": "c3",
            "embedding": "m1",
            "prompt": "p1",
        },
        metrics={
            "latency_seconds": 2.5,
            "retrieval_seconds": 0.2,
            "generation_seconds": 2.3,
            "attempts": 1,
            "context_chunk_ids": ["MB01-p001-c001"],
            "context_token_count": 640,
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 100,
            },
        },
    )

    assert response.metrics.latency_seconds == 2.5
    assert response.metrics.retrieval_seconds == 0.2
    assert response.metrics.generation_seconds == 2.3
    assert response.metrics.attempts == 1
    assert response.metrics.context_chunk_ids == ["MB01-p001-c001"]
    assert response.metrics.context_token_count == 640
    assert response.metrics.usage.input_tokens == 1200
    assert response.metrics.usage.output_tokens == 100
    assert response.retrieval_results[0].used_in_answer is True
    assert response.settings.chunk_version == "c3"


def test_query_response_allows_unavailable_metrics():
    response = QueryResponse(
        request_id="request-1",
        status="error",
        answer="요청을 처리하지 못했습니다.",
        reason="retrieval_error:ConnectError",
        citations=[],
        retrieval_mode="hybrid",
        metrics=None,
    )

    assert response.metrics is None
