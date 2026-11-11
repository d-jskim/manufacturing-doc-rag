"""평가 요청의 실행 메타데이터 검증을 확인한다."""

import pytest
from pydantic import ValidationError

from backend.api.schemas import QueryRequest


def test_service_request_does_not_require_execution_metadata():
    request = QueryRequest(
        question="CV-20 기능을 알려 주세요.",
        retrieval_mode="hybrid",
    )

    assert request.source == "service"
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