"""API 요청·응답 형식. 형식에 맞지 않는 요청은 FastAPI가 422로 거부한다."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class QueryRequest(BaseModel):
    question: str = Field(
        max_length=2000,
        description="사용자 질문",
    )
    retrieval_mode: str | None = Field(
        default=None,
        description="검색 방식. 생략하면 운영 설정의 기본값을 사용한다.",
    )
    chunk_version: Literal["c1", "c2", "c3"] | None = Field(
        default=None,
        description="청킹 버전. 생략하면 운영 설정의 기본값을 사용한다.",
    )
    source: Literal["service", "evaluation", "security"] = Field(
        default="service",
        description="요청 출처",
    )
    run_id: str | None = Field(
        default=None,
        max_length=100,
        description="평가 또는 보안 실행 ID",
    )
    question_id: str | None = Field(
        default=None,
        max_length=100,
        description="평가 질문 ID",
    )
    user_id: str | None = Field(
        default=None,
        max_length=100,
        description="서비스 사용자 ID",
    )
    session_id: str | None = Field(
        default=None,
        max_length=100,
        description="서비스 대화 세션 ID",
    )

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """공백만 있는 질문은 빈 질문으로 거부한다."""
        if not value.strip():
            raise ValueError("질문이 비어 있습니다.")
        return value.strip()

    @model_validator(mode="after")
    def require_execution_metadata(self):
        """평가·보안 요청에는 실행 ID와 질문 ID를 요구한다."""
        if self.source in {"evaluation", "security"}:
            if not self.run_id or not self.question_id:
                raise ValueError(
                    "평가·보안 요청에는 run_id와 question_id가 필요합니다."
                )
        return self


class Citation(BaseModel):
    source_id: str
    book_id: str
    file_name: str
    page_start: int
    page_end: int


class RetrievalResult(BaseModel):
    rank: int
    chunk_id: str
    book_id: str
    file_name: str
    page_start: int
    page_end: int
    text: str
    score: float
    score_type: str
    used_in_answer: bool
    metadata: dict = Field(default_factory=dict)


class QuerySettings(BaseModel):
    dataset: str
    chunk_version: str
    embedding: str
    prompt: str


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class QueryMetrics(BaseModel):
    latency_seconds: float
    retrieval_seconds: float
    generation_seconds: float
    attempts: int
    context_chunk_ids: list[str]
    context_token_count: int
    usage: TokenUsage | None


class QueryResponse(BaseModel):
    request_id: str
    status: Literal[
        "answered",
        "needs_clarification",
        "abstained",
        "error",
    ]
    answer: str
    reason: str | None
    citations: list[Citation]
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    retrieval_mode: str
    settings: QuerySettings | None = None
    metrics: QueryMetrics | None = None
