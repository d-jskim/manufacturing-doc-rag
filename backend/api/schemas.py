"""API 요청·응답 형식. 형식에 맞지 않는 요청은 FastAPI가 422로 거부한다."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(max_length=2000, description="사용자 질문")
    retrieval_mode: str | None = Field(default=None, description="검색 방식(생략하면 운영 설정의 기본값)")

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """공백만 있는 질문도 빈 질문으로 거부한다."""
        if not value.strip():
            raise ValueError("질문이 비어 있습니다.")
        return value.strip()


class Citation(BaseModel):
    source_id: str
    book_id: str
    file_name: str
    page_start: int
    page_end: int


class QueryResponse(BaseModel):
    request_id: str
    status: Literal["answered", "needs_clarification", "abstained", "error"]
    answer: str
    reason: str | None
    citations: list[Citation]
    retrieval_mode: str