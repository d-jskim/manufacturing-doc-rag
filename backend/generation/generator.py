"""검색된 청크와 질문으로 프롬프트를 만들고, LLM 답변을 받아 형식과 자료 ID를 검증한다."""

import json
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

STATUSES = ("answered", "needs_clarification", "abstained")
ERROR_ANSWER = "요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."


class LLMAnswer(BaseModel):
    status: str
    answer: str
    reason: str
    source_ids: list[str]


def build_context(chunks: list[dict]) -> tuple[str, dict[str, dict]]:
    """청크마다 이번 요청에서만 쓰는 자료 ID(S1, S2…)를 붙이고, 자료 ID와 출처의 대응표를 만든다."""
    blocks, sources = [], {}
    for number, chunk in enumerate(chunks, start=1):
        source_id = f"S{number}"
        start, end = chunk["page_start"], chunk["page_end"]
        pages = str(start) if start == end else f"{start}-{end}"
        sources[source_id] = {
            "source_id": source_id, "chunk_id": chunk["chunk_id"], "book_id": chunk["book_id"],
            "file_name": chunk["file_name"], "page_start": start, "page_end": end,
        }
        blocks.append(f"[{source_id}]\n문서 ID: {chunk['book_id']}\n문서명: {chunk['file_name']}\n"
                      f"PDF 페이지: {pages}\n내용: {chunk['text']}")
    return "\n\n".join(blocks), sources


def parse_output(content: str, sources: dict) -> tuple[LLMAnswer | None, str | None]:
    """형식, 상태 값, 자료 ID를 검사한다. 통과하면 (답변, None), 실패하면 (None, 오류 이름)."""
    try:
        parsed = LLMAnswer.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError):
        return None, "output_invalid_json"
    if parsed.status not in STATUSES:
        return None, "output_invalid_status"
    if any(source_id not in sources for source_id in parsed.source_ids):
        return None, "output_unknown_source_id"  # 문맥에 없던 자료 ID는 통과시키지 않음
    return parsed, None


def total_usage(usages: list[dict | None]) -> dict | None:
    """응답이 없거나 사용량을 확인하지 못한 응답이 하나라도 있으면 확인 불가(None)로 남긴다."""
    if not usages or any(u is None for u in usages):
        return None
    return {"input_tokens": sum(u["input_tokens"] for u in usages),
            "output_tokens": sum(u["output_tokens"] for u in usages)}


class Generator:
    def __init__(self, prompt_config: dict):
        self.prompt_version = prompt_config["prompt_version"]
        self.settings = prompt_config["generation"]
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_config["system"]), ("human", prompt_config["human"])])
        options = {
            "model": self.settings["model"],
            "max_tokens": self.settings["max_output_tokens"],
            "timeout": self.settings["timeout_seconds"],
            "max_retries": self.settings["api_retries"],
        }
        for key in ("temperature", "seed"):  # 모델이 지원하는 값만 지정
            if self.settings.get(key) is not None:
                options[key] = self.settings[key]
        self._llm = ChatOpenAI(**options).bind(response_format={"type": "json_object"})

    def generate(self, question: str, chunks: list[dict]) -> dict:
        context, sources = build_context(chunks)
        messages = self._prompt.format_messages(question=question, context=context)
        usages: list[dict | None] = []
        start = time.perf_counter()
        error = None
        attempts = 0

        for attempts in range(1, self.settings["output_retries"] + 2):
            try:
                response = self._llm.invoke(messages)
            except Exception as exc:  # 시간 초과, 인증, 잘못된 요청, 서버 오류 등 모델 호출 실패
                error = f"model_error:{type(exc).__name__}: {exc}"
                break
            usages.append(response.usage_metadata)
            parsed, error = parse_output(response.content, sources)
            if parsed:
                return {
                    "status": parsed.status, "answer": parsed.answer, "reason": parsed.reason,
                    "citations": [sources[s] for s in parsed.source_ids],
                    **self._record(attempts, start, usages, None, context),
                }

        return {"status": "error", "answer": ERROR_ANSWER, "reason": error, "citations": [],
                **self._record(attempts, start, usages, error, context)}

    def _record(self, attempts: int, start: float, usages: list, error: str | None, context: str) -> dict:
        return {
            "prompt_version": self.prompt_version, "model": self.settings["model"],
            "attempts": attempts, "latency_seconds": round(time.perf_counter() - start, 3),
            "usage": total_usage(usages), "error": error, "final_context": context,
        }
