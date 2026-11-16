"""FastAPI 앱: /health, /query. 실행: uv run uvicorn backend.api.main:app"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

from backend.api.schemas import (
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
)
from backend.generation.generator import ERROR_ANSWER
from backend.pipeline import RAGPipeline

pipeline: RAGPipeline | None = None


def active_config_path() -> Path:
    """서비스 기본 설정 또는 실험 실행에서 지정한 설정 경로를 반환한다."""
    configured = os.getenv("RAG_ACTIVE_CONFIG_PATH")
    if not configured:
        return ROOT / "configs" / "active.yaml"

    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 검색 파이프라인을 준비하고 종료 시 기록을 전송한다."""
    global pipeline
    pipeline = RAGPipeline(active_config_path())
    yield
    pipeline.flush()


app = FastAPI(
    title="제조 문서 RAG",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    if pipeline is None:
        return {"status": "starting", "dataset": None}

    config = pipeline.config
    return {
        "status": "ok",
        "dataset": config["dataset"],
        "chunk_version": config["chunk_version"],
        "embedding": config["embedding"],
        "prompt": config["prompt"],
        "retrieval_modes": config["retrieval_modes"],
        "final_k": config["final_k"],
        "context_token_budget": config["context_token_budget"],
    }


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(request: QueryRequest) -> JSONResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="API가 아직 준비되지 않았습니다.")
    mode = (
        request.retrieval_mode
        or pipeline.config["default_retrieval_mode"]
    )
    chunk_version = request.chunk_version or pipeline.config["chunk_version"]
    if mode not in pipeline.retrieval_modes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"지원하지 않는 검색 방식: {mode}. "
                f"가능한 값: {pipeline.retrieval_modes}"
            ),
        )

    request_id = str(uuid.uuid4())
    extra_metadata = {
        key: value
        for key, value in {
            "run_id": request.run_id,
            "question_id": request.question_id,
        }.items()
        if value is not None
    }

    try:
        result = pipeline.query(
            question=request.question,
            retrieval_mode=mode,
            chunk_version=chunk_version,
            request_id=request_id,
            source=request.source,
            extra_metadata=extra_metadata,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as exc:
        body = QueryResponse(
            request_id=request_id,
            status="error",
            answer=ERROR_ANSWER,
            reason=f"retrieval_error:{type(exc).__name__}",
            citations=[],
            retrieval_results=[],
            retrieval_mode=mode,
            settings={
                "dataset": pipeline.config["dataset"],
                "chunk_version": chunk_version,
                "embedding": pipeline.embedder.model_code,
                "prompt": pipeline.generator.prompt_version,
            },
            metrics=None,
        )
        return JSONResponse(
            status_code=502,
            content=body.model_dump(),
        )

    if result["status"] == "error":
        result["reason"] = result["reason"].split(": ", 1)[0]

    metrics = QueryMetrics(
        latency_seconds=result["latency_seconds"],
        retrieval_seconds=result["retrieval"]["seconds"],
        generation_seconds=result["generation_seconds"],
        attempts=result["attempts"],
        context_chunk_ids=result["retrieval"]["context_chunk_ids"],
        context_token_count=result["retrieval"]["context_token_count"],
        usage=result["usage"],
    )
    context_chunk_ids = set(result["retrieval"]["context_chunk_ids"])
    retrieval_results = [
        RetrievalResult(
            rank=rank,
            chunk_id=candidate["chunk_id"],
            book_id=candidate["book_id"],
            file_name=candidate["file_name"],
            page_start=candidate["page_start"],
            page_end=candidate["page_end"],
            text=candidate["text"],
            score=candidate["score"],
            score_type=result["retrieval"]["score_type"],
            used_in_answer=candidate["chunk_id"] in context_chunk_ids,
            metadata={
                key: candidate[key]
                for key in (
                    "dense_rank",
                    "keyword_rank",
                    "rrf_score",
                    "rerank_rank",
                    "rerank_score",
                )
                if key in candidate and candidate[key] is not None
            },
        )
        for rank, candidate in enumerate(
            result["retrieval_results"],
            start=1,
        )
    ]
    body = QueryResponse(
        request_id=result["request_id"],
        status=result["status"],
        answer=result["answer"],
        reason=result["reason"],
        citations=result["citations"],
        retrieval_results=retrieval_results,
        retrieval_mode=result["retrieval_mode"],
        settings=result["settings"],
        metrics=metrics,
    )
    return JSONResponse(
        status_code=(
            502
            if result["status"] == "error"
            else 200
        ),
        content=body.model_dump(),
    )
