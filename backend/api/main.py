"""FastAPI 앱: /health, /query. 실행: uv run uvicorn backend.api.main:app"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

from backend.api.schemas import QueryRequest, QueryResponse
from backend.generation.generator import ERROR_ANSWER
from backend.pipeline import RAGPipeline

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 검색 파이프라인을 준비하고 종료 시 기록을 전송한다."""
    global pipeline
    pipeline = RAGPipeline()
    yield
    pipeline.flush()


app = FastAPI(
    title="제조 문서 RAG",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(request: QueryRequest) -> JSONResponse:
    mode = (
        request.retrieval_mode
        or pipeline.config["default_retrieval_mode"]
    )
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
            request_id=request_id,
            source=request.source,
            extra_metadata=extra_metadata,
        )
    except Exception as exc:
        body = QueryResponse(
            request_id=request_id,
            status="error",
            answer=ERROR_ANSWER,
            reason=f"retrieval_error:{type(exc).__name__}",
            citations=[],
            retrieval_mode=mode,
        )
        return JSONResponse(
            status_code=502,
            content=body.model_dump(),
        )

    if result["status"] == "error":
        result["reason"] = result["reason"].split(": ", 1)[0]

    body = QueryResponse(
        **{
            key: result[key]
            for key in QueryResponse.model_fields
        }
    )
    return JSONResponse(
        status_code=(
            502
            if result["status"] == "error"
            else 200
        ),
        content=body.model_dump(),
    )