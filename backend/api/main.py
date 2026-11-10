"""FastAPI 앱: /health, /query. 실행: uv run uvicorn backend.api.main:app"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)  # 시스템 환경 변수에 같은 이름이 있어도 .env 값을 사용

from backend.api.schemas import QueryRequest, QueryResponse
from backend.generation.generator import ERROR_ANSWER
from backend.pipeline import RAGPipeline

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 켜질 때 임베딩 모델과 Qdrant 연결을 준비하고, 모든 요청이 같이 쓴다."""
    global pipeline
    pipeline = RAGPipeline()
    yield
    pipeline.flush()  # 평소에는 자동 전송, 서버가 꺼질 때 아직 보내지 않은 기록을 마저 보냄


app = FastAPI(title="제조 문서 RAG", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> JSONResponse:
    mode = request.retrieval_mode or pipeline.config["default_retrieval_mode"]
    if mode not in pipeline.retrieval_modes:
        raise HTTPException(status_code=422,
                            detail=f"지원하지 않는 검색 방식: {mode}. 가능한 값: {pipeline.retrieval_modes}")

    request_id = str(uuid.uuid4())  # 요청 ID는 서버가 만든다
    try:
        result = pipeline.query(request.question, mode, request_id)
    except Exception as exc:  # 검색 단계 실패(Qdrant 연결 등)
        body = QueryResponse(request_id=request_id, status="error", answer=ERROR_ANSWER,
                             reason=f"retrieval_error:{type(exc).__name__}", citations=[], retrieval_mode=mode)
        return JSONResponse(status_code=502, content=body.model_dump())

    if result["status"] == "error":
        # 응답에는 오류 종류만 보내고, 자세한 메시지는 실행 기록(Langfuse)에만 남긴다
        result["reason"] = result["reason"].split(": ", 1)[0]
    body = QueryResponse(**{key: result[key] for key in QueryResponse.model_fields})
    return JSONResponse(status_code=502 if result["status"] == "error" else 200, content=body.model_dump())