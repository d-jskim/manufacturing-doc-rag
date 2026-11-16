"""운영 설정(configs/active.yaml)에 따라 검색과 답변 생성을 한 번에 실행하고, 요청마다 Langfuse에 기록한다.
API와 평가 실험이 함께 쓴다."""

import json
import time
import uuid
from pathlib import Path

import yaml
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient

from backend.core.token_counter import TiktokenCounter
from backend.generation.generator import Generator
from backend.retrieval.context import fill_context
from backend.retrieval.embedder import Embedder
from backend.retrieval.keyword import BM25Index, make_tokenizer
from backend.retrieval.reranker import Reranker
from backend.retrieval.rrf import rrf_fuse
from backend.retrieval.vector_store import search

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
PROCESSED_DIR = ROOT / "data" / "processed"
SCORE_TYPES = {
    "dense": "cosine",
    "hybrid": "rrf",
    "hybrid_rerank": "cross_encoder",
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class RAGPipeline:
    def __init__(self, active_config_path: Path = CONFIGS_DIR / "active.yaml"):
        """서버가 켜질 때 한 번만 만든다. 검색 모델과 색인, 생성기를 미리 준비한다."""
        self.config = load_yaml(active_config_path)
        embedding_config = load_yaml(
            CONFIGS_DIR / "embedding" / f"{self.config['embedding']}.yaml"
        )
        prompt_config = load_yaml(
            CONFIGS_DIR / "prompts" / f"{self.config['prompt']}.yaml"
        )
        self.retrieval_configs = {
            mode: load_yaml(CONFIGS_DIR / "retrieval" / f"{version}.yaml")
            for mode, version in self.config["retrieval_modes"].items()
        }
        for mode, retrieval_config in self.retrieval_configs.items():
            method = retrieval_config["method"]
            uses_rrf = method in {"hybrid", "hybrid_rerank"}
            if method not in SCORE_TYPES or (
                uses_rrf and retrieval_config["fusion"] != "rrf"
            ):
                raise ValueError(
                    f"지원하지 않는 검색 설정: "
                    f"{mode} → {retrieval_config['retrieval_version']}"
                )

        self.embedder = Embedder(embedding_config)
        self.client = QdrantClient(url=self.config["qdrant_url"])
        self.generator = Generator(prompt_config)
        self.counter = TiktokenCounter(
            prompt_config["generation"]["model"],
            "o200k_base",
        )
        self.langfuse = get_client()
        self.keyword_indexes = {}
        self.rerankers = self._build_rerankers()

    def _keyword_index(self, chunk_version: str, tokenizer_name: str) -> BM25Index:
        """청킹 버전과 토크나이저별 BM25 색인을 처음 요청할 때 만든다."""
        cache_key = (chunk_version, tokenizer_name)
        if cache_key in self.keyword_indexes:
            return self.keyword_indexes[cache_key]
        chunks_path = (
            PROCESSED_DIR
            / self.config["dataset"]
            / chunk_version
            / "chunks.jsonl"
        )
        chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.keyword_indexes[cache_key] = BM25Index(
            chunks, make_tokenizer(tokenizer_name)
        )
        return self.keyword_indexes[cache_key]

    def _build_rerankers(self) -> dict:
        """재순위 검색 방식별 Cross-encoder를 서버 시작 시 한 번 만든다."""
        return {
            mode: Reranker(config)
            for mode, config in self.retrieval_configs.items()
            if config["method"] == "hybrid_rerank"
        }

    @property
    def retrieval_modes(self) -> list[str]:
        return list(self.retrieval_configs)

    def retrieve(
        self,
        question: str,
        retrieval_mode: str,
        chunk_version: str | None = None,
    ) -> dict:
        """질문을 검색하고 선택한 방식으로 후보를 정렬한 뒤 문맥을 채운다."""
        chunk_version = chunk_version or self.config["chunk_version"]
        collection = (
            f"{self.config['dataset']}_{chunk_version}_{self.embedder.model_code}"
        )
        retrieval_config = self.retrieval_configs[retrieval_mode]
        method = retrieval_config["method"]
        query_vector = self.embedder.embed_query(question)

        if method == "dense":
            candidates = search(
                self.client,
                collection,
                query_vector,
                retrieval_config["candidate_count"],
                exact=retrieval_config["exact"],
            )
        else:
            dense = search(
                self.client,
                collection,
                query_vector,
                retrieval_config["dense_candidate_count"],
                exact=retrieval_config["exact"],
            )
            keyword = self._keyword_index(
                chunk_version, retrieval_config["keyword_tokenizer"]
            ).search(
                question,
                retrieval_config["keyword_candidate_count"],
            )
            candidates = rrf_fuse(
                dense,
                keyword,
                k=retrieval_config["rrf_k"],
                limit=retrieval_config["candidate_count"],
            )
            if method == "hybrid_rerank":
                candidates = self.rerankers[retrieval_mode].rerank(
                    question,
                    candidates,
                )

        final_k = self.config["final_k"]
        selected_candidates = candidates[:final_k]
        context, used = fill_context(
            selected_candidates,
            self.counter,
            self.config["context_token_budget"],
        )
        return {
            "retrieval_version": retrieval_config["retrieval_version"],
            "chunk_version": chunk_version,
            "collection": collection,
            "score_type": SCORE_TYPES[method],
            "final_k": final_k,
            "candidates": candidates,
            "context": context,
            "context_token_count": used,
        }

    def query(
        self,
        question: str,
        retrieval_mode: str | None = None,
        request_id: str | None = None,
        source: str = "service",
        extra_metadata: dict | None = None,
        chunk_version: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """질문 하나를 검색부터 답변까지 처리한다. 요청 하나가 Langfuse trace 하나가 된다.

        source: service(서비스 질문) 또는 evaluation(평가 실행)
        extra_metadata: 평가 시 실행 ID·질문 ID 등
        """
        retrieval_mode = (
            retrieval_mode or self.config["default_retrieval_mode"]
        )
        chunk_version = chunk_version or self.config["chunk_version"]
        request_id = request_id or str(uuid.uuid4())
        retrieval_version = self.retrieval_configs[
            retrieval_mode
        ]["retrieval_version"]
        metadata = {
            "source": source,
            "request_id": request_id,
            "dataset": self.config["dataset"],
            "chunk_version": chunk_version,
            "embedding": self.embedder.model_code,
            "prompt": self.generator.prompt_version,
            "retrieval": retrieval_version,
            "retrieval_mode": retrieval_mode,
            **({"user_id": user_id} if user_id else {}),
            **({"session_id": session_id} if session_id else {}),
            **(extra_metadata or {}),
        }
        trace_id = self.langfuse.create_trace_id(seed=request_id)
        start = time.perf_counter()

        with self.langfuse.start_as_current_observation(
            name="rag_query",
            as_type="span",
            input={"question": question},
            trace_context={"trace_id": trace_id},
        ) as root:
            with propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                trace_name="rag_query",
                tags=[source, retrieval_mode],
                metadata={key: str(value) for key, value in metadata.items()},
            ):
                try:
                    with self.langfuse.start_as_current_observation(
                        name="retrieval",
                        as_type="retriever",
                        input={"question": question},
                    ) as span:
                        retrieval = self.retrieve(
                            question, retrieval_mode, chunk_version
                        )
                        retrieval_seconds = time.perf_counter() - start
                        retrieval_record = {
                            "version": retrieval["retrieval_version"],
                            "collection": retrieval["collection"],
                            "score_type": retrieval["score_type"],
                            "final_k": retrieval["final_k"],
                            "candidates": [
                                {
                                    "chunk_id": candidate["chunk_id"],
                                    "score": round(candidate["score"], 4),
                                    **{
                                        key: round(candidate[key], 4)
                                        for key in (
                                            "rrf_score",
                                            "rerank_score",
                                        )
                                        if key in candidate
                                    },
                                    **{
                                        key: candidate[key]
                                        for key in (
                                            "dense_rank",
                                            "keyword_rank",
                                            "rerank_rank",
                                        )
                                        if key in candidate
                                    },
                                }
                                for candidate in retrieval["candidates"]
                            ],
                            "context_chunk_ids": [
                                candidate["chunk_id"]
                                for candidate in retrieval["context"]
                            ],
                            "context_token_count": retrieval[
                                "context_token_count"
                            ],
                            "seconds": round(retrieval_seconds, 3),
                        }
                        span.update(output=retrieval_record)

                    result = self.generator.generate(
                        question,
                        retrieval["context"],
                        callbacks=[CallbackHandler()],
                    )
                    result["generation_seconds"] = result.pop(
                        "latency_seconds"
                    )

                    total_seconds = round(time.perf_counter() - start, 3)
                    settings = {
                        "dataset": self.config["dataset"],
                        "chunk_version": chunk_version,
                        "embedding": self.embedder.model_code,
                        "prompt": self.generator.prompt_version,
                    }
                    context_chunk_ids = set(
                        retrieval_record["context_chunk_ids"]
                    )
                    root_retrieval_results = [
                        {
                            "rank": rank,
                            "chunk_id": candidate["chunk_id"],
                            "file_name": candidate["file_name"],
                            "page_start": candidate["page_start"],
                            "page_end": candidate["page_end"],
                            "text": candidate["text"],
                            "score": candidate["score"],
                            "score_type": retrieval["score_type"],
                            "used_in_answer": candidate["chunk_id"]
                            in context_chunk_ids,
                        }
                        for rank, candidate in enumerate(
                            retrieval["candidates"][: retrieval["final_k"]],
                            start=1,
                        )
                    ]
                    output = {
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "status": result["status"],
                        "answer": result["answer"],
                        "reason": result["reason"],
                        "citations": result["citations"],
                        "settings": settings,
                        "retrieval": retrieval_record,
                        "retrieval_results": root_retrieval_results,
                        "timings": {
                            "total_seconds": total_seconds,
                            "retrieval_seconds": retrieval_record["seconds"],
                            "generation_seconds": result[
                                "generation_seconds"
                            ],
                        },
                        "usage": result["usage"],
                        "error": result["error"],
                    }
                    usage_details = None
                    if result["usage"]:
                        input_tokens = result["usage"]["input_tokens"]
                        output_tokens = result["usage"]["output_tokens"]
                        usage_details = {
                            "input": input_tokens,
                            "output": output_tokens,
                            "total": input_tokens + output_tokens,
                        }
                    root_metadata = {
                        **metadata,
                        "status": result["status"],
                        "error": result["error"],
                    }
                    if result["status"] == "error":
                        root.update(
                            output=output,
                            metadata=root_metadata,
                            usage_details=usage_details,
                            level="ERROR",
                            status_message=result["error"],
                        )
                    else:
                        root.update(
                            output=output,
                            metadata=root_metadata,
                            usage_details=usage_details,
                        )
                except Exception as exc:
                    error = f"pipeline_error:{type(exc).__name__}"
                    root.update(
                        output={
                            "request_id": request_id,
                            "trace_id": trace_id,
                            "status": "error",
                            "error": error,
                        },
                        metadata={**metadata, "status": "error", "error": error},
                        level="ERROR",
                        status_message=error,
                    )
                    raise

        return {
            "request_id": request_id,
            "trace_id": trace_id,
            "retrieval_mode": retrieval_mode,
            "retrieval_results": retrieval["candidates"][
                : retrieval["final_k"]
            ],
            "settings": settings,
            **result,
            "retrieval": retrieval_record,
            "latency_seconds": total_seconds,
        }

    def flush(self) -> None:
        """남아 있는 Langfuse 기록을 보낸다. 서버가 꺼질 때와 스크립트가 끝날 때 부른다."""
        self.langfuse.flush()
