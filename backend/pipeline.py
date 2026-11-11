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
        self.collection = (
            f"{self.config['dataset']}_"
            f"{self.config['chunk_version']}_"
            f"{self.embedder.model_code}"
        )
        self.langfuse = get_client()
        self.keyword_indexes = self._build_keyword_indexes()
        self.rerankers = self._build_rerankers()

    def _build_keyword_indexes(self) -> dict:
        """하이브리드 검색이 있으면 단어 분리 방식별 BM25 색인을 한 번 만든다."""
        tokenizer_names = {
            config["keyword_tokenizer"]
            for config in self.retrieval_configs.values()
            if config["method"] in {"hybrid", "hybrid_rerank"}
        }
        if not tokenizer_names:
            return {}

        chunks_path = (
            PROCESSED_DIR
            / self.config["dataset"]
            / self.config["chunk_version"]
            / "chunks.jsonl"
        )
        chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return {
            name: BM25Index(chunks, make_tokenizer(name))
            for name in tokenizer_names
        }

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

    def retrieve(self, question: str, retrieval_mode: str) -> dict:
        """질문을 검색하고 선택한 방식으로 후보를 정렬한 뒤 문맥을 채운다."""
        retrieval_config = self.retrieval_configs[retrieval_mode]
        method = retrieval_config["method"]
        query_vector = self.embedder.embed_query(question)

        if method == "dense":
            candidates = search(
                self.client,
                self.collection,
                query_vector,
                retrieval_config["candidate_count"],
                exact=retrieval_config["exact"],
            )
        else:
            dense = search(
                self.client,
                self.collection,
                query_vector,
                retrieval_config["dense_candidate_count"],
                exact=retrieval_config["exact"],
            )
            keyword = self.keyword_indexes[
                retrieval_config["keyword_tokenizer"]
            ].search(
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

        context, used = fill_context(
            candidates,
            self.counter,
            self.config["context_token_budget"],
        )
        return {
            "retrieval_version": retrieval_config["retrieval_version"],
            "score_type": SCORE_TYPES[method],
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
    ) -> dict:
        """질문 하나를 검색부터 답변까지 처리한다. 요청 하나가 Langfuse trace 하나가 된다.

        source: service(서비스 질문) 또는 evaluation(평가 실행)
        extra_metadata: 평가 시 실행 ID·질문 ID 등
        """
        retrieval_mode = (
            retrieval_mode or self.config["default_retrieval_mode"]
        )
        request_id = request_id or str(uuid.uuid4())
        retrieval_version = self.retrieval_configs[
            retrieval_mode
        ]["retrieval_version"]
        metadata = {
            "source": source,
            "request_id": request_id,
            "dataset": self.config["dataset"],
            "chunk_version": self.config["chunk_version"],
            "embedding": self.embedder.model_code,
            "prompt": self.generator.prompt_version,
            "retrieval": retrieval_version,
            "retrieval_mode": retrieval_mode,
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
                trace_name="rag_query",
                tags=[source, retrieval_mode],
                metadata={
                    key: str(value)
                    for key, value in metadata.items()
                },
            ):
                with self.langfuse.start_as_current_observation(
                    name="retrieval",
                    as_type="retriever",
                    input={"question": question},
                ) as span:
                    retrieval = self.retrieve(question, retrieval_mode)
                    retrieval_seconds = time.perf_counter() - start
                    retrieval_record = {
                        "version": retrieval["retrieval_version"],
                        "collection": self.collection,
                        "score_type": retrieval["score_type"],
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

            output = {
                key: result[key]
                for key in (
                    "status",
                    "answer",
                    "reason",
                    "citations",
                )
            }
            if result["status"] == "error":
                root.update(
                    output=output,
                    level="ERROR",
                    status_message=result["error"],
                )
            else:
                root.update(output=output)

        return {
            "request_id": request_id,
            "trace_id": trace_id,
            "retrieval_mode": retrieval_mode,
            **result,
            "retrieval": retrieval_record,
            "latency_seconds": round(
                time.perf_counter() - start,
                3,
            ),
        }

    def flush(self) -> None:
        """남아 있는 Langfuse 기록을 보낸다. 서버가 꺼질 때와 스크립트가 끝날 때 부른다."""
        self.langfuse.flush()