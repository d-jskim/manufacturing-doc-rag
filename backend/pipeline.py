"""운영 설정(configs/active.yaml)에 따라 검색과 답변 생성을 한 번에 실행하고, 요청마다 Langfuse에 기록한다.
API와 평가 실험이 함께 쓴다."""

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
from backend.retrieval.vector_store import search

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class RAGPipeline:
    def __init__(self, active_config_path: Path = CONFIGS_DIR / "active.yaml"):
        """서버가 켜질 때 한 번만 만든다. 임베딩 모델과 Qdrant 연결을 미리 준비한다."""
        self.config = load_yaml(active_config_path)
        embedding_config = load_yaml(CONFIGS_DIR / "embedding" / f"{self.config['embedding']}.yaml")
        prompt_config = load_yaml(CONFIGS_DIR / "prompts" / f"{self.config['prompt']}.yaml")
        self.retrieval_configs = {
            mode: load_yaml(CONFIGS_DIR / "retrieval" / f"{version}.yaml")
            for mode, version in self.config["retrieval_modes"].items()
        }
        self.embedder = Embedder(embedding_config)
        self.client = QdrantClient(url=self.config["qdrant_url"])
        self.generator = Generator(prompt_config)
        self.counter = TiktokenCounter(prompt_config["generation"]["model"], "o200k_base")
        self.collection = f"{self.config['dataset']}_{self.config['chunk_version']}_{self.embedder.model_code}"
        self.langfuse = get_client()  # .env의 LANGFUSE_PUBLIC_KEY·SECRET_KEY·BASE_URL 사용

    @property
    def retrieval_modes(self) -> list[str]:
        return list(self.retrieval_configs)

    def retrieve(self, question: str, retrieval_mode: str) -> dict:
        """질문 임베딩 → 후보 검색 → 문맥 한도까지 담기."""
        retrieval_config = self.retrieval_configs[retrieval_mode]
        candidates = search(self.client, self.collection, self.embedder.embed_query(question),
                            retrieval_config["candidate_count"], exact=retrieval_config["exact"])
        context, used = fill_context(candidates, self.counter, self.config["context_token_budget"])
        return {"retrieval_version": retrieval_config["retrieval_version"], "candidates": candidates,
                "context": context, "context_token_count": used}

    def query(self, question: str, retrieval_mode: str | None = None, request_id: str | None = None,
              source: str = "service", extra_metadata: dict | None = None) -> dict:
        """질문 하나를 검색부터 답변까지 처리한다. 요청 하나가 Langfuse trace 하나가 된다.

        source: service(서비스 질문) 또는 evaluation(평가 실행)
        extra_metadata: 평가 시 실행 ID·질문 ID 등
        """
        retrieval_mode = retrieval_mode or self.config["default_retrieval_mode"]
        request_id = request_id or str(uuid.uuid4())  # 요청 ID는 서버가 만든다
        retrieval_version = self.retrieval_configs[retrieval_mode]["retrieval_version"]
        metadata = {
            "source": source, "request_id": request_id, "dataset": self.config["dataset"],
            "chunk_version": self.config["chunk_version"], "embedding": self.embedder.model_code,
            "prompt": self.generator.prompt_version, "retrieval": retrieval_version,
            "retrieval_mode": retrieval_mode, **(extra_metadata or {}),
        }
        trace_id = self.langfuse.create_trace_id(seed=request_id)  # 요청 ID로 trace를 찾을 수 있게 함
        start = time.perf_counter()

        with self.langfuse.start_as_current_observation(
            name="rag_query", as_type="span", input={"question": question},
            trace_context={"trace_id": trace_id},
        ) as root:
            with propagate_attributes(trace_name="rag_query", tags=[source, retrieval_mode],
                                      metadata={key: str(value) for key, value in metadata.items()}):
                # 1. 검색
                with self.langfuse.start_as_current_observation(
                    name="retrieval", as_type="retriever", input={"question": question},
                ) as span:
                    retrieval = self.retrieve(question, retrieval_mode)
                    retrieval_seconds = time.perf_counter() - start
                    retrieval_record = {
                        "version": retrieval["retrieval_version"],
                        "collection": self.collection,
                        "candidates": [{"chunk_id": c["chunk_id"], "score": round(c["score"], 4)}
                                       for c in retrieval["candidates"]],
                        "context_chunk_ids": [c["chunk_id"] for c in retrieval["context"]],
                        "context_token_count": retrieval["context_token_count"],
                        "seconds": round(retrieval_seconds, 3),
                    }
                    span.update(output=retrieval_record)

                # 2. 답변 생성(LLM 호출은 CallbackHandler가 실제 입출력·토큰을 기록)
                result = self.generator.generate(question, retrieval["context"], callbacks=[CallbackHandler()])
                result["generation_seconds"] = result.pop("latency_seconds")

            output = {key: result[key] for key in ("status", "answer", "reason", "citations")}
            if result["status"] == "error":
                root.update(output=output, level="ERROR", status_message=result["error"])
            else:
                root.update(output=output)

        return {
            "request_id": request_id,
            "trace_id": trace_id,
            "retrieval_mode": retrieval_mode,
            **result,
            "retrieval": retrieval_record,
            "latency_seconds": round(time.perf_counter() - start, 3),
        }

    def flush(self) -> None:
        """남아 있는 Langfuse 기록을 보낸다. 서버가 꺼질 때와 스크립트가 끝날 때 부른다."""
        self.langfuse.flush()