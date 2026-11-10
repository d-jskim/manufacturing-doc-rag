"""운영 설정(configs/active.yaml)에 따라 검색과 답변 생성을 한 번에 실행한다. API와 평가 실험이 함께 쓴다."""

import time
import uuid
from pathlib import Path

import yaml
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

    def query(self, question: str, retrieval_mode: str | None = None, request_id: str | None = None) -> dict:
        """질문 하나를 검색부터 답변까지 처리하고, 실행 기록을 함께 돌려준다."""
        retrieval_mode = retrieval_mode or self.config["default_retrieval_mode"]
        request_id = request_id or str(uuid.uuid4())  # 요청 ID는 서버가 만든다
        start = time.perf_counter()

        retrieval = self.retrieve(question, retrieval_mode)
        retrieval_seconds = time.perf_counter() - start
        result = self.generator.generate(question, retrieval["context"])
        result["generation_seconds"] = result.pop("latency_seconds")

        return {
            "request_id": request_id,
            "retrieval_mode": retrieval_mode,
            **result,
            "retrieval": {
                "version": retrieval["retrieval_version"],
                "collection": self.collection,
                "candidates": [{"chunk_id": c["chunk_id"], "score": round(c["score"], 4)}
                               for c in retrieval["candidates"]],
                "context_chunk_ids": [c["chunk_id"] for c in retrieval["context"]],
                "context_token_count": retrieval["context_token_count"],
                "seconds": round(retrieval_seconds, 3),
            },
            "latency_seconds": round(time.perf_counter() - start, 3),
        }