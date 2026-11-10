"""질문 하나를 검색부터 답변 생성까지 실행해 흐름을 점검한다."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)  # OPENAI_API_KEY

from backend.core.token_counter import TiktokenCounter
from backend.generation.generator import Generator
from backend.retrieval.context import fill_context
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import search


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--dataset", default="manufacturing_basic")
    parser.add_argument("--chunk", default="c1")
    parser.add_argument("--token-budget", type=int, default=1582)
    args = parser.parse_args()

    embedding_config = load_yaml(ROOT / "configs" / "embedding" / "m1.yaml")
    retrieval_config = load_yaml(ROOT / "configs" / "retrieval" / "r1.yaml")
    prompt_config = load_yaml(ROOT / "configs" / "prompts" / "p1.yaml")

    # 1. 검색: 질문 임베딩 → Qdrant 후보 → 문맥 한도까지 담기
    embedder = Embedder(embedding_config)
    client = QdrantClient(url="http://127.0.0.1:6333")
    collection = f"{args.dataset}_{args.chunk}_{embedder.model_code}"
    candidates = search(client, collection, embedder.embed_query(args.question),
                        retrieval_config["candidate_count"], exact=retrieval_config["exact"])
    counter = TiktokenCounter(prompt_config["generation"]["model"], "o200k_base")
    context, used = fill_context(candidates, counter, args.token_budget)
    print(f"검색: {collection}, 문맥 청크 {len(context)}개, {used}토큰")

    # 2. 답변 생성
    result = Generator(prompt_config).generate(args.question, context)
    print(json.dumps({k: v for k, v in result.items() if k != "final_context"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()