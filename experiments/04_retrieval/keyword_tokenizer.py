"""BM25 키워드 검색에서 글을 자르는 방식(띄어쓰기 vs Kiwi)에 따라 근거를 찾는 정도가 달라지는지 비교한다.
하이브리드 검색(r2)의 키워드 검색 설정을 정하기 위한 사전 확인이다."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # backend·evaluation 모듈을 불러오기 위해 저장소 루트를 경로에 추가

from backend.core.token_counter import TiktokenCounter
from backend.retrieval.context import fill_context
from backend.retrieval.keyword import BM25Index, make_tokenizer
from evaluation.retrieval import score_question, summarize_scores

DATASET = "manufacturing_basic"
CHUNK_VERSION = "c1"          # 청킹 실험 결과
CONTEXT_TOKEN_BUDGET = 1582   # 토큰 한도 실험 결과
CANDIDATE_COUNT = 20          # Naive 검색(r1)과 같음
TOKENIZERS = ("whitespace", "kiwi")


def main() -> None:
    chunks_path = ROOT / "data" / "processed" / DATASET / CHUNK_VERSION / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    questions = json.loads((ROOT / "data" / "raw" / DATASET / "evaluation" / "development.json")
                           .read_text(encoding="utf-8"))
    counter = TiktokenCounter("gpt-6-luna", "o200k_base")

    summary, rows_by_tokenizer = {}, {}
    for name in TOKENIZERS:
        start = time.perf_counter()
        index = BM25Index(chunks, make_tokenizer(name))
        build_seconds = time.perf_counter() - start

        rows = []
        for item in questions:
            context, _ = fill_context(index.search(item["question"], CANDIDATE_COUNT), counter, CONTEXT_TOKEN_BUDGET)
            rows.append(score_question(item, context))

        summary[name] = {**summarize_scores(rows), "index_build_seconds": round(build_seconds, 2)}
        rows_by_tokenizer[name] = rows
        metrics = summary[name]
        print(f"{name}: Hit@k {metrics['hit_at_k']}, Recall@k {metrics['recall_at_k']}, "
              f"근거 문장 {metrics['quote_recall_at_k']}, 평균 k {metrics['mean_k']}, "
              f"색인 {metrics['index_build_seconds']}초")

    results_dir = ROOT / "experiments" / "04_retrieval" / "results" / DATASET
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result = {
        "summary": {
            "dataset": DATASET, "chunk_version": CHUNK_VERSION, "retrieval": "bm25_only",
            "context_token_budget": CONTEXT_TOKEN_BUDGET, "candidate_count": CANDIDATE_COUNT,
            "tokenizers": summary,
        },
        "questions": rows_by_tokenizer,
    }
    (results_dir / f"keyword_tokenizer_{timestamp}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()