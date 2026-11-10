"""문맥 토큰 한도 결정을 위한 측정: 페이지 토큰 수와 질문별 근거 수."""

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # backend 모듈을 불러오기 위해 저장소 루트를 경로에 추가

from backend.core.token_counter import HuggingFaceCounter, TiktokenCounter
from backend.ingestion.extract import extract_folder


def percentile(values: list[int], p: int) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def summarize(values: list[int], p: int) -> dict:
    return {"min": min(values), "median": statistics.median(values), f"p{p}": percentile(values, p), "max": max(values)}


def count_evidence(evaluation_dir: Path, file_names: list[str]) -> list[dict]:
    """질문마다 등록된 근거의 고유 페이지 수와 고유 문서 수를 센다."""
    rows = []
    for file_name in file_names:
        items = json.loads((evaluation_dir / file_name).read_text(encoding="utf-8"))
        for item in items:
            evidences = [e for group in item["evidence_groups"] for e in group]
            rows.append({
                "id": item["id"],
                "expected_action": item["expected_action"],
                "evidence_page_count": len({(e["book_id"], e["page"]) for e in evidences}),
                "evidence_doc_count": len({e["book_id"] for e in evidences}),
            })
    return rows


def choose_context_page_count(evidence_page_counts: list[int], coverage: float) -> int:
    """근거 페이지 수가 n 이하인 질문의 비율이 coverage 이상이 되는 가장 작은 n."""
    for n in range(1, max(evidence_page_counts) + 1):
        if sum(c <= n for c in evidence_page_counts) / len(evidence_page_counts) >= coverage:
            return n
    return max(evidence_page_counts)


def run(config: dict, dataset: str, timestamp: str) -> None:
    """데이터셋 하나를 측정하고, 데이터셋마다 실행 ID와 결과 파일을 따로 남긴다."""
    paths = {key: ROOT / value.format(dataset=dataset) for key, value in config["paths"].items()}
    p = config["percentile"]
    run_id = f"e01_c0_p0_r0_{timestamp}"

    # 1. 페이지 추출과 두 기준의 토큰 측정
    pages, failed_files = extract_folder(paths["documents"])
    generation_counter = TiktokenCounter(config["generation_tokenizer"]["model"], config["generation_tokenizer"]["fallback"])
    embedding_config = config["embedding_tokenizer"]
    embedding_counter = HuggingFaceCounter(embedding_config["model"], embedding_config["prefix"])

    page_rows = [
        {
            "book_id": page.book_id,
            "page": page.page,
            "char_count": len(page.text),
            "generation_input_token_count": generation_counter.count(page.text),
            "embedding_input_token_count": embedding_counter.count(page.text),
        }
        for page in pages
    ]
    generation_input_token_counts = [r["generation_input_token_count"] for r in page_rows]
    embedding_input_token_counts = [r["embedding_input_token_count"] for r in page_rows]
    extracted_page_count = len(pages)
    over_limit_page_count = sum(t > embedding_config["max_tokens"] for t in embedding_input_token_counts)

    # 2. 질문별 근거 수 (근거가 없는 질문은 N 산정에서 제외)
    question_rows = count_evidence(paths["evaluation"], config["evaluation_files"])
    evidence_question_rows = [r for r in question_rows if r["evidence_page_count"] > 0]
    evidence_page_counts = [r["evidence_page_count"] for r in evidence_question_rows]

    # 3. N(문맥에 담을 페이지 수)과 문맥 토큰 한도
    context_page_count = choose_context_page_count(evidence_page_counts, config["question_coverage"])
    context_token_budget = math.ceil(percentile(generation_input_token_counts, p) * context_page_count)

    summary = {
        "run_id": run_id,
        "dataset": dataset,
        "extraction": {"extracted_page_count": extracted_page_count, "failed_files": failed_files},
        "generation_tokenizer": {"name": generation_counter.name, "approximate": generation_counter.is_approximate},
        "generation_input_token_count": summarize(generation_input_token_counts, p),
        "embedding_input_token_count": summarize(embedding_input_token_counts, p),
        "embedding_input_limit": {
            "max_tokens": embedding_config["max_tokens"],
            "extracted_page_count": extracted_page_count,
            "over_limit_page_count": over_limit_page_count,
            "over_limit_page_ratio": round(over_limit_page_count / extracted_page_count, 3),
        },
        "chars_per_embedding_input_token": round(sum(r["char_count"] for r in page_rows) / sum(embedding_input_token_counts), 2),
        "questions": {
            "question_count": len(question_rows),
            "no_evidence_question_count": len(question_rows) - len(evidence_question_rows),
        },
        "evidence_page_count_distribution": dict(sorted(Counter(evidence_page_counts).items())),
        "evidence_doc_count_distribution": dict(sorted(Counter(r["evidence_doc_count"] for r in evidence_question_rows).items())),
        "context_page_count": context_page_count,
        "context_token_budget": context_token_budget,
    }

    paths["results"].mkdir(parents=True, exist_ok=True)
    result = {"summary": summary, "pages": page_rows, "questions": question_rows}
    (paths["results"] / f"{run_id}_token_budget.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{dataset}: 페이지 {extracted_page_count}쪽, N={context_page_count}, "
          f"문맥 토큰 한도 {context_token_budget}")


def main() -> None:
    """설정의 datasets를 차례로 실행한다. --dataset을 주면 그 하나만 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="하나만 실행할 때 데이터셋 이름(생략하면 설정의 datasets 전체)")
    args = parser.parse_args()

    config = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text(encoding="utf-8"))
    datasets = [args.dataset] if args.dataset else config["datasets"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    for dataset in datasets:
        run(config, dataset, timestamp)


if __name__ == "__main__":
    main()