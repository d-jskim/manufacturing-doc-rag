"""세 검색 방식에서 k=2·3·5의 검색 지표를 비교한다."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from backend.pipeline import RAGPipeline
from evaluation.retrieval import score_question, summarize_scores

DATASET = "manufacturing_basic"
QUESTION_PATH = (
    ROOT
    / "data"
    / "raw"
    / DATASET
    / "evaluation"
    / "development.json"
)
RESULTS_DIR = (
    ROOT
    / "experiments"
    / "04_retrieval"
    / "results"
    / DATASET
)

MODES = ("naive", "hybrid", "rerank")
K_VALUES = (2, 3, 5)


def mean(values: list[float]) -> float | None:
    """값이 있으면 평균을 소수점 넷째 자리까지 반환한다."""
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def main() -> None:
    questions = json.loads(
        QUESTION_PATH.read_text(encoding="utf-8")
    )
    pipeline = RAGPipeline()

    missing_modes = [
        mode
        for mode in MODES
        if mode not in pipeline.retrieval_modes
    ]
    if missing_modes:
        raise ValueError(
            f"활성 검색 방식에 없음: {missing_modes}"
        )

    rows = {
        mode: {
            f"k{k}": []
            for k in K_VALUES
        }
        for mode in MODES
    }
    retrieval_seconds = {
        mode: []
        for mode in MODES
    }

    for index, item in enumerate(questions, start=1):
        print(
            f"[{index}/{len(questions)}] "
            f"{item['id']}: {item['question']}"
        )

        for mode in MODES:
            started = time.perf_counter()
            retrieval = pipeline.retrieve(
                item["question"],
                mode,
            )
            elapsed = time.perf_counter() - started
            retrieval_seconds[mode].append(elapsed)

            candidates = retrieval["candidates"]

            for k in K_VALUES:
                row = score_question(
                    item,
                    candidates[:k],
                )
                rows[mode][f"k{k}"].append(
                    {
                        "question": item["question"],
                        **row,
                    }
                )

    summary = {
        mode: {
            f"k{k}": summarize_scores(
                rows[mode][f"k{k}"]
            )
            for k in K_VALUES
        }
        for mode in MODES
    }

    for mode in MODES:
        summary[mode]["retrieval_seconds"] = {
            "mean": mean(retrieval_seconds[mode]),
            "count": len(retrieval_seconds[mode]),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = (
        f"e04_"
        f"{pipeline.config['chunk_version']}_"
        f"{pipeline.embedder.model_code}_"
        f"{pipeline.generator.prompt_version}_"
        f"k235_"
        f"{timestamp}"
    )

    result = {
        "run_id": run_id,
        "dataset": DATASET,
        "question_source": str(
            QUESTION_PATH.relative_to(ROOT)
        ),
        "question_count": len(questions),
        "retrieval_modes": {
            mode: pipeline.retrieval_configs[mode]
            for mode in MODES
        },
        "k_values": list(K_VALUES),
        "summary": summary,
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{run_id}.json"
    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"결과 저장: {result_path}")


if __name__ == "__main__":
    main()