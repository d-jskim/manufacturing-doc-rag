"""답변 생성부터 보고서까지 공통 평가 파이프라인을 실행한다."""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import DATASETS, DEFAULT_DATASET
from evaluation.report import build_report
from evaluation.run_comparison import DEFAULT_BASE_URL, DEFAULT_MODES, run_comparison
from evaluation.run_faithfulness import run_faithfulness
from evaluation.run_judge import run_judge
from evaluation.run_retrieval import run_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--question-set", default="development")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--report-status",
        choices=("draft", "approved"),
        default="approved",
    )
    return parser.parse_args()


def run_evaluation(
    dataset: str = DEFAULT_DATASET,
    question_set: str = "development",
    base_url: str = DEFAULT_BASE_URL,
    modes: list[str] | None = None,
    limit: int | None = None,
    run_id: str | None = None,
    timeout: float = 180.0,
    report_status: str = "approved",
) -> dict[str, Path]:
    answer_path = run_comparison(
        dataset=dataset,
        question_set=question_set,
        base_url=base_url,
        modes=modes,
        limit=limit,
        run_id=run_id,
        timeout=timeout,
    )
    source_run_id = answer_path.stem
    retrieval_path = run_retrieval(dataset, source_run_id, limit)
    faithfulness_path = run_faithfulness(dataset, source_run_id, limit)
    judge_path = run_judge(dataset, source_run_id, limit)
    report_path = build_report(dataset, source_run_id, report_status)
    return {
        "answers": answer_path,
        "retrieval": retrieval_path,
        "faithfulness": faithfulness_path,
        "judge": judge_path,
        "report": report_path,
    }


def main() -> None:
    args = parse_args()
    run_evaluation(
        dataset=args.dataset,
        question_set=args.question_set,
        base_url=args.base_url,
        modes=args.modes,
        limit=args.limit,
        run_id=args.run_id,
        timeout=args.timeout,
        report_status=args.report_status,
    )


if __name__ == "__main__":
    main()
