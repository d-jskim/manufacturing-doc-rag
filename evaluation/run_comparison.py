"""세 검색 방식으로 일반 평가 질문을 API에 보내고 운영 지표를 저장한다."""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_MODES = ("naive", "hybrid", "rerank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--run-id",
        default=None,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    return parser.parse_args()


def make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"e04_c1_m1_p1_all_{timestamp}"


def load_questions(limit: int | None) -> list[dict]:
    questions = json.loads(
        QUESTION_PATH.read_text(encoding="utf-8")
    )
    if limit is not None:
        return questions[:limit]
    return questions


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def summarize(records: list[dict]) -> dict:
    summary = {}

    for mode in DEFAULT_MODES:
        mode_records = [
            record
            for record in records
            if record["retrieval_mode"] == mode
        ]
        statuses = Counter(
            record["response"].get(
                "status",
                "rejected",
            )
            for record in mode_records
        )
        normal_records = [
            record
            for record in mode_records
            if record["response"].get("status")
            in {
                "answered",
                "needs_clarification",
                "abstained",
            }
        ]
        system_error_count = sum(
            record["response"].get("status") == "error"
            for record in mode_records
        )
        latencies = [
            record["response"]["metrics"]["latency_seconds"]
            for record in normal_records
            if record["response"].get("metrics")
        ]
        usages = [
            record["response"]["metrics"]["usage"]
            for record in mode_records
            if record["response"].get("metrics")
            and record["response"]["metrics"].get("usage")
        ]

        summary[mode] = {
            "request_count": len(mode_records),
            "status_counts": dict(statuses),
            "normal_latency_seconds": {
                "mean": mean(latencies),
                "count": len(latencies),
            },
            "system_error_count": system_error_count,
            "system_error_rate": (
                round(
                    system_error_count
                    / len(mode_records),
                    4,
                )
                if mode_records
                else None
            ),
            "token_usage": {
                "input_tokens": sum(
                    usage["input_tokens"]
                    for usage in usages
                ),
                "output_tokens": sum(
                    usage["output_tokens"]
                    for usage in usages
                ),
                "confirmed_request_count": len(usages),
            },
        }

    return summary


def save_result(
    result_path: Path,
    run_id: str,
    args: argparse.Namespace,
    question_count: int,
    records: list[dict],
) -> None:
    result = {
        "run_id": run_id,
        "dataset": DATASET,
        "question_source": str(
            QUESTION_PATH.relative_to(ROOT)
        ),
        "question_count": question_count,
        "requested_modes": args.modes,
        "base_url": args.base_url,
        "timeout_seconds": args.timeout,
        "summary": summarize(records),
        "records": records,
    }
    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    run_id = args.run_id or make_run_id()
    questions = load_questions(args.limit)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path = RESULTS_DIR / f"{run_id}.json"

    if result_path.exists():
        previous = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        records = previous["records"]
    else:
        records = []

    completed = {
        (
            record["question_id"],
            record["retrieval_mode"],
        )
        for record in records
    }

    print(f"run_id: {run_id}")
    print(f"result_path: {result_path}")
    print(
        f"requests: "
        f"{len(questions)} questions × "
        f"{len(args.modes)} modes"
    )

    with httpx.Client(
        timeout=args.timeout,
    ) as client:
        for question in questions:
            for mode in args.modes:
                key = (question["id"], mode)
                if key in completed:
                    print(
                        f"skip: {question['id']} / {mode}"
                    )
                    continue

                payload = {
                    "question": question["question"],
                    "retrieval_mode": mode,
                    "source": "evaluation",
                    "run_id": run_id,
                    "question_id": question["id"],
                }
                started = time.perf_counter()

                try:
                    response = client.post(
                        f"{args.base_url}/query",
                        json=payload,
                    )
                    try:
                        body = response.json()
                    except ValueError:
                        body = {
                            "status": "error",
                            "reason": "invalid_json_response",
                            "raw_response": response.text,
                        }
                    http_status = response.status_code
                except Exception as exc:
                    http_status = 0
                    body = {
                        "status": "error",
                        "reason": (
                            f"client_error:"
                            f"{type(exc).__name__}"
                        ),
                    }

                record = {
                    "question_id": question["id"],
                    "question": question["question"],
                    "retrieval_mode": mode,
                    "http_status": http_status,
                    "client_latency_seconds": round(
                        time.perf_counter() - started,
                        3,
                    ),
                    "response": body,
                }
                records.append(record)

                save_result(
                    result_path,
                    run_id,
                    args,
                    len(questions),
                    records,
                )

                print(
                    f"{question['id']} / {mode}: "
                    f"HTTP {http_status}, "
                    f"{body.get('status')}"
                )

    save_result(
        result_path,
        run_id,
        args,
        len(questions),
        records,
    )
    print(
        json.dumps(
            summarize(records),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"완료: {result_path}")


if __name__ == "__main__":
    main()