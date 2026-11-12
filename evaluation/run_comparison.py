"""세 검색 방식으로 일반 평가 질문을 API에 보내고 운영 지표를 저장한다."""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import (
    DATASETS,
    DEFAULT_DATASET,
    evaluation_path,
    load_json,
    relative_path,
    results_dir,
    write_json,
)

DEFAULT_MODES = ("naive", "hybrid", "rerank")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


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
    return parser.parse_args()


def make_run_id(active_config: dict, question_set: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    chunk = active_config["chunk_version"]
    embedding = active_config["embedding"]
    prompt = active_config["prompt"]
    return f"e04_{chunk}_{embedding}_{prompt}_{question_set}_all_{timestamp}"


def load_questions(question_path: Path, limit: int | None) -> list[dict]:
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    if limit is not None:
        return questions[:limit]
    return questions


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def load_active_config(
    client: httpx.Client,
    base_url: str,
    expected_dataset: str,
    modes: list[str],
) -> dict:
    response = client.get(f"{base_url.rstrip('/')}/health")
    response.raise_for_status()
    health = response.json()
    active_dataset = health.get("dataset")

    if active_dataset != expected_dataset:
        raise RuntimeError(
            f"데이터셋 불일치: 실행={expected_dataset}, API={active_dataset}"
        )

    active_modes = health.get("retrieval_modes", {})
    unavailable = [mode for mode in modes if mode not in active_modes]
    if unavailable:
        raise RuntimeError(f"API에 없는 검색 방식: {', '.join(unavailable)}")

    return {key: value for key, value in health.items() if key != "status"}


def summarize(records: list[dict], modes: list[str]) -> dict:
    summary = {}

    for mode in modes:
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
    dataset: str,
    question_set: str,
    question_path: Path,
    modes: list[str],
    base_url: str,
    timeout: float,
    active_config: dict,
    question_count: int,
    records: list[dict],
) -> None:
    write_json(
        result_path,
        {
            "run_id": run_id,
            "dataset": dataset,
            "active_config": active_config,
            "question_set": question_set,
            "question_source": relative_path(question_path),
            "question_count": question_count,
            "requested_modes": modes,
            "base_url": base_url,
            "timeout_seconds": timeout,
            "summary": summarize(records, modes),
            "records": records,
        },
    )


def validate_resume(
    previous: dict,
    active_config: dict,
    question_path: Path,
    modes: list[str],
) -> None:
    checks = {
        "active_config": active_config,
        "question_source": relative_path(question_path),
        "requested_modes": modes,
    }
    mismatches = [key for key, value in checks.items() if previous.get(key) != value]
    if mismatches:
        raise RuntimeError(f"기존 실행 조건 불일치: {', '.join(mismatches)}")


def run_comparison(
    dataset: str = DEFAULT_DATASET,
    question_set: str = "development",
    base_url: str = DEFAULT_BASE_URL,
    modes: list[str] | None = None,
    limit: int | None = None,
    run_id: str | None = None,
    timeout: float = 180.0,
) -> Path:
    modes = list(modes or DEFAULT_MODES)
    question_path = evaluation_path(dataset, question_set)
    questions = load_questions(question_path, limit)
    result_dir = results_dir(dataset)
    result_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=timeout) as client:
        active_config = load_active_config(client, base_url, dataset, modes)
        run_id = run_id or make_run_id(active_config, question_set)
        result_path = result_dir / f"{run_id}.json"

        if result_path.exists():
            previous = load_json(result_path)
            validate_resume(previous, active_config, question_path, modes)
            records = previous["records"]
        else:
            records = []

        completed = {
            (record["question_id"], record["retrieval_mode"])
            for record in records
        }

        print(f"run_id: {run_id}")
        print(f"result_path: {result_path}")
        print(f"requests: {len(questions)} questions × {len(modes)} modes")

        for question in questions:
            for mode in modes:
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
                        f"{base_url.rstrip('/')}/query",
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
                    dataset,
                    question_set,
                    question_path,
                    modes,
                    base_url,
                    timeout,
                    active_config,
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
        dataset,
        question_set,
        question_path,
        modes,
        base_url,
        timeout,
        active_config,
        len(questions),
        records,
    )
    print(json.dumps(summarize(records, modes), ensure_ascii=False, indent=2))
    print(f"완료: {result_path}")
    return result_path


def main() -> None:
    args = parse_args()
    run_comparison(
        dataset=args.dataset,
        question_set=args.question_set,
        base_url=args.base_url,
        modes=args.modes,
        limit=args.limit,
        run_id=args.run_id,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
