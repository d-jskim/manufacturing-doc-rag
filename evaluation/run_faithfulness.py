"""답변 생성에 실제 사용된 문맥으로 RAGAS Faithfulness를 평가한다."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import (
    DATASETS,
    DEFAULT_DATASET,
    EVALUATION_MODEL,
    ROOT,
    load_answer_result,
    load_chunks,
    load_json,
    metric_run_id,
    relative_path,
    results_dir,
    select_records,
    write_json,
)

load_dotenv(ROOT / ".env", override=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def summarize(rows: list[dict], modes: list[str]) -> dict:
    summary = {}

    for mode in modes:
        mode_rows = [row for row in rows if row["retrieval_mode"] == mode]
        scores = [
            row["faithfulness"]
            for row in mode_rows
            if row["faithfulness"] is not None
        ]
        summary[mode] = {
            "mean_faithfulness": (
                round(sum(scores) / len(scores), 4)
                if scores
                else None
            ),
            "evaluated_count": len(scores),
            "error_count": sum(
                row["faithfulness"] is None for row in mode_rows
            ),
        }

    return summary


def save_result(
    output_path: Path,
    answer_path: Path,
    active_config: dict,
    modes: list[str],
    rows: list[dict],
) -> None:
    write_json(
        output_path,
        {
            "run_id": output_path.stem,
            "source_run_id": answer_path.stem,
            "metric": "ragas_faithfulness",
            "model": EVALUATION_MODEL,
            "dataset": active_config["dataset"],
            "active_config": active_config,
            "answer_result": relative_path(answer_path),
            "summary": summarize(rows, modes),
            "rows": rows,
        },
    )


def run_faithfulness(
    dataset: str,
    run_id: str,
    limit: int | None = None,
) -> Path:
    answer_path, answer_result, active_config = load_answer_result(dataset, run_id)
    chunks = load_chunks(dataset, active_config["chunk_version"])
    modes = answer_result["requested_modes"]
    answer_records = select_records(answer_result["records"], limit)

    output_run_id = metric_run_id(run_id, "faithfulness")
    output_path = results_dir(dataset) / f"{output_run_id}.json"
    rows = load_json(output_path)["rows"] if output_path.exists() else []
    completed = {
        (row["question_id"], row["retrieval_mode"]) for row in rows
    }

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=180.0,
    )
    llm = llm_factory(
        model=EVALUATION_MODEL,
        provider="openai",
        client=client,
        seed=42,
        max_tokens=2048,
    )
    metric = Faithfulness(llm=llm)

    for record in answer_records:
        key = (record["question_id"], record["retrieval_mode"])
        if key in completed:
            continue

        response = record["response"]
        metrics = response.get("metrics") or {}
        context_chunk_ids = metrics.get("context_chunk_ids", [])
        context_token_count = metrics.get("context_token_count")
        started = time.perf_counter()

        try:
            if not context_chunk_ids:
                raise ValueError("실제 사용 문맥 정보가 없습니다.")

            contexts = [
                chunks[chunk_id]["text"]
                for chunk_id in context_chunk_ids
            ]
            answer = response["answer"]
            if response.get("reason"):
                answer += f"\n사유: {response['reason']}"

            result = metric.score(
                user_input=record["question"],
                response=answer,
                retrieved_contexts=contexts,
            )
            score = float(result.value)
            error = None
        except Exception as exc:
            score = None
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "question_id": record["question_id"],
                "retrieval_mode": record["retrieval_mode"],
                "status": response["status"],
                "context_chunk_ids": context_chunk_ids,
                "context_token_count": context_token_count,
                "faithfulness": score,
                "seconds": round(time.perf_counter() - started, 3),
                "error": error,
            }
        )
        save_result(output_path, answer_path, active_config, modes, rows)
        print(f"{key[0]} / {key[1]}: {score}")

    print(json.dumps(summarize(rows, modes), ensure_ascii=False, indent=2))
    print(f"완료: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    run_faithfulness(args.dataset, args.run_id, args.limit)


if __name__ == "__main__":
    main()
