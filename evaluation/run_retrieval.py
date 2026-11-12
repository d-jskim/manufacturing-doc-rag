"""답변 생성에 실제 사용된 문맥으로 Hit@k와 Recall@k를 계산한다."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import (
    DATASETS,
    DEFAULT_DATASET,
    load_answer_result,
    load_chunks,
    load_json,
    metric_run_id,
    project_path,
    relative_path,
    results_dir,
    select_records,
    write_json,
)
from evaluation.retrieval import score_question, summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def run_retrieval(
    dataset: str,
    run_id: str,
    limit: int | None = None,
) -> Path:
    answer_path, answer_result, active_config = load_answer_result(dataset, run_id)
    ground_truth_path = project_path(answer_result["question_source"])
    ground_truth = {item["id"]: item for item in load_json(ground_truth_path)}
    chunks = load_chunks(dataset, active_config["chunk_version"])
    modes = answer_result["requested_modes"]
    records = select_records(answer_result["records"], limit)
    final_k = active_config["final_k"]
    k_key = f"k{final_k}"
    rows = {mode: {k_key: []} for mode in modes}
    retrieval_seconds = {mode: [] for mode in modes}

    for record in records:
        mode = record["retrieval_mode"]
        response = record["response"]
        metrics = response.get("metrics") or {}
        context_chunk_ids = metrics.get("context_chunk_ids", [])
        context = [chunks[chunk_id] for chunk_id in context_chunk_ids]
        score = score_question(ground_truth[record["question_id"]], context)
        rows[mode][k_key].append({"question": record["question"], **score})

        if metrics.get("retrieval_seconds") is not None:
            retrieval_seconds[mode].append(metrics["retrieval_seconds"])

    summary = {}
    for mode in modes:
        summary[mode] = {
            k_key: summarize_scores(rows[mode][k_key]),
            "retrieval_seconds": {
                "mean": mean(retrieval_seconds[mode]),
                "count": len(retrieval_seconds[mode]),
            },
        }

    output_run_id = metric_run_id(run_id, "retrieval")
    output_path = results_dir(dataset) / f"{output_run_id}.json"
    write_json(
        output_path,
        {
            "run_id": output_run_id,
            "source_run_id": run_id,
            "dataset": dataset,
            "active_config": active_config,
            "question_source": relative_path(ground_truth_path),
            "question_count": len({record["question_id"] for record in records}),
            "retrieval_modes": active_config["retrieval_modes"],
            "k_values": [final_k],
            "answer_result": relative_path(answer_path),
            "summary": summary,
            "rows": rows,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"완료: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    run_retrieval(args.dataset, args.run_id, args.limit)


if __name__ == "__main__":
    main()
