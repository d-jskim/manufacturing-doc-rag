"""평가 실행 결과를 Streamlit에서 선택할 수 있는 보고서로 묶는다."""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import (
    DATASETS,
    DEFAULT_DATASET,
    ROOT,
    load_answer_result,
    load_json,
    metric_run_id,
    results_dir,
)

CONFIG_DIRS = {
    "chunk": "chunking",
    "embedding": "embedding",
    "prompt": "prompts",
    "retrieval": "retrieval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("draft", "approved"), default="approved")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_definition(identifier: str, definition_type: str) -> dict:
    directory = CONFIG_DIRS[definition_type]
    config = load_yaml(ROOT / "configs" / directory / f"{identifier}.yaml")
    return {
        "type": definition_type,
        "label": config["label"],
        "description": config["description"],
    }


def created_at_from_run_id(run_id: str) -> tuple[str, str]:
    match = re.search(r"_(\d{8})_(\d{4})$", run_id)
    if match is None:
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M"), now.strftime("%Y%m%d_%H%M")

    value = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M")
    return value.strftime("%Y-%m-%d %H:%M"), value.strftime("%Y%m%d_%H%M")


def validate_metric_result(path: Path, source_run_id: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"평가 결과가 없습니다: {path}")
    result = load_json(path)
    if result.get("source_run_id") != source_run_id:
        raise ValueError(f"원본 실행 ID가 일치하지 않습니다: {path}")
    return result


def build_report(
    dataset: str,
    run_id: str,
    status: str = "approved",
) -> Path:
    answer_path, answers, active_config = load_answer_result(dataset, run_id)
    result_dir = results_dir(dataset)
    retrieval_path = result_dir / f"{metric_run_id(run_id, 'retrieval')}.json"
    faithfulness_path = result_dir / f"{metric_run_id(run_id, 'faithfulness')}.json"
    judge_path = result_dir / f"{metric_run_id(run_id, 'judge')}.json"

    validate_metric_result(retrieval_path, run_id)
    faithfulness = validate_metric_result(faithfulness_path, run_id)
    judge = validate_metric_result(judge_path, run_id)

    modes = answers["requested_modes"]
    retrieval_modes = active_config["retrieval_modes"]
    mode_ids = [retrieval_modes[mode] for mode in modes]
    definitions = {
        active_config["chunk_version"]: load_definition(
            active_config["chunk_version"],
            "chunk",
        ),
        active_config["embedding"]: load_definition(
            active_config["embedding"],
            "embedding",
        ),
        active_config["prompt"]: load_definition(
            active_config["prompt"],
            "prompt",
        ),
        **{
            identifier: load_definition(identifier, "retrieval")
            for identifier in mode_ids
        },
    }
    mode_labels = [definitions[identifier]["label"] for identifier in mode_ids]
    prompt_config = load_yaml(
        ROOT / "configs" / "prompts" / f"{active_config['prompt']}.yaml"
    )
    created_at, timestamp = created_at_from_run_id(run_id)
    question_set = answers.get("question_set", "development")
    report_id = "_".join(
        [
            "e04",
            "retrieval",
            active_config["chunk_version"],
            active_config["embedding"],
            active_config["prompt"],
            "-".join(mode_ids),
            question_set,
            timestamp,
        ]
    )
    report = {
        "report_id": report_id,
        "report_label": f"{'·'.join(mode_labels)} · {created_at}",
        "source_run_id": run_id,
        "experiment": "04_retrieval",
        "dataset": dataset,
        "created_at": created_at,
        "status": status,
        "title": "제조 문서 RAG 성능 평가",
        "dataset_label": dataset.replace("_", " ").title(),
        "comparison_label": " / ".join(mode_labels),
        "purpose": (
            f"{question_set} 질문 {answers['question_count']}개 기준 "
            "RAG 구성별 품질 및 운영 지표 비교"
        ),
        "result_files": {
            "retrieval": retrieval_path.name,
            "answers": answer_path.name,
            "faithfulness": faithfulness_path.name,
            "judge": judge_path.name,
        },
        "settings": {
            "chunk_version": active_config["chunk_version"],
            "embedding": active_config["embedding"],
            "prompt_version": active_config["prompt"],
            "retrieval_modes": {
                mode: retrieval_modes[mode] for mode in modes
            },
            "final_k": active_config["final_k"],
            "context_token_budget": active_config["context_token_budget"],
            "generation_model": prompt_config["generation"]["model"],
            "evaluation_questions": answers["question_count"],
            "question_set": question_set,
            "faithfulness_errors": sum(
                item["error_count"] for item in faithfulness["summary"].values()
            ),
            "judge_errors": sum(
                item["evaluation_error_count"]
                for item in judge["summary"].values()
            ),
        },
        "id_definitions": definitions,
    }

    report_dir = result_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report_id}_report.yaml"
    report_path.write_text(
        yaml.safe_dump(report, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"완료: {report_path}")
    return report_path


def main() -> None:
    args = parse_args()
    build_report(args.dataset, args.run_id, args.status)


if __name__ == "__main__":
    main()
