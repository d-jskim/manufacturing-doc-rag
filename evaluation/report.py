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


def build_result_analysis(
    modes: list[str],
    final_k: int,
    retrieval: dict,
    answers: dict,
    faithfulness: dict,
) -> dict:
    items = {}

    for mode in modes:
        search = retrieval["summary"][mode][f"k{final_k}"]
        service = answers["summary"][mode]
        faith = faithfulness["summary"][mode]["mean_faithfulness"]
        items[mode] = {
            "summary": (
                f"Hit@{final_k} {search['hit_at_k']:.2%}, "
                f"Recall@{final_k} {search['recall_at_k']:.2%}, "
                f"Faithfulness {faith:.4f}"
            ),
            "limitation": (
                f"시스템 오류 {service['system_error_count']}건, "
                f"평균 검색 {retrieval['summary'][mode]['retrieval_seconds']['mean']:.3f}초, "
                f"평균 응답 {service['normal_latency_seconds']['mean']:.3f}초"
            ),
        }

    return {"title": "Result Analysis", "items": items}


def build_failure_analysis(
    modes: list[str],
    final_k: int,
    active_config: dict,
    manifest: dict,
    retrieval: dict,
    answers: dict,
    faithfulness: dict,
    judge: dict,
) -> dict:
    items = []
    chunk_count = manifest.get("chunk_count", 0)
    over_limit = manifest.get("over_limit_chunk_count", 0)

    if over_limit:
        ratio = over_limit / chunk_count if chunk_count else 0
        max_tokens = manifest.get("chunk_config", {}).get("max_encoded_tokens")
        items.append(
            {
                "failure_id": "e04_embedding_input_limit",
                "type": "data_issue",
                "title": "청크의 임베딩 입력 한도 초과",
                "status": "confirmed",
                "affected_versions": [
                    active_config["chunk_version"],
                    active_config["embedding"],
                ],
                "expectation": "Basic Baseline을 그대로 적용해 데이터 난이도 증가에 따른 입력 한도 초과 규모를 확인함",
                "observed": f"전체 {chunk_count:,}개 중 {over_limit:,}개({ratio:.1%})가 {max_tokens}토큰 입력 한도를 초과함",
                "possible_cause": "페이지 단위 청크 길이가 임베딩 모델의 최대 입력 길이보다 긴 문서가 포함됨",
                "evidence": [
                    f"청킹 버전: {active_config['chunk_version']}",
                    f"임베딩 모델: {active_config['embedding']}",
                    f"입력 한도 초과 청크: {over_limit:,}개",
                ],
                "follow_up": "Baseline 결과를 보존한 뒤 C2·C3 또는 상위 문맥을 보강한 분할 청킹으로 재실험함",
            }
        )

    error_modes = [
        mode for mode in modes if answers["summary"][mode]["system_error_count"]
    ]
    if error_modes:
        items.append(
            {
                "failure_id": "e04_generation_errors",
                "type": "execution_error",
                "title": "답변 생성 시스템 오류 발생",
                "status": "confirmed",
                "affected_versions": [active_config["retrieval_modes"][mode] for mode in error_modes],
                "expectation": "모든 평가 질문에서 정상 응답을 생성함",
                "observed": ", ".join(
                    f"{mode} {answers['summary'][mode]['system_error_count']}건"
                    for mode in error_modes
                ),
                "possible_cause": "검색·재정렬·생성 단계의 처리 시간과 외부 모델 호출 오류가 누적된 것으로 추정함",
                "evidence": [
                    f"{mode} 오류율 {answers['summary'][mode]['system_error_rate']:.2%}"
                    for mode in error_modes
                ],
                "follow_up": "오류 유형별 로그를 분리하고 동일 질문을 재실행해 재현 여부를 확인함",
            }
        )

    if "rerank" in modes:
        rerank_seconds = retrieval["summary"]["rerank"]["retrieval_seconds"]["mean"]
        reference_mode = "naive" if "naive" in modes else modes[0]
        reference_seconds = retrieval["summary"][reference_mode]["retrieval_seconds"]["mean"]
        if rerank_seconds > max(5, reference_seconds * 10):
            items.append(
                {
                    "failure_id": "e04_rerank_latency",
                    "type": "execution_error",
                    "title": "Reranking 검색 지연 증가",
                    "status": "confirmed",
                    "affected_versions": [active_config["retrieval_modes"]["rerank"]],
                    "expectation": "Reranking의 품질 개선 효과를 서비스 응답이 가능한 시간 안에서 확보함",
                    "observed": f"Reranking 평균 검색 {rerank_seconds:.3f}초, {reference_mode} {reference_seconds:.3f}초",
                    "possible_cause": "후보 청크별 Cross-encoder 재정렬 비용이 데이터 규모와 함께 증가한 것으로 추정함",
                    "evidence": [
                        f"Reranking 검색 시간이 {reference_mode}보다 {rerank_seconds / reference_seconds:.1f}배 김"
                    ],
                    "follow_up": "후보 수·배치 처리·재정렬 모델 실행 장치를 분리해 지연 원인을 측정함",
                }
            )

    recall = {
        mode: retrieval["summary"][mode][f"k{final_k}"]["recall_at_k"]
        for mode in modes
    }
    faith = {
        mode: faithfulness["summary"][mode]["mean_faithfulness"]
        for mode in modes
        if faithfulness["summary"][mode]["mean_faithfulness"] is not None
    }
    if faith:
        best_recall = max(recall, key=recall.get)
        worst_faith = min(faith, key=faith.get)
        if best_recall == worst_faith and len(modes) > 1:
            items.append(
                {
                    "failure_id": "e04_retrieval_answer_mismatch",
                    "type": "hypothesis_miss",
                    "title": "검색 개선과 답변 근거 충실도 불일치",
                    "status": "hypothesis",
                    "affected_versions": [active_config["retrieval_modes"][best_recall]],
                    "expectation": "Recall이 높은 검색 방식이 더 높은 Faithfulness를 보일 것으로 예상함",
                    "observed": f"{best_recall} Recall@{final_k} {recall[best_recall]:.2%}, Faithfulness {faith[best_recall]:.4f}",
                    "possible_cause": "관련 청크를 더 찾았더라도 조합된 문맥의 경쟁 정보나 생성 단계가 답변 근거성을 낮췄을 수 있음",
                    "evidence": [
                        f"최고 Recall 방식: {best_recall}",
                        f"최저 Faithfulness 방식: {worst_faith}",
                    ],
                    "follow_up": "동일 질문의 검색 청크와 생성 답변을 대조해 문맥 조합·인용 실패 유형을 분류함",
                }
            )

    faith_errors = sum(item["error_count"] for item in faithfulness["summary"].values())
    judge_errors = sum(item["evaluation_error_count"] for item in judge["summary"].values())
    if faith_errors or judge_errors:
        items.append(
            {
                "failure_id": "e04_evaluation_incomplete",
                "type": "evaluation_issue",
                "title": "자동 평가 일부 미완료",
                "status": "confirmed",
                "affected_versions": [active_config["retrieval_modes"][mode] for mode in modes],
                "expectation": "모든 답변의 Faithfulness와 Judge 평가를 완료함",
                "observed": f"Faithfulness 오류 {faith_errors}건, Judge 오류 {judge_errors}건",
                "possible_cause": "평가 모델 출력 길이 초과 또는 외부 API 연결 오류가 발생함",
                "evidence": [
                    f"Faithfulness 오류: {faith_errors}건",
                    f"Judge 오류: {judge_errors}건",
                ],
                "follow_up": "정상 평가 결과는 유지하고 오류 항목만 재개해 보고서를 다시 생성함",
            }
        )

    return {"title": "Failure Analysis", "items": items}


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
