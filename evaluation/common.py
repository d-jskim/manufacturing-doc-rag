"""평가 실행에서 공통으로 사용하는 경로와 결과 파일 도우미."""

import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "manufacturing_basic"
EVALUATION_MODEL = "gpt-6-luna"
DATASETS = tuple(
    sorted(
        path.parent.parent.name
        for path in (ROOT / "data" / "raw").glob("*/evaluation/development.json")
    )
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def project_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    return ROOT.joinpath(*PurePosixPath(normalized).parts)


def evaluation_path(dataset: str, question_set: str) -> Path:
    path = ROOT / "data" / "raw" / dataset / "evaluation" / f"{question_set}.json"
    if not path.exists():
        raise FileNotFoundError(f"평가 질문 파일이 없습니다: {path}")
    return path


def results_dir(dataset: str) -> Path:
    return ROOT / "experiments" / "04_retrieval" / "results" / dataset


def load_chunks(dataset: str, chunk_version: str) -> dict:
    path = ROOT / "data" / "processed" / dataset / chunk_version / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"청크 파일이 없습니다: {path}")
    return {
        chunk["chunk_id"]: chunk
        for chunk in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def select_records(records: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return records
    question_ids = list(dict.fromkeys(record["question_id"] for record in records))
    selected_ids = set(question_ids[:limit])
    return [
        record for record in records if record["question_id"] in selected_ids
    ]


def metric_run_id(source_run_id: str, metric: str) -> str:
    if "_all_" in source_run_id:
        return source_run_id.replace("_all_", f"_{metric}_", 1)
    return f"{source_run_id}_{metric}"


def load_answer_result(dataset: str, run_id: str) -> tuple[Path, dict, dict]:
    answer_path = results_dir(dataset) / f"{run_id}.json"
    if not answer_path.exists():
        raise FileNotFoundError(f"답변 결과가 없습니다: {answer_path}")

    answer_result = load_json(answer_path)
    active_config = answer_result.get("active_config")
    if active_config is None:
        raise ValueError(
            "active_config가 없는 이전 결과입니다. 새 실행 결과를 사용하세요."
        )
    if active_config["dataset"] != dataset:
        raise ValueError(
            f"데이터셋 불일치: 실행={dataset}, 결과={active_config['dataset']}"
        )
    return answer_path, answer_result, active_config
