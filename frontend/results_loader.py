"""실험 결과와 YAML 보고서를 불러오는 모듈."""

import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def results_dir(experiment: str, dataset: str) -> Path:
    return (
        EXPERIMENTS_DIR
        / experiment
        / "results"
        / dataset
    )


def reports_dir(experiment: str, dataset: str) -> Path:
    return results_dir(experiment, dataset) / "reports"


def _has_report(dataset_dir: Path) -> bool:
    legacy_report = dataset_dir / "report.yaml"
    report_directory = dataset_dir / "reports"

    return legacy_report.exists() or (
        report_directory.exists()
        and any(
            report_directory.glob("*_report.yaml")
        )
    )


def list_datasets(experiment: str) -> list[str]:
    """보고서가 존재하는 데이터셋 이름을 반환한다."""
    base = EXPERIMENTS_DIR / experiment / "results"

    return sorted(
        path.name
        for path in base.iterdir()
        if path.is_dir() and _has_report(path)
    )


def list_result_files(
    experiment: str,
    dataset: str,
) -> list[Path]:
    """데이터셋 폴더의 결과 JSON을 최신순으로 반환한다."""
    return sorted(
        results_dir(experiment, dataset).glob("*.json"),
        reverse=True,
    )


def _report_sort_key(path: Path) -> tuple[str, str]:
    with path.open(encoding="utf-8") as file:
        report = yaml.safe_load(file) or {}

    return (
        str(report.get("created_at", "")),
        path.name,
    )


def list_report_files(
    experiment: str,
    dataset: str,
) -> list[Path]:
    """승인된 실행별 보고서를 최신순으로 반환한다."""
    directory = reports_dir(experiment, dataset)

    if not directory.exists():
        return []

    report_files = []

    for path in directory.glob("*_report.yaml"):
        with path.open(encoding="utf-8") as file:
            report = yaml.safe_load(file) or {}

        if report.get("status") == "approved":
            report_files.append(path)

    return sorted(
        report_files,
        key=_report_sort_key,
        reverse=True,
    )


def load_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_report(
    experiment: str,
    dataset: str,
) -> dict:
    """기존 단일 report.yaml을 불러온다."""
    path = (
        results_dir(experiment, dataset)
        / "report.yaml"
    )

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_report_file(path: Path) -> dict:
    """선택한 실행별 보고서를 불러온다."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_value(result: dict, key: str):
    """점으로 구분된 키로 summary 값을 찾는다."""
    value = result["summary"]

    for part in key.split("."):
        value = value[part]

    return value