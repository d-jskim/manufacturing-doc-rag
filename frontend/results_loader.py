"""실험 결과 JSON과 report.yaml을 불러오는 모듈(이후 API 호출로 전환)."""
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def list_result_files(experiment: str) -> list[Path]:
    """results/ 바로 아래의 유효 결과만 최신순으로 반환함(invalid/ 제외)."""
    results_dir = EXPERIMENTS_DIR / experiment / "results"
    return sorted(results_dir.glob("*.json"), reverse=True)


def load_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_report(experiment: str) -> dict:
    with (EXPERIMENTS_DIR / experiment / "report.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_value(result: dict, key: str):
    """'a.b' 형태의 키로 summary 안의 값을 꺼냄(report.yaml의 result_key용)."""
    value = result["summary"]
    for part in key.split("."):
        value = value[part]
    return value