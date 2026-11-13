"""Basic 확정 설정을 다른 데이터셋에 그대로 적용해 RAG Baseline을 평가한다."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.report import build_report
from evaluation.run_comparison import run_comparison
from evaluation.run_faithfulness import run_faithfulness
from evaluation.run_judge import run_judge
from evaluation.run_retrieval import run_retrieval

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments" / "04_retrieval" / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def write_runtime_config(dataset: str, settings: dict) -> Path:
    path = ROOT / "data" / "runtime" / f"e04_{dataset}_baseline.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {"dataset": dataset, **settings},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def prepare_index(dataset: str, settings: dict) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_index.py"),
        "--dataset",
        dataset,
        "--chunk-version",
        settings["chunk_version"],
        "--embedding",
        settings["embedding"],
        "--qdrant-url",
        settings["qdrant_url"],
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def start_api(config_path: Path, host: str, port: int) -> subprocess.Popen:
    environment = {**os.environ, "RAG_ACTIVE_CONFIG_PATH": str(config_path)}
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        if os.name == "nt"
        else 0
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.api.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=environment,
        creationflags=creationflags,
    )


def wait_for_api(server: subprocess.Popen, base_url: str, dataset: str) -> None:
    deadline = time.monotonic() + 300
    with httpx.Client(timeout=5) as client:
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"실험 API가 종료되었습니다: {server.returncode}")
            try:
                health = client.get(f"{base_url}/health").json()
                if health.get("status") == "ok":
                    if health.get("dataset") != dataset:
                        raise RuntimeError(
                            f"데이터셋 불일치: {dataset} != {health.get('dataset')}"
                        )
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(2)
    raise TimeoutError(f"실험 API 준비 시간 초과: {dataset}")


def stop_api(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    if os.name == "nt":
        server.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def run_dataset(dataset: str, config: dict, chunk_version: str) -> None:
    settings = {**config["settings"], "chunk_version": chunk_version}
    host = config["api_host"]
    port = config["api_port"]
    base_url = f"http://{host}:{port}"

    print(f"\n=== {dataset}: Baseline 색인 준비 ===", flush=True)
    prepare_index(dataset, settings)
    runtime_config = write_runtime_config(dataset, settings)
    server = start_api(runtime_config, host, port)
    try:
        wait_for_api(server, base_url, dataset)
        print(f"=== {dataset}: 답변 생성 ===", flush=True)
        answer_path = run_comparison(
            dataset=dataset,
            question_set=config["question_set"],
            base_url=base_url,
            timeout=config["request_timeout_seconds"],
        )
    finally:
        stop_api(server)

    run_id = answer_path.stem
    print(f"=== {dataset}: 검색 지표 계산 ===", flush=True)
    run_retrieval(dataset, run_id)
    print(f"=== {dataset}: Faithfulness 평가 ===", flush=True)
    run_faithfulness(dataset, run_id)
    print(f"=== {dataset}: Judge 평가 ===", flush=True)
    run_judge(dataset, run_id)
    print(f"=== {dataset}: 보고서 생성 ===", flush=True)
    build_report(dataset, run_id)


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser()
    datasets = sorted({"manufacturing_basic", *config["datasets"]})
    parser.add_argument("--dataset", choices=datasets)
    parser.add_argument(
        "--chunk-version",
        choices=("c1", "c2", "c3"),
        default=config["settings"]["chunk_version"],
    )
    return parser.parse_args()


def main() -> None:
    config = load_config()
    args = parse_args()
    datasets = [args.dataset] if args.dataset else config["datasets"]
    for dataset in datasets:
        run_dataset(dataset, config, args.chunk_version)


if __name__ == "__main__":
    main()
