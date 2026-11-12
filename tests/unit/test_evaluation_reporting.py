import json

import yaml

from evaluation import report as report_module
from evaluation import run_retrieval as retrieval_module


def active_config() -> dict:
    return {
        "dataset": "manufacturing_basic",
        "chunk_version": "c1",
        "embedding": "m1",
        "prompt": "p1",
        "retrieval_modes": {"naive": "r1"},
        "final_k": 3,
        "context_token_budget": 2373,
    }


def test_run_retrieval_uses_recorded_context(monkeypatch, tmp_path):
    run_id = "e04_c1_m1_p1_development_all_20261112_1200"
    answer_path = tmp_path / f"{run_id}.json"
    answers = {
        "requested_modes": ["naive"],
        "question_source": "data/raw/manufacturing_basic/evaluation/development.json",
        "records": [
            {
                "question_id": "q1",
                "question": "질문",
                "retrieval_mode": "naive",
                "response": {
                    "metrics": {
                        "context_chunk_ids": ["chunk-1"],
                        "retrieval_seconds": 0.2,
                    }
                },
            }
        ],
    }
    truth = [
        {
            "id": "q1",
            "evidence_groups": [
                [{"book_id": "MB01", "page": 1, "quote": "근거 문장"}]
            ],
        }
    ]
    chunks = {
        "chunk-1": {
            "chunk_id": "chunk-1",
            "book_id": "MB01",
            "text": "근거 문장",
        }
    }

    monkeypatch.setattr(
        retrieval_module,
        "load_answer_result",
        lambda dataset, source_run_id: (
            answer_path,
            answers,
            active_config(),
        ),
    )
    monkeypatch.setattr(retrieval_module, "project_path", lambda value: tmp_path)
    monkeypatch.setattr(retrieval_module, "relative_path", lambda path: path.name)
    monkeypatch.setattr(retrieval_module, "load_json", lambda path: truth)
    monkeypatch.setattr(
        retrieval_module,
        "load_chunks",
        lambda dataset, chunk_version: chunks,
    )
    monkeypatch.setattr(retrieval_module, "results_dir", lambda dataset: tmp_path)

    output_path = retrieval_module.run_retrieval(
        "manufacturing_basic",
        run_id,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["summary"]["naive"]["k3"]["hit_at_k"] == 1.0
    assert result["summary"]["naive"]["retrieval_seconds"]["mean"] == 0.2


def test_build_report_links_result_files(monkeypatch, tmp_path):
    run_id = "e04_c1_m1_p1_development_all_20261112_1200"
    answer_path = tmp_path / f"{run_id}.json"
    answers = {
        "requested_modes": ["naive"],
        "question_set": "development",
        "question_count": 30,
    }

    monkeypatch.setattr(
        report_module,
        "load_answer_result",
        lambda dataset, source_run_id: (
            answer_path,
            answers,
            active_config(),
        ),
    )
    monkeypatch.setattr(report_module, "results_dir", lambda dataset: tmp_path)

    def metric_result(path, source_run_id):
        if "faithfulness" in path.name:
            return {"summary": {"naive": {"error_count": 0}}}
        if "judge" in path.name:
            return {"summary": {"naive": {"evaluation_error_count": 0}}}
        return {"summary": {"naive": {"k3": {"hit_at_k": 1.0}}}}

    monkeypatch.setattr(report_module, "validate_metric_result", metric_result)

    report_path = report_module.build_report("manufacturing_basic", run_id)
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "approved"
    assert report["source_run_id"] == run_id
    assert report["result_files"]["answers"] == answer_path.name
    assert report["settings"]["question_set"] == "development"
