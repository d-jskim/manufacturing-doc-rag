from evaluation.common import metric_run_id, project_path, select_records
from evaluation.run_comparison import make_run_id


def test_metric_run_id_replaces_answer_marker():
    source = "e04_c1_m1_p1_development_all_20261112_1200"

    assert metric_run_id(source, "judge") == (
        "e04_c1_m1_p1_development_judge_20261112_1200"
    )


def test_project_path_accepts_windows_separator():
    path = project_path("data\\raw\\manufacturing_basic")

    assert path.parts[-3:] == ("data", "raw", "manufacturing_basic")


def test_select_records_limits_by_question_not_mode():
    records = [
        {"question_id": "q1", "retrieval_mode": "naive"},
        {"question_id": "q1", "retrieval_mode": "hybrid"},
        {"question_id": "q2", "retrieval_mode": "naive"},
    ]

    assert select_records(records, 1) == records[:2]


def test_run_id_contains_active_config_and_question_set():
    active_config = {
        "chunk_version": "c2",
        "embedding": "m1",
        "prompt": "p1",
    }

    run_id = make_run_id(active_config, "additional")

    assert run_id.startswith("e04_c2_m1_p1_additional_all_")
