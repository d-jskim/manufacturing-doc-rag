"""SQLite 사용자별 대화와 인용 저장을 검증한다."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

from chat_store import (
    create_session,
    initialize,
    list_sessions,
    list_users,
    load_comparisons,
    load_messages,
    save_comparison,
    save_exchange,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "chat.sqlite3"
    initialize(path)
    return path


def test_sample_users_have_security_context(db_path):
    users = list_users(db_path)
    assert len(users) == 2
    assert all(user["department"] and user["position"] and user["role"] for user in users)
    initialize(db_path)
    assert len(list_users(db_path)) == 2


def test_exchange_and_citation_persist_and_are_scoped(db_path):
    user_id = list_users(db_path)[0]["user_id"]
    other_id = list_users(db_path)[1]["user_id"]
    create_session(user_id, "session-a", "온도 질문", db_path)
    result = {
        "answer": "80도 이하입니다.",
        "status": "answered",
        "reason": "문서 확인",
        "retrieval_mode": "rerank",
        "request_id": "request-a",
        "settings": {
            "dataset": "manufacturing_basic",
            "chunk_version": "c3",
            "embedding": "m1",
            "prompt": "p1",
        },
        "metrics": {"latency_seconds": 1.25},
        "retrieval_results": [],
        "citations": [{
            "source_id": "S1", "book_id": "B1", "file_name": "manual.pdf",
            "page_start": 3, "page_end": 4,
        }],
    }
    save_exchange(user_id, "session-a", "온도는?", result, db_path)
    assert list_sessions(user_id, db_path)[0]["session_id"] == "session-a"
    assert list_sessions(other_id, db_path) == []
    messages = load_messages(user_id, "session-a", db_path)
    assert messages[0]["answer"] == "80도 이하입니다."
    assert messages[0]["citations"][0]["page_start"] == 3
    assert messages[0]["settings"]["chunk_version"] == "c3"
    assert messages[0]["metrics"]["latency_seconds"] == 1.25
    assert load_messages(other_id, "session-a", db_path) == []
    with pytest.raises(ValueError):
        save_exchange(other_id, "session-a", "다른 사용자", result, db_path)


def test_comparison_results_are_grouped_and_separate_from_messages(db_path):
    user_id = list_users(db_path)[0]["user_id"]
    create_session(user_id, "session-b", "비교 질문", db_path)
    results = []
    for mode in ("naive", "hybrid", "rerank"):
        results.append({
            "answer": f"{mode} 답변",
            "status": "answered",
            "reason": None,
            "retrieval_mode": mode,
            "request_id": f"request-{mode}",
            "settings": {
                "dataset": "manufacturing_basic",
                "chunk_version": "c3",
                "embedding": "m1",
                "prompt": "p1",
            },
            "metrics": {"latency_seconds": 1.0},
            "citations": [],
            "retrieval_results": [],
        })
    save_comparison(
        user_id,
        "session-b",
        "같은 질문",
        results,
        "comparison-1",
        db_path,
    )

    assert load_messages(user_id, "session-b", db_path) == []
    comparisons = load_comparisons(user_id, "session-b", db_path)
    assert comparisons[0]["group_id"] == "comparison-1"
    assert [item["retrieval_mode"] for item in comparisons[0]["results"]] == [
        "naive",
        "hybrid",
        "rerank",
    ]
