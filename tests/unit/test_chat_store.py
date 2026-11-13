"""SQLite 사용자별 대화와 인용 저장을 검증한다."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

from chat_store import create_session, initialize, list_sessions, list_users, load_messages, save_exchange


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
    assert load_messages(other_id, "session-a", db_path) == []
    with pytest.raises(ValueError):
        save_exchange(other_id, "session-a", "다른 사용자", result, db_path)
