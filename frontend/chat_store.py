"""SQLite 기반 샘플 사용자와 채팅 이력 저장소."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "chat" / "chat.sqlite3"

SAMPLE_USERS = (
    ("demo_quality", "김민수", "품질관리팀", "대리", "품질 담당자"),
    ("demo_field", "이서연", "설비운영팀", "사원", "현장 작업자"),
)


@contextmanager
def connect(db_path: Path = DEFAULT_DB):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize(db_path: Path = DEFAULT_DB) -> None:
    with connect(db_path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                position TEXT NOT NULL,
                role TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                retrieval_mode TEXT NOT NULL,
                request_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS answer_citations (
                citation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL REFERENCES chat_messages(message_id),
                source_id TEXT NOT NULL,
                book_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL
            );
        """)
        db.executemany(
            "INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", SAMPLE_USERS
        )


def list_users(db_path: Path = DEFAULT_DB) -> list[dict]:
    with connect(db_path) as db:
        return [dict(row) for row in db.execute("SELECT * FROM users ORDER BY user_id")]


def list_sessions(user_id: str, db_path: Path = DEFAULT_DB) -> list[dict]:
    with connect(db_path) as db:
        rows = db.execute(
            "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC, rowid DESC",
            (user_id,),
        )
        return [dict(row) for row in rows]


def create_session(user_id: str, session_id: str, title: str, db_path: Path = DEFAULT_DB) -> None:
    with connect(db_path) as db:
        db.execute(
            "INSERT INTO chat_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
            (session_id, user_id, title),
        )


def save_exchange(user_id: str, session_id: str, question: str, result: dict, db_path: Path = DEFAULT_DB) -> None:
    with connect(db_path) as db:
        owner = db.execute(
            "SELECT 1 FROM chat_sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if owner is None:
            raise ValueError("사용자의 대화 세션을 찾을 수 없습니다.")
        cursor = db.execute(
            """INSERT INTO chat_messages
               (session_id, question, answer, status, reason, retrieval_mode, request_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, question, result["answer"], result["status"], result.get("reason"),
             result["retrieval_mode"], result.get("request_id")),
        )
        db.executemany(
            """INSERT INTO answer_citations
               (message_id, source_id, book_id, file_name, page_start, page_end)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(cursor.lastrowid, c["source_id"], c["book_id"], c["file_name"],
              c["page_start"], c["page_end"]) for c in result.get("citations", [])],
        )


def load_messages(user_id: str, session_id: str, db_path: Path = DEFAULT_DB) -> list[dict]:
    with connect(db_path) as db:
        rows = db.execute(
            """SELECT m.* FROM chat_messages m
               JOIN chat_sessions s ON s.session_id = m.session_id
               WHERE s.user_id = ? AND s.session_id = ? ORDER BY m.message_id""",
            (user_id, session_id),
        ).fetchall()
        messages = []
        for row in rows:
            message = dict(row)
            message["citations"] = [
                dict(citation) for citation in db.execute(
                    "SELECT * FROM answer_citations WHERE message_id = ? ORDER BY citation_id",
                    (row["message_id"],),
                )
            ]
            messages.append(message)
        return messages
