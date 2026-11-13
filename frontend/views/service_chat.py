"""서비스 > AI 어시스턴트."""

import sqlite3
import uuid
from pathlib import Path

import httpx
import streamlit as st
import yaml

from chat_store import create_session, list_sessions, load_messages, save_exchange

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
API_URL = "http://127.0.0.1:8000"
EXPECTED = {
    "dataset": "manufacturing_basic",
    "chunk_version": "c1",
    "embedding": "m1",
    "prompt": "p1",
}
RETRIEVAL_VERSIONS = {"naive": "r1", "hybrid": "r2", "rerank": "r3"}


def load_label(category: str, identifier: str) -> str:
    path = CONFIGS / category / f"{identifier}.yaml"
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return f"({identifier.upper()}) {config['label']}"


def show_exchange(message: dict) -> None:
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        st.write(message["answer"])
        st.caption(f"상태: {message['status']} · 검색: {message['retrieval_mode']}")
        if message.get("reason"):
            st.caption(f"판단 이유: {message['reason']}")
        for citation in message.get("citations", []):
            start = citation["page_start"]
            end = citation["page_end"]
            pages = f"{start}쪽" if start == end else f"{start}–{end}쪽"
            st.write(f"- {citation['file_name']} · {pages}")


st.title("AI 어시스턴트")
user_id = st.session_state.chat_user_id
sessions = list_sessions(user_id)
session_titles = {session["session_id"]: session["title"] for session in sessions}
if st.session_state.get("chat_session_id") not in session_titles:
    st.session_state.chat_session_id = None

def show_options(name: str, options: list[tuple[str, bool]], key: str) -> None:
    label_section, option_section = st.columns([1, 6], vertical_alignment="center", gap="small")
    with label_section:
        st.caption(name)
    with option_section:
        option_row = st.container(
            key=f"setting_options_{key}",
            horizontal=True,
            wrap=False,
            vertical_alignment="center",
            gap="small",
        )
        with option_row:
            for option, available in options:
                st.radio(
                    f"{name}: {option}",
                    [option],
                    index=0 if available else None,
                    disabled=not available,
                    label_visibility="collapsed",
                    key=f"setting_{name}_{option}",
                )


st.markdown("""
    <style>
    .st-key-chat_settings [data-testid="stRadio"] label p,
    .st-key-chat_settings [data-testid="stCaptionContainer"] p {
        font-size: 0.8rem;
        white-space: nowrap;
    }
    .st-key-chat_settings [class*="st-key-setting_options_"] {
        justify-content: flex-start;
    }
    .st-key-chat_settings [class*="st-key-setting_options_"] > div {
        flex: 0 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

with st.container(key="chat_settings", gap=None):
    show_options("데이터셋", [
        ("Basic", True), ("Intermediate", False), ("Advanced", False),
    ], "dataset")
    show_options("청킹 방식", [
        (load_label("chunking", "c1"), True),
        (load_label("chunking", "c2"), False),
        (load_label("chunking", "c3"), False),
    ], "chunking")
    show_options("임베딩 모델", [(load_label("embedding", "m1"), True)], "embedding")
    show_options("프롬프트", [
        (load_label("prompts", "p1"), True),
        ("(P2) 미정", False),
        ("(P3) 미정", False),
    ], "prompt")
    retrieval_label, retrieval_options = st.columns([1, 6], vertical_alignment="center", gap="small")
    with retrieval_label:
        st.caption("검색 방식")
    with retrieval_options:
        retrieval_mode = st.radio(
            "검색 방식",
            options=list(RETRIEVAL_VERSIONS),
            format_func=lambda mode: load_label("retrieval", RETRIEVAL_VERSIONS[mode]),
            horizontal=True,
            label_visibility="collapsed",
        )

new_chat, previous_chats = st.columns([1, 4], vertical_alignment="top")
with new_chat:
    if st.button("＋ 새 대화"):
        st.session_state.chat_session_id = None
        st.rerun()

with previous_chats:
    with st.expander("이전 대화"):
        if not sessions:
            st.caption("저장된 대화가 없습니다.")
        for session in sessions:
            if st.button(session["title"], key=f"session_{session['session_id']}"):
                st.session_state.chat_session_id = session["session_id"]
                st.rerun()

selected = st.session_state.chat_session_id
st.caption(f"현재 대화: {session_titles[selected] if selected else '새 대화'}")

if selected:
    for message in load_messages(user_id, selected):
        show_exchange(message)

question = st.text_area("질문", placeholder="장비의 기능, 설정, 점검 절차 등 업무에 필요한 내용을 질문해 주세요.")
submit = st.button("질문하기", type="primary")

if submit:
    if not question.strip():
        st.warning("질문을 입력해 주세요.")
    elif len(question.strip()) > 2000:
        st.warning("질문은 2,000자 이내로 입력해 주세요.")
    else:
        try:
            with st.spinner("문서를 검색하고 답변을 생성하고 있습니다."):
                with httpx.Client(timeout=180.0) as client:
                    health = client.get(f"{API_URL}/health")
                    health.raise_for_status()
                    active = health.json()
                    if active.get("status") != "ok":
                        st.warning("API가 아직 준비되지 않았습니다.")
                    elif any(active.get(key) != value for key, value in EXPECTED.items()):
                        st.error("화면에 표시된 설정과 API의 활성 설정이 다릅니다.")
                    else:
                        response = client.post(
                            f"{API_URL}/query",
                            json={"question": question, "retrieval_mode": retrieval_mode, "source": "service"},
                        )
                        response.raise_for_status()
                        result = response.json()
                        session_id = selected or str(uuid.uuid4())
                        if not selected:
                            create_session(user_id, session_id, question.strip()[:60])
                        save_exchange(user_id, session_id, question.strip(), result)
                        st.session_state.chat_session_id = session_id
                        st.rerun()
        except httpx.RequestError:
            st.error("API 서버에 연결하지 못했습니다. 서버가 실행 중인지 확인해 주세요.")
        except httpx.HTTPStatusError as error:
            st.error(f"API 요청에 실패했습니다. 상태 코드: {error.response.status_code}")
        except sqlite3.Error:
            st.error("대화 저장에 실패했습니다. 데이터베이스 상태를 확인해 주세요.")
