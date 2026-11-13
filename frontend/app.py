"""Streamlit 앱 진입점: 샘플 로그인과 공통 메뉴 구성."""

import sqlite3

import streamlit as st

from chat_store import initialize, list_users

st.set_page_config(page_title="제조 문서 RAG", layout="wide")

try:
    initialize()
    users = list_users()
except sqlite3.Error:
    st.error("사용자 데이터베이스를 열지 못했습니다.")
    st.stop()

user_by_id = {user["user_id"]: user for user in users}
user_id = st.session_state.get("chat_user_id")

if user_id not in user_by_id:
    st.title("제조 문서 RAG")
    st.caption("시연용 샘플 로그인입니다. 실제 인증과 문서 접근 권한은 적용되지 않습니다.")
    chosen_user = st.selectbox(
        "샘플 사용자",
        options=list(user_by_id),
        format_func=lambda key: (
            f"{user_by_id[key]['name']} · {user_by_id[key]['department']} · "
            f"{user_by_id[key]['position']} · {user_by_id[key]['role']}"
        ),
    )
    if st.button("로그인", type="primary"):
        st.session_state.chat_user_id = chosen_user
        st.session_state.chat_session_id = None
        st.rerun()
    st.stop()

user = user_by_id[user_id]
header_title, header_user = st.columns([5, 1])
with header_title:
    st.markdown("**제조 문서 RAG**")
with header_user:
    with st.popover(f"👤 {user['name']}"):
        st.write(f"{user['department']} · {user['position']}")
        st.caption(user["role"])
        if st.button("로그아웃"):
            st.session_state.chat_user_id = None
            st.session_state.chat_session_id = None
            st.rerun()

pages = {
    "서비스": [
        st.Page("views/service_chat.py", title="AI 어시스턴트", default=True),
    ],
    "관리자": [
        st.Page("views/admin_dashboard.py", title="대시보드(준비 중)"),
        st.Page("views/admin_active_config.py", title="운영 설정"),
    ],
    "실험": [
    st.Page(
        "views/experiments_token_budget.py",
        title="01 토큰 한도",
    ),
    st.Page(
        "views/experiments_chunking.py",
        title="02 청킹",
    ),
    st.Page(
        "views/experiments_prompt.py",
        title="03 프롬프트",
    ),
    st.Page(
        "views/experiments_rag_comparison.py",
        title="04 RAG 방식 비교",
    ),
],
}

st.navigation(pages).run()
