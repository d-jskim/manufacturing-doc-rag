"""Streamlit 앱 진입점: 메뉴(st.navigation) 구성."""
import streamlit as st

st.set_page_config(page_title="제조 문서 RAG", layout="wide")

pages = {
    "서비스": [
        st.Page("views/service_chat.py", title="AI 어시스턴트", default=True),
    ],
    "관리자": [
        st.Page("views/admin_dashboard.py", title="대시보드(준비 중)"),
        st.Page("views/admin_active_config.py", title="운영 설정"),
    ],
    "실험": [
        st.Page("views/experiments_token_budget.py", title="토큰 한도"),
        st.Page("views/experiments_chunking.py", title="청킹"),
    ],
}

st.navigation(pages).run()