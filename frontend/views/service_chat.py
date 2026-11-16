"""서비스 > AI 어시스턴트."""

import json
import sqlite3
import uuid
from functools import lru_cache
from pathlib import Path

import httpx
import streamlit as st
import yaml

from chat_store import (
    create_session,
    list_sessions,
    load_comparisons,
    load_messages,
    save_comparison,
    save_exchange,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
PROCESSED = ROOT / "data" / "processed" / "manufacturing_basic"
API_URL = "http://127.0.0.1:8000"
EXPECTED = {
    "dataset": "manufacturing_basic",
    "embedding": "m1",
    "prompt": "p1",
}
RETRIEVAL_VERSIONS = {"naive": "r1", "hybrid": "r2", "rerank": "r3"}
SAMPLE_QUESTION = (
    "현장에서 네트워크로 받은 시험 자료를 확인하고 기록을 파일로 내보내려고 합니다. "
    "CV-10과 CV-20 중 어떤 모델을 사용해야 하나요?"
)


def load_label(category: str, identifier: str) -> str:
    path = CONFIGS / category / f"{identifier}.yaml"
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return f"({identifier.upper()}) {config['label']}"


def page_label(item: dict) -> str:
    start = item["page_start"]
    end = item["page_end"]
    return f"{start}쪽" if start == end else f"{start}–{end}쪽"


@lru_cache(maxsize=3)
def load_chunks(chunk_version: str) -> dict[str, dict]:
    path = PROCESSED / chunk_version / "chunks.jsonl"
    if not path.exists():
        return {}
    return {
        chunk["chunk_id"]: chunk
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (chunk := json.loads(line))
    }


def legacy_evidence(message: dict) -> list[dict]:
    metrics = message.get("metrics") or {}
    evidence = []
    for rank, chunk_id in enumerate(metrics.get("context_chunk_ids", []), start=1):
        parts = chunk_id.split("_")
        if len(parts) < 3:
            continue
        chunk = load_chunks(parts[-2]).get(chunk_id)
        if chunk:
            evidence.append({**chunk, "rank": rank, "used_in_answer": True})
    return evidence


def available_evidence(message: dict) -> list[dict]:
    return message.get("retrieval_results", []) or legacy_evidence(message)


def document_sources(message: dict) -> list[dict]:
    citations = message.get("citations", [])
    if citations:
        return citations
    return [
        result
        for result in available_evidence(message)
        if result.get("used_in_answer", True)
    ]


def show_retrieval_results(results: list[dict]) -> None:
    with st.expander("검색 과정 보기"):
        for result in results:
            used = " · 답변 문맥 사용" if result["used_in_answer"] else ""
            st.markdown(
                f"**{result['rank']}위 · {result['file_name']} · "
                f"{page_label(result)}**"
            )
            st.caption(
                f"점수 {result['score']:.4f} ({result['score_type']}){used} · "
                f"청크 ID: {result['chunk_id']}"
            )
            st.write(result["text"])
            if result.get("metadata"):
                metadata = " · ".join(
                    f"{key}: {value}" for key, value in result["metadata"].items()
                )
                st.caption(metadata)
            if result["rank"] < len(results):
                st.divider()


def show_sources(message: dict) -> None:
    scored_results = message.get("retrieval_results", [])
    results = available_evidence(message)
    sources = document_sources(message)
    if not sources:
        return
    st.markdown("**문서 근거**")
    for source in sources:
        st.markdown(f"- **{source['file_name']} · {page_label(source)}**")
        matching = next(
            (
                result
                for result in results
                if result["file_name"] == source["file_name"]
                and result["page_start"] <= source["page_end"]
                and result["page_end"] >= source["page_start"]
            ),
            None,
        )
        if matching:
            text = " ".join(matching["text"].split())
            summary = text[:260]
            st.markdown(f"> {summary}{'…' if len(text) > len(summary) else ''}")
    if scored_results:
        show_retrieval_results(scored_results)


def show_answer(message: dict) -> None:
    st.write(message["answer"])
    metrics = message.get("metrics") or {}
    latency = (
        f" · {metrics['latency_seconds']:.2f}초"
        if metrics.get("latency_seconds") is not None
        else ""
    )
    st.caption(
        f"상태: {message['status']} · 검색: {message['retrieval_mode']}{latency}"
    )
    if message.get("reason"):
        st.caption(f"판단 이유: {message['reason']}")
    show_sources(message)


def show_exchange(message: dict) -> None:
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        show_answer(message)


def show_options(name: str, options: list[tuple[str, bool]], key: str) -> None:
    label_section, option_section = st.columns(
        [1, 6], vertical_alignment="center", gap="small"
    )
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


def show_selection(
    name: str,
    options: list[str],
    key: str,
    index: int,
    format_func,
) -> str:
    label_section, option_section = st.columns(
        [1, 6], vertical_alignment="center", gap="small"
    )
    with label_section:
        st.caption(name)
    with option_section:
        return st.radio(
            name,
            options=options,
            index=index,
            format_func=format_func,
            horizontal=True,
            label_visibility="collapsed",
            key=key,
        )


def request_query(
    client: httpx.Client,
    question: str,
    mode: str,
    chunk: str,
    user_id: str,
    session_id: str,
) -> dict:
    response = client.post(
        f"{API_URL}/query",
        json={
            "question": question,
            "retrieval_mode": mode,
            "chunk_version": chunk,
            "source": "service",
            "user_id": user_id,
            "session_id": session_id,
        },
    )
    if response.status_code != 502:
        response.raise_for_status()
    return response.json()


def run_queries(
    client: httpx.Client,
    question: str,
    modes: list[str],
    chunk_version: str,
    user_id: str,
    selected_session_id: str | None,
) -> tuple[str, list[dict]]:
    session_id = selected_session_id or str(uuid.uuid4())
    results = [
        request_query(
            client,
            question,
            mode,
            chunk_version,
            user_id,
            session_id,
        )
        for mode in modes
    ]
    return session_id, results


def validate_api(client: httpx.Client) -> None:
    health = client.get(f"{API_URL}/health")
    health.raise_for_status()
    active = health.json()
    if active.get("status") != "ok":
        raise RuntimeError("API가 아직 준비되지 않았습니다.")
    if any(active.get(key) != value for key, value in EXPECTED.items()):
        raise RuntimeError("화면에 표시된 설정과 API의 활성 설정이 다릅니다.")


def comparison_summary(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        evidence = document_sources(result)
        metrics = result.get("metrics") or {}
        top_source = "-"
        if evidence:
            top_source = f"{evidence[0]['file_name']} · {page_label(evidence[0])}"
        rows.append(
            {
                "검색 방식": load_label(
                    "retrieval", RETRIEVAL_VERSIONS[result["retrieval_mode"]]
                ),
                "응답 상태": result["status"],
                "1순위 출처": top_source,
                "근거 수": len(evidence),
                "응답 시간(초)": metrics.get("latency_seconds"),
                "시스템 오류": "예" if result["status"] == "error" else "아니요",
            }
        )
    return rows


def show_comparison(comparison: dict, expanded: bool = False) -> None:
    with st.expander("검색 방식 비교", expanded=expanded):
        with st.container(border=True):
            st.markdown("**비교 질문**")
            st.write(comparison["question"])
        results = comparison["results"]
        st.dataframe(comparison_summary(results), hide_index=True, width="stretch")
        result_by_mode = {result["retrieval_mode"]: result for result in results}
        modes = [mode for mode in RETRIEVAL_VERSIONS if mode in result_by_mode]
        tabs = st.tabs(
            [load_label("retrieval", RETRIEVAL_VERSIONS[mode]) for mode in modes]
        )
        for tab, mode in zip(tabs, modes, strict=True):
            with tab:
                show_answer(result_by_mode[mode])


st.title("AI 어시스턴트")
user_id = st.session_state.chat_user_id
sessions = list_sessions(user_id)
session_titles = {session["session_id"]: session["title"] for session in sessions}
if st.session_state.get("chat_session_id") not in session_titles:
    st.session_state.chat_session_id = None


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
    chunk_version = show_selection(
        "청킹 방식",
        ["c1", "c2", "c3"],
        "setting_chunk_version",
        2,
        lambda version: load_label("chunking", version),
    )
    show_options("임베딩 모델", [(load_label("embedding", "m1"), True)], "embedding")
    show_options("프롬프트", [
        (load_label("prompts", "p1"), True),
        ("(P2) 미정", False),
        ("(P3) 미정", False),
    ], "prompt")
    retrieval_mode = show_selection(
        "검색 방식",
        list(RETRIEVAL_VERSIONS),
        "setting_retrieval_mode",
        1,
        lambda mode: load_label("retrieval", RETRIEVAL_VERSIONS[mode]),
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
    comparisons = load_comparisons(user_id, selected)
    for index, comparison in enumerate(comparisons):
        show_comparison(comparison, expanded=index == len(comparisons) - 1)

question = st.text_area(
    "질문",
    placeholder="장비 오류, 점검 절차 또는 작업 조건을 질문해 주세요.",
    key="service_question",
)
with st.container(horizontal=True, vertical_alignment="center"):
    submit = st.button("질문하기", type="primary")
    st.button(
        "샘플 질문",
        on_click=lambda: st.session_state.update(service_question=SAMPLE_QUESTION),
    )
    compare = st.button("검색 방식 비교")

if submit or compare:
    if not question.strip():
        st.warning("질문을 입력해 주세요.")
    elif len(question.strip()) > 2000:
        st.warning("질문은 2,000자 이내로 입력해 주세요.")
    else:
        try:
            spinner = (
                "세 가지 검색 방식의 결과를 비교하고 있습니다."
                if compare
                else "문서를 검색하고 답변을 생성하고 있습니다."
            )
            with st.spinner(spinner):
                with httpx.Client(timeout=180.0) as client:
                    validate_api(client)
                    modes = list(RETRIEVAL_VERSIONS) if compare else [retrieval_mode]
                    session_id, results = run_queries(
                        client,
                        question.strip(),
                        modes,
                        chunk_version,
                        user_id,
                        selected,
                    )
                if not selected:
                    create_session(user_id, session_id, question.strip()[:60])
                if compare:
                    save_comparison(
                        user_id,
                        session_id,
                        question.strip(),
                        results,
                        str(uuid.uuid4()),
                    )
                else:
                    save_exchange(
                        user_id, session_id, question.strip(), results[0]
                    )
                st.session_state.chat_session_id = session_id
                st.rerun()
        except RuntimeError as error:
            st.error(str(error))
        except httpx.RequestError:
            st.error("API 서버에 연결하지 못했습니다. 서버가 실행 중인지 확인해 주세요.")
        except httpx.HTTPStatusError as error:
            st.error(f"API 요청에 실패했습니다. 상태 코드: {error.response.status_code}")
        except sqlite3.Error:
            st.error("대화 저장에 실패했습니다. 데이터베이스 상태를 확인해 주세요.")
