"""관리자 > Langfuse 서비스 대시보드."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv
from langfuse import get_client

ROOT = Path(__file__).resolve().parents[2]
SEOUL = ZoneInfo("Asia/Seoul")
RETRIEVAL_LABELS = {
    "naive": "(R1) Naive",
    "hybrid": "(R2) Hybrid",
    "rerank": "(R3) Hybrid + Reranking",
}
NORMAL_STATUSES = {"answered", "needs_clarification", "abstained"}


def configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def parse_value(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_observation(observation) -> dict:
    metadata = observation.metadata or {}
    input_data = parse_value(observation.input) or {}
    output_data = parse_value(observation.output) or {}
    if not isinstance(input_data, dict):
        input_data = {}
    if not isinstance(output_data, dict):
        output_data = {}

    timings = output_data.get("timings") or {}
    usage = output_data.get("usage") or observation.usage_details or {}
    input_tokens = usage.get("input_tokens", usage.get("input"))
    output_tokens = usage.get("output_tokens", usage.get("output"))
    error = output_data.get("error") or observation.status_message
    status = output_data.get("status") or metadata.get("status")
    if error and not status:
        status = "error"
    settings = dict(output_data.get("settings") or {})
    for key in (
        "dataset",
        "chunk_version",
        "embedding",
        "prompt",
        "retrieval",
        "retrieval_mode",
    ):
        if key not in settings and metadata.get(key) is not None:
            settings[key] = metadata[key]
    return {
        "trace_id": observation.trace_id,
        "source": metadata.get("source"),
        "start_time": observation.start_time,
        "user_id": observation.user_id or metadata.get("user_id") or "-",
        "session_id": observation.session_id or metadata.get("session_id") or "-",
        "question": input_data.get("question", "-"),
        "retrieval_mode": metadata.get("retrieval_mode", "-"),
        "status": status or "-",
        "latency_seconds": timings.get("total_seconds", observation.latency),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "settings": settings,
        "retrieval_results": output_data.get("retrieval_results", []),
        "citations": output_data.get("citations", []),
        "error": error,
    }


def fetch_service_requests(client, start: datetime, end: datetime) -> list[dict]:
    source_filter = json.dumps(
        [
            {
                "type": "stringObject",
                "column": "metadata",
                "key": "source",
                "operator": "=",
                "value": "service",
            }
        ]
    )
    response = client.api.observations.get_many(
        fields="basic,time,io,metadata,usage,metrics,trace_context",
        expand_metadata=(
            "source,user_id,session_id,request_id,dataset,chunk_version,"
            "embedding,prompt,retrieval,retrieval_mode,status,error"
        ),
        limit=500,
        name="rag_query",
        is_root_observation=True,
        from_start_time=start,
        to_start_time=end,
        filter=source_filter,
    )
    rows = [normalize_observation(item) for item in response.data]
    return [
        row for row in rows if row["trace_id"] and row["source"] == "service"
    ]


def filter_requests(
    rows: list[dict], user_id: str, retrieval_mode: str, status: str
) -> list[dict]:
    return [
        row
        for row in rows
        if (user_id == "전체" or row["user_id"] == user_id)
        and (retrieval_mode == "전체" or row["retrieval_mode"] == retrieval_mode)
        and (status == "전체" or row["status"] == status)
    ]


def average(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def normal_count(rows: list[dict]) -> int:
    return sum(row["status"] in NORMAL_STATUSES for row in rows)


def confirmed_usage_count(rows: list[dict]) -> int:
    return sum(
        row["input_tokens"] is not None and row["output_tokens"] is not None
        for row in rows
    )


def mode_summary(rows: list[dict]) -> list[dict]:
    summary = []
    for mode in RETRIEVAL_LABELS:
        selected = [row for row in rows if row["retrieval_mode"] == mode]
        if not selected:
            continue
        errors = sum(row["status"] == "error" for row in selected)
        average_latency = average(selected, "latency_seconds")
        summary.append(
            {
                "검색 방식": RETRIEVAL_LABELS[mode],
                "요청 수": len(selected),
                "평균 응답 시간(초)": (
                    round(average_latency, 3)
                    if average_latency is not None
                    else "-"
                ),
                "오류율": f"{errors / len(selected):.1%}",
            }
        )
    return summary


def format_time(value: datetime) -> str:
    return value.astimezone(SEOUL).strftime("%m-%d %H:%M:%S")


def recent_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "시각": format_time(row["start_time"]),
            "사용자": row["user_id"],
            "세션": row["session_id"],
            "질문": row["question"],
            "검색 방식": RETRIEVAL_LABELS.get(
                row["retrieval_mode"], row["retrieval_mode"]
            ),
            "응답 상태": row["status"],
            "응답 시간(초)": row["latency_seconds"],
            "토큰 수": (
                row["input_tokens"] + row["output_tokens"]
                if row["input_tokens"] is not None
                and row["output_tokens"] is not None
                else "-"
            ),
        }
        for row in sorted(rows, key=lambda item: item["start_time"], reverse=True)
    ]


def show_request_detail(row: dict) -> None:
    st.markdown("#### 적용 설정")
    st.dataframe([row["settings"]], hide_index=True, width="stretch")

    st.markdown("#### 검색 청크")
    chunks = [
        {
            "순위": item.get("rank"),
            "청크 ID": item.get("chunk_id"),
            "문서": item.get("file_name"),
            "페이지": item.get("page_start"),
            "점수": item.get("score"),
            "답변 사용": "예" if item.get("used_in_answer") else "아니요",
        }
        for item in row["retrieval_results"]
    ]
    if chunks:
        st.dataframe(chunks, hide_index=True, width="stretch")
    else:
        st.caption("검색 청크 기록이 없습니다.")

    st.markdown("#### 사용 근거")
    citations = [
        {
            "자료 ID": item.get("source_id"),
            "문서": item.get("file_name"),
            "시작 페이지": item.get("page_start"),
            "끝 페이지": item.get("page_end"),
        }
        for item in row["citations"]
    ]
    if citations:
        st.dataframe(citations, hide_index=True, width="stretch")
    else:
        st.caption("사용 근거 기록이 없습니다.")

    if row["error"]:
        st.markdown("#### 오류 정보")
        st.code(str(row["error"]), language=None)
    st.caption(f"Langfuse trace ID: {row['trace_id']}")


def render() -> None:
    st.title("대시보드")
    load_dotenv(ROOT / ".env", override=False)
    if not configured():
        st.warning("Langfuse 연결 정보가 설정되지 않았습니다.")
        return

    period_options = {"최근 1일": 1, "최근 7일": 7, "최근 30일": 30}
    period = st.selectbox("조회 기간", list(period_options), index=1)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=period_options[period])
    try:
        rows = fetch_service_requests(get_client(), start, end)
    except Exception:
        st.warning("Langfuse 서비스 기록을 불러오지 못했습니다.")
        return

    if not rows:
        st.caption("조회 기간에 서비스 요청이 없습니다.")
        return

    filter_user, filter_mode, filter_status = st.columns(3)
    with filter_user:
        user_id = st.selectbox(
            "사용자", ["전체", *sorted({row["user_id"] for row in rows})]
        )
    with filter_mode:
        retrieval_mode = st.selectbox(
            "검색 방식",
            ["전체", *[mode for mode in RETRIEVAL_LABELS if any(
                row["retrieval_mode"] == mode for row in rows
            )]],
            format_func=lambda value: RETRIEVAL_LABELS.get(value, value),
        )
    with filter_status:
        status = st.selectbox(
            "응답 상태", ["전체", *sorted({row["status"] for row in rows})]
        )

    filtered = filter_requests(rows, user_id, retrieval_mode, status)
    normal = normal_count(filtered)
    average_latency = average(filtered, "latency_seconds")
    average_input = average(filtered, "input_tokens")
    average_output = average(filtered, "output_tokens")
    metric_columns = st.columns(5)
    values = (
        ("서비스 질문 수", str(len(filtered))),
        ("정상 응답률", f"{normal / len(filtered):.1%}" if filtered else "-"),
        (
            "평균 응답 시간",
            f"{average_latency:.2f}초" if average_latency is not None else "-",
        ),
        ("평균 입력 토큰", f"{average_input:.0f}" if average_input is not None else "-"),
        (
            "평균 출력 토큰",
            f"{average_output:.0f}" if average_output is not None else "-",
        ),
    )
    for column, (label, value) in zip(metric_columns, values, strict=True):
        column.metric(label, value)
    confirmed_usage = confirmed_usage_count(filtered)
    st.caption(f"토큰 사용량 확인: {confirmed_usage}/{len(filtered)}건")

    st.subheader("검색 방식별 현황")
    summary = mode_summary(filtered)
    if summary:
        st.dataframe(summary, hide_index=True, width="stretch")
    else:
        st.caption("선택한 조건의 요청이 없습니다.")

    st.subheader("최근 요청")
    if not filtered:
        st.caption("선택한 조건의 요청이 없습니다.")
        return
    ordered = sorted(filtered, key=lambda item: item["start_time"], reverse=True)
    st.dataframe(recent_rows(ordered), hide_index=True, width="stretch")
    selected_trace = st.selectbox(
        "요청 선택",
        options=[row["trace_id"] for row in ordered],
        format_func=lambda trace_id: next(
            f"{format_time(row['start_time'])} · {row['question'][:70]}"
            for row in ordered
            if row["trace_id"] == trace_id
        ),
    )
    selected = next(row for row in ordered if row["trace_id"] == selected_trace)
    show_request_detail(selected)


if __name__ == "__main__":
    render()
