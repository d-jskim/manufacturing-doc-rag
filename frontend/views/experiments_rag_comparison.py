"""실험 > RAG 방식 비교."""

import altair as alt
import pandas as pd
import streamlit as st

from results_loader import (
    list_datasets,
    list_report_files,
    load_report_file,
    load_result,
    results_dir,
)

EXPERIMENT = "04_retrieval"
MODES = ("naive", "hybrid", "rerank")
MODE_LABELS = {
    "naive": "Naive",
    "hybrid": "Hybrid",
    "rerank": "Hybrid + Reranking",
}
MODE_ORDER = list(MODE_LABELS.values())
MODE_COLORS = ["#94A3B8", "#60A5FA", "#2563EB"]
JUDGE_CRITERIA = ("facts", "conditions", "evidence", "decision")

datasets = list_datasets(EXPERIMENT)
dataset = st.selectbox("데이터셋", datasets)
report_files = list_report_files(EXPERIMENT, dataset)

if not report_files:
    st.error("승인된 실행 보고서가 없습니다.")
    st.stop()

reports = {path: load_report_file(path) for path in report_files}
selected_report_path = st.selectbox(
    "보고서(실행 ID)",
    report_files,
    format_func=lambda path: reports[path].get("report_label", path.stem),
)

report = reports[selected_report_path]
base = results_dir(EXPERIMENT, dataset)

# 선택한 실행별 보고서에 지정된 결과 파일만 사용한다.
retrieval = load_result(base / report["result_files"]["retrieval"])
answers = load_result(base / report["result_files"]["answers"])
faithfulness = load_result(base / report["result_files"]["faithfulness"])
judge = load_result(base / report["result_files"]["judge"])
settings = report["settings"]

st.header(report["title"])
st.markdown(
    f"- **데이터셋:** {report['dataset_label']}\n"
    f"- **비교 구성:** {report['comparison_label']}\n"
    f"- **평가 목적:** {report['purpose']}"
)

columns = st.columns(4)
columns[0].metric(
    "평가 질문 수",
    settings["evaluation_questions"],
)
columns[1].metric(
    "상위 청크 수 (k)",
    settings["final_k"],
)
columns[2].metric(
    "문맥 토큰 한도",
    f"{settings['context_token_budget']:,}",
)
columns[3].metric(
    "생성 모델",
    settings["generation_model"],
)

quality_rows = []
coverage_rows = []
operation_rows = []

for mode in MODES:
    label = MODE_LABELS[mode]
    retrieval_summary = retrieval["summary"][mode]
    search = retrieval_summary["k3"]
    service = answers["summary"][mode]
    faith = faithfulness["summary"][mode]
    scores = judge["summary"][mode]
    statuses = service["status_counts"]

    quality_rows.append(
        {
            "방식": label,
            "Hit@3": search["hit_at_k"] * 100,
            "Recall@3": search["recall_at_k"] * 100,
            # RAGAS 논문 표기처럼 0~1 점수를 유지한다.
            "Faithfulness": faith["mean_faithfulness"],
            "사실 충족률": scores["facts"]["pass_rate"] * 100,
            "조건 충족률": (
                scores["conditions"]["pass_rate"] * 100
            ),
            "근거 충족률": (
                scores["evidence"]["pass_rate"] * 100
            ),
            "판단 충족률": (
                scores["decision"]["pass_rate"] * 100
            ),
        }
    )

    coverage_rows.append(
        {
            "방식": label,
            "검색 지표 산출": search["evaluated_count"],
            "검색 지표 해당 없음": search["not_applicable_count"],
            "Faithfulness 산출": faith["evaluated_count"],
            "Faithfulness 평가 불가": faith["error_count"],
            "Judge 채점": scores["request_count"],
            "Judge 오류": scores["evaluation_error_count"],
            "Judge 해당 없음 (사실/조건/근거/판단)": "/".join(
                str(scores[criterion]["not_applicable_count"])
                for criterion in JUDGE_CRITERIA
            ),
        }
    )

    operation_rows.append(
        {
            "방식": label,
            "요청 수": service["request_count"],
            "검색 시간(초)": retrieval_summary[
                "retrieval_seconds"
            ]["mean"],
            "전체 응답 시간(초)": service[
                "normal_latency_seconds"
            ]["mean"],
            "응답 시간 집계": service[
                "normal_latency_seconds"
            ]["count"],
            "오류율": service["system_error_rate"] * 100,
            "오류 건수": service["system_error_count"],
            "입력 토큰": service["token_usage"]["input_tokens"],
            "출력 토큰": service["token_usage"]["output_tokens"],
            "토큰 확인 요청": service["token_usage"][
                "confirmed_request_count"
            ],
            "모델 호출 비용": "확인 불가",
            "답변/추가확인/보류": (
                f"{statuses.get('answered', 0)}/"
                f"{statuses.get('needs_clarification', 0)}/"
                f"{statuses.get('abstained', 0)}"
            ),
        }
    )

quality = pd.DataFrame(quality_rows)
coverage = pd.DataFrame(coverage_rows)
operations = pd.DataFrame(operation_rows)

st.subheader("품질 비교")

# 표는 검색, Faithfulness, Judge 순서로 배치한다.
# 비율 지표만 %로 표시하고 Faithfulness는 0~1 점수를 유지한다.
st.dataframe(
    quality,
    hide_index=True,
    width="stretch",
    column_config={
        "Hit@3": st.column_config.NumberColumn(format="%.1f%%"),
        "Recall@3": st.column_config.NumberColumn(format="%.1f%%"),
        "Faithfulness": st.column_config.NumberColumn(
            format="%.4f"
        ),
        "사실 충족률": st.column_config.NumberColumn(
            format="%.1f%%"
        ),
        "조건 충족률": st.column_config.NumberColumn(
            format="%.1f%%"
        ),
        "근거 충족률": st.column_config.NumberColumn(
            format="%.1f%%"
        ),
        "판단 충족률": st.column_config.NumberColumn(
            format="%.1f%%"
        ),
    },
)


def chart_title(text: str) -> alt.TitleParams:
    """표 머리글과 비슷한 형태로 차트 제목을 표시한다."""
    return alt.TitleParams(
        text=text,
        anchor="start",
        color="#4B5563",
        fontSize=16,
        fontWeight="normal",
        offset=10,
    )


def column_rule() -> alt.Chart:
    """표처럼 지표 열 사이에 세로 구분선을 표시한다."""
    return (
        alt.Chart(pd.DataFrame({"value": [0]}))
        .mark_rule(color="#D1D5DB")
        .encode(x=alt.value(0))
    )


def make_metric_chart(
    metric: str,
    domain: list[float],
    is_ratio: bool,
) -> alt.Chart:
    """지표의 원래 단위를 유지한 막대그래프를 만든다."""
    chart_data = quality[["방식", metric]].copy()

    # 단위는 축에 표시하므로 막대에는 숫자만 표시한다.
    chart_data["표시값"] = chart_data[metric].map(
        lambda value: (
            f"{value:.1f}"
            if is_ratio
            else f"{value:.4f}"
        )
    )

    base_chart = alt.Chart(chart_data).encode(
        y=alt.Y(
            "방식:N",
            title=None,
            sort=MODE_ORDER,
            axis=None,
        ),
        color=alt.Color(
            "방식:N",
            legend=None,
            sort=MODE_ORDER,
            scale=alt.Scale(
                domain=MODE_ORDER,
                range=MODE_COLORS,
            ),
        ),
        tooltip=[
            alt.Tooltip("방식:N"),
            alt.Tooltip("표시값:N", title=metric),
        ],
    )

    bars = base_chart.mark_bar(size=26).encode(
        x=alt.X(
            f"{metric}:Q",
            title=None,
            scale=alt.Scale(domain=domain),
            axis=alt.Axis(
                title=None,
                labelBound=False,
                labelFlush=False,
                labelOverlap=False,
                values=(
                    [0, 20, 40, 60, 80, 100]
                    if is_ratio
                    else [0, 0.2, 0.4, 0.6, 0.8, 1]
                ),
                labelExpr=(
                    "datum.value === 100 "
                    "? '100 (%)' : format(datum.value, '.0f')"
                    if is_ratio
                    else "format(datum.value, '.1f')"
                ),
            ),
        )
    )

    # 막대 색상을 상속하지 않도록 별도 차트에서 수치를 표시한다.
    labels = (
        alt.Chart(chart_data)
        .mark_text(
            align="left",
            baseline="middle",
            dx=6,
            color="white",
            fontWeight="bold",
        )
        .encode(
            x=alt.value(0),
            y=alt.Y(
                "방식:N",
                title=None,
                sort=MODE_ORDER,
                axis=None,
            ),
            text=alt.Text("표시값:N"),
        )
    )

    return (column_rule() + bars + labels).properties(
        title=chart_title(metric),
        width=260,
        height=105,
    )


# 표의 첫 번째 열처럼 검색 방식 이름만 배치한다.
method_data = pd.DataFrame({"방식": MODE_ORDER})
method_labels = (
    alt.Chart(method_data)
    .mark_text(
        align="left",
        baseline="middle",
        color="#4B5563",
        fontSize=14,
    )
    .encode(
        x=alt.value(0),
        y=alt.Y(
            "방식:N",
            title=None,
            sort=MODE_ORDER,
            axis=None,
        ),
        text=alt.Text("방식:N"),
    )
)
method_chart = method_labels.properties(
    title=chart_title(" "),
    width=165,
    height=105,
)

# 표의 열 순서와 같은 순서로 대표 지표를 배치한다.
charts = [
    make_metric_chart(
        "Hit@3",
        [0, 100],
        True,
    ),
    make_metric_chart(
        "Recall@3",
        [0, 100],
        True,
    ),
    make_metric_chart(
        "Faithfulness",
        [0, 1],
        False,
    ),
    make_metric_chart(
        "근거 충족률",
        [0, 100],
        True,
    ),
]

# Faithfulness는 단위가 다르므로 지표별 x축을 독립적으로 사용한다.
st.altair_chart(
    alt.hconcat(
        method_chart,
        *charts,
        spacing=12,
    ).resolve_scale(
        x="independent",
        y="shared",
    ).configure_view(strokeOpacity=0),
    width="stretch",
)

st.subheader("자동 평가 처리 현황")
st.markdown(
    "- **검색 지표:** 규칙 기반 계산\n"
    "- **Faithfulness:** RAGAS + gpt-6-luna\n"
    "- **Judge:** gpt-6-luna"
)

st.dataframe(
    coverage,
    hide_index=True,
    width="stretch",
)

st.subheader("운영 지표")

st.dataframe(
    operations,
    hide_index=True,
    width="stretch",
    column_config={
        "검색 시간(초)": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "전체 응답 시간(초)": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "오류율": st.column_config.NumberColumn(
            format="%.1f%%"
        ),
        "입력 토큰": st.column_config.NumberColumn(
            format="%d"
        ),
        "출력 토큰": st.column_config.NumberColumn(
            format="%d"
        ),
    },
)
