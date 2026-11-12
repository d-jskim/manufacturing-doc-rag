"""실험 > 03 프롬프트."""

import altair as alt
import pandas as pd
import streamlit as st

from components.report import (
    format_experiment_id,
    show_failure_analysis,
    show_experiment_id_definitions,
    show_result_analysis,
)
from results_loader import (
    list_datasets,
    list_report_files,
    load_report_file,
    load_result,
    results_dir,
)

EXPERIMENT = "03_prompt"


def percent(value):
    return value * 100 if value is not None else None


dataset = st.selectbox("데이터셋", list_datasets(EXPERIMENT))
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
settings = report["settings"]
baseline_prompt = report["baseline_prompt"]

st.header(report["title"])
st.markdown(
    f"- **데이터셋:** {report['dataset_label']}\n"
    f"- **평가 목적:** {report['purpose']}"
)

columns = st.columns(4)
columns[0].metric("평가 질문 수", settings["evaluation_questions"])
columns[1].metric("기준 프롬프트", format_experiment_id(report, baseline_prompt))
columns[2].metric(
    "고정 청킹",
    format_experiment_id(report, settings["chunk_version"]),
)
columns[3].metric("생성 모델", settings["generation_model"])
show_experiment_id_definitions(
    report,
    [
        settings["chunk_version"],
        settings["embedding"],
        baseline_prompt,
        *settings["retrieval_modes"].values(),
    ],
)

quality_rows = []
operation_rows = []

for prompt_version, source in report["prompt_results"].items():
    source_base = results_dir(
        source["source_experiment"],
        source["source_dataset"],
    )
    answers = load_result(source_base / source["files"]["answers"])
    faithfulness = load_result(source_base / source["files"]["faithfulness"])
    judge = load_result(source_base / source["files"]["judge"])

    prompt_label = format_experiment_id(report, prompt_version)

    for mode, retrieval_id in settings["retrieval_modes"].items():
        mode_label = format_experiment_id(report, retrieval_id)
        answer_summary = answers["summary"][mode]
        faith_summary = faithfulness["summary"][mode]
        judge_summary = judge["summary"][mode]
        statuses = answer_summary["status_counts"]

        quality_rows.append(
            {
                "프롬프트": prompt_label,
                "검색 방식": mode_label,
                "Faithfulness": faith_summary["mean_faithfulness"],
                "사실 충족률": percent(
                    judge_summary["facts"]["pass_rate"]
                ),
                "조건 충족률": percent(
                    judge_summary["conditions"]["pass_rate"]
                ),
                "근거 충족률": percent(
                    judge_summary["evidence"]["pass_rate"]
                ),
                "판단 충족률": percent(
                    judge_summary["decision"]["pass_rate"]
                ),
                "Faithfulness 평가 수": faith_summary[
                    "evaluated_count"
                ],
                "Faithfulness 오류": faith_summary["error_count"],
                "Judge 오류": judge_summary[
                    "evaluation_error_count"
                ],
            }
        )

        operation_rows.append(
            {
                "프롬프트": prompt_label,
                "검색 방식": mode_label,
                "요청 수": answer_summary["request_count"],
                "평균 응답 시간(초)": answer_summary[
                    "normal_latency_seconds"
                ]["mean"],
                "시스템 오류율": percent(
                    answer_summary["system_error_rate"]
                ),
                "입력 토큰": answer_summary["token_usage"][
                    "input_tokens"
                ],
                "출력 토큰": answer_summary["token_usage"][
                    "output_tokens"
                ],
                "답변/추가확인/보류": (
                    f"{statuses.get('answered', 0)}/"
                    f"{statuses.get('needs_clarification', 0)}/"
                    f"{statuses.get('abstained', 0)}"
                ),
            }
        )

quality = pd.DataFrame(quality_rows)
operations = pd.DataFrame(operation_rows)

st.subheader("프롬프트 품질 Baseline")

st.dataframe(
    quality,
    hide_index=True,
    width="stretch",
    column_config={
        "Faithfulness": st.column_config.NumberColumn(format="%.4f"),
        "사실 충족률": st.column_config.NumberColumn(format="%.1f%%"),
        "조건 충족률": st.column_config.NumberColumn(format="%.1f%%"),
        "근거 충족률": st.column_config.NumberColumn(format="%.1f%%"),
        "판단 충족률": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

prompt_options = quality["프롬프트"].drop_duplicates().tolist()
selected_prompt = (
    st.selectbox("그래프 프롬프트", prompt_options)
    if len(prompt_options) > 1
    else prompt_options[0]
)
selected_quality = quality[quality["프롬프트"] == selected_prompt]
retrieval_order = selected_quality["검색 방식"].tolist()
retrieval_colors = ["#94A3B8", "#60A5FA", "#2563EB"]

judge_metrics = ["사실 충족률", "조건 충족률", "근거 충족률", "판단 충족률"]
judge_data = selected_quality.melt(
    id_vars="검색 방식",
    value_vars=judge_metrics,
    var_name="지표",
    value_name="값",
)

judge_base = alt.Chart(judge_data).encode(
    x=alt.X("지표:N", title=None, sort=judge_metrics, axis=alt.Axis(labelAngle=0)),
    xOffset=alt.XOffset("검색 방식:N", sort=retrieval_order),
    y=alt.Y(
        "값:Q",
        title="충족률 (%)",
        scale=alt.Scale(domain=[0, 100]),
        axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]),
    ),
    color=alt.Color(
        "검색 방식:N",
        title="검색 방식",
        sort=retrieval_order,
        scale=alt.Scale(domain=retrieval_order, range=retrieval_colors),
        legend=alt.Legend(orient="top", direction="horizontal"),
    ),
    tooltip=[
        alt.Tooltip("검색 방식:N"),
        alt.Tooltip("지표:N"),
        alt.Tooltip("값:Q", title="충족률", format=".1f"),
    ],
)

judge_bars = judge_base.mark_bar(size=22)
judge_labels = judge_base.mark_text(dy=-8, fontSize=11).encode(
    text=alt.Text("값:Q", format=".1f")
)
judge_chart = (judge_bars + judge_labels).properties(
    title=alt.TitleParams(
        text=f"{selected_prompt} · Judge 품질 비교",
        anchor="start",
        fontSize=16,
        fontWeight="normal",
    ),
    height=320,
)

st.altair_chart(judge_chart, width="stretch")

faithfulness_base = alt.Chart(selected_quality).encode(
    x=alt.X("검색 방식:N", title=None, sort=retrieval_order, axis=alt.Axis(labelAngle=0)),
    y=alt.Y(
        "Faithfulness:Q",
        title="Faithfulness",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1]),
    ),
    color=alt.Color(
        "검색 방식:N",
        legend=None,
        sort=retrieval_order,
        scale=alt.Scale(domain=retrieval_order, range=retrieval_colors),
    ),
    tooltip=[
        alt.Tooltip("검색 방식:N"),
        alt.Tooltip("Faithfulness:Q", format=".4f"),
    ],
)

faithfulness_bars = faithfulness_base.mark_bar(size=48)
faithfulness_labels = faithfulness_base.mark_text(dy=-8, fontSize=12).encode(
    text=alt.Text("Faithfulness:Q", format=".4f")
)
faithfulness_chart = (faithfulness_bars + faithfulness_labels).properties(
    title=alt.TitleParams(
        text=f"{selected_prompt} · Faithfulness 비교",
        anchor="start",
        fontSize=16,
        fontWeight="normal",
    ),
    height=280,
)

st.altair_chart(faithfulness_chart, width="stretch")

st.subheader("운영 지표")

st.dataframe(
    operations,
    hide_index=True,
    width="stretch",
    column_config={
        "평균 응답 시간(초)": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "시스템 오류율": st.column_config.NumberColumn(
            format="%.1f%%"
        ),
        "입력 토큰": st.column_config.NumberColumn(format="%d"),
        "출력 토큰": st.column_config.NumberColumn(format="%d"),
    },
)

analysis_labels = {
    version: format_experiment_id(report, version)
    for version in report["prompt_results"]
}

show_result_analysis(
    report.get("result_analysis"),
    analysis_labels,
)
show_failure_analysis(report.get("failure_analysis"))

security = report.get("security_evaluation")

if security:
    st.subheader("Security Evaluation")

    if security["status"] == "not_evaluated":
        st.info(security["description"])
    else:
        st.markdown(security["description"])
