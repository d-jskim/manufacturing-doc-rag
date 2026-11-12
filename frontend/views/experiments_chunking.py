"""실험 > 청킹 실험."""

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

EXPERIMENT = "02_chunking"
CHUNK_VERSIONS = ("c1", "c2", "c3")

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
settings = report["settings"]

# 선택한 실행별 보고서에 지정된 결과 파일만 사용한다.
results = {
    version: load_result(base / report["result_files"][version])
    for version in CHUNK_VERSIONS
}

st.header(report["title"])
st.markdown(
    f"- **데이터셋:** {report['dataset_label']}\n"
    f"- **평가 목적:** {report['purpose']}"
)
show_experiment_id_definitions(
    report,
    [*CHUNK_VERSIONS, settings["embedding"], "p0", settings["retrieval"]],
)

columns = st.columns(4)
columns[0].metric("평가 질문 수", settings["evaluation_questions"])
columns[1].metric("지표 산출 질문 수", settings["evaluated_questions"])
columns[2].metric("문맥 토큰 한도", f"{settings['context_token_budget']:,}")
columns[3].metric(
    "선택 청킹 버전",
    format_experiment_id(report, settings["selected_chunk_version"]),
)

rows = []

for version in CHUNK_VERSIONS:
    summary = results[version]["summary"]
    chunk_stats = summary["chunk_stats"]
    retrieval = summary["retrieval_metrics"]
    cost = summary["cost"]

    rows.append(
        {
            "청킹 버전": version.upper(),
            "청크 수": chunk_stats["chunk_count"],
            "중앙 청크 토큰": chunk_stats["chunk_encoded_token_count"]["median"],
            "제한 초과 청크": chunk_stats["over_limit_chunk_count"],
            "Hit@k": retrieval["hit_at_k"] * 100,
            "Recall@k": retrieval["recall_at_k"] * 100,
            "인용 Recall@k": retrieval["quote_recall_at_k"] * 100,
            "평균 k": retrieval["mean_k"],
            "총 처리 시간(초)": cost["total_seconds"],
            "문서 임베딩 토큰": cost["token_counts"]["embedding_passage"],
        }
    )

comparison = pd.DataFrame(rows)

st.subheader("검색 품질 비교")

st.dataframe(
    comparison[["청킹 버전", "Hit@k", "Recall@k", "인용 Recall@k", "평균 k"]],
    hide_index=True,
    width="stretch",
    column_config={
        "Hit@k": st.column_config.NumberColumn(format="%.2f%%"),
        "Recall@k": st.column_config.NumberColumn(format="%.2f%%"),
        "인용 Recall@k": st.column_config.NumberColumn(format="%.2f%%"),
        "평균 k": st.column_config.NumberColumn(format="%.2f"),
    },
)

quality_data = comparison.melt(
    id_vars="청킹 버전",
    value_vars=["Hit@k", "Recall@k", "인용 Recall@k"],
    var_name="지표",
    value_name="값",
)

chunk_labels = {
    version.upper(): format_experiment_id(report, version)
    for version in CHUNK_VERSIONS
}

quality_data["청킹 방식"] = quality_data["청킹 버전"].map(chunk_labels)

chunk_order = [chunk_labels[version.upper()] for version in CHUNK_VERSIONS]
metric_order = ["Hit@k", "Recall@k", "인용 Recall@k"]

base_chart = alt.Chart(quality_data).encode(
    x=alt.X(
        "지표:N",
        title=None,
        sort=metric_order,
        axis=alt.Axis(labelAngle=0, labelPadding=10),
    ),
    xOffset=alt.XOffset("청킹 방식:N", sort=chunk_order),
    y=alt.Y(
        "값:Q",
        title="성능 (%)",
        scale=alt.Scale(domain=[0, 100]),
        axis=alt.Axis(values=[0, 20, 40, 60, 80, 100], grid=True),
    ),
    color=alt.Color(
        "청킹 방식:N",
        title="청킹 방식",
        sort=chunk_order,
        scale=alt.Scale(
            domain=chunk_order,
            range=["#2563EB", "#94A3B8", "#F59E0B"],
        ),
        legend=alt.Legend(
            orient="top",
            direction="horizontal",
            columns=3,
            labelLimit=220,
        ),
    ),
    tooltip=[
        alt.Tooltip("지표:N"),
        alt.Tooltip("청킹 방식:N"),
        alt.Tooltip("값:Q", title="성능", format=".2f"),
    ],
)

bars = base_chart.mark_bar(size=28)

labels = base_chart.mark_text(dy=-8, fontSize=12).encode(
    text=alt.Text("값:Q", format=".2f")
)

quality_chart = (
    (bars + labels)
    .properties(
        title=alt.TitleParams(
            text="청킹 방식별 검색 성능 비교",
            anchor="start",
            fontSize=16,
            fontWeight="normal",
        ),
        height=360,
    )
    .configure_view(stroke=None)
)

st.altair_chart(quality_chart, width="stretch")

st.subheader("청크 및 처리 비용 비교")

st.dataframe(
    comparison[
        [
            "청킹 버전",
            "청크 수",
            "중앙 청크 토큰",
            "제한 초과 청크",
            "총 처리 시간(초)",
            "문서 임베딩 토큰",
        ]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "중앙 청크 토큰": st.column_config.NumberColumn(format="%.0f"),
        "총 처리 시간(초)": st.column_config.NumberColumn(format="%.3f"),
        "문서 임베딩 토큰": st.column_config.NumberColumn(format="%d"),
    },
)

analysis_labels = {
    version: chunk_labels[version.upper()]
    for version in CHUNK_VERSIONS
}

show_result_analysis(report.get("result_analysis"), analysis_labels)
show_failure_analysis(report.get("failure_analysis"))

selected = settings["selected_chunk_version"]
selected_summary = results[selected]["summary"]
selected_metrics = selected_summary["retrieval_metrics"]
selected_stats = selected_summary["chunk_stats"]

st.subheader("선택 결과")
st.success(f"선택된 청킹 버전: {format_experiment_id(report, selected)}")

st.markdown(
    f"- **Hit@k:** {selected_metrics['hit_at_k']:.2%}\n"
    f"- **Recall@k:** {selected_metrics['recall_at_k']:.2%}\n"
    f"- **인용 Recall@k:** {selected_metrics['quote_recall_at_k']:.2%}\n"
    f"- **제한 초과 청크:** {selected_stats['over_limit_chunk_count']:,}개"
)
