"""실험 > 토큰 한도."""
import streamlit as st

from components.report import (
    format_value, show_conclusion, show_count_bars, show_header,
    show_histogram, show_metrics, show_predictions,
)
from results_loader import (
    list_datasets,
    list_report_files,
    load_report_file,
    load_result,
    results_dir,
)

EXPERIMENT = "01_token_budget"

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
selected = results_dir(EXPERIMENT, dataset) / report["result_file"]

show_header(report["title"], report["purpose"])

result = load_result(selected)
summary = result["summary"]
pages = result["pages"]
limit = summary["embedding_input_limit"]
max_tokens = limit["max_tokens"]
n = summary["context_page_count"]
generation_name = summary["generation_tokenizer"]["name"]

show_metrics([
    {"label": f"문맥 토큰 한도 · {generation_name}",
     "value": format_value("context_token_budget", summary["context_token_budget"])},
    {"label": "N(문맥 페이지 수)", "value": format_value("context_page_count", n)},
    {"label": f"{max_tokens} 초과 페이지 · E5",
     "value": format_value("over_limit_page_ratio", limit["over_limit_page_ratio"])},
    {"label": "토큰당 글자 수 · E5",
     "value": format_value("chars", summary["chars_per_embedding_input_token"])},
])

left, right = st.columns(2)
with left:
    st.markdown("**페이지별 토큰 수 분포**")
    e5_tab, generation_tab = st.tabs(["임베딩(E5)", f"생성 모델({generation_name})"])
    with e5_tab:
        show_histogram(
            [page["embedding_input_token_count"] for page in pages],
            max_tokens, f"{max_tokens} 기준선", "페이지 토큰 수", "페이지 수",
        )
    with generation_tab:
        p90 = summary["generation_input_token_count"]["p90"]
        show_histogram(
            [page["generation_input_token_count"] for page in pages],
            p90, f"p90 {format_value('p90', p90)}", "페이지 토큰 수", "페이지 수",
        )

with right:
    st.markdown("**질문별 근거 페이지 수 분포**")
    counts = {0: summary["questions"]["no_evidence_question_count"]}
    counts.update({int(k): v for k, v in summary["evidence_page_count_distribution"].items()})
    show_count_bars(counts, n, f"N={n}", "근거 페이지 수", "질문 수")

prediction_col, conclusion_col = st.columns([3, 2])
with prediction_col:
    st.markdown("**예측 대비 실제**")
    show_predictions(report["predictions"], result)
with conclusion_col:
    show_conclusion(report["conclusion"])
