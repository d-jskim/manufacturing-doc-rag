"""보고서 화면 부품: 실험 보고서 화면에서 공통으로 쓰는 표시 함수 모음."""
import altair as alt
import pandas as pd
import streamlit as st

from results_loader import get_value

THRESHOLD_COLOR = "#DC2626"


def format_value(key: str, value) -> str:
    """개수 비율(_ratio)은 %, 정수 값(712.0 포함)은 천 단위 쉼표, 그 외는 그대로 표시함."""
    if key.endswith("_ratio"):
        return f"{value:.1%}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def show_header(title: str, purpose: str) -> None:
    st.title(title)
    st.caption(f"목적: {purpose}")


def show_metrics(metrics: list[dict]) -> None:
    """metrics: [{"label": ..., "value": ...}] 형태. 카드를 한 줄로 배치함."""
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        column.metric(metric["label"], metric["value"])


def threshold_layer(x: float, label: str) -> alt.Chart:
    """세로 기준선과 이름표."""
    data = pd.DataFrame({"x": [x], "label": [label]})
    rule = alt.Chart(data).mark_rule(color=THRESHOLD_COLOR, strokeWidth=2).encode(x="x:Q")
    text = alt.Chart(data).mark_text(
        color=THRESHOLD_COLOR, align="left", dx=4, fontWeight="bold"
    ).encode(x="x:Q", y=alt.value(10), text="label:N")
    return rule + text


def show_histogram(values: list[int], threshold: float, threshold_label: str,
                   x_title: str, y_title: str) -> None:
    """값 분포 히스토그램과 기준선."""
    data = pd.DataFrame({"value": values})
    bars = alt.Chart(data).mark_bar(color="#9CA3AF").encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title=x_title),
        y=alt.Y("count():Q", title=y_title),
    )
    st.altair_chart(bars + threshold_layer(threshold, threshold_label))


def show_count_bars(counts: dict[int, int], threshold: int, threshold_label: str,
                    x_title: str, y_title: str) -> None:
    """구간별 개수 막대와 기준선. 기준선은 threshold 막대 오른쪽 끝에 그림."""
    data = pd.DataFrame({"value": list(counts), "count": list(counts.values())})
    bars = alt.Chart(data).mark_bar(color="#9CA3AF", size=40).encode(
        x=alt.X("value:Q", title=x_title, axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("count:Q", title=y_title),
    )
    st.altair_chart(bars + threshold_layer(threshold + 0.5, threshold_label))


def show_predictions(predictions: list[dict], result: dict) -> None:
    """report.yaml의 예측과 결과 JSON의 실제 값을 표로 비교함."""
    rows = []
    for prediction in predictions:
        key = prediction["result_key"]
        actual = get_value(result, key)
        rows.append({
            "항목": prediction["item"],
            "예측": str(prediction["predicted"]),
            "실제": format_value(key, actual),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_conclusion(conclusion: dict) -> None:
    st.markdown("**결정**")
    st.markdown("\n".join(f"- {item}" for item in conclusion["decisions"]))
    st.markdown("**한계**")
    st.markdown("\n".join(f"- {item}" for item in conclusion["limitations"]))