"""실험 화면이 Basic 데이터만 사용하도록 고정됐는지 검증한다."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_VIEWS = (
    "experiments_token_budget.py",
    "experiments_chunking.py",
    "experiments_prompt.py",
    "experiments_rag_comparison.py",
)


class StopRender(Exception):
    pass


def load_rag_comparison(monkeypatch):
    fake_streamlit = ModuleType("streamlit")
    fake_streamlit.selectbox = lambda *args, **kwargs: (_ for _ in ()).throw(
        StopRender
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.syspath_prepend(str(ROOT / "frontend"))
    path = ROOT / "frontend" / "views" / "experiments_rag_comparison.py"
    spec = importlib.util.spec_from_file_location("rag_comparison_check", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except StopRender:
        pass
    return module


def test_all_experiment_views_are_fixed_to_basic_without_dataset_selector():
    for name in EXPERIMENT_VIEWS:
        path = ROOT / "frontend" / "views" / name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }

        assert assignments["DATASET"] == "manufacturing_basic"
        assert "list_datasets" not in source
        assert 'selectbox("데이터셋"' not in source


def test_judge_display_uses_passed_over_evaluated(monkeypatch):
    module = load_rag_comparison(monkeypatch)

    assert module.format_judge_evaluation(90.0, 30, 0) == "90.0% (27/30)"
    assert (
        module.format_judge_evaluation(100.0, 29, 1)
        == "100.0% (29/29) · 해당 없음 1건"
    )


def test_prompt_overview_is_stored_in_source_config_and_used_by_view():
    config_path = ROOT / "configs" / "prompts" / "p1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    view_source = (
        ROOT / "frontend" / "views" / "experiments_prompt.py"
    ).read_text(encoding="utf-8")

    assert config["overview"] == [
        "질문의 적용 대상과 조건 확인",
        "검색 자료의 적용 가능성과 문서 근거 검증",
        "근거가 충분하면 답변",
        "사용자 조건이 부족하면 추가 확인",
        "근거가 부족하거나 충돌하면 답변 보류",
        "상태·답변·판단 이유·자료 ID를 JSON으로 반환",
    ]
    assert 'prompt_config["overview"]' in view_source
