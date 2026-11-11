"""Cross-encoder를 가짜 객체로 대체해 Reranker의 정렬 로직을 검사한다."""

from backend.retrieval import reranker as reranker_module

CONFIG = {
    "reranker_model": "test-reranker",
    "reranker_device": "cuda",
    "reranker_max_length": 1024,
    "reranker_batch_size": 4,
}


class FakeCrossEncoder:
    """미리 정한 점수를 반환하고 호출 내용을 저장한다."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls = []
        self.init_args = None

    def predict(
        self,
        pairs: list[list[str]],
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[float]:
        self.calls.append(
            {
                "pairs": pairs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
        )
        return self.scores


def make_reranker(monkeypatch, scores: list[float]):
    """가짜 Cross-encoder를 사용하는 Reranker를 만든다."""
    fake_model = FakeCrossEncoder(scores)

    def fake_cross_encoder(model_name: str, device: str, max_length: int):
        fake_model.init_args = {
            "model_name": model_name,
            "device": device,
            "max_length": max_length,
        }
        return fake_model

    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        fake_cross_encoder,
    )
    return reranker_module.Reranker(CONFIG), fake_model


def make_candidate(
    chunk_id: str,
    text: str,
    score: float,
    dense_rank: int | None,
    keyword_rank: int | None,
) -> dict:
    """시험에 사용할 RRF 후보를 만든다."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "dense_rank": dense_rank,
        "keyword_rank": keyword_rank,
    }


def test_loads_model_from_config(monkeypatch):
    reranker, fake_model = make_reranker(monkeypatch, [])

    assert reranker.model_name == "test-reranker"
    assert reranker.device == "cuda"
    assert reranker.max_length == 1024
    assert reranker.batch_size == 4
    assert fake_model.init_args == {
        "model_name": "test-reranker",
        "device": "cuda",
        "max_length": 1024,
    }


def test_returns_empty_list_without_prediction(monkeypatch):
    reranker, fake_model = make_reranker(monkeypatch, [])

    result = reranker.rerank("질문", [])

    assert result == []
    assert fake_model.calls == []


def test_sorts_candidates_and_preserves_search_metadata(monkeypatch):
    reranker, fake_model = make_reranker(
        monkeypatch,
        scores=[0.2, 0.9, 0.5],
    )
    candidates = [
        make_candidate("c1", "첫 번째 청크", 0.031, 1, 3),
        make_candidate("c2", "두 번째 청크", 0.029, 2, 1),
        make_candidate("c3", "세 번째 청크", 0.025, 3, None),
    ]

    result = reranker.rerank("설정을 변경할 수 있나요?", candidates)

    assert [candidate["chunk_id"] for candidate in result] == [
        "c2",
        "c3",
        "c1",
    ]
    assert [candidate["rerank_rank"] for candidate in result] == [1, 2, 3]
    assert [candidate["rerank_score"] for candidate in result] == [
        0.9,
        0.5,
        0.2,
    ]
    assert [candidate["rrf_score"] for candidate in result] == [
        0.029,
        0.025,
        0.031,
    ]
    assert result[0]["dense_rank"] == 2
    assert result[0]["keyword_rank"] == 1
    assert result[0]["score"] == result[0]["rerank_score"]

    assert fake_model.calls == [
        {
            "pairs": [
                ["설정을 변경할 수 있나요?", "첫 번째 청크"],
                ["설정을 변경할 수 있나요?", "두 번째 청크"],
                ["설정을 변경할 수 있나요?", "세 번째 청크"],
            ],
            "batch_size": 4,
            "show_progress_bar": False,
        }
    ]


def test_preserves_rrf_order_when_rerank_scores_are_equal(monkeypatch):
    reranker, _ = make_reranker(
        monkeypatch,
        scores=[0.7, 0.7],
    )
    candidates = [
        make_candidate("c1", "첫 번째 청크", 0.031, 1, 2),
        make_candidate("c2", "두 번째 청크", 0.030, 2, 1),
    ]

    result = reranker.rerank("질문", candidates)

    assert [candidate["chunk_id"] for candidate in result] == ["c1", "c2"]
    assert [candidate["rerank_rank"] for candidate in result] == [1, 2]