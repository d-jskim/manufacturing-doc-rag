"""RAGPipeline의 검색 후보 선택 동작을 검사한다."""

from backend import pipeline as pipeline_module


class FakeEmbedder:
    model_code = "m1"

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


def test_retrieve_uses_final_k_and_keeps_all_candidates(monkeypatch):
    candidates = [
        {
            "chunk_id": f"c{index}",
            "book_id": "MB01",
            "text": f"청크 {index}",
            "score": 1.0 / index,
        }
        for index in range(1, 6)
    ]
    captured = {}

    def fake_search(client, collection, query_vector, limit, exact):
        return candidates

    def fake_fill_context(selected, counter, token_budget):
        captured["selected"] = selected
        captured["token_budget"] = token_budget
        return selected, 1200

    monkeypatch.setattr(
        pipeline_module,
        "search",
        fake_search,
    )
    monkeypatch.setattr(
        pipeline_module,
        "fill_context",
        fake_fill_context,
    )

    pipeline = pipeline_module.RAGPipeline.__new__(
        pipeline_module.RAGPipeline
    )
    pipeline.config = {
        "final_k": 3,
        "context_token_budget": 2373,
    }
    pipeline.retrieval_configs = {
        "naive": {
            "retrieval_version": "r1",
            "method": "dense",
            "candidate_count": 20,
            "exact": True,
        }
    }
    pipeline.embedder = FakeEmbedder()
    pipeline.client = object()
    pipeline.collection = "test_collection"
    pipeline.counter = object()

    result = pipeline.retrieve(
        "질문",
        "naive",
    )

    assert [
        candidate["chunk_id"]
        for candidate in captured["selected"]
    ] == ["c1", "c2", "c3"]
    assert captured["token_budget"] == 2373

    assert result["final_k"] == 3
    assert len(result["candidates"]) == 5
    assert len(result["context"]) == 3
    assert result["context_token_count"] == 1200